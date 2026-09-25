from __future__ import annotations

import copy
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

# Every ROS process in this project must use the same DDS implementation.
# The remote agent and robot_executor.py use Fast DDS, so direct users of this
# module must use Fast DDS too.
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

import rclpy

from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener


@dataclass(frozen=True)
class MotionResult:
    success: bool
    message: str = ""
    error_code: int | None = None

    def __bool__(self):
        return self.success


@dataclass(frozen=True)
class PoseMM:
    x: float
    y: float
    z: float

    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class SO101Config:

    planning_group: str = "manipulator"

    base_frame: str = "base_link"
    end_effector_frame: str = "gripper_frame_link"

    move_action: str = "/move_action"
    ik_service: str = "/compute_ik"
    move_group_node: str = "/move_group"

    joint_states_topic: str = "/follower/joint_states"

    planning_pipeline: str = "ompl"

    planning_time: float = 5.0
    planning_attempts: int = 10

    velocity_scaling: float = 0.20
    acceleration_scaling: float = 0.20

    # Kept for compatibility with callers that constructed SO101Config
    # before position goals were sent directly to MoveGroup.
    ik_timeout: float = 0.8

    arm_joints: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )

    joint_tolerance_rad: float = 0.01

    # MoveIt receives a position constraint for gripper_frame_link,
    # just like its RViz MotionPlanning client.  The goal is a sphere
    # around the requested XYZ, expressed in base_frame.
    position_goal_tolerance_mm: float = 3.0

    # SO-101 has five arm joints, so an arbitrary XYZ + quaternion is
    # generally over-constrained.  Position-only planning is therefore
    # the practical default.  Set this to True only when tool attitude is
    # important; lock_current_orientation() then defines the attitude.
    constrain_orientation: bool = False
    orientation_tolerance_rad: float = 0.15

    # PickIK otherwise tries to satisfy the arbitrary orientation sampled
    # by MoveIt for a position-only goal.  That cannot generally work for
    # a five-axis arm.  The library switches rotation_scale to zero only
    # while planning this command and restores the previous value after it.
    manage_pick_ik_position_only: bool = True
    parameter_timeout: float = 2.0

    # Keep the jaws from rolling around the tool axis while MoveIt is
    # otherwise free to choose an orientation that reaches the XYZ goal.
    preserve_wrist_roll: bool = True
    wrist_roll_joint: str = "wrist_roll"
    wrist_roll_tolerance_rad: float = 0.02

    # Сначала пробуем естественный yaw.
    # Потом чуть крутим его, но не меняем наклон схвата.
    yaw_search_deg: tuple[float, ...] = (
        0.0,
        5.0,
        -5.0,
        10.0,
        -10.0,
        20.0,
        -20.0,
        30.0,
        -30.0,
        45.0,
        -45.0,
        60.0,
        -60.0,
        90.0,
        -90.0,
    )

    # Проверяем реальное положение после движения.
    verify_position_tolerance_mm: float = 5.0
    verify_timeout: float = 1.5

    gripper_action: str = (
        "/follower/gripper_controller/gripper_cmd"
    )

    gripper_joint: str = "gripper"

    gripper_open_position: float = 1.50
    gripper_closed_position: float = -0.16

    gripper_tolerance_rad: float = 0.03

    # Current feedback is supplied by the patched Feetech driver in amperes.
    # About twenty-three STS current counts give a firmer grip while still
    # stopping well below the motor protection-current limit.
    # Tune only with disposable test objects, never a hand.
    gripper_contact_current_amp: float = 0.150
    # One over-current sample stops immediately.
    gripper_contact_samples: int = 1
    gripper_current_poll_interval: float = 0.02
    gripper_close_timeout: float = 25.0

    # Add only a small preload after contact to retain the object.
    gripper_grasp_preload_rad: float = 0.050
    gripper_hold_timeout: float = 3.0

    # The action server becomes discoverable shortly before ros2_control
    # finishes activating the gripper controller.  Retry goals that are
    # explicitly rejected during that small startup window.
    gripper_goal_retries: int = 6
    gripper_goal_retry_delay: float = 0.20

    server_timeout: float = 10.0
    command_timeout: float = 30.0

    shoulder_pan_x_mm: float = 38.8353
    shoulder_pan_y_mm: float = 0.0


# ============================================================
# MATH
# ============================================================


def _normalize_angle(angle: float) -> float:

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def _normalize_quaternion(
    q: Quaternion,
) -> Quaternion:

    norm = math.sqrt(
        q.x * q.x
        + q.y * q.y
        + q.z * q.z
        + q.w * q.w
    )

    result = Quaternion()

    if norm < 1e-12:
        result.w = 1.0
        return result

    result.x = q.x / norm
    result.y = q.y / norm
    result.z = q.z / norm
    result.w = q.w / norm

    return result


def _quat_multiply(
    a: Quaternion,
    b: Quaternion,
) -> Quaternion:

    result = Quaternion()

    result.w = (
        a.w * b.w
        - a.x * b.x
        - a.y * b.y
        - a.z * b.z
    )

    result.x = (
        a.w * b.x
        + a.x * b.w
        + a.y * b.z
        - a.z * b.y
    )

    result.y = (
        a.w * b.y
        - a.x * b.z
        + a.y * b.w
        + a.z * b.x
    )

    result.z = (
        a.w * b.z
        + a.x * b.y
        - a.y * b.x
        + a.z * b.w
    )

    return _normalize_quaternion(result)


def _yaw_quaternion(
    yaw: float,
) -> Quaternion:

    result = Quaternion()

    result.z = math.sin(
        yaw / 2.0
    )

    result.w = math.cos(
        yaw / 2.0
    )

    return result


# ============================================================
# ARM
# ============================================================


class SO101Arm:

    def __init__(
        self,
        config: SO101Config | None = None,
    ):

        self.config = (
            config
            or SO101Config()
        )

        self._command_lock = (
            threading.Lock()
        )

        self._goal_lock = (
            threading.Lock()
        )

        self._state_lock = (
            threading.Lock()
        )

        self._current_move_goal = None
        self._current_gripper_goal = None

        self._last_joint_state = None
        self._joint_positions = {}
        self._joint_efforts = {}
        self._joint_state_seq = 0

        self._fixed_orientation = None

        # Направление руки в XY
        # в момент фиксации ориентации.
        self._orientation_anchor_heading = None

        self._shutdown = False

        # ====================================================
        # ROS
        # ====================================================

        self._context = Context()

        rclpy.init(
            args=None,
            context=self._context,
        )

        self._node = Node(
            f"so101_control_{os.getpid()}",
            context=self._context,
        )

        # ====================================================
        # MOVEIT
        # ====================================================

        self._move_client = ActionClient(
            self._node,
            MoveGroup,
            self.config.move_action,
        )

        self._move_group_parameters = (
            AsyncParameterClient(
                self._node,
                self.config.move_group_node,
            )
        )

        # ====================================================
        # GRIPPER
        # ====================================================

        self._gripper_client = (
            ActionClient(
                self._node,
                ParallelGripperCommand,
                self.config.gripper_action,
            )
        )

        # ====================================================
        # JOINT STATES
        # ====================================================

        self._joint_subscription = (
            self._node.create_subscription(
                JointState,
                self.config.joint_states_topic,
                self._joint_callback,
                10,
            )
        )

        # ====================================================
        # TF
        # ====================================================

        self._tf_buffer = Buffer()

        self._tf_listener = (
            TransformListener(
                self._tf_buffer,
                self._node,
                spin_thread=False,
            )
        )

        # ====================================================
        # EXECUTOR
        # ====================================================

        self._executor = (
            MultiThreadedExecutor(
                num_threads=3,
                context=self._context,
            )
        )

        self._executor.add_node(
            self._node
        )

        self._spin_thread = (
            threading.Thread(
                target=self._executor.spin,
                daemon=True,
                name="so101_ros_executor",
            )
        )

        self._spin_thread.start()

        self._connect()

        if self.config.constrain_orientation:

            self.lock_current_orientation()

    # ========================================================
    # STATE
    # ========================================================

    def _joint_callback(
        self,
        msg: JointState,
    ):

        with self._state_lock:

            self._last_joint_state = (
                copy.deepcopy(msg)
            )

            self._joint_positions = {
                name: float(position)
                for name, position
                in zip(
                    msg.name,
                    msg.position,
                )
            }

            self._joint_efforts = {
                name: float(effort)
                for name, effort
                in zip(
                    msg.name,
                    msg.effort,
                )
            }

            self._joint_state_seq += 1

    def _current_joint_state(
        self,
    ):

        with self._state_lock:

            if (
                self._last_joint_state
                is None
            ):
                return None

            return copy.deepcopy(
                self._last_joint_state
            )

    def _get_joint_position(
        self,
        name: str,
    ):

        with self._state_lock:

            value = (
                self._joint_positions.get(
                    name
                )
            )

        if value is None:
            return None

        return float(value)

    def _get_joint_effort(
        self,
        name: str,
    ):

        with self._state_lock:

            value = self._joint_efforts.get(
                name
            )

        if (
            value is None
            or not math.isfinite(value)
        ):
            return None

        return float(value)

    def _get_joint_effort_sample(
        self,
        name: str,
    ):

        with self._state_lock:

            sequence = self._joint_state_seq
            value = self._joint_efforts.get(
                name
            )

        if (
            value is None
            or not math.isfinite(value)
        ):
            return sequence, None

        return sequence, float(value)

    # ========================================================
    # FUTURES
    # ========================================================

    @staticmethod
    def _wait_future(
        future,
        timeout: float,
    ):

        event = threading.Event()

        future.add_done_callback(
            lambda _: event.set()
        )

        if not event.wait(timeout):
            return None

        return future.result()

    # ========================================================
    # CONNECT
    # ========================================================

    def _connect(self):

        self._node.get_logger().info(
            "Connecting to SO-101..."
        )

        if not (
            self._move_client.wait_for_server(
                timeout_sec=(
                    self.config.server_timeout
                )
            )
        ):

            raise RuntimeError(
                "MoveIt action server "
                f"{self.config.move_action} "
                "not available."
            )

        if not (
            self._gripper_client.wait_for_server(
                timeout_sec=(
                    self.config.server_timeout
                )
            )
        ):

            raise RuntimeError(
                "Gripper action server "
                f"{self.config.gripper_action} "
                "not available."
            )

        deadline = (
            time.monotonic()
            + self.config.server_timeout
        )

        while (
            time.monotonic()
            < deadline
        ):

            if (
                self._current_joint_state()
                is not None
            ):
                break

            time.sleep(0.05)

        else:

            raise RuntimeError(
                "No joint states received."
            )

        self._node.get_logger().info(
            "SO-101 control ready."
        )

    def is_ready(self):

        return (
            self._move_client.server_is_ready()
            and
            self._gripper_client.server_is_ready()
        )

    # ========================================================
    # POSE
    # ========================================================

    def get_pose(
        self,
        timeout: float = 2.0,
    ) -> PoseMM:

        transform = (
            self._tf_buffer.lookup_transform(
                self.config.base_frame,
                self.config.end_effector_frame,
                Time(),
                timeout=Duration(
                    seconds=timeout
                ),
            )
        )

        position = (
            transform.transform.translation
        )

        rotation = (
            transform.transform.rotation
        )

        return PoseMM(
            x=position.x * 1000.0,
            y=position.y * 1000.0,
            z=position.z * 1000.0,

            qx=rotation.x,
            qy=rotation.y,
            qz=rotation.z,
            qw=rotation.w,
        )

    def get_gripper_current(self) -> float:

        current = self._get_joint_effort(
            self.config.gripper_joint
        )

        if current is None:

            raise RuntimeError(
                "Gripper current feedback is "
                "not available."
            )

        return abs(current)

    # ========================================================
    # ORIENTATION
    # ========================================================

    def lock_current_orientation(
        self,
    ) -> PoseMM:

        pose = self.get_pose()

        q = Quaternion()

        q.x = pose.qx
        q.y = pose.qy
        q.z = pose.qz
        q.w = pose.qw

        self._fixed_orientation = (
            _normalize_quaternion(q)
        )

        self._orientation_anchor_pose = pose

        self._orientation_anchor_heading = (
            math.atan2(
                pose.y,
                pose.x,
            )
        )

        self.config.constrain_orientation = True

        self._node.get_logger().info(
            "Tool orientation locked."
        )

        return pose

    def free_orientation(self):

        self.config.constrain_orientation = False

        self._node.get_logger().info(
            "Tool orientation is free."
        )

    def _candidate_orientations(
        self,
        x_mm: float,
        y_mm: float,
    ):
        if (
            self._fixed_orientation is None
            or self._orientation_anchor_heading is None
        ):
            self.lock_current_orientation()

        px = self.config.shoulder_pan_x_mm
        py = self.config.shoulder_pan_y_mm

        # Текущая точка, в которой была зафиксирована orientation
        anchor = self._orientation_anchor_pose

        current_heading = math.atan2(
            anchor.y - py,
            anchor.x - px,
        )

        target_heading = math.atan2(
            y_mm - py,
            x_mm - px,
        )

        yaw_delta = _normalize_angle(
            target_heading - current_heading
        )

        candidates = []

        for offset_deg in (
            self.config.yaw_search_deg
        ):

            candidate_yaw = (
                yaw_delta
                + math.radians(offset_deg)
            )

            candidate = _quat_multiply(
                _yaw_quaternion(candidate_yaw),
                self._fixed_orientation,
            )

            candidates.append(
                (candidate_yaw, candidate)
            )

        return candidates

    # ========================================================
    # PICK IK MODE
    # ========================================================

    def _pick_ik_rotation_scale_name(self):

        return (
            "robot_description_kinematics."
            f"{self.config.planning_group}."
            "rotation_scale"
        )

    def _get_pick_ik_rotation_scale(self):

        if not (
            self._move_group_parameters
            .wait_for_services(
                timeout_sec=(
                    self.config
                    .parameter_timeout
                )
            )
        ):

            raise RuntimeError(
                "MoveIt parameter services are "
                "not available."
            )

        response = self._wait_future(
            self._move_group_parameters
            .get_parameters([
                self._pick_ik_rotation_scale_name()
            ]),
            self.config.parameter_timeout,
        )

        if (
            response is None
            or not response.values
        ):

            raise RuntimeError(
                "Could not read PickIK "
                "rotation_scale."
            )

        value = response.values[0]

        if (
            value.type
            != Parameter.Type.DOUBLE.value
        ):

            return None

        return float(value.double_value)

    def _set_pick_ik_rotation_scale(
        self,
        value: float,
    ):

        response = self._wait_future(
            self._move_group_parameters
            .set_parameters([
                Parameter(
                    self._pick_ik_rotation_scale_name(),
                    Parameter.Type.DOUBLE,
                    float(value),
                )
            ]),
            self.config.parameter_timeout,
        )

        if (
            response is None
            or not response.results
            or not response.results[0].successful
        ):

            reason = "timeout"

            if (
                response is not None
                and response.results
            ):
                reason = response.results[0].reason

            raise RuntimeError(
                "Could not set PickIK "
                "rotation_scale: "
                f"{reason or 'rejected'}"
            )

    @contextmanager
    def _position_only_ik_mode(self):

        if (
            self.config.constrain_orientation
            or not self.config
            .manage_pick_ik_position_only
        ):

            yield
            return

        previous_scale = (
            self._get_pick_ik_rotation_scale()
        )

        # A solver other than PickIK may not expose this parameter.
        if previous_scale is None:

            yield
            return

        changed = not math.isclose(
            previous_scale,
            0.0,
            abs_tol=1e-12,
        )

        if changed:

            self._set_pick_ik_rotation_scale(
                0.0
            )

            self._node.get_logger().info(
                "PickIK position-only mode enabled."
            )

        try:
            yield
        finally:

            if changed:

                try:
                    self._set_pick_ik_rotation_scale(
                        previous_scale
                    )
                except Exception as exc:
                    self._node.get_logger().error(
                        "Could not restore PickIK "
                        "rotation_scale: "
                        f"{exc}"
                    )

    # ========================================================
    # MOVE GROUP
    # ========================================================

    def _build_move_goal(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        velocity: float,
        acceleration: float,
    ):

        stamp = (
            self._node
            .get_clock()
            .now()
            .to_msg()
        )

        position = PositionConstraint()

        position.header.frame_id = (
            self.config.base_frame
        )

        position.header.stamp = stamp

        position.link_name = (
            self.config.end_effector_frame
        )

        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [
            (
                float(
                    self.config
                    .position_goal_tolerance_mm
                )
                / 1000.0
            )
        ]

        region_pose = Pose()

        region_pose.position.x = (
            x_mm / 1000.0
        )

        region_pose.position.y = (
            y_mm / 1000.0
        )

        region_pose.position.z = (
            z_mm / 1000.0
        )

        region_pose.orientation.w = 1.0

        position.constraint_region.primitives = [
            region
        ]

        position.constraint_region.primitive_poses = [
            region_pose
        ]

        position.weight = 1.0

        goal_constraints = []

        wrist_roll_constraint = None

        if (
            not self.config.constrain_orientation
            and self.config.preserve_wrist_roll
        ):

            wrist_roll = self._get_joint_position(
                self.config.wrist_roll_joint
            )

            if wrist_roll is None:

                raise RuntimeError(
                    "No wrist roll joint state."
                )

            wrist_roll_constraint = (
                JointConstraint()
            )

            wrist_roll_constraint.joint_name = (
                self.config.wrist_roll_joint
            )

            wrist_roll_constraint.position = (
                wrist_roll
            )

            wrist_roll_constraint.tolerance_above = (
                self.config
                .wrist_roll_tolerance_rad
            )

            wrist_roll_constraint.tolerance_below = (
                self.config
                .wrist_roll_tolerance_rad
            )

            wrist_roll_constraint.weight = 1.0

        if self.config.constrain_orientation:

            for (
                yaw,
                quaternion,
            ) in self._candidate_orientations(
                x_mm,
                y_mm,
            ):

                constraints = Constraints()

                constraints.name = (
                    "so101_pose_target_"
                    f"{math.degrees(yaw):.1f}deg"
                )

                constraints.position_constraints = [
                    copy.deepcopy(position)
                ]

                orientation = (
                    OrientationConstraint()
                )

                orientation.header.frame_id = (
                    self.config.base_frame
                )

                orientation.header.stamp = stamp

                orientation.link_name = (
                    self.config.end_effector_frame
                )

                orientation.orientation = quaternion

                tolerance = (
                    float(
                        self.config
                        .orientation_tolerance_rad
                    )
                )

                orientation.absolute_x_axis_tolerance = (
                    tolerance
                )

                orientation.absolute_y_axis_tolerance = (
                    tolerance
                )

                orientation.absolute_z_axis_tolerance = (
                    tolerance
                )

                orientation.weight = 1.0

                constraints.orientation_constraints = [
                    orientation
                ]

                goal_constraints.append(
                    constraints
                )

        else:

            constraints = Constraints()

            constraints.name = (
                "so101_position_target"
            )

            constraints.position_constraints = [
                position
            ]

            if wrist_roll_constraint is not None:

                constraints.joint_constraints = [
                    wrist_roll_constraint
                ]

            goal_constraints.append(
                constraints
            )

        goal = MoveGroup.Goal()

        request = goal.request

        request.group_name = (
            self.config.planning_group
        )

        request.pipeline_id = (
            self.config.planning_pipeline
        )

        request.num_planning_attempts = (
            self.config.planning_attempts
        )

        request.allowed_planning_time = (
            self.config.planning_time
        )

        request.max_velocity_scaling_factor = (
            velocity
        )

        request.max_acceleration_scaling_factor = (
            acceleration
        )

        request.goal_constraints = (
            goal_constraints
        )

        if wrist_roll_constraint is not None:

            request.path_constraints.name = (
                "so101_hold_wrist_roll"
            )

            request.path_constraints.joint_constraints = [
                copy.deepcopy(
                    wrist_roll_constraint
                )
            ]

        request.start_state.is_diff = True

        options = (
            goal.planning_options
        )

        options.plan_only = False

        options.look_around = False

        options.replan = True

        options.replan_attempts = 2

        options.replan_delay = 0.2

        options.planning_scene_diff.is_diff = True

        options.planning_scene_diff.robot_state.is_diff = (
            True
        )

        return goal

    # ========================================================
    # VERIFY
    # ========================================================

    def _verify_position(
        self,
        x: float,
        y: float,
        z: float,
    ):

        deadline = (
            time.monotonic()
            + self.config.verify_timeout
        )

        last_pose = self.get_pose()

        while True:

            pose = self.get_pose()

            error = math.sqrt(
                (pose.x - x) ** 2
                + (pose.y - y) ** 2
                + (pose.z - z) ** 2
            )

            last_pose = pose

            if (
                error
                <=
                self.config
                .verify_position_tolerance_mm
            ):

                return (
                    True,
                    pose,
                    error,
                )

            if (
                time.monotonic()
                >= deadline
            ):

                return (
                    False,
                    last_pose,
                    error,
                )

            time.sleep(0.08)

    # ========================================================
    # MOVE
    # ========================================================

    def move(
        self,
        x: float,
        y: float,
        z: float,
        *,
        speed: float | None = None,
        acceleration: float | None = None,
        timeout: float | None = None,
    ) -> MotionResult:

        x = float(x)
        y = float(y)
        z = float(z)

        if not all(
            math.isfinite(v)
            for v in (
                x,
                y,
                z,
            )
        ):

            raise ValueError(
                "XYZ must be finite."
            )

        if max(
            abs(x),
            abs(y),
            abs(z),
        ) > 1000.0:

            raise ValueError(
                "XYZ are millimetres."
            )

        velocity = (
            self.config.velocity_scaling
            if speed is None
            else float(speed)
        )

        accel = (
            self.config.acceleration_scaling
            if acceleration is None
            else float(acceleration)
        )

        if not (
            0.0 < velocity <= 1.0
            and
            0.0 < accel <= 1.0
        ):

            raise ValueError(
                "Speed and acceleration scaling "
                "must be in (0, 1]."
            )

        position_tolerance = float(
            self.config
            .position_goal_tolerance_mm
        )

        if (
            not math.isfinite(position_tolerance)
            or position_tolerance <= 0.0
        ):

            raise ValueError(
                "Position goal tolerance "
                "must be positive."
            )

        if self.config.constrain_orientation:

            orientation_tolerance = float(
                self.config
                .orientation_tolerance_rad
            )

            if (
                not math.isfinite(
                    orientation_tolerance
                )
                or orientation_tolerance <= 0.0
                or orientation_tolerance > math.pi
            ):

                raise ValueError(
                    "Orientation tolerance must "
                    "be in (0, pi]."
                )

        if self.config.preserve_wrist_roll:

            wrist_tolerance = float(
                self.config
                .wrist_roll_tolerance_rad
            )

            if (
                not math.isfinite(wrist_tolerance)
                or wrist_tolerance <= 0.0
                or wrist_tolerance > math.pi
            ):

                raise ValueError(
                    "Wrist roll tolerance must "
                    "be in (0, pi]."
                )

        with (
            self._command_lock,
            self._position_only_ik_mode(),
        ):

            goal = (
                self._build_move_goal(
                    x,
                    y,
                    z,
                    velocity,
                    accel,
                )
            )

            orientation_mode = (
                "locked orientation"
                if self.config.constrain_orientation
                else "position-only, wrist roll held"
            )

            self._node.get_logger().info(
                f"MOVE → "
                f"x={x:.1f} "
                f"y={y:.1f} "
                f"z={z:.1f} mm "
                f"({orientation_mode})"
            )

            goal_handle = (
                self._wait_future(
                    self._move_client
                    .send_goal_async(goal),

                    self.config
                    .server_timeout,
                )
            )

            if (
                goal_handle is None
                or
                not goal_handle.accepted
            ):

                return MotionResult(
                    False,
                    "MoveIt rejected goal.",
                )

            with self._goal_lock:

                self._current_move_goal = (
                    goal_handle
                )

            command_timeout = (
                self.config.command_timeout
                if timeout is None
                else float(timeout)
            )

            try:

                wrapped_result = (
                    self._wait_future(
                        goal_handle
                        .get_result_async(),

                        command_timeout,
                    )
                )

                if wrapped_result is None:

                    goal_handle.cancel_goal_async()

                    return MotionResult(
                        False,
                        "Move timed out.",
                    )

                action_status = (
                    int(
                        wrapped_result.status
                    )
                )

                error_code = (
                    int(
                        wrapped_result
                        .result
                        .error_code
                        .val
                    )
                )

                success = (
                    action_status
                    ==
                    GoalStatus.STATUS_SUCCEEDED

                    and

                    error_code
                    ==
                    MoveItErrorCodes.SUCCESS
                )

                if not success:

                    return MotionResult(
                        False,
                        (
                            "MoveIt failed. "
                            f"error_code="
                            f"{error_code}, "
                            f"status="
                            f"{action_status}"
                        ),
                        error_code,
                    )

                (
                    verified,
                    pose,
                    xyz_error,
                ) = (
                    self._verify_position(
                        x,
                        y,
                        z,
                    )
                )

                if not verified:

                    return MotionResult(
                        False,
                        (
                            "MoveIt SUCCESS but "
                            "real pose differs: "
                            f"x={pose.x:.1f} "
                            f"y={pose.y:.1f} "
                            f"z={pose.z:.1f} mm, "
                            f"error="
                            f"{xyz_error:.1f} mm"
                        ),
                        error_code,
                    )

                return MotionResult(
                    True,
                    (
                        f"Reached "
                        f"x={pose.x:.1f} "
                        f"y={pose.y:.1f} "
                        f"z={pose.z:.1f} mm "
                        f"(error="
                        f"{xyz_error:.1f} mm)"
                    ),
                    error_code,
                )

            finally:

                with self._goal_lock:

                    if (
                        self._current_move_goal
                        is goal_handle
                    ):

                        self._current_move_goal = None

    # ========================================================
    # RELATIVE
    # ========================================================

    def move_relative(
        self,
        dx: float,
        dy: float,
        dz: float,
        **kwargs,
    ) -> MotionResult:

        pose = self.get_pose()

        return self.move(
            pose.x + float(dx),
            pose.y + float(dy),
            pose.z + float(dz),
            **kwargs,
        )

    # ========================================================
    # GRIPPER
    # ========================================================

    def _send_gripper_goal(
        self,
        goal: ParallelGripperCommand.Goal,
    ):

        attempts = max(
            1,
            int(self.config.gripper_goal_retries),
        )

        for attempt in range(attempts):

            goal_handle = self._wait_future(
                self._gripper_client.send_goal_async(goal),
                self.config.server_timeout,
            )

            # A timeout is ambiguous: the server may still accept the goal
            # later, so sending a duplicate would be unsafe.
            if goal_handle is None:
                return None

            if goal_handle.accepted:
                return goal_handle

            if attempt + 1 < attempts:

                if attempt == 0:
                    self._node.get_logger().warning(
                        "Gripper controller rejected a goal while starting; "
                        "waiting for it to become active."
                    )

                time.sleep(
                    max(
                        0.0,
                        float(
                            self.config
                            .gripper_goal_retry_delay
                        ),
                    )
                )

        return goal_handle

    def _gripper_hold_target(
        self,
        contact_position: float,
    ) -> float:

        closing_direction = math.copysign(
            1.0,
            (
                self.config
                .gripper_closed_position
                - self.config
                .gripper_open_position
            ),
        )

        preload = abs(
            float(
                self.config
                .gripper_grasp_preload_rad
            )
        )

        target = (
            float(contact_position)
            + closing_direction * preload
        )

        lower = min(
            self.config.gripper_open_position,
            self.config.gripper_closed_position,
        )

        upper = max(
            self.config.gripper_open_position,
            self.config.gripper_closed_position,
        )

        return min(
            upper,
            max(lower, target),
        )

    def _hold_gripper_after_contact(
        self,
        contact_position: float,
    ):

        hold_target = self._gripper_hold_target(
            contact_position
        )

        goal = ParallelGripperCommand.Goal()

        goal.command.name = [
            self.config.gripper_joint
        ]

        goal.command.position = [
            hold_target
        ]

        goal_handle = self._send_gripper_goal(goal)

        if (
            goal_handle is None
            or not goal_handle.accepted
        ):

            return (
                False,
                hold_target,
                "hold command was rejected",
            )

        with self._goal_lock:

            self._current_gripper_goal = (
                goal_handle
            )

        try:

            wrapped_result = self._wait_future(
                goal_handle.get_result_async(),
                self.config.gripper_hold_timeout,
            )

            if wrapped_result is None:

                goal_handle.cancel_goal_async()

                return (
                    False,
                    hold_target,
                    "hold command timed out",
                )

            result = wrapped_result.result

            held = bool(
                result.reached_goal
                or result.stalled
            )

            return (
                held,
                hold_target,
                (
                    "held"
                    if held
                    else "hold position was not reached"
                ),
            )

        finally:

            with self._goal_lock:

                if (
                    self._current_gripper_goal
                    is goal_handle
                ):

                    self._current_gripper_goal = None

    def set_gripper(
        self,
        position: float,
        *,
        timeout: float = 10.0,
        allow_stall: bool = False,
    ) -> MotionResult:

        target = float(position)

        with self._command_lock:

            before = (
                self._get_joint_position(
                    self.config.gripper_joint
                )
            )

            if before is None:

                return MotionResult(
                    False,
                    "No gripper state.",
                )

            self._node.get_logger().info(
                f"GRIPPER → "
                f"{target:.3f} rad "
                f"(position="
                f"{before:.3f})"
            )

            position_error = abs(before - target)

            if (
                position_error
                <= self.config.gripper_tolerance_rad
            ):

                return MotionResult(
                    True,
                    (
                        "Gripper is already at target "
                        f"({before:.3f} rad, "
                        f"error={position_error:.3f} rad)."
                    ),
                )

            goal = (
                ParallelGripperCommand.Goal()
            )

            goal.command.name = [
                self.config.gripper_joint
            ]

            goal.command.position = [
                target
            ]

            goal_handle = self._send_gripper_goal(goal)

            if (
                goal_handle is None
                or
                not goal_handle.accepted
            ):

                return MotionResult(
                    False,
                    "Gripper rejected command.",
                )

            with self._goal_lock:

                self._current_gripper_goal = (
                    goal_handle
                )

            try:

                wrapped_result = (
                    self._wait_future(
                        goal_handle
                        .get_result_async(),

                        timeout,
                    )
                )

                if wrapped_result is None:

                    goal_handle.cancel_goal_async()

                    return MotionResult(
                        False,
                        "Gripper timeout.",
                    )

                result = (
                    wrapped_result.result
                )

                time.sleep(0.1)

                after = (
                    self._get_joint_position(
                        self.config.gripper_joint
                    )
                )

                if after is None:

                    return MotionResult(
                        False,
                        "No gripper state.",
                    )

                reached = (
                    abs(
                        after
                        - target
                    )
                    <=
                    self.config
                    .gripper_tolerance_rad
                )

                moved = (
                    abs(after - before) > 0.01
                )

                if reached:

                    return MotionResult(
                        True,
                        (
                            f"Gripper reached "
                            f"{after:.3f}"
                        ),
                    )

                if (
                    allow_stall
                    and result.stalled
                    and moved
                ):

                    return MotionResult(
                        True,
                        (
                            "Gripper stalled after "
                            f"moving to {after:.3f}; "
                            "holding contact."
                        ),
                    )

                return MotionResult(
                    False,
                    (
                        "Gripper did not move "
                        "correctly. "
                        f"before={before:.3f}, "
                        f"after={after:.3f}, "
                        f"target={target:.3f}, "
                        f"stalled="
                        f"{bool(result.stalled)}"
                    ),
                )

            finally:

                with self._goal_lock:

                    if (
                        self._current_gripper_goal
                        is goal_handle
                    ):

                        self._current_gripper_goal = None

    def open(self):

        return self.set_gripper(
            self.config
            .gripper_open_position,
            allow_stall=False,
        )

    def _close_with_current_feedback(self):

        threshold = float(
            self.config
            .gripper_contact_current_amp
        )

        samples_required = int(
            self.config
            .gripper_contact_samples
        )

        poll_interval = float(
            self.config
            .gripper_current_poll_interval
        )

        timeout = float(
            self.config
            .gripper_close_timeout
        )

        if (
            not math.isfinite(threshold)
            or threshold <= 0.0
        ):
            raise ValueError(
                "Gripper contact current must "
                "be finite and positive."
            )

        if samples_required < 1:
            raise ValueError(
                "Gripper contact samples must "
                "be at least one."
            )

        if (
            not math.isfinite(poll_interval)
            or poll_interval <= 0.0
        ):
            raise ValueError(
                "Gripper current poll interval "
                "must be finite and positive."
            )

        if (
            not math.isfinite(timeout)
            or timeout <= 0.0
        ):
            raise ValueError(
                "Gripper close timeout must "
                "be finite and positive."
            )

        with self._command_lock:

            before = self._get_joint_position(
                self.config.gripper_joint
            )

            sequence, current = (
                self._get_joint_effort_sample(
                    self.config.gripper_joint
                )
            )

            if before is None:

                return MotionResult(
                    False,
                    "No gripper position state.",
                )

            if current is None:

                return MotionResult(
                    False,
                    (
                        "Current feedback is unavailable; "
                        "close refused for safety. Rebuild "
                        "and restart the Feetech driver."
                    ),
                )

            if abs(current) >= threshold:

                return MotionResult(
                    False,
                    (
                        "Gripper current is already high "
                        f"({abs(current):.3f} A); close "
                        "refused for safety."
                    ),
                )

            target = float(
                self.config
                .gripper_closed_position
            )

            goal = ParallelGripperCommand.Goal()

            goal.command.name = [
                self.config.gripper_joint
            ]

            goal.command.position = [target]

            goal_handle = self._send_gripper_goal(goal)

            if (
                goal_handle is None
                or not goal_handle.accepted
            ):

                return MotionResult(
                    False,
                    "Gripper rejected close command.",
                )

            with self._goal_lock:

                self._current_gripper_goal = (
                    goal_handle
                )

            result_future = (
                goal_handle.get_result_async()
            )

            deadline = time.monotonic() + timeout
            over_current_samples = 0
            last_sequence = sequence

            try:

                while time.monotonic() < deadline:

                    sequence, current = (
                        self._get_joint_effort_sample(
                            self.config.gripper_joint
                        )
                    )

                    if sequence != last_sequence:

                        last_sequence = sequence

                        if (
                            current is not None
                            and abs(current) >= threshold
                        ):
                            over_current_samples += 1
                        else:
                            over_current_samples = 0

                    if (
                        over_current_samples
                        >= samples_required
                    ):

                        contact_position = (
                            self._get_joint_position(
                                self.config.gripper_joint
                            )
                        )

                        goal_handle.cancel_goal_async()

                        if contact_position is None:

                            return MotionResult(
                                False,
                                "Contact detected, but no "
                                "gripper position is available.",
                            )

                        (
                            held,
                            hold_target,
                            hold_message,
                        ) = self._hold_gripper_after_contact(
                            contact_position
                        )

                        if not held:

                            return MotionResult(
                                False,
                                (
                                    "Current contact detected at "
                                    f"{abs(current):.3f} A, but "
                                    f"{hold_message}."
                                ),
                            )

                        return MotionResult(
                            True,
                            (
                                "Object/contact detected at "
                                f"{abs(current):.3f} A and "
                                f"{contact_position:.3f} rad; "
                                f"holding at {hold_target:.3f} rad."
                            ),
                        )

                    if result_future.done():

                        wrapped_result = (
                            result_future.result()
                        )

                        after = self._get_joint_position(
                            self.config.gripper_joint
                        )

                        if after is None:

                            return MotionResult(
                                False,
                                "No gripper position state.",
                            )

                        reached = (
                            abs(after - target)
                            <= self.config
                            .gripper_tolerance_rad
                        )

                        if (
                            reached
                            or wrapped_result
                            .result.reached_goal
                        ):

                            return MotionResult(
                                True,
                                (
                                    "Gripper closed fully; "
                                    "no contact threshold "
                                    "was reached."
                                ),
                            )

                        moved = abs(after - before) > 0.05

                        if (
                            wrapped_result.result.stalled
                            and moved
                        ):
                            (
                                held,
                                hold_target,
                                hold_message,
                            ) = self._hold_gripper_after_contact(
                                after
                            )

                            if held:
                                current_text = (
                                    f"{abs(current):.3f} A"
                                    if current is not None
                                    else "unavailable"
                                )

                                return MotionResult(
                                    True,
                                    (
                                        "Object/contact detected by "
                                        "controller stall at "
                                        f"{after:.3f} rad "
                                        f"(current={current_text}); "
                                        "holding at "
                                        f"{hold_target:.3f} rad."
                                    ),
                                )

                            return MotionResult(
                                False,
                                (
                                    "Gripper stalled on contact, but "
                                    f"{hold_message}."
                                ),
                            )

                        return MotionResult(
                            False,
                            (
                                "Gripper stopped without a "
                                "reliable current contact signal; "
                                "command aborted."
                            ),
                        )

                    time.sleep(poll_interval)

                goal_handle.cancel_goal_async()

                position = self._get_joint_position(
                    self.config.gripper_joint
                )

                if position is not None:
                    self._hold_gripper_after_contact(
                        position
                    )

                return MotionResult(
                    False,
                    "Gripper close timed out and was stopped.",
                )

            finally:

                with self._goal_lock:

                    if (
                        self._current_gripper_goal
                        is goal_handle
                    ):
                        self._current_gripper_goal = None

    def close(self):
        return self._close_with_current_feedback()

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        with self._goal_lock:

            move_goal = (
                self._current_move_goal
            )

            gripper_goal = (
                self._current_gripper_goal
            )

        if move_goal is not None:

            try:
                move_goal.cancel_goal_async()
            except Exception:
                pass

        if gripper_goal is not None:

            try:
                gripper_goal.cancel_goal_async()
            except Exception:
                pass

        self._node.get_logger().warning(
            "STOP requested."
        )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self):

        if self._shutdown:
            return

        self._shutdown = True

        self.stop()

        try:
            self._executor.shutdown(
                timeout_sec=2.0
            )
        except Exception:
            pass

        try:
            self._node.destroy_node()
        except Exception:
            pass

        try:
            self._context.shutdown()
        except Exception:
            pass

        if self._spin_thread.is_alive():

            self._spin_thread.join(
                timeout=2.0
            )

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.shutdown()

    def __del__(self):

        try:
            self.shutdown()
        except Exception:
            pass
