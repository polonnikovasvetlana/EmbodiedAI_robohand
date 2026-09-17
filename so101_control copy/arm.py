from __future__ import annotations

import copy
import math
import os
import threading
import time
from dataclasses import dataclass

import rclpy

from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
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

    joint_states_topic: str = "/follower/joint_states"

    planning_pipeline: str = "ompl"

    planning_time: float = 5.0
    planning_attempts: int = 10

    velocity_scaling: float = 0.20
    acceleration_scaling: float = 0.20

    ik_timeout: float = 0.8

    arm_joints: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )

    joint_tolerance_rad: float = 0.01

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
    verify_position_tolerance_mm: float = 15.0
    verify_timeout: float = 1.5

    gripper_action: str = (
        "/follower/gripper_controller/gripper_cmd"
    )

    gripper_joint: str = "gripper"

    gripper_open_position: float = 1.50
    gripper_closed_position: float = -0.16

    gripper_tolerance_rad: float = 0.03

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

        self._ik_client = (
            self._node.create_client(
                GetPositionIK,
                self.config.ik_service,
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

        # Сохраняем текущий наклон инструмента.
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
            self._ik_client.wait_for_service(
                timeout_sec=(
                    self.config.server_timeout
                )
            )
        ):

            raise RuntimeError(
                "IK service "
                f"{self.config.ik_service} "
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
            self._ik_client.service_is_ready()
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

        self._node.get_logger().info(
            "Tool orientation locked."
        )

        return pose

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

        orientation = _quat_multiply(
            _yaw_quaternion(yaw_delta),
            self._fixed_orientation,
        )

        return [
            (yaw_delta, orientation)
        ]

    # ========================================================
    # IK
    # ========================================================

    def _solve_ik(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
    ):

        joint_state = (
            self._current_joint_state()
        )

        if joint_state is None:

            return None

        last_error = None

        candidates = (
            self._candidate_orientations(
                x_mm,
                y_mm,
            )
        )

        for (
            yaw,
            orientation,
        ) in candidates:

            request = (
                GetPositionIK.Request()
            )

            ik = request.ik_request

            ik.group_name = (
                self.config.planning_group
            )

            ik.ik_link_name = (
                self.config.end_effector_frame
            )

            ik.robot_state.joint_state = (
                copy.deepcopy(
                    joint_state
                )
            )

            ik.robot_state.is_diff = False

            ik.avoid_collisions = True

            ik.timeout = (
                Duration(
                    seconds=(
                        self.config.ik_timeout
                    )
                ).to_msg()
            )

            ik.pose_stamped.header.frame_id = (
                self.config.base_frame
            )

            ik.pose_stamped.header.stamp = (
                self._node
                .get_clock()
                .now()
                .to_msg()
            )

            ik.pose_stamped.pose.position.x = (
                x_mm / 1000.0
            )

            ik.pose_stamped.pose.position.y = (
                y_mm / 1000.0
            )

            ik.pose_stamped.pose.position.z = (
                z_mm / 1000.0
            )

            ik.pose_stamped.pose.orientation = (
                orientation
            )

            self._node.get_logger().info(
                f"IK → "
                f"x={x_mm:.1f} "
                f"y={y_mm:.1f} "
                f"z={z_mm:.1f} mm "
                f"yawΔ={math.degrees(yaw):.1f}°"
            )

            response = (
                self._wait_future(
                    self._ik_client.call_async(
                        request
                    ),
                    self.config.ik_timeout
                    + 1.0,
                )
            )

            if response is None:

                last_error = "timeout"
                continue

            error_code = (
                int(
                    response.error_code.val
                )
            )

            last_error = error_code

            if (
                error_code
                != MoveItErrorCodes.SUCCESS
            ):
                continue

            solution = dict(
                zip(
                    response
                    .solution
                    .joint_state
                    .name,

                    response
                    .solution
                    .joint_state
                    .position,
                )
            )

            joint_targets = {}

            for name in (
                self.config.arm_joints
            ):

                if name not in solution:
                    break

                joint_targets[name] = (
                    float(
                        solution[name]
                    )
                )

            if (
                len(joint_targets)
                != len(
                    self.config.arm_joints
                )
            ):
                continue

            self._node.get_logger().info(
                "IK SUCCESS "
                f"yawΔ="
                f"{math.degrees(yaw):.1f}°"
            )

            return (
                joint_targets,
                yaw,
            )

        self._node.get_logger().warning(
            "IK FAILED for all yaw "
            f"candidates. "
            f"last_error={last_error}"
        )

        return None

    # ========================================================
    # MOVE GROUP
    # ========================================================

    def _build_move_goal(
        self,
        joint_targets,
        velocity: float,
        acceleration: float,
    ):

        constraints = Constraints()

        constraints.name = (
            "so101_ik_target"
        )

        for name in (
            self.config.arm_joints
        ):

            constraint = (
                JointConstraint()
            )

            constraint.joint_name = name

            constraint.position = (
                joint_targets[name]
            )

            constraint.tolerance_above = (
                self.config
                .joint_tolerance_rad
            )

            constraint.tolerance_below = (
                self.config
                .joint_tolerance_rad
            )

            constraint.weight = 1.0

            constraints.joint_constraints.append(
                constraint
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

        request.goal_constraints = [
            constraints
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

        with self._command_lock:

            ik_result = (
                self._solve_ik(
                    x,
                    y,
                    z,
                )
            )

            if ik_result is None:

                return MotionResult(
                    False,
                    (
                        "IK failed for all "
                        "allowed yaw orientations."
                    ),
                    -31,
                )

            (
                joint_targets,
                used_yaw,
            ) = ik_result

            goal = (
                self._build_move_goal(
                    joint_targets,
                    velocity,
                    accel,
                )
            )

            self._node.get_logger().info(
                f"MOVE → "
                f"x={x:.1f} "
                f"y={y:.1f} "
                f"z={z:.1f} mm "
                f"yawΔ="
                f"{math.degrees(used_yaw):.1f}°"
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

            self._node.get_logger().info(
                f"GRIPPER → "
                f"{target:.3f} rad "
                f"(current="
                f"{before:.3f})"
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

            goal_handle = (
                self._wait_future(
                    self._gripper_client
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
                    abs(
                        after
                        - before
                    )
                    > 0.01
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
                    and
                    result.stalled
                    and
                    moved
                ):

                    return MotionResult(
                        True,
                        (
                            "Gripper stalled "
                            "after moving."
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

    def close(self):

        return self.set_gripper(
            self.config
            .gripper_closed_position,
            allow_stall=True,
        )

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