import os

# All ROS processes on both computers must use the same middleware.  The
# remote agent runs on Fast DDS, so keep the executor on Fast DDS as well.
# MoveIt and the controllers must be launched with the same setting.
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

import json
import math
import queue
import threading
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.utilities import get_rmw_implementation_identifier
from std_msgs.msg import String

from so101_control import SO101Arm


# ============================================================
# TOPICS
# ============================================================

COMMAND_TOPIC = "/voice/robot_command"
STATUS_TOPIC = "/robot_executor/status"


# ============================================================
# CALIBRATION
# ============================================================

ABSOLUTE_TARGET_X_OFFSET_MM = 25.0
ABSOLUTE_TARGET_Y_OFFSET_MM = 50.0
CAMERA_GRASP_Z_OFFSET_MM = 35.0


# Если робот физически приехал достаточно близко,
# не убиваем весь task.
MOVE_ACCEPT_ERROR_MM = 30.0
REL_ACCEPT_ERROR_MM = 30.0


# ============================================================
# ROBOT EXECUTOR
# ============================================================

class RobotExecutor(Node):

    def __init__(self):

        super().__init__("robot_executor")

        self.get_logger().info(
            "ROS middleware: "
            f"{get_rmw_implementation_identifier()}"
        )

        # ====================================================
        # ROBOT
        # ====================================================

        self.get_logger().info(
            "Initializing SO101Arm..."
        )

        self.arm = SO101Arm()

        self.get_logger().info(
            "SO101Arm initialized"
        )

        # ====================================================
        # ROS
        # ====================================================

        self.command_sub = self.create_subscription(
            String,
            COMMAND_TOPIC,
            self.command_callback,
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            STATUS_TOPIC,
            10,
        )

        # ====================================================
        # TASK QUEUE
        # ====================================================

        self.command_queue = queue.Queue(
            maxsize=10
        )

        self.worker_thread = threading.Thread(
            target=self.worker,
            daemon=True,
        )

        self.worker_thread.start()

        # ====================================================
        # LOG
        # ====================================================

        self.get_logger().info(
            f"Robot executor listening on {COMMAND_TOPIC}"
        )

        self.get_logger().info(
            f"Offsets: "
            f"X +{ABSOLUTE_TARGET_X_OFFSET_MM:.1f} mm, "
            f"Y +{ABSOLUTE_TARGET_Y_OFFSET_MM:.1f} mm"
        )

        self.get_logger().info(
            f"MOVE tolerance: "
            f"{MOVE_ACCEPT_ERROR_MM:.1f} mm"
        )

        self.get_logger().info(
            f"REL tolerance: "
            f"{REL_ACCEPT_ERROR_MM:.1f} mm"
        )

    # ========================================================
    # COMMAND CALLBACK
    # ========================================================

    def command_callback(self, msg: String):

        self.get_logger().info("")
        self.get_logger().info(
            "======================================"
        )
        self.get_logger().info(
            "NEW AGENT COMMAND"
        )
        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            f"RAW: {msg.data}"
        )

        try:

            command = json.loads(
                msg.data
            )

        except json.JSONDecodeError as e:

            self.get_logger().error(
                f"Invalid JSON: {e}"
            )

            self.publish_status(
                task_id=None,
                state="error",
                message="Invalid JSON",
            )

            return

        # ====================================================
        # IMMEDIATE STOP
        # ====================================================

        if command.get("action") == "stop":

            self.get_logger().warning(
                "STOP COMMAND RECEIVED"
            )

            try:

                self.arm.stop()

                self.publish_status(
                    task_id=command.get(
                        "task_id"
                    ),
                    state="stopped",
                )

            except Exception as e:

                self.get_logger().error(
                    f"STOP failed: {e}"
                )

            return

        # ====================================================
        # ADD TO QUEUE
        # ====================================================

        try:

            self.command_queue.put_nowait(
                command
            )

            self.get_logger().info(
                "Task added to queue"
            )

        except queue.Full:

            self.get_logger().error(
                "Command queue is full"
            )

            self.publish_status(
                task_id=command.get(
                    "task_id"
                ),
                state="error",
                message="Command queue full",
            )

    # ========================================================
    # WORKER
    # ========================================================

    def worker(self):

        while rclpy.ok():

            try:

                task = self.command_queue.get(
                    timeout=0.2
                )

            except queue.Empty:
                continue

            try:

                self.execute_task(
                    task
                )

            except Exception as e:

                task_id = task.get(
                    "task_id"
                )

                self.get_logger().error(
                    f"Task failed: {e}"
                )

                self.publish_status(
                    task_id=task_id,
                    state="error",
                    message=str(e),
                )

            finally:

                self.command_queue.task_done()

    # ========================================================
    # EXECUTE TASK
    # ========================================================

    def execute_task(self, task):

        task_id = task.get(
            "task_id",
            f"task_{uuid.uuid4().hex[:8]}"
        )

        if "steps" in task:
            steps = task["steps"]
        else:
            steps = [task]

        if not isinstance(
            steps,
            list
        ):
            raise ValueError(
                "'steps' must be a list"
            )

        self.get_logger().info("")
        self.get_logger().info(
            "======================================"
        )
        self.get_logger().info(
            f"START TASK: {task_id}"
        )
        self.get_logger().info(
            f"{len(steps)} steps"
        )
        self.get_logger().info(
            "======================================"
        )

        self.publish_status(
            task_id=task_id,
            state="running",
            step=0,
        )

        for index, step in enumerate(
            steps
        ):

            self.get_logger().info("")
            self.get_logger().info(
                "--------------------------------------"
            )

            self.get_logger().info(
                f"[{task_id}] "
                f"step {index + 1}/{len(steps)}: "
                f"{step}"
            )

            self.publish_status(
                task_id=task_id,
                state="running",
                step=index + 1,
            )

            self.execute_step(
                step
            )

        self.get_logger().info("")
        self.get_logger().info(
            "======================================"
        )
        self.get_logger().info(
            f"TASK COMPLETED: {task_id}"
        )
        self.get_logger().info(
            "======================================"
        )

        self.publish_status(
            task_id=task_id,
            state="done",
            step=len(steps),
        )

    # ========================================================
    # EXECUTE STEP
    # ========================================================

    def execute_step(self, step):

        action = step.get(
            "action"
        )

        if action is None:
            raise ValueError(
                "Step has no 'action'"
            )

        action = str(
            action
        ).strip().lower()

        # ====================================================
        # MOVE XYZ
        # ====================================================

        if action == "move":

            x_raw = float(
                step["x"]
            )

            y_raw = float(
                step["y"]
            )

            z_raw = float(
                step["z"]
            )

            frame = str(
                step.get("frame", "")
            ).strip().lower()

            # Camera detections describe the top of the object. Move the
            # gripper lower for the grasp, but never below table Z=0.
            if (
                frame == "camera"
                and math.isfinite(z_raw)
            ):
                z = max(
                    0.0,
                    z_raw - CAMERA_GRASP_Z_OFFSET_MM,
                )
            else:
                z = z_raw

            # calibration offsets

            x = (
                x_raw
                + ABSOLUTE_TARGET_X_OFFSET_MM
            )

            y = (
                y_raw
                + ABSOLUTE_TARGET_Y_OFFSET_MM
            )

            self.get_logger().info(
                f"MOVE RAW → "
                f"frame={frame or 'unspecified'} "
                f"x={x_raw:.1f} "
                f"y={y_raw:.1f} "
                f"z={z_raw:.1f}"
            )

            if frame == "camera":

                self.get_logger().info(
                    f"CAMERA GRASP Z → "
                    f"max(0.0, {z_raw:.1f} - "
                    f"{CAMERA_GRASP_Z_OFFSET_MM:.1f}) "
                    f"= {z:.1f} mm"
                )

            self.get_logger().info(
                f"MOVE CALIBRATED → "
                f"x={x:.1f} "
                f"y={y:.1f} "
                f"z={z:.1f}"
            )

            result = self.arm.move(
                x,
                y,
                z,
            )

            after = self.arm.get_pose()

            error = math.sqrt(
                (after.x - x) ** 2
                +
                (after.y - y) ** 2
                +
                (after.z - z) ** 2
            )

            self.get_logger().info(
                f"MOVE ACTUAL → "
                f"x={after.x:.1f} "
                f"y={after.y:.1f} "
                f"z={after.z:.1f}"
            )

            self.get_logger().info(
                f"MOVE ERROR → "
                f"{error:.1f} mm"
            )

            if result:

                self.get_logger().info(
                    "MOVE DONE ✓"
                )

                return

            # MoveIt мог реально приехать,
            # но arm.py посчитал error слишком большим.

            if (
                error
                <= MOVE_ACCEPT_ERROR_MM
            ):

                self.get_logger().warning(
                    f"MOVE accepted anyway ✓ "
                    f"(error={error:.1f} mm)"
                )

                return

            raise RuntimeError(
                f"MOVE failed: "
                f"{result.message}. "
                f"Real error={error:.1f} mm"
            )

        # ====================================================
        # MOVE XY
        # ====================================================

        elif action == "move_xy":

            x_raw = float(
                step["x"]
            )

            y_raw = float(
                step["y"]
            )

            # calibration offsets

            x = (
                x_raw
                + ABSOLUTE_TARGET_X_OFFSET_MM
            )

            y = (
                y_raw
                + ABSOLUTE_TARGET_Y_OFFSET_MM
            )

            # current real Z

            before = self.arm.get_pose()

            z = before.z

            self.get_logger().info(
                f"MOVE_XY RAW → "
                f"x={x_raw:.1f} "
                f"y={y_raw:.1f}"
            )

            self.get_logger().info(
                f"MOVE_XY CALIBRATED → "
                f"x={x:.1f} "
                f"y={y:.1f} "
                f"z stays {z:.1f}"
            )

            result = self.arm.move(
                x,
                y,
                z,
            )

            # Получаем реальную позу после движения

            after = self.arm.get_pose()

            error = math.sqrt(
                (after.x - x) ** 2
                +
                (after.y - y) ** 2
                +
                (after.z - z) ** 2
            )

            self.get_logger().info(
                f"MOVE_XY ACTUAL → "
                f"x={after.x:.1f} "
                f"y={after.y:.1f} "
                f"z={after.z:.1f}"
            )

            self.get_logger().info(
                f"MOVE_XY ERROR → "
                f"{error:.1f} mm"
            )

            # Нормальный success

            if result:

                self.get_logger().info(
                    "MOVE_XY DONE ✓"
                )

                return

            # MoveIt SUCCESS,
            # но arm.py заругался на pose difference.

            if (
                error
                <= MOVE_ACCEPT_ERROR_MM
            ):

                self.get_logger().warning(
                    f"MOVE_XY accepted anyway ✓ "
                    f"(error={error:.1f} mm <= "
                    f"{MOVE_ACCEPT_ERROR_MM:.1f} mm)"
                )

                return

            raise RuntimeError(
                f"MOVE_XY failed: "
                f"{result.message}. "
                f"Real error={error:.1f} mm"
            )

        # ====================================================
        # RELATIVE MOVE
        # ====================================================

        elif action == "rel":

            dx = float(
                step.get(
                    "dx",
                    0
                )
            )

            dy = float(
                step.get(
                    "dy",
                    0
                )
            )

            dz = float(
                step.get(
                    "dz",
                    0
                )
            )

            before = self.arm.get_pose()

            target_x = (
                before.x
                + dx
            )

            target_y = (
                before.y
                + dy
            )

            target_z = (
                before.z
                + dz
            )

            self.get_logger().info(
                f"REL → "
                f"dx={dx:.1f} "
                f"dy={dy:.1f} "
                f"dz={dz:.1f}"
            )

            self.get_logger().info(
                f"REL START → "
                f"x={before.x:.1f} "
                f"y={before.y:.1f} "
                f"z={before.z:.1f}"
            )

            self.get_logger().info(
                f"REL TARGET → "
                f"x={target_x:.1f} "
                f"y={target_y:.1f} "
                f"z={target_z:.1f}"
            )

            result = self.arm.move_relative(
                dx,
                dy,
                dz,
            )

            after = self.arm.get_pose()

            error = math.sqrt(
                (after.x - target_x) ** 2
                +
                (after.y - target_y) ** 2
                +
                (after.z - target_z) ** 2
            )

            self.get_logger().info(
                f"REL ACTUAL → "
                f"x={after.x:.1f} "
                f"y={after.y:.1f} "
                f"z={after.z:.1f}"
            )

            self.get_logger().info(
                f"REL ERROR → "
                f"{error:.1f} mm"
            )

            if result:

                self.get_logger().info(
                    "REL DONE ✓"
                )

                return

            if (
                error
                <= REL_ACCEPT_ERROR_MM
            ):

                self.get_logger().warning(
                    f"REL accepted anyway ✓ "
                    f"(error={error:.1f} mm <= "
                    f"{REL_ACCEPT_ERROR_MM:.1f} mm)"
                )

                return

            raise RuntimeError(
                f"REL failed: "
                f"{result.message}. "
                f"Real error={error:.1f} mm"
            )

        # ====================================================
        # OPEN
        # ====================================================

        elif action == "open":

            self.get_logger().info(
                "GRIPPER → OPEN"
            )

            result = self.arm.open()

            if not result:

                raise RuntimeError(
                    f"OPEN failed: "
                    f"{result.message}"
                )

            self.get_logger().info(
                "GRIPPER OPEN ✓"
            )

        # ====================================================
        # CLOSE
        # ====================================================

        elif action == "close":

            self.get_logger().info(
                "GRIPPER → CLOSE"
            )

            result = self.arm.close()

            if not result:

                raise RuntimeError(
                    f"CLOSE failed: "
                    f"{result.message}"
                )

            self.get_logger().info(
                "GRIPPER CLOSED ✓"
            )

        # ====================================================
        # POSE
        # ====================================================

        elif action == "pose":

            pose = self.arm.get_pose()

            self.get_logger().info(
                f"CURRENT POSE → "
                f"x={pose.x:.1f} "
                f"y={pose.y:.1f} "
                f"z={pose.z:.1f}"
            )

        # ====================================================
        # WAIT
        # ====================================================

        elif action == "wait":

            seconds = float(
                step.get(
                    "seconds",
                    1.0
                )
            )

            seconds = max(
                0.0,
                min(
                    seconds,
                    10.0
                )
            )

            self.get_logger().info(
                f"WAIT → {seconds:.2f}s"
            )

            time.sleep(
                seconds
            )

        # ====================================================
        # STOP
        # ====================================================

        elif action == "stop":

            self.get_logger().warning(
                "STOP"
            )

            self.arm.stop()

        # ====================================================
        # UNKNOWN
        # ====================================================

        else:

            raise ValueError(
                f"Unknown action: {action}"
            )

    # ========================================================
    # STATUS
    # ========================================================

    def publish_status(
        self,
        task_id,
        state,
        step=None,
        message=None,
    ):

        data = {
            "task_id": task_id,
            "state": state,
        }

        if step is not None:
            data["step"] = step

        if message is not None:
            data["message"] = message

        msg = String()

        msg.data = json.dumps(
            data
        )

        self.status_pub.publish(
            msg
        )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self):

        self.get_logger().info(
            "Shutting down robot executor..."
        )

        try:
            self.arm.stop()
        except Exception:
            pass

        try:
            self.arm.shutdown()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = RobotExecutor()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.shutdown()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":

    main()
