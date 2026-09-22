import json
import re
import statistics
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


# ============================================================
# TOPICS
# ============================================================

DETECTION_TOPIC = "/detected_objects"
COMMAND_TOPIC = "/agent/robot_command"


# ============================================================
# DETECTION SETTINGS
# ============================================================

SAMPLES_N = 5


# ============================================================
# OBJECTS
# ============================================================

PICK_CLASS = "mouse"
PICK_COLOR = "blue"

TARGET_CLASS = "traffic light"
TARGET_COLOR = "blue"


# ============================================================
# MOVEMENT SETTINGS
# ============================================================

# Для захвата:
#
# grasp_z = object_z - 20 mm
#
# Но никогда не ниже 0.
GRASP_OFFSET_MM = 20.0


# После захвата подняться
LIFT_MM = 190.0


# Перед отпусканием опуститься
DROP_DOWN_MM = 50.0


# ============================================================
# EXECUTOR OFFSETS
# ============================================================
#
# robot_executor.py делает:
#
# real_x = command_x + 30
# real_y = command_y + 40
#
# Поэтому для финальной абсолютной позы
# надо компенсировать эти offsets.
# ============================================================

EXECUTOR_X_OFFSET_MM = 30.0
EXECUTOR_Y_OFFSET_MM = 40.0


# ============================================================
# FINAL / HOME POSE
# ============================================================
#
# Это РЕАЛЬНАЯ поза робота, куда хотим приехать:
#
# SO101 > pose
# x=-78.7
# y=6.7
# z=474.7
#
# ============================================================

HOME_ROBOT_X_MM = -78.7
HOME_ROBOT_Y_MM = 6.7
HOME_ROBOT_Z_MM = 474.7


# Что реально надо отправить executor'у,
# чтобы после его offsets получить HOME pose.

HOME_COMMAND_X_MM = (
    HOME_ROBOT_X_MM
    - EXECUTOR_X_OFFSET_MM
)

HOME_COMMAND_Y_MM = (
    HOME_ROBOT_Y_MM
    - EXECUTOR_Y_OFFSET_MM
)

HOME_COMMAND_Z_MM = (
    HOME_ROBOT_Z_MM
)


# ============================================================
# HELPERS
# ============================================================

def parse_color(
    detection_id: str
) -> str:

    """
    Example:

    object_id=3:color=blue:distance_mm=640.0
    """

    match = re.search(
        r"color=([^:]+)",
        detection_id
    )

    if match is None:
        return ""

    return (
        match.group(1)
        .strip()
        .lower()
    )


def parse_distance_mm(
    detection_id: str
):

    match = re.search(
        r"distance_mm=([-+]?[0-9]*\.?[0-9]+)",
        detection_id
    )

    if match is None:
        return None

    return float(
        match.group(1)
    )


def get_detection_data(det):

    if len(det.results) == 0:
        return None

    result = det.results[0]

    # --------------------------------------------------------
    # Class
    # --------------------------------------------------------

    class_name = (
        result.hypothesis.class_id
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    score = float(
        result.hypothesis.score
    )

    # --------------------------------------------------------
    # Position
    #
    # CV publishes metres.
    # Robot works in mm.
    # --------------------------------------------------------

    pos = (
        result.pose.pose.position
    )

    x_mm = (
        float(pos.x)
        * 1000.0
    )

    y_mm = (
        float(pos.y)
        * 1000.0
    )

    z_mm = (
        float(pos.z)
        * 1000.0
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    color = parse_color(
        det.id
    )

    distance_mm = parse_distance_mm(
        det.id
    )

    return {

        "class":
            class_name,

        "color":
            color,

        "score":
            score,

        "x_mm":
            x_mm,

        "y_mm":
            y_mm,

        "z_mm":
            z_mm,

        "distance_mm":
            distance_mm,

        "id_raw":
            det.id,
    }


# ============================================================
# FAKE AGENT
# ============================================================

class FakeAgent(Node):

    def __init__(self):

        super().__init__(
            "fake_agent"
        )

        # ====================================================
        # CV SUBSCRIBER
        # ====================================================

        self.det_sub = self.create_subscription(
            Detection2DArray,
            DETECTION_TOPIC,
            self.detection_callback,
            10,
        )

        # ====================================================
        # COMMAND PUBLISHER
        # ====================================================

        self.command_pub = self.create_publisher(
            String,
            COMMAND_TOPIC,
            10,
        )

        # ====================================================
        # STATE
        # ====================================================

        self.pick_samples = []
        self.target_samples = []

        self.sent = False

        self.message_count = 0

        self.last_msg_time = None

        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        self.create_timer(
            2.0,
            self.diagnostic_timer,
        )

        # ====================================================
        # START LOG
        # ====================================================

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "FAKE AGENT STARTED"
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            f"PICK: "
            f"{PICK_CLASS} {PICK_COLOR}"
        )

        self.get_logger().info(
            f"TARGET: "
            f"{TARGET_CLASS} {TARGET_COLOR}"
        )

        self.get_logger().info(
            f"GRASP OFFSET: "
            f"{GRASP_OFFSET_MM:.1f} mm"
        )

        self.get_logger().info(
            f"LIFT: "
            f"{LIFT_MM:.1f} mm"
        )

        self.get_logger().info(
            f"DROP DOWN: "
            f"{DROP_DOWN_MM:.1f} mm"
        )

        self.get_logger().info(
            ""
        )

        self.get_logger().info(
            "FINAL REAL ROBOT POSE:"
        )

        self.get_logger().info(
            f"X={HOME_ROBOT_X_MM:.1f} "
            f"Y={HOME_ROBOT_Y_MM:.1f} "
            f"Z={HOME_ROBOT_Z_MM:.1f}"
        )

        self.get_logger().info(
            "Command sent to executor:"
        )

        self.get_logger().info(
            f"X={HOME_COMMAND_X_MM:.1f} "
            f"Y={HOME_COMMAND_Y_MM:.1f} "
            f"Z={HOME_COMMAND_Z_MM:.1f}"
        )

        self.get_logger().info(
            ""
        )

        self.get_logger().info(
            "Waiting for CV..."
        )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def diagnostic_timer(self):

        publishers = (
            self.get_publishers_info_by_topic(
                DETECTION_TOPIC
            )
        )

        types = [
            info.topic_type
            for info
            in publishers
        ]

        self.get_logger().info(
            f"[DIAG] "
            f"publishers={len(publishers)}, "
            f"types={types}, "
            f"messages={self.message_count}"
        )

        if len(publishers) == 0:

            self.get_logger().warning(
                "NO CV publisher"
            )

            return

        if self.message_count == 0:

            self.get_logger().warning(
                "CV publisher exists, "
                "but no messages received"
            )

    # ========================================================
    # DETECTION CALLBACK
    # ========================================================

    def detection_callback(
        self,
        msg: Detection2DArray,
    ):

        # Уже один раз отправили task.
        if self.sent:
            return

        self.message_count += 1

        self.last_msg_time = (
            time.time()
        )

        self.get_logger().info(
            ""
        )

        self.get_logger().info(
            "--------------------------------------"
        )

        self.get_logger().info(
            f"CV FRAME #{self.message_count}"
        )

        # ====================================================
        # READ ALL OBJECTS
        # ====================================================

        objects = []

        for i, det in enumerate(
            msg.detections
        ):

            data = get_detection_data(
                det
            )

            if data is None:
                continue

            objects.append(
                data
            )

            self.get_logger().info(
                f"#{i + 1}: "
                f"{data['class']} "
                f"{data['color']} "
                f"score={data['score']:.3f}"
            )

            self.get_logger().info(
                f"    "
                f"X={data['x_mm']:.1f} "
                f"Y={data['y_mm']:.1f} "
                f"Z={data['z_mm']:.1f}"
            )

        # ====================================================
        # FIND PICK OBJECT
        # ====================================================

        pick_object = None

        for obj in objects:

            if (
                obj["class"]
                == PICK_CLASS
                and
                obj["color"]
                == PICK_COLOR
            ):

                pick_object = obj

                break

        # ====================================================
        # FIND TARGET
        # ====================================================

        target_object = None

        for obj in objects:

            if (
                obj["class"]
                == TARGET_CLASS
                and
                obj["color"]
                == TARGET_COLOR
            ):

                target_object = obj

                break

        # ====================================================
        # LOG DETECTION
        # ====================================================

        if pick_object is None:

            self.get_logger().warning(
                "BLUE MOUSE NOT FOUND"
            )

        else:

            self.get_logger().info(
                "✓ BLUE MOUSE FOUND"
            )

        if target_object is None:

            self.get_logger().warning(
                "BLUE TRAFFIC LIGHT NOT FOUND"
            )

        else:

            self.get_logger().info(
                "✓ BLUE TRAFFIC LIGHT FOUND"
            )

        # Need both objects.

        if (
            pick_object is None
            or
            target_object is None
        ):

            return

        # ====================================================
        # COLLECT SAMPLES
        # ====================================================

        self.pick_samples.append(
            (
                pick_object["x_mm"],
                pick_object["y_mm"],
                pick_object["z_mm"],
            )
        )

        self.target_samples.append(
            (
                target_object["x_mm"],
                target_object["y_mm"],
                target_object["z_mm"],
            )
        )

        # Keep last N

        self.pick_samples = (
            self.pick_samples[
                -SAMPLES_N:
            ]
        )

        self.target_samples = (
            self.target_samples[
                -SAMPLES_N:
            ]
        )

        count = len(
            self.pick_samples
        )

        self.get_logger().info(
            f"SAMPLE {count}/{SAMPLES_N}"
        )

        if count < SAMPLES_N:
            return

        # ====================================================
        # MEDIAN PICK POSITION
        # ====================================================

        pick_x = statistics.median(
            p[0]
            for p
            in self.pick_samples
        )

        pick_y = statistics.median(
            p[1]
            for p
            in self.pick_samples
        )

        object_z = statistics.median(
            p[2]
            for p
            in self.pick_samples
        )

        # ====================================================
        # CALCULATE GRASP Z
        # ====================================================

        raw_grasp_z = (
            object_z
            - GRASP_OFFSET_MM
        )

        # Safety:
        # never command negative Z

        grasp_z = max(
            0.0,
            raw_grasp_z,
        )

        # ====================================================
        # MEDIAN TARGET POSITION
        # ====================================================

        target_x = statistics.median(
            p[0]
            for p
            in self.target_samples
        )

        target_y = statistics.median(
            p[1]
            for p
            in self.target_samples
        )

        target_z = statistics.median(
            p[2]
            for p
            in self.target_samples
        )

        # ====================================================
        # LOG FINAL POSITIONS
        # ====================================================

        self.get_logger().info(
            ""
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "STABLE COORDINATES"
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            f"MOUSE CV → "
            f"X={pick_x:.1f} "
            f"Y={pick_y:.1f} "
            f"Z={object_z:.1f}"
        )

        self.get_logger().info(
            f"MOUSE GRASP → "
            f"X={pick_x:.1f} "
            f"Y={pick_y:.1f} "
            f"Z={grasp_z:.1f}"
        )

        self.get_logger().info(
            f"GRASP Z calculation: "
            f"{object_z:.1f} - "
            f"{GRASP_OFFSET_MM:.1f} = "
            f"{raw_grasp_z:.1f}"
        )

        if raw_grasp_z < 0:

            self.get_logger().warning(
                f"Grasp Z would be "
                f"{raw_grasp_z:.1f} mm"
            )

            self.get_logger().warning(
                "Clamped to Z=0.0 mm"
            )

        self.get_logger().info(
            f"TARGET → "
            f"X={target_x:.1f} "
            f"Y={target_y:.1f} "
            f"Z={target_z:.1f}"
        )

        # ====================================================
        # BUILD TASK
        # ====================================================
        #
        # 1 OPEN
        #
        # 2 MOVE mouse XYZ
        #
        # 3 CLOSE
        #
        # 4 REL +150 Z
        #
        # 5 MOVE_XY traffic light
        #
        # 6 REL -50 Z
        #
        # 7 OPEN
        #
        # 8 MOVE HOME
        #
        # ====================================================

        task = {

            "task_id":
                "fake_pick_mouse_to_traffic_light",

            "steps": [

                # =================================================
                # 1. OPEN GRIPPER
                # =================================================

                {
                    "action":
                        "open"
                },

                # =================================================
                # 2. MOVE TO MOUSE
                # =================================================

                {
                    "action":
                        "move",

                    "x":
                        round(
                            pick_x,
                            1
                        ),

                    "y":
                        round(
                            pick_y,
                            1
                        ),

                    "z":
                        round(
                            grasp_z,
                            1
                        ),
                },

                # =================================================
                # 3. CLOSE
                # =================================================

                {
                    "action":
                        "close"
                },

                # =================================================
                # 4. LIFT OBJECT
                # =================================================

                {
                    "action":
                        "rel",

                    "dx":
                        0,

                    "dy":
                        0,

                    "dz":
                        LIFT_MM,
                },

                # =================================================
                # 5. MOVE ABOVE TRAFFIC LIGHT
                # =================================================

                {
                    "action":
                        "move_xy",

                    "x":
                        round(
                            target_x,
                            1
                        ),

                    "y":
                        round(
                            target_y,
                            1
                        ),
                },

                # =================================================
                # 6. LOWER
                # =================================================

                {
                    "action":
                        "rel",

                    "dx":
                        0,

                    "dy":
                        0,

                    "dz":
                        -DROP_DOWN_MM,
                },

                # =================================================
                # 7. RELEASE OBJECT
                # =================================================

                {
                    "action":
                        "open"
                },

                # =================================================
                # 8. GO TO FINAL / HOME POSITION
                #
                # IMPORTANT:
                #
                # Executor adds:
                #
                # X + 30
                # Y + 40
                #
                # Therefore we send compensated coordinates.
                #
                # command:
                #   X = -108.7
                #   Y =  -33.3
                #   Z =  474.7
                #
                # actual robot target:
                #   X = -78.7
                #   Y =   6.7
                #   Z = 474.7
                # =================================================

                {
                    "action":
                        "move",

                    "x":
                        round(
                            HOME_COMMAND_X_MM,
                            1
                        ),

                    "y":
                        round(
                            HOME_COMMAND_Y_MM,
                            1
                        ),

                    "z":
                        round(
                            HOME_COMMAND_Z_MM,
                            1
                        ),
                },
            ]
        }

        # ====================================================
        # LOG TASK
        # ====================================================

        self.get_logger().info(
            ""
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "FINAL TASK"
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            json.dumps(
                task,
                indent=2
            )
        )

        self.get_logger().info(
            ""
        )

        self.get_logger().info(
            "FINAL HOME MOVE:"
        )

        self.get_logger().info(
            f"command → "
            f"X={HOME_COMMAND_X_MM:.1f} "
            f"Y={HOME_COMMAND_Y_MM:.1f} "
            f"Z={HOME_COMMAND_Z_MM:.1f}"
        )

        self.get_logger().info(
            f"expected real pose → "
            f"X={HOME_ROBOT_X_MM:.1f} "
            f"Y={HOME_ROBOT_Y_MM:.1f} "
            f"Z={HOME_ROBOT_Z_MM:.1f}"
        )

        # ====================================================
        # SEND TASK
        # ====================================================

        msg_out = String()

        msg_out.data = json.dumps(
            task
        )

        self.command_pub.publish(
            msg_out
        )

        # ====================================================
        # BLOCK REPEATED SEND
        # ====================================================

        self.sent = True

        self.get_logger().info(
            ""
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "✓ TASK SENT"
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "Fake agent will not send again"
        )


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = FakeAgent()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()