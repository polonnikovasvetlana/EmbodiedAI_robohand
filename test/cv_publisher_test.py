import cv2
import json
import sys
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

# Allow direct execution from the test directory to use project settings.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calibration_config import TABLE_HOMOGRAPHY_FILE, origin_pixel

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)


# ============================================================
# SETTINGS
# ============================================================

COLOR_TOPIC = "/camera/camera/color/image_raw"
DETECTION_TOPIC = "/detected_objects"

CALIBRATION_CROSS_SIZE = 28
CALIBRATION_CROSS_THICKNESS = 5


# ============================================================
# QOS
# ============================================================

SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def calibration_origin_pixel(image_width, image_height):
    if TABLE_HOMOGRAPHY_FILE.is_file():
        try:
            payload = json.loads(
                TABLE_HOMOGRAPHY_FILE.read_text(encoding="utf-8")
            )
            matrix = np.asarray(
                payload["homography_pixel_to_mm"],
                dtype=np.float32
            )
            inverse = np.linalg.inv(matrix)
            origin = cv2.perspectiveTransform(
                np.array([[[0.0, 0.0]]], dtype=np.float32),
                inverse.astype(np.float32)
            )[0, 0]
            return tuple(np.round(origin).astype(int))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    return origin_pixel(image_width, image_height)


# ============================================================
# NODE
# ============================================================

class CVVisualizerNode(Node):

    def __init__(self):

        super().__init__(
            "cv_visualizer_node"
        )

        self.objects = []

        # ====================================================
        # SUBSCRIBERS
        # ====================================================

        self.image_subscription = (
            self.create_subscription(
                Image,
                COLOR_TOPIC,
                self.image_callback,
                SENSOR_QOS
            )
        )

        self.objects_subscription = (
            self.create_subscription(
                String,
                DETECTION_TOPIC,
                self.objects_callback,
                1
            )
        )

        self.get_logger().info(
            "CV visualizer started."
        )

        self.get_logger().info(
            f"RGB: {COLOR_TOPIC}"
        )

        self.get_logger().info(
            f"Detections: {DETECTION_TOPIC}"
        )


    # ========================================================
    # DETECTIONS
    # ========================================================

    def objects_callback(self, msg):

        try:

            data = json.loads(
                msg.data
            )

            self.objects = data.get(
                "objects",
                []
            )

        except Exception as e:

            self.get_logger().error(
                f"Detection message error: {e}"
            )


    # ========================================================
    # IMAGE
    # ========================================================

    def image_callback(self, msg):

        try:

            frame = self.image_to_numpy(
                msg
            )

        except Exception as e:

            self.get_logger().error(
                f"Image conversion error: {e}"
            )

            return

        # ====================================================
        # DRAW DETECTIONS
        # ====================================================

        for obj in self.objects:

            object_id = obj.get(
                "id",
                -1
            )

            object_type = obj.get(
                "type",
                "object"
            )

            color = obj.get(
                "color",
                "unknown"
            )

            height = obj.get(
                "height_mm",
                0.0
            )

            cx, cy = obj.get(
                "position",
                [0, 0]
            )

            # =================================================
            # CONTOUR
            # =================================================

            contour_data = obj.get(
                "contour",
                []
            )

            if contour_data:

                contour = np.asarray(
                    contour_data,
                    dtype=np.int32
                ).reshape(
                    -1,
                    1,
                    2
                )

                cv2.drawContours(
                    frame,
                    [contour],
                    -1,
                    (0, 255, 0),
                    2
                )

                x, y, w, h = (
                    cv2.boundingRect(
                        contour
                    )
                )

            else:

                x = cx - 30
                y = cy - 30
                w = 60
                h = 60

            # =================================================
            # CENTER
            # =================================================

            cv2.circle(
                frame,
                (int(cx), int(cy)),
                5,
                (0, 0, 255),
                -1
            )

            # =================================================
            # TEXT
            # =================================================

            cv2.putText(
                frame,
                f"Object {object_id}",
                (
                    x,
                    max(y - 30, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{color} {object_type}",
                (
                    x,
                    max(y - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"position: ({cx}, {cy})",
                (
                    x,
                    y + h + 18
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )

            position_mm = obj.get(
                "position_mm",
                [0.0, 0.0]
            )

            cv2.putText(
                frame,
                f"mm: X={position_mm[0]:.1f} Y={position_mm[1]:.1f}",
                (
                    x,
                    y + h + 52
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )

            cv2.putText(
                frame,
                f"height: {height:.0f} mm",
                (
                    x,
                    y + h + 69
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )

        calibration_point = calibration_origin_pixel(
            msg.width,
            msg.height
        )

        cv2.drawMarker(
            frame,
            calibration_point,
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            CALIBRATION_CROSS_SIZE,
            CALIBRATION_CROSS_THICKNESS
        )

        cv2.putText(
            frame,
            "0.0",
            (
                calibration_point[0] + 12,
                calibration_point[1] - 12
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # ====================================================
        # WINDOW
        # ====================================================

        cv2.imshow(
            "Object detection",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            if rclpy.ok():
                rclpy.shutdown()


    # ========================================================
    # ROS IMAGE -> OPENCV
    # ========================================================

    def image_to_numpy(self, msg):

        encoding = msg.encoding.lower()

        if encoding not in (
            "rgb8",
            "bgr8"
        ):

            raise RuntimeError(
                f"Unsupported image encoding: "
                f"{msg.encoding}"
            )

        raw = np.frombuffer(
            msg.data,
            dtype=np.uint8
        )

        rows = raw.reshape(
            msg.height,
            msg.step
        )

        frame = rows[
            :,
            :msg.width * 3
        ].reshape(
            msg.height,
            msg.width,
            3
        )

        # RealSense обычно публикует RGB8,
        # OpenCV ожидает BGR
        if encoding == "rgb8":

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

        else:

            frame = frame.copy()

        return frame


    # ========================================================
    # DESTROY
    # ========================================================

    def destroy_node(self):

        cv2.destroyAllWindows()

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = CVVisualizerNode()

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