"""Interactive ArUco calibration for the tabletop coordinate system."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calibration_config import (
    ARUCO_DICTIONARY,
    MANIPULATOR_MARKER_HEIGHT_MM,
    MANIPULATOR_MARKER_ID,
    TABLE_HOMOGRAPHY_FILE,
    TABLE_MARKER_IDS,
    TABLE_MARKER_POSITIONS_MM,
    validate_aruco_measurements,
)

COLOR_TOPIC = "/camera/camera/color/image_raw"
WINDOW_NAME = "ArUco calibration"

SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def get_dictionary():
    dictionary_id = getattr(cv2.aruco, ARUCO_DICTIONARY)
    get_predefined = getattr(
        cv2.aruco,
        "getPredefinedDictionary",
        None,
    )
    if get_predefined is not None:
        return get_predefined(dictionary_id)

    dictionary_get = getattr(cv2.aruco, "Dictionary_get", None)
    if dictionary_get is not None:
        return dictionary_get(dictionary_id)

    raise RuntimeError("This OpenCV build has no ArUco dictionary factory")


def get_detector_parameters():
    create_parameters = getattr(
        cv2.aruco,
        "DetectorParameters_create",
        None,
    )
    if create_parameters is not None:
        return create_parameters()

    parameters_class = getattr(
        cv2.aruco,
        "DetectorParameters",
        None,
    )
    if parameters_class is not None:
        return parameters_class()

    raise RuntimeError("This OpenCV build has no ArUco detector parameters")


def image_to_bgr(msg):
    encoding = msg.encoding.lower()
    if encoding not in ("rgb8", "bgr8"):
        raise RuntimeError(f"Unsupported RGB encoding: {msg.encoding}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    rows = raw.reshape(msg.height, msg.step)
    frame = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if encoding == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(frame)


def marker_centers(corners, ids):
    centers = {}
    if ids is None:
        return centers
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        centers[int(marker_id)] = marker_corners.reshape(4, 2).mean(axis=0)
    return centers


class ArucoCalibrationNode(Node):
    def __init__(self):
        super().__init__("aruco_calibration")
        validate_aruco_measurements()
        self.dictionary = get_dictionary()
        self.parameters = get_detector_parameters()
        self.latest_frame = None
        self.latest_centers = {}
        self.saved = False
        self.subscription = self.create_subscription(
            Image,
            COLOR_TOPIC,
            self.image_callback,
            SENSOR_QOS,
        )
        self.get_logger().info(
            "ArUco calibration: place IDs 0-3 on the tabletop and ID 10 "
            "on the manipulator base. Press S to save, Q to quit."
        )

    def image_callback(self, msg):
        try:
            frame = image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().error(f"Image conversion error: {exc}")
            return

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.dictionary,
            parameters=self.parameters,
        )
        self.latest_frame = frame
        self.latest_centers = marker_centers(corners, ids)
        display = frame.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

        for marker_id, center in self.latest_centers.items():
            point = tuple(np.round(center).astype(int))
            label = str(marker_id)
            if marker_id == MANIPULATOR_MARKER_ID:
                label += f" | base Z~{MANIPULATOR_MARKER_HEIGHT_MM:.0f} mm"
            cv2.putText(
                display,
                label,
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        missing = [
            marker_id
            for marker_id in TABLE_MARKER_IDS
            if marker_id not in self.latest_centers
        ]
        status = "READY: press S to save" if not missing else (
            "Missing table IDs: " + ", ".join(map(str, missing))
        )
        cv2.putText(
            display,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0) if not missing else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            self.save_calibration()
        elif key == ord("q"):
            rclpy.shutdown()

    def save_calibration(self):
        missing = [
            marker_id
            for marker_id in TABLE_MARKER_IDS
            if marker_id not in self.latest_centers
        ]
        if missing or self.latest_frame is None:
            self.get_logger().warn(
                "Cannot save; missing table IDs: "
                + ", ".join(map(str, missing))
            )
            return

        source_points = np.array(
            [self.latest_centers[marker_id] for marker_id in TABLE_MARKER_IDS],
            dtype=np.float32,
        )
        target_points = np.array(
            [TABLE_MARKER_POSITIONS_MM[marker_id][:2] for marker_id in TABLE_MARKER_IDS],
            dtype=np.float32,
        )
        matrix, _ = cv2.findHomography(source_points, target_points, 0)
        if matrix is None:
            self.get_logger().error("Could not compute table homography")
            return

        payload = {
            "image_width": int(self.latest_frame.shape[1]),
            "image_height": int(self.latest_frame.shape[0]),
            "table_marker_ids": list(TABLE_MARKER_IDS),
            "homography_pixel_to_mm": matrix.tolist(),
        }
        TABLE_HOMOGRAPHY_FILE.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        self.saved = True
        self.get_logger().info(
            f"Saved table calibration to {TABLE_HOMOGRAPHY_FILE}"
        )

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
