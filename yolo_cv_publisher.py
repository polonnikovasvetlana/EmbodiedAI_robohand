"""ROS 2 publisher that combines YOLO detections with RealSense depth.

The node is a drop-in alternative to ``cv_publisher.py``: it subscribes to
the same color/depth topics and publishes vision_msgs to ``/detected_objects``.
"""

import json
import time
from pathlib import Path

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
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from calibration_config import TABLE_HOMOGRAPHY_FILE


COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
DETECTION_TOPIC = "/detected_objects"

CONFIDENCE = 0.40
IOU = 0.45
PROCESS_EVERY_N_FRAMES = 3

# Ignore invalid depth and obvious RealSense outliers when calculating range.
MIN_DEPTH_MM = 100.0
MAX_DEPTH_MM = 10000.0
CALIBRATION_FRAMES = 30

SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def default_model_path():
    """Return only the project-trained weights; never fall back to COCO."""
    model_path = Path(__file__).resolve().parent / "best.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Custom YOLO weights not found: {model_path}")
    return str(model_path)


def image_to_numpy(msg):
    """Convert a ROS RGB image to contiguous BGR for OpenCV/YOLO."""
    encoding = msg.encoding.lower()
    if encoding not in ("rgb8", "bgr8"):
        raise RuntimeError(f"Unsupported RGB encoding: {msg.encoding}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    rows = raw.reshape(msg.height, msg.step)
    frame = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)

    if encoding == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(frame)


def depth_to_numpy(msg):
    """Convert a ROS depth image to millimetres as float32."""
    encoding = msg.encoding.upper()
    if encoding in ("16UC1", "MONO16"):
        raw = np.frombuffer(msg.data, dtype=np.uint16)
        rows = raw.reshape(msg.height, msg.step // np.dtype(np.uint16).itemsize)
        return rows[:, : msg.width].astype(np.float32)

    if encoding == "32FC1":
        raw = np.frombuffer(msg.data, dtype=np.float32)
        rows = raw.reshape(msg.height, msg.step // np.dtype(np.float32).itemsize)
        return rows[:, : msg.width] * 1000.0

    raise RuntimeError(f"Unsupported depth encoding: {msg.encoding}")


def load_table_homography():
    if not TABLE_HOMOGRAPHY_FILE.is_file():
        return None
    try:
        payload = json.loads(
            TABLE_HOMOGRAPHY_FILE.read_text(encoding="utf-8")
        )
        matrix = np.asarray(
            payload["homography_pixel_to_mm"],
            dtype=np.float32,
        )
        if matrix.shape != (3, 3):
            raise ValueError("homography must be a 3x3 matrix")
        return matrix
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Warning: cannot load table homography: {exc}")
        return None


TABLE_HOMOGRAPHY = load_table_homography()


def get_color_name(hue, saturation, value):
    if saturation < 25:
        if value < 55:
            return "black"
        if value < 170:
            return "gray"
        return "white"
    if hue < 8 or hue >= 172:
        return "red"
    if hue < 22:
        return "orange"
    if hue < 38:
        return "yellow"
    if hue < 80:
        return "green"
    if hue < 100:
        return "cyan"
    if hue < 130:
        return "blue"
    if hue < 155:
        return "purple"
    if hue < 172:
        return "pink"
    return "unknown"


class YoloCVPublisher(Node):
    def __init__(self):
        super().__init__("yolo_cv_publisher")

        self.declare_parameter("confidence", CONFIDENCE)
        self.declare_parameter("iou", IOU)
        self.declare_parameter("process_every_n_frames", PROCESS_EVERY_N_FRAMES)

        self.model_path = default_model_path()
        self.confidence = float(self.get_parameter("confidence").value)
        self.iou = float(self.get_parameter("iou").value)
        self.process_every_n_frames = max(
            1,
            int(self.get_parameter("process_every_n_frames").value),
        )

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"YOLO classes: {self.model.names}")
        if TABLE_HOMOGRAPHY is None:
            self.get_logger().warn(
                "table_homography.json was not loaded; table coordinates "
                "will be unavailable"
            )
        else:
            self.get_logger().info(
                f"Loaded table calibration: {TABLE_HOMOGRAPHY_FILE}"
            )

        self.latest_depth_msg = None
        self.calibration_frames = []
        self.background_depth = None
        self.frame_counter = 0
        self.last_object_count = None
        self.stats_start = time.perf_counter()
        self.stats_frames = 0

        self.color_subscription = self.create_subscription(
            Image,
            COLOR_TOPIC,
            self.color_callback,
            SENSOR_QOS,
        )
        self.depth_subscription = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            SENSOR_QOS,
        )
        self.publisher = self.create_publisher(
            Detection2DArray,
            DETECTION_TOPIC,
            1,
        )

        self.get_logger().info(f"RGB: {COLOR_TOPIC}")
        self.get_logger().info(f"Depth: {DEPTH_TOPIC}")
        self.get_logger().info(f"Detections: {DETECTION_TOPIC}")

    def depth_callback(self, msg):
        # Keep only the newest frame so slow inference never builds a queue.
        self.latest_depth_msg = msg

    def color_callback(self, msg):
        if self.latest_depth_msg is None:
            return

        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            return

        started = time.perf_counter()
        depth_msg = self.latest_depth_msg

        try:
            frame = image_to_numpy(msg)
            depth_mm = depth_to_numpy(depth_msg)
        except Exception as exc:
            self.get_logger().error(f"Image conversion error: {exc}")
            return

        if self.background_depth is None:
            self.calibrate_depth(depth_mm)
            return

        try:
            result = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                verbose=False,
            )[0]
            objects = self.make_objects(result, frame, depth_mm)
        except Exception as exc:
            self.get_logger().error(f"YOLO inference error: {exc}")
            return

        message = Detection2DArray()
        message.header = msg.header
        message.detections = [
            self.to_ros_detection(obj)
            for obj in objects
        ]
        self.publisher.publish(message)

        if len(objects) != self.last_object_count:
            self.get_logger().info(f"Detected {len(objects)} object(s)")
            self.last_object_count = len(objects)

        self.report_performance(started)

    def calibrate_depth(self, depth_mm):
        depth_float = depth_mm.astype(np.float32)
        depth_float[depth_float <= 0] = np.nan
        self.calibration_frames.append(depth_float)
        count = len(self.calibration_frames)
        if count == 1:
            self.get_logger().info(
                "Keep the table empty: starting YOLO table calibration..."
            )
        if count % 5 == 0:
            self.get_logger().info(
                f"YOLO calibration: {count}/{CALIBRATION_FRAMES}"
            )
        if count < CALIBRATION_FRAMES:
            return
        with np.errstate(all="ignore"):
            self.background_depth = np.nanmedian(
                np.stack(self.calibration_frames, axis=0),
                axis=0,
            ).astype(np.float32)
        self.calibration_frames.clear()
        self.get_logger().info("YOLO table calibration finished.")

    def make_objects(self, result, frame, depth_mm):
        """Build objects with table coordinates, height, range, and colour."""
        detected = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detected

        color_height, color_width = frame.shape[:2]
        depth_height, depth_width = depth_mm.shape
        names = result.names

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        for coords, confidence, class_id in zip(xyxy, confidences, class_ids):
            x1, y1, x2, y2 = self.clip_box(coords, color_width, color_height)
            if x2 <= x1 or y2 <= y1:
                continue

            dx1 = int(round(x1 * depth_width / color_width))
            dx2 = int(round(x2 * depth_width / color_width))
            dy1 = int(round(y1 * depth_height / color_height))
            dy2 = int(round(y2 * depth_height / color_height))
            dx1, dx2 = np.clip((dx1, dx2), 0, depth_width).astype(int)
            dy1, dy2 = np.clip((dy1, dy2), 0, depth_height).astype(int)

            distance_mm = self.box_distance(depth_mm[dy1:dy2, dx1:dx2])
            height_mm = self.box_height(depth_mm[dy1:dy2, dx1:dx2], x1, y1, x2, y2)
            color = self.box_color(frame[y1:y2, x1:x2])
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            elif 0 <= class_id < len(names):
                class_name = str(names[class_id])
            else:
                class_name = str(class_id)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            position_mm = self.pixel_to_table_mm(center_x, center_y)

            detected.append(
                {
                    "type": class_name,
                    "color": color,
                    "height_mm": height_mm,
                    "class_id": int(class_id),
                    "confidence": round(float(confidence), 3),
                    "position": [center_x, center_y],
                    "position_mm": position_mm,
                    "bbox": [x1, y1, x2, y2],
                    # Rectangle contour keeps cv_publisher consumers working.
                    "contour": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "distance_mm": distance_mm,
                }
            )

        detected.sort(key=lambda obj: obj["position"][0])
        for object_id, obj in enumerate(detected, start=1):
            obj["id"] = object_id
        return detected

    @staticmethod
    def to_ros_detection(obj):
        detection = Detection2D()
        x1, y1, x2, y2 = obj["bbox"]
        detection.id = (
            f"object_id={obj['id']}:color={obj['color']}:"
            f"distance_mm={obj['distance_mm']}"
        )
        detection.bbox.center.position.x = (x1 + x2) / 2.0
        detection.bbox.center.position.y = (y1 + y2) / 2.0
        detection.bbox.size_x = float(x2 - x1)
        detection.bbox.size_y = float(y2 - y1)

        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = str(obj["type"])
        result.hypothesis.score = float(obj["confidence"])
        position_mm = obj.get("position_mm")
        if isinstance(position_mm, list) and len(position_mm) == 2:
            result.pose.pose.position.x = float(position_mm[0]) / 1000.0
            result.pose.pose.position.y = float(position_mm[1]) / 1000.0
        result.pose.pose.position.z = float(obj.get("height_mm", 0.0)) / 1000.0
        detection.results = [result]
        return detection

    @staticmethod
    def box_color(roi):
        if roi.size == 0:
            return "unknown"
        height, width = roi.shape[:2]
        margin_x = width // 4
        margin_y = height // 4
        central = roi[
            margin_y : max(margin_y + 1, height - margin_y),
            margin_x : max(margin_x + 1, width - margin_x),
        ]
        hsv = cv2.cvtColor(central, cv2.COLOR_BGR2HSV)
        median = np.median(hsv.reshape(-1, 3), axis=0)
        return get_color_name(*median.astype(int))

    def box_height(self, depth_roi, x1, y1, x2, y2):
        if self.background_depth is None or depth_roi.size == 0:
            return 0.0
        background_roi = self.background_depth[y1:y2, x1:x2]
        valid = (
            np.isfinite(background_roi)
            & np.isfinite(depth_roi)
            & (depth_roi >= MIN_DEPTH_MM)
            & (depth_roi <= MAX_DEPTH_MM)
        )
        differences = background_roi[valid] - depth_roi[valid]
        differences = differences[(differences > 0) & (differences < 1000)]
        if differences.size == 0:
            return 0.0
        return round(float(np.percentile(differences, 75)), 1)

    @staticmethod
    def pixel_to_table_mm(x, y):
        if TABLE_HOMOGRAPHY is None:
            return None
        point = np.array([[[float(x), float(y)]]], dtype=np.float32)
        table_point = cv2.perspectiveTransform(point, TABLE_HOMOGRAPHY)[0, 0]
        return [round(float(table_point[0]), 1), round(float(table_point[1]), 1)]

    @staticmethod
    def clip_box(coords, width, height):
        x1, y1, x2, y2 = (int(round(float(value))) for value in coords)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        return x1, y1, x2, y2

    @staticmethod
    def box_distance(depth_roi):
        """Use the central part of a box to reduce background contamination."""
        if depth_roi.size == 0:
            return None

        height, width = depth_roi.shape
        margin_x = width // 4
        margin_y = height // 4
        central = depth_roi[
            margin_y : max(margin_y + 1, height - margin_y),
            margin_x : max(margin_x + 1, width - margin_x),
        ]
        valid = central[
            np.isfinite(central)
            & (central >= MIN_DEPTH_MM)
            & (central <= MAX_DEPTH_MM)
        ]
        if valid.size == 0:
            return None
        return round(float(np.median(valid)), 1)

    def report_performance(self, started):
        processing_ms = (time.perf_counter() - started) * 1000.0
        self.stats_frames += 1
        now = time.perf_counter()
        elapsed = now - self.stats_start
        if elapsed < 2.0:
            return

        hz = self.stats_frames / elapsed
        self.get_logger().info(
            f"YOLO: {hz:.1f} Hz | processing: {processing_ms:.1f} ms"
        )
        self.stats_frames = 0
        self.stats_start = now


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloCVPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
