"""Show YOLO detections over the RealSense RGB stream.

Run ``yolo_cv_publisher.py`` and this script after starting the RealSense ROS
driver. Depth remains subscribed for diagnostics/range data. Press Q to exit.
"""

import json

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
from vision_msgs.msg import Detection2DArray


COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
DETECTION_TOPIC = "/detected_objects"

SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def depth_to_numpy(msg):
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


def color_to_numpy(msg):
    encoding = msg.encoding.lower()
    if encoding not in ("rgb8", "bgr8"):
        raise RuntimeError(f"Unsupported color encoding: {msg.encoding}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    rows = raw.reshape(msg.height, msg.step)
    frame = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if encoding == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(frame)


class YoloDepthVisualizer(Node):
    def __init__(self):
        super().__init__("yolo_depth_visualizer")
        self.objects = []
        self.source_width = None
        self.source_height = None
        self.latest_depth_mm = None

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
        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            DETECTION_TOPIC,
            self.detection_callback,
            1,
        )

        self.get_logger().info("YOLO RGB visualizer started. Press Q to quit.")
        self.get_logger().info(f"RGB: {COLOR_TOPIC}")
        self.get_logger().info(f"Depth: {DEPTH_TOPIC}")
        self.get_logger().info(f"Detections: {DETECTION_TOPIC}")

    def detection_callback(self, msg):
        objects = []
        for detection in msg.detections:
            if not detection.results:
                continue
            result = detection.results[0]
            metadata = self.parse_metadata(detection.id)
            object_id = metadata.get("object_id", "?")
            center = result.pose.pose.position
            bbox = detection.bbox
            center_px = bbox.center.position
            half_width = bbox.size_x / 2.0
            half_height = bbox.size_y / 2.0
            objects.append(
                {
                    "id": object_id,
                    "type": result.hypothesis.class_id,
                    "color": metadata.get("color", "unknown"),
                    "confidence": result.hypothesis.score,
                    "position_mm": [center.x * 1000.0, center.y * 1000.0],
                    "height_mm": center.z * 1000.0,
                    "distance_mm": self.parse_float(
                        metadata.get("distance_mm")
                    ),
                    "bbox": [
                        center_px.x - half_width,
                        center_px.y - half_height,
                        center_px.x + half_width,
                        center_px.y + half_height,
                    ],
                }
            )
        self.objects = objects
        self.source_width = 640
        self.source_height = 480

    @staticmethod
    def parse_metadata(value):
        metadata = {}
        for part in value.split(":"):
            if "=" in part:
                key, item = part.split("=", 1)
                metadata[key] = item
        return metadata

    @staticmethod
    def parse_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def depth_callback(self, msg):
        try:
            self.latest_depth_mm = depth_to_numpy(msg)
        except Exception as exc:
            self.get_logger().error(f"Depth conversion error: {exc}")

    def color_callback(self, msg):
        try:
            frame = color_to_numpy(msg)
        except Exception as exc:
            self.get_logger().error(f"Color conversion error: {exc}")
            return

        scale_x = msg.width / self.source_width if self.source_width else 1.0
        scale_y = msg.height / self.source_height if self.source_height else 1.0

        for obj in self.objects:
            bbox = obj.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            x1 = int(round(x1 * scale_x))
            x2 = int(round(x2 * scale_x))
            y1 = int(round(y1 * scale_y))
            y2 = int(round(y2 * scale_y))
            x1 = max(0, min(x1, msg.width - 1))
            y1 = max(0, min(y1, msg.height - 1))
            x2 = max(0, min(x2, msg.width - 1))
            y2 = max(0, min(y2, msg.height - 1))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            object_type = obj.get("type", "object")
            color = obj.get("color", "unknown")
            object_id = obj.get("id", "?")
            confidence = float(obj.get("confidence", 0.0))
            distance = obj.get("distance_mm")
            position_mm = obj.get("position_mm")
            height_mm = obj.get("height_mm")
            labels = [
                f"Object {object_id}: {object_type} {color} {confidence:.2f}"
            ]
            if isinstance(position_mm, list) and len(position_mm) == 2:
                labels.append(
                    f"X: {float(position_mm[0]):.0f} mm  "
                    f"Y: {float(position_mm[1]):.0f} mm"
                )
            if height_mm is not None:
                labels.append(f"Height: {float(height_mm):.0f} mm")
            if distance is not None:
                labels.append(f"Distance: {float(distance) / 10.0:.1f} cm")
            else:
                labels.append("Distance: N/A")

            font_scale = 0.48
            line_height = 18
            text_sizes = [
                cv2.getTextSize(
                    line,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    1,
                )[0]
                for line in labels
            ]
            block_width = min(
                max(size[0] for size in text_sizes) + 8,
                msg.width - x1,
            )
            block_height = line_height * len(labels) + 6
            block_x = x1
            block_y = y2 + 4
            if block_y + block_height > msg.height:
                block_y = max(0, y1 - block_height - 4)
            cv2.rectangle(
                frame,
                (block_x, block_y),
                (min(block_x + block_width, msg.width - 1), block_y + block_height),
                (0, 0, 0),
                -1,
            )
            for line_index, line in enumerate(labels):
                cv2.putText(
                    frame,
                    line,
                    (block_x + 3, block_y + 15 + line_index * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            f"Objects: {len(self.objects)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("YOLO detections on RGB", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            if rclpy.ok():
                rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthVisualizer()
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
