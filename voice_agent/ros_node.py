"""Optional ROS 2 preview bridge; deliberately has no robot action client.

Accepts JSON Transcript messages, not arbitrary joint/pose instructions.
Run as ``python -m voice_agent.ros_node`` in a ROS-enabled interpreter.
"""

import json
import sys

from .errors import VoiceError
from .perception import DetectionStore, finite_number
from .pipeline import preview


def process_transcript(payload, detections, min_confidence=0.5):
    try:
        data = json.loads(payload)
    except (ValueError, TypeError) as exc:
        raise VoiceError("invalid_transcript", "Expected transcript JSON.") from exc
    if (not isinstance(data, dict) or set(data) != {"text", "confidence", "language"}
            or not isinstance(data["text"], str) or data["language"] != "en"):
        raise VoiceError("invalid_transcript", "Expected exactly text, confidence, and language='en'.")
    confidence = data["confidence"]
    if not finite_number(confidence) or not 0 <= confidence <= 1:
        raise VoiceError("low_confidence", "Invalid transcription confidence.")
    if confidence < min_confidence:
        raise VoiceError("low_confidence", "Repeat the command with clearer speech.")
    return preview(data["text"], detections)


def make_node():
    # Lazy imports keep CLI/parser/test usage independent of ROS installation.
    from rclpy.node import Node
    from std_msgs.msg import String

    class VoicePreviewNode(Node):
        def __init__(self):
            super().__init__("voice_agent_preview")
            self.declare_parameter("max_detection_age", 2.0)
            self.declare_parameter("min_transcription_confidence", 0.5)
            self.declare_parameter("min_detection_confidence", 0.4)
            self.store = DetectionStore(
                max_age=self.get_parameter("max_detection_age").value,
                min_confidence=self.get_parameter("min_detection_confidence").value,
            )
            self.min_confidence = self.get_parameter("min_transcription_confidence").value
            if not finite_number(self.min_confidence) or not 0 <= self.min_confidence <= 1:
                raise VoiceError("invalid_config", "Transcription confidence must be in [0, 1].")
            self.intents = self.create_publisher(String, "/voice_agent/intent_preview", 10)
            self.status = self.create_publisher(String, "/voice_agent/status", 10)
            self.detection_sub = self.create_subscription(String, "/detected_objects", self.on_detections, 1)
            self.transcript_sub = self.create_subscription(String, "/voice_agent/transcript", self.on_transcript, 10)
            self.get_logger().info("Preview only: no robot controller is connected.")

        def report(self, error):
            self.status.publish(String(data=json.dumps(error.as_dict())))

        def on_detections(self, message):
            try:
                self.store.update(message.data)
                stamp = self.store.payload.get("stamp")
                if stamp is not None:
                    source_ns = stamp["sec"] * 10**9 + stamp["nanosec"]
                    age = (self.get_clock().now().nanoseconds - source_ns) / 1e9
                    if age < -0.1 or age > self.store.max_age:
                        self.store.clear()
                        raise VoiceError("stale_detections", "Source image timestamp is stale or in the future.")
                    # Include pre-receipt age, not merely time in this process.
                    self.store.received_at -= max(age, 0)
            except VoiceError as exc:
                self.report(exc)

        def on_transcript(self, message):
            try:
                result = process_transcript(message.data, self.store, self.min_confidence)
                self.intents.publish(String(data=json.dumps(result, allow_nan=False)))
            except VoiceError as exc:
                self.report(exc)

    return VoicePreviewNode()


def main(args=None):
    try:
        import rclpy
    except ImportError:
        print("ROS 2 unavailable: source your ROS setup and use its compatible Python interpreter.", file=sys.stderr)
        return 2
    node = None
    try:
        rclpy.init(args=args)
        node = make_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
