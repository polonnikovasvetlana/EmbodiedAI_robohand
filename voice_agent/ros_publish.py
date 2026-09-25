"""Publish one CLI transcript file using a separate ROS-compatible Python."""

import argparse
import json
import sys
import time
from pathlib import Path

from .errors import VoiceError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.transcript.read_text())
        if (not isinstance(data, dict) or set(data) != {"text", "confidence", "language"}
                or not isinstance(data["text"], str)):
            raise ValueError("Expected JSON from 'python -m voice_agent stt'.")
        encoded = json.dumps(data, allow_nan=False)
        import rclpy
        from rclpy.duration import Duration
        from std_msgs.msg import String
    except (ImportError, OSError, ValueError) as exc:
        print(json.dumps(VoiceError("publish_unavailable", str(exc)).as_dict()), file=sys.stderr)
        return 2
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("voice_transcript_sender")
        publisher = node.create_publisher(String, "/voice_agent/transcript", 10)
        deadline = time.monotonic() + 5
        while publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise VoiceError("no_subscriber", "Start voice_agent.ros_node before sending a transcript.")
            rclpy.spin_once(node, timeout_sec=0.1)
        publisher.publish(String(data=encoded))
        if not publisher.wait_for_all_acked(Duration(seconds=2)):
            raise VoiceError("delivery_timeout", "Transcript acknowledgement timed out.")
        return 0
    except VoiceError as exc:
        print(json.dumps(exc.as_dict()), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
