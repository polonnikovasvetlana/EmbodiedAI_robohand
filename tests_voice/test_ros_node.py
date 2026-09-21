"""Exercise node callbacks without ROS middleware or any hardware imports."""

import json
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from voice_agent.ros_node import make_node


class RosCallbackTests(unittest.TestCase):
    def setUp(self):
        class FakeNode:
            def __init__(self, name):
                self.params = {}
                self.publishers = {}
                self.subscriptions = {}

            def declare_parameter(self, name, value):
                self.params[name] = value

            def get_parameter(self, name):
                return SimpleNamespace(value=self.params[name])

            def create_publisher(self, kind, topic, depth):
                result = Mock()
                self.publishers[topic] = result
                return result

            def create_subscription(self, kind, topic, callback, depth):
                self.subscriptions[topic] = callback
                return Mock()

            def get_logger(self):
                return Mock()

            def get_clock(self):
                return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100 * 10**9))

        node_module = ModuleType("rclpy.node")
        node_module.Node = FakeNode
        msg_module = ModuleType("std_msgs.msg")
        msg_module.String = SimpleNamespace
        with patch.dict(sys.modules, {"rclpy.node": node_module, "std_msgs.msg": msg_module}):
            self.node = make_node()

    def send_detection(self, stamp=100):
        payload = {"objects": [{"id": 1, "type": "bottle", "position": [10, 20], "confidence": 0.9}],
                   "stamp": {"sec": stamp, "nanosec": 0}}
        self.node.on_detections(SimpleNamespace(data=json.dumps(payload)))

    def send_command(self, text="bring me the bottle", confidence=0.9):
        self.node.on_transcript(SimpleNamespace(data=json.dumps({
            "text": text, "confidence": confidence, "language": "en"})))

    def test_fresh_detection_publishes_preview_only(self):
        self.send_detection()
        self.send_command()
        result = json.loads(self.node.intents.publish.call_args.args[0].data)
        self.assertFalse(result["execution_allowed"])
        self.assertIsNone(result["target"]["robot_pose"])
        self.assertEqual(set(self.node.publishers), {"/voice_agent/intent_preview", "/voice_agent/status"})

    def test_stale_and_future_source_invalidates_target(self):
        for stamp in (90, 110):
            self.send_detection(stamp)
            self.send_command()
            self.assertIsNone(self.node.store.payload)
            self.node.intents.publish.assert_not_called()

    def test_invalid_frame_and_low_confidence_do_not_publish(self):
        self.send_detection()
        self.send_command(confidence=0.1)
        self.node.intents.publish.assert_not_called()
        self.node.on_detections(SimpleNamespace(data="null"))
        self.send_command()
        self.node.intents.publish.assert_not_called()

    def test_stop_does_not_need_perception(self):
        self.send_command("Stop")
        result = json.loads(self.node.intents.publish.call_args.args[0].data)
        self.assertEqual(result["command"], {"action": "stop", "object": None})
        self.assertIsNone(result["target"])
