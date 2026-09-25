import json
import subprocess
import sys
import unittest
from pathlib import Path

from voice_agent import Command, VoiceError, parse_command
from voice_agent.perception import DetectionStore
from voice_agent.pipeline import preview
from voice_agent.ros_node import process_transcript

FIXTURE = Path(__file__).resolve().parents[1] / "examples/voice_detections.json"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.store = DetectionStore(clock=lambda: self.now)
        self.data = json.loads(FIXTURE.read_text())
        self.store.update(self.data)

    def error(self, code, fn, *args):
        with self.assertRaises(VoiceError) as caught:
            fn(*args)
        self.assertEqual(caught.exception.code, code)

    def test_commands(self):
        for text, action, obj in [
            ("Bring me the bottle", "bring", "bottle"),
            ("Bring me the cup", "bring", "cup"),
            ("Pick up the bottle", "pick_up", "bottle"),
            ("  PLEASE pick UP a cup!  ", "pick_up", "cup"),
            ("Stop", "stop", None), ("Stop, please.", "stop", None),
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_command(text).as_dict(), {"action": action, "object": obj})

    def test_rejections(self):
        self.error("empty_speech", parse_command, "  ")
        self.error("unknown_object", parse_command, "bring me the elephant")
        for text in ["don't bring me the cup", "bring me the cup and stop", "stop then pick up the cup",
                     "move 1 2 3", "bring me the red cup", "pick up", "stop? bring me the cup"]:
            with self.subTest(text=text):
                self.error("unsupported_command", parse_command, text)
        self.error("unsupported_command", Command, "move", "cup")
        self.error("unsupported_command", Command, "stop", "cup")

    def test_preview_has_no_robot_pose(self):
        result = preview("Bring me the bottle", self.store)
        self.assertEqual(result["target"]["pixel_position"], [210, 180])
        self.assertEqual(result["target"]["distance_mm"], 650)
        self.assertIsNone(result["target"]["robot_pose"])
        self.assertFalse(result["execution_allowed"])

    def test_stop_without_detections(self):
        self.store.clear()
        self.assertIsNone(preview("Stop", self.store)["target"])

    def test_missing_stale_and_disappearing_object(self):
        self.now = 13
        self.error("stale_detections", preview, "bring me the cup", self.store)
        self.store.update({"objects": []})
        self.error("object_not_detected", preview, "bring me the cup", self.store)

    def test_ambiguous(self):
        self.data["objects"].append(dict(self.data["objects"][0], id=3))
        self.store.update(self.data)
        self.error("ambiguous_object", preview, "pick up the bottle", self.store)

    def test_low_detection_confidence(self):
        self.data["objects"][0]["confidence"] = 0.1
        self.store.update(self.data)
        self.error("low_detection_confidence", preview, "pick up the bottle", self.store)

    def test_shape_is_not_semantic_object(self):
        self.store.update({"objects": [{"type": "circle", "position": [5, 5], "position_mm": [1, -2]}]})
        self.error("object_not_detected", preview, "bring me the cup", self.store)

    def test_invalid_snapshot_clears_cache(self):
        for data in ["{", [], {"objects": [None]}, {"objects": [{"type": "cup", "position": [float('nan'), 1]}]}]:
            with self.subTest(data=data):
                self.store.update(self.data)
                self.error("invalid_detections", self.store.update, data)
                self.error("object_not_detected", preview, "bring me the cup", self.store)

    def test_null_depth_does_not_become_xyz(self):
        self.data["objects"][0]["distance_mm"] = None
        self.store.update(self.data)
        result = preview("bring me the bottle", self.store)
        self.assertTrue(result["target"]["requires_calibration"])
        self.assertIsNone(result["target"]["robot_pose"])

    def test_ros_transcript_boundary(self):
        payload = {"text": "bring me the cup", "confidence": 0.9, "language": "en"}
        self.assertEqual(process_transcript(json.dumps(payload), self.store)["command"]["object"], "cup")
        payload["confidence"] = 0.1
        self.error("low_confidence", process_transcript, json.dumps(payload), self.store)
        payload["confidence"] = float("nan")
        self.error("low_confidence", process_transcript, json.dumps(payload), self.store)
        payload["joints"] = [1, 2, 3]
        self.error("invalid_transcript", process_transcript, json.dumps(payload), self.store)

    def test_cli(self):
        root = FIXTURE.parents[1]
        result = subprocess.run([sys.executable, "-m", "voice_agent", "pipeline", "--text",
                                 "bring me the cup", "--detections", str(FIXTURE)],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["execution_allowed"])
        result = subprocess.run([sys.executable, "-m", "voice_agent", "parse", "move 1 2 3"],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["error"]["code"], "unsupported_command")


if __name__ == "__main__":
    unittest.main()
