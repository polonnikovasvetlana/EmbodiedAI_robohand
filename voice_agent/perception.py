"""Adapter for the repository's /detected_objects JSON snapshots.

Coordinates remain observations, never robot goals. IDs are frame-local.
"""

import copy
import json
import math
import time

from .errors import VoiceError


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def pair(value):
    return isinstance(value, list) and len(value) == 2 and all(map(finite_number, value))


class DetectionStore:
    def __init__(self, max_age=2.0, min_confidence=0.4, clock=time.monotonic):
        if not finite_number(max_age) or max_age <= 0:
            raise VoiceError("invalid_config", "Detection max age must be positive.")
        if not finite_number(min_confidence) or not 0 <= min_confidence <= 1:
            raise VoiceError("invalid_config", "Detection confidence must be in [0, 1].")
        self.max_age = max_age
        self.min_confidence = min_confidence
        self.clock = clock
        self.clear()

    def clear(self):
        self.payload = None
        self.received_at = None

    def update(self, payload):
        # A broken frame must not leave the previous target usable.
        self.clear()
        try:
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
                raise ValueError("Expected an object containing an objects list.")
            for obj in payload["objects"]:
                if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
                    raise ValueError("Each detection needs a string type.")
                if not pair(obj.get("position")) or min(obj["position"]) < 0:
                    raise ValueError("position must contain two nonnegative pixel coordinates.")
                if "confidence" in obj and (
                    not finite_number(obj["confidence"]) or not 0 <= obj["confidence"] <= 1
                ):
                    raise ValueError("Invalid detection confidence.")
                if "position_mm" in obj and not pair(obj["position_mm"]):
                    raise ValueError("position_mm must contain two finite coordinates.")
                for field in ("distance_mm", "height_mm"):
                    value = obj.get(field)
                    if value is not None and (not finite_number(value) or value < 0):
                        raise ValueError(f"Invalid {field}.")
                if "id" in obj and (type(obj["id"]) is not int or obj["id"] < 1):
                    raise ValueError("Detection id must be a positive integer.")
            for key in ("image_width", "image_height"):
                if key in payload and (type(payload[key]) is not int or payload[key] <= 0):
                    raise ValueError(f"Invalid {key}.")
            for obj in payload["objects"]:
                for axis, key in enumerate(("image_width", "image_height")):
                    if key in payload and obj["position"][axis] >= payload[key]:
                        raise ValueError("Pixel coordinate outside image.")
            stamp = payload.get("stamp")
            if stamp is not None:
                if (not isinstance(stamp, dict)
                        or type(stamp.get("sec")) is not int
                        or type(stamp.get("nanosec")) is not int
                        or stamp["sec"] < 0 or not 0 <= stamp["nanosec"] < 10**9):
                    raise ValueError("Invalid detection stamp.")
        except (ValueError, TypeError) as exc:
            raise VoiceError("invalid_detections", str(exc)) from exc
        self.payload = copy.deepcopy(payload)
        self.received_at = self.clock()

    def resolve(self, object_name):
        if self.payload is None:
            raise VoiceError("object_not_detected", "No detection snapshot is available.")
        age = self.clock() - self.received_at
        if age < 0 or age > self.max_age:
            raise VoiceError("stale_detections", "Detection snapshot expired; wait for a new frame.")
        matches = [obj for obj in self.payload["objects"] if obj["type"].lower() == object_name]
        if not matches:
            raise VoiceError("object_not_detected", f"No {object_name} detected. Use the YOLO publisher for semantic labels.")
        # Do not silently select one of several bottles based on confidence.
        if len(matches) != 1:
            raise VoiceError("ambiguous_object", f"Multiple {object_name} detections; target selection is required.")
        obj = matches[0]
        if obj.get("confidence", 0.0) < self.min_confidence:
            raise VoiceError("low_detection_confidence", "Semantic detection confidence is missing or too low.")
        result = {
            "id": obj.get("id"),
            "type": obj["type"],
            "confidence": obj.get("confidence"),
            "pixel_position": list(obj["position"]),
            "distance_mm": obj.get("distance_mm"),
            "workspace_xy_mm": copy.deepcopy(obj.get("position_mm")),
            "height_mm": obj.get("height_mm"),
            "coordinate_frame": "unverified_observation",
            "robot_pose": None,
            "requires_calibration": True,
            "source_stamp": copy.deepcopy(self.payload.get("stamp")),
            "age_seconds": age,
        }
        return result
