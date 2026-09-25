"""Pure preview pipeline: no imports from SO101Arm, MoveIt, or motor drivers."""

from .commands import parse_command


def preview(text, detections):
    command = parse_command(text)
    target = None if command.action == "stop" else detections.resolve(command.object)
    return {
        "ok": True,
        "schema_version": 1,
        "mode": "preview",
        "command": command.as_dict(),
        "target": target,
        "execution_allowed": False,
    }
