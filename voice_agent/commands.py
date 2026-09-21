"""Closed command grammar. Free-form text never becomes executable code."""

import re
from dataclasses import asdict, dataclass

from .errors import VoiceError

OBJECTS = frozenset({"bottle", "cup"})


@dataclass(frozen=True)
class Command:
    action: str
    object: str | None = None

    def __post_init__(self):
        if self.action not in ("bring", "pick_up", "stop"):
            raise VoiceError("unsupported_command", "Allowed actions: bring, pick_up, stop.")
        if self.action == "stop":
            if self.object is not None:
                raise VoiceError("unsupported_command", "Stop cannot have an object.")
        elif not isinstance(self.object, str) or self.object not in OBJECTS:
            raise VoiceError("unknown_object", "Supported objects: bottle, cup.")

    def as_dict(self):
        return asdict(self)


def parse_command(text: str) -> Command:
    if not isinstance(text, str):
        raise VoiceError("unsupported_command", "Command must be text.")
    text = " ".join(text.lower().split()).strip(" .!?")
    if not text:
        raise VoiceError("empty_speech", "No speech was recognized. Please try again.")
    text = re.sub(r"^please\s+", "", text)
    text = re.sub(r"(?:,?\s+please)$", "", text)
    if text == "stop":
        return Command("stop")
    match = re.fullmatch(r"(bring(?: me)?|pick up) (?:the |a )?([a-z]+)", text)
    if not match:
        raise VoiceError("unsupported_command", "Say 'Bring me the bottle', 'Pick up the cup', or 'Stop'.")
    action, object_name = match.groups()
    return Command("pick_up" if action == "pick up" else "bring", object_name)
