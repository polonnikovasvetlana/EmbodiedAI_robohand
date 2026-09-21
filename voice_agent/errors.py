class VoiceError(Exception):
    """An expected, machine-readable pipeline failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def as_dict(self):
        return {"ok": False, "error": {"code": self.code, "message": str(self)}}
