"""Bounded microphone capture and English faster-whisper transcription.

Optional audio dependencies are imported only when audio is requested.
"""

import math
from dataclasses import asdict, dataclass

from .errors import VoiceError


def audio_backend():
    try:
        import sounddevice
        return sounddevice
    except (ImportError, OSError) as exc:
        raise VoiceError("audio_unavailable", "Install sounddevice and PortAudio in the active environment.") from exc


def list_devices():
    sd = audio_backend()
    try:
        return [dict(info, index=index, is_default=index == sd.default.device[0])
                for index, info in enumerate(sd.query_devices())
                if info["max_input_channels"] > 0]
    except Exception as exc:
        raise VoiceError("no_microphone", f"Cannot enumerate input devices: {exc}") from exc


def record(seconds=5.0, device=None, backend=None):
    """Use PortAudio's default input unless explicitly overridden.

Capture at 16 kHz if supported, otherwise at the device's native rate and
resample. Never fall back to a different microphone after an explicit choice.
"""
    import numpy as np

    if not math.isfinite(seconds) or not 0 < seconds <= 30:
        raise VoiceError("invalid_config", "Recording duration must be in (0, 30] seconds.")
    sd = backend if backend is not None else audio_backend()
    if isinstance(device, str):
        device = device.strip() or None
        if device is not None and device.lstrip("-").isdigit():
            device = int(device)
    try:
        info = sd.query_devices(device, "input")
        if info["max_input_channels"] < 1:
            raise ValueError("Selected device has no input channels.")
    except Exception as exc:
        raise VoiceError("no_microphone", f"No usable input device ({device!r}): {exc}. Run 'devices' or set --device.") from exc
    rate = 16000
    try:
        sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=rate)
    except Exception:
        try:
            rate = int(info["default_samplerate"])
            if rate <= 0:
                raise ValueError("Invalid native sample rate.")
            sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=rate)
        except Exception as exc:
            raise VoiceError("audio_unavailable", f"Microphone does not support mono capture: {exc}") from exc
    try:
        with sd.InputStream(device=device, samplerate=rate, channels=1, dtype="float32") as stream:
            audio, overflowed = stream.read(max(1, int(seconds * rate)))
        if overflowed:
            raise VoiceError("audio_overflow", "Input audio was lost. Please record again.")
    except VoiceError:
        raise
    except Exception as exc:
        raise VoiceError("audio_unavailable", f"Microphone capture failed: {exc}") from exc
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if rate != 16000:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise VoiceError("missing_dependency", "Install scipy to resample native microphone audio.") from exc
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    return audio


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float
    language: str = "en"

    def as_dict(self):
        return asdict(self)


class SpeechToText:
    def __init__(self, model="tiny.en", min_confidence=0.5, model_instance=None,
                 cache_dir=None, local_files_only=False):
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise VoiceError("invalid_config", "Transcription confidence must be in [0, 1].")
        self.min_confidence = min_confidence
        if model_instance is not None:
            self.model = model_instance
            return
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model, device="cpu", compute_type="int8",
                                      download_root=cache_dir, local_files_only=local_files_only)
        except Exception as exc:
            raise VoiceError("model_unavailable", f"Could not load Whisper model {model!r}: {exc}. Use --model with a local model directory or download it first.") from exc

    def transcribe(self, audio):
        import numpy as np

        if not isinstance(audio, (str, bytes)):
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim != 1 or not np.isfinite(audio).all():
                raise VoiceError("invalid_audio", "Audio must be finite mono float32 samples at 16 kHz.")
            if audio.size == 0 or np.max(np.abs(audio)) < 1e-5:
                raise VoiceError("empty_speech", "No audible speech in the recording.")
        try:
            segments, _ = self.model.transcribe(
                audio, language="en", vad_filter=True, beam_size=5,
                condition_on_previous_text=False, temperature=0.0,
            )
            # Inference is lazy: consume the generator inside the error handler.
            segments = list(segments)
        except Exception as exc:
            raise VoiceError("transcription_failed", f"Whisper transcription failed: {exc}") from exc
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        if not text:
            raise VoiceError("empty_speech", "No speech was recognized. Please try again.")
        scores = []
        for segment in segments:
            if not segment.text.strip():
                continue
            logprob, no_speech = segment.avg_logprob, segment.no_speech_prob
            if (not math.isfinite(logprob) or not math.isfinite(no_speech)
                    or not 0 <= no_speech <= 1):
                raise VoiceError("low_confidence", "Transcription confidence is invalid; repeat the command.")
            # Conservative segment score, NOT a calibrated probability.
            scores.append(min(math.exp(min(0.0, logprob)), 1.0 - no_speech))
        confidence = min(scores)
        if confidence < self.min_confidence:
            raise VoiceError("low_confidence", f"Transcription score {confidence:.3f} is below {self.min_confidence:.3f}; repeat the command.")
        return Transcript(text, confidence)
