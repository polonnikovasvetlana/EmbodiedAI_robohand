import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 5  # seconds

print("Loading Whisper model...")

model = WhisperModel(
    "tiny.en",
    device="cpu",
    compute_type="int8"
)

print("Whisper ready.")

input("\nPress ENTER, then speak for 5 seconds...")

print("Listening...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = np.squeeze(audio)

print("Transcribing...")

segments, info = model.transcribe(
    audio,
    language="en",
    vad_filter=True
)

text = " ".join(segment.text.strip() for segment in segments)

print("\nYOU SAID:")
print(text)