import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from voice_agent.errors import VoiceError
from voice_agent.stt import SpeechToText, record


class AudioTests(unittest.TestCase):
    def error(self, code, fn, *args, **kwargs):
        with self.assertRaises(VoiceError) as caught:
            fn(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def backend(self, rate=16000, overflow=False):
        sd = Mock()
        sd.query_devices.return_value = {"max_input_channels": 1, "default_samplerate": rate}
        stream = Mock()
        stream.read.side_effect = lambda frames: (np.ones((frames, 1), dtype=np.float32), overflow)
        class Context:
            def __enter__(self):
                return stream
            def __exit__(self, *args):
                pass
        sd.InputStream.return_value = Context()
        return sd

    def test_default_microphone(self):
        sd = self.backend()
        audio = record(0.1, backend=sd)
        sd.query_devices.assert_called_once_with(None, "input")
        self.assertEqual(sd.InputStream.call_args.kwargs["device"], None)
        self.assertEqual(audio.shape, (1600,))

    def test_explicit_device_and_resampling(self):
        sd = self.backend(48000)
        sd.check_input_settings.side_effect = [ValueError("unsupported rate"), None]
        audio = record(0.1, device="3", backend=sd)
        self.assertEqual(sd.InputStream.call_args.kwargs["device"], 3)
        self.assertEqual(sd.InputStream.call_args.kwargs["samplerate"], 48000)
        self.assertEqual(audio.shape, (1600,))
        self.assertEqual(audio.dtype, np.float32)

    def test_no_microphone_and_overflow(self):
        sd = self.backend()
        sd.query_devices.side_effect = ValueError("No default input")
        self.error("no_microphone", record, backend=sd)
        sd.InputStream.assert_not_called()
        self.error("audio_overflow", record, 0.1, backend=self.backend(overflow=True))

    def test_bad_duration(self):
        for duration in [0, -1, 31, float("nan")]:
            self.error("invalid_config", record, duration)

    def stt(self, text="Bring me the bottle", logprob=-0.1, no_speech=0.01):
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(text=text, avg_logprob=logprob,
                                                             no_speech_prob=no_speech)]), None)
        return SpeechToText(model_instance=model)

    def test_transcript(self):
        stt = self.stt()
        result = stt.transcribe(np.ones(1600))
        self.assertEqual(result.text, "Bring me the bottle")
        self.assertGreater(result.confidence, 0.5)
        self.assertTrue(stt.model.transcribe.call_args.kwargs["vad_filter"])

    def test_silence_empty_and_low_confidence(self):
        self.error("empty_speech", self.stt().transcribe, np.zeros(1600))
        self.error("empty_speech", self.stt(text=" ").transcribe, np.ones(1600))
        for kwargs in [{"logprob": -2}, {"no_speech": 0.9}, {"logprob": float("nan")}]:
            self.error("low_confidence", self.stt(**kwargs).transcribe, np.ones(1600))

    def test_generator_failure(self):
        def broken():
            raise RuntimeError("inference failed")
            yield
        stt = self.stt()
        stt.model.transcribe.return_value = (broken(), None)
        self.error("transcription_failed", stt.transcribe, np.ones(1600))

    def test_audio_to_target(self):
        import json
        from pathlib import Path
        from voice_agent.perception import DetectionStore
        from voice_agent.pipeline import preview

        audio = record(0.1, backend=self.backend())
        transcript = self.stt().transcribe(audio)
        store = DetectionStore()
        fixture = Path(__file__).resolve().parents[1] / "examples/voice_detections.json"
        store.update(json.loads(fixture.read_text()))
        result = preview(transcript.text, store)
        self.assertEqual(result["command"], {"action": "bring", "object": "bottle"})
        self.assertEqual(result["target"]["id"], 1)
        self.assertFalse(result["execution_allowed"])


if __name__ == "__main__":
    unittest.main()
