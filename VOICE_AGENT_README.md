# Whisper voice agent

The `voice_agent` package converts English speech into structured command JSON
using **faster-whisper** (CPU/int8, `tiny.en` by default). Microphone input uses
sounddevice/PortAudio, with bounded mono recording and resampling to 16 kHz when
needed. Audio files are also supported. Low-confidence or empty speech is rejected.

Natural-language parsing uses a small deterministic grammar: `bring [me] [the|a]
bottle|cup`, `pick up [the|a] bottle|cup`, and `stop`, with optional `please`.
For example, “Pick up the bottle” becomes `{"action":"pick_up","object":"bottle"}`.
Unsupported, negated, and compound commands are rejected.

## Run

Use Python 3.10+ from the repository root. Audio requires PortAudio and the
dependencies below; parser and fixture pipeline commands need only Python.

```bash
python -m pip install -r requirements-voice.txt
python -m voice_agent parse "Pick up the bottle"
python -m voice_agent pipeline --text "Bring me the cup" --detections examples/voice_detections.json
python -m voice_agent devices
python -m voice_agent stt --seconds 5
python -m voice_agent listen --detections examples/voice_detections.json
python -m unittest discover -s tests_voice -v
```

Audio commands load the model before prompting for microphone input. The first
load may download model weights; `--model /path/to/model --local-files-only`
uses a local model. `--device` selects an input and `--audio /path/to/command.wav`
uses a recording. `speech_to_text.py` is the original standalone prototype;
the package CLI provides validation and the command pipeline.

## Mock testing and control boundary

The example detections are a mock fixture. The pipeline resolves a unique bottle
or cup and rejects missing, stale, ambiguous, or low-confidence detections.
Tests use mocked audio, Whisper, and ROS callbacks without hardware or downloads.
Run only `tests_voice`; other repository test scripts can access physical hardware.

The AI layer is separated from physical robot control: results have
`execution_allowed: false` and `robot_pose: null`. No movement is executed.
`stop` is currently an intent only, not a physical emergency stop.

## Next steps

Validate live ROS 2 transport using the optional `voice_agent.ros_node` preview
bridge and real detections. Calibrate aligned RGB/depth and camera-to-robot TF,
then develop a separate executor for MoveIt and SO-101 with fresh-target checks,
collision checking, grasp policies, cancellation, and explicit execution control.
Test that executor in simulation before separately authorized hardware testing.
See [the detailed guide](docs/voice_agent.md) for topic contracts and setup.
