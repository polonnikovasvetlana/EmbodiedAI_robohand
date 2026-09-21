# Voice agent

The `voice_agent` package uses **faster-whisper** for English speech-to-text
(`tiny.en`, CPU/int8). Microphone input uses sounddevice/PortAudio with bounded
mono recording and resampling to 16 kHz when needed. Audio files are also supported;
empty or low-confidence speech is rejected.

Natural-language command parsing uses a small deterministic grammar for
“Bring me the bottle/cup”, “Pick up the bottle/cup”, and “Stop”, with optional
“please”. For example, “Pick up the bottle” produces:

```json
{"action":"pick_up","object":"bottle"}
```

Unsupported, negated, and compound commands are rejected.

## Examples

Run from the repository root with Python 3.10+. Audio needs PortAudio and the
listed dependencies; the parser and mock pipeline need only Python.

```bash
python -m pip install -r requirements-voice.txt
python -m voice_agent parse "Pick up the bottle"
python -m voice_agent pipeline --text "Bring me the cup" --detections examples/voice_detections.json
python -m voice_agent devices
python -m voice_agent stt --seconds 5
python -m voice_agent listen --detections examples/voice_detections.json
python -m unittest discover -s tests_voice -v
```

Audio commands load the model, then prompt before recording. The first load may
download model weights. Use `--device` to select a microphone or
`--audio /path/to/command.wav` to transcribe a file.

## Testing and robot control

The mock detection fixture lets the pipeline match commands to a unique bottle
or cup without a camera. Tests mock audio, Whisper, and ROS callbacks; they check
parsing, confidence, stale or ambiguous detections, and preview results. Run only
`tests_voice`: other repository test scripts can access physical hardware.

The AI layer is separate from physical robot control. Results always contain
`execution_allowed: false` and `robot_pose: null`; no movement is executed.
“Stop” is an intent, not a physical emergency stop.

## Next steps

Validate the optional ROS 2 preview bridge with live detections. Calibrate aligned
RGB/depth and camera-to-robot transforms, then build a separate MoveIt / SO-101
executor with fresh-target checks, collision checking, grasp policies, and
cancellation. Test in simulation before separately authorized hardware testing.
See [the detailed guide](docs/voice_agent.md) for ROS topics and setup.
