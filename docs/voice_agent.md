# Voice-to-agent pipeline

The new `voice_agent` package is preview-only. It contains no robot-controller,
joint-angle, motor, trajectory, or MoveIt client. No physical action is executed,
including when the command is `stop`. Spoken Stop is an intent, not an emergency
stop: capture and transcription take time, and no controller is connected yet.

## Repository architecture inspected before implementation

All 20 existing Python files and three shell scripts were inspected (including
the untracked `speech_to_text.py`). Existing files were left unchanged.

| Existing files | Role |
| --- | --- |
| `cv_publisher.py`, `calibration_config.py` | Depth-background segmentation, shape/color labels, approximate calibrated tabletop XY in mm |
| `yolo_cv_publisher.py` | YOLO semantic labels, pixel centers, bbox confidence and depth range |
| `so101_control/arm.py`, `__init__.py` | ROS 2 MoveGroup `/move_action`, gripper action, joint feedback and TF; Cartesian inputs in mm, current-limited grasp logic |
| `so101_control copy/arm.py`, `__init__.py` | Older controller using `/compute_ik` then joint constraints |
| `start.sh`, `start_cv.sh`, `start_yolo.sh` | Camera + detector + visualizer launchers |
| `test/cv_publisher_test.py`, `test/yolo_depth_test.py` | ROS image/detection visualizers |
| `test/YOLO_test.py`, `test/intel_cam_test.py`, `test/cv1_simple_test.py`, `test/depth_test.py` | Direct RealSense/vision experiments |
| `test/test_so101_control.py` | Interactive ROS arm controller |
| `test/test_pan.py`, `test/test_so101.py`, `test/so101_xyz.py` | Hardware trajectory, torque and LeRobot Cartesian experiments |
| `test_sts_raw.py`, `read_sts_limits.py`, `shit/move_xyz.py` | Direct Feetech/LeRobot diagnostics and control |
| `speech_to_text.py` | Existing five-second Whisper prototype; preserved |
| `how_to_work.txt`, `.vscode/`, `conda-list.txt` | Workstation setup references; external `so101_real_ws`, Jazzy, domain 42; environment snapshot from another workstation |

There is no local ROS package manifest or launch package. The new bridge follows
the existing standalone Python ROS-node layout. Some existing test scripts
access hardware at import time; do **not** run repository-wide test discovery.

## Run locally

From the repository root:

```bash
conda activate agent35
python -m voice_agent devices
python -m voice_agent stt
python -m voice_agent parse "Bring me the bottle"
python -m voice_agent parse "Pick up the bottle"
python -m voice_agent parse "Stop"
python -m voice_agent pipeline --text "Bring me the cup" --detections examples/voice_detections.json
python -m voice_agent listen --detections examples/voice_detections.json
python -m unittest discover -s tests_voice -v
```

`stt` loads the model, waits for Enter, records five seconds, and prints transcript
JSON. `listen` additionally parses and matches the recorded command against a
fixture. Fixtures are explicitly marked as such; these coordinates are not live.
`pipeline` and `parse` need only Python's standard library and never open audio,
ROS, or hardware. Errors are JSON on stderr with exit code 2; Ctrl+C returns 130.

Options for both audio modes:

```bash
python -m voice_agent stt --device 3 --seconds 4
python -m voice_agent stt --device "USB" --model base.en
python -m voice_agent stt --audio /path/to/command.wav
python -m voice_agent stt --model /path/to/local-whisper-model --local-files-only
```

Device indexes/names come from `devices`. Default is PortAudio's system input;
`VOICE_INPUT_DEVICE` overrides it, and `--device` takes precedence. Device names
must match unambiguously. Capture tries mono 16 kHz, then the selected device's
native rate with `scipy.signal.resample_poly` to 16 kHz. An invalid explicit
device fails instead of switching microphones. Audio overflow is rejected.

`tiny.en` uses CPU/int8. Its initial load downloads model weights unless already
cached or `--local-files-only` is set. Use `--cache-dir` to choose the download
location. Dependencies are in `requirements-voice.txt`; PortAudio is a system
library. No LLM, API key, GPU, or cloud transcription service is needed.

The transcription confidence is a conservative heuristic: the minimum across
nonempty segments of `min(exp(avg_logprob), 1 - no_speech_prob)`. It is **not** a
calibrated probability. Default rejection threshold is 0.5, configurable with
`--min-confidence`. VAD and silence checks reduce false transcripts but do not
guarantee recognition accuracy. Validate the threshold with actual voices/noise.

The exact supported grammar is `bring [me] [the|a] bottle|cup`,
`pick up [the|a] bottle|cup`, and `stop`, with optional leading/trailing `please`,
case/whitespace normalization and terminal punctuation. Other actions, compound
commands and negations fail closed. Objects outside bottle/cup return
`unknown_object`. Commands have exactly `action` and `object`:

```json
{"action": "bring", "object": "bottle"}
```

`pick up` becomes `pick_up`; Stop becomes `{"action":"stop","object":null}`.

## Vision contract and ROS preview

Both existing publishers emit `std_msgs/msg/String` JSON on `/detected_objects`:
`{"objects":[...]}`. Only run one publisher on this topic at a time.

* YOLO: `type` is e.g. bottle/cup; `position` is `[u,v]` in pixels;
  `distance_mm` is median depth (nullable); `confidence` is detector confidence;
  top-level image dimensions and RGB `stamp` are present. Its `height_mm=0` is
  a placeholder, not measured object height. IDs are reassigned left-to-right
  every frame and cannot be used as persistent tracking IDs.
* Classical CV: `type` is circle/square/rectangle/oval/object, not bottle/cup.
  `position_mm` is approximate tabletop XY from `calibration_config.py` and
  `height_mm` is height above the calibrated background. No timestamp or
  semantic confidence is published. This cannot resolve bottle/cup requests.

The adapter rejects malformed, absent, expired (default 2 seconds), low-confidence
(default 0.4), and ambiguous detections. Empty/new-invalid frames invalidate the
old target. ROS checks YOLO source time as well as monotonic receipt age; source
and subscriber must share the ROS clock (`use_sim_time` when appropriate).
Unstamped legacy messages can only be aged from receipt. Offline fixtures are
deliberately treated as fresh snapshots when loaded, without source-clock checks.

Preview targets expose `pixel_position`, `distance_mm`, `workspace_xy_mm`,
`height_mm`, confidence, source stamp and age. `robot_pose` is always null,
`requires_calibration` is true, and `execution_allowed` is false. Null depth is
permitted as an observation, never converted into a robot goal.

In a terminal with ROS 2 Jazzy and its compatible Python available:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
python3 -m voice_agent.ros_node
```

The bridge subscribes to `/voice_agent/transcript` and `/detected_objects`, and
publishes `/voice_agent/intent_preview` or errors on `/voice_agent/status`, all
`std_msgs/msg/String`. Transcript JSON must contain exactly `text`, `confidence`,
and `language: "en"`. Confidence is validated again at this boundary. QoS is
reliable/volatile, matching the existing detection publishers. Topics can be
remapped using standard `--ros-args -r`; thresholds are ROS parameters:

```bash
python3 -m voice_agent.ros_node --ros-args -p max_detection_age:=2.0 -p min_transcription_confidence:=0.5
ros2 topic echo /voice_agent/intent_preview
ros2 topic echo /voice_agent/status
```

For separate Conda and ROS interpreters, record in the Conda terminal:

```bash
python -m voice_agent stt > /tmp/voice-transcript.json
```

Then send the resulting JSON file from a ROS terminal in the same repository:

```bash
python3 -m voice_agent.ros_publish /tmp/voice-transcript.json
```

The sender waits up to five seconds for a subscriber. This process boundary avoids
forcing Conda's native libraries into the ROS Python environment. Use a fresh
transcript each time; the sender is a manual one-shot transport, not a durable
command queue. DDS acknowledgement does not imply robot execution.

## Later physical-controller integration

Implement a separate, explicitly enabled task executor after manual calibration
and approval for hardware work. Keep the language package free of controller
imports. Do not connect a motor subscriber to the preview topic: create a distinct
execution interface with request IDs, expiration, cancellation and result status.

The executor must validate the command allowlist, re-resolve a fresh unique
target, and obtain a calibrated robot-frame grasp pose. YOLO currently samples
the raw depth image by scaled RGB bounding boxes; scaling is not RGB/depth
registration. Establish aligned depth, camera intrinsics, camera-to-base TF and
image/depth time synchronization before calculating XYZ. Classical tabletop XY
also needs validation against `base_link`; do not assume its origin is the arm
origin. Neither publisher currently supplies a ready-to-execute robot grasp pose.

The deterministic executor owns approach/grasp/lift poses, collision/workspace
checks, gripper policies and the delivery location for `bring`. It can then call
the existing `SO101Arm.move/open/close` methods and inspect `MotionResult`.
The agent must never supply joint targets, arbitrary poses, or motor commands.
Stop must bypass normal task queues, cancel the task state machine and invoke
controller cancellation; validate cancellation during planning, movement and
gripper close. The existing `stop()` alone does not implement a task-level stop
latch. These physical behaviors are intentionally not implemented or exercised
by this voice preview package.

## Manual verification still required

1. Confirm microphone selection, record each supported phrase, test silence/noise
   and adjust the confidence threshold if required.
2. Run the ROS bridge with the existing YOLO publisher on the robot workstation;
   check transcript delivery, fresh detections, ambiguity, source-clock settings
   and camera disconnection. No MoveIt launch is needed for this preview test.
3. Calibrate registered RGB/depth and robot-frame transforms, define grasp and
   delivery policies, and separately approve and test physical execution/Stop.

## Local validation on this workstation

* Python 3.12.14 in `agent35`: 24 isolated tests passed, covering parser, audio
  failure cases/resampling, confidence, target matching, CLI subprocesses and
  ROS callbacks with mocked middleware. No existing hardware tests were imported.
* All four requested phrases were run through the CLI; fixture-based target
  resolution and structured error exits were exercised successfully.
* Outside the sandbox, sounddevice listed six input devices and system-default
  input index 10 (`default`). Indexes can change between sessions.
* `tiny.en` downloaded to `/tmp/agenticbot-whisper`. Actual microphone capture and
  Whisper inference ran for one second. The result was rejected with score 0.411
  below threshold 0.5. This confirms the audio/inference path and rejection, not
  recognition accuracy for the command phrases. Reuse this temporary model cache
  with `--cache-dir /tmp/agenticbot-whisper`, or use a persistent cache later.
* ROS 2 / `rclpy` and `/opt/ros/jazzy` are absent here. Live DDS transport and the
  ROS sender remain untested; callback tests do not substitute for that test.
* No robot motion, controller connection, or existing-file modification occurred.

Implementation references: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and [sounddevice usage](https://github.com/spatialaudio/python-sounddevice/blob/master/doc/usage.rst).
