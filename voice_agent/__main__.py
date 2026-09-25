import argparse
import json
import os
import sys
from pathlib import Path

from .commands import parse_command
from .errors import VoiceError
from .perception import DetectionStore
from .pipeline import preview


def main(argv=None):
    parser = argparse.ArgumentParser(description="Voice-to-agent CLI. Always preview-only; never moves a robot.")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("devices", help="List microphone devices")
    parse = modes.add_parser("parse", help="Parse English text into strict command JSON")
    parse.add_argument("text")
    pipeline = modes.add_parser("pipeline", help="Match a command to a detection fixture without ROS")
    pipeline.add_argument("--text", required=True)
    pipeline.add_argument("--detections", type=Path)
    for name in ("stt", "listen"):
        mode = modes.add_parser(name, help="Transcribe audio" if name == "stt" else "Transcribe, parse, and match detections")
        mode.add_argument("--device", default=os.environ.get("VOICE_INPUT_DEVICE"), help="Input index/name; default: system input")
        mode.add_argument("--seconds", type=float, default=5)
        mode.add_argument("--audio", help="Audio file instead of microphone")
        mode.add_argument("--model", default="tiny.en")
        mode.add_argument("--cache-dir")
        mode.add_argument("--local-files-only", action="store_true")
        mode.add_argument("--min-confidence", type=float, default=0.5)
        mode.add_argument("--no-prompt", action="store_true", help="Record immediately after model loading")
        if name == "listen":
            mode.add_argument("--detections", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "devices":
            from .stt import list_devices
            result = list_devices()
            if not result:
                raise VoiceError("no_microphone", "No input devices are visible.")
        elif args.mode == "parse":
            result = parse_command(args.text).as_dict()
        else:
            if args.mode in ("stt", "listen"):
                from .stt import SpeechToText, record
                print("Loading Whisper...", file=sys.stderr)
                stt = SpeechToText(args.model, args.min_confidence, cache_dir=args.cache_dir,
                                   local_files_only=args.local_files_only)
                if args.audio:
                    audio = args.audio
                else:
                    if not args.no_prompt:
                        print("Press ENTER, then speak.", file=sys.stderr)
                        input()
                    print(f"Listening for {args.seconds:g} seconds...", file=sys.stderr)
                    audio = record(args.seconds, args.device)
                transcript = stt.transcribe(audio)
                text = transcript.text
            else:
                text = args.text
            if args.mode == "stt":
                result = transcript.as_dict()
            else:
                store = DetectionStore()
                # Parse first so Stop does not depend on perception or its file.
                if parse_command(text).action != "stop" and args.detections:
                    store.update(args.detections.read_text())
                result = preview(text, store)
                result["detection_source"] = "fixture" if args.detections else "none"
                if args.mode == "listen":
                    result["transcript"] = transcript.as_dict()
        print(json.dumps(result, allow_nan=False))
        return 0
    except VoiceError as exc:
        print(json.dumps(exc.as_dict()), file=sys.stderr)
        return 2
    except (OSError, EOFError) as exc:
        print(json.dumps(VoiceError("input_error", str(exc)).as_dict()), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
