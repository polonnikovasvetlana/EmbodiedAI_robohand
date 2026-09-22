#!/usr/bin/env python3
"""Automatically collect RGB images from an Intel RealSense for a YOLO dataset."""

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None


DEFAULT_OUTPUT_DIR = Path("/home/svetlana/Pictures/IW_dataset")
IMAGE_NAME_RE = re.compile(r"^image_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
WINDOW_NAME = "YOLO dataset capture"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save RealSense RGB frames for a YOLO dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="number of saved images per second (default: 2)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100 (default: 95)",
    )
    return parser.parse_args()


def find_next_image_number(output_dir):
    """Continue after the greatest existing image_N file number."""
    greatest_number = -1
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = IMAGE_NAME_RE.match(path.name)
        if match:
            greatest_number = max(greatest_number, int(match.group(1)))
    return greatest_number + 1


def save_image(frame, output_dir, next_number, quality):
    """Save one clean frame without overwriting an existing image."""
    image_path = output_dir / f"image_{next_number}.jpg"
    while image_path.exists():
        next_number += 1
        image_path = output_dir / f"image_{next_number}.jpg"

    saved = cv2.imwrite(
        str(image_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not saved:
        raise RuntimeError(f"Could not save image: {image_path}")
    return image_path, next_number + 1


def draw_status(frame, auto_mode, seconds_left, saved_count, next_number):
    """Draw preview-only status text on a copy of the camera frame."""
    preview = frame.copy()
    overlay = preview.copy()
    cv2.rectangle(overlay, (0, 0), (preview.shape[1], 82), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, preview, 0.42, 0, preview)

    if not auto_mode:
        status = "MANUAL: SPACE = photo, ENTER = auto"
        status_color = (0, 200, 255)
    else:
        status = f"AUTO: next photo in {seconds_left:.2f} s, ESC = stop"
        status_color = (90, 255, 90)

    cv2.putText(
        preview,
        status,
        (16, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"Saved this run: {saved_count}    Next file: image_{next_number}.jpg",
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def add_flash(preview, saved_name):
    """Add a bright tint and border so a saved frame is easy to notice."""
    white = np.full_like(preview, 255)
    preview = cv2.addWeighted(preview, 0.55, white, 0.45, 0)
    cv2.rectangle(
        preview,
        (5, 5),
        (preview.shape[1] - 6, preview.shape[0] - 6),
        (0, 255, 0),
        10,
    )
    cv2.putText(
        preview,
        f"SAVED: {saved_name}",
        (20, preview.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 120, 0),
        2,
        cv2.LINE_AA,
    )
    return preview


def main():
    args = parse_args()
    if rs is None:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment. "
            "Install it with: python3 -m pip install pyrealsense2"
        )
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than zero")
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    next_number = find_next_image_number(output_dir)

    interval = 1.0 / args.fps
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        30,
    )

    print(f"Saving images to: {output_dir}")
    print(f"Starting with: image_{next_number}.jpg")
    print("Controls: Space = one photo, Enter = auto mode, Esc = manual mode, Q = exit")

    pipeline_started = False
    try:
        pipeline.start(config)
        pipeline_started = True
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        next_capture_at = time.monotonic()
        flash_until = 0.0
        last_saved_name = ""
        saved_count = 0
        auto_mode = False

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            now = time.monotonic()

            if auto_mode and now >= next_capture_at:
                image_path, next_number = save_image(
                    frame,
                    output_dir,
                    next_number,
                    args.quality,
                )
                last_saved_name = image_path.name
                print(f"Saved {last_saved_name}")
                saved_count += 1
                flash_until = now + 0.16

                # Keep a steady cadence without trying to catch up after a delay.
                next_capture_at += interval
                if next_capture_at <= now:
                    next_capture_at = now + interval

            seconds_left = 0.0 if not auto_mode else max(0.0, next_capture_at - now)
            preview = draw_status(
                frame,
                auto_mode,
                seconds_left,
                saved_count,
                next_number,
            )
            if now < flash_until:
                preview = add_flash(preview, last_saved_name)

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (10, 13):
                if not auto_mode:
                    auto_mode = True
                    next_capture_at = time.monotonic()
                    print("Auto capture started")
            elif key == 27:
                if auto_mode:
                    auto_mode = False
                    print("Auto capture stopped; manual mode enabled")
            elif key == ord(" ") and not auto_mode:
                image_path, next_number = save_image(
                    frame,
                    output_dir,
                    next_number,
                    args.quality,
                )
                last_saved_name = image_path.name
                print(f"Saved manually: {last_saved_name}")
                saved_count += 1
                flash_until = time.monotonic() + 0.16

    except RuntimeError as exc:
        raise SystemExit(f"Camera error: {exc}") from exc
    finally:
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()

    print(f"Done. Saved {saved_count} images this run.")


if __name__ == "__main__":
    main()
