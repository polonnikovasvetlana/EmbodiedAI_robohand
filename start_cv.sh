#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_PID=""
CV_PID=""

cleanup() {
    if [[ -n "$CV_PID" ]]; then
        kill "$CV_PID" 2>/dev/null || true
    fi
    if [[ -n "$CAMERA_PID" ]]; then
        kill -- "-$CAMERA_PID" 2>/dev/null || kill "$CAMERA_PID" 2>/dev/null || true
    fi
    wait "$CV_PID" 2>/dev/null || true
    wait "$CAMERA_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    source /opt/ros/jazzy/setup.bash
fi

set -u

setsid ros2 launch realsense2_camera rs_launch.py \
    enable_sync:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 &
CAMERA_PID=$!

sleep 4

python3 "$PROJECT_DIR/cv_publisher.py" &
CV_PID=$!

sleep 2

python3 "$PROJECT_DIR/test/cv_publisher_test.py"
