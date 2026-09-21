#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_PID=""

cleanup() {
    if [[ -n "$CAMERA_PID" ]]; then
        kill -- "-$CAMERA_PID" 2>/dev/null || kill "$CAMERA_PID" 2>/dev/null || true
    fi
    wait "$CAMERA_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

source /opt/ros/jazzy/setup.bash

set -u

setsid ros2 launch realsense2_camera rs_launch.py \
    enable_sync:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 &
CAMERA_PID=$!

sleep 4
python3 "$PROJECT_DIR/aruco_calibration.py"
