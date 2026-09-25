#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_PID=""
CV_PID=""

cleanup() {
    if [[ -n "$CV_PID" ]]; then
        kill "$CV_PID" 2>/dev/null || true
        wait "$CV_PID" 2>/dev/null || true
    fi

    if [[ -n "$CAMERA_PID" ]]; then
        # ros2 launch creates child processes. Kill the whole session so the
        # camera node cannot remain orphaned and keep the USB device busy.
        kill -- "-$CAMERA_PID" 2>/dev/null || kill "$CAMERA_PID" 2>/dev/null || true
        wait "$CAMERA_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    source /opt/ros/jazzy/setup.bash
fi

set -u

if pgrep -f '/realsense2_camera/realsense2_camera_node' >/dev/null; then
    echo "ERROR: RealSense camera node is already running."
    echo "Stop the old node first: pkill -f realsense2_camera_node"
    exit 1
fi

setsid ros2 launch realsense2_camera rs_launch.py \
    enable_sync:=true \
    align_depth.enable:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 &
CAMERA_PID=$!

sleep 4

if ! kill -0 "$CAMERA_PID" 2>/dev/null; then
    echo "ERROR: RealSense launch stopped during startup."
    exit 1
fi

python3 "$PROJECT_DIR/yolo_cv_publisher.py" &
CV_PID=$!

sleep 2

python3 "$PROJECT_DIR/test/yolo_depth_test.py"
