#!/bin/bash

ros2 launch realsense2_camera rs_launch.py \
    enable_sync:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 &
CAMERA_PID=$!

sleep 1.5

python3 cv_publisher.py &
CV_PID=$!

sleep 1.5

python3 /home/svetlana/AgenticBotIW/test/cv_publisher_test.py

kill $CAMERA_PID $CV_PID