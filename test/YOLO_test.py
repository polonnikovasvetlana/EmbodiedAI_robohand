import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


# Готовая pretrained YOLO
model = YOLO("yolo26n.pt")


# RealSense
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

pipeline.start(config)

print("Camera started")
print("Press Q to quit")


try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # YOLO
        results = model(
            frame,
            conf=0.4,
            verbose=False
        )

        # Картинка с bbox + названиями объектов
        annotated_frame = results[0].plot()

        cv2.imshow("RealSense + YOLO", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()