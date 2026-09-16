import pyrealsense2 as rs
import numpy as np
import cv2

# Настраиваем камеру
pipeline = rs.pipeline()
config = rs.config()

# RGB
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Depth
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Запуск камеры
pipeline.start(config)

print("Camera started. Press Q to exit.")

try:
    while True:
        # Получаем новый кадр
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # RealSense frame -> numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Красивое отображение depth
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Показываем два окна
        cv2.imshow("RGB", color_image)
        cv2.imshow("Depth", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()