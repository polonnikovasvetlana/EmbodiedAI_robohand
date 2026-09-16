import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 640
HEIGHT = 480
FPS = 30

# Всё, что выше поверхности стола больше чем на столько мм,
# считаем объектом
MIN_HEIGHT_MM = 10

# Игнорируем слишком высокие вещи
MAX_HEIGHT_MM = 250

# Минимальная площадь объекта
MIN_AREA = 500

# Сколько кадров использовать для запоминания пустого стола
CALIBRATION_FRAMES = 30


# ============================================================
# REALSENSE
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    WIDTH,
    HEIGHT,
    rs.format.bgr8,
    FPS
)

config.enable_stream(
    rs.stream.depth,
    WIDTH,
    HEIGHT,
    rs.format.z16,
    FPS
)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align = rs.align(rs.stream.color)

background_depth = None


# ============================================================
# COLOR
# ============================================================

def get_color_name(h, s, v):

    # -------- grayscale --------

    if s < 25:

        if v < 55:
            return "black"

        elif v < 170:
            return "gray"

        else:
            return "white"


    # -------- beige --------

    if 8 <= h <= 30 and s < 90 and v > 100:
        return "beige"


    # -------- brown --------

    if 5 <= h <= 20 and v < 150:
        return "brown"


    # -------- normal colors --------

    if h < 8 or h >= 172:
        return "red"

    elif h < 22:
        return "orange"

    elif h < 38:
        return "yellow"

    elif h < 80:
        return "green"

    elif h < 100:
        return "cyan"

    elif h < 130:
        return "blue"

    elif h < 155:
        return "purple"

    elif h < 172:
        return "pink"

    return "unknown"


# ============================================================
# SHAPE
# ============================================================

def classify_shape(cnt):

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter <= 0:
        return "object"

    circularity = (
        4 * np.pi * area /
        (perimeter * perimeter)
    )

    approx = cv2.approxPolyDP(
        cnt,
        0.03 * perimeter,
        True
    )

    vertices = len(approx)

    rect = cv2.minAreaRect(cnt)
    (_, _), (w, h), _ = rect

    if w <= 0 or h <= 0:
        return "object"

    aspect = min(w, h) / max(w, h)

    rect_area = w * h
    fill = area / rect_area if rect_area > 0 else 0


    # Круг
    if circularity > 0.82:
        return "circle"


    # Square / rectangle
    if 4 <= vertices <= 6 and fill > 0.72:

        if aspect > 0.78:
            return "square"

        else:
            return "rectangle"


    # Например мышка
    if circularity > 0.50:
        return "oval"


    return "object"


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_table():

    print()
    print("REMOVE movable objects from the table!")
    print("Do not move the camera.")
    print("Calibrating...")
    print()

    collected = []

    while len(collected) < CALIBRATION_FRAMES:

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        depth = np.asanyarray(
            depth_frame.get_data()
        ).astype(np.float32)

        depth *= depth_scale

        depth[depth <= 0] = np.nan

        collected.append(depth)

        print(
            f"{len(collected)}/{CALIBRATION_FRAMES}",
            end="\r"
        )


    stack = np.stack(collected)

    background = np.nanmedian(
        stack,
        axis=0
    )

    print()
    print("TABLE CALIBRATED")
    print()

    return background


# ============================================================
# DETECTION
# ============================================================

def detect_objects(
    frame,
    depth,
    background
):

    # --------------------------------------------------------
    # Height over table
    # --------------------------------------------------------

    height = background - depth

    min_h = MIN_HEIGHT_MM / 1000
    max_h = MAX_HEIGHT_MM / 1000


    valid = (
        np.isfinite(depth)
        &
        np.isfinite(background)
    )


    mask_bool = (
        valid
        &
        (height > min_h)
        &
        (height < max_h)
    )


    mask = (
        mask_bool.astype(np.uint8)
        * 255
    )


    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    kernel3 = np.ones(
        (3, 3),
        np.uint8
    )

    kernel5 = np.ones(
        (5, 5),
        np.uint8
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel3
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel5
    )


    # --------------------------------------------------------
    # CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < MIN_AREA:
            continue


        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue


        cx = int(
            M["m10"] /
            M["m00"]
        )

        cy = int(
            M["m01"] /
            M["m00"]
        )


        # ----------------------------------------------------
        # OBJECT MASK
        # ----------------------------------------------------

        obj_mask = np.zeros(
            mask.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            obj_mask,
            [cnt],
            -1,
            255,
            -1
        )


        # немного уменьшаем mask внутрь,
        # чтобы не брать цвет стола по краям

        obj_mask = cv2.erode(
            obj_mask,
            np.ones((5, 5), np.uint8),
            iterations=1
        )


        pixels = hsv[
            obj_mask == 255
        ]


        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        if len(pixels) > 0:

            h = int(np.median(pixels[:, 0]))
            s = int(np.median(pixels[:, 1]))
            v = int(np.median(pixels[:, 2]))

            color = get_color_name(
                h, s, v
            )

        else:

            color = "unknown"


        # ----------------------------------------------------
        # SHAPE
        # ----------------------------------------------------

        shape = classify_shape(cnt)


        # ----------------------------------------------------
        # HEIGHT
        # ----------------------------------------------------

        heights = height[
            obj_mask == 255
        ]

        heights = heights[
            np.isfinite(heights)
        ]

        if len(heights):

            object_height = np.median(
                heights
            ) * 1000

        else:

            object_height = 0


        # ----------------------------------------------------
        # DRAW REAL CONTOUR
        # ----------------------------------------------------

        cv2.drawContours(
            frame,
            [cnt],
            -1,
            (0, 255, 0),
            2
        )


        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )


        x, y, w, h_box = cv2.boundingRect(cnt)


        label = f"{color} {shape}"


        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"H: {object_height:.0f} mm",
            (x, y + h_box + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )


        cv2.putText(
            frame,
            f"({cx}, {cy})",
            (x, y + h_box + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )


    return mask


# ============================================================
# MAIN
# ============================================================

print()
print("C = calibrate empty table")
print("Q = quit")
print()


try:

    while True:

        frames = pipeline.wait_for_frames()

        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue


        frame = np.asanyarray(
            color_frame.get_data()
        )


        depth = np.asanyarray(
            depth_frame.get_data()
        ).astype(np.float32)

        depth *= depth_scale

        depth[depth <= 0] = np.nan


        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        if background_depth is not None:

            mask = detect_objects(
                frame,
                depth,
                background_depth
            )

            cv2.imshow(
                "Depth mask",
                mask
            )


        else:

            cv2.putText(
                frame,
                "Remove objects and press C",
                (120, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )


        cv2.imshow(
            "Object detection",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):
            break


        if key == ord("c"):

            background_depth = (
                calibrate_table()
            )


finally:

    pipeline.stop()
    cv2.destroyAllWindows()