import pyrealsense2 as rs
import numpy as np
import cv2


# ============================================================
# SETTINGS
# ============================================================

# Максимальная разница глубины между соседними пикселями,
# чтобы считать их частью одной гладкой поверхности
LOCAL_DEPTH_STEP_MM = 1.5

# Минимальная площадь поверхности в пикселях
# Уменьшай, если не видит маленькие поверхности
MIN_SURFACE_AREA = 150

# Ограничение по расстоянию от камеры
MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 1.50

# Размер ROI
ROI_WIDTH = 500
ROI_HEIGHT = 380

# Максимум отображаемых поверхностей
MAX_SURFACES = 15

# Прозрачность заливки
OVERLAY_ALPHA = 0.45


# ============================================================
# COLORS
# ============================================================

COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 150, 0),
    (150, 0, 255),
    (0, 150, 255),
    (150, 255, 0),
]


# ============================================================
# REALSENSE SETUP
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)


# ============================================================
# DEPTH FILTERS
# ============================================================

spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
hole_filter = rs.hole_filling_filter()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # GET FRAMES
        # ----------------------------------------------------

        frames = pipeline.wait_for_frames()

        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue


        # ----------------------------------------------------
        # FILTER DEPTH
        # ----------------------------------------------------

        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_filter.process(depth_frame)


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        color = np.asanyarray(
            color_frame.get_data()
        )

        depth_raw = np.asanyarray(
            depth_frame.get_data()
        )


        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        depth = (
            depth_raw.astype(np.float32)
            * depth_scale
        )


        h, w = depth.shape


        # ====================================================
        # ROI
        # ====================================================

        cx = w // 2
        cy = h // 2

        x1 = max(
            0,
            cx - ROI_WIDTH // 2
        )

        x2 = min(
            w,
            cx + ROI_WIDTH // 2
        )

        y1 = max(
            0,
            cy - ROI_HEIGHT // 2
        )

        y2 = min(
            h,
            cy + ROI_HEIGHT // 2
        )


        roi_depth = depth[
            y1:y2,
            x1:x2
        ]

        roi_color = color[
            y1:y2,
            x1:x2
        ]


        # ====================================================
        # VALID DEPTH MASK
        # ====================================================

        valid = (
            (roi_depth > MIN_DEPTH_M)
            &
            (roi_depth < MAX_DEPTH_M)
        )


        # ====================================================
        # LOCAL DEPTH DIFFERENCE
        # ====================================================

        tolerance = (
            LOCAL_DEPTH_STEP_MM
            / 1000.0
        )


        # Разница с соседями
        diff_left = np.zeros_like(
            roi_depth
        )

        diff_right = np.zeros_like(
            roi_depth
        )

        diff_up = np.zeros_like(
            roi_depth
        )

        diff_down = np.zeros_like(
            roi_depth
        )


        diff_left[:, 1:] = np.abs(
            roi_depth[:, 1:]
            -
            roi_depth[:, :-1]
        )

        diff_right[:, :-1] = np.abs(
            roi_depth[:, :-1]
            -
            roi_depth[:, 1:]
        )

        diff_up[1:, :] = np.abs(
            roi_depth[1:, :]
            -
            roi_depth[:-1, :]
        )

        diff_down[:-1, :] = np.abs(
            roi_depth[:-1, :]
            -
            roi_depth[1:, :]
        )


        # ====================================================
        # FLAT SURFACE MASK
        # ====================================================

        flat_mask = (
            valid
            &
            (diff_left < tolerance)
            &
            (diff_right < tolerance)
            &
            (diff_up < tolerance)
            &
            (diff_down < tolerance)
        )


        # Убираем шум
        flat_mask = (
            flat_mask.astype(np.uint8)
            * 255
        )


        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        flat_mask = cv2.morphologyEx(
            flat_mask,
            cv2.MORPH_OPEN,
            kernel
        )


        # ====================================================
        # CONNECTED COMPONENTS
        # ====================================================

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                flat_mask,
                connectivity=8
            )
        )


        # ====================================================
        # SORT SURFACES BY AREA
        # ====================================================

        surfaces = []

        for label in range(
            1,
            num_labels
        ):

            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area < MIN_SURFACE_AREA:
                continue


            surface_mask = (
                labels == label
            )


            depths = roi_depth[
                surface_mask
            ]


            depths = depths[
                depths > 0
            ]


            if len(depths) == 0:
                continue


            median_depth = np.median(
                depths
            )


            # Разброс depth внутри поверхности
            depth_std = np.std(
                depths
            )


            surfaces.append(
                {
                    "label": label,
                    "area": area,
                    "depth": median_depth,
                    "std": depth_std
                }
            )


        surfaces = sorted(
            surfaces,
            key=lambda s: s["area"],
            reverse=True
        )


        surfaces = surfaces[
            :MAX_SURFACES
        ]


        # ====================================================
        # DRAW
        # ====================================================

        overlay = color.copy()


        for i, surface in enumerate(
            surfaces
        ):

            label = surface["label"]

            mask = (
                labels == label
            )


            plane_color = COLORS[
                i % len(COLORS)
            ]


            # -----------------------------------------------
            # COLOR SURFACE
            # -----------------------------------------------

            ys, xs = np.where(mask)

            global_x = xs + x1
            global_y = ys + y1


            original_pixels = overlay[
                global_y,
                global_x
            ].astype(np.float32)


            new_color = np.array(
                plane_color,
                dtype=np.float32
            )


            overlay[
                global_y,
                global_x
            ] = (

                original_pixels
                * (1.0 - OVERLAY_ALPHA)

                +

                new_color
                * OVERLAY_ALPHA

            ).astype(np.uint8)


            # -----------------------------------------------
            # CENTER
            # -----------------------------------------------

            center_x = int(
                np.mean(global_x)
            )

            center_y = int(
                np.mean(global_y)
            )


            # -----------------------------------------------
            # LABEL
            # -----------------------------------------------

            depth_m = surface["depth"]

            text = (
                f"S{i + 1}: "
                f"{depth_m:.3f} m"
            )


            cv2.circle(
                overlay,
                (
                    center_x,
                    center_y
                ),
                6,
                plane_color,
                -1
            )


            cv2.putText(
                overlay,
                text,
                (
                    center_x + 10,
                    center_y - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                plane_color,
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # ROI
        # ====================================================

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            1
        )


        # ====================================================
        # INFO
        # ====================================================

        cv2.putText(
            overlay,
            f"Surfaces: {len(surfaces)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            f"Depth step: {LOCAL_DEPTH_STEP_MM:.1f} mm",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "D435 Surface Detection",
            overlay
        )


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break


finally:

    pipeline.stop()
    cv2.destroyAllWindows()