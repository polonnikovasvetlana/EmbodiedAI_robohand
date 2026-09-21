import cv2
import json
import time
import warnings
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from calibration_config import (
    TABLE_MARKER_POSITIONS_MM,
    TABLE_MARKER_SIZE_MM,
    TABLE_HOMOGRAPHY_FILE,
    WORKSPACE_HEIGHT_CM,
    WORKSPACE_WIDTH_CM,
    origin_pixel,
)

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)


# ============================================================
# SETTINGS
# ============================================================

COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"

DETECTION_TOPIC = "/detected_objects"

MIN_HEIGHT_MM = 10.0
MAX_HEIGHT_MM = 250.0

MIN_AREA = 500
MASK_OPEN_SIZE = 5
MASK_CLOSE_SIZE = 9
MARKER_EXCLUSION_MARGIN_MM = 35.0

CALIBRATION_FRAMES = 30

# camera 30 Hz
# 2 -> ~15 Hz CV
# 3 -> ~10 Hz CV
# 6 -> ~5 Hz CV
PROCESS_EVERY_N_FRAMES = 3

CONTOUR_EPSILON = 1.5


def load_table_homography():
    if not TABLE_HOMOGRAPHY_FILE.is_file():
        return None
    try:
        payload = json.loads(
            TABLE_HOMOGRAPHY_FILE.read_text(encoding="utf-8")
        )
        matrix = np.asarray(
            payload["homography_pixel_to_mm"],
            dtype=np.float32
        )
        if matrix.shape != (3, 3):
            raise ValueError("homography must be a 3x3 matrix")
        return matrix
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Warning: cannot load table homography: {exc}")
        return None


TABLE_HOMOGRAPHY = load_table_homography()


def table_mask(image_shape):
    if TABLE_HOMOGRAPHY is None:
        return None

    height, width = image_shape[:2]
    inverse = np.linalg.inv(TABLE_HOMOGRAPHY).astype(np.float32)
    positions = np.asarray(
        [TABLE_MARKER_POSITIONS_MM[marker_id][:2] for marker_id in (0, 2, 3, 1)],
        dtype=np.float32,
    )
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    table_corners_mm = np.array(
        [[[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]],
        dtype=np.float32,
    )
    table_corners_px = cv2.perspectiveTransform(
        table_corners_mm,
        inverse,
    )[0].round().astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, table_corners_px, 255)

    for marker_id in (0, 1, 2, 3):
        center_x, center_y = TABLE_MARKER_POSITIONS_MM[marker_id][:2]
        half_size = (
            TABLE_MARKER_SIZE_MM / 2.0
            + MARKER_EXCLUSION_MARGIN_MM
        )
        marker_corners_mm = np.array(
            [[[
                center_x - half_size,
                center_y - half_size,
            ], [
                center_x + half_size,
                center_y - half_size,
            ], [
                center_x + half_size,
                center_y + half_size,
            ], [
                center_x - half_size,
                center_y + half_size,
            ]]],
            dtype=np.float32,
        )
        marker_corners_px = cv2.perspectiveTransform(
            marker_corners_mm,
            inverse,
        )[0].round().astype(np.int32)
        cv2.fillConvexPoly(mask, marker_corners_px, 0)

    return mask


def fill_mask_holes(mask):
    flood = mask.copy()
    flood_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2),
        dtype=np.uint8
    )
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return mask | holes


def pixel_to_workspace_mm(cx, cy, image_width, image_height):
    if TABLE_HOMOGRAPHY is not None:
        source = np.array(
            [[[float(cx), float(cy)]]],
            dtype=np.float32
        )
        target = cv2.perspectiveTransform(
            source,
            TABLE_HOMOGRAPHY
        )[0, 0]
        return round(float(target[0]), 1), round(float(target[1]), 1)

    origin_x, origin_y = origin_pixel(
        image_width,
        image_height
    )
    x_mm = (
        (cy - origin_y)
        * WORKSPACE_HEIGHT_CM
        * 10.0
        / image_height
    )
    y_mm = (
        (origin_x - cx)
        * WORKSPACE_WIDTH_CM
        * 10.0
        / image_width
    )
    return round(x_mm, 1), round(y_mm, 1)


# ============================================================
# QOS
# ============================================================

SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


# ============================================================
# COLOR CLASSIFICATION
# ============================================================

def get_color_name(h, s, v):

    if s < 25:

        if v < 55:
            return "black"

        elif v < 170:
            return "gray"

        return "white"

    if 8 <= h <= 30 and s < 90 and v > 100:
        return "beige"

    if 5 <= h <= 20 and v < 150:
        return "brown"

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
# SHAPE CLASSIFICATION
# ============================================================

def classify_shape(contour):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter <= 0:
        return "object"

    circularity = (
        4.0 * np.pi * area
        /
        (perimeter * perimeter)
    )

    approx = cv2.approxPolyDP(
        contour,
        0.03 * perimeter,
        True
    )

    vertices = len(approx)

    rect = cv2.minAreaRect(
        contour
    )

    (_, _), (w, h), _ = rect

    if w <= 0 or h <= 0:
        return "object"

    aspect = (
        min(w, h)
        /
        max(w, h)
    )

    rect_area = w * h

    fill = (
        area / rect_area
        if rect_area > 0
        else 0
    )

    if circularity > 0.82:
        return "circle"

    if (
        4 <= vertices <= 6
        and fill > 0.72
    ):

        if aspect > 0.78:
            return "square"

        return "rectangle"

    if circularity > 0.50:
        return "oval"

    return "object"


# ============================================================
# NODE
# ============================================================

class CVDetectorNode(Node):

    def __init__(self):

        super().__init__(
            "cv_detector_node"
        )

        self.get_logger().info(
            "Starting CV detector..."
        )

        # Последнее сообщение depth.
        # Не копируем numpy на каждом depth frame.
        self.latest_depth_msg = None

        # Calibration
        self.calibration_frames = []
        self.background_depth = None

        self.frame_counter = 0

        self.last_object_count = None

        # kernels создаём один раз
        self.kernel3 = np.ones(
            (3, 3),
            dtype=np.uint8
        )

        self.kernel5 = np.ones(
            (5, 5),
            dtype=np.uint8
        )

        # Performance statistics
        self.stats_start = time.perf_counter()
        self.stats_frames = 0

        # ====================================================
        # SUBSCRIBERS
        # ====================================================

        self.color_subscription = (
            self.create_subscription(
                Image,
                COLOR_TOPIC,
                self.color_callback,
                SENSOR_QOS
            )
        )

        self.depth_subscription = (
            self.create_subscription(
                Image,
                DEPTH_TOPIC,
                self.depth_callback,
                SENSOR_QOS
            )
        )

        # ====================================================
        # PUBLISHER
        # ====================================================

        self.publisher = self.create_publisher(
            String,
            DETECTION_TOPIC,
            1
        )

        self.get_logger().info(
            f"RGB: {COLOR_TOPIC}"
        )

        self.get_logger().info(
            f"Depth: {DEPTH_TOPIC}"
        )

        self.get_logger().warn(
            "Keep the table EMPTY during calibration."
        )


    # ========================================================
    # DEPTH CALLBACK
    # ========================================================

    def depth_callback(self, msg):

        # Ничего не считаем.
        # Просто сохраняем последнее сообщение.
        self.latest_depth_msg = msg


    # ========================================================
    # IMAGE CONVERSION
    # ========================================================

    def color_to_numpy(self, msg):

        if msg.encoding.lower() not in (
            "rgb8",
            "bgr8"
        ):

            raise RuntimeError(
                f"Unsupported RGB encoding: {msg.encoding}"
            )

        raw = np.frombuffer(
            msg.data,
            dtype=np.uint8
        )

        rows = raw.reshape(
            msg.height,
            msg.step
        )

        frame = rows[
            :,
            :msg.width * 3
        ].reshape(
            msg.height,
            msg.width,
            3
        )

        return frame


    def depth_to_numpy(self, msg):

        encoding = msg.encoding.upper()

        # RealSense default depth
        if encoding in (
            "16UC1",
            "MONO16"
        ):

            raw = np.frombuffer(
                msg.data,
                dtype=np.uint16
            )

            rows = raw.reshape(
                msg.height,
                msg.step // 2
            )

            depth = rows[
                :,
                :msg.width
            ]

            # D435 depth values here are mm
            return depth

        elif encoding == "32FC1":

            raw = np.frombuffer(
                msg.data,
                dtype=np.float32
            )

            rows = raw.reshape(
                msg.height,
                msg.step // 4
            )

            depth_m = rows[
                :,
                :msg.width
            ]

            return depth_m * 1000.0

        raise RuntimeError(
            f"Unsupported depth encoding: {msg.encoding}"
        )


    # ========================================================
    # COLOR CALLBACK
    # ========================================================

    def color_callback(self, msg):

        if self.latest_depth_msg is None:
            return

        # ====================================================
        # DROP FRAMES FIRST
        # ====================================================

        self.frame_counter += 1

        if (
            self.frame_counter
            % PROCESS_EVERY_N_FRAMES
            != 0
        ):
            return

        start = time.perf_counter()

        depth_msg = self.latest_depth_msg

        # ====================================================
        # DIMENSION CHECK
        # ====================================================

        if (
            msg.width != depth_msg.width
            or
            msg.height != depth_msg.height
        ):

            self.get_logger().error(
                "RGB/depth size mismatch: "
                f"RGB {msg.width}x{msg.height}, "
                f"Depth {depth_msg.width}x{depth_msg.height}"
            )

            return

        # ====================================================
        # NUMPY
        # ====================================================

        try:

            frame = self.color_to_numpy(
                msg
            )

            depth = self.depth_to_numpy(
                depth_msg
            )

        except Exception as e:

            self.get_logger().error(
                f"Image conversion error: {e}"
            )

            return

        # ====================================================
        # CALIBRATION
        # ====================================================

        if self.background_depth is None:

            self.calibrate(
                depth
            )

            return

        # ====================================================
        # DETECTION
        # ====================================================

        objects = self.detect_objects(
            frame,
            depth,
            msg.encoding.lower()
        )

        count = len(objects)

        if count != self.last_object_count:

            self.get_logger().info(
                f"Detected {count} object(s)"
            )

            self.last_object_count = count

        # ====================================================
        # PUBLISH
        # ====================================================

        message = String()

        message.data = json.dumps(
            {
                "objects": objects
            },
            ensure_ascii=False
        )

        self.publisher.publish(
            message
        )

        # ====================================================
        # PERFORMANCE
        # ====================================================

        processing_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        self.stats_frames += 1

        now = time.perf_counter()

        dt = now - self.stats_start

        if dt >= 2.0:

            hz = (
                self.stats_frames
                / dt
            )

            self.get_logger().info(
                f"CV: {hz:.1f} Hz | "
                f"processing: {processing_ms:.1f} ms"
            )

            self.stats_frames = 0
            self.stats_start = now


    # ========================================================
    # CALIBRATION
    # ========================================================

    def calibrate(self, depth):

        depth_float = depth.astype(
            np.float32
        )

        depth_float[
            depth_float <= 0
        ] = np.nan

        self.calibration_frames.append(
            depth_float
        )

        count = len(
            self.calibration_frames
        )

        if count == 1:

            self.get_logger().info(
                "Starting table calibration..."
            )

        if count % 5 == 0:

            self.get_logger().info(
                f"Calibration: "
                f"{count}/"
                f"{CALIBRATION_FRAMES}"
            )

        if count < CALIBRATION_FRAMES:
            return

        stack = np.stack(
            self.calibration_frames,
            axis=0
        )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore",
                RuntimeWarning
            )

            self.background_depth = (
                np.nanmedian(
                    stack,
                    axis=0
                ).astype(
                    np.float32
                )
            )

        self.calibration_frames.clear()

        valid = (
            np.isfinite(
                self.background_depth
            ).sum()
            /
            self.background_depth.size
            * 100.0
        )

        self.get_logger().info(
            f"Calibration finished. "
            f"Valid depth: {valid:.1f}%"
        )

        self.get_logger().info(
            "Detector ready."
        )


    # ========================================================
    # DETECTION
    # ========================================================

    def detect_objects(
        self,
        frame,
        depth,
        color_encoding
    ):

        # ====================================================
        # HEIGHT MAP
        # ====================================================

        valid = (
            (depth > 0)
            &
            np.isfinite(
                self.background_depth
            )
        )

        height_map = (
            self.background_depth
            - depth
        )

        object_pixels = (
            valid
            &
            (
                height_map
                > MIN_HEIGHT_MM
            )
            &
            (
                height_map
                < MAX_HEIGHT_MM
            )
        )

        workspace = table_mask(depth.shape)
        if workspace is not None:
            object_pixels &= workspace > 0

        mask = (
            object_pixels.astype(
                np.uint8
            )
            * 255
        )

        # ====================================================
        # CLEAN MASK
        # ====================================================

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MASK_OPEN_SIZE, MASK_OPEN_SIZE)
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel
        )

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MASK_CLOSE_SIZE, MASK_CLOSE_SIZE)
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel
        )


        # ====================================================
        # CONTOURS
        # ====================================================

        contours, _ = (
            cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        )

        detected = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < MIN_AREA:
                continue

            # ================================================
            # CENTER
            # ================================================

            moments = cv2.moments(
                contour
            )

            if moments["m00"] == 0:
                continue

            cx = int(
                moments["m10"]
                /
                moments["m00"]
            )

            cy = int(
                moments["m01"]
                /
                moments["m00"]
            )

            # ================================================
            # ROI
            # ================================================

            x, y, w, h = cv2.boundingRect(
                contour
            )

            roi = frame[
                y:y + h,
                x:x + w
            ]

            height_roi = height_map[
                y:y + h,
                x:x + w
            ]

            local_contour = contour.copy()

            local_contour[:, :, 0] -= x
            local_contour[:, :, 1] -= y

            object_mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            cv2.drawContours(
                object_mask,
                [local_contour],
                -1,
                255,
                -1
            )

            object_mask = cv2.erode(
                object_mask,
                self.kernel5,
                iterations=1
            )

            # ================================================
            # COLOR
            # ================================================

            if color_encoding == "rgb8":

                hsv_roi = cv2.cvtColor(
                    roi,
                    cv2.COLOR_RGB2HSV
                )

            else:

                hsv_roi = cv2.cvtColor(
                    roi,
                    cv2.COLOR_BGR2HSV
                )

            pixels = hsv_roi[
                object_mask == 255
            ]

            if len(pixels) > 0:

                median_hsv = np.median(
                    pixels,
                    axis=0
                )

                color = get_color_name(
                    int(median_hsv[0]),
                    int(median_hsv[1]),
                    int(median_hsv[2])
                )

            else:

                color = "unknown"

            # ================================================
            # SHAPE
            # ================================================

            object_type = classify_shape(
                contour
            )

            # ================================================
            # HEIGHT
            # ================================================

            heights = height_roi[
                object_mask == 255
            ]

            heights = heights[
                np.isfinite(
                    heights
                )
            ]

            if len(heights) > 0:

                height_mm = float(
                    np.median(
                        heights
                    )
                )

            else:

                height_mm = 0.0

            # ================================================
            # SIMPLIFY CONTOUR
            # ================================================

            simple_contour = (
                cv2.approxPolyDP(
                    contour,
                    CONTOUR_EPSILON,
                    True
                )
            )

            # ================================================
            # RESULT
            # ================================================

            detected.append(
                {
                    "type": object_type,

                    "color": color,

                    "position": [
                        cx,
                        cy
                    ],

                    "position_mm": list(
                        pixel_to_workspace_mm(
                            cx,
                            cy,
                            frame.shape[1],
                            frame.shape[0]
                        )
                    ),

                    "height_mm": round(
                        height_mm,
                        1
                    ),

                    "contour": (
                        simple_contour
                        .reshape(-1, 2)
                        .tolist()
                    )
                }
            )

        # ====================================================
        # SORT + IDS
        # ====================================================

        detected.sort(
            key=lambda obj:
            obj["position"][0]
        )

        for index, obj in enumerate(
            detected,
            start=1
        ):

            obj["id"] = index

        return detected


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = CVDetectorNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":
    main()