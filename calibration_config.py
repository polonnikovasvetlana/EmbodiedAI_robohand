"""Camera/table calibration measurements.

Table marker coordinates are measured in millimetres from the projection of
the manipulator origin onto the tabletop. Marker 10 is on the manipulator,
so its image center is not itself a point on the tabletop plane.
"""

from pathlib import Path

# ArUco dictionary and IDs used by the calibration board.
ARUCO_DICTIONARY = "DICT_4X4_50"
TABLE_MARKER_IDS = (0, 1, 2, 3)
MANIPULATOR_MARKER_ID = 10
MANIPULATOR_MARKER_HEIGHT_MM = 130.0

# Printed marker side lengths. These are used for validation and pose scale.
TABLE_MARKER_SIZE_MM = 70.0
MANIPULATOR_MARKER_SIZE_MM = 25.0

# Table marker centers in the table/robot coordinate system, in millimetres.
# Measure from the tabletop point directly below the manipulator origin to
# each table marker centre.
# +X points toward the bottom of the image/table; +Y points to the left.
# Replace the example values with ruler measurements before using ArUco pose.
TABLE_MARKER_POSITIONS_MM = {
    0: (0.0, -210.0, 0.0),
    1: (280.0, -210.0, 0.0),
    2: (0.0, 210.0, 0.0),
    3: (360.0, 210.0, 0.0),
}

# Offset from the centre of marker 10 to the robot TCP/gripper point.
# Measure this once if the marker centre is not exactly the TCP.
MANIPULATOR_MARKER_TO_TCP_MM = (0.0, 0.0, 0.0)

TABLE_HOMOGRAPHY_FILE = Path(__file__).resolve().parent / "table_homography.json"


def validate_aruco_measurements():
    missing = [
        marker_id
        for marker_id, position in TABLE_MARKER_POSITIONS_MM.items()
        if any(value is None for value in position)
    ]
    if missing:
        raise ValueError(
            "Set measured positions for table markers: "
            + ", ".join(str(marker_id) for marker_id in missing)
        )

# Camera image size used by the RealSense launch script.
IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480

# Visible tabletop dimensions measured with a ruler.
WORKSPACE_WIDTH_CM = 63.0
WORKSPACE_HEIGHT_CM = 45.0

# Origin position measured from the tabletop edges.
# The positive X axis points down; the positive Y axis points left.
ORIGIN_FROM_RIGHT_CM = 34.0
ORIGIN_FROM_TOP_CM = 5.0


def origin_pixel(image_width=IMAGE_WIDTH_PX, image_height=IMAGE_HEIGHT_PX):
    origin_x = (
        WORKSPACE_WIDTH_CM - ORIGIN_FROM_RIGHT_CM
    ) * image_width / WORKSPACE_WIDTH_CM
    origin_y = (
        ORIGIN_FROM_TOP_CM
        * image_height
        / WORKSPACE_HEIGHT_CM
    )
    return round(origin_x), round(origin_y)
