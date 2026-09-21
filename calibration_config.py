"""Manual measurements for the flat tabletop calibration."""

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
