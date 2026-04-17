"""Convert AI Hub polygon to YOLO-format bounding box."""
import numpy as np

from common.label_parser import polygon_to_bbox


def polygon_to_yolo_bbox(polygon: np.ndarray, img_w: int, img_h: int) -> tuple:
    """Return (x_center, y_center, w, h) normalized to [0, 1] in YOLO format.

    Raises ValueError if the polygon has zero width or height in either axis."""
    x_min, y_min, x_max, y_max = polygon_to_bbox(polygon)
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w <= 0 or box_h <= 0:
        raise ValueError(
            f"degenerate polygon: bbox {x_min, y_min, x_max, y_max}"
        )
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    w = box_w / img_w
    h = box_h / img_h
    return (float(x_center), float(y_center), float(w), float(h))
