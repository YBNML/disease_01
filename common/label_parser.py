import numpy as np
import cv2


def parse_antn_pt(pt_str: str) -> np.ndarray:
    """Parse AI Hub ANTN_PT string '[x1|x2|...],[y1|y2|...]' to (N, 2) int array."""
    xs_str, ys_str = pt_str.split("],[")
    xs = [int(v) for v in xs_str.strip("[]").split("|")]
    ys = [int(v) for v in ys_str.strip("[]").split("|")]
    if len(xs) != len(ys):
        raise ValueError(f"x/y length mismatch: {len(xs)} vs {len(ys)}")
    return np.array(list(zip(xs, ys)), dtype=np.int32)


def polygon_to_bbox(polygon: np.ndarray) -> tuple:
    """Return (x_min, y_min, x_max, y_max) from (N, 2) polygon."""
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def polygon_to_mask(polygon: np.ndarray, h: int, w: int) -> np.ndarray:
    """Rasterize polygon into a (H, W) uint8 binary mask (1 inside, 0 outside)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask
