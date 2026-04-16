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


import json
from pathlib import Path


def _parse_resolution(rsoltn: str) -> tuple:
    """'(1920,1080)' -> (1920, 1080). Returns (0, 0) on parse failure."""
    try:
        cleaned = rsoltn.strip().strip("()")
        w, h = cleaned.split(",")
        return (int(w), int(h))
    except (ValueError, AttributeError):
        return (0, 0)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_sample(json_path) -> dict:
    """Load AI Hub JSON label and return a normalized dict."""
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    info = d.get("Info", {}) or {}
    ann = d.get("Annotations", {}) or {}
    env = d.get("Environment", {}) or {}

    pt_str = ann.get("ANTN_PT")
    has_polygon = pt_str is not None
    polygon = parse_antn_pt(pt_str) if has_polygon else None

    return {
        "json_path": json_path,
        "image_file_name": info.get("IMAGE_FILE_NM", json_path.stem),
        "class_code": ann.get("OBJECT_CLASS_CODE"),
        "has_polygon": has_polygon,
        "polygon": polygon,
        "image_size": _parse_resolution(info.get("RSOLTN", "")),
        "metadata": {
            "camera": info.get("CMRA_INFO"),
            "location": info.get("LCINFO"),
            "place_type": info.get("IMAGE_OBTAIN_PLACE_TY"),
            "growth_stage": info.get("GRWH_STEP_CODE"),
            "date": info.get("OCPRD"),
            "env": {
                "solar": _safe_float(env.get("SOLRAD_QY")),
                "rain": _safe_float(env.get("AFR")),
                "temp": _safe_float(env.get("TP")),
                "humidity": _safe_float(env.get("HD")),
                "soil_moisture": _safe_float(env.get("SOIL_MITR")),
            },
        },
    }
