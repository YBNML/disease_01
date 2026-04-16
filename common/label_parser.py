import numpy as np


def parse_antn_pt(pt_str: str) -> np.ndarray:
    """Parse AI Hub ANTN_PT string '[x1|x2|...],[y1|y2|...]' to (N, 2) int array."""
    xs_str, ys_str = pt_str.split("],[")
    xs = [int(v) for v in xs_str.strip("[]").split("|")]
    ys = [int(v) for v in ys_str.strip("[]").split("|")]
    if len(xs) != len(ys):
        raise ValueError(f"x/y length mismatch: {len(xs)} vs {len(ys)}")
    return np.array(list(zip(xs, ys)), dtype=np.int32)
