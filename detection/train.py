"""Training entrypoint for YOLOv8 citrus detection.

Usage:
    python -m detection.train --config detection/config.yaml
    python -m detection.train --config detection/config.yaml --override train.epochs=5
"""
from __future__ import annotations
import argparse
from pathlib import Path

from common.config import load_config, apply_overrides
from common.utils import set_seed, get_device


def _resolve_device(device_cfg: str) -> str:
    """ultralytics accepts 'mps' | 'cpu' | '0' (cuda). 'auto' → mps or cpu."""
    if device_cfg != "auto":
        return device_cfg
    return "mps" if get_device("auto").type == "mps" else "cpu"


def build_ultralytics_kwargs(cfg: dict) -> dict:
    """Flatten our config into keyword args accepted by YOLO(...).train(...)."""
    device = _resolve_device(cfg["train"]["device"])
    return {
        "data": cfg["data"]["data_yaml"],
        "epochs": cfg["train"]["epochs"],
        "batch": cfg["train"]["batch"],
        "imgsz": cfg["model"]["imgsz"],
        "lr0": cfg["train"]["lr0"],
        "workers": cfg["train"]["workers"],
        "patience": cfg["train"]["patience"],
        "device": device,
        "project": cfg["output"]["project"],
        "name": cfg["output"]["name"],
        "seed": cfg["seed"],
        "exist_ok": True,  # allow re-runs into same name (ultralytics will increment)
    }


def main(config_path: str, overrides: list | None = None):
    cfg = apply_overrides(load_config(config_path), overrides or [])
    set_seed(cfg["seed"])

    # Import ultralytics lazily so `import detection.train` doesn't pull the
    # whole ultralytics stack (keeps test startup fast).
    from ultralytics import YOLO

    kwargs = build_ultralytics_kwargs(cfg)
    model = YOLO(cfg["model"]["name"])
    results = model.train(**kwargs)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.override)
