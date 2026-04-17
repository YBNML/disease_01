"""Evaluation wrapper for a YOLOv8 detection checkpoint.

Usage:
    python -m detection.eval --config detection/config.yaml \
        --ckpt outputs/detection/run/weights/best.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path

from common.config import load_config
from detection.train import _resolve_device


def build_val_kwargs(cfg: dict) -> dict:
    device = _resolve_device(cfg["train"]["device"])
    return {
        "data": cfg["data"]["data_yaml"],
        "imgsz": cfg["model"]["imgsz"],
        "batch": cfg["train"]["batch"],
        "workers": cfg["train"]["workers"],
        "device": device,
        "project": str(Path(cfg["output"]["project"]).resolve()),
        "name": cfg["output"]["name"] + "_eval",
        "exist_ok": True,
    }


def main(config_path: str, ckpt_path: str):
    cfg = load_config(config_path)

    from ultralytics import YOLO
    model = YOLO(ckpt_path)
    kwargs = build_val_kwargs(cfg)
    results = model.val(**kwargs)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    main(args.config, args.ckpt)
