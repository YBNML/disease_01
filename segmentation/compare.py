"""Run segmentation architecture comparison.

Usage:
    python -m segmentation.compare --config segmentation/compare_config.yaml
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from common.config import load_config
from common.utils import set_seed, get_device, make_output_dir
from segmentation.train import main as train_main
from segmentation.eval import main as eval_main
from segmentation.model import build_model
from classification.benchmark import count_parameters


CSV_COLUMNS = [
    "label", "architecture", "encoder",
    "params",
    "miou", "pixel_accuracy",
    "iou_bg", "iou_normal", "iou_canker",
    "dice_normal", "dice_canker",
    "latency_bs1_ms", "throughput_bs_fps",
]


def _sync(device: str) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _benchmark(model, imgsz: int, device: str, warmup: int, iters: int, bs: int) -> float:
    """Seconds per forward at batch size bs."""
    net = model.to(device).eval()
    x = torch.randn(bs, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            net(x)
        _sync(device)
    return (time.perf_counter() - t0) / iters


def _build_cfg(common: dict, entry: dict) -> dict:
    cfg = copy.deepcopy(common)
    cfg["model"] = {
        "num_classes": 3,
        "architecture": entry["architecture"],
        "encoder_name": entry["encoder_name"],
        "encoder_weights": "imagenet",
    }
    return cfg


def main(config_path: str) -> Path:
    cfg = load_config(config_path)
    common = cfg["common"]
    bench = cfg["benchmark"]

    set_seed(common["seed"])
    device = get_device(common["device"]).type

    out_root = Path(common["output"]["root"])
    out_dir = make_output_dir(out_root, task="compare")

    rows = []
    for entry in cfg["models"]:
        label = entry["label"]
        print(f"\n===== {label} =====")
        per_cfg = _build_cfg(common, entry)
        # Route output to this model's subdir inside the compare root
        per_cfg["output"]["root"] = str(out_dir / label)

        per_cfg_path = out_dir / f"{label}.yaml"
        per_cfg_path.write_text(yaml.safe_dump(per_cfg))

        # 1) Train
        run_dir = Path(train_main(str(per_cfg_path)))
        best_ckpt = run_dir / "ckpt" / "best.pt"

        # 2) Eval
        eval_result = eval_main(str(per_cfg_path), str(best_ckpt),
                                num_qualitative_samples=0)

        # 3) Benchmark — build fresh model, load weights
        model = build_model(
            num_classes=3,
            architecture=entry["architecture"],
            encoder_name=entry["encoder_name"],
            encoder_weights=None,
        )
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()

        params = count_parameters(model)
        lat_bs1 = _benchmark(model, bench["imgsz"], device,
                             bench["warmup"], bench["iters"], bs=1) * 1000.0
        bs_thr = bench["batch_sizes"][-1]
        thr_lat = _benchmark(model, bench["imgsz"], device,
                             bench["warmup"], bench["iters"], bs=bs_thr)
        thr_fps = bs_thr / thr_lat

        iou = eval_result["iou_per_class"]
        dice = eval_result["dice_per_class"]

        rows.append({
            "label": label,
            "architecture": entry["architecture"],
            "encoder": entry["encoder_name"],
            "params": params,
            "miou": eval_result["miou"],
            "pixel_accuracy": eval_result["pixel_accuracy"],
            "iou_bg": iou[0],
            "iou_normal": iou[1],
            "iou_canker": iou[2],
            "dice_normal": dice[1],
            "dice_canker": dice[2],
            "latency_bs1_ms": lat_bs1,
            "throughput_bs_fps": thr_fps,
        })

    # CSV
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})

    # Markdown
    md = ["# Segmentation Architecture Comparison", "",
          "| Label | Arch | Encoder | Params | mIoU | pixAcc | IoU bg | IoU normal | IoU canker | Dice normal | Dice canker | Latency bs=1 (ms) | Throughput (FPS) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['label']} | {r['architecture']} | {r['encoder']} | "
            f"{r['params']:,} | {r['miou']:.4f} | {r['pixel_accuracy']:.4f} | "
            f"{r['iou_bg']:.4f} | {r['iou_normal']:.4f} | {r['iou_canker']:.4f} | "
            f"{r['dice_normal']:.4f} | {r['dice_canker']:.4f} | "
            f"{r['latency_bs1_ms']:.2f} | {r['throughput_bs_fps']:.1f} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n")

    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    print(f"\nComparison written to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
