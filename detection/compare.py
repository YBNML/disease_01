"""Run YOLOv8 variants comparison (n/s/m) for citrus detection.

Usage:
    python -m detection.compare --config detection/compare_config.yaml
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
from detection.train import build_ultralytics_kwargs


CSV_COLUMNS = [
    "model",
    "params",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "latency_bs1_ms",
    "throughput_bs_fps",
]


def _build_cfg_for_model(common: dict, model_entry: dict) -> dict:
    """Build a flat config dict (matching detection/config.yaml shape) for one model."""
    cfg = copy.deepcopy(common)
    cfg["model"] = {**common["model"], "name": model_entry["name"]}
    cfg["output"] = {**common["output"], "name": model_entry["label"]}
    return cfg


def _write_cfg(cfg: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg))


def _sync(device: str) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _benchmark(model, imgsz: int, device: str, warmup: int, iters: int, bs: int) -> float:
    """Return seconds per forward at given batch size (raw net, no letterbox)."""
    import torch
    net = model.model.to(device).eval()
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


def main(config_path: str) -> Path:
    cfg = load_config(config_path)
    common = cfg["common"]
    bench = cfg["benchmark"]

    set_seed(common["seed"])
    out_root = Path(common["output"]["project"])
    out_dir = make_output_dir(out_root, task="compare")

    from ultralytics import YOLO
    from detection.train import main as train_main
    from detection.eval import main as eval_main

    rows = []
    device = get_device(common["train"]["device"]).type

    for entry in cfg["models"]:
        label = entry["label"]
        print(f"\n===== {label} =====")

        # Build a flat per-model config (matching detection/config.yaml shape).
        # output.project points at out_dir so each model lands in out_dir/<label>/.
        per_cfg = _build_cfg_for_model(common, entry)
        per_cfg["output"]["project"] = str(out_dir.resolve())
        per_cfg["output"]["name"] = label

        per_cfg_path = out_dir / f"{label}.yaml"
        _write_cfg(per_cfg, per_cfg_path)

        # 1) Train
        train_main(str(per_cfg_path))

        # Ultralytics puts output under <project>/<name>/. Find best.pt.
        run_dir = out_dir / label
        best_ckpt = run_dir / "weights" / "best.pt"

        # 2) Eval — eval.main writes to a separate subdir to avoid clobbering train run
        eval_cfg = copy.deepcopy(per_cfg)
        eval_cfg["output"]["name"] = f"{label}_eval"
        eval_cfg_path = out_dir / f"{label}_eval.yaml"
        _write_cfg(eval_cfg, eval_cfg_path)
        val_results = eval_main(str(eval_cfg_path), str(best_ckpt))

        # Extract metrics (ultralytics Results.box has map50, map, mp, mr)
        map50 = float(val_results.box.map50)
        map50_95 = float(val_results.box.map)
        mp = float(val_results.box.mp)
        mr = float(val_results.box.mr)

        # 3) Benchmark
        model = YOLO(str(best_ckpt))
        params = sum(p.numel() for p in model.model.parameters())
        lat_bs1 = _benchmark(model, bench["imgsz"], device,
                             bench["warmup"], bench["iters"], bs=1) * 1000.0
        thr_lat = _benchmark(model, bench["imgsz"], device,
                             bench["warmup"], bench["iters"],
                             bs=bench["batch_sizes"][-1])
        thr_fps = bench["batch_sizes"][-1] / thr_lat

        rows.append({
            "model": label,
            "params": params,
            "map50": map50,
            "map50_95": map50_95,
            "precision": mp,
            "recall": mr,
            "latency_bs1_ms": lat_bs1,
            "throughput_bs_fps": thr_fps,
        })

    # Write CSV
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})

    # Markdown
    md_lines = [
        "# YOLOv8 Variants Comparison (Detection)",
        "",
        "| Model | Params | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Latency bs=1 (ms) | Throughput (FPS) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['params']:,} | "
            f"{r['map50']:.4f} | {r['map50_95']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['latency_bs1_ms']:.2f} | {r['throughput_bs_fps']:.1f} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(md_lines) + "\n")

    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    print(f"\nComparison written to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
