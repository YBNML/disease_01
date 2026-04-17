"""Run training + benchmark for multiple classification backbones.

Usage:
    python -m classification.compare --config classification/compare_config.yaml

Writes:
    <out_root>/<timestamp>/
      <model_name>/              # each model's training output (as per train.py)
      comparison.csv             # one row per model with metrics + speed
      comparison.md              # human-readable markdown table
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import copy

import torch

from common.config import load_config
from common.utils import set_seed, get_device, make_output_dir
from classification.train import main as train_main
from classification.eval import main as eval_main
from classification.model import build_model
from classification.benchmark import (
    count_parameters, measure_inference_latency, measure_throughput,
)


CSV_COLUMNS = [
    "model",
    "params",
    "accuracy",
    "f1_positive",
    "precision_positive",
    "recall_positive",
    "auc",
    "latency_bs1_ms",
    "throughput_bs_fps",       # populated from the largest configured batch size
]


def _build_model_config(common: dict, model_entry: dict) -> dict:
    """Copy `common` and add model.name/pretrained keys so train.main() can consume it."""
    cfg = copy.deepcopy(common)
    cfg["model"] = {
        "name": model_entry["name"],
        "num_classes": 2,
        "pretrained": model_entry.get("pretrained", True),
    }
    return cfg


def _write_model_config_yaml(cfg: dict, path: Path) -> None:
    import yaml
    path.write_text(yaml.safe_dump(cfg))


def _benchmark_one(model_name: str, ckpt_path: Path, image_size: int,
                   device: str, warmup: int, iters: int,
                   batch_sizes: list) -> dict:
    model = build_model(name=model_name, num_classes=2, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = count_parameters(model)
    results = {"params": params}

    sample_bs1 = torch.randn(batch_sizes[0], 3, image_size, image_size)
    results["latency_bs1_ms"] = measure_inference_latency(
        model, sample_bs1, device=device, warmup=warmup, iters=iters
    ) * 1000.0

    bs_throughput = batch_sizes[-1]
    sample_bsN = torch.randn(bs_throughput, 3, image_size, image_size)
    results["throughput_bs_fps"] = measure_throughput(
        model, sample_bsN, device=device, warmup=warmup, iters=iters
    )
    results["throughput_batch_size"] = bs_throughput
    return results


def _write_markdown(rows: list, path: Path) -> None:
    lines = ["# Classification Backbone Comparison", ""]
    lines.append("| Model | Params | Acc | F1 (canker) | AUC | "
                 "Latency bs=1 (ms) | Throughput (FPS) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['params']:,} | "
            f"{r['accuracy']:.4f} | {r['f1_positive']:.4f} | {r['auc']:.4f} | "
            f"{r['latency_bs1_ms']:.2f} | {r['throughput_bs_fps']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


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
        name = entry["name"]
        print(f"\n===== {name} =====")
        per_model_cfg = _build_model_config(common, entry)
        per_model_cfg["output"]["root"] = str(out_dir / name)

        per_cfg_path = out_dir / f"{name}.yaml"
        _write_model_config_yaml(per_model_cfg, per_cfg_path)

        run_dir = train_main(str(per_cfg_path))
        best_ckpt = Path(run_dir) / "ckpt" / "best.pt"
        eval_result = eval_main(str(per_cfg_path), str(best_ckpt))

        bench_result = _benchmark_one(
            name, best_ckpt,
            image_size=common["data"]["image_size"],
            device=device,
            warmup=bench["warmup"],
            iters=bench["iters"],
            batch_sizes=bench["batch_sizes"],
        )

        rows.append({
            "model": name,
            "params": bench_result["params"],
            "accuracy": eval_result["accuracy"],
            "f1_positive": eval_result["f1_positive"],
            "precision_positive": eval_result["precision_positive"],
            "recall_positive": eval_result["recall_positive"],
            "auc": eval_result["auc"],
            "latency_bs1_ms": bench_result["latency_bs1_ms"],
            "throughput_bs_fps": bench_result["throughput_bs_fps"],
        })

    # write CSV
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})

    # write markdown
    _write_markdown(rows, out_dir / "comparison.md")

    # also dump raw rows JSON for traceability
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2))

    print(f"\nComparison written to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
