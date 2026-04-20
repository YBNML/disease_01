"""Quantize the best classifier (ViT-Small fp32) to INT8 (dynamic or post-training)
and compare accuracy, size, latency vs fp32 baseline.

Usage:
    python scripts/quantize_and_benchmark.py \
        --ckpt outputs/classification_compare/compare/2026-04-19_01-13-33/vit_small_patch16_224/run/*/ckpt/best.pt \
        --config classification/config.yaml \
        --out docs/results/2026-04-20-quantization.md

Produces a markdown report with side-by-side comparison.
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── import project modules ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load_config
from common.dataset import ClassificationDataset
from classification.model import build_model
from classification.transforms import build_transforms
from classification.metrics import ClassificationMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collate(batch: list) -> dict:
    imgs = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"image": imgs, "label": labels}


def _resolve_ckpt(pattern: str) -> Path:
    """Expand glob wildcard in --ckpt argument and return a single Path."""
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found matching: {pattern!r}")
    if len(matches) > 1:
        print(f"[warn] multiple checkpoints found, using first: {matches[0]}")
    return Path(matches[0])


def _load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = build_model(name="vit_small_patch16_224", num_classes=2, pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    key = "model" if "model" in state else "state_dict" if "state_dict" in state else None
    sd = state[key] if key else state
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def _eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Return accuracy and F1 (canker = class 1)."""
    metrics = ClassificationMetrics(num_classes=2, positive_class=1)
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(imgs)
            metrics.update(logits, labels)
    return metrics.compute()


def _measure_latency_bs1(model: nn.Module, device: torch.device,
                          image_size: int = 224,
                          n_warmup: int = 10, n_iters: int = 50) -> float:
    """Return mean latency in ms for batch-size-1 inference on CPU."""
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(sum(times) / len(times))


def _measure_throughput_bs32(model: nn.Module, device: torch.device,
                              image_size: int = 224,
                              n_warmup: int = 5, n_iters: int = 20) -> float:
    """Return throughput in images/sec for batch-size-32 on CPU."""
    dummy = torch.randn(32, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy)
    elapsed = time.perf_counter() - t0
    return float(32 * n_iters / elapsed)


def _model_size_mb(model: nn.Module) -> float:
    """Serialize model to a temp file and report size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        torch.save(model.state_dict(), f.name)
        return Path(f.name).stat().st_size / (1024 ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize ViT-Small and benchmark")
    parser.add_argument("--ckpt", required=True,
                        help="Path (or glob pattern) to best.pt checkpoint")
    parser.add_argument("--config", default="classification/config.yaml",
                        help="Classification config YAML for val-set settings")
    parser.add_argument("--out", default="docs/results/quantization.md",
                        help="Output markdown report path")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    # Always use CPU for quantized-op stability
    device = torch.device("cpu")
    print(f"[info] device: {device}")

    # ── Resolve checkpoint ──────────────────────────────────────────────────
    ckpt_path = _resolve_ckpt(args.ckpt)
    print(f"[info] checkpoint: {ckpt_path}")

    # ── Val dataloader ──────────────────────────────────────────────────────
    cfg = load_config(args.config)
    db = Path(cfg["data"]["database_root"])
    val_tf = build_transforms(args.image_size, train=False)
    val_ds = ClassificationDataset(db, split="val", transform=val_tf)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    print(f"[info] val set: {len(val_ds)} samples")

    # ── FP32 evaluation ─────────────────────────────────────────────────────
    print("[info] evaluating fp32 model …")
    fp32_model = _load_model(ckpt_path, device)
    fp32_metrics = _eval_accuracy(fp32_model, val_loader, device)
    fp32_size = _model_size_mb(fp32_model)
    fp32_latency = _measure_latency_bs1(fp32_model, device, args.image_size)
    fp32_throughput = _measure_throughput_bs32(fp32_model, device, args.image_size)
    print(f"  accuracy={fp32_metrics['accuracy']:.4f}  "
          f"f1_canker={fp32_metrics['f1_positive']:.4f}  "
          f"size={fp32_size:.1f}MB  lat={fp32_latency:.1f}ms  "
          f"tput={fp32_throughput:.1f}fps")

    # ── INT8 dynamic quantization ────────────────────────────────────────────
    print("[info] applying torch.ao dynamic INT8 quantization (Linear layers) …")
    # qnnpack is required on Apple Silicon (ARM64); fbgemm is not available.
    # Fall back gracefully if qnnpack is also absent.
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = torch.backends.quantized.supported_engines[0]
    int8_model = torch.ao.quantization.quantize_dynamic(
        fp32_model, {nn.Linear}, dtype=torch.qint8
    )
    int8_model.eval()

    print("[info] evaluating int8 model …")
    int8_metrics = _eval_accuracy(int8_model, val_loader, device)
    int8_size = _model_size_mb(int8_model)
    int8_latency = _measure_latency_bs1(int8_model, device, args.image_size)
    int8_throughput = _measure_throughput_bs32(int8_model, device, args.image_size)
    print(f"  accuracy={int8_metrics['accuracy']:.4f}  "
          f"f1_canker={int8_metrics['f1_positive']:.4f}  "
          f"size={int8_size:.1f}MB  lat={int8_latency:.1f}ms  "
          f"tput={int8_throughput:.1f}fps")

    # ── Compose report ───────────────────────────────────────────────────────
    size_reduction = (1 - int8_size / fp32_size) * 100
    latency_speedup = fp32_latency / int8_latency if int8_latency > 0 else float("nan")
    acc_drop = (fp32_metrics["accuracy"] - int8_metrics["accuracy"]) * 100

    report_md = f"""# Quantization Study — ViT-Small/16

**Checkpoint**: `{ckpt_path}`
**Val samples**: {len(val_ds)}
**Device**: CPU (required for quantized ops)

| Variant | Size (MB) | Accuracy | F1 (canker) | Latency bs=1 (ms) | Throughput bs=32 (FPS) |
|---|---:|---:|---:|---:|---:|
| fp32 (baseline) | {fp32_size:.1f} | {fp32_metrics['accuracy']:.4f} | {fp32_metrics['f1_positive']:.4f} | {fp32_latency:.1f} | {fp32_throughput:.1f} |
| int8 (dynamic) | {int8_size:.1f} | {int8_metrics['accuracy']:.4f} | {int8_metrics['f1_positive']:.4f} | {int8_latency:.1f} | {int8_throughput:.1f} |

## Summary

- **Size reduction**: {size_reduction:.1f}%
- **Latency speedup** (bs=1): {latency_speedup:.2f}×
- **Accuracy drop**: {acc_drop:.2f} pp
"""

    raw_data = {
        "checkpoint": str(ckpt_path),
        "val_samples": len(val_ds),
        "fp32": {
            "size_mb": round(fp32_size, 3),
            "accuracy": round(fp32_metrics["accuracy"], 6),
            "f1_canker": round(fp32_metrics["f1_positive"], 6),
            "latency_bs1_ms": round(fp32_latency, 3),
            "throughput_bs32_fps": round(fp32_throughput, 2),
        },
        "int8_dynamic": {
            "size_mb": round(int8_size, 3),
            "accuracy": round(int8_metrics["accuracy"], 6),
            "f1_canker": round(int8_metrics["f1_positive"], 6),
            "latency_bs1_ms": round(int8_latency, 3),
            "throughput_bs32_fps": round(int8_throughput, 2),
        },
        "delta": {
            "size_reduction_pct": round(size_reduction, 2),
            "latency_speedup": round(latency_speedup, 3),
            "accuracy_drop_pp": round(acc_drop, 4),
        },
    }

    # ── Write outputs ────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"\n[info] report written → {out_path}")
    print(f"[info] json written  → {json_path}")
    print("\n" + "=" * 60)
    print(report_md)


if __name__ == "__main__":
    main()
