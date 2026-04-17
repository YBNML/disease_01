"""Evaluation entrypoint: load a checkpoint, compute metrics on val split,
save confusion_matrix.png and metrics.json alongside the checkpoint.

Usage:
    python classification/eval.py --config classification/config.yaml \
        --ckpt outputs/classification/<run>/ckpt/best.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.config import load_config
from common.dataset import ClassificationDataset
from common.utils import set_seed, get_device
from classification.transforms import build_transforms
from classification.model import build_model
from classification.metrics import ClassificationMetrics
from classification.train import _collate


def _save_confusion_matrix(cm: np.ndarray, accuracy: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["normal", "canker"],
                yticklabels=["normal", "canker"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (acc={accuracy:.2%})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(config_path: str, ckpt_path: str) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg["model"]["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_tf = build_transforms(cfg["data"]["image_size"], train=False)
    val_ds = ClassificationDataset(Path(cfg["data"]["database_root"]),
                                   split="val", transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"],
                            collate_fn=_collate)

    metrics = ClassificationMetrics(num_classes=cfg["model"]["num_classes"])
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(imgs)
            metrics.update(logits, labels)
    result = metrics.compute()

    out_dir = Path(ckpt_path).parent.parent  # ckpt/ lives inside run dir
    cm = np.asarray(result["confusion_matrix"])
    _save_confusion_matrix(cm, result["accuracy"], out_dir / "confusion_matrix.png")

    # JSON-safe version (convert ndarray to list)
    json_result = {**result, "confusion_matrix": cm.tolist()}
    (out_dir / "metrics.json").write_text(json.dumps(json_result, indent=2))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    r = main(args.config, args.ckpt)
    print(json.dumps({k: v for k, v in r.items() if k != "confusion_matrix"}, indent=2))
