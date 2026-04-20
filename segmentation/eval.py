"""Evaluation entrypoint for semantic segmentation checkpoints.

Writes metrics.json and a few qualitative composite images
(original | GT mask | predicted mask) to <run_dir>/qualitative/.

Usage:
    python -m segmentation.eval --config segmentation/config.yaml \
        --ckpt outputs/segmentation/run/<timestamp>/ckpt/best.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.config import load_config
from common.dataset import SegmentationDataset
from common.utils import set_seed, get_device
from segmentation.transforms import build_transforms
from segmentation.model import build_model
from segmentation.metrics import SegmentationMetrics
from segmentation.train import _collate


CLASS_COLORS = np.array([
    [0, 0, 0],       # 0 = bg (black)
    [0, 255, 0],     # 1 = normal (green)
    [255, 0, 0],     # 2 = canker (red)
], dtype=np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def _denormalize(img: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization, return uint8 HxWx3 RGB."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = img.detach().cpu() * std + mean
    x = (x.clamp(0, 1) * 255).to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


def _save_qualitative(imgs: torch.Tensor, gts: torch.Tensor, preds: torch.Tensor,
                      out_dir: Path, max_samples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(imgs.size(0), max_samples)
    for i in range(n):
        rgb = _denormalize(imgs[i])
        gt_rgb = _mask_to_rgb(gts[i].detach().cpu().numpy())
        pred_rgb = _mask_to_rgb(preds[i].detach().cpu().numpy())
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, arr, title in zip(axes, (rgb, gt_rgb, pred_rgb),
                                  ("original", "GT", "pred")):
            ax.imshow(arr)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:03d}.png", dpi=100)
        plt.close(fig)


def main(config_path: str, ckpt_path: str,
         num_qualitative_samples: int | None = None) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        architecture=cfg["model"].get("architecture", "Unet"),
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_tf = build_transforms(cfg["data"]["image_size"], train=False)
    val_ds = SegmentationDataset(Path(cfg["data"]["database_root"]),
                                 split="val", transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["data"]["num_workers"],
                            collate_fn=_collate)

    metrics = SegmentationMetrics(num_classes=cfg["model"]["num_classes"])
    out_dir = Path(ckpt_path).parent.parent
    qdir = out_dir / "qualitative"

    q_budget = num_qualitative_samples if num_qualitative_samples is not None \
        else cfg["eval"].get("num_qualitative_samples", 0)
    q_saved = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            metrics.update(logits, masks)
            if q_saved < q_budget:
                remaining = q_budget - q_saved
                _save_qualitative(imgs, masks, preds, qdir, max_samples=remaining)
                q_saved += min(imgs.size(0), remaining)

    result = metrics.compute()
    json_result = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in result.items()}
    (out_dir / "metrics.json").write_text(json.dumps(json_result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--samples", type=int, default=None,
                        help="override num qualitative samples to save")
    args = parser.parse_args()
    r = main(args.config, args.ckpt, num_qualitative_samples=args.samples)
    printable = {k: v for k, v in r.items() if k != "confusion_matrix"}
    print(json.dumps(printable, indent=2))
