"""Training entrypoint for 3-class citrus semantic segmentation.

Usage:
    python -m segmentation.train --config segmentation/config.yaml
    python -m segmentation.train --config segmentation/config.yaml --override train.lr=0.0005
"""
from __future__ import annotations
import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.config import load_config, apply_overrides
from common.dataset import SegmentationDataset
from common.utils import set_seed, get_device, make_output_dir
from segmentation.transforms import build_transforms
from segmentation.model import build_model
from segmentation.losses import CombinedLoss
from segmentation.metrics import SegmentationMetrics


def _build_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("segmentation.train")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _collate(batch: list) -> dict:
    imgs = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"].long() for b in batch])
    paths = [b["image_path"] for b in batch]
    return {"image": imgs, "mask": masks, "image_path": paths}


def main(config_path: str, overrides: list | None = None) -> Path:
    cfg = apply_overrides(load_config(config_path), overrides or [])
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    out_dir = make_output_dir(cfg["output"]["root"], task="run")
    (out_dir / "ckpt").mkdir()
    (out_dir / "tb").mkdir()
    shutil.copy(config_path, out_dir / "config.yaml")

    logger = _build_logger(out_dir)
    logger.info(f"config: {cfg}")
    logger.info(f"device: {device}")
    logger.info(f"out_dir: {out_dir}")

    db = Path(cfg["data"]["database_root"])
    train_tf = build_transforms(cfg["data"]["image_size"], train=True)
    val_tf = build_transforms(cfg["data"]["image_size"], train=False)
    train_ds = SegmentationDataset(db, split="train", transform=train_tf)
    val_ds = SegmentationDataset(db, split="val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"],
                              shuffle=True,
                              num_workers=cfg["data"]["num_workers"],
                              collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["data"]["num_workers"],
                            collate_fn=_collate)

    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )
    loss_fn = CombinedLoss(
        num_classes=cfg["model"]["num_classes"],
        ce_weight=cfg["train"]["ce_weight"],
        dice_weight=cfg["train"]["dice_weight"],
    ).to(device)

    tb = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_miou = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tr_loss_sum = 0.0
        n_seen = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
        tr_loss = tr_loss_sum / max(n_seen, 1)
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        n_seen = 0
        metrics = SegmentationMetrics(num_classes=cfg["model"]["num_classes"])
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                val_loss_sum += loss.item() * imgs.size(0)
                n_seen += imgs.size(0)
                metrics.update(logits, masks)
        val_loss = val_loss_sum / max(n_seen, 1)
        m = metrics.compute()

        logger.info(
            f"epoch {epoch+1}/{cfg['train']['epochs']} "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"miou={m['miou']:.4f} pixAcc={m['pixel_accuracy']:.4f} "
            f"iou_classes={['{:.3f}'.format(x) for x in m['iou_per_class']]}"
        )
        tb.add_scalar("train/loss", tr_loss, epoch)
        tb.add_scalar("val/loss", val_loss, epoch)
        tb.add_scalar("val/miou", m["miou"], epoch)
        tb.add_scalar("val/pixel_accuracy", m["pixel_accuracy"], epoch)
        for i, iou in enumerate(m["iou_per_class"]):
            tb.add_scalar(f"val/iou_class_{i}", iou, epoch)
        tb.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": m,
            "config": cfg,
        }, out_dir / "ckpt" / "last.pt")
        if m["miou"] > best_miou:
            best_miou = m["miou"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": m,
                "config": cfg,
            }, out_dir / "ckpt" / "best.pt")
            logger.info(f"  new best miou={best_miou:.4f}")

    tb.close()
    logger.info("training complete")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.override)
