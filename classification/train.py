"""Training entrypoint for citrus disease classification.

Usage:
    python classification/train.py --config classification/config.yaml
    python classification/train.py --config classification/config.yaml --override train.lr=0.0005

The script creates outputs/<task>/<timestamp>/ containing:
- ckpt/last.pt, ckpt/best.pt
- tb/ (tensorboard logs)
- train.log
- config.yaml (snapshot)
"""
from __future__ import annotations
import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.config import load_config, apply_overrides
from common.dataset import ClassificationDataset
from common.utils import set_seed, get_device, make_output_dir
from classification.transforms import build_transforms
from classification.model import build_model
from classification.sampler import build_weighted_sampler
from classification.metrics import ClassificationMetrics


def _build_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("classification.train")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers on re-entry (e.g. tests)
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


def _build_optimizer(model, cfg):
    name = cfg["train"]["optimizer"].lower()
    if name != "adamw":
        raise ValueError(f"unsupported optimizer: {name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )


def _build_scheduler(optimizer, cfg, num_epochs):
    name = cfg["train"]["scheduler"].lower()
    if name != "cosine":
        raise ValueError(f"unsupported scheduler: {name}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


def _labels_from_dataset(ds: ClassificationDataset) -> list:
    """Extract all labels from a ClassificationDataset without reading images.
    Uses label_parser directly on the cached json paths."""
    from common.label_parser import load_sample
    from common.dataset import CLASS_NAME_TO_LABEL
    return [CLASS_NAME_TO_LABEL[load_sample(jp)["class_code"]]
            for _, jp in ds.items]


def main(config_path: str, overrides: list | None = None) -> Path:
    cfg = apply_overrides(load_config(config_path), overrides or [])

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    # outputs/<task>/<timestamp>/
    out_dir = make_output_dir(cfg["output"]["root"], task="run")
    (out_dir / "ckpt").mkdir()
    (out_dir / "tb").mkdir()
    shutil.copy(config_path, out_dir / "config.yaml")

    logger = _build_logger(out_dir)
    logger.info(f"config: {cfg}")
    logger.info(f"device: {device}")
    logger.info(f"out_dir: {out_dir}")

    # datasets
    db = Path(cfg["data"]["database_root"])
    train_tf = build_transforms(cfg["data"]["image_size"], train=True)
    val_tf = build_transforms(cfg["data"]["image_size"], train=False)
    train_ds = ClassificationDataset(db, split="train", transform=train_tf)
    val_ds = ClassificationDataset(db, split="val", transform=val_tf)

    # sampler for imbalance
    if cfg["train"]["use_weighted_sampler"]:
        labels = _labels_from_dataset(train_ds)
        sampler = build_weighted_sampler(labels)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"],
        sampler=sampler, shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        collate_fn=_collate,
    )

    model = build_model(cfg["model"]["num_classes"], cfg["model"]["pretrained"]).to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, cfg["train"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    tb = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_f1 = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss_sum = 0.0
        n_seen = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
        train_loss = train_loss_sum / max(n_seen, 1)
        scheduler.step()

        model.eval()
        metrics = ClassificationMetrics(num_classes=cfg["model"]["num_classes"])
        val_loss_sum = 0.0
        n_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                n_seen += imgs.size(0)
                metrics.update(logits, labels)
        val_loss = val_loss_sum / max(n_seen, 1)
        m = metrics.compute()

        logger.info(
            f"epoch {epoch+1}/{cfg['train']['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={m['accuracy']:.4f} val_f1_pos={m['f1_positive']:.4f} "
            f"val_auc={m['auc']:.4f}"
        )
        tb.add_scalar("train/loss", train_loss, epoch)
        tb.add_scalar("val/loss", val_loss, epoch)
        tb.add_scalar("val/accuracy", m["accuracy"], epoch)
        tb.add_scalar("val/f1_positive", m["f1_positive"], epoch)
        tb.add_scalar("val/auc", m["auc"], epoch)
        tb.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": m,
            "config": cfg,
        }, out_dir / "ckpt" / "last.pt")
        if m["f1_positive"] > best_f1:
            best_f1 = m["f1_positive"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": m,
                "config": cfg,
            }, out_dir / "ckpt" / "best.pt")
            logger.info(f"  new best f1_positive={best_f1:.4f}")

    tb.close()
    logger.info("training complete")
    return out_dir


def _collate(batch: list) -> dict:
    """Stack {'image': tensor, 'label': int, ...} dicts into batched tensors."""
    imgs = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    paths = [b["image_path"] for b in batch]
    return {"image": imgs, "label": labels, "image_path": paths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.override)
