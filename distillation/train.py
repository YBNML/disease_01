"""Knowledge-distillation training: ViT-Small (frozen teacher) → MobileNetV3-Large (student).

Usage:
    python distillation/train.py \
        --config distillation/config.yaml \
        --teacher-ckpt outputs/classification_compare/compare/2026-04-19_01-13-33/vit_small_patch16_224/run/*/ckpt/best.pt

Creates outputs/distillation/run/<timestamp>/ containing:
- ckpt/last.pt, ckpt/best.pt
- tb/ (tensorboard logs)
- train.log
- config.yaml (snapshot)
"""
from __future__ import annotations
import argparse
import glob
import logging
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load_config, apply_overrides
from common.dataset import ClassificationDataset
from common.utils import set_seed, get_device, make_output_dir
from classification.transforms import build_transforms
from classification.model import build_model
from classification.sampler import build_weighted_sampler
from classification.metrics import ClassificationMetrics
from distillation.loss import DistillationLoss


# ---------------------------------------------------------------------------
# Private helpers (duplicated from classification.train for independence)
# ---------------------------------------------------------------------------

def _build_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("distillation.train")
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


def _build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["train"]["optimizer"].lower()
    if name != "adamw":
        raise ValueError(f"unsupported optimizer: {name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )


def _build_scheduler(optimizer, cfg: dict, num_epochs: int):
    name = cfg["train"]["scheduler"].lower()
    if name != "cosine":
        raise ValueError(f"unsupported scheduler: {name}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


def _labels_from_dataset(ds: ClassificationDataset) -> list:
    from common.label_parser import load_sample
    from common.dataset import CLASS_NAME_TO_LABEL
    return [CLASS_NAME_TO_LABEL[load_sample(jp)["class_code"]]
            for _, jp in ds.items]


def _collate(batch: list) -> dict:
    imgs = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    paths = [b["image_path"] for b in batch]
    return {"image": imgs, "label": labels, "image_path": paths}


def _resolve_ckpt(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found matching: {pattern!r}")
    if len(matches) > 1:
        print(f"[warn] multiple teacher checkpoints found, using first: {matches[0]}")
    return Path(matches[0])


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(config_path: str,
         teacher_ckpt: str,
         overrides: list | None = None) -> Path:

    cfg = apply_overrides(load_config(config_path), overrides or [])

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    # outputs/distillation/run/<timestamp>/
    out_dir = make_output_dir(cfg["output"]["root"], task="run")
    (out_dir / "ckpt").mkdir()
    (out_dir / "tb").mkdir()
    shutil.copy(config_path, out_dir / "config.yaml")

    logger = _build_logger(out_dir)
    logger.info(f"config: {cfg}")
    logger.info(f"device: {device}")
    logger.info(f"out_dir: {out_dir}")

    # ── Teacher model (frozen) ───────────────────────────────────────────────
    teacher_ckpt_path = _resolve_ckpt(teacher_ckpt)
    logger.info(f"loading teacher from: {teacher_ckpt_path}")
    teacher = build_model(
        name=cfg["teacher"]["name"],
        num_classes=cfg["teacher"]["num_classes"],
        pretrained=False,
    ).to(device)
    state = torch.load(teacher_ckpt_path, map_location=device)
    key = "model" if "model" in state else "state_dict" if "state_dict" in state else None
    sd = state[key] if key else state
    teacher.load_state_dict(sd)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    logger.info("teacher frozen in eval mode")

    # ── Student model ────────────────────────────────────────────────────────
    logger.info(f"building student: {cfg['student']['name']}")
    student = build_model(
        name=cfg["student"]["name"],
        num_classes=cfg["student"]["num_classes"],
        pretrained=cfg["student"]["pretrained"],
    ).to(device)

    # ── Dataloaders ──────────────────────────────────────────────────────────
    db = Path(cfg["data"]["database_root"])
    train_tf = build_transforms(cfg["data"]["image_size"], train=True)
    val_tf = build_transforms(cfg["data"]["image_size"], train=False)
    train_ds = ClassificationDataset(db, split="train", transform=train_tf)
    val_ds = ClassificationDataset(db, split="val", transform=val_tf)

    if cfg["train"]["use_weighted_sampler"]:
        labels_list = _labels_from_dataset(train_ds)
        sampler = build_weighted_sampler(labels_list)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=_collate,
    )

    logger.info(f"train={len(train_ds)}  val={len(val_ds)}")

    # ── Optimizer / scheduler / loss ─────────────────────────────────────────
    optimizer = _build_optimizer(student, cfg)
    scheduler = _build_scheduler(optimizer, cfg, cfg["train"]["epochs"])
    kd_loss_fn = DistillationLoss(
        alpha=cfg["distill"]["alpha"],
        temperature=cfg["distill"]["temperature"],
    )

    tb = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_f1 = -1.0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(cfg["train"]["epochs"]):
        student.train()
        train_loss_sum = 0.0
        n_seen = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            student_logits = student(imgs)
            with torch.no_grad():
                teacher_logits = teacher(imgs)

            loss = kd_loss_fn(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)

        train_loss = train_loss_sum / max(n_seen, 1)
        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        student.eval()
        metrics = ClassificationMetrics(num_classes=cfg["student"]["num_classes"])
        val_loss_sum = 0.0
        n_seen = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                student_logits = student(imgs)
                teacher_logits = teacher(imgs)
                loss = kd_loss_fn(student_logits, teacher_logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                n_seen += imgs.size(0)
                metrics.update(student_logits, labels)

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

        # Save last
        torch.save({
            "model": student.state_dict(),
            "epoch": epoch,
            "metrics": m,
            "config": cfg,
        }, out_dir / "ckpt" / "last.pt")

        # Save best on f1_positive
        if m["f1_positive"] > best_f1:
            best_f1 = m["f1_positive"]
            torch.save({
                "model": student.state_dict(),
                "epoch": epoch,
                "metrics": m,
                "config": cfg,
            }, out_dir / "ckpt" / "best.pt")
            logger.info(f"  new best f1_positive={best_f1:.4f}")

    tb.close()
    logger.info("distillation training complete")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: ViT-Small → MobileNetV3-Large"
    )
    parser.add_argument("--config", default="distillation/config.yaml")
    parser.add_argument("--teacher-ckpt", required=True,
                        help="Path (or glob) to teacher best.pt checkpoint")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.teacher_ckpt, args.override)
