"""Accumulating metrics for multiclass semantic segmentation.

Maintains a running pixel-level confusion matrix and derives IoU / Dice /
pixel accuracy from it on demand. Classes with zero ground-truth AND zero
prediction pixels are excluded from the mean (mIoU).
"""
from typing import Dict
import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        """logits: (B, C, H, W); mask: (B, H, W) long/int."""
        with torch.no_grad():
            preds = logits.argmax(dim=1).detach().cpu().numpy().ravel()
            trues = mask.detach().cpu().numpy().ravel().astype(np.int64)
        # Accumulate confusion matrix
        valid = (trues >= 0) & (trues < self.num_classes) & \
                (preds >= 0) & (preds < self.num_classes)
        idx = trues[valid] * self.num_classes + preds[valid]
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self._cm += binc.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict:
        cm = self._cm
        total = cm.sum()
        pixel_acc = float(np.diag(cm).sum() / total) if total > 0 else 0.0

        iou_per_class = []
        dice_per_class = []
        for c in range(self.num_classes):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            denom_iou = tp + fp + fn
            denom_dice = 2 * tp + fp + fn
            iou = tp / denom_iou if denom_iou > 0 else 0.0
            dice = 2 * tp / denom_dice if denom_dice > 0 else 0.0
            iou_per_class.append(iou)
            dice_per_class.append(dice)

        # mIoU over classes that had any ground truth OR predictions
        active = [c for c in range(self.num_classes)
                  if cm[c, :].sum() > 0 or cm[:, c].sum() > 0]
        miou = float(np.mean([iou_per_class[c] for c in active])) if active else 0.0

        return {
            "miou": miou,
            "iou_per_class": iou_per_class,
            "dice_per_class": dice_per_class,
            "pixel_accuracy": pixel_acc,
            "confusion_matrix": cm.copy(),
        }
