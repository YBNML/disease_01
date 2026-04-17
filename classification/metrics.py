"""Accumulating metrics for binary classification evaluation."""
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
)


class ClassificationMetrics:
    """Accumulates per-batch predictions and computes summary metrics on demand.

    Designed for binary classification (num_classes=2) but confusion matrix
    handles arbitrary num_classes. `positive_class` identifies which class is
    the 'positive' for precision/recall/F1/AUC (default: 1 = 궤양병)."""

    def __init__(self, num_classes: int = 2, positive_class: int = 1):
        self.num_classes = num_classes
        self.positive_class = positive_class
        self.reset()

    def reset(self) -> None:
        self._labels = []
        self._preds = []
        self._scores = []  # softmax probability for positive_class

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate a batch. `logits` shape (B, C), `labels` shape (B,)."""
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        self._labels.extend(labels.detach().cpu().tolist())
        self._preds.extend(preds.detach().cpu().tolist())
        self._scores.extend(probs[:, self.positive_class].detach().cpu().tolist())

    def compute(self) -> Dict:
        if not self._labels:
            return {"accuracy": 0.0, "f1_positive": 0.0, "auc": 0.0,
                    "confusion_matrix": np.zeros((self.num_classes, self.num_classes),
                                                 dtype=np.int64)}
        y_true = np.asarray(self._labels)
        y_pred = np.asarray(self._preds)
        y_score = np.asarray(self._scores)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(self.num_classes)), zero_division=0
        )
        # AUC requires both classes present
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = float("nan")

        return {
            "accuracy": float(acc),
            "precision_per_class": prec.tolist(),
            "recall_per_class": rec.tolist(),
            "f1_per_class": f1.tolist(),
            "f1_positive": float(f1[self.positive_class]),
            "precision_positive": float(prec[self.positive_class]),
            "recall_positive": float(rec[self.positive_class]),
            "auc": float(auc),
            "confusion_matrix": cm,
        }
