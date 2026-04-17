"""Weighted sampling for imbalanced binary classification."""
from typing import Sequence, Optional
import numpy as np
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(labels: Sequence[int]) -> np.ndarray:
    """Return per-class weight = 1 / class_count."""
    labels_arr = np.asarray(labels)
    num_classes = int(labels_arr.max()) + 1
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    # avoid div-by-zero if a class is missing (shouldn't happen here)
    counts[counts == 0] = 1.0
    return 1.0 / counts


def build_weighted_sampler(
    labels: Sequence[int],
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """WeightedRandomSampler for class-balanced sampling.
    `num_samples` defaults to len(labels) (one epoch worth)."""
    class_w = compute_class_weights(labels)
    sample_w = class_w[np.asarray(labels)]
    if num_samples is None:
        num_samples = len(labels)
    return WeightedRandomSampler(
        weights=sample_w.tolist(),
        num_samples=num_samples,
        replacement=True,
    )
