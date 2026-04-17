import numpy as np
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from classification.sampler import build_weighted_sampler, compute_class_weights


def test_compute_class_weights_inverse_frequency():
    labels = [0, 0, 0, 1]  # 3 normal, 1 canker
    w = compute_class_weights(labels)
    assert w.shape == (2,)
    # class 0 has 3 samples → weight 1/3; class 1 has 1 sample → weight 1
    assert abs(w[0] - 1 / 3) < 1e-6
    assert abs(w[1] - 1.0) < 1e-6


def test_build_weighted_sampler_returns_sampler():
    labels = [0, 0, 0, 1, 1]
    sampler = build_weighted_sampler(labels)
    assert isinstance(sampler, WeightedRandomSampler)
    assert len(sampler) == len(labels)


def test_build_weighted_sampler_draws_balanced():
    """With enough draws, minority class should appear close to 50%."""
    labels = [0] * 100 + [1] * 20  # heavy imbalance
    sampler = build_weighted_sampler(labels)
    # Draw len(labels)*10 samples from a long sampler to check the distribution
    long_sampler = build_weighted_sampler(labels, num_samples=10_000)
    indices = list(iter(long_sampler))
    drawn_labels = [labels[i] for i in indices]
    counts = Counter(drawn_labels)
    # After weighting, both classes should be roughly equal (within 10%)
    ratio = counts[1] / (counts[0] + counts[1])
    assert 0.4 < ratio < 0.6
