import numpy as np
import torch
from classification.transforms import build_transforms


def test_train_transform_output_shape():
    t = build_transforms(image_size=224, train=True)
    img = (np.random.rand(500, 400, 3) * 255).astype(np.uint8)
    out = t(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_val_transform_output_shape():
    t = build_transforms(image_size=224, train=False)
    img = (np.random.rand(500, 400, 3) * 255).astype(np.uint8)
    out = t(img)
    assert out.shape == (3, 224, 224)


def test_val_transform_is_deterministic():
    """Val transform has no randomness — same input → same output."""
    t = build_transforms(image_size=224, train=False)
    img = (np.random.rand(500, 400, 3) * 255).astype(np.uint8)
    out1 = t(img)
    out2 = t(img)
    torch.testing.assert_close(out1, out2)


def test_train_transform_is_random():
    """Train transform includes randomness — two calls likely differ."""
    import random
    random.seed(0)
    torch.manual_seed(0)
    t = build_transforms(image_size=224, train=True)
    img = (np.random.rand(500, 400, 3) * 255).astype(np.uint8)
    out1 = t(img)
    out2 = t(img)
    # extremely unlikely to be identical with flip+colorjitter
    assert not torch.allclose(out1, out2)


def test_normalization_stats_applied():
    """After ImageNet normalization, pure white (255) input produces well-known values."""
    t = build_transforms(image_size=32, train=False)  # small size to speed up
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    out = t(white)
    # BGR input (from cv2) becomes RGB after transform; all channels should be
    # (1.0 - mean) / std per ImageNet constants
    expected_ch0 = (1.0 - 0.485) / 0.229  # R
    assert abs(out[0].mean().item() - expected_ch0) < 1e-3
