import numpy as np
import torch
from segmentation.transforms import build_transforms


def test_train_transform_output_shapes():
    t = build_transforms(image_size=128, train=True)
    img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    mask = np.random.randint(0, 3, (200, 300), dtype=np.uint8)
    out = t(image=img, mask=mask)
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 128, 128)
    assert out["image"].dtype == torch.float32
    # mask keeps uint8/int64 dtype but shape must match (H, W)
    assert out["mask"].shape == (128, 128)


def test_val_transform_output_shapes():
    t = build_transforms(image_size=128, train=False)
    img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    out = t(image=img, mask=mask)
    assert out["image"].shape == (3, 128, 128)
    assert out["mask"].shape == (128, 128)


def test_val_transform_preserves_mask_values():
    """Val transform only resizes+normalizes; mask class ids must stay {0,1,2}."""
    t = build_transforms(image_size=128, train=False)
    img = np.full((200, 300, 3), 200, dtype=np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[50:150, 50:200] = 2  # a 2-labeled region
    out = t(image=img, mask=mask)
    unique = set(torch.unique(out["mask"]).tolist())
    assert unique.issubset({0, 2})


def test_val_transform_is_deterministic():
    t = build_transforms(image_size=64, train=False)
    img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    mask = np.random.randint(0, 3, (100, 100), dtype=np.uint8)
    o1 = t(image=img, mask=mask)
    o2 = t(image=img, mask=mask)
    torch.testing.assert_close(o1["image"], o2["image"])
    torch.testing.assert_close(o1["mask"].float(), o2["mask"].float())
