import torch
from segmentation.metrics import SegmentationMetrics


def _logits_from_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    B, H, W = mask.shape
    logits = torch.full((B, num_classes, H, W), -10.0)
    for c in range(num_classes):
        logits[:, c][mask == c] = 10.0
    return logits


def test_metrics_perfect_prediction():
    m = SegmentationMetrics(num_classes=3)
    mask = torch.randint(0, 3, (2, 16, 16))
    logits = _logits_from_mask(mask, 3)
    m.update(logits, mask)
    res = m.compute()
    assert abs(res["miou"] - 1.0) < 1e-6
    assert abs(res["pixel_accuracy"] - 1.0) < 1e-6
    assert all(abs(v - 1.0) < 1e-6 for v in res["iou_per_class"])


def test_metrics_all_wrong_assignment():
    """Predict every pixel as class 0 while the mask is all class 1."""
    m = SegmentationMetrics(num_classes=3)
    mask = torch.ones(1, 8, 8, dtype=torch.long)
    logits = torch.zeros(1, 3, 8, 8)
    logits[:, 0] = 10.0
    m.update(logits, mask)
    res = m.compute()
    # class 1 should have IoU=0, class 0 should have IoU=0 (pred all 0 but no gt),
    # class 2 has no gt and no pred → skipped or 0
    assert res["iou_per_class"][1] == 0.0
    assert res["pixel_accuracy"] == 0.0


def test_metrics_confusion_matrix_counts():
    m = SegmentationMetrics(num_classes=3)
    # GT all class 1, predict half class 1 and half class 2
    mask = torch.ones(1, 4, 4, dtype=torch.long)
    logits = torch.zeros(1, 3, 4, 4)
    logits[:, 1, :, :2] = 10.0
    logits[:, 2, :, 2:] = 10.0
    m.update(logits, mask)
    res = m.compute()
    cm = res["confusion_matrix"]
    # 8 correct (class 1) + 8 mispredicted as class 2
    assert cm[1, 1] == 8
    assert cm[1, 2] == 8


def test_metrics_reset():
    m = SegmentationMetrics(num_classes=3)
    mask = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = _logits_from_mask(mask, 3)
    m.update(logits, mask)
    m.reset()
    res = m.compute()
    # should not raise; miou may be 0/NaN with empty state
    assert "miou" in res


def test_metrics_dice_per_class_for_perfect():
    m = SegmentationMetrics(num_classes=3)
    mask = torch.randint(0, 3, (1, 8, 8))
    logits = _logits_from_mask(mask, 3)
    m.update(logits, mask)
    res = m.compute()
    assert all(abs(d - 1.0) < 1e-6 for d in res["dice_per_class"])
