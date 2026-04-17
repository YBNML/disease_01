import torch
from segmentation.losses import CombinedLoss


def _perfect_logits(mask: torch.Tensor, num_classes: int, scale: float = 20.0) -> torch.Tensor:
    """Build logits that near-perfectly predict `mask`. Shape (B, C, H, W)."""
    B, H, W = mask.shape
    logits = torch.full((B, num_classes, H, W), -scale)
    for c in range(num_classes):
        logits[:, c][mask == c] = scale
    return logits


def test_combined_loss_near_zero_on_perfect_prediction():
    loss_fn = CombinedLoss(num_classes=3, ce_weight=0.5, dice_weight=0.5)
    mask = torch.randint(0, 3, (2, 16, 16))
    logits = _perfect_logits(mask, num_classes=3)
    loss = loss_fn(logits, mask)
    assert loss.item() < 0.05


def test_combined_loss_positive_on_random():
    loss_fn = CombinedLoss(num_classes=3, ce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(2, 3, 16, 16)
    mask = torch.randint(0, 3, (2, 16, 16))
    loss = loss_fn(logits, mask)
    assert loss.item() > 0.1


def test_combined_loss_weights_respected():
    """With dice_weight=1 and ce_weight=0, total should be pure Dice (no CE term)."""
    loss_fn_mix = CombinedLoss(num_classes=3, ce_weight=0.5, dice_weight=0.5)
    loss_fn_dice = CombinedLoss(num_classes=3, ce_weight=0.0, dice_weight=1.0)
    logits = torch.randn(2, 3, 16, 16)
    mask = torch.randint(0, 3, (2, 16, 16))
    mix = loss_fn_mix(logits, mask).item()
    dice = loss_fn_dice(logits, mask).item()
    # They should differ (unless CE=0 by coincidence, which is essentially impossible)
    assert abs(mix - dice) > 1e-3


def test_combined_loss_returns_scalar():
    loss_fn = CombinedLoss(num_classes=3)
    logits = torch.randn(1, 3, 8, 8)
    mask = torch.zeros(1, 8, 8, dtype=torch.long)
    loss = loss_fn(logits, mask)
    assert loss.dim() == 0
