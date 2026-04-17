"""Combined CrossEntropy + Dice loss for multiclass semantic segmentation."""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """weight_ce * CrossEntropy + weight_dice * Dice.

    Expects `logits` of shape (B, C, H, W) and `mask` of shape (B, H, W)
    with integer class ids in [0, C)."""

    def __init__(self, num_classes: int = 3,
                 ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask may arrive as uint8 from albumentations; CE needs long
        mask = mask.long()
        ce = self.ce(logits, mask)
        dice = self.dice(logits, mask)
        return self.ce_weight * ce + self.dice_weight * dice
