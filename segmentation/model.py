"""smp U-Net builder for 3-class citrus semantic segmentation."""
from typing import Optional
import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    num_classes: int = 3,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    """Return smp.Unet with the given encoder and output classes."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
