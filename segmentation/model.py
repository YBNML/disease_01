"""smp segmentation model builder supporting multiple architectures."""
from typing import Optional
import segmentation_models_pytorch as smp
import torch.nn as nn


ARCHITECTURES = {
    "Unet": smp.Unet,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "UnetPlusPlus": smp.UnetPlusPlus,
}


def build_model(
    num_classes: int = 3,
    architecture: str = "Unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    """Return an smp segmentation model.

    architecture: one of Unet / DeepLabV3Plus / FPN / PSPNet / UnetPlusPlus
    encoder_name: any timm/smp encoder name (resnet34, efficientnet-b0, mobilenet_v2, ...)
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture!r}. "
                         f"Choose from {sorted(ARCHITECTURES)}")
    cls = ARCHITECTURES[architecture]
    return cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
