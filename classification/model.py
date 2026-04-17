"""Classification model builder supporting torchvision and timm backbones."""
from typing import Optional
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights


TORCHVISION_MODELS = {"resnet50"}


def build_model(name: str = "resnet50",
                num_classes: int = 2,
                pretrained: bool = True) -> nn.Module:
    """Return a classification model by name.

    - 'resnet50' uses torchvision (to match the original P1 baseline).
    - Any other name is treated as a timm model name and created via
      `timm.create_model(name, pretrained=..., num_classes=...)`.
    """
    if name in TORCHVISION_MODELS:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # fall through to timm
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
