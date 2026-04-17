"""ResNet50 model builder for binary citrus disease classification."""
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Return torchvision ResNet50 with the final FC replaced for `num_classes`."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
