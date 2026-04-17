import torch
import pytest
from classification.model import build_model


def test_resnet50_output_shape():
    model = build_model(name="resnet50", num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2)


def test_default_resnet50_param_count():
    model = build_model(name="resnet50", num_classes=2, pretrained=False)
    n = sum(p.numel() for p in model.parameters())
    # ResNet50 is ~25M params
    assert 20_000_000 < n < 30_000_000


@pytest.mark.parametrize("name", [
    "efficientnet_b0",
    "convnext_tiny",
    "mobilenetv3_large_100",
    "vit_small_patch16_224",
])
def test_timm_backbones_build_and_infer(name):
    model = build_model(name=name, num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2)


def test_build_model_raises_on_unknown_name():
    with pytest.raises(Exception):
        build_model(name="definitely_not_a_real_model", num_classes=2, pretrained=False)
