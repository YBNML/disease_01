import torch
from segmentation.model import build_model


def test_model_output_shape_3class():
    model = build_model(num_classes=3, encoder_name="resnet34", encoder_weights=None)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    # (B, num_classes, H, W)
    assert y.shape == (2, 3, 128, 128)


def test_model_output_shape_different_size():
    model = build_model(num_classes=3, encoder_name="resnet34", encoder_weights=None)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 256, 256)


def test_model_unpretrained_builds_without_download():
    model = build_model(num_classes=3, encoder_name="resnet34", encoder_weights=None)
    params = sum(p.numel() for p in model.parameters())
    # ResNet34 U-Net has ~24M params
    assert 10_000_000 < params < 50_000_000
