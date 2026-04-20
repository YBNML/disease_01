import torch
import pytest
from segmentation.model import build_model


def test_unet_resnet34_output_shape():
    model = build_model(num_classes=3, architecture="Unet",
                        encoder_name="resnet34", encoder_weights=None)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3, 128, 128)


def test_unet_output_shape_different_size():
    model = build_model(num_classes=3, architecture="Unet",
                        encoder_name="resnet34", encoder_weights=None)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 256, 256)


def test_unet_param_count_in_expected_range():
    model = build_model(num_classes=3, architecture="Unet",
                        encoder_name="resnet34", encoder_weights=None)
    params = sum(p.numel() for p in model.parameters())
    assert 10_000_000 < params < 50_000_000


@pytest.mark.parametrize("architecture", ["DeepLabV3Plus", "FPN"])
def test_other_architectures_build_and_infer(architecture):
    model = build_model(num_classes=3, architecture=architecture,
                        encoder_name="resnet34", encoder_weights=None)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 128, 128)


def test_build_model_raises_on_unknown_architecture():
    with pytest.raises(ValueError):
        build_model(num_classes=3, architecture="NotARealArch",
                    encoder_name="resnet34", encoder_weights=None)
