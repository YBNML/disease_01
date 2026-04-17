import torch
from classification.model import build_model


def test_model_produces_correct_output_shape():
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2)


def test_model_final_fc_replaced_to_num_classes():
    model = build_model(num_classes=2, pretrained=False)
    # torchvision ResNet50 stores the final classifier as `.fc`
    assert hasattr(model, "fc")
    assert model.fc.out_features == 2


def test_model_pretrained_false_runs_quickly():
    """Fast sanity: unpretrained build completes without downloading."""
    model = build_model(num_classes=2, pretrained=False)
    assert sum(p.numel() for p in model.parameters()) > 10_000_000
