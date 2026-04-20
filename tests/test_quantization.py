"""Smoke-test: dynamic INT8 quantization runs without errors on a tiny Linear model.

Note: On Apple Silicon (ARM64), only 'qnnpack' is available as the quantization
engine. We set it explicitly before quantizing.
"""
import torch
import torch.nn as nn
import pytest


def _set_quant_engine():
    """Set the best available quantization engine for this platform."""
    engines = torch.backends.quantized.supported_engines
    if "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"
    elif engines:
        torch.backends.quantized.engine = engines[0]
    else:
        pytest.skip("No quantization engine available on this platform")


def test_quantize_dynamic_linear_model():
    """quantize_dynamic on a simple Linear model runs inference without errors."""
    _set_quant_engine()

    class TinyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 8)
            self.fc2 = nn.Linear(8, 2)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = TinyLinear().eval()
    q_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    dummy = torch.randn(4, 16)
    out = q_model(dummy)
    assert out.shape == (4, 2), f"unexpected output shape: {out.shape}"


def test_quantize_dynamic_preserves_output_shape():
    """Output shape is unchanged after quantization."""
    _set_quant_engine()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            )

        def forward(self, x):
            return self.layers(x)

    model = MLP().eval()
    q_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    dummy = torch.randn(8, 32)
    fp32_out = model(dummy)
    int8_out = q_model(dummy)
    assert fp32_out.shape == int8_out.shape
