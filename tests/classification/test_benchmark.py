import torch
import torch.nn as nn
from classification.benchmark import (
    count_parameters, measure_inference_latency, measure_throughput,
)


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_count_parameters_returns_positive_int():
    m = _TinyClassifier()
    n = count_parameters(m)
    assert isinstance(n, int)
    assert n > 0
    # conv (3*8*9 + 8) + fc (8*2 + 2) = 216+8 + 16+2 = 242
    assert n == 242


def test_count_parameters_trainable_only():
    m = _TinyClassifier()
    for p in m.conv.parameters():
        p.requires_grad_(False)
    total = count_parameters(m, trainable_only=False)
    trainable = count_parameters(m, trainable_only=True)
    assert total > trainable


def test_measure_inference_latency_positive():
    m = _TinyClassifier()
    m.eval()
    x = torch.randn(1, 3, 32, 32)
    lat = measure_inference_latency(m, x, device="cpu", warmup=2, iters=5)
    assert lat > 0.0  # seconds per forward


def test_measure_throughput_positive():
    m = _TinyClassifier()
    m.eval()
    x = torch.randn(8, 3, 32, 32)
    fps = measure_throughput(m, x, device="cpu", warmup=2, iters=5)
    assert fps > 0.0  # images per second
