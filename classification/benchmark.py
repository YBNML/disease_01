"""Benchmark utilities: parameter count, inference latency, throughput.

Handles MPS/CPU devices correctly (proper synchronization before timing).
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Total (or trainable-only) parameter count."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _sync(device: str) -> None:
    """Block until all queued kernels on the device finish (for accurate timing)."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def measure_inference_latency(
    model: nn.Module,
    input_sample: torch.Tensor,
    device: str = "cpu",
    warmup: int = 5,
    iters: int = 30,
) -> float:
    """Mean seconds per forward pass for `input_sample` on `device`.

    `input_sample` is used as-is (not modified). Batch size is whatever the
    sample has (use batch=1 for single-image latency)."""
    model = model.to(device).eval()
    x = input_sample.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync(device)

        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        _sync(device)
        elapsed = time.perf_counter() - start

    return elapsed / iters


def measure_throughput(
    model: nn.Module,
    input_sample: torch.Tensor,
    device: str = "cpu",
    warmup: int = 5,
    iters: int = 30,
) -> float:
    """Images per second on `device`, given a batched `input_sample` (N, C, H, W)."""
    latency = measure_inference_latency(model, input_sample, device, warmup, iters)
    batch_size = input_sample.size(0)
    return batch_size / latency
