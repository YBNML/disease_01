# Quantization Study — ViT-Small/16

**Checkpoint**: `outputs/classification_compare/compare/2026-04-19_01-13-33/vit_small_patch16_224/run/2026-04-19_02-09-35/ckpt/best.pt`
**Val samples**: 427
**Device**: CPU (required for quantized ops)

| Variant | Size (MB) | Accuracy | F1 (canker) | Latency bs=1 (ms) | Throughput bs=32 (FPS) |
|---|---:|---:|---:|---:|---:|
| fp32 (baseline) | 82.7 | 0.9906 | 0.9884 | 18.5 | 88.9 |
| int8 (dynamic) | 22.0 | 0.9906 | 0.9884 | 32.2 | 38.5 |

## Summary

- **Size reduction**: 73.4%
- **Latency speedup** (bs=1): 0.58×
- **Accuracy drop**: 0.00 pp
