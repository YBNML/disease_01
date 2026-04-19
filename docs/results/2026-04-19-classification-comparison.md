# Classification Backbone Comparison

| Model | Params | Acc | F1 (canker) | AUC | Latency bs=1 (ms) | Throughput (FPS) |
|---|---:|---:|---:|---:|---:|---:|
| resnet50 | 23,512,130 | 0.9883 | 0.9856 | 0.9975 | 27.86 | 83.8 |
| efficientnet_b0 | 4,010,110 | 0.9742 | 0.9679 | 0.9902 | 19.68 | 160.6 |
| convnext_tiny | 27,821,666 | 0.9883 | 0.9857 | 0.9994 | 7.18 | 109.8 |
| mobilenetv3_large_100 | 4,204,594 | 0.9766 | 0.9708 | 0.9954 | 13.12 | 264.6 |
| vit_small_patch16_224 | 21,666,434 | 0.9906 | 0.9884 | 0.9990 | 5.94 | 175.0 |
