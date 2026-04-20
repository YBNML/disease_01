# Segmentation Architecture Comparison

| Label | Arch | Encoder | Params | mIoU | pixAcc | IoU bg | IoU normal | IoU canker | Dice normal | Dice canker | Latency bs=1 (ms) | Throughput (FPS) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unet_resnet34 | Unet | resnet34 | 24,436,659 | 0.9616 | 0.9948 | 0.9978 | 0.9579 | 0.9291 | 0.9785 | 0.9632 | 26.23 | 40.5 |
| unet_efficientnet-b0 | Unet | efficientnet-b0 | 6,251,759 | 0.9077 | 0.9889 | 0.9976 | 0.9083 | 0.8172 | 0.9520 | 0.8994 | 24.63 | 43.3 |
| deeplabv3plus_resnet34 | DeepLabV3Plus | resnet34 | 22,437,971 | 0.9777 | 0.9966 | 0.9977 | 0.9732 | 0.9621 | 0.9864 | 0.9807 | 30.89 | 36.1 |
| fpn_resnet34 | FPN | resnet34 | 23,155,651 | 0.9677 | 0.9955 | 0.9977 | 0.9654 | 0.9400 | 0.9824 | 0.9691 | 24.95 | 41.6 |
