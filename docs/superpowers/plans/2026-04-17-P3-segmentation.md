# P3 — Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** segmentation_models_pytorch U-Net (ResNet34 encoder) 기반 3-class semantic segmentation 파이프라인을 구축하고, 실제 데이터로 smoke 학습까지 검증한다.

**Architecture:** `common.dataset.SegmentationDataset` 재사용. `segmentation/` 하위에 transforms(albumentations) / model(smp) / losses(CE+Dice) / metrics(mIoU, Dice, pixAcc) / config / train / eval 모듈을 분리. Best checkpoint는 val mIoU 기준.

**Tech Stack:** PyTorch (MPS), segmentation_models_pytorch, albumentations, numpy, pytest

**Prereq:**
- P0, P1, P2 완료
- Spec §8, `common.dataset.SegmentationDataset` 활용

---

## File Structure

**Create:**
- `segmentation/__init__.py`
- `segmentation/transforms.py` — `build_transforms(image_size, train)` albumentations 팩토리
- `segmentation/model.py` — `build_model(num_classes, encoder_name, encoder_weights)` smp 래퍼
- `segmentation/losses.py` — `CombinedLoss` (CE + Dice)
- `segmentation/metrics.py` — `SegmentationMetrics` 집계기
- `segmentation/config.yaml` — 학습 설정
- `segmentation/train.py` — 학습 엔트리포인트
- `segmentation/eval.py` — 평가 + 정성 시각화
- `tests/segmentation/__init__.py`
- `tests/segmentation/conftest.py` — synthetic fixture 재노출
- `tests/segmentation/test_transforms.py`
- `tests/segmentation/test_model.py`
- `tests/segmentation/test_losses.py`
- `tests/segmentation/test_metrics.py`
- `tests/segmentation/test_train.py`
- `tests/segmentation/test_eval.py`

**Modify:**
- `README.md` — P3 체크박스

---

## Working Directory & Environment

- Root: `<project root>`
- Activation:
  ```bash
  source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE
  ```

---

### Task 1: Scaffold `segmentation/` package

**Files:**
- Create: `segmentation/__init__.py`, `tests/segmentation/__init__.py`, `tests/segmentation/conftest.py`

- [ ] **Step 1: Create dirs and markers**

```bash
cd <project root>
mkdir -p segmentation tests/segmentation
touch segmentation/__init__.py tests/segmentation/__init__.py
```

- [ ] **Step 2: Create conftest.py to share fixtures**

Create `tests/segmentation/conftest.py`:

```python
from tests.common.conftest import (
    synthetic_dataset_root,
    sample_json_with_polygon,
    sample_json_no_polygon,
)

__all__ = [
    "synthetic_dataset_root",
    "sample_json_with_polygon",
    "sample_json_no_polygon",
]
```

- [ ] **Step 3: Verify pytest still green**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && pytest
```

Expected: 56 passed.

- [ ] **Step 4: Commit**

```bash
git add segmentation/ tests/segmentation/
git commit -m "$(cat <<'EOF'
chore: scaffold segmentation package

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: TDD — transforms

**Files:**
- Create: `tests/segmentation/test_transforms.py`
- Create: `segmentation/transforms.py`

- [ ] **Step 1: Write failing tests**

Create `tests/segmentation/test_transforms.py`:

```python
import numpy as np
import torch
from segmentation.transforms import build_transforms


def test_train_transform_output_shapes():
    t = build_transforms(image_size=128, train=True)
    img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    mask = np.random.randint(0, 3, (200, 300), dtype=np.uint8)
    out = t(image=img, mask=mask)
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 128, 128)
    assert out["image"].dtype == torch.float32
    # mask keeps uint8/int64 dtype but shape must match (H, W)
    assert out["mask"].shape == (128, 128)


def test_val_transform_output_shapes():
    t = build_transforms(image_size=128, train=False)
    img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    out = t(image=img, mask=mask)
    assert out["image"].shape == (3, 128, 128)
    assert out["mask"].shape == (128, 128)


def test_val_transform_preserves_mask_values():
    """Val transform only resizes+normalizes; mask class ids must stay {0,1,2}."""
    t = build_transforms(image_size=128, train=False)
    img = np.full((200, 300, 3), 200, dtype=np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[50:150, 50:200] = 2  # a 2-labeled region
    out = t(image=img, mask=mask)
    unique = set(torch.unique(out["mask"]).tolist())
    assert unique.issubset({0, 2})


def test_val_transform_is_deterministic():
    t = build_transforms(image_size=64, train=False)
    img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    mask = np.random.randint(0, 3, (100, 100), dtype=np.uint8)
    o1 = t(image=img, mask=mask)
    o2 = t(image=img, mask=mask)
    torch.testing.assert_close(o1["image"], o2["image"])
    torch.testing.assert_close(o1["mask"].float(), o2["mask"].float())
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/segmentation/test_transforms.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement transforms**

Create `segmentation/transforms.py`:

```python
"""Image + mask transforms for segmentation training and validation.

Input is a BGR numpy image (H, W, 3) uint8 from cv2.imread and a (H, W) uint8
mask with values in {0: bg, 1: normal fruit, 2: canker fruit}. The pipeline
converts BGR→RGB (via albumentations), then applies geometric/photometric
transforms, and finally normalizes with ImageNet stats and stacks to tensor.
Mask is resized with nearest-neighbor interpolation to preserve class ids.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 512, train: bool = True):
    """Return an albumentations Compose accepting image=<uint8 HxWx3> and
    mask=<uint8 HxW>, producing image=<float32 3xHxW> and mask=<int64 HxW>."""
    base = [
        # AI Hub images are BGR from cv2; albumentations' geometry ops are
        # channel-agnostic, but Normalize expects values in the convention the
        # user supplies. We convert BGR→RGB first so ImageNet stats apply
        # correctly after normalization.
        A.Lambda(image=lambda x, **kw: x[..., ::-1].copy(), mask=None),
        A.Resize(image_size, image_size, interpolation=1, mask_interpolation=0),
    ]
    if train:
        base += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ColorJitter(0.1, 0.1, 0.1, p=0.5),
        ]
    base += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(base)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/segmentation/test_transforms.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add segmentation/transforms.py tests/segmentation/test_transforms.py
git commit -m "$(cat <<'EOF'
feat(segmentation): add albumentations transform factory for image+mask

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: TDD — model builder

**Files:**
- Create: `tests/segmentation/test_model.py`
- Create: `segmentation/model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/segmentation/test_model.py`:

```python
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
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/segmentation/test_model.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement model builder**

Create `segmentation/model.py`:

```python
"""smp U-Net builder for 3-class citrus semantic segmentation."""
from typing import Optional
import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    num_classes: int = 3,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    """Return smp.Unet with the given encoder and output classes."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/segmentation/test_model.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add segmentation/model.py tests/segmentation/test_model.py
git commit -m "$(cat <<'EOF'
feat(segmentation): add smp U-Net model builder

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: TDD — combined loss (CE + Dice)

**Files:**
- Create: `tests/segmentation/test_losses.py`
- Create: `segmentation/losses.py`

- [ ] **Step 1: Write failing tests**

Create `tests/segmentation/test_losses.py`:

```python
import torch
from segmentation.losses import CombinedLoss


def _perfect_logits(mask: torch.Tensor, num_classes: int, scale: float = 20.0) -> torch.Tensor:
    """Build logits that near-perfectly predict `mask`. Shape (B, C, H, W)."""
    B, H, W = mask.shape
    logits = torch.full((B, num_classes, H, W), -scale)
    for c in range(num_classes):
        logits[:, c][mask == c] = scale
    return logits


def test_combined_loss_near_zero_on_perfect_prediction():
    loss_fn = CombinedLoss(num_classes=3, ce_weight=0.5, dice_weight=0.5)
    mask = torch.randint(0, 3, (2, 16, 16))
    logits = _perfect_logits(mask, num_classes=3)
    loss = loss_fn(logits, mask)
    assert loss.item() < 0.05


def test_combined_loss_positive_on_random():
    loss_fn = CombinedLoss(num_classes=3, ce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(2, 3, 16, 16)
    mask = torch.randint(0, 3, (2, 16, 16))
    loss = loss_fn(logits, mask)
    assert loss.item() > 0.1


def test_combined_loss_weights_respected():
    """With dice_weight=1 and ce_weight=0, total should be pure Dice (no CE term)."""
    loss_fn_mix = CombinedLoss(num_classes=3, ce_weight=0.5, dice_weight=0.5)
    loss_fn_dice = CombinedLoss(num_classes=3, ce_weight=0.0, dice_weight=1.0)
    logits = torch.randn(2, 3, 16, 16)
    mask = torch.randint(0, 3, (2, 16, 16))
    mix = loss_fn_mix(logits, mask).item()
    dice = loss_fn_dice(logits, mask).item()
    # They should differ (unless CE=0 by coincidence, which is essentially impossible)
    assert abs(mix - dice) > 1e-3


def test_combined_loss_returns_scalar():
    loss_fn = CombinedLoss(num_classes=3)
    logits = torch.randn(1, 3, 8, 8)
    mask = torch.zeros(1, 8, 8, dtype=torch.long)
    loss = loss_fn(logits, mask)
    assert loss.dim() == 0
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/segmentation/test_losses.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement combined loss**

Create `segmentation/losses.py`:

```python
"""Combined CrossEntropy + Dice loss for multiclass semantic segmentation."""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """weight_ce * CrossEntropy + weight_dice * Dice.

    Expects `logits` of shape (B, C, H, W) and `mask` of shape (B, H, W)
    with integer class ids in [0, C)."""

    def __init__(self, num_classes: int = 3,
                 ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask may arrive as uint8 from albumentations; CE needs long
        mask = mask.long()
        ce = self.ce(logits, mask)
        dice = self.dice(logits, mask)
        return self.ce_weight * ce + self.dice_weight * dice
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/segmentation/test_losses.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add segmentation/losses.py tests/segmentation/test_losses.py
git commit -m "$(cat <<'EOF'
feat(segmentation): add CombinedLoss (CE + Dice)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: TDD — metrics aggregator

**Files:**
- Create: `tests/segmentation/test_metrics.py`
- Create: `segmentation/metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/segmentation/test_metrics.py`:

```python
import torch
from segmentation.metrics import SegmentationMetrics


def _logits_from_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    B, H, W = mask.shape
    logits = torch.full((B, num_classes, H, W), -10.0)
    for c in range(num_classes):
        logits[:, c][mask == c] = 10.0
    return logits


def test_metrics_perfect_prediction():
    m = SegmentationMetrics(num_classes=3)
    mask = torch.randint(0, 3, (2, 16, 16))
    logits = _logits_from_mask(mask, 3)
    m.update(logits, mask)
    res = m.compute()
    assert abs(res["miou"] - 1.0) < 1e-6
    assert abs(res["pixel_accuracy"] - 1.0) < 1e-6
    assert all(abs(v - 1.0) < 1e-6 for v in res["iou_per_class"])


def test_metrics_all_wrong_assignment():
    """Predict every pixel as class 0 while the mask is all class 1."""
    m = SegmentationMetrics(num_classes=3)
    mask = torch.ones(1, 8, 8, dtype=torch.long)
    logits = torch.zeros(1, 3, 8, 8)
    logits[:, 0] = 10.0
    m.update(logits, mask)
    res = m.compute()
    # class 1 should have IoU=0, class 0 should have IoU=0 (pred all 0 but no gt),
    # class 2 has no gt and no pred → skipped or 0
    assert res["iou_per_class"][1] == 0.0
    assert res["pixel_accuracy"] == 0.0


def test_metrics_confusion_matrix_counts():
    m = SegmentationMetrics(num_classes=3)
    # GT all class 1, predict half class 1 and half class 2
    mask = torch.ones(1, 4, 4, dtype=torch.long)
    logits = torch.zeros(1, 3, 4, 4)
    logits[:, 1, :, :2] = 10.0
    logits[:, 2, :, 2:] = 10.0
    m.update(logits, mask)
    res = m.compute()
    cm = res["confusion_matrix"]
    # 8 correct (class 1) + 8 mispredicted as class 2
    assert cm[1, 1] == 8
    assert cm[1, 2] == 8


def test_metrics_reset():
    m = SegmentationMetrics(num_classes=3)
    mask = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = _logits_from_mask(mask, 3)
    m.update(logits, mask)
    m.reset()
    res = m.compute()
    # should not raise; miou may be 0/NaN with empty state
    assert "miou" in res


def test_metrics_dice_per_class_for_perfect():
    m = SegmentationMetrics(num_classes=3)
    mask = torch.randint(0, 3, (1, 8, 8))
    logits = _logits_from_mask(mask, 3)
    m.update(logits, mask)
    res = m.compute()
    assert all(abs(d - 1.0) < 1e-6 for d in res["dice_per_class"])
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/segmentation/test_metrics.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement metrics aggregator**

Create `segmentation/metrics.py`:

```python
"""Accumulating metrics for multiclass semantic segmentation.

Maintains a running pixel-level confusion matrix and derives IoU / Dice /
pixel accuracy from it on demand. Classes with zero ground-truth AND zero
prediction pixels are excluded from the mean (mIoU).
"""
from typing import Dict
import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        """logits: (B, C, H, W); mask: (B, H, W) long/int."""
        with torch.no_grad():
            preds = logits.argmax(dim=1).detach().cpu().numpy().ravel()
            trues = mask.detach().cpu().numpy().ravel().astype(np.int64)
        # Accumulate confusion matrix
        valid = (trues >= 0) & (trues < self.num_classes) & \
                (preds >= 0) & (preds < self.num_classes)
        idx = trues[valid] * self.num_classes + preds[valid]
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self._cm += binc.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict:
        cm = self._cm
        total = cm.sum()
        pixel_acc = float(np.diag(cm).sum() / total) if total > 0 else 0.0

        iou_per_class = []
        dice_per_class = []
        for c in range(self.num_classes):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            denom_iou = tp + fp + fn
            denom_dice = 2 * tp + fp + fn
            iou = tp / denom_iou if denom_iou > 0 else 0.0
            dice = 2 * tp / denom_dice if denom_dice > 0 else 0.0
            iou_per_class.append(iou)
            dice_per_class.append(dice)

        # mIoU over classes that had any ground truth OR predictions
        active = [c for c in range(self.num_classes)
                  if cm[c, :].sum() > 0 or cm[:, c].sum() > 0]
        miou = float(np.mean([iou_per_class[c] for c in active])) if active else 0.0

        return {
            "miou": miou,
            "iou_per_class": iou_per_class,
            "dice_per_class": dice_per_class,
            "pixel_accuracy": pixel_acc,
            "confusion_matrix": cm.copy(),
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/segmentation/test_metrics.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add segmentation/metrics.py tests/segmentation/test_metrics.py
git commit -m "$(cat <<'EOF'
feat(segmentation): add SegmentationMetrics (mIoU / Dice / pixAcc)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Write `config.yaml`

**Files:**
- Create: `segmentation/config.yaml`

- [ ] **Step 1: Write YAML**

Create `segmentation/config.yaml`:

```yaml
data:
  database_root: database
  num_workers: 4
  batch_size: 8
  image_size: 512

model:
  num_classes: 3                 # bg / normal fruit / canker fruit
  encoder_name: resnet34
  encoder_weights: imagenet

train:
  epochs: 50
  lr: 0.001
  weight_decay: 0.0001
  optimizer: adamw
  scheduler: cosine
  ce_weight: 0.5
  dice_weight: 0.5

eval:
  save_qualitative_every_n_epochs: 5
  num_qualitative_samples: 4

output:
  root: outputs/segmentation

seed: 42
device: auto
```

- [ ] **Step 2: Verify YAML loads**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "from common.config import load_config; c = load_config('segmentation/config.yaml'); print(c)"
```

- [ ] **Step 3: Commit**

```bash
git add segmentation/config.yaml
git commit -m "$(cat <<'EOF'
feat(segmentation): add training config.yaml

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Implement `train.py` (TDD via tiny smoke test)

**Files:**
- Create: `tests/segmentation/test_train.py`
- Create: `segmentation/train.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/segmentation/test_train.py`:

```python
"""Integration test: segmentation train runs one epoch on synthetic data."""
from pathlib import Path
import yaml
import pytest


@pytest.fixture
def mini_config(tmp_path, synthetic_dataset_root):
    out_root = tmp_path / "outputs"
    cfg = {
        "data": {
            "database_root": str(synthetic_dataset_root),
            "num_workers": 0,
            "batch_size": 2,
            "image_size": 64,  # tiny for speed
        },
        "model": {
            "num_classes": 3,
            "encoder_name": "resnet34",
            "encoder_weights": None,  # no download in tests
        },
        "train": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "ce_weight": 0.5,
            "dice_weight": 0.5,
        },
        "eval": {"save_qualitative_every_n_epochs": 100, "num_qualitative_samples": 0},
        "output": {"root": str(out_root)},
        "seed": 0,
        "device": "cpu",
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, out_root


def test_seg_train_runs_one_epoch(mini_config):
    cfg_path, out_root = mini_config
    from segmentation.train import main
    out_dir = main(str(cfg_path))
    assert Path(out_dir).exists()
    assert (Path(out_dir) / "ckpt" / "last.pt").exists()
    assert (Path(out_dir) / "ckpt" / "best.pt").exists()
```

- [ ] **Step 2: Run to verify fail**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && pytest tests/segmentation/test_train.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `train.py`**

Create `segmentation/train.py`:

```python
"""Training entrypoint for 3-class citrus semantic segmentation.

Usage:
    python -m segmentation.train --config segmentation/config.yaml
    python -m segmentation.train --config segmentation/config.yaml --override train.lr=0.0005
"""
from __future__ import annotations
import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.config import load_config, apply_overrides
from common.dataset import SegmentationDataset
from common.utils import set_seed, get_device, make_output_dir
from segmentation.transforms import build_transforms
from segmentation.model import build_model
from segmentation.losses import CombinedLoss
from segmentation.metrics import SegmentationMetrics


def _build_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("segmentation.train")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _collate(batch: list) -> dict:
    imgs = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"].long() for b in batch])
    paths = [b["image_path"] for b in batch]
    return {"image": imgs, "mask": masks, "image_path": paths}


def main(config_path: str, overrides: list | None = None) -> Path:
    cfg = apply_overrides(load_config(config_path), overrides or [])
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    out_dir = make_output_dir(cfg["output"]["root"], task="run")
    (out_dir / "ckpt").mkdir()
    (out_dir / "tb").mkdir()
    shutil.copy(config_path, out_dir / "config.yaml")

    logger = _build_logger(out_dir)
    logger.info(f"config: {cfg}")
    logger.info(f"device: {device}")
    logger.info(f"out_dir: {out_dir}")

    db = Path(cfg["data"]["database_root"])
    train_tf = build_transforms(cfg["data"]["image_size"], train=True)
    val_tf = build_transforms(cfg["data"]["image_size"], train=False)
    train_ds = SegmentationDataset(db, split="train", transform=train_tf)
    val_ds = SegmentationDataset(db, split="val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"],
                              shuffle=True,
                              num_workers=cfg["data"]["num_workers"],
                              collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["data"]["num_workers"],
                            collate_fn=_collate)

    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )
    loss_fn = CombinedLoss(
        num_classes=cfg["model"]["num_classes"],
        ce_weight=cfg["train"]["ce_weight"],
        dice_weight=cfg["train"]["dice_weight"],
    ).to(device)

    tb = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_miou = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tr_loss_sum = 0.0
        n_seen = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
        tr_loss = tr_loss_sum / max(n_seen, 1)
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        n_seen = 0
        metrics = SegmentationMetrics(num_classes=cfg["model"]["num_classes"])
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                val_loss_sum += loss.item() * imgs.size(0)
                n_seen += imgs.size(0)
                metrics.update(logits, masks)
        val_loss = val_loss_sum / max(n_seen, 1)
        m = metrics.compute()

        logger.info(
            f"epoch {epoch+1}/{cfg['train']['epochs']} "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"miou={m['miou']:.4f} pixAcc={m['pixel_accuracy']:.4f} "
            f"iou_classes={['{:.3f}'.format(x) for x in m['iou_per_class']]}"
        )
        tb.add_scalar("train/loss", tr_loss, epoch)
        tb.add_scalar("val/loss", val_loss, epoch)
        tb.add_scalar("val/miou", m["miou"], epoch)
        tb.add_scalar("val/pixel_accuracy", m["pixel_accuracy"], epoch)
        for i, iou in enumerate(m["iou_per_class"]):
            tb.add_scalar(f"val/iou_class_{i}", iou, epoch)
        tb.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": m,
            "config": cfg,
        }, out_dir / "ckpt" / "last.pt")
        if m["miou"] > best_miou:
            best_miou = m["miou"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": m,
                "config": cfg,
            }, out_dir / "ckpt" / "best.pt")
            logger.info(f"  ↑ new best miou={best_miou:.4f}")

    tb.close()
    logger.info("training complete")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.override)
```

- [ ] **Step 4: Run integration test**

```bash
pytest tests/segmentation/test_train.py -v
```

Expected: 1 passed (under ~60s on CPU).

- [ ] **Step 5: Commit**

```bash
git add segmentation/train.py tests/segmentation/test_train.py
git commit -m "$(cat <<'EOF'
feat(segmentation): add train.py with one-epoch integration test

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Implement `eval.py` (with qualitative viz)

**Files:**
- Create: `tests/segmentation/test_eval.py`
- Create: `segmentation/eval.py`

- [ ] **Step 1: Write failing test**

Create `tests/segmentation/test_eval.py`:

```python
"""Integration test: eval loads a segmentation checkpoint and writes metrics."""
import json
from pathlib import Path
import pytest


@pytest.fixture
def trained_output(tmp_path, synthetic_dataset_root):
    import yaml
    from segmentation.train import main as train_main

    out_root = tmp_path / "outputs"
    cfg = {
        "data": {
            "database_root": str(synthetic_dataset_root),
            "num_workers": 0,
            "batch_size": 2,
            "image_size": 64,
        },
        "model": {"num_classes": 3, "encoder_name": "resnet34", "encoder_weights": None},
        "train": {"epochs": 1, "lr": 0.001, "weight_decay": 0.0,
                  "optimizer": "adamw", "scheduler": "cosine",
                  "ce_weight": 0.5, "dice_weight": 0.5},
        "eval": {"save_qualitative_every_n_epochs": 100, "num_qualitative_samples": 0},
        "output": {"root": str(out_root)},
        "seed": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_dir = train_main(str(cfg_path))
    return out_dir, cfg_path


def test_seg_eval_produces_metrics(trained_output):
    out_dir, cfg_path = trained_output
    from segmentation.eval import main as eval_main
    result = eval_main(str(cfg_path), str(Path(out_dir) / "ckpt" / "best.pt"))
    assert "miou" in result
    assert "pixel_accuracy" in result
    assert "iou_per_class" in result
    assert len(result["iou_per_class"]) == 3


def test_seg_eval_writes_metrics_json_and_qualitative(trained_output):
    out_dir, cfg_path = trained_output
    from segmentation.eval import main as eval_main
    eval_main(str(cfg_path), str(Path(out_dir) / "ckpt" / "best.pt"),
              num_qualitative_samples=2)
    metrics_path = Path(out_dir) / "metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert "miou" in data

    # qualitative/: original | GT | pred overlay PNGs
    qdir = Path(out_dir) / "qualitative"
    assert qdir.exists()
    pngs = list(qdir.glob("*.png"))
    assert len(pngs) >= 2
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/segmentation/test_eval.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `eval.py`**

Create `segmentation/eval.py`:

```python
"""Evaluation entrypoint for semantic segmentation checkpoints.

Writes metrics.json and a few qualitative composite images
(original | GT mask | predicted mask) to <run_dir>/qualitative/.

Usage:
    python -m segmentation.eval --config segmentation/config.yaml \
        --ckpt outputs/segmentation/run/<timestamp>/ckpt/best.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.config import load_config
from common.dataset import SegmentationDataset
from common.utils import set_seed, get_device
from segmentation.transforms import build_transforms
from segmentation.model import build_model
from segmentation.metrics import SegmentationMetrics
from segmentation.train import _collate


CLASS_COLORS = np.array([
    [0, 0, 0],       # 0 = bg (black)
    [0, 255, 0],     # 1 = normal (green)
    [255, 0, 0],     # 2 = canker (red)
], dtype=np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def _denormalize(img: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization, return uint8 HxWx3 RGB."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = img.detach().cpu() * std + mean
    x = (x.clamp(0, 1) * 255).to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


def _save_qualitative(imgs: torch.Tensor, gts: torch.Tensor, preds: torch.Tensor,
                      out_dir: Path, max_samples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(imgs.size(0), max_samples)
    for i in range(n):
        rgb = _denormalize(imgs[i])
        gt_rgb = _mask_to_rgb(gts[i].detach().cpu().numpy())
        pred_rgb = _mask_to_rgb(preds[i].detach().cpu().numpy())
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, arr, title in zip(axes, (rgb, gt_rgb, pred_rgb),
                                  ("original", "GT", "pred")):
            ax.imshow(arr)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:03d}.png", dpi=100)
        plt.close(fig)


def main(config_path: str, ckpt_path: str,
         num_qualitative_samples: int | None = None) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_tf = build_transforms(cfg["data"]["image_size"], train=False)
    val_ds = SegmentationDataset(Path(cfg["data"]["database_root"]),
                                 split="val", transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["data"]["num_workers"],
                            collate_fn=_collate)

    metrics = SegmentationMetrics(num_classes=cfg["model"]["num_classes"])
    out_dir = Path(ckpt_path).parent.parent
    qdir = out_dir / "qualitative"

    q_budget = num_qualitative_samples if num_qualitative_samples is not None \
        else cfg["eval"].get("num_qualitative_samples", 0)
    q_saved = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            metrics.update(logits, masks)
            if q_saved < q_budget:
                remaining = q_budget - q_saved
                _save_qualitative(imgs, masks, preds, qdir, max_samples=remaining)
                q_saved += min(imgs.size(0), remaining)

    result = metrics.compute()
    json_result = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in result.items()}
    (out_dir / "metrics.json").write_text(json.dumps(json_result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--samples", type=int, default=None,
                        help="override num qualitative samples to save")
    args = parser.parse_args()
    r = main(args.config, args.ckpt, num_qualitative_samples=args.samples)
    printable = {k: v for k, v in r.items() if k != "confusion_matrix"}
    print(json.dumps(printable, indent=2))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/segmentation/test_eval.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add segmentation/eval.py tests/segmentation/test_eval.py
git commit -m "$(cat <<'EOF'
feat(segmentation): add eval.py with qualitative viz

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Real-data smoke training (3 epochs)

**Files:** (no source changes)

- [ ] **Step 1: Run 3-epoch training on real data**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && python -m segmentation.train --config segmentation/config.yaml --override train.epochs=3
```

Expected behavior:
- Downloads ResNet34 ImageNet weights on first run (~80 MB)
- Uses MPS
- Trains on 699 polygon-labeled images at 512×512, batch 8
- Creates `outputs/segmentation/run/<timestamp>/`
- Final train_loss decreasing; val mIoU > 0 after epoch 3

Time estimate: 3 epochs × ~87 batches × ~2s/batch ≈ 8–15 min.

If MPS fails, retry with `--override device=cpu` (may be much slower — up to 45 min — but acceptable as fallback).

If wall time exceeds 30 min on MPS or 60 min on CPU, STOP and report.

- [ ] **Step 2: Run eval on best checkpoint**

```bash
BEST=$(ls -td outputs/segmentation/run/*/ 2>/dev/null | head -1)ckpt/best.pt
echo "best: $BEST"
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && python -m segmentation.eval --config segmentation/config.yaml --ckpt "$BEST" --samples 4
```

Expected:
- Prints mIoU, pixel_accuracy, per-class IoU, per-class Dice
- Writes `metrics.json`
- Writes 4 qualitative composite PNGs in `<run>/qualitative/`

- [ ] **Step 3: Report findings** — do NOT commit outputs.

Report:
- Run directory path
- Wall time for 3 epochs
- First and last epoch train_loss, miou
- Final eval metrics (mIoU, per-class IoU, pixel accuracy)
- Qualitative samples saved count
- Any warnings

---

### Task 10: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update P3 section**

Read `README.md`, then:
1. Change `[ ] P3 — Segmentation` to `[x] P3 — Segmentation` in "Phase status"
2. Insert a new section right AFTER `## Training & Evaluation (P2 — Detection)` and BEFORE `## Phase status`:

```markdown
## Training & Evaluation (P3 — Segmentation)

3-class semantic segmentation (background / normal fruit / canker fruit).
Only the 787 polygon-labeled images (train 699 / val 88) are used.

```bash
# train
python -m segmentation.train --config segmentation/config.yaml

# quick smoke (3 epochs)
python -m segmentation.train --config segmentation/config.yaml --override train.epochs=3

# evaluate best checkpoint and save 4 qualitative samples
python -m segmentation.eval --config segmentation/config.yaml \
    --ckpt outputs/segmentation/run/<timestamp>/ckpt/best.pt --samples 4
```

Outputs in `outputs/segmentation/run/<timestamp>/`:
- `ckpt/best.pt`, `ckpt/last.pt`
- `train.log`, `tb/` (TensorBoard)
- `config.yaml` snapshot
- `metrics.json` (mIoU, per-class IoU, Dice, pixel accuracy, pixel confusion matrix)
- `qualitative/sample_XXX.png` — original | GT mask | pred mask side-by-side
```

- [ ] **Step 2: Run full test suite**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && pytest
```

Expected: P0 (24) + P1 (19) + P2 (13) + P3 (~21) ≈ 77 tests.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: mark P3 segmentation complete, add usage section

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Criteria (P3 complete when)

1. `pytest` passes green (P0 + P1 + P2 + P3 tests).
2. `python -m segmentation.train --config segmentation/config.yaml --override train.epochs=3` runs end-to-end on real data without crash.
3. `python -m segmentation.eval ...` writes `metrics.json` and 4 qualitative PNGs.
4. Git history: atomic commits per task.
5. README updated with P3 status and usage.

## Notes for Executor

- **Do not commit** `outputs/` — gitignored.
- On first real-data run, smp will download the ResNet34 ImageNet weights (~80 MB).
- Use `python -m segmentation.<module>` form to ensure `common` imports resolve from project root.
- Integration tests use `encoder_weights=None` and `device=cpu` — no downloads, no MPS required.
- If albumentations `A.Lambda` with `mask=None` raises a type error on your version, replace with a conditional BGR→RGB done inside `Dataset.__getitem__` — flag this as a concern if needed.
