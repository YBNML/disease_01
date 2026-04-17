# P1 — Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ResNet50 기반 감귤 정상/궤양병 이진 분류 파이프라인을 `common/` 공통 모듈 위에 구축하고, 실제 데이터로 2-epoch smoke 학습까지 검증한다.

**Architecture:** `classification/` 하위에 transforms / model / metrics / config / train / eval 모듈을 분리. 공통 데이터 로딩은 `common.dataset.ClassificationDataset` 재사용. YAML config + argparse로 실험 파라미터 관리. TensorBoard + 파일 로거. Best checkpoint는 val F1 (궤양병) 기준.

**Tech Stack:** PyTorch, torchvision ResNet50, torchmetrics, PyYAML, TensorBoard, numpy, pytest

**Prereq:**
- P0 완료 (`common/` 모듈 동작, MPS 검증, env `disease_01` 준비됨)
- Spec: `docs/superpowers/specs/2026-04-17-citrus-disease-cv-design.md` §6

---

## File Structure

**Create:**
- `classification/__init__.py`
- `classification/config.yaml` — 학습 설정
- `classification/transforms.py` — `build_transforms(image_size, train)` 팩토리
- `classification/model.py` — `build_model(num_classes, pretrained)` 팩토리
- `classification/sampler.py` — `build_weighted_sampler(labels)` for class imbalance
- `classification/metrics.py` — `ClassificationMetrics` 집계기 (accuracy, F1 per-class, confusion)
- `classification/train.py` — 학습 엔트리포인트
- `classification/eval.py` — 평가 엔트리포인트
- `tests/classification/__init__.py`
- `tests/classification/test_transforms.py`
- `tests/classification/test_model.py`
- `tests/classification/test_sampler.py`
- `tests/classification/test_metrics.py`

**Modify:**
- `README.md` — P1 체크박스 갱신

---

## Working Directory & Environment

- Root: `<project root>`
- Activate env + KMP workaround before any Python command:
  ```bash
  source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE
  ```

---

### Task 1: Scaffold `classification/` package

**Files:**
- Create: `classification/__init__.py` (empty)
- Create: `tests/classification/__init__.py` (empty)

- [ ] **Step 1: Create dirs + markers**

```bash
cd <project root>
mkdir -p classification tests/classification
touch classification/__init__.py tests/classification/__init__.py
```

- [ ] **Step 2: Confirm pytest still collects**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && pytest
```

Expected: 24 passed (same as P0 final).

- [ ] **Step 3: Commit**

```bash
git add classification/ tests/classification/
git commit -m "$(cat <<'EOF'
chore: scaffold classification package

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: TDD — transforms

**Files:**
- Create: `tests/classification/test_transforms.py`
- Create: `classification/transforms.py`

- [ ] **Step 1: Write failing tests**

Create `tests/classification/test_transforms.py`:

```python
import numpy as np
import torch
from classification.transforms import build_transforms


def test_train_transform_output_shape():
    t = build_transforms(image_size=224, train=True)
    img = (np.random.rand(500, 400, 3) * 255).astype(np.uint8)
    out = t(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_val_transform_output_shape():
    t = build_transforms(image_size=224, train=False)
    img = (np.random.rand(500, 400, 3) * 255).astype(np.uint8)
    out = t(img)
    assert out.shape == (3, 224, 224)


def test_val_transform_is_deterministic():
    """Val transform has no randomness — same input → same output."""
    t = build_transforms(image_size=224, train=False)
    img = (np.random.rand(500, 400, 3) * 255).astype(np.uint8)
    out1 = t(img)
    out2 = t(img)
    torch.testing.assert_close(out1, out2)


def test_train_transform_is_random():
    """Train transform includes randomness — two calls likely differ."""
    import random
    random.seed(0)
    torch.manual_seed(0)
    t = build_transforms(image_size=224, train=True)
    img = (np.random.rand(500, 400, 3) * 255).astype(np.uint8)
    out1 = t(img)
    out2 = t(img)
    # extremely unlikely to be identical with flip+colorjitter
    assert not torch.allclose(out1, out2)


def test_normalization_stats_applied():
    """After ImageNet normalization, pure white (255) input produces well-known values."""
    t = build_transforms(image_size=32, train=False)  # small size to speed up
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    out = t(white)
    # BGR input (from cv2) becomes RGB after transform; all channels should be
    # (1.0 - mean) / std per ImageNet constants
    expected_ch0 = (1.0 - 0.485) / 0.229  # R
    assert abs(out[0].mean().item() - expected_ch0) < 1e-3
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/classification/test_transforms.py -v
```

Expected: `ModuleNotFoundError: No module named 'classification.transforms'`.

- [ ] **Step 3: Implement transforms**

Create `classification/transforms.py`:

```python
"""Image transforms for classification training and validation.

Input is a BGR numpy array (H, W, 3) uint8 from cv2.imread. The pipeline
converts BGR→RGB, then applies torchvision transforms to produce a normalized
float tensor of shape (3, image_size, image_size).
"""
from typing import Callable
import cv2
import numpy as np
import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_transforms(image_size: int = 224, train: bool = True) -> Callable:
    """Return a callable that converts a BGR uint8 numpy image into a
    normalized float tensor of shape (3, image_size, image_size)."""
    resize_short = int(round(image_size * 256 / 224))  # keep 256/224 ratio
    if train:
        pipeline = T.Compose([
            T.Lambda(_bgr_to_rgb),
            T.ToPILImage(),
            T.Resize(resize_short),
            T.CenterCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ColorJitter(0.1, 0.1, 0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        pipeline = T.Compose([
            T.Lambda(_bgr_to_rgb),
            T.ToPILImage(),
            T.Resize(resize_short),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return pipeline
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/classification/test_transforms.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add classification/transforms.py tests/classification/test_transforms.py
git commit -m "$(cat <<'EOF'
feat(classification): add BGR→RGB+normalize transform factory

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: TDD — model builder

**Files:**
- Create: `tests/classification/test_model.py`
- Create: `classification/model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/classification/test_model.py`:

```python
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
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/classification/test_model.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement model builder**

Create `classification/model.py`:

```python
"""ResNet50 model builder for binary citrus disease classification."""
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Return torchvision ResNet50 with the final FC replaced for `num_classes`."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/classification/test_model.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add classification/model.py tests/classification/test_model.py
git commit -m "$(cat <<'EOF'
feat(classification): add ResNet50 model builder

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: TDD — WeightedRandomSampler builder

**Files:**
- Create: `tests/classification/test_sampler.py`
- Create: `classification/sampler.py`

- [ ] **Step 1: Write failing tests**

Create `tests/classification/test_sampler.py`:

```python
import numpy as np
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from classification.sampler import build_weighted_sampler, compute_class_weights


def test_compute_class_weights_inverse_frequency():
    labels = [0, 0, 0, 1]  # 3 normal, 1 canker
    w = compute_class_weights(labels)
    assert w.shape == (2,)
    # class 0 has 3 samples → weight 1/3; class 1 has 1 sample → weight 1
    assert abs(w[0] - 1 / 3) < 1e-6
    assert abs(w[1] - 1.0) < 1e-6


def test_build_weighted_sampler_returns_sampler():
    labels = [0, 0, 0, 1, 1]
    sampler = build_weighted_sampler(labels)
    assert isinstance(sampler, WeightedRandomSampler)
    assert len(sampler) == len(labels)


def test_build_weighted_sampler_draws_balanced():
    """With enough draws, minority class should appear close to 50%."""
    labels = [0] * 100 + [1] * 20  # heavy imbalance
    sampler = build_weighted_sampler(labels)
    # Draw len(labels)*10 samples from a long sampler to check the distribution
    long_sampler = build_weighted_sampler(labels, num_samples=10_000)
    indices = list(iter(long_sampler))
    drawn_labels = [labels[i] for i in indices]
    counts = Counter(drawn_labels)
    # After weighting, both classes should be roughly equal (within 10%)
    ratio = counts[1] / (counts[0] + counts[1])
    assert 0.4 < ratio < 0.6
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/classification/test_sampler.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement sampler module**

Create `classification/sampler.py`:

```python
"""Weighted sampling for imbalanced binary classification."""
from typing import Sequence, Optional
import numpy as np
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(labels: Sequence[int]) -> np.ndarray:
    """Return per-class weight = 1 / class_count."""
    labels_arr = np.asarray(labels)
    num_classes = int(labels_arr.max()) + 1
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    # avoid div-by-zero if a class is missing (shouldn't happen here)
    counts[counts == 0] = 1.0
    return 1.0 / counts


def build_weighted_sampler(
    labels: Sequence[int],
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """WeightedRandomSampler for class-balanced sampling.
    `num_samples` defaults to len(labels) (one epoch worth)."""
    class_w = compute_class_weights(labels)
    sample_w = class_w[np.asarray(labels)]
    if num_samples is None:
        num_samples = len(labels)
    return WeightedRandomSampler(
        weights=sample_w.tolist(),
        num_samples=num_samples,
        replacement=True,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/classification/test_sampler.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add classification/sampler.py tests/classification/test_sampler.py
git commit -m "$(cat <<'EOF'
feat(classification): add weighted sampler for class imbalance

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: TDD — metrics aggregator

**Files:**
- Create: `tests/classification/test_metrics.py`
- Create: `classification/metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/classification/test_metrics.py`:

```python
import numpy as np
import torch
from classification.metrics import ClassificationMetrics


def test_metrics_perfect_prediction():
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0]])
    labels = torch.tensor([0, 1, 0])
    m.update(logits, labels)
    result = m.compute()
    assert result["accuracy"] == 1.0
    assert result["f1_positive"] == 1.0
    # 2 true negatives, 1 true positive
    cm = result["confusion_matrix"]
    assert cm[0, 0] == 2 and cm[1, 1] == 1
    assert cm[0, 1] == 0 and cm[1, 0] == 0


def test_metrics_all_wrong():
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    logits = torch.tensor([[-10.0, 10.0], [10.0, -10.0]])
    labels = torch.tensor([0, 1])
    m.update(logits, labels)
    result = m.compute()
    assert result["accuracy"] == 0.0
    assert result["f1_positive"] == 0.0


def test_metrics_mixed_batch_f1():
    """2 TP, 1 FP, 1 FN → precision=2/3, recall=2/3, F1=2/3 on positive class."""
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    # pred positive for samples 0,1,2; pred negative for sample 3
    # true positive for samples 0,1,3; true negative for sample 2
    # so: 0=TP, 1=TP, 2=FP, 3=FN
    logits = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    labels = torch.tensor([1, 1, 0, 1])
    m.update(logits, labels)
    result = m.compute()
    assert abs(result["f1_positive"] - 2 / 3) < 1e-6


def test_metrics_reset():
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    logits = torch.tensor([[10.0, -10.0]])
    labels = torch.tensor([0])
    m.update(logits, labels)
    m.reset()
    # After reset, computing with no data should not raise but may return NaN / 0
    # We at least require no exception
    result = m.compute()
    assert "accuracy" in result


def test_metrics_auc_binary():
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    # rank-correct scores: positives get higher score → AUC = 1
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 3.0]])
    labels = torch.tensor([0, 1, 0, 1])
    m.update(logits, labels)
    result = m.compute()
    assert abs(result["auc"] - 1.0) < 1e-6
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/classification/test_metrics.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement metrics aggregator**

Create `classification/metrics.py`:

```python
"""Accumulating metrics for binary classification evaluation."""
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
)


class ClassificationMetrics:
    """Accumulates per-batch predictions and computes summary metrics on demand.

    Designed for binary classification (num_classes=2) but confusion matrix
    handles arbitrary num_classes. `positive_class` identifies which class is
    the 'positive' for precision/recall/F1/AUC (default: 1 = 궤양병)."""

    def __init__(self, num_classes: int = 2, positive_class: int = 1):
        self.num_classes = num_classes
        self.positive_class = positive_class
        self.reset()

    def reset(self) -> None:
        self._labels = []
        self._preds = []
        self._scores = []  # softmax probability for positive_class

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate a batch. `logits` shape (B, C), `labels` shape (B,)."""
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        self._labels.extend(labels.detach().cpu().tolist())
        self._preds.extend(preds.detach().cpu().tolist())
        self._scores.extend(probs[:, self.positive_class].detach().cpu().tolist())

    def compute(self) -> Dict:
        if not self._labels:
            return {"accuracy": 0.0, "f1_positive": 0.0, "auc": 0.0,
                    "confusion_matrix": np.zeros((self.num_classes, self.num_classes),
                                                 dtype=np.int64)}
        y_true = np.asarray(self._labels)
        y_pred = np.asarray(self._preds)
        y_score = np.asarray(self._scores)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(self.num_classes)), zero_division=0
        )
        # AUC requires both classes present
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = float("nan")

        return {
            "accuracy": float(acc),
            "precision_per_class": prec.tolist(),
            "recall_per_class": rec.tolist(),
            "f1_per_class": f1.tolist(),
            "f1_positive": float(f1[self.positive_class]),
            "precision_positive": float(prec[self.positive_class]),
            "recall_positive": float(rec[self.positive_class]),
            "auc": float(auc),
            "confusion_matrix": cm,
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/classification/test_metrics.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add classification/metrics.py tests/classification/test_metrics.py
git commit -m "$(cat <<'EOF'
feat(classification): add ClassificationMetrics aggregator

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Write `config.yaml`

**Files:**
- Create: `classification/config.yaml`

- [ ] **Step 1: Write YAML**

Create `classification/config.yaml`:

```yaml
data:
  database_root: ../database
  num_workers: 4
  batch_size: 32
  image_size: 224

model:
  num_classes: 2
  pretrained: true

train:
  epochs: 30
  lr: 0.0001
  weight_decay: 0.0001
  optimizer: adamw
  scheduler: cosine
  use_weighted_sampler: true

eval:
  save_misclassified: true
  save_qualitative_every_n_epochs: 5

output:
  root: ../outputs/classification

seed: 42
device: auto
```

- [ ] **Step 2: Verify YAML loads**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "from common.config import load_config; c = load_config('classification/config.yaml'); print(c)"
```

Expected: prints the nested dict without errors.

- [ ] **Step 3: Commit**

```bash
git add classification/config.yaml
git commit -m "$(cat <<'EOF'
feat(classification): add training config.yaml

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Implement `train.py` (TDD via tiny smoke test)

**Files:**
- Create: `classification/train.py`
- Create: `tests/classification/test_train.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/classification/test_train.py`:

```python
"""Integration test: train.py runs one epoch on synthetic data without errors."""
import shutil
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
            "image_size": 32,  # tiny for speed
        },
        "model": {"num_classes": 2, "pretrained": False},  # no download in test
        "train": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "use_weighted_sampler": True,
        },
        "eval": {"save_misclassified": False, "save_qualitative_every_n_epochs": 100},
        "output": {"root": str(out_root)},
        "seed": 0,
        "device": "cpu",  # tests must not require MPS
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, out_root


def test_train_runs_one_epoch(mini_config, monkeypatch):
    """Smoke: train.main(cfg_path) runs one epoch on synthetic data, produces ckpt."""
    cfg_path, out_root = mini_config
    from classification.train import main
    exit_dir = main(str(cfg_path))
    # returns the output dir it created
    assert Path(exit_dir).exists()
    # ckpt/last.pt must exist
    assert (Path(exit_dir) / "ckpt" / "last.pt").exists()
```

Note: the `synthetic_dataset_root` fixture is in `tests/common/conftest.py`. To reuse it, we need to tell pytest where to find it. Add a `conftest.py` at `tests/classification/` that imports the fixture:

Create `tests/classification/conftest.py`:

```python
# Share fixtures with tests/common/
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

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/classification/test_train.py -v
```

Expected: `ImportError` from `classification.train`.

- [ ] **Step 3: Implement `train.py`**

Create `classification/train.py`:

```python
"""Training entrypoint for citrus disease classification.

Usage:
    python classification/train.py --config classification/config.yaml
    python classification/train.py --config classification/config.yaml --override train.lr=0.0005

The script creates outputs/<task>/<timestamp>/ containing:
- ckpt/last.pt, ckpt/best.pt
- tb/ (tensorboard logs)
- train.log
- config.yaml (snapshot)
"""
from __future__ import annotations
import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.config import load_config, apply_overrides
from common.dataset import ClassificationDataset
from common.utils import set_seed, get_device, make_output_dir
from classification.transforms import build_transforms
from classification.model import build_model
from classification.sampler import build_weighted_sampler
from classification.metrics import ClassificationMetrics


def _build_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("classification.train")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers on re-entry (e.g. tests)
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


def _build_optimizer(model, cfg):
    name = cfg["train"]["optimizer"].lower()
    if name != "adamw":
        raise ValueError(f"unsupported optimizer: {name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )


def _build_scheduler(optimizer, cfg, num_epochs):
    name = cfg["train"]["scheduler"].lower()
    if name != "cosine":
        raise ValueError(f"unsupported scheduler: {name}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


def _labels_from_dataset(ds: ClassificationDataset) -> list:
    """Extract all labels from a ClassificationDataset without reading images.
    Uses label_parser directly on the cached json paths."""
    from common.label_parser import load_sample
    from common.dataset import CLASS_NAME_TO_LABEL
    return [CLASS_NAME_TO_LABEL[load_sample(jp)["class_code"]]
            for _, jp in ds.items]


def main(config_path: str, overrides: list | None = None) -> Path:
    cfg = apply_overrides(load_config(config_path), overrides or [])

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    # outputs/<task>/<timestamp>/
    out_dir = make_output_dir(cfg["output"]["root"], task="run")
    (out_dir / "ckpt").mkdir()
    (out_dir / "tb").mkdir()
    shutil.copy(config_path, out_dir / "config.yaml")

    logger = _build_logger(out_dir)
    logger.info(f"config: {cfg}")
    logger.info(f"device: {device}")
    logger.info(f"out_dir: {out_dir}")

    # datasets
    db = Path(cfg["data"]["database_root"])
    train_tf = build_transforms(cfg["data"]["image_size"], train=True)
    val_tf = build_transforms(cfg["data"]["image_size"], train=False)
    train_ds = ClassificationDataset(db, split="train", transform=train_tf)
    val_ds = ClassificationDataset(db, split="val", transform=val_tf)

    # sampler for imbalance
    if cfg["train"]["use_weighted_sampler"]:
        labels = _labels_from_dataset(train_ds)
        sampler = build_weighted_sampler(labels)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"],
        sampler=sampler, shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        collate_fn=_collate,
    )

    model = build_model(cfg["model"]["num_classes"], cfg["model"]["pretrained"]).to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, cfg["train"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    tb = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_f1 = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss_sum = 0.0
        n_seen = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
        train_loss = train_loss_sum / max(n_seen, 1)
        scheduler.step()

        model.eval()
        metrics = ClassificationMetrics(num_classes=cfg["model"]["num_classes"])
        val_loss_sum = 0.0
        n_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                n_seen += imgs.size(0)
                metrics.update(logits, labels)
        val_loss = val_loss_sum / max(n_seen, 1)
        m = metrics.compute()

        logger.info(
            f"epoch {epoch+1}/{cfg['train']['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={m['accuracy']:.4f} val_f1_pos={m['f1_positive']:.4f} "
            f"val_auc={m['auc']:.4f}"
        )
        tb.add_scalar("train/loss", train_loss, epoch)
        tb.add_scalar("val/loss", val_loss, epoch)
        tb.add_scalar("val/accuracy", m["accuracy"], epoch)
        tb.add_scalar("val/f1_positive", m["f1_positive"], epoch)
        tb.add_scalar("val/auc", m["auc"], epoch)
        tb.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": m,
            "config": cfg,
        }, out_dir / "ckpt" / "last.pt")
        if m["f1_positive"] > best_f1:
            best_f1 = m["f1_positive"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": m,
                "config": cfg,
            }, out_dir / "ckpt" / "best.pt")
            logger.info(f"  ↑ new best f1_positive={best_f1:.4f}")

    tb.close()
    logger.info("training complete")
    return out_dir


def _collate(batch: list) -> dict:
    """Stack {'image': tensor, 'label': int, ...} dicts into batched tensors."""
    imgs = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    paths = [b["image_path"] for b in batch]
    return {"image": imgs, "label": labels, "image_path": paths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.override)
```

- [ ] **Step 4: Run integration test**

```bash
pytest tests/classification/test_train.py -v
```

Expected: 1 passed (completes one epoch on synthetic 8-image dataset).

- [ ] **Step 5: Commit**

```bash
git add classification/train.py tests/classification/test_train.py tests/classification/conftest.py
git commit -m "$(cat <<'EOF'
feat(classification): add train.py with one-epoch integration test

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Implement `eval.py`

**Files:**
- Create: `classification/eval.py`
- Create: `tests/classification/test_eval.py`

- [ ] **Step 1: Write failing test**

Create `tests/classification/test_eval.py`:

```python
"""Integration test: eval.py loads a checkpoint and produces a metrics.json."""
import json
from pathlib import Path
import pytest


@pytest.fixture
def trained_output(tmp_path, synthetic_dataset_root):
    """Train one epoch on synthetic data to produce a checkpoint to evaluate."""
    import yaml
    from classification.train import main as train_main

    out_root = tmp_path / "outputs"
    cfg = {
        "data": {
            "database_root": str(synthetic_dataset_root),
            "num_workers": 0,
            "batch_size": 2,
            "image_size": 32,
        },
        "model": {"num_classes": 2, "pretrained": False},
        "train": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "use_weighted_sampler": True,
        },
        "eval": {"save_misclassified": False, "save_qualitative_every_n_epochs": 100},
        "output": {"root": str(out_root)},
        "seed": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_dir = train_main(str(cfg_path))
    return out_dir, cfg_path


def test_eval_produces_metrics_json(trained_output):
    out_dir, cfg_path = trained_output
    from classification.eval import main as eval_main
    result = eval_main(str(cfg_path), str(Path(out_dir) / "ckpt" / "best.pt"))
    assert "accuracy" in result
    assert "f1_positive" in result
    assert "confusion_matrix" in result


def test_eval_writes_confusion_matrix_image(trained_output, tmp_path):
    out_dir, cfg_path = trained_output
    from classification.eval import main as eval_main
    eval_main(str(cfg_path), str(Path(out_dir) / "ckpt" / "best.pt"))
    cm_path = Path(out_dir) / "confusion_matrix.png"
    assert cm_path.exists()
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/classification/test_eval.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `eval.py`**

Create `classification/eval.py`:

```python
"""Evaluation entrypoint: load a checkpoint, compute metrics on val split,
save confusion_matrix.png and metrics.json alongside the checkpoint.

Usage:
    python classification/eval.py --config classification/config.yaml \
        --ckpt outputs/classification/<run>/ckpt/best.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.config import load_config
from common.dataset import ClassificationDataset
from common.utils import set_seed, get_device
from classification.transforms import build_transforms
from classification.model import build_model
from classification.metrics import ClassificationMetrics
from classification.train import _collate


def _save_confusion_matrix(cm: np.ndarray, accuracy: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["normal", "canker"],
                yticklabels=["normal", "canker"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (acc={accuracy:.2%})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(config_path: str, ckpt_path: str) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg["model"]["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_tf = build_transforms(cfg["data"]["image_size"], train=False)
    val_ds = ClassificationDataset(Path(cfg["data"]["database_root"]),
                                   split="val", transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"],
                            collate_fn=_collate)

    metrics = ClassificationMetrics(num_classes=cfg["model"]["num_classes"])
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(imgs)
            metrics.update(logits, labels)
    result = metrics.compute()

    out_dir = Path(ckpt_path).parent.parent  # ckpt/ lives inside run dir
    cm = np.asarray(result["confusion_matrix"])
    _save_confusion_matrix(cm, result["accuracy"], out_dir / "confusion_matrix.png")

    # JSON-safe version (convert ndarray to list)
    json_result = {**result, "confusion_matrix": cm.tolist()}
    (out_dir / "metrics.json").write_text(json.dumps(json_result, indent=2))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    r = main(args.config, args.ckpt)
    print(json.dumps({k: v for k, v in r.items() if k != "confusion_matrix"}, indent=2))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/classification/test_eval.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add classification/eval.py tests/classification/test_eval.py
git commit -m "$(cat <<'EOF'
feat(classification): add eval.py with confusion matrix output

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Real-data smoke training run (2 epochs)

**Files:** (no source changes)

- [ ] **Step 1: Run 2-epoch training on real data**

Use YAML override to shrink epochs. Keep batch_size and image_size at config defaults. Use MPS (default). Command:

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && python classification/train.py --config classification/config.yaml --override train.epochs=2
```

Expected:
- Completes without crash
- `outputs/classification/<timestamp>/ckpt/best.pt` exists (~100MB, ResNet50)
- `train.log` shows monotonically decreasing train_loss and val_f1_pos > 0 after epoch 2
- `tb/` has tensorboard event file

Acceptance: val F1 on positive class > 0 (very low bar — just ensure the pipeline runs and produces predictions, not that the 2-epoch model is good).

If anything crashes (MPS op unsupported, memory issue, etc.) — STOP and report as BLOCKED with full traceback.

- [ ] **Step 2: Run eval on that checkpoint**

```bash
BEST=$(ls -td outputs/classification/*/ | head -1)ckpt/best.pt
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && python classification/eval.py --config classification/config.yaml --ckpt "$BEST"
```

Expected: prints accuracy/F1/AUC to stdout; creates `confusion_matrix.png` and `metrics.json` in the same run dir.

- [ ] **Step 3: Do NOT commit outputs/ (they are gitignored)**

Just note in the report:
- run directory path
- key metrics (accuracy, F1 positive, AUC)
- train log summary (first and last epoch losses)

---

### Task 10: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update P1 checkbox and add usage section**

Edit `README.md`:
- Change `[ ] P1 — Classification` to `[x] P1 — Classification`
- Add a new section right before `## Phase status`:

```markdown
## Training & Evaluation (P1)

```bash
# train
python classification/train.py --config classification/config.yaml

# override epochs via CLI
python classification/train.py --config classification/config.yaml --override train.epochs=10

# evaluate best checkpoint
python classification/eval.py --config classification/config.yaml \
    --ckpt outputs/classification/<run>/ckpt/best.pt

# tensorboard
tensorboard --logdir outputs/classification
```
```

- [ ] **Step 2: Run full test suite as final sanity**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && pytest
```

Expected: all tests pass (P0 24 + P1 added tests: 5 transforms + 3 model + 3 sampler + 5 metrics + 1 train + 2 eval = 43 total; exact count may differ slightly).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: mark P1 classification complete, add usage section

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Criteria (P1 complete when)

1. `pytest` passes green (all P0 + P1 tests).
2. `python classification/train.py --config classification/config.yaml --override train.epochs=2` runs to completion on MPS without error.
3. `python classification/eval.py --config classification/config.yaml --ckpt <run>/ckpt/best.pt` produces `confusion_matrix.png` and `metrics.json`.
4. Git history shows atomic commits per task.
5. README updated with P1 status and usage.

## Notes for Executor

- **Do not** commit `outputs/` — it's gitignored.
- Use `cpu` device for pytest integration tests (MPS not guaranteed to be fast for tiny synthetic batches; consistency matters more than speed).
- The `synthetic_dataset_root` fixture is shared from `tests/common/conftest.py` via a re-export in `tests/classification/conftest.py`.
- If the integration tests take longer than ~30s per, consider reducing `image_size` further or confirming the test actually uses `num_workers=0`.
- MPS-specific: `torch.utils.tensorboard` writer should work on MPS; no special handling needed.
