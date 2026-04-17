# P2 — Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ultralytics YOLOv8s 기반 감귤 열매 정상/궤양병 탐지 파이프라인을 구축하고, 실제 데이터로 smoke 학습까지 검증한다.

**Architecture:** AI Hub polygon 라벨을 YOLO txt 포맷으로 변환하는 `prepare_yolo.py` 스크립트와, ultralytics의 자체 trainer/validator를 얇게 감싼 `train.py` / `eval.py`로 구성. 데이터 변환 로직은 TDD로 검증하고, ultralytics 래퍼는 실데이터 smoke 학습으로 통합 검증.

**Tech Stack:** ultralytics YOLOv8, PyTorch (MPS), PyYAML, numpy, pytest

**Prereq:**
- P0, P1 완료
- Spec §7, `common.label_parser.polygon_to_bbox` 활용

---

## File Structure

**Create:**
- `detection/__init__.py`
- `detection/yolo_format.py` — `polygon_to_yolo_bbox(polygon, img_w, img_h)` 변환 함수
- `detection/prepare_yolo.py` — JSON 라벨을 YOLO 포맷으로 변환하는 CLI 스크립트
- `detection/config.yaml` — 학습 설정
- `detection/train.py` — ultralytics YOLO 학습 래퍼
- `detection/eval.py` — ultralytics YOLO 평가 래퍼
- `tests/detection/__init__.py`
- `tests/detection/conftest.py` — `synthetic_dataset_root` 재노출
- `tests/detection/test_yolo_format.py`
- `tests/detection/test_prepare_yolo.py`
- `tests/detection/test_train.py`
- `tests/detection/test_eval.py`

**Modify:**
- `README.md` — P2 체크박스

**Runtime outputs (gitignored):**
- `detection/data/{train,val}/{images,labels}/` — prepare_yolo 결과
- `detection/data/data.yaml` — ultralytics가 읽는 데이터 정의
- `outputs/detection/<timestamp>/` — 학습/평가 결과

---

## Working Directory & Environment

- Root: `<project root>`
- Activation:
  ```bash
  source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE
  ```

---

### Task 1: Scaffold `detection/` package

**Files:**
- Create: `detection/__init__.py` (empty)
- Create: `tests/detection/__init__.py` (empty)
- Create: `tests/detection/conftest.py`

- [ ] **Step 1: Create dirs and markers**

```bash
cd <project root>
mkdir -p detection tests/detection
touch detection/__init__.py tests/detection/__init__.py
```

- [ ] **Step 2: Create conftest.py to share fixtures**

Create `tests/detection/conftest.py`:

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

Expected: 43 passed (P0+P1 state).

- [ ] **Step 4: Commit**

```bash
git add detection/ tests/detection/
git commit -m "$(cat <<'EOF'
chore: scaffold detection package

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: TDD — `polygon_to_yolo_bbox`

**Files:**
- Create: `tests/detection/test_yolo_format.py`
- Create: `detection/yolo_format.py`

- [ ] **Step 1: Write failing tests**

Create `tests/detection/test_yolo_format.py`:

```python
import numpy as np
import pytest
from detection.yolo_format import polygon_to_yolo_bbox


def test_polygon_to_yolo_bbox_basic():
    # polygon covers a 40×40 square from (100,100) to (140,140)
    # in a 1000×1000 image
    poly = np.array([[100, 100], [140, 100], [140, 140], [100, 140]], dtype=np.int32)
    x, y, w, h = polygon_to_yolo_bbox(poly, img_w=1000, img_h=1000)
    # x_center = 120/1000 = 0.12
    assert abs(x - 0.12) < 1e-6
    # y_center = 120/1000 = 0.12
    assert abs(y - 0.12) < 1e-6
    # w = 40/1000 = 0.04
    assert abs(w - 0.04) < 1e-6
    # h = 40/1000 = 0.04
    assert abs(h - 0.04) < 1e-6


def test_polygon_to_yolo_bbox_non_square_image():
    poly = np.array([[0, 0], [960, 0], [960, 540], [0, 540]], dtype=np.int32)
    x, y, w, h = polygon_to_yolo_bbox(poly, img_w=1920, img_h=1080)
    # Full half of the image starting from top-left
    # bbox (0,0)-(960,540); x_center = 480/1920 = 0.25; y_center = 270/1080 = 0.25
    # w = 960/1920 = 0.5; h = 540/1080 = 0.5
    assert abs(x - 0.25) < 1e-6
    assert abs(y - 0.25) < 1e-6
    assert abs(w - 0.5) < 1e-6
    assert abs(h - 0.5) < 1e-6


def test_polygon_to_yolo_bbox_clamped_to_unit_interval():
    """Edge coords should stay within [0, 1]."""
    poly = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.int32)
    x, y, w, h = polygon_to_yolo_bbox(poly, img_w=1920, img_h=1080)
    for v in (x, y, w, h):
        assert 0.0 <= v <= 1.0


def test_polygon_to_yolo_bbox_zero_dimension_raises():
    """A degenerate polygon should raise a clear error."""
    poly = np.array([[100, 100], [100, 100]], dtype=np.int32)  # single point
    with pytest.raises(ValueError):
        polygon_to_yolo_bbox(poly, img_w=1000, img_h=1000)
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/detection/test_yolo_format.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement conversion**

Create `detection/yolo_format.py`:

```python
"""Convert AI Hub polygon to YOLO-format bounding box."""
import numpy as np

from common.label_parser import polygon_to_bbox


def polygon_to_yolo_bbox(polygon: np.ndarray, img_w: int, img_h: int) -> tuple:
    """Return (x_center, y_center, w, h) normalized to [0, 1] in YOLO format.

    Raises ValueError if the polygon has zero width or height in either axis."""
    x_min, y_min, x_max, y_max = polygon_to_bbox(polygon)
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w <= 0 or box_h <= 0:
        raise ValueError(
            f"degenerate polygon: bbox {x_min, y_min, x_max, y_max}"
        )
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    w = box_w / img_w
    h = box_h / img_h
    return (float(x_center), float(y_center), float(w), float(h))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/detection/test_yolo_format.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add detection/yolo_format.py tests/detection/test_yolo_format.py
git commit -m "$(cat <<'EOF'
feat(detection): add polygon_to_yolo_bbox normalization

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: TDD — `prepare_yolo` builder

**Files:**
- Create: `tests/detection/test_prepare_yolo.py`
- Create: `detection/prepare_yolo.py`

- [ ] **Step 1: Write failing tests**

Create `tests/detection/test_prepare_yolo.py`:

```python
from pathlib import Path
import yaml
import pytest
from detection.prepare_yolo import prepare_split, prepare_all, CLASS_NAMES


def test_prepare_split_train_creates_structure(synthetic_dataset_root, tmp_path):
    dest = tmp_path / "yolo_data"
    n = prepare_split(synthetic_dataset_root, dest, split="train")
    # synthetic fixture has 1 normal + 1 canker polygon = 2 samples
    assert n == 2

    images = list((dest / "train" / "images").iterdir())
    labels = list((dest / "train" / "labels").iterdir())
    assert len(images) == 2
    assert len(labels) == 2

    # each label is one line: "<cls> <x> <y> <w> <h>"
    for lf in labels:
        lines = lf.read_text().strip().splitlines()
        assert len(lines) == 1
        parts = lines[0].split()
        assert len(parts) == 5
        cls = int(parts[0])
        coords = [float(p) for p in parts[1:]]
        assert cls in (0, 1)
        assert all(0.0 <= c <= 1.0 for c in coords)


def test_prepare_split_labels_match_classes(synthetic_dataset_root, tmp_path):
    dest = tmp_path / "yolo_data"
    prepare_split(synthetic_dataset_root, dest, split="train")

    # 열매_정상 = class 0, 열매_궤양병 = class 1
    # (as defined by CLASS_NAMES order)
    for lf in (dest / "train" / "labels").iterdir():
        cls = int(lf.read_text().split()[0])
        if "00FT" in lf.stem:  # normal fixture stem
            assert cls == CLASS_NAMES.index("normal")
        elif "01FT" in lf.stem:  # canker fixture stem
            assert cls == CLASS_NAMES.index("canker")


def test_prepare_all_creates_data_yaml(synthetic_dataset_root, tmp_path):
    dest = tmp_path / "yolo_data"
    summary = prepare_all(synthetic_dataset_root, dest)
    assert summary["train"] == 2
    assert summary["val"] == 2

    yaml_path = dest / "data.yaml"
    assert yaml_path.exists()
    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["names"] == CLASS_NAMES
    assert doc["train"] == "train/images"
    assert doc["val"] == "val/images"
    assert Path(doc["path"]).resolve() == dest.resolve()


def test_prepare_split_skips_images_without_polygon(synthetic_dataset_root, tmp_path):
    """Synthetic fixture has 4 images per split total; only 2 have polygons."""
    dest = tmp_path / "yolo_data"
    n = prepare_split(synthetic_dataset_root, dest, split="val")
    assert n == 2
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/detection/test_prepare_yolo.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `prepare_yolo.py`**

Create `detection/prepare_yolo.py`:

```python
"""Convert AI Hub citrus polygon labels into YOLO training format.

Usage (as a script):
    python detection/prepare_yolo.py --source database --dest detection/data

Produces:
    <dest>/train/images/*.jpg
    <dest>/train/labels/*.txt
    <dest>/val/images/*.jpg
    <dest>/val/labels/*.txt
    <dest>/data.yaml

Only images that have polygon annotations are copied (~787 out of 3834).
Each YOLO label file contains one line (one box per image):
    <class_id> <x_center> <y_center> <w> <h>   # all normalized to [0, 1]
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import yaml

from common.dataset import SPLIT_DIRS, _iter_label_files
from common.label_parser import load_sample
from detection.yolo_format import polygon_to_yolo_bbox


# Order defines class_id: index 0 → 정상, index 1 → 궤양병
CLASS_NAMES = ["normal", "canker"]
CLASS_CODE_TO_ID = {"감귤_정상": 0, "감귤_궤양병": 1}


def prepare_split(database_root, dest_root, split: str) -> int:
    """Build <dest_root>/<split>/{images,labels}/ from polygon-labeled samples.
    Returns the number of samples written."""
    database_root = Path(database_root)
    dest_root = Path(dest_root)
    img_dir = dest_root / split / "images"
    lbl_dir = dest_root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_path, json_path in _iter_label_files(database_root, split):
        info = load_sample(json_path)
        if not info["has_polygon"]:
            continue
        img_w, img_h = info["image_size"]
        if img_w <= 0 or img_h <= 0:
            continue
        try:
            xc, yc, w, h = polygon_to_yolo_bbox(info["polygon"], img_w, img_h)
        except ValueError:
            continue  # skip degenerate polygons

        cls_id = CLASS_CODE_TO_ID[info["class_code"]]
        stem = json_path.stem
        # copy image using the original extension
        shutil.copy2(img_path, img_dir / img_path.name)
        (lbl_dir / f"{stem}.txt").write_text(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        n += 1
    return n


def prepare_all(database_root, dest_root) -> dict:
    """Build both train and val splits, plus data.yaml."""
    dest_root = Path(dest_root)
    summary = {}
    for split in ("train", "val"):
        summary[split] = prepare_split(database_root, dest_root, split)

    yaml_path = dest_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(dest_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": CLASS_NAMES,
    }, sort_keys=False))
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="AI Hub database root")
    p.add_argument("--dest", required=True, help="Output YOLO data directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = prepare_all(args.source, args.dest)
    print(f"train: {summary['train']}  val: {summary['val']}")
    print(f"wrote {Path(args.dest) / 'data.yaml'}")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/detection/test_prepare_yolo.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add detection/prepare_yolo.py tests/detection/test_prepare_yolo.py
git commit -m "$(cat <<'EOF'
feat(detection): add prepare_yolo script for AI Hub→YOLO conversion

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Run `prepare_yolo` on real data

**Files:** (no source changes)

- [ ] **Step 1: Generate YOLO dataset from real data**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && cd <project root> && python detection/prepare_yolo.py --source database --dest detection/data
```

Expected stdout: `train: 699  val: 88`.

- [ ] **Step 2: Verify structure and counts**

```bash
ls detection/data/train/images | wc -l
ls detection/data/train/labels | wc -l
ls detection/data/val/images | wc -l
ls detection/data/val/labels | wc -l
cat detection/data/data.yaml
head -1 detection/data/train/labels/*.txt | head -3
```

Expected:
- train/images and train/labels both 699
- val/images and val/labels both 88
- data.yaml has correct names/paths

Acceptance: counts match 699 / 88 exactly (same as P0 polygon totals).

- [ ] **Step 3: Do NOT commit `detection/data/`** — it's gitignored.

No commit for this task; just a runtime verification.

---

### Task 5: Write `detection/config.yaml`

**Files:**
- Create: `detection/config.yaml`

- [ ] **Step 1: Write config**

Create `detection/config.yaml`:

```yaml
data:
  data_yaml: detection/data/data.yaml

model:
  name: yolov8s.pt       # pretrained on COCO
  imgsz: 640

train:
  epochs: 100
  batch: 16
  lr0: 0.01
  workers: 4
  patience: 30           # early stopping patience
  device: auto           # auto | mps | cpu | 0 (cuda)

output:
  project: outputs/detection
  name: run              # becomes outputs/detection/run/

seed: 42
```

- [ ] **Step 2: Verify YAML loads**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && cd <project root> && python -c "from common.config import load_config; c = load_config('detection/config.yaml'); print(c)"
```

Expected: prints the nested dict without errors.

- [ ] **Step 3: Commit**

```bash
git add detection/config.yaml
git commit -m "$(cat <<'EOF'
feat(detection): add YOLOv8 training config.yaml

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Implement `train.py`

**Files:**
- Create: `tests/detection/test_train.py`
- Create: `detection/train.py`

- [ ] **Step 1: Write failing test (import-only)**

The training wrapper is a thin adapter over ultralytics; unit-testing it end-to-end would download weights and take minutes, so the test only verifies that the module imports, exposes `main(config_path, overrides)`, and that config parsing works.

Create `tests/detection/test_train.py`:

```python
import pytest
import yaml


def test_train_module_imports():
    from detection import train
    assert hasattr(train, "main")


def test_train_build_ultralytics_kwargs_from_config(tmp_path):
    from detection.train import build_ultralytics_kwargs
    cfg = {
        "data": {"data_yaml": "detection/data/data.yaml"},
        "model": {"name": "yolov8s.pt", "imgsz": 640},
        "train": {"epochs": 5, "batch": 16, "lr0": 0.01, "workers": 4,
                  "patience": 30, "device": "mps"},
        "output": {"project": "outputs/detection", "name": "run"},
        "seed": 42,
    }
    kwargs = build_ultralytics_kwargs(cfg)
    assert kwargs["data"] == "detection/data/data.yaml"
    assert kwargs["epochs"] == 5
    assert kwargs["batch"] == 16
    assert kwargs["imgsz"] == 640
    assert kwargs["device"] == "mps"
    assert kwargs["project"] == "outputs/detection"
    assert kwargs["name"] == "run"
    assert kwargs["seed"] == 42


def test_train_device_auto_resolves_to_mps_or_cpu(tmp_path):
    from detection.train import build_ultralytics_kwargs
    cfg = {
        "data": {"data_yaml": "x"},
        "model": {"name": "yolov8s.pt", "imgsz": 640},
        "train": {"epochs": 1, "batch": 1, "lr0": 0.01, "workers": 0,
                  "patience": 10, "device": "auto"},
        "output": {"project": "o", "name": "r"},
        "seed": 0,
    }
    kwargs = build_ultralytics_kwargs(cfg)
    assert kwargs["device"] in ("mps", "cpu")
```

- [ ] **Step 2: Run to verify fail**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && pytest tests/detection/test_train.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement `detection/train.py`**

Create `detection/train.py`:

```python
"""Training entrypoint for YOLOv8 citrus detection.

Usage:
    python detection/train.py --config detection/config.yaml
    python detection/train.py --config detection/config.yaml --override train.epochs=5
"""
from __future__ import annotations
import argparse
from pathlib import Path

from common.config import load_config, apply_overrides
from common.utils import set_seed, get_device


def _resolve_device(device_cfg: str) -> str:
    """ultralytics accepts 'mps' | 'cpu' | '0' (cuda). 'auto' → mps or cpu."""
    if device_cfg != "auto":
        return device_cfg
    return "mps" if get_device("auto").type == "mps" else "cpu"


def build_ultralytics_kwargs(cfg: dict) -> dict:
    """Flatten our config into keyword args accepted by YOLO(...).train(...)."""
    device = _resolve_device(cfg["train"]["device"])
    return {
        "data": cfg["data"]["data_yaml"],
        "epochs": cfg["train"]["epochs"],
        "batch": cfg["train"]["batch"],
        "imgsz": cfg["model"]["imgsz"],
        "lr0": cfg["train"]["lr0"],
        "workers": cfg["train"]["workers"],
        "patience": cfg["train"]["patience"],
        "device": device,
        "project": cfg["output"]["project"],
        "name": cfg["output"]["name"],
        "seed": cfg["seed"],
        "exist_ok": True,  # allow re-runs into same name (ultralytics will increment)
    }


def main(config_path: str, overrides: list | None = None):
    cfg = apply_overrides(load_config(config_path), overrides or [])
    set_seed(cfg["seed"])

    # Import ultralytics lazily so `import detection.train` doesn't pull the
    # whole ultralytics stack (keeps test startup fast).
    from ultralytics import YOLO

    kwargs = build_ultralytics_kwargs(cfg)
    model = YOLO(cfg["model"]["name"])
    results = model.train(**kwargs)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.override)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/detection/test_train.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add detection/train.py tests/detection/test_train.py
git commit -m "$(cat <<'EOF'
feat(detection): add YOLOv8 train.py wrapper

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Implement `eval.py`

**Files:**
- Create: `tests/detection/test_eval.py`
- Create: `detection/eval.py`

- [ ] **Step 1: Write failing test (import + config only)**

Create `tests/detection/test_eval.py`:

```python
def test_eval_module_imports():
    from detection import eval as deval
    assert hasattr(deval, "main")


def test_eval_build_val_kwargs():
    from detection.eval import build_val_kwargs
    cfg = {
        "data": {"data_yaml": "detection/data/data.yaml"},
        "model": {"name": "yolov8s.pt", "imgsz": 640},
        "train": {"batch": 16, "workers": 4, "device": "mps"},
        "output": {"project": "outputs/detection", "name": "val"},
    }
    kwargs = build_val_kwargs(cfg)
    assert kwargs["data"] == "detection/data/data.yaml"
    assert kwargs["imgsz"] == 640
    assert kwargs["device"] == "mps"
    assert kwargs["project"] == "outputs/detection"
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/detection/test_eval.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `detection/eval.py`**

Create `detection/eval.py`:

```python
"""Evaluation wrapper for a YOLOv8 detection checkpoint.

Usage:
    python detection/eval.py --config detection/config.yaml \
        --ckpt outputs/detection/run/weights/best.pt
"""
from __future__ import annotations
import argparse

from common.config import load_config
from detection.train import _resolve_device


def build_val_kwargs(cfg: dict) -> dict:
    device = _resolve_device(cfg["train"]["device"])
    return {
        "data": cfg["data"]["data_yaml"],
        "imgsz": cfg["model"]["imgsz"],
        "batch": cfg["train"]["batch"],
        "workers": cfg["train"]["workers"],
        "device": device,
        "project": cfg["output"]["project"],
        "name": cfg["output"]["name"] + "_eval",
        "exist_ok": True,
    }


def main(config_path: str, ckpt_path: str):
    cfg = load_config(config_path)

    from ultralytics import YOLO
    model = YOLO(ckpt_path)
    kwargs = build_val_kwargs(cfg)
    results = model.val(**kwargs)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    main(args.config, args.ckpt)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/detection/test_eval.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add detection/eval.py tests/detection/test_eval.py
git commit -m "$(cat <<'EOF'
feat(detection): add YOLOv8 eval.py wrapper

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Real-data smoke training (5 epochs)

**Files:** (no source changes)

- [ ] **Step 1: Run YOLOv8 smoke training**

First-run caveat: ultralytics will download `yolov8s.pt` (~20 MB) on first use. Allow ~30s for this.

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && cd <project root> && python detection/train.py --config detection/config.yaml --override train.epochs=5
```

Expected behavior:
- Downloads yolov8s.pt on first run
- Uses MPS (auto-detected)
- Trains on 699 images, validates on 88, 5 epochs total
- Creates `outputs/detection/run/` (or `run2`, `run3`... if already exists) containing:
  - `weights/best.pt`, `weights/last.pt`
  - `results.csv`, `results.png`
  - `confusion_matrix.png`, various P/R curves
- Final stdout shows mAP@0.5 and mAP@0.5:0.95

Time estimate: 5 epochs × 699 images at 640 imgsz on M4 MPS ≈ 10–25 minutes total (YOLOv8 is more expensive per image than ResNet50 classification).

If ultralytics crashes on MPS (known: some older versions fall back to CPU automatically; should still complete), report as DONE_WITH_CONCERNS with the actual device used. If it hangs >40 minutes, STOP and report.

- [ ] **Step 2: Run eval on best checkpoint**

Ultralytics writes checkpoints to `outputs/detection/run/weights/best.pt` (or the incremented run dir). Find the latest run dir and point eval at it:

```bash
BEST=$(ls -td outputs/detection/run*/weights/best.pt 2>/dev/null | head -1)
echo "best: $BEST"
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && cd <project root> && python detection/eval.py --config detection/config.yaml --ckpt "$BEST"
```

Expected: ultralytics prints mAP@0.5, mAP@0.5:0.95, per-class AP. Creates `outputs/detection/run_eval/` with metrics artifacts.

- [ ] **Step 3: Report findings**

Do NOT commit `outputs/` or `detection/data/` (both gitignored).

Report:
- Run directory path
- Wall time for 5 epochs
- Final mAP@0.5 and mAP@0.5:0.95
- Per-class AP (normal vs canker)
- Any warnings (device fallback, NaN, etc.)

---

### Task 9: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update P2 section**

Read `README.md`, then:
1. Change `[ ] P2 — Detection` to `[x] P2 — Detection` in "Phase status"
2. Insert a new section right AFTER `## Training & Evaluation (P1)` and BEFORE `## Phase status`:

```markdown
## Training & Evaluation (P2 — Detection)

```bash
# one-time: convert AI Hub polygons to YOLO format (~787 images)
python detection/prepare_yolo.py --source database --dest detection/data

# train
python detection/train.py --config detection/config.yaml

# quick smoke (5 epochs)
python detection/train.py --config detection/config.yaml --override train.epochs=5

# evaluate best checkpoint
python detection/eval.py --config detection/config.yaml \
    --ckpt outputs/detection/run/weights/best.pt
```

Outputs are written by ultralytics to `outputs/detection/run*/`:
- `weights/best.pt`, `weights/last.pt`
- `results.csv`, `results.png`
- `confusion_matrix.png`, P/R curves
```

- [ ] **Step 2: Run full test suite**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && cd <project root> && pytest
```

Expected: all P0+P1+P2 tests pass. P2 adds 4 + 4 + 3 + 2 = 13 tests → total ~56.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: mark P2 detection complete, add usage section

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Criteria (P2 complete when)

1. `pytest` passes green (P0 + P1 + P2 tests).
2. `python detection/prepare_yolo.py --source database --dest detection/data` produces 699/88 split correctly.
3. `python detection/train.py --config detection/config.yaml --override train.epochs=5` runs to completion on MPS (or falls back cleanly to CPU).
4. `python detection/eval.py --config detection/config.yaml --ckpt <best.pt>` reports mAP metrics.
5. Git history: atomic commit per task.
6. README updated with P2 status and usage.

## Notes for Executor

- **Do not commit** `detection/data/` or `outputs/` — both gitignored.
- Ultralytics downloads `yolov8s.pt` on first run; allow time.
- If MPS has issues with YOLOv8 (rare but possible), override `train.device=cpu` as fallback; still acceptable for smoke run.
- Ultralytics auto-increments run dir names (`run`, `run2`, `run3`, ...). Use `ls -td` in bash to find latest.
- YOLOv8 has its own augmentation (mosaic, mixup, etc.) — do not add our own transforms.
