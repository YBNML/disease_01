# P0 — Common Module + Env Setup + Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Citrus disease CV 프로젝트의 공통 인프라(공통 데이터 로딩/라벨 파싱/설정/유틸)를 TDD로 구축하고, 기존 코드를 정리하며, 재현 가능한 conda 환경을 준비한다.

**Architecture:** 단일 `common/` 패키지가 세 ML 태스크 모두에서 재사용되는 공통 기능을 제공. 태스크별 코드는 후속 plan(P1/P2/P3)에서 이 공통 모듈을 import해서 사용한다.

**Tech Stack:** Python 3.11, PyTorch (MPS), numpy, pandas, opencv-python, pillow, pyyaml, pytest, conda

**Prereq:** 설계 스펙 `docs/superpowers/specs/2026-04-17-citrus-disease-cv-design.md`

---

## File Structure

**Create:**
- `.gitignore`
- `environment.yml`
- `README.md` (stub)
- `common/__init__.py`
- `common/label_parser.py` — JSON ANTN_PT 문자열을 polygon/bbox/mask로 변환
- `common/config.py` — YAML 로더 + dataclass
- `common/utils.py` — seed, device, output dir
- `common/dataset.py` — ClassificationDataset, SegmentationDataset
- `tests/__init__.py`
- `tests/common/__init__.py`
- `tests/common/conftest.py` — 공통 fixture (synthetic JSON/이미지)
- `tests/common/test_label_parser.py`
- `tests/common/test_config.py`
- `tests/common/test_utils.py`
- `tests/common/test_dataset.py`

**Move to `_archive/`:**
- `main.py`, `backup.py`, `otsu.py`, `otsu2.py`, `test_model.pt`

**Delete:**
- `HF01_00FT_000001.jpg` (database에 동일 파일 존재)
- `_vis/`, `_vis_polygon.py` (시각화 임시 산출물)
- `.DS_Store`

**Not touched:** `willmer/` (사용자가 직접 옮김), `database/`

---

## Working Directory

모든 상대 경로는 `<project root>`(이하 프로젝트 루트) 기준.

---

### Task 1: Initialize git repo

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Initialize git**

```bash
cd <project root>
git init
```

Expected: `Initialized empty Git repository in .../disease_01/.git/`

- [ ] **Step 2: Write `.gitignore`**

Create `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/

# macOS
.DS_Store

# ML artifacts
*.pt
*.pth
*.ckpt
outputs/
checkpoints/
runs/

# Dataset (too large)
database/

# Detection working dirs
detection/data/

# Archive (kept local, not in version control)
_archive/

# IDE
.vscode/
.idea/

# Conda
*.lock

# Temp visualization
_vis/
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: init git repo and add .gitignore"
```

Expected: first commit created.

---

### Task 2: Archive old code

**Files:**
- Create: `_archive/` directory
- Move: `main.py`, `backup.py`, `otsu.py`, `otsu2.py`, `test_model.pt` → `_archive/`
- Delete: `HF01_00FT_000001.jpg`, `_vis/`, `_vis_polygon.py`, `.DS_Store`

- [ ] **Step 1: Create `_archive/` and move files**

```bash
cd <project root>
mkdir -p _archive
mv main.py backup.py otsu.py otsu2.py test_model.pt _archive/
```

Expected: `ls _archive/` shows 5 files.

- [ ] **Step 2: Delete unused artifacts**

```bash
rm -f HF01_00FT_000001.jpg .DS_Store
rm -rf _vis _vis_polygon.py
```

- [ ] **Step 3: Verify root is clean**

```bash
ls
```

Expected output (file order may vary): `_archive  database  docs  willmer` (plus `.gitignore`, `.git/`).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: archive legacy code and remove stale artifacts"
```

Note: `_archive/` is gitignored so the commit only records deletions of tracked paths. That's expected; the archive is kept locally on disk.

---

### Task 3: Create `environment.yml`

**Files:**
- Create: `environment.yml`

- [ ] **Step 1: Write `environment.yml`**

```yaml
name: disease_01
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - pyyaml
  - tqdm
  - pillow
  - opencv
  - pytest
  - pip:
      - torch
      - torchvision
      - torchmetrics
      - tensorboard
      - albumentations
      - ultralytics
      - segmentation-models-pytorch
      - timm
```

- [ ] **Step 2: Commit**

```bash
git add environment.yml
git commit -m "chore: add conda environment.yml for disease_01"
```

---

### Task 4: Create conda env and verify

**Files:** (no changes, environment setup only)

- [ ] **Step 1: Create env**

```bash
cd <project root>
conda env create -f environment.yml
```

Expected: env created at `/opt/homebrew/Caskroom/miniforge/base/envs/disease_01`. Will take several minutes.

- [ ] **Step 2: Activate env**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate disease_01
```

- [ ] **Step 3: Verify MPS availability**

```bash
python -c "import torch; print('torch:', torch.__version__); print('mps available:', torch.backends.mps.is_available())"
```

Expected: `mps available: True`

- [ ] **Step 4: Verify other key imports**

```bash
python -c "import cv2, numpy, yaml, torchvision, ultralytics, segmentation_models_pytorch as smp, albumentations; print('all imports OK')"
```

Expected: `all imports OK`

(No commit — environment is external to the repo.)

---

### Task 5: Create `common/` and `tests/` package skeletons

**Files:**
- Create: `common/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `tests/common/__init__.py` (empty)
- Create: `pytest.ini`

- [ ] **Step 1: Create directories and empty package markers**

```bash
cd <project root>
mkdir -p common tests/common
touch common/__init__.py tests/__init__.py tests/common/__init__.py
```

- [ ] **Step 2: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
```

- [ ] **Step 3: Run pytest to confirm setup**

```bash
pytest
```

Expected: `no tests ran` (or similar, exit code 5).

- [ ] **Step 4: Commit**

```bash
git add common/ tests/ pytest.ini
git commit -m "chore: scaffold common package and pytest config"
```

---

### Task 6: TDD — `label_parser.parse_antn_pt`

**Files:**
- Create: `tests/common/test_label_parser.py`
- Create: `common/label_parser.py`

- [ ] **Step 1: Write failing test**

Create `tests/common/test_label_parser.py`:

```python
import numpy as np
import pytest
from common.label_parser import parse_antn_pt


def test_parse_antn_pt_basic():
    s = "[10|20|30],[100|200|300]"
    pts = parse_antn_pt(s)
    assert pts.shape == (3, 2)
    assert pts.dtype == np.int32
    np.testing.assert_array_equal(pts, [[10, 100], [20, 200], [30, 300]])


def test_parse_antn_pt_single_point():
    s = "[5],[9]"
    pts = parse_antn_pt(s)
    assert pts.shape == (1, 2)
    np.testing.assert_array_equal(pts, [[5, 9]])


def test_parse_antn_pt_mismatched_lengths_raises():
    s = "[1|2|3],[10|20]"
    with pytest.raises(ValueError):
        parse_antn_pt(s)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/common/test_label_parser.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'common.label_parser'` (or similar).

- [ ] **Step 3: Implement `parse_antn_pt`**

Create `common/label_parser.py`:

```python
import numpy as np


def parse_antn_pt(pt_str: str) -> np.ndarray:
    """Parse AI Hub ANTN_PT string '[x1|x2|...],[y1|y2|...]' to (N, 2) int array."""
    xs_str, ys_str = pt_str.split("],[")
    xs = [int(v) for v in xs_str.strip("[]").split("|")]
    ys = [int(v) for v in ys_str.strip("[]").split("|")]
    if len(xs) != len(ys):
        raise ValueError(f"x/y length mismatch: {len(xs)} vs {len(ys)}")
    return np.array(list(zip(xs, ys)), dtype=np.int32)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/common/test_label_parser.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add common/label_parser.py tests/common/test_label_parser.py
git commit -m "feat(common): add parse_antn_pt with tests"
```

---

### Task 7: TDD — `label_parser.polygon_to_bbox` + `polygon_to_mask`

**Files:**
- Modify: `tests/common/test_label_parser.py`
- Modify: `common/label_parser.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/common/test_label_parser.py`:

```python
from common.label_parser import polygon_to_bbox, polygon_to_mask


def test_polygon_to_bbox():
    poly = np.array([[10, 20], [30, 15], [25, 40]], dtype=np.int32)
    bbox = polygon_to_bbox(poly)
    assert bbox == (10, 15, 30, 40)


def test_polygon_to_mask_square():
    poly = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.int32)
    mask = polygon_to_mask(poly, h=10, w=10)
    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    # interior pixel
    assert mask[4, 4] == 1
    # outside pixel
    assert mask[0, 0] == 0


def test_polygon_to_mask_empty_outside():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32)
    mask = polygon_to_mask(poly, h=5, w=5)
    assert mask.sum() >= 1
    assert mask.max() == 1
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/common/test_label_parser.py -v
```

Expected: FAIL on bbox/mask tests with `ImportError`.

- [ ] **Step 3: Implement both functions**

Append to `common/label_parser.py`:

```python
import cv2


def polygon_to_bbox(polygon: np.ndarray) -> tuple:
    """Return (x_min, y_min, x_max, y_max) from (N, 2) polygon."""
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def polygon_to_mask(polygon: np.ndarray, h: int, w: int) -> np.ndarray:
    """Rasterize polygon into a (H, W) uint8 binary mask (1 inside, 0 outside)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/common/test_label_parser.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add common/label_parser.py tests/common/test_label_parser.py
git commit -m "feat(common): add polygon_to_bbox and polygon_to_mask"
```

---

### Task 8: TDD — `label_parser.load_sample`

**Files:**
- Create: `tests/common/conftest.py`
- Modify: `tests/common/test_label_parser.py`
- Modify: `common/label_parser.py`

- [ ] **Step 1: Add shared fixture**

Create `tests/common/conftest.py`:

```python
import json
import numpy as np
import cv2
import pytest
from pathlib import Path


@pytest.fixture
def sample_json_with_polygon(tmp_path):
    """Write a synthetic AI Hub-style JSON label with polygon, return its path."""
    payload = {
        "Info": {
            "IMAGE_FILE_NM": "TEST_000001",
            "RSOLTN": "(1920,1080)",
            "CMRA_INFO": "samsung",
            "LCINFO": "F02",
            "IMAGE_OBTAIN_PLACE_TY": "노지",
            "GRWH_STEP_CODE": "6",
            "OCPRD": "08-05",
            "SPCIES_NM": "온주밀감",
        },
        "Annotations": {
            "ANTN_ID": "1",
            "ANTN_TY": "polygon",
            "OBJECT_CLASS_CODE": "감귤_궤양병",
            "ANTN_PT": "[100|200|300|200],[100|50|100|200]",
        },
        "Environment": {
            "SOLRAD_QY": "34.7",
            "AFR": "0",
            "TP": "29.6",
            "HD": "64.8",
            "SOIL_MITR": "74.9",
        },
    }
    p = tmp_path / "TEST_000001.json"
    p.write_text(json.dumps(payload, ensure_ascii=False))
    return p


@pytest.fixture
def sample_json_no_polygon(tmp_path):
    """JSON label with null annotations (no polygon)."""
    payload = {
        "Info": {
            "IMAGE_FILE_NM": "TEST_000002",
            "RSOLTN": "(1920,1080)",
            "CMRA_INFO": "Xiaomi",
            "LCINFO": "F19",
            "IMAGE_OBTAIN_PLACE_TY": "온실",
            "GRWH_STEP_CODE": "7",
            "OCPRD": "10-05",
            "SPCIES_NM": "온주밀감",
        },
        "Annotations": {
            "ANTN_ID": None,
            "ANTN_TY": None,
            "OBJECT_CLASS_CODE": "감귤_정상",
            "ANTN_PT": None,
        },
        "Environment": {
            "SOLRAD_QY": "20.1",
            "AFR": "0",
            "TP": "25.0",
            "HD": "60.0",
            "SOIL_MITR": "70.0",
        },
    }
    p = tmp_path / "TEST_000002.json"
    p.write_text(json.dumps(payload, ensure_ascii=False))
    return p


@pytest.fixture
def synthetic_dataset_root(tmp_path):
    """Create a minimal AI Hub-style directory tree with 2 normal + 2 canker images,
    where 1 of each has a polygon. Returns the DS root path."""
    root = tmp_path / "database"
    for split, split_dir in [("1.Training", "TS1.감귤"), ("2.Validation", "VS1.감귤")]:
        split_lbl = {"1.Training": "TL1.감귤", "2.Validation": "VL1.감귤"}[split]
        for cls_name, cls_code, dbyhs in [("열매_정상", "감귤_정상", "00"),
                                           ("열매_궤양병", "감귤_궤양병", "01")]:
            img_dir = root / split / "원천데이터" / split_dir / cls_name
            lbl_dir = root / split / "라벨링데이터" / split_lbl / cls_name
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            for i in range(2):
                stem = f"HF01_{dbyhs}FT_{i:06d}"
                # 64x64 solid-color image
                img = np.full((64, 64, 3), 200, dtype=np.uint8)
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)

                # polygon only for the first sample of each class
                has_poly = (i == 0)
                payload = {
                    "Info": {
                        "IMAGE_FILE_NM": stem,
                        "RSOLTN": "(64,64)",
                        "CMRA_INFO": "samsung",
                        "LCINFO": "F02",
                        "IMAGE_OBTAIN_PLACE_TY": "노지",
                        "GRWH_STEP_CODE": "6",
                        "OCPRD": "08-05",
                        "SPCIES_NM": "온주밀감",
                    },
                    "Annotations": {
                        "ANTN_ID": "1" if has_poly else None,
                        "ANTN_TY": "polygon" if has_poly else None,
                        "OBJECT_CLASS_CODE": cls_code,
                        "ANTN_PT": "[10|50|50|10],[10|10|50|50]" if has_poly else None,
                    },
                    "Environment": {
                        "SOLRAD_QY": "30.0", "AFR": "0",
                        "TP": "25.0", "HD": "60.0", "SOIL_MITR": "70.0",
                    },
                }
                (lbl_dir / f"{stem}.json").write_text(
                    json.dumps(payload, ensure_ascii=False)
                )
    return root
```

- [ ] **Step 2: Add failing test**

Append to `tests/common/test_label_parser.py`:

```python
from common.label_parser import load_sample


def test_load_sample_with_polygon(sample_json_with_polygon):
    s = load_sample(sample_json_with_polygon)
    assert s["class_code"] == "감귤_궤양병"
    assert s["has_polygon"] is True
    assert s["polygon"].shape == (4, 2)
    assert s["image_size"] == (1920, 1080)
    assert s["metadata"]["camera"] == "samsung"
    assert s["metadata"]["location"] == "F02"
    assert s["metadata"]["env"]["temp"] == 29.6


def test_load_sample_without_polygon(sample_json_no_polygon):
    s = load_sample(sample_json_no_polygon)
    assert s["class_code"] == "감귤_정상"
    assert s["has_polygon"] is False
    assert s["polygon"] is None
    assert s["metadata"]["env"]["humidity"] == 60.0
```

- [ ] **Step 3: Run tests to verify fail**

```bash
pytest tests/common/test_label_parser.py -v
```

Expected: new tests fail with `ImportError: cannot import name 'load_sample'`.

- [ ] **Step 4: Implement `load_sample`**

Append to `common/label_parser.py`:

```python
import json
from pathlib import Path


def _parse_resolution(rsoltn: str) -> tuple:
    """'(1920,1080)' -> (1920, 1080). Returns (0, 0) on parse failure."""
    try:
        cleaned = rsoltn.strip().strip("()")
        w, h = cleaned.split(",")
        return (int(w), int(h))
    except (ValueError, AttributeError):
        return (0, 0)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_sample(json_path) -> dict:
    """Load AI Hub JSON label and return a normalized dict."""
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    info = d.get("Info", {}) or {}
    ann = d.get("Annotations", {}) or {}
    env = d.get("Environment", {}) or {}

    pt_str = ann.get("ANTN_PT")
    has_polygon = pt_str is not None
    polygon = parse_antn_pt(pt_str) if has_polygon else None

    return {
        "json_path": json_path,
        "image_file_name": info.get("IMAGE_FILE_NM", json_path.stem),
        "class_code": ann.get("OBJECT_CLASS_CODE"),
        "has_polygon": has_polygon,
        "polygon": polygon,
        "image_size": _parse_resolution(info.get("RSOLTN", "")),
        "metadata": {
            "camera": info.get("CMRA_INFO"),
            "location": info.get("LCINFO"),
            "place_type": info.get("IMAGE_OBTAIN_PLACE_TY"),
            "growth_stage": info.get("GRWH_STEP_CODE"),
            "date": info.get("OCPRD"),
            "env": {
                "solar": _safe_float(env.get("SOLRAD_QY")),
                "rain": _safe_float(env.get("AFR")),
                "temp": _safe_float(env.get("TP")),
                "humidity": _safe_float(env.get("HD")),
                "soil_moisture": _safe_float(env.get("SOIL_MITR")),
            },
        },
    }
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/common/test_label_parser.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add common/label_parser.py tests/common/test_label_parser.py tests/common/conftest.py
git commit -m "feat(common): add load_sample with shared test fixtures"
```

---

### Task 9: TDD — `config.load_config`

**Files:**
- Create: `tests/common/test_config.py`
- Create: `common/config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/common/test_config.py`:

```python
import pytest
from common.config import load_config, apply_overrides


def test_load_config_parses_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "data:\n"
        "  batch_size: 32\n"
        "  image_size: 224\n"
        "train:\n"
        "  lr: 0.0001\n"
        "  epochs: 30\n"
        "seed: 42\n"
    )
    cfg = load_config(str(p))
    assert cfg["data"]["batch_size"] == 32
    assert cfg["train"]["lr"] == 0.0001
    assert cfg["seed"] == 42


def test_apply_overrides_dotted_path():
    cfg = {"train": {"lr": 0.0001, "epochs": 30}, "seed": 42}
    out = apply_overrides(cfg, ["train.lr=0.0005", "seed=7"])
    assert out["train"]["lr"] == 0.0005
    assert out["seed"] == 7
    assert out["train"]["epochs"] == 30


def test_apply_overrides_type_coercion():
    cfg = {"train": {"epochs": 30, "use_sampler": True}}
    out = apply_overrides(cfg, ["train.epochs=50", "train.use_sampler=false"])
    assert out["train"]["epochs"] == 50
    assert out["train"]["use_sampler"] is False


def test_apply_overrides_unknown_key_raises():
    cfg = {"train": {"lr": 0.0001}}
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.nonexistent=1"])
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/common/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `common/config.py`**

Create `common/config.py`:

```python
import yaml
from copy import deepcopy


def load_config(path: str) -> dict:
    """Load YAML file and return a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce(value: str):
    """Coerce a CLI override string to bool/int/float/str."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply 'dotted.path=value' overrides to cfg. Raises KeyError if the path
    does not already exist — overrides must patch known keys."""
    out = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        cursor = out
        for p in parts[:-1]:
            if not isinstance(cursor, dict) or p not in cursor:
                raise KeyError(f"override path not found: {key}")
            cursor = cursor[p]
        last = parts[-1]
        if not isinstance(cursor, dict) or last not in cursor:
            raise KeyError(f"override path not found: {key}")
        cursor[last] = _coerce(raw)
    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/common/test_config.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add common/config.py tests/common/test_config.py
git commit -m "feat(common): add YAML config loader with CLI overrides"
```

---

### Task 10: TDD — `utils` (seed, device, output dir)

**Files:**
- Create: `tests/common/test_utils.py`
- Create: `common/utils.py`

- [ ] **Step 1: Write failing tests**

Create `tests/common/test_utils.py`:

```python
import random
import re
import numpy as np
import torch
from common.utils import set_seed, get_device, make_output_dir


def test_set_seed_python_numpy_torch():
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_get_device_is_mps_or_cpu():
    dev = get_device()
    assert dev.type in ("mps", "cpu")


def test_get_device_forced_cpu():
    dev = get_device("cpu")
    assert dev.type == "cpu"


def test_make_output_dir_creates_timestamped_dir(tmp_path):
    out = make_output_dir(root=tmp_path, task="classification")
    assert out.exists() and out.is_dir()
    assert out.parent.name == "classification"
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", out.name)
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/common/test_utils.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `common/utils.py`**

Create `common/utils.py`:

```python
import os
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for python random, numpy, and torch (CPU + MPS if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Return torch.device for the requested backend. 'auto' → mps > cpu."""
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "mps":
        return torch.device("mps")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_output_dir(root, task: str) -> Path:
    """Create outputs/<task>/<timestamp>/ and return its Path."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(root) / task / ts
    out.mkdir(parents=True, exist_ok=False)
    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/common/test_utils.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add common/utils.py tests/common/test_utils.py
git commit -m "feat(common): add seed/device/output-dir helpers"
```

---

### Task 11: TDD — `ClassificationDataset`

**Files:**
- Create: `tests/common/test_dataset.py`
- Create: `common/dataset.py`

- [ ] **Step 1: Write failing tests**

Create `tests/common/test_dataset.py`:

```python
import torch
from common.dataset import ClassificationDataset


def test_classification_dataset_train_size(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="train", transform=None)
    # 2 normal + 2 canker in Training
    assert len(ds) == 4


def test_classification_dataset_val_size(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="val", transform=None)
    assert len(ds) == 4  # same count in synthetic fixture


def test_classification_dataset_item_shape(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="train", transform=None)
    sample = ds[0]
    assert "image" in sample
    assert "label" in sample
    assert "metadata" in sample
    # label is 0 (normal) or 1 (canker)
    assert sample["label"] in (0, 1)
    # image from fixture is 64x64x3
    assert sample["image"].shape == (64, 64, 3)


def test_classification_dataset_labels_balanced(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="train", transform=None)
    labels = [ds[i]["label"] for i in range(len(ds))]
    assert labels.count(0) == 2
    assert labels.count(1) == 2


def test_classification_dataset_metadata_populated(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="train", transform=None)
    meta = ds[0]["metadata"]
    assert meta["camera"] == "samsung"
    assert "env" in meta
    assert meta["env"]["temp"] == 25.0
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/common/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ClassificationDataset`**

Create `common/dataset.py`:

```python
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset

from .label_parser import load_sample


CLASS_NAME_TO_LABEL = {"감귤_정상": 0, "감귤_궤양병": 1}

SPLIT_DIRS = {
    "train": {
        "split": "1.Training",
        "img": "원천데이터/TS1.감귤",
        "lbl": "라벨링데이터/TL1.감귤",
    },
    "val": {
        "split": "2.Validation",
        "img": "원천데이터/VS1.감귤",
        "lbl": "라벨링데이터/VL1.감귤",
    },
}

CLASS_DIRS = ["열매_정상", "열매_궤양병"]


def _iter_label_files(database_root, split: str):
    """Yield (image_path, json_path) pairs for a given split across both classes."""
    root = Path(database_root)
    cfg = SPLIT_DIRS[split]
    split_dir = root / cfg["split"]
    for cls_dir in CLASS_DIRS:
        img_dir = split_dir / cfg["img"] / cls_dir
        lbl_dir = split_dir / cfg["lbl"] / cls_dir
        if not lbl_dir.exists():
            continue
        for jp in sorted(lbl_dir.iterdir()):
            if jp.suffix.lower() != ".json":
                continue
            stem = jp.stem
            # image filename may have any extension; take first match
            cands = list(img_dir.glob(f"{stem}.*"))
            if not cands:
                continue
            yield cands[0], jp


class ClassificationDataset(Dataset):
    """All images in the split, labeled 0 (정상) / 1 (궤양병) from folder names.
    Loads images as BGR numpy arrays (H, W, 3) uint8. Transform (if given) is
    called with the numpy image and may return anything (tensor/array)."""

    def __init__(self, database_root, split: str = "train", transform=None):
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.database_root = Path(database_root)
        self.split = split
        self.transform = transform
        self.items = list(_iter_label_files(self.database_root, split))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, json_path = self.items[idx]
        info = load_sample(json_path)
        label = CLASS_NAME_TO_LABEL[info["class_code"]]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {img_path}")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "metadata": info["metadata"],
            "image_path": str(img_path),
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/common/test_dataset.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add common/dataset.py tests/common/test_dataset.py
git commit -m "feat(common): add ClassificationDataset with metadata"
```

---

### Task 12: TDD — `SegmentationDataset`

**Files:**
- Modify: `tests/common/test_dataset.py`
- Modify: `common/dataset.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/common/test_dataset.py`:

```python
import numpy as np
from common.dataset import SegmentationDataset


def test_segmentation_dataset_filters_polygon_only(synthetic_dataset_root):
    ds = SegmentationDataset(synthetic_dataset_root, split="train", transform=None)
    # fixture has 1 normal + 1 canker with polygons (others null)
    assert len(ds) == 2


def test_segmentation_dataset_mask_shape_and_values(synthetic_dataset_root):
    ds = SegmentationDataset(synthetic_dataset_root, split="train", transform=None)
    sample = ds[0]
    assert sample["image"].shape == (64, 64, 3)
    assert sample["mask"].shape == (64, 64)
    assert sample["mask"].dtype == np.uint8
    # mask values must be in {0, 1, 2}: bg, normal fruit, canker fruit
    assert set(np.unique(sample["mask"]).tolist()).issubset({0, 1, 2})
    # something non-zero should exist (polygon was rasterized)
    assert sample["mask"].max() in (1, 2)


def test_segmentation_dataset_mask_class_matches_folder(synthetic_dataset_root):
    ds = SegmentationDataset(synthetic_dataset_root, split="train", transform=None)
    for i in range(len(ds)):
        s = ds[i]
        non_bg = np.unique(s["mask"][s["mask"] != 0]).tolist()
        # a single-class mask: either {1} for normal or {2} for canker
        assert non_bg in ([1], [2])
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/common/test_dataset.py -v
```

Expected: new tests fail with `ImportError: cannot import name 'SegmentationDataset'`.

- [ ] **Step 3: Implement `SegmentationDataset`**

Append to `common/dataset.py`:

```python
import numpy as np
from .label_parser import load_sample, polygon_to_mask


SEG_CLASS_FROM_CODE = {"감귤_정상": 1, "감귤_궤양병": 2}


class SegmentationDataset(Dataset):
    """Only images that have polygon labels. Produces a 3-class semantic mask:
    0 = background, 1 = 정상 감귤, 2 = 궤양병 감귤.
    Transform (if given) is called as transform(image, mask) and should return a
    dict {'image': ..., 'mask': ...} (albumentations style)."""

    def __init__(self, database_root, split: str = "train", transform=None):
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.database_root = Path(database_root)
        self.split = split
        self.transform = transform
        self.items = [
            (ip, jp) for ip, jp in _iter_label_files(self.database_root, split)
            if load_sample(jp)["has_polygon"]
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, json_path = self.items[idx]
        info = load_sample(json_path)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {img_path}")
        h, w = image.shape[:2]

        cls_value = SEG_CLASS_FROM_CODE[info["class_code"]]
        binary = polygon_to_mask(info["polygon"], h=h, w=w)  # 0/1
        mask = (binary * cls_value).astype(np.uint8)         # 0 or cls_value

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        return {
            "image": image,
            "mask": mask,
            "metadata": info["metadata"],
            "image_path": str(img_path),
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/common/test_dataset.py -v
```

Expected: 8 passed (5 classification + 3 segmentation).

- [ ] **Step 5: Commit**

```bash
git add common/dataset.py tests/common/test_dataset.py
git commit -m "feat(common): add SegmentationDataset with 3-class masks"
```

---

### Task 13: Smoke test against real database

**Files:**
- Create: `scripts/smoke_test_datasets.py`

- [ ] **Step 1: Write smoke-test script**

Create `scripts/smoke_test_datasets.py`:

```python
"""Smoke test: verify ClassificationDataset and SegmentationDataset work
against the real AI Hub data. Prints sizes and one sample from each."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.dataset import ClassificationDataset, SegmentationDataset

DB = ROOT / "database"

print("== ClassificationDataset ==")
for split in ("train", "val"):
    ds = ClassificationDataset(DB, split=split)
    print(f"  {split}: n={len(ds)}")
    s = ds[0]
    print(f"    sample image={s['image'].shape}, label={s['label']}, "
          f"camera={s['metadata']['camera']}")

print("\n== SegmentationDataset ==")
for split in ("train", "val"):
    ds = SegmentationDataset(DB, split=split)
    print(f"  {split}: n={len(ds)}")
    s = ds[0]
    import numpy as np
    print(f"    sample image={s['image'].shape}, mask={s['mask'].shape}, "
          f"unique={np.unique(s['mask']).tolist()}")

print("\nSmoke test OK.")
```

- [ ] **Step 2: Run smoke test**

```bash
cd <project root>
python scripts/smoke_test_datasets.py
```

Expected:
```
== ClassificationDataset ==
  train: n=3407
    sample image=(1080, 1920, 3), label=..., camera=...
  val: n=427
    sample image=(1080, 1920, 3), label=..., camera=...

== SegmentationDataset ==
  train: n=699
    sample image=(1080, 1920, 3), mask=(1080, 1920), unique=[0, <1 or 2>]
  val: n=88
    sample image=(1080, 1920, 3), mask=(1080, 1920), unique=[0, <1 or 2>]

Smoke test OK.
```

If the numbers differ significantly from spec (3407/427/699/88) — STOP and investigate.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test_datasets.py
git commit -m "test(common): add dataset smoke test against real data"
```

---

### Task 14: Write initial `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write stub README**

Create `README.md`:

````markdown
# Citrus Disease CV

감귤(온주밀감) 정상/궤양병 감별을 위한 Classification / Detection / Segmentation 파이프라인.

## Structure

```
disease_01/
├── common/             # 공통 모듈 (dataset, label_parser, config, utils)
├── classification/     # P1 (TBD)
├── detection/          # P2 (TBD)
├── segmentation/       # P3 (TBD)
├── database/           # AI Hub 데이터 (gitignored, 로컬만)
├── docs/superpowers/   # 설계 spec / 구현 plan
├── _archive/           # 기존 코드 보관 (gitignored)
└── environment.yml
```

## Environment

```bash
conda env create -f environment.yml
conda activate disease_01
```

## Run tests

```bash
pytest
```

## Smoke test against real data

```bash
python scripts/smoke_test_datasets.py
```

## Design

- Spec: `docs/superpowers/specs/2026-04-17-citrus-disease-cv-design.md`
- Plans: `docs/superpowers/plans/`

## Phase status

- [x] P0 — Common module
- [ ] P1 — Classification
- [ ] P2 — Detection
- [ ] P3 — Segmentation
````

- [ ] **Step 2: Run full test suite as final sanity check**

```bash
pytest
```

Expected: all tests pass (label_parser 8, config 4, utils 4, dataset 8 = 24 total; numbers may shift slightly).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add initial README"
```

---

## Acceptance Criteria (P0 complete when)

1. `pytest` passes green for `tests/common/` (no skipped tests).
2. `python scripts/smoke_test_datasets.py` prints sizes matching the spec (3407 / 427 / 699 / 88).
3. `conda activate disease_01 && python -c "import torch; assert torch.backends.mps.is_available()"` succeeds.
4. Project root contains only: `common/`, `tests/`, `scripts/`, `database/`, `docs/`, `_archive/`, `willmer/` (if user hasn't moved it), `.gitignore`, `environment.yml`, `pytest.ini`, `README.md`.
5. Git history shows atomic commits (one per task).

## Notes for Executor

- Working directory: `<project root>`
- Conda env `disease_01` must be activated for all Python execution after Task 4.
- `_archive/` and `database/` are gitignored and should not appear in commits.
- If Task 4 (`conda env create`) fails due to resolver issues, try creating an empty env and installing via `pip install -r requirements.txt` as a fallback — but do not skip MPS verification in Step 3.
- Do NOT touch `willmer/` — the user is relocating it manually.
