# P1b — Classification Architecture Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 5개 classification 백본(ResNet50 / EfficientNet-B0 / ConvNeXt-Tiny / MobileNetV3-Large / ViT-Small/16)을 동일 조건으로 학습·평가·추론 벤치마킹해서 성능 / 추론속도 / 파라미터 수를 비교한다.

**Architecture:** P1의 `classification/model.py`에 `timm` 백엔드 추가. 공통 벤치마크 유틸(`classification/benchmark.py`)로 FPS·latency·param 측정. `classification/compare.py`가 모델 목록을 순회하며 학습 → 평가 → 벤치마크 → CSV/Markdown 보고서 생성.

**Tech Stack:** PyTorch (MPS), timm (이미 env에 있음), torchvision, numpy, pandas, pytest

**Prereq:**
- P0 / P1 완료
- Spec "Phase 2 — 모델 아키텍처 비교" 섹션

---

## File Structure

**Create:**
- `classification/benchmark.py` — 추론 속도/파라미터 측정 유틸
- `classification/compare.py` — 다중 모델 학습·평가·벤치마크 러너
- `classification/compare_config.yaml` — 비교 실험 config
- `tests/classification/test_benchmark.py`
- `tests/classification/test_compare.py`

**Modify:**
- `classification/model.py` — `timm` 백엔드 지원 추가 (`model_name` 파라미터)
- `tests/classification/test_model.py` — 기존 assertion을 백엔드 중립적으로 조정
- `classification/config.yaml` — `model.name` 필드 추가 (선택사항, 하위 호환)
- `README.md` — 비교 실험 섹션 추가

---

## Working Directory & Environment

- Root: `<project root>`
- Activation:
  ```bash
  source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE
  ```

---

### Task 1: Extend `build_model` to support timm backbones

**Files:**
- Modify: `classification/model.py`
- Modify: `tests/classification/test_model.py`

- [ ] **Step 1: Update tests to support both backends**

Edit `tests/classification/test_model.py` to this EXACT content (replace the existing file):

```python
import torch
import pytest
from classification.model import build_model


def test_resnet50_output_shape():
    model = build_model(name="resnet50", num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2)


def test_default_resnet50_param_count():
    model = build_model(name="resnet50", num_classes=2, pretrained=False)
    n = sum(p.numel() for p in model.parameters())
    # ResNet50 is ~25M params
    assert 20_000_000 < n < 30_000_000


@pytest.mark.parametrize("name", [
    "efficientnet_b0",
    "convnext_tiny",
    "mobilenetv3_large_100",
    "vit_small_patch16_224",
])
def test_timm_backbones_build_and_infer(name):
    model = build_model(name=name, num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2)


def test_build_model_raises_on_unknown_name():
    with pytest.raises(Exception):
        build_model(name="definitely_not_a_real_model", num_classes=2, pretrained=False)
```

- [ ] **Step 2: Run to verify fail**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && pytest tests/classification/test_model.py -v
```

Expected: existing tests fail because `build_model` signature changed (no `name` parameter yet).

- [ ] **Step 3: Update `build_model`**

Replace the content of `classification/model.py` with:

```python
"""Classification model builder supporting torchvision and timm backbones."""
from typing import Optional
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights


TORCHVISION_MODELS = {"resnet50"}


def build_model(name: str = "resnet50",
                num_classes: int = 2,
                pretrained: bool = True) -> nn.Module:
    """Return a classification model by name.

    - 'resnet50' uses torchvision (to match the original P1 baseline).
    - Any other name is treated as a timm model name and created via
      `timm.create_model(name, pretrained=..., num_classes=...)`.
    """
    if name in TORCHVISION_MODELS:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # fall through to timm
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/classification/test_model.py -v
```

Expected: all tests pass. (The parameterized timm tests will each build their models without pretrained weights — these should be fast because no download happens.)

- [ ] **Step 5: Run full suite to catch regressions in train/eval that use `build_model`**

```bash
pytest
```

Expected: all tests pass. **If P1's `test_train.py` or `test_eval.py` fail** because they call `build_model(num_classes, pretrained)` positionally, find those test files and update the calls to pass `name="resnet50"` (or use keyword form). Also check `classification/train.py` and `classification/eval.py` — their call sites need updating too.

Specifically, update these call sites to pass `name` (either hardcoded "resnet50" or from `cfg["model"]["name"]` if you add that key — but for this step keep it simple):

- `classification/train.py`: find `build_model(cfg["model"]["num_classes"], cfg["model"]["pretrained"])` and change to `build_model(name=cfg["model"].get("name", "resnet50"), num_classes=cfg["model"]["num_classes"], pretrained=cfg["model"]["pretrained"])`.
- `classification/eval.py`: same pattern (change `build_model(cfg["model"]["num_classes"], pretrained=False)` → `build_model(name=cfg["model"].get("name", "resnet50"), num_classes=cfg["model"]["num_classes"], pretrained=False)`).

Re-run full pytest after these edits. Expected: 75 tests still pass.

- [ ] **Step 6: Commit**

```bash
git add classification/model.py classification/train.py classification/eval.py tests/classification/test_model.py
git commit -m "$(cat <<'EOF'
feat(classification): add timm backbone support to build_model

Keeps torchvision ResNet50 as default for backward compatibility; routes
any other name to timm.create_model. Enables cross-architecture comparison.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: TDD — benchmark utilities

**Files:**
- Create: `tests/classification/test_benchmark.py`
- Create: `classification/benchmark.py`

- [ ] **Step 1: Write failing tests**

Create `tests/classification/test_benchmark.py`:

```python
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
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/classification/test_benchmark.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement benchmark utilities**

Create `classification/benchmark.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/classification/test_benchmark.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add classification/benchmark.py tests/classification/test_benchmark.py
git commit -m "$(cat <<'EOF'
feat(classification): add benchmark utils (params/latency/throughput)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Write `compare_config.yaml`

**Files:**
- Create: `classification/compare_config.yaml`

- [ ] **Step 1: Write YAML**

Create `classification/compare_config.yaml`:

```yaml
# Common settings applied to every candidate model
common:
  data:
    database_root: database
    num_workers: 4
    batch_size: 32
    image_size: 224
  train:
    epochs: 5                  # short budget — data hits >98% in 2 epochs already
    lr: 0.0001
    weight_decay: 0.0001
    optimizer: adamw
    scheduler: cosine
    use_weighted_sampler: true
  eval:
    save_misclassified: false
    save_qualitative_every_n_epochs: 100
  output:
    root: outputs/classification_compare
  seed: 42
  device: auto

# Benchmark settings (measured after training each model)
benchmark:
  warmup: 5
  iters: 30
  batch_sizes: [1, 32]         # latency at 1, throughput at 32

# Ordered list of candidate models
models:
  - name: resnet50                      # torchvision baseline
    pretrained: true
  - name: efficientnet_b0               # timm
    pretrained: true
  - name: convnext_tiny                 # timm
    pretrained: true
  - name: mobilenetv3_large_100         # timm
    pretrained: true
  - name: vit_small_patch16_224         # timm (22M params)
    pretrained: true
```

- [ ] **Step 2: Verify YAML loads**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && python -c "from common.config import load_config; c = load_config('classification/compare_config.yaml'); print(len(c['models']), 'models')"
```

Expected: `5 models`.

- [ ] **Step 3: Commit**

```bash
git add classification/compare_config.yaml
git commit -m "$(cat <<'EOF'
feat(classification): add compare_config.yaml listing 5 backbones

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Implement `compare.py` runner

**Files:**
- Create: `tests/classification/test_compare.py`
- Create: `classification/compare.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/classification/test_compare.py`:

```python
"""Integration test: compare.py runs training+benchmark on 2 synthetic models."""
import json
from pathlib import Path
import yaml
import pytest


@pytest.fixture
def mini_compare_config(tmp_path, synthetic_dataset_root):
    out_root = tmp_path / "outputs"
    cfg = {
        "common": {
            "data": {
                "database_root": str(synthetic_dataset_root),
                "num_workers": 0,
                "batch_size": 2,
                "image_size": 32,  # tiny to keep it fast
            },
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
        },
        "benchmark": {"warmup": 1, "iters": 3, "batch_sizes": [1, 4]},
        # Two tiny timm models — downloads avoided by pretrained: False
        "models": [
            {"name": "resnet50", "pretrained": False},
            {"name": "mobilenetv3_large_100", "pretrained": False},
        ],
    }
    p = tmp_path / "compare.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, out_root


def test_compare_runs_and_writes_report(mini_compare_config):
    cfg_path, out_root = mini_compare_config
    from classification.compare import main
    out_dir = main(str(cfg_path))
    out_dir = Path(out_dir)

    # report files
    csv = out_dir / "comparison.csv"
    md = out_dir / "comparison.md"
    assert csv.exists()
    assert md.exists()

    # each model got its own sub-run
    runs = [d for d in out_dir.iterdir() if d.is_dir()]
    assert len(runs) >= 2


def test_compare_csv_has_expected_columns(mini_compare_config):
    cfg_path, _ = mini_compare_config
    from classification.compare import main
    out_dir = Path(main(str(cfg_path)))
    csv = (out_dir / "comparison.csv").read_text().splitlines()
    header = csv[0].split(",")
    required = {"model", "params", "accuracy", "f1_positive", "auc",
                "latency_bs1_ms", "throughput_bs4_fps"}
    assert required.issubset(set(header))
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/classification/test_compare.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `compare.py`**

Create `classification/compare.py`:

```python
"""Run training + benchmark for multiple classification backbones.

Usage:
    python -m classification.compare --config classification/compare_config.yaml

Writes:
    <out_root>/<timestamp>/
      <model_name>/              # each model's training output (as per train.py)
      comparison.csv             # one row per model with metrics + speed
      comparison.md              # human-readable markdown table
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import copy

import torch

from common.config import load_config
from common.utils import set_seed, get_device, make_output_dir
from classification.train import main as train_main
from classification.eval import main as eval_main
from classification.model import build_model
from classification.benchmark import (
    count_parameters, measure_inference_latency, measure_throughput,
)


CSV_COLUMNS = [
    "model",
    "params",
    "accuracy",
    "f1_positive",
    "precision_positive",
    "recall_positive",
    "auc",
    "latency_bs1_ms",
    "throughput_bs_fps",       # populated from the largest configured batch size
]


def _build_model_config(common: dict, model_entry: dict) -> dict:
    """Copy `common` and add model.name/pretrained keys so train.main() can consume it."""
    cfg = copy.deepcopy(common)
    cfg["model"] = {
        "name": model_entry["name"],
        "num_classes": 2,
        "pretrained": model_entry.get("pretrained", True),
    }
    return cfg


def _write_model_config_yaml(cfg: dict, path: Path) -> None:
    import yaml
    path.write_text(yaml.safe_dump(cfg))


def _benchmark_one(model_name: str, ckpt_path: Path, image_size: int,
                   device: str, warmup: int, iters: int,
                   batch_sizes: list) -> dict:
    model = build_model(name=model_name, num_classes=2, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = count_parameters(model)
    results = {"params": params}

    sample_bs1 = torch.randn(batch_sizes[0], 3, image_size, image_size)
    results["latency_bs1_ms"] = measure_inference_latency(
        model, sample_bs1, device=device, warmup=warmup, iters=iters
    ) * 1000.0

    bs_throughput = batch_sizes[-1]
    sample_bsN = torch.randn(bs_throughput, 3, image_size, image_size)
    results["throughput_bs_fps"] = measure_throughput(
        model, sample_bsN, device=device, warmup=warmup, iters=iters
    )
    results["throughput_batch_size"] = bs_throughput
    return results


def _write_markdown(rows: list, path: Path) -> None:
    lines = ["# Classification Backbone Comparison", ""]
    lines.append("| Model | Params | Acc | F1 (canker) | AUC | "
                 "Latency bs=1 (ms) | Throughput (FPS) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['params']:,} | "
            f"{r['accuracy']:.4f} | {r['f1_positive']:.4f} | {r['auc']:.4f} | "
            f"{r['latency_bs1_ms']:.2f} | {r['throughput_bs_fps']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(config_path: str) -> Path:
    cfg = load_config(config_path)
    common = cfg["common"]
    bench = cfg["benchmark"]

    set_seed(common["seed"])
    device = get_device(common["device"]).type

    out_root = Path(common["output"]["root"])
    out_dir = make_output_dir(out_root, task="compare")

    rows = []
    for entry in cfg["models"]:
        name = entry["name"]
        print(f"\n===== {name} =====")
        per_model_cfg = _build_model_config(common, entry)
        per_model_cfg["output"]["root"] = str(out_dir / name)

        per_cfg_path = out_dir / f"{name}.yaml"
        _write_model_config_yaml(per_model_cfg, per_cfg_path)

        run_dir = train_main(str(per_cfg_path))
        best_ckpt = Path(run_dir) / "ckpt" / "best.pt"
        eval_result = eval_main(str(per_cfg_path), str(best_ckpt))

        bench_result = _benchmark_one(
            name, best_ckpt,
            image_size=common["data"]["image_size"],
            device=device,
            warmup=bench["warmup"],
            iters=bench["iters"],
            batch_sizes=bench["batch_sizes"],
        )

        rows.append({
            "model": name,
            "params": bench_result["params"],
            "accuracy": eval_result["accuracy"],
            "f1_positive": eval_result["f1_positive"],
            "precision_positive": eval_result["precision_positive"],
            "recall_positive": eval_result["recall_positive"],
            "auc": eval_result["auc"],
            "latency_bs1_ms": bench_result["latency_bs1_ms"],
            "throughput_bs_fps": bench_result["throughput_bs_fps"],
        })

    # write CSV
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})

    # write markdown
    _write_markdown(rows, out_dir / "comparison.md")

    # also dump raw rows JSON for traceability
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2))

    print(f"\nComparison written to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
```

- [ ] **Step 4: Run integration test**

```bash
pytest tests/classification/test_compare.py -v
```

Expected: 2 passed. Each test trains 2 tiny models for 1 epoch on CPU — should complete in well under a minute.

- [ ] **Step 5: Commit**

```bash
git add classification/compare.py tests/classification/test_compare.py
git commit -m "$(cat <<'EOF'
feat(classification): add compare.py multi-backbone benchmarking runner

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Real-data comparison run

**Files:** (no source changes)

- [ ] **Step 1: Launch the comparison**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && python -m classification.compare --config classification/compare_config.yaml
```

This will:
- Download pretrained weights for each model (resnet50, efficientnet_b0, convnext_tiny, mobilenetv3_large_100, vit_small_patch16_224) on first run — ~several hundred MB total
- Train each on 3407 images for 5 epochs (batch 32, img 224, MPS)
- Eval each on 427 val images
- Measure latency (batch 1) and throughput (batch 32) on MPS
- Write `comparison.csv`, `comparison.md`, `comparison.json`

**Time estimate per model**: 5 epochs × ~3 min ≈ 15 min (matching P1 baseline timing).
**Total**: 5 × 15 = **~75 min**. Add download overhead (~10 min first time). Expect ~80-90 minutes.

If any model crashes (MPS op unsupported, etc.), isolate the failing model and:
- Option 1: retry with `--override common.device=cpu` for just that model (much slower but works)
- Option 2: remove it from the model list and report as DONE_WITH_CONCERNS

If total wall time exceeds 3 hours, STOP.

- [ ] **Step 2: Report findings**

Do NOT commit `outputs/` (gitignored). Just report:
- Output directory path
- Contents of `comparison.md` (paste it)
- Total wall time
- Which device each model used (usually all MPS)
- Any models that failed or had issues

---

### Task 6: Commit `comparison.md` to docs (for history)

**Files:**
- Create: `docs/results/2026-04-17-classification-comparison.md`

- [ ] **Step 1: Copy the generated markdown into docs**

Since `outputs/` is gitignored, we preserve the comparison report by copying it into `docs/results/`:

```bash
cd <project root>
mkdir -p docs/results
# Find the newest compare run
LATEST=$(ls -td outputs/classification_compare/compare/*/ | head -1)
cp "${LATEST}comparison.md" docs/results/2026-04-17-classification-comparison.md
cp "${LATEST}comparison.csv" docs/results/2026-04-17-classification-comparison.csv
cp "${LATEST}comparison.json" docs/results/2026-04-17-classification-comparison.json
```

- [ ] **Step 2: Commit**

```bash
git add docs/results/
git commit -m "$(cat <<'EOF'
docs: add P1b classification backbone comparison results

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add comparison section and reference**

Read `README.md`. Insert a new section right AFTER `## Training & Evaluation (P3 — Segmentation)` and BEFORE `## Phase status`:

```markdown
## Backbone Comparison (P1b — Classification)

Compare 5 backbones (ResNet50 / EfficientNet-B0 / ConvNeXt-Tiny /
MobileNetV3-Large / ViT-Small/16) under identical conditions — same data,
same optimizer, same budget — and report accuracy vs inference speed vs
parameter count.

```bash
python -m classification.compare --config classification/compare_config.yaml
```

Outputs in `outputs/classification_compare/compare/<timestamp>/`:
- `<model_name>/run/<sub-timestamp>/` — per-model training + eval artifacts
- `comparison.csv`, `comparison.md`, `comparison.json` — aggregated report

Latest results: [`docs/results/2026-04-17-classification-comparison.md`](docs/results/2026-04-17-classification-comparison.md)
```

- [ ] **Step 2: Run full test suite**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate disease_01 && export KMP_DUPLICATE_LIB_OK=TRUE && pytest
```

Expected: all tests still pass. P1b adds: 5 (model parametrized) + 4 (benchmark) + 2 (compare) = 11 new tests, minus any existing model tests replaced. Exact total depends on parametrize counts — should be ~85-86 tests.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: add P1b backbone comparison usage section

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Criteria (P1b complete when)

1. `pytest` passes green.
2. `python -m classification.compare --config classification/compare_config.yaml` runs end-to-end.
3. Produces `comparison.csv` / `comparison.md` / `comparison.json` with all 5 models.
4. `docs/results/2026-04-17-classification-comparison.md` committed with real numbers.
5. README updated.

## Notes for Executor

- timm is already in `environment.yml`. First-time use will cache pretrained weights to `~/.cache/torch/hub/checkpoints/` or `~/.cache/huggingface/`.
- ViT-Small has fixed input size 224 (the model name encodes it). The compare config uses `image_size: 224` so this is fine.
- If `ConvNeXt-Tiny` or `ViT-Small` has MPS ops that fall back to CPU internally, ultralytics-style warnings may print. That's expected and acceptable — just note which models triggered fallbacks.
- Inference benchmarks should use the same `device` as training (usually MPS) so the FPS numbers reflect realistic deployment on this hardware.
- Do not alter training hyperparams per model — the comparison is only fair when all models see the same budget/lr/etc.
