# Citrus Disease CV

감귤(온주밀감) 정상/궤양병 감별을 위한 Classification / Detection / Segmentation 파이프라인.

## Structure

```
disease_01/
├── common/             # 공통 모듈 (dataset, label_parser, config, utils)
├── classification/     # P1 (planned)
├── detection/          # P2 (planned)
├── segmentation/       # P3 (planned)
├── scripts/            # 유틸리티 스크립트 (smoke test 등)
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

**Important (macOS OpenMP workaround):** 이 프로젝트는 numpy / torch / opencv 여러 라이브러리가 각자 OpenMP를 링크해서 실행 시 충돌이 발생합니다. 실행 전 아래 환경변수를 설정해야 합니다:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

쉘에 영구 적용하려면 `~/.zshrc` 등에 추가하거나 실행 스크립트 상단에 두세요.

## Run tests

```bash
pytest
```

## Smoke test against real data

```bash
python scripts/smoke_test_datasets.py
```

Expected output:
- Classification train: 3407 / val: 427
- Segmentation train: 699 / val: 88

## Design

- Spec: `docs/superpowers/specs/2026-04-17-citrus-disease-cv-design.md`
- Plans: `docs/superpowers/plans/`

## Training & Evaluation (P1)

All commands run from the project root.

```bash
# train on full data (default: 30 epochs, MPS)
python classification/train.py --config classification/config.yaml

# quick smoke training (2 epochs)
python classification/train.py --config classification/config.yaml --override train.epochs=2

# evaluate best checkpoint
python classification/eval.py --config classification/config.yaml \
    --ckpt outputs/classification/<run>/ckpt/best.pt

# view training curves
tensorboard --logdir outputs/classification
```

Outputs land in `outputs/classification/<timestamp>/`:
- `ckpt/best.pt`, `ckpt/last.pt` — model checkpoints
- `train.log` — per-epoch log
- `tb/` — TensorBoard events
- `config.yaml` — frozen config snapshot
- `confusion_matrix.png`, `metrics.json` — from `eval.py`

## Training & Evaluation (P2 — Detection)

Use `-m` module form so `common` imports resolve correctly.

```bash
# one-time: convert AI Hub polygons to YOLO format (~787 images)
python -m detection.prepare_yolo --source database --dest detection/data

# train
python -m detection.train --config detection/config.yaml

# quick smoke (5 epochs)
python -m detection.train --config detection/config.yaml --override train.epochs=5

# evaluate best checkpoint
python -m detection.eval --config detection/config.yaml \
    --ckpt outputs/detection/run/weights/best.pt
```

Outputs land in `outputs/detection/run*/`:
- `weights/best.pt`, `weights/last.pt`
- `results.csv`, `results.png`
- `confusion_matrix.png`, P/R curves

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

## Phase status

- [x] P0 — Common module
- [x] P1 — Classification
- [x] P2 — Detection
- [x] P3 — Segmentation
