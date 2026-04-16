# Citrus Disease CV Pipeline — Design Spec

- **Date**: 2026-04-17
- **Project directory**: `<project root>`
- **Dataset**: AI Hub 감귤 데이터셋 (온주밀감, 정상 / 궤양병)

## 1. Goal

AI Hub 감귤 데이터셋을 사용해 **정상 vs 궤양병** 감별을 세 가지 컴퓨터 비전 태스크로 구현한다.

1. **Classification** — 이미지 전체를 정상/궤양병으로 분류
2. **Detection** — 이미지 속 감귤 열매를 bbox로 검출하고 정상/궤양병 분류
3. **Segmentation** — 픽셀 단위로 배경/정상/궤양병 세 클래스 분할

기존 `main.py` (단일 classification) 를 완전히 대체하고, 공통 모듈 기반 재사용 가능한 구조로 재설계한다.

## 2. Scope

### In Scope (Phase 1)
- 3개 태스크의 학습/평가 파이프라인 구축
- 공통 데이터 로딩 및 라벨 파싱 모듈
- YAML 기반 config, TensorBoard 로깅
- 재현 가능한 실험 환경 (seed, env 정의, timestamped outputs)

### Out of Scope (Phase 2 — 차후)
- 멀티모달 학습 (이미지 + 환경 feature 결합)
- 카메라/지역 기반 도메인 일반화 실험
- 모델 아키텍처 비교 및 fine-tuning 실험
- 웹 / 배포용 추론 API

### 기존 코드 처리
| 파일 | 처리 |
|---|---|
| `main.py`, `backup.py`, `otsu.py`, `otsu2.py`, `test_model.pt` | `_archive/`로 이동 |
| `HF01_00FT_000001.jpg` | 삭제 (`database/`에 동일 파일 있음) |
| `willmer/` | 사용자가 직접 다른 위치로 이동 (건드리지 않음) |
| `_vis/`, `_vis_polygon.py` | polygon 시각화 임시 산출물, `_archive/` 또는 삭제 |

## 3. Environment

- **Hardware**: Mac mini M4, 32GB RAM, Apple Silicon (arm64)
- **Accelerator**: MPS (CUDA 없음) — 일부 연산 CPU fallback 가능성 인지
- **Conda env (신규)**: `disease_01`
  - Python 3.11
  - PyTorch (MPS 지원 버전)
  - torchvision, ultralytics, segmentation_models_pytorch, timm
  - opencv-python, pillow, albumentations
  - numpy, pandas, scikit-learn, matplotlib, seaborn
  - pyyaml, tqdm, tensorboard, torchmetrics
- **Env 정의**: `environment.yml` 파일로 재현 가능하게 관리
- **시각화(임시)**: 기존 `depth_estimation` env 재사용 (설계 단계 한정, 코드 개발 시점엔 신규 env 사용)

## 4. Dataset

### 전체 현황
| Split | 클래스 | 이미지 | JSON 라벨 | Polygon 있음 |
|---|---|---:|---:|---:|
| Train | 정상 | 2,035 | 2,035 | 471 |
| Train | 궤양병 | 1,372 | 1,372 | 228 |
| Valid | 정상 | 255 | 255 | 59 |
| Valid | 궤양병 | 172 | 172 | 29 |
| **합계** | | **3,834** | **3,834** | **787** |

### 태스크별 데이터 할당
| 태스크 | Train | Valid | 라벨 소스 |
|---|---:|---:|---|
| Classification | 3,407 | 427 | 폴더명 (`열매_정상`=0, `열매_궤양병`=1) |
| Detection | 699 | 88 | JSON polygon → bbox 변환 |
| Segmentation | 699 | 88 | JSON polygon → 픽셀 마스크 |

### 라벨 포맷
JSON 구조:
```json
{
  "Info": { "IMAGE_FILE_NM", "RSOLTN", "CMRA_INFO", "LCINFO",
            "GRWH_STEP_CODE", "IMAGE_OBTAIN_PLACE_TY", "OCPRD", ... },
  "Annotations": {
    "ANTN_ID": "1" | null,
    "ANTN_TY": "polygon" | null,
    "OBJECT_CLASS_CODE": "감귤_정상" | "감귤_궤양병",
    "ANTN_PT": "[x1|x2|...],[y1|y2|...]" | null
  },
  "Environment": { "SOLRAD_QY", "AFR", "TP", "HD", "SOIL_MITR" }
}
```

### 중요 관찰
- 이미지는 **이미 흰색 배경으로 전처리되어 있음** → 기존 `segment_fruit()` HSV 마스킹 불필요
- Polygon은 **감귤 열매의 외곽선** (병변 위치 아님)
- 이미지 해상도: 1920×1080 (전처리 단계에서 Resize로 흡수)
- 클래스 불균형: 정상:궤양병 ≈ 1.48:1 (약한 불균형)

### Split 전략
- AI Hub가 제공한 **`1.Training/` vs `2.Validation/` 폴더 split을 그대로 사용**
- 기존 `main.py`의 `random.sample()` 기반 8:2 분할 로직은 폐기 (`_archive/` 이동)
- 세 태스크 모두 동일한 split 사용 (실험 간 비교 공정성)

## 5. Architecture

### 폴더 구조
```
disease_01/
├── database/                    # AI Hub 데이터 (유지)
├── common/                      # 공통 모듈
│   ├── __init__.py
│   ├── dataset.py               # ClassificationDataset / DetectionDataset / SegmentationDataset
│   ├── label_parser.py          # ANTN_PT → polygon/bbox/mask
│   ├── config.py                # YAML 로더 + dataclass
│   └── utils.py                 # seed, device, paths, logging
├── classification/
│   ├── train.py
│   ├── eval.py
│   └── config.yaml
├── detection/
│   ├── prepare_yolo.py          # JSON → YOLO txt + data.yaml 생성
│   ├── train.py
│   ├── eval.py
│   ├── config.yaml
│   └── data/                    # YOLO 포맷 변환된 이미지/라벨 (gitignore)
├── segmentation/
│   ├── train.py
│   ├── eval.py
│   └── config.yaml
├── outputs/                     # 실험 결과 (gitignore)
│   └── <task>/<timestamp>/
│       ├── train.log
│       ├── tb/                  # tensorboard events
│       ├── ckpt/                # best.pt, last.pt
│       ├── qualitative/         # 예측 시각화
│       └── config.yaml          # 실험 시점 config 스냅샷
├── checkpoints/                 # 최종 선택된 모델 복사본 (선택적)
├── _archive/                    # 기존 코드 보관
│   ├── main.py
│   ├── backup.py
│   ├── otsu.py
│   ├── otsu2.py
│   └── test_model.pt
├── docs/superpowers/specs/      # 설계 문서
├── environment.yml
├── .gitignore
└── README.md
```

### 모듈 책임

#### `common/label_parser.py`
- `parse_antn_pt(pt_str: str) -> np.ndarray` — `"[x1|x2|...],[y1|y2|...]"` → shape `(N, 2)` int array
- `polygon_to_bbox(polygon: np.ndarray) -> tuple[int,int,int,int]` — `(x_min, y_min, x_max, y_max)`
- `polygon_to_mask(polygon: np.ndarray, h: int, w: int) -> np.ndarray` — binary mask `(H, W) uint8`
- `load_sample(json_path: Path) -> dict` — 파싱된 샘플 정보 (class, polygon, size, metadata)

#### `common/dataset.py`
세 개의 Dataset 클래스. 각각 `__getitem__`이 반환하는 구조:

**공통 메타데이터** (모든 태스크, Phase 2 대비):
```python
metadata = {
    "camera": "samsung",
    "location": "F02",
    "place_type": "노지",
    "growth_stage": "6",
    "env": {
        "solar": 34.7, "temp": 29.6, "humidity": 64.8,
        "soil_moisture": 74.9, "rain": 0.0
    }
}
```

- `ClassificationDataset`: `{image, label, metadata}` — 전체 이미지 사용
- `DetectionDataset`: YOLO 학습은 ultralytics에 위임. Dataset 클래스는 `prepare_yolo.py`에서 변환 시 사용 (학습 loop는 ultralytics 내부)
- `SegmentationDataset`: `{image, mask, metadata}` — polygon 있는 이미지만 필터링

#### `common/config.py`
- `load_config(path: str) -> dict`
- `@dataclass` 기반 스키마 (train/data/model/output 섹션)
- argparse로 YAML 경로 + 선택적 override 받기

#### `common/utils.py`
- `set_seed(seed: int)` — random/numpy/torch 시드 통합
- `get_device() -> torch.device` — auto-detect (mps > cpu)
- `make_output_dir(task: str) -> Path` — 타임스탬프 폴더 생성, config 스냅샷 저장
- 로거 설정 helper

## 6. Task 1: Classification

### 모델
- `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`
- Final FC 2 classes로 교체

### 전처리
```python
T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.2),
    T.ColorJitter(0.1, 0.1, 0.1),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```
Validation은 aug 없이 `Resize→CenterCrop→ToTensor→Normalize`만.

### 하이퍼파라미터
| 항목 | 값 |
|---|---|
| Input size | 224 |
| Batch size | 32 |
| Epochs | 30 |
| Optimizer | AdamW |
| LR | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| 불균형 처리 | `WeightedRandomSampler` (class inverse frequency) |

### 평가 지표
- Accuracy, Precision/Recall/F1 (per-class), ROC-AUC, Confusion Matrix
- **Best checkpoint 기준**: val F1 (궤양병 class)
- 오분류 샘플 이미지 저장

## 7. Task 2: Detection

### 접근
- ultralytics YOLOv8 사용, 자체 학습 루프·augmentation·metric 모두 내장
- 직접 학습 루프 구현하지 않음

### 데이터 변환 (`prepare_yolo.py`)
- JSON polygon → bbox → YOLO txt 포맷 (class_id x_center y_center w h, normalized)
- 이미지를 `detection/data/{train,val}/images/`에 symlink 또는 복사
- 라벨을 `detection/data/{train,val}/labels/`에 저장
- `detection/data/data.yaml` 자동 생성 (클래스명, 경로)
- **polygon 있는 787장만 사용**

### 모델 & 학습
| 항목 | 값 |
|---|---|
| Model | `yolov8s.pt` (COCO pretrained) |
| imgsz | 640 |
| Batch | 16 |
| Epochs | 100 |
| lr0 | 0.01 (YOLO 기본) |
| Optimizer | SGD (YOLO 기본) |
| Augmentation | ultralytics 기본 (mosaic, mixup, flip) |
| Device | mps |

### 평가 지표
- mAP@0.5, mAP@0.5:0.95, per-class AP, Precision/Recall
- ultralytics 내장 출력 사용
- 예측 결과 이미지 저장 (bbox overlay)
- **Best checkpoint**: val mAP@0.5 (ultralytics 기본)

## 8. Task 3: Segmentation

### 태스크 정의
- **3-class semantic segmentation**:
  - class 0: 배경
  - class 1: 정상 감귤
  - class 2: 궤양병 감귤

### 모델
- `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=3)`

### 전처리 (albumentations)
```python
A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ColorJitter(0.1, 0.1, 0.1, p=0.5),
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ToTensorV2(),
])
```
Validation은 `Resize→Normalize→ToTensor`만. 이미지와 mask를 동시에 변환.

### 마스크 생성
- 이미지 로드 시 polygon → `(H, W)` mask 생성 → resize 512×512
- 클래스 값: 배경 0, 정상이면 1, 궤양병이면 2 (폴더명 기반 결정)

### 하이퍼파라미터
| 항목 | 값 |
|---|---|
| Input size | 512 |
| Batch size | 8 |
| Epochs | 50 |
| Optimizer | AdamW |
| LR | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | 0.5 × CrossEntropyLoss + 0.5 × DiceLoss |

### 평가 지표
- mIoU, per-class IoU, Dice coefficient, Pixel Accuracy, 픽셀 단위 confusion matrix
- **Best checkpoint**: val mIoU (정상/궤양병 IoU에 집중)
- 정성 시각화: 원본 | GT mask | 예측 mask 3분할 이미지 저장

## 9. Configuration

### 방식
YAML + argparse + dataclass 하이브리드

### 파일 위치
- 각 태스크 폴더: `classification/config.yaml`, `detection/config.yaml`, `segmentation/config.yaml`
- 실험 실행 시 `outputs/<task>/<timestamp>/config.yaml`로 스냅샷 저장 (재현성)

### CLI
```bash
python classification/train.py --config classification/config.yaml
python classification/train.py --config classification/config.yaml --lr 0.0005  # override
```

### 공통 config 섹션 구조
```yaml
data:
  database_root: ../database
  num_workers: 4
  batch_size: 32
  image_size: 224

model:
  name: resnet50
  pretrained: true
  num_classes: 2

train:
  epochs: 30
  lr: 0.0001
  weight_decay: 0.0001
  optimizer: adamw
  scheduler: cosine

output:
  root: ../outputs/classification

seed: 42
device: auto
```

## 10. Logging

### 도구
- **TensorBoard**: loss / metric / LR 곡선, 샘플 예측 이미지 (5 epoch마다)
- **Python logging**: 콘솔 + `train.log` 파일
- **tqdm**: 진행률
- **YOLOv8**: ultralytics 내장 로거 (자동으로 TensorBoard 연동)

### TensorBoard 기록 항목
- `train/loss`, `val/loss` (epoch마다)
- `train/lr` (epoch마다)
- `val/<metric>` (태스크별: accuracy/f1, mAP, mIoU 등)
- `qualitative/<sample_i>` 이미지 (5 epoch마다)

### 실행
```bash
tensorboard --logdir outputs/classification
# http://localhost:6006
```

## 11. Reproducibility & Operations

- `set_seed(42)` — random, numpy, torch (CPU & MPS)
- 실험별 config 스냅샷 저장
- 타임스탬프 폴더 (`YYYY-MM-DD_HH-MM-SS`) 로 덮어쓰기 방지
- `environment.yml`로 env 재현 가능

## 12. Phase 2 Hooks (Future)

Phase 1 구현 시 Phase 2를 위해 준비해두는 것:

- **`common/dataset.py`에서 metadata 필드 항상 로딩** (학습엔 미사용)
  - `CMRA_INFO`, `LCINFO`, `IMAGE_OBTAIN_PLACE_TY`, `GRWH_STEP_CODE`
  - Environment: `SOLRAD_QY`, `AFR`, `TP`, `HD`, `SOIL_MITR`
- 이로써 Phase 2에서 멀티모달 · 도메인 일반화 · 카메라/지역 split 실험을 구현 가능

## 13. Deliverables

Phase 1 완료 기준:
1. `common/` 모듈 (dataset, label_parser, config, utils)
2. 각 태스크의 `train.py`, `eval.py`, `config.yaml` 동작
3. `environment.yml`로 재현 가능한 env
4. 3개 태스크 각각 최소 1회 학습 완료 + best checkpoint 저장
5. 각 태스크 평가 결과 (metric 요약) 기록
6. `_archive/` 로 기존 코드 정리
7. `README.md`: 실행 방법, 폴더 구조, 결과 요약
