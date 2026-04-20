# disease_01 — 감귤 궤양병 CV 파이프라인

감귤(온주밀감) **정상 vs 궤양병** 감별을 위한 Classification / Detection / Segmentation 엔드투엔드 파이프라인. 공통 모듈 위에 세 가지 컴퓨터 비전 태스크를 TDD로 구축하고, 5개 백본을 동일 조건으로 비교한 결과까지 포함.

- **데이터**: AI Hub 감귤 병충해 데이터셋 (3,834장 — 정상 2,290 / 궤양병 1,544)
- **하드웨어**: Apple Silicon (Mac mini M4, 32GB, MPS backend)
- **태스크**: Classification / Detection / Segmentation + 백본 비교

## 🏆 주요 성과

| Phase | 태스크 | 모델 | 핵심 지표 |
|---|---|---|---|
| P1 | Classification | ResNet50 (2 epochs) | **Acc 98.83%**, F1 0.986, AUC 0.999, canker Recall **100%** |
| P2 | Detection | YOLOv8s (5 epochs) | **mAP@0.5 0.994**, 궤양병 AP 0.995 |
| P3 | Segmentation | smp U-Net + ResNet34 (3 epochs) | **mIoU 0.940**, 궤양병 IoU 0.886, pixel acc 99.3% |
| P1b | Backbone 비교 | 5종 비교 | **ViT-Small/16 종합 우승** (F1 0.988, latency 5.94ms) |

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [아키텍처](#2-아키텍처)
3. [데이터 파이프라인 & 라벨 파싱](#3-데이터-파이프라인--라벨-파싱)
4. [알고리즘 & 학습 로직](#4-알고리즘--학습-로직)
   - [P1 Classification (ResNet50)](#p1--classification-resnet50)
   - [P2 Detection (YOLOv8s)](#p2--detection-yolov8s)
   - [P3 Segmentation (U-Net + ResNet34)](#p3--segmentation-u-net--resnet34)
   - [P1b 백본 비교](#p1b--백본-비교-방법론)
5. [결과 & 해석](#5-결과--해석)
6. [한계 & 주의사항](#6-한계--주의사항)
7. [실행 방법](#7-실행-방법)
8. [프로젝트 구조 & 환경](#8-프로젝트-구조--환경)
9. [분석 노트북 & 설계 문서](#9-분석-노트북--설계-문서)
10. [진행 상황 & 향후 계획](#10-진행-상황--향후-계획)

---

## 1. 프로젝트 개요

### 문제 정의

감귤(온주밀감)에 발생하는 **궤양병(canker)** 을 이미지로 자동 감별. 궤양병은 감귤류 주요 질병으로 수확량·상품성에 직접 영향을 미침. 조기 감별은 선별·검역 비용을 크게 줄인다.

### 세 가지 CV 태스크로 접근
| 태스크 | 출력 | 활용 시나리오 |
|---|---|---|
| **Classification** | 이미지 전체 → 정상 or 궤양병 (1개 라벨) | 선별 라인에서 빠른 이진 판정 |
| **Detection** | 이미지 속 감귤 bbox + 클래스 | 다중 과일이 있는 장면에서 위치 정보까지 |
| **Segmentation** | 픽셀마다 클래스(배경/정상/궤양병) | 픽셀 단위 정확 분할 — 가장 정교 |

### 데이터셋 구성
| Split | 정상 | 궤양병 | 합계 | Polygon 있음 |
|---|---:|---:|---:|---:|
| Train | 2,035 | 1,372 | 3,407 | 699 |
| Validation | 255 | 172 | 427 | 88 |
| **합계** | **2,290** | **1,544** | **3,834** | **787** |

- 이미지 해상도 1920×1080, 배경이 **이미 흰색으로 전처리됨**
- Classification은 폴더명(정상=0, 궤양병=1)으로 라벨 부여 → 전체 사용
- Detection/Segmentation은 JSON의 polygon 라벨이 필요 → 787장만 사용
- Polygon은 **감귤 열매의 외곽선**이며 병변(lesion) 자체는 아님 → 중요한 한계 (§6 참조)

---

## 2. 아키텍처

### 설계 원칙
1. **공통 모듈 재사용 (DRY)** — 데이터 로딩·라벨 파싱·설정·유틸을 `common/` 하나로 통일
2. **태스크별 독립성** — 각 태스크(`classification/`, `detection/`, `segmentation/`)는 공통 모듈만 의존
3. **TDD** — 85개 unit/integration test로 회귀 방지
4. **재현성** — 타임스탬프 출력 디렉토리 + config 스냅샷 + 고정 seed(42)
5. **Config-driven** — YAML + CLI override로 하이퍼파라미터 실험

### 모듈 구성
```
disease_01/
├── common/                  # 공통 모듈
│   ├── dataset.py           # ClassificationDataset / SegmentationDataset
│   ├── label_parser.py      # JSON ANTN_PT → polygon / bbox / mask
│   ├── config.py            # YAML 로더 + CLI override
│   └── utils.py             # seed / device(MPS/CPU) / output dir
├── classification/          # P1 + P1b (백본 비교)
│   ├── model.py             # ResNet50 (torchvision) 또는 timm 백본
│   ├── transforms.py        # ImageNet normalize + aug
│   ├── sampler.py           # WeightedRandomSampler (class 불균형)
│   ├── metrics.py           # Acc / F1 / AUC / Confusion Matrix
│   ├── benchmark.py         # latency / FPS / params 측정
│   ├── compare.py           # 다중 백본 학습·벤치마크 러너
│   ├── train.py / eval.py   # 학습·평가 엔트리포인트
│   └── config.yaml
├── detection/               # P2
│   ├── yolo_format.py       # polygon → YOLO bbox 정규화
│   ├── prepare_yolo.py      # AI Hub → YOLO 포맷 변환 CLI
│   ├── train.py / eval.py   # ultralytics YOLOv8 래퍼
│   └── config.yaml
├── segmentation/            # P3
│   ├── model.py             # smp Unet(encoder=resnet34)
│   ├── transforms.py        # albumentations (image + mask 동시 변환)
│   ├── losses.py            # CombinedLoss (CE + Dice)
│   ├── metrics.py           # mIoU / Dice / pixel accuracy
│   ├── train.py / eval.py
│   └── config.yaml
├── tests/                   # 85개 test
├── scripts/
├── docs/
│   ├── analysis/            # Jupyter 분석 노트북 6종
│   ├── results/             # 실험 결과 아카이브 (JSON/CSV/MD)
│   ├── superpowers/specs/   # 전체 설계 스펙
│   └── superpowers/plans/   # Phase별 구현 계획
├── database/                # AI Hub 데이터 (gitignored)
└── environment.yml
```

---

## 3. 데이터 파이프라인 & 라벨 파싱

### AI Hub JSON 라벨 구조

각 이미지마다 JSON 하나. 주요 필드:
```json
{
  "Info": {
    "IMAGE_FILE_NM": "HF01_01FT_000001",
    "RSOLTN": "(1920,1080)",
    "CMRA_INFO": "samsung",         // samsung / Xiaomi / LGE
    "LCINFO": "F02",                // 10개 지역
    "IMAGE_OBTAIN_PLACE_TY": "노지",  // 노지 / 온실
    "GRWH_STEP_CODE": "6",           // 성장 단계
    "OCPRD": "08-05"                 // 촬영 일자
  },
  "Annotations": {
    "ANTN_TY": "polygon" | null,
    "OBJECT_CLASS_CODE": "감귤_정상" | "감귤_궤양병",
    "ANTN_PT": "[x1|x2|...],[y1|y2|...]" | null
  },
  "Environment": {
    "SOLRAD_QY": "34.7",  // 일사량
    "TP": "29.6",         // 기온(°C)
    "HD": "64.8",         // 습도(%)
    "SOIL_MITR": "74.9",  // 토양 수분
    "AFR": "0"            // 강수량
  }
}
```

### 라벨 변환 로직 (`common/label_parser.py`)

#### ① ANTN_PT 문자열 파싱
AI Hub의 독특한 폴리곤 포맷 `"[x1|x2|...],[y1|y2|...]"` → `(N, 2)` int array:
```python
parse_antn_pt("[10|20|30],[100|200|300]")
# → np.array([[10,100], [20,200], [30,300]], dtype=int32)
```

#### ② Polygon → Bounding Box
```python
polygon_to_bbox(polygon) = (x_min, y_min, x_max, y_max)
# axis-aligned bbox (polygon 꼭짓점의 min/max)
```

#### ③ Polygon → Binary Mask
```python
polygon_to_mask(polygon, H, W) = uint8 array shape (H, W)
# cv2.fillPoly로 polygon 내부를 1, 외부를 0으로 래스터화
```

#### ④ YOLO 포맷 정규화 (`detection/yolo_format.py`)
```python
polygon_to_yolo_bbox(polygon, img_w, img_h) = (x_center, y_center, w, h)
# 모두 [0, 1]로 normalize — YOLO 학습 라벨 표준
```

#### ⑤ 3-class Segmentation Mask
```python
# polygon → binary mask → 클래스 값으로 스케일
mask = polygon_to_mask(polygon, H, W)  # 0 / 1
mask *= {"감귤_정상": 1, "감귤_궤양병": 2}[class_code]
# 최종: 0=bg, 1=정상 감귤, 2=궤양병 감귤
```

### Dataset 클래스
- `ClassificationDataset` — 이미지 + 폴더명 라벨 + 메타데이터(카메라·지역·기상) 반환 (3,834장 전체)
- `SegmentationDataset` — polygon 있는 것만 필터링 → 이미지 + 3-class mask 반환 (787장)
- 두 클래스 모두 반환 dict에 **metadata** 포함 → 향후 멀티모달 학습 때 바로 활용 가능

---

## 4. 알고리즘 & 학습 로직

### P1 — Classification (ResNet50)

#### 모델: ResNet50
- **Residual Block** — `y = F(x) + x` 구조로 gradient vanishing 완화 → 50-layer 깊은 네트워크 학습 가능
- ImageNet 1K 사전학습 가중치 사용 → 감귤 데이터셋(3,400장) 규모에 적합한 **transfer learning**
- 마지막 FC layer를 `nn.Linear(2048, 2)` 로 교체 (원래 1000-class → 2-class)

#### 전처리 파이프라인
```
BGR numpy image (cv2.imread)
  → BGR→RGB 변환
  → Resize(256) → CenterCrop(224)                    # ImageNet 표준 크기
  → (학습 시) HFlip(0.5) + VFlip(0.2) + ColorJitter(0.1,0.1,0.1)
  → ToTensor (0~255 → 0~1)
  → Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])  # ImageNet 통계
```
- Val은 augmentation 제외 (결정론적)
- **ImageNet 통계로 정규화**하는 이유: pretrained weights가 이 분포 기준으로 학습되었기 때문

#### 학습 알고리즘
| 구성요소 | 선택 | 근거 |
|---|---|---|
| **Loss** | CrossEntropyLoss | 표준 이진/다중 분류 손실. `log-softmax + NLL` 결합 |
| **Optimizer** | AdamW (lr=1e-4, wd=1e-4) | Adam + weight decay 분리 → pretrained fine-tuning에 stable |
| **Scheduler** | CosineAnnealingLR | LR이 cosine 곡선으로 0까지 감소 → local minima 탈출 + 안정적 수렴 |
| **Batch** | 32 | M4 MPS 메모리 균형 |
| **Epoch** | 30 (실험은 2) | 데이터가 쉬워서 2 epoch에 수렴 |
| **Sampler** | `WeightedRandomSampler` | 클래스 불균형 (정상:궤양병 ≈ 1.48:1) 완화 — 각 클래스 확률 = 1/count, 배치에서 balanced 샘플링 |

#### Best Checkpoint 기준
**Val F1 (궤양병 class)** — accuracy가 아님. 이유:
- 의료·질병 진단에서는 **놓치지 않는 것(recall)** 이 중요
- Precision과 Recall의 조화평균인 F1이 불균형 데이터에서 accuracy보다 신뢰할 수 있는 지표
- 궤양병(positive class)에 특화

#### 평가 지표
- Accuracy — 전체 정확도 (sanity check)
- **Precision/Recall/F1 per-class** — 불균형 데이터 핵심
- **ROC-AUC** — threshold 무관한 ranking 성능
- **Confusion Matrix** — 오분류 유형 시각 분석

---

### P2 — Detection (YOLOv8s)

#### 모델: YOLOv8 (Ultralytics)
- **One-stage detector** — region proposal 없이 feature map에서 직접 bbox + class 예측 → Faster R-CNN 대비 실시간
- **Anchor-free** (v8부터) — anchor 튜닝 불필요, 작은 객체까지 robust
- **Decoupled head** — classification head / regression head 분리 → 최적화 안정성 ↑
- COCO 사전학습 가중치 사용

#### 데이터 변환 (AI Hub → YOLO format)
Polygon 라벨 → bbox → YOLO 텍스트 파일:
```
<class_id> <x_center> <y_center> <width> <height>   # 모두 [0, 1] normalized
```
`detection/prepare_yolo.py`가 1회 변환. 출력:
```
detection/data/
├── train/images/*.jpg      # 699장
├── train/labels/*.txt
├── val/images/*.jpg         # 88장
├── val/labels/*.txt
└── data.yaml                # ultralytics가 읽는 데이터 정의
```

#### 학습 알고리즘
Ultralytics가 내부적으로 처리 (우리 래퍼는 설정만 전달):

| 구성요소 | 값/방식 |
|---|---|
| **Loss** | BCE (classification) + CIoU (box regression) + DFL (distribution focal loss) |
| **Optimizer** | SGD with momentum (YOLO 기본) |
| **LR** | lr0=0.01, lrf=0.01 (cosine warmdown) |
| **Augmentation** | mosaic, mixup, HSV jitter, flip — 내장 |
| **Batch** | 16 @ imgsz=640 |

#### 평가 지표 (mAP 계열)
- **mAP@0.5** — IoU ≥ 0.5를 "정답"으로 간주한 Average Precision 평균
- **mAP@0.5:0.95** — IoU threshold를 0.5~0.95 10단계로 평균한 엄격한 지표 (COCO 표준)
- Per-class AP — normal / canker 각각
- Precision / Recall curves

---

### P3 — Segmentation (U-Net + ResNet34)

#### 모델: U-Net with ResNet34 encoder
- **Encoder-Decoder 구조** — encoder로 해상도 줄이며 의미 추출 → decoder로 해상도 복원
- **Skip connections** — encoder 중간 feature를 decoder에 직접 연결 → **공간 세부 정보 보존** (segmentation 정확도 핵심)
- **ResNet34 encoder** — 경량 + pretrained ImageNet weights 사용 (transfer learning)
- `segmentation_models_pytorch` 라이브러리로 생성 — `smp.Unet(encoder_name="resnet34", classes=3)`

```
입력 (3×512×512)
  ↓ encoder (ResNet34 backbone, 해상도 1/2, 1/4, 1/8, 1/16, 1/32씩 축소)
  ↓ bottleneck
  ↓ decoder (transposed conv로 업샘플, skip connections concat)
출력 (3×512×512)  ← 3 채널 = bg / 정상 / 궤양병 각 확률
```

#### 3-class Semantic Segmentation
- class 0: 배경 (흰색 영역)
- class 1: 정상 감귤 (픽셀)
- class 2: 궤양병 감귤 (픽셀)

단일 모델로 **분할 + 분류 동시 해결** (별도 classifier 불필요).

#### 전처리 (albumentations)
이미지와 마스크를 **동시에 같은 랜덤 변환으로** 처리해야 함 → `albumentations` 사용:
```python
A.Compose([
    A.Lambda(image=bgr_to_rgb, mask=None),    # image만 변환
    A.Resize(512, 512,
             interpolation=BILINEAR,          # image: bilinear
             mask_interpolation=NEAREST),     # mask: nearest (클래스 값 보존!)
    A.HorizontalFlip(p=0.5),                  # image+mask 동기화
    A.VerticalFlip(p=0.2),
    A.ColorJitter(0.1, 0.1, 0.1, p=0.5),     # image만 영향
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ToTensorV2(),                             # (H,W,C)→(C,H,W) tensor
])
```
핵심: **마스크는 Nearest Neighbor로 리사이즈** — bilinear하면 class 1과 2 사이에 1.5 같은 중간값이 생겨 loss가 망가짐.

#### 학습 알고리즘
| 구성요소 | 선택 | 근거 |
|---|---|---|
| **Loss** | 0.5 × CrossEntropyLoss + 0.5 × DiceLoss | 아래 설명 |
| **Optimizer** | AdamW (lr=1e-3, wd=1e-4) | scratch decoder라 P1보다 lr 높음 |
| **Scheduler** | CosineAnnealingLR | |
| **Batch** | 8 @ 512×512 | 해상도 × 3채널 → 메모리 큼 |
| **Epoch** | 50 (실험은 3) | |

**왜 CE + Dice 조합인가?**
- **CE (CrossEntropy)**: 픽셀 단위 분류 손실, gradient 안정
- **Dice Loss**: `1 - 2|GT∩Pred| / (|GT|+|Pred|)` — 겹침을 직접 최적화. **클래스 불균형에 강함** (소수 클래스 픽셀도 잘 학습)
- 픽셀 수 관점에서 배경이 압도적으로 많아 CE만으로는 배경으로 편향 → Dice로 상쇄

#### 평가 지표 (픽셀 기반)
- **mIoU (mean Intersection over Union)** — 핵심 지표
  `IoU(c) = TP(c) / (TP(c) + FP(c) + FN(c))`
- **Per-class IoU** — bg / normal / canker 각각
- **Dice coefficient** — 2×TP / (2×TP + FP + FN), IoU와 상관 있지만 다름
- **Pixel Accuracy** — 참고용 (배경 때문에 높게 나옴)
- 픽셀 단위 Confusion Matrix

---

### P1b — 백본 비교 방법론

#### 개요
5개 아키텍처를 **동일한 데이터·optimizer·epoch(5)·seed(42)** 로 학습하여 공정 비교.

| 모델 | 특징 |
|---|---|
| **ResNet50** (torchvision) | CNN baseline, residual blocks, ImageNet pretrained |
| **EfficientNet-B0** (timm) | compound scaling (depth × width × resolution) |
| **ConvNeXt-Tiny** (timm) | 최신 CNN, Transformer 디자인 아이디어 차용 (LayerNorm, GELU, inverted bottleneck) |
| **MobileNetV3-Large** (timm) | depthwise separable conv + SE block + h-swish, 모바일 최적화 |
| **ViT-Small/16** (timm) | Vision Transformer, 이미지 patch를 토큰으로 처리 |

구현: `classification/model.py`의 `build_model(name, ...)` 팩토리가 `TORCHVISION_MODELS` set에 있으면 torchvision, 아니면 `timm.create_model()`로 라우팅.

#### 측정 항목

| 지표 | 목적 | 측정 방법 |
|---|---|---|
| **Accuracy / F1 / AUC** | 성능 | val set 평가 |
| **Latency @ batch=1** (ms) | 단일 이미지 실시간 처리 | `measure_inference_latency` with MPS sync |
| **Throughput @ batch=32** (FPS) | 배치 처리량 | `measure_throughput` |
| **Parameter count** | 메모리·배포 크기 | `sum(p.numel() for p in params)` |

#### MPS 벤치마킹 정확도 주의점
MPS는 kernel queue가 비동기이므로 timing 전후 `torch.mps.synchronize()` 필수:
```python
for _ in range(warmup): model(x)          # 커널 컴파일·캐시 warmup
torch.mps.synchronize()                   # 큐 비움
t0 = time.perf_counter()
for _ in range(iters): model(x)
torch.mps.synchronize()
elapsed = time.perf_counter() - t0
```
(sync 누락 시 latency가 비현실적으로 작게 측정됨)

#### Pareto Frontier
- F1과 latency는 trade-off 관계 (보통 더 큰 모델 = 높은 F1 + 느린 latency)
- **Pareto optimal** = 다른 모델에 의해 "dominated" 되지 않는 모델 (F1도 높고 latency도 낮음)
- P1b 결과에서 ViT-Small이 유일하게 Pareto-optimal이었다 (§5 참조)

---

## 5. 결과 & 해석

### 5.1 P1 Classification (ResNet50, 2 epochs on MPS)

```
Val accuracy   : 98.83%
F1 (궤양병)    : 0.9856
Precision      : 0.9771
Recall         : 1.0000   ← 궤양병을 단 한 장도 놓치지 않음
AUC            : 0.9975
```

**Confusion Matrix (val 427장)**:
```
               Pred normal   Pred canker
True normal       250             5       (FP: 5)
True canker         0           172       (FN: 0)
```

#### 해석
- **Recall 100%** (궤양병 0장 놓침) — 의료/농업 스크리닝 용도로 매우 이상적. "일단 의심되면 다 잡는" 모델
- Precision 97.7% — 정상인데 궤양병으로 오판한 것이 5/255장 (약 2%). 2차 검수 단계에서 걸러내면 OK
- **2 epoch에 수렴**한 이유: 데이터가 시각적으로 구분이 명확 + ImageNet pretrained가 이미 "질감 차이"를 인식함
- AUC 0.999 → 거의 완벽한 ranking 성능

### 5.2 P2 Detection (YOLOv8s, 5 epochs on MPS)

```
mAP@0.5       : 0.994
mAP@0.5:0.95  : 0.989
Normal  AP@0.5 : 0.994  (P=1.000, R=0.916)
Canker  AP@0.5 : 0.995  (P=0.959, R=1.000)
```

#### 해석
- mAP@0.5:0.95가 0.989 — IoU threshold가 엄격해도 유지됨 → bbox 위치가 **매우 정확**
- 궤양병 recall 100% 재확인 (P1과 일관)
- **하지만 해석상 주의**: 이미지에 감귤이 1개만 있고 배경이 흰색이라 **검출 자체는 거의 trivial**. 이 성능은 "위치 찾기 + 분류"라기보다 "단일 객체 전체가 감귤인 것을 확인하고 분류"에 가까움. §6 한계 참조.

### 5.3 P3 Segmentation (U-Net + ResNet34, 3 epochs on MPS)

```
mIoU                 : 0.9403
Pixel accuracy       : 99.26%
Per-class IoU:
  배경               : 0.998
  정상 감귤          : 0.937
  궤양병 감귤        : 0.886
Per-class Dice:
  배경               : 0.999
  정상 감귤          : 0.967
  궤양병 감귤        : 0.940
```

#### 해석
- 전체 mIoU 0.940 — 3 epoch에 이 정도면 우수
- **클래스별 격차**: 배경(0.998) > 정상(0.937) > 궤양병(0.886)
  - 배경이 거의 완벽한 이유: 흰색 균일
  - 궤양병이 가장 낮은 이유: 정상 부위와 **병변 부위가 한 과일에 섞여 있어** 경계가 모호. Polygon은 과일 외곽선이지 병변 경계가 아니라서 학습 어려움
- Pixel accuracy 99.26%는 배경 픽셀이 압도적이라 부풀려짐 — mIoU가 더 신뢰할 지표

### 5.4 P1b 백본 비교 (5 models, 5 epochs each on MPS)

| Model | Params | Acc | F1 (궤양병) | AUC | Latency bs=1 (ms) | Throughput bs=32 (FPS) |
|---|---:|---:|---:|---:|---:|---:|
| **ViT-Small/16** 🏆 | 21.7M | **99.06%** | **0.988** | 0.9990 | **5.94** | 175.0 |
| ConvNeXt-Tiny | 27.8M | 98.83% | 0.986 | **0.9994** | 7.18 | 109.8 |
| ResNet50 | 23.5M | 98.83% | 0.986 | 0.9975 | 27.86 | 83.8 |
| MobileNetV3-L | 4.2M | 97.66% | 0.971 | 0.9954 | 13.12 | **264.6** |
| EfficientNet-B0 | **4.0M** | 97.42% | 0.968 | 0.9902 | 19.68 | 160.6 |

#### 핵심 관찰

1. **모든 모델이 97% 이상** — 데이터가 쉬움 → 정확도 차이는 미미. **실질 차별화는 속도/크기**에서 발생
2. **ViT-Small 종합 우승** — 정확도도 최고, latency도 최저(ResNet50 대비 4.7배 빠름)
   - 이유 추정: ViT는 M4 Neural Engine의 행렬 연산 패턴에 잘 맞음. MPS 상에서 attention이 conv보다 효율적
3. **MobileNetV3는 throughput 왕** — 264 FPS (ResNet50의 3.2배). 파라미터 4.2M → 모바일/엣지 후보
4. **ResNet50 (baseline)이 가장 느림** — 레거시 아키텍처가 드러남. 현대 backbone으로 대체할 여지가 큼
5. **Pareto optimal**: ViT-Small (다른 모델에 F1·latency 모두 우월하지 않음). ConvNeXt는 최고 AUC지만 latency에서 ViT에 밀림

#### 배포 시나리오별 추천
| 시나리오 | 추천 모델 | 이유 |
|---|---|---|
| 고정확도 서버 | ViT-Small | F1/latency 모두 최상 |
| 모바일 / 엣지 | MobileNetV3-L | 4.2M params, 264 FPS |
| 저메모리 환경 (≤10M) | EfficientNet-B0 | 4.0M params, 적정 성능 |
| 레거시 호환 | ResNet50 | 가장 보편적, 도구 지원 광범위 |

상세 분석은 [`docs/analysis/05_backbone_comparison.ipynb`](docs/analysis/05_backbone_comparison.ipynb) (실행 완료, GitHub에서 바로 렌더링).

---

## 6. 한계 & 주의사항

### 🚨 Polygon 라벨의 의미적 한계
AI Hub 데이터셋의 polygon은 **감귤 열매의 외곽선**이지 **병변(lesion) 영역이 아님**. 즉:
- 현재 Segmentation은 "과일 vs 배경 + 과일 전체의 병변 여부"를 학습
- **픽셀 단위로 '이 점은 병변'인지 알려주지 않음**
- 진정한 의미의 "병변 분할"이 필요하다면 별도 라벨링 작업이 선행되어야 함

### 🚨 데이터가 쉬움
- **배경이 흰색으로 사전 전처리**되어 있음 → 자연 환경 배경에 대한 robustness 미검증
- 이미지당 감귤이 1개 → Detection 난이도가 낮음 (다중 객체 장면 필요)
- 클래스 2개만 (binary) → 다른 병해(검은점병 등)와 구분하는 성능은 미확인

### 🚨 도메인 Shift 미검증
- 카메라 3종(Samsung·Xiaomi·LGE), 지역 10곳, 노지/온실 섞임
- 현재는 **random split** (시기/카메라 고려 안 함) → 실제 배포에서 성능 저하 가능
- 개선 방향: **leave-one-camera-out** / **leave-one-location-out** 교차검증

### 🚨 초단기 학습
- 2~5 epoch만 학습 (실험 목적). 30+ epoch로 돌리면 성능이 더 나올 가능성
- Early stopping, LR 탐색 등 hyperparameter tuning 생략

### 🚨 MPS 특이사항
- macOS Apple Silicon MPS는 일부 연산이 CPU fallback되며 경고 출력
- 파라미터·성능은 일관되나 **latency 벤치마크는 MPS ↔ CUDA ↔ CPU 간 직접 비교 어려움**
- 여기 수치는 "M4 MPS 환경에서의 상대 비교"로만 해석

---

## 7. 실행 방법

모든 명령은 **프로젝트 루트**에서 실행. 사전에 conda env 활성화 및 `KMP_DUPLICATE_LIB_OK=TRUE` 설정 필요 (§8 참조).

### P1 — Classification
```bash
# 전체 데이터 학습 (기본: 30 epoch, MPS)
python classification/train.py --config classification/config.yaml

# 빠른 스모크 학습 (2 epoch)
python classification/train.py --config classification/config.yaml --override train.epochs=2

# 최고 체크포인트 평가
python classification/eval.py --config classification/config.yaml \
    --ckpt outputs/classification/<run>/ckpt/best.pt

# 학습 곡선 확인
tensorboard --logdir outputs/classification
```

**출력** (`outputs/classification/<timestamp>/`): `ckpt/best.pt`, `ckpt/last.pt`, `train.log`, `tb/`, `config.yaml`, `confusion_matrix.png`, `metrics.json`

### P2 — Detection
```bash
# 1회 변환: AI Hub polygon → YOLO 포맷 (787장)
python -m detection.prepare_yolo --source database --dest detection/data

# 학습
python -m detection.train --config detection/config.yaml

# 빠른 스모크 (5 epoch)
python -m detection.train --config detection/config.yaml --override train.epochs=5

# 평가
python -m detection.eval --config detection/config.yaml \
    --ckpt outputs/detection/run/weights/best.pt
```

**출력** (`outputs/detection/run*/`): `weights/best.pt`, `results.csv`, `results.png`, `confusion_matrix.png`, P/R 곡선

### P3 — Segmentation
```bash
# 학습
python -m segmentation.train --config segmentation/config.yaml

# 빠른 스모크 (3 epoch)
python -m segmentation.train --config segmentation/config.yaml --override train.epochs=3

# 평가 + 정성 샘플 4개 저장
python -m segmentation.eval --config segmentation/config.yaml \
    --ckpt outputs/segmentation/run/<timestamp>/ckpt/best.pt --samples 4
```

**출력** (`outputs/segmentation/run/<timestamp>/`): `ckpt/best.pt`, `metrics.json` (mIoU, per-class IoU/Dice, pixel CM), `qualitative/sample_*.png` (원본|GT|예측 병렬 시각화)

### P1b — 백본 비교
```bash
python -m classification.compare --config classification/compare_config.yaml
```
**출력**: `<run>/comparison.csv`, `comparison.md`, `comparison.json`. 각 모델별 학습·평가·벤치마크 자동 수행.

---

## 8. 프로젝트 구조 & 환경

### 환경 설정
```bash
conda env create -f environment.yml
conda activate disease_01
```

#### ⚠️ macOS OpenMP 우회 필수
numpy / torch / opencv 가 각자 OpenMP를 링크하여 실행 시 충돌:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```
`~/.zshrc` 등에 추가하거나 실행 스크립트 상단에 둘 것.

### 테스트
```bash
pytest          # 전체 85개 테스트 (P0: 24 + P1: 19 + P2: 13 + P3: 19 + P1b: 10)
```

### 실제 데이터 스모크 테스트
```bash
python scripts/smoke_test_datasets.py
```
예상 출력: Classification train 3,407 / val 427, Segmentation train 699 / val 88.

---

## 9. 분석 노트북 & 설계 문서

### Jupyter 분석 노트북 (`docs/analysis/`)
GitHub에서 바로 렌더링 가능. 로컬에서는 conda env 활성화 후 재실행.

| # | 노트북 | 내용 |
|---|---|---|
| 01 | [`01_dataset_analysis.ipynb`](docs/analysis/01_dataset_analysis.ipynb) | AI Hub EDA — 클래스 분포, polygon 커버리지, 메타데이터 |
| 02 | [`02_classification_results.ipynb`](docs/analysis/02_classification_results.ipynb) | P1 심층 — 학습 곡선, CM 해석, 오분류 샘플, ROC |
| 03 | [`03_detection_results.ipynb`](docs/analysis/03_detection_results.ipynb) | P2 — bbox 시각화, per-class AP, 실패 케이스 |
| 04 | [`04_segmentation_results.ipynb`](docs/analysis/04_segmentation_results.ipynb) | P3 — mask 시각화, IoU/Dice, 실패 케이스 |
| 05 | [`05_backbone_comparison.ipynb`](docs/analysis/05_backbone_comparison.ipynb) | P1b — Pareto, 배포 시나리오 (**실행 완료**) |
| 06 | [`06_lessons_learned.md`](docs/analysis/06_lessons_learned.md) | 설계 근거, 트러블슈팅, 다음 계획 |

각 노트북 하단에 **"📝 Your turn"** 섹션으로 분석 노트를 직접 추가할 수 있음.

### 설계 및 계획 문서
- 전체 설계 스펙: [`docs/superpowers/specs/2026-04-17-citrus-disease-cv-design.md`](docs/superpowers/specs/2026-04-17-citrus-disease-cv-design.md)
- Phase별 구현 계획: [`docs/superpowers/plans/`](docs/superpowers/plans/)
- 실험 결과 아카이브: [`docs/results/`](docs/results/)

---

## 10. 진행 상황 & 향후 계획

### 완료
- [x] **P0** — Common module (data loading, label parsing, config, utils)
- [x] **P1** — Classification (ResNet50)
- [x] **P2** — Detection (YOLOv8s)
- [x] **P3** — Segmentation (smp U-Net)
- [x] **P1b** — Classification 백본 비교 (5 models)
- [x] **Jupyter 분석 노트북** 6종
- [x] GitHub 공개 (보안 정리 완료)

### 예정
- [ ] **P2b** — Detection variants 비교 (YOLOv8 n/s/m/l)
- [ ] **P3b** — Segmentation encoder/decoder 조합 비교 (Unet/DeepLabV3+/FPN × resnet34/efficientnet-b0/mobilenet_v2)
- [ ] **멀티모달 학습** — 이미지 + 환경 feature(기온/습도/일사량) concat → 성능 개선 확인
- [ ] **도메인 일반화** — 카메라/지역 기반 cross-device 평가
- [ ] **병변 단위 라벨링** — 과일 외곽선이 아닌 canker lesion 자체 분할
- [ ] **배포** — CoreML export, 모바일 앱 연동

---

## 데이터셋 출처

[AI Hub 감귤 병충해 영상 데이터](https://aihub.or.kr/) — 온주밀감 정상/궤양병 이미지. 라이선스상 저장소에는 포함하지 않음.

## 라이선스

본 저장소의 코드는 학습·연구 목적 참고용.
