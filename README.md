# disease_01 — 감귤 궤양병 CV 파이프라인

감귤(온주밀감) **정상 vs 궤양병** 감별을 위한 Classification / Detection / Segmentation 엔드투엔드 파이프라인.

- **데이터**: AI Hub 감귤 병충해 데이터셋 (3,834장 — 정상 2,290 / 궤양병 1,544)
- **하드웨어**: Apple Silicon (M4, MPS) 기준
- **태스크**: 세 가지 CV 태스크를 공통 모듈 위에 구현

## 주요 성과

| Phase | 태스크 | 모델 | 핵심 지표 |
|---|---|---|---|
| P1 | Classification | ResNet50 (2 epochs) | **Acc 98.83%**, F1 0.986, AUC 0.999 |
| P2 | Detection | YOLOv8s (5 epochs) | **mAP@0.5 0.994**, 궤양병 AP 0.995 |
| P3 | Segmentation | smp U-Net + ResNet34 (3 epochs) | **mIoU 0.940**, 궤양병 IoU 0.886 |
| P1b | Backbone 비교 | 5종 비교 | ViT-Small/16 종합 우승 (F1 0.988, latency 5.94ms) |

## 프로젝트 구조

```
disease_01/
├── common/             # 공통 모듈 (dataset, label_parser, config, utils)
├── classification/     # P1 — 이진 분류 (정상/궤양병)
├── detection/          # P2 — YOLOv8 객체 검출
├── segmentation/       # P3 — 3-class semantic segmentation
├── scripts/            # 유틸리티 스크립트 (smoke test 등)
├── database/           # AI Hub 데이터 (gitignored, 로컬만)
├── docs/
│   ├── superpowers/    # 설계 스펙 및 구현 계획
│   └── results/        # 실험 결과 기록
├── _archive/           # 기존 코드 보관 (gitignored)
└── environment.yml
```

## 환경 설정

```bash
conda env create -f environment.yml
conda activate disease_01
```

### ⚠️ macOS OpenMP 우회 필수

numpy / torch / opencv 라이브러리가 각자 OpenMP를 링크해서 실행 시 충돌이 발생합니다. 실행 전 환경변수 설정:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

쉘에 영구 적용하려면 `~/.zshrc` 등에 추가하세요.

## 테스트

```bash
pytest
```

총 85개 테스트 (P0: 24 + P1: 19 + P2: 13 + P3: 19 + P1b: 10).

## 실제 데이터 스모크 테스트

```bash
python scripts/smoke_test_datasets.py
```

예상 출력:
- Classification: train 3,407 / val 427
- Segmentation: train 699 / val 88 (polygon 라벨 있는 것만)

## P1 — Classification (ResNet50)

모든 명령은 프로젝트 루트에서 실행합니다.

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

출력 경로: `outputs/classification/<timestamp>/`
- `ckpt/best.pt`, `ckpt/last.pt` — 모델 체크포인트
- `train.log` — epoch별 로그
- `tb/` — TensorBoard 이벤트
- `config.yaml` — 실행 시점 config 스냅샷
- `confusion_matrix.png`, `metrics.json` — `eval.py` 산출물

## P2 — Detection (YOLOv8s)

`common` 모듈 import를 위해 `-m` 형식으로 실행합니다.

```bash
# 1회 변환: AI Hub polygon을 YOLO 포맷으로 (약 787장)
python -m detection.prepare_yolo --source database --dest detection/data

# 학습
python -m detection.train --config detection/config.yaml

# 빠른 스모크 (5 epoch)
python -m detection.train --config detection/config.yaml --override train.epochs=5

# 최고 체크포인트 평가
python -m detection.eval --config detection/config.yaml \
    --ckpt outputs/detection/run/weights/best.pt
```

출력 경로: `outputs/detection/run*/`
- `weights/best.pt`, `weights/last.pt`
- `results.csv`, `results.png`
- `confusion_matrix.png`, P/R 곡선

## P3 — Segmentation (smp U-Net)

3-class semantic segmentation (**배경 / 정상 과일 / 궤양병 과일**).
Polygon 라벨이 있는 787장만 사용 (train 699 / val 88).

```bash
# 학습
python -m segmentation.train --config segmentation/config.yaml

# 빠른 스모크 (3 epoch)
python -m segmentation.train --config segmentation/config.yaml --override train.epochs=3

# 최고 체크포인트 평가 및 정성 샘플 4개 저장
python -m segmentation.eval --config segmentation/config.yaml \
    --ckpt outputs/segmentation/run/<timestamp>/ckpt/best.pt --samples 4
```

출력 경로: `outputs/segmentation/run/<timestamp>/`
- `ckpt/best.pt`, `ckpt/last.pt`
- `train.log`, `tb/` (TensorBoard)
- `config.yaml` 스냅샷
- `metrics.json` — mIoU, 클래스별 IoU/Dice, pixel accuracy, 픽셀 confusion matrix
- `qualitative/sample_XXX.png` — 원본 | GT 마스크 | 예측 마스크 병렬 시각화

## P1b — Classification 백본 비교

5개 백본(ResNet50 / EfficientNet-B0 / ConvNeXt-Tiny / MobileNetV3-Large / ViT-Small/16)을 **동일 조건**(같은 데이터 / optimizer / 5 epoch)으로 학습하고, 정확도 · 추론 속도 · 파라미터 수를 비교합니다.

```bash
python -m classification.compare --config classification/compare_config.yaml
```

출력 경로: `outputs/classification_compare/compare/<timestamp>/`
- `<model_name>/run/<sub-timestamp>/` — 모델별 학습/평가 아티팩트
- `comparison.csv`, `comparison.md`, `comparison.json` — 통합 리포트

**최신 결과**: [`docs/results/2026-04-19-classification-comparison.md`](docs/results/2026-04-19-classification-comparison.md)

요약:
| Model | Params | Acc | F1 (궤양병) | Latency (ms) | FPS |
|---|---:|---:|---:|---:|---:|
| **ViT-Small/16** 🏆 | 21.7M | **99.06%** | **0.988** | **5.94** | 175.0 |
| ConvNeXt-Tiny | 27.8M | 98.83% | 0.986 | 7.18 | 109.8 |
| ResNet50 | 23.5M | 98.83% | 0.986 | 27.86 | 83.8 |
| MobileNetV3-L | 4.2M | 97.66% | 0.971 | 13.12 | **264.6** |
| EfficientNet-B0 | **4.0M** | 97.42% | 0.968 | 19.68 | 160.6 |

## 설계 및 계획 문서

- 전체 설계: [`docs/superpowers/specs/2026-04-17-citrus-disease-cv-design.md`](docs/superpowers/specs/2026-04-17-citrus-disease-cv-design.md)
- 구현 계획 (Phase별): [`docs/superpowers/plans/`](docs/superpowers/plans/)

## 진행 상황

- [x] P0 — Common module
- [x] P1 — Classification
- [x] P2 — Detection
- [x] P3 — Segmentation
- [x] P1b — Classification 백본 비교
- [ ] P2b — Detection variants 비교 (YOLOv8 n/s/m 등)
- [ ] P3b — Segmentation encoder/decoder 조합 비교

## 데이터셋 출처

[AI Hub 감귤 병충해 영상 데이터](https://aihub.or.kr/) — 온주밀감 정상/궤양병 이미지. 라이선스상 저장소에는 포함하지 않습니다.

## 라이선스

본 저장소의 코드는 학습·연구 목적 참고용입니다.
