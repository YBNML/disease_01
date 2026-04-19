# P0~P1b 전반의 배운 점 & 다음 단계

---

## 1. 설계 결정의 배경과 근거

### 왜 3개 태스크를 공통 모듈 위에 올렸는가?

감귤 병해 데이터셋은 동일 이미지에서 분류(병해 유무), 탐지(bbox), 분할(mask)을 동시에 추출할 수 있다.
세 태스크를 별도 코드베이스로 관리하면 데이터 전처리, 학습 루프, 평가 로직이 중복되어 유지보수 비용이 3배로 늘어난다.
공통 모듈(`src/shared/`)에 데이터 로딩, 변환(augmentation), 평가 유틸리티를 집중시키고 태스크별 헤드만 분리함으로써
코드 중복을 최소화하고, 새 태스크 추가 시 헤드만 작성하면 되는 확장 구조를 확보했다.

### Classification 데이터가 왜 3,834장 전부인데 Detection/Segmentation은 787장만?

AI Hub 감귤 데이터셋은 전체 이미지 중 일부에만 polygon 라벨이 포함되어 있다.
분류 태스크는 이미지 레벨 레이블(정상/비정상)만 필요하므로 3,834장 전체를 사용할 수 있다.
반면 탐지·분할은 polygon 좌표가 있는 이미지만 사용 가능하며, 해당 이미지 수가 787장이었다.
이 비대칭은 의도적 설계가 아닌 데이터셋 원본의 라벨링 범위에 의한 제약이다.

### 왜 P1b에서 ViT-Small이 최고 성능이었는가?

- **global self-attention**: Transformer는 입력 전체의 long-range dependency를 학습한다. 과일 표면의 병반은 특정 위치에 고정되지 않으므로 global feature가 유리하다.
- **ImageNet21k pretrain**: ViT-Small은 ImageNet-1k 대비 더 풍부한 표현을 사전학습한 모델 가중치를 시작점으로 사용했다.
- **MPS 하드웨어 가속**: Apple M4 Neural Engine이 Attention 행렬 연산을 효율적으로 가속해 latency가 가장 낮았고, 이는 동일 epoch 내 더 많은 gradient update와 안정적 수렴으로 이어졌다.
- **파라미터 효율**: 21.7M으로 ResNet50(23.5M)·ConvNeXt-Tiny(27.8M)보다 작으면서 더 높은 성능을 달성했다.

### 왜 polygon을 bbox/mask로 변환해서 다용도로 썼는가?

원본 라벨은 polygon 형식이다. polygon → bbox 변환으로 YOLO 탐지 학습 데이터를 생성하고,
polygon → binary mask 변환으로 Segmentation 학습 데이터를 생성했다.
단일 소스(polygon)에서 두 포맷을 자동 생성함으로써 라벨 불일치 위험을 없애고
새 데이터가 추가될 때도 한 번의 변환 파이프라인만 실행하면 세 태스크가 동기화된다.

---

## 2. 트러블슈팅 기록

### macOS OpenMP 충돌 (`KMP_DUPLICATE_LIB_OK=TRUE`)

**증상**: PyTorch DataLoader 멀티프로세싱 실행 시 `OMP: Error #15: Initializing libomp140.x86_64.dll` 유사 충돌 발생.  
**원인**: macOS에서 Intel MKL과 libomp가 동시에 로드될 때 OpenMP 라이브러리 충돌.  
**해결**: 환경 변수 `KMP_DUPLICATE_LIB_OK=TRUE` 설정. 학습 스크립트 상단 및 Jupyter 실행 커맨드에 추가.  
**교훈**: Apple Silicon + Conda 환경에서는 항상 설정해두는 것이 안전하다.

### albumentations `A.Lambda` multiprocessing pickling 이슈

**증상**: `num_workers > 0`에서 `PicklingError: Can't pickle <function <lambda>...>` 발생.  
**원인**: `A.Lambda`에 익명 함수(lambda)를 전달하면 pickle 직렬화가 불가능해 worker 프로세스로 전달 실패.  
**해결**: lambda를 명시적 named function으로 교체. `def apply_fn(image, **kw): ...` 형태로 모듈 레벨에 정의.  
**교훈**: DataLoader worker를 사용할 경우 모든 transform은 pickle-safe해야 한다.

### YOLO output path가 ultralytics 기본 `runs_dir`로 가는 문제

**증상**: `model.train()` 실행 후 결과가 `runs/detect/train*/` 로 저장되어 프로젝트 디렉터리 구조와 불일치.  
**원인**: ultralytics는 기본적으로 현재 작업 디렉터리 기준 `runs/` 폴더에 결과를 저장.  
**해결**: `model.train(project='outputs/detection', name='yolov8n_run1')` 명시적 경로 지정.  
**교훈**: ultralytics API는 `project`/`name` 인자로 출력 경로를 완전히 제어할 수 있다.

### Python 프로세스 세션 드랍 / silent death

**증상**: SSH 세션 종료 또는 macOS 절전 후 장시간 학습 프로세스가 조용히 종료됨.  
**원인**: 터미널 세션 hang-up 시그널(SIGHUP) 전파 + macOS 자동 절전으로 프로세스 일시정지.  
**해결**:
```bash
nohup python train.py > logs/train.log 2>&1 & disown
caffeinate -i -w $!   # 절전 방지
```
**교훈**: 장시간 학습은 반드시 `nohup + disown`으로 세션과 분리하고, macOS에서는 `caffeinate`로 절전을 막아야 한다.

### config의 `database_root: ../database` 경로 실수

**증상**: 스크립트를 프로젝트 루트 외 다른 디렉터리에서 실행하면 데이터 경로를 찾지 못함.  
**원인**: YAML config의 상대 경로가 실행 위치(CWD)에 따라 달라짐.  
**해결**: config 로더에서 경로를 항상 `config 파일 위치 기준`으로 절대 경로로 변환.
```python
cfg_dir = Path(config_path).parent
database_root = (cfg_dir / cfg['database_root']).resolve()
```
**교훈**: config 파일의 상대 경로는 항상 config 파일 자신의 위치를 기준으로 resolve해야 CWD에 독립적이 된다.

---

## 3. 재현성 & 투명성 원칙

### Timestamped output + config snapshot

모든 실험 결과는 `outputs/{task}/{timestamp}/` 디렉터리에 저장하고, 실험 시작 시 사용한 config 파일을 해당 디렉터리에 함께 복사한다. 나중에 어떤 하이퍼파라미터로 학습했는지 코드를 뒤질 필요 없이 output 폴더만 열면 확인 가능하다.

### 학습 로그, TensorBoard events, 체크포인트 함께 저장

`train.log` (텍스트 로그), `events.out.tfevents.*` (TensorBoard), `best.pth` / `last.pth` (체크포인트)를 한 디렉터리에 묶어 관리한다. 실험 재현 시 체크포인트에서 resume하거나, TensorBoard로 학습 곡선을 즉시 확인할 수 있다.

### 75+ pytest tests로 모듈 단위 회귀 방지

`tests/` 아래 데이터 로더, 모델 빌더, 변환 파이프라인, 평가 함수 등 75개 이상의 단위 테스트를 작성했다. 새 기능 추가 후 `pytest` 한 번으로 기존 동작이 깨지지 않았는지 확인한다.

### atomic commit — 1 task 1 commit

50개+ 커밋이 각각 단일 논리 단위를 담당한다. 대규모 "everything" 커밋 없이 `git bisect`로 버그 도입 시점을 추적할 수 있고, PR 리뷰어가 변경 의도를 파악하기 쉽다.

---

## 4. 데이터가 말하는 것

### 모든 모델이 97%+ 정확도 → 문제가 쉬움

5개 백본 모두 97~99% 정확도를 기록했다. 정확도 차이는 최대 1.6%p에 불과해 통계적으로 유의미한 차이라고 단정하기 어렵다.

### 원인 분석

1. **배경 전처리**: AI Hub 데이터는 과일을 흰색 배경에서 촬영하거나 배경을 제거한 상태. 모델이 배경 텍스처에서 쉽게 힌트를 얻을 수 있다.
2. **polygon 라벨은 과일 외곽선**: 병반(lesion) 자체가 아닌 과일 전체 윤곽을 라벨링했다. 즉, 분류 태스크는 "이 과일이 병든 과일인가"이지 "병반 픽셀은 어디인가"가 아니다.
3. **클래스 2개**: 정상/비정상 이진 분류. 다중 병해 종류 구분이 없어 문제 난이도가 낮다.

### 결론: 차별화 포인트는 정확도가 아니라 속도/크기

모든 모델이 충분히 정확하므로 실제 배포 결정은 **latency, throughput, params**가 주도해야 한다.
더 어려운 데이터(병반 단위 라벨, 다중 병해 분류, 저화질 현장 이미지)가 도입될 때 비로소 백본 간 정확도 차이가 의미를 가질 것이다.

---

## 5. 다음 단계 (아직 미진행)

### P2b — Detection variants 비교

YOLOv8 nano/small/medium/large 4개 모델을 동일 조건에서 학습하고 mAP50, mAP50-95, latency, model size를 비교한다.
P1b와 동일한 비교 framework(compare.py + compare_config.yaml)를 재사용할 계획이다.

### P3b — Segmentation encoder/decoder 조합 비교

`segmentation_models.pytorch(smp)` 기반으로 아키텍처 × 인코더 조합을 비교한다.
- **디코더**: UNet, DeepLabV3+, FPN
- **인코더**: resnet34, efficientnet-b0, mobilenet_v2
- **지표**: mIoU, Dice, latency

### 멀티모달 학습 (Phase 2 spec)

이미지 피처와 환경 피처(기온, 습도, 일사량, 강수량 등)를 concat해 분류/탐지 성능이 향상되는지 검증한다.
환경 데이터는 촬영 메타데이터 또는 기상청 API에서 취득할 수 있다.

### 도메인 일반화

특정 제조사(Samsung) 카메라로만 학습하고 다른 제조사(Xiaomi, LGE) 카메라 이미지로 평가한다.
Cross-device robustness를 측정해 실제 농가 배포 가능성을 검증한다.

### 병변 단위 라벨링

현재 polygon은 과일 외곽선이다. 진정한 의미의 segmentation은 canker lesion 자체를 픽셀 단위로 분할해야 한다.
추가 라벨링 작업이 필요하며, 이를 통해 모델 간 성능 차이가 더 명확하게 드러날 것으로 예상된다.

### 실제 배포 파이프라인

- **CoreML export**: `torch.export` + `coremltools`로 `.mlpackage` 변환 후 iOS/macOS 앱에서 on-device 추론 검증
- **모바일 앱 연동**: SwiftUI 또는 Flutter 앱에서 CoreML 모델을 사용하는 end-to-end 파이프라인 구축

---

## 6. 참고 자료

### 데이터셋

<!-- AI Hub 감귤 병해 데이터셋: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=161 -->

### 라이브러리 공식 문서

- [timm (PyTorch Image Models)](https://timm.fast.ai/)
- [segmentation_models.pytorch (smp)](https://smp.readthedocs.io/)
- [ultralytics YOLOv8](https://docs.ultralytics.com/)
- [albumentations](https://albumentations.ai/docs/)

### 관련 논문

| 모델 | 논문 |
|------|------|
| ResNet | He et al., "Deep Residual Learning for Image Recognition," CVPR 2016 |
| EfficientNet | Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019 |
| MobileNetV3 | Howard et al., "Searching for MobileNetV3," ICCV 2019 |
| ConvNeXt | Liu et al., "A ConvNet for the 2020s," CVPR 2022 |
| ViT | Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021 |
| YOLOv8 | Jocher et al., Ultralytics YOLOv8, 2023 |
| U-Net | Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015 |
| DeepLabV3+ | Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Segmentation," ECCV 2018 |
