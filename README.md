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

## Phase status

- [x] P0 — Common module
- [ ] P1 — Classification
- [ ] P2 — Detection
- [ ] P3 — Segmentation
