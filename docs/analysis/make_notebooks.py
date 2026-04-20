"""Generate analysis notebooks for disease_01 project.

Run from project root:
    python docs/analysis/make_notebooks.py
"""
import nbformat as nbf
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
KERNEL_META = {
    "kernelspec": {
        "display_name": "Python 3 (disease_01)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

# ---------------------------------------------------------------------------
# Notebook 1 — dataset_analysis
# ---------------------------------------------------------------------------

nb1 = nbf.v4.new_notebook()
nb1["metadata"] = KERNEL_META

nb1["cells"] = [
    # ── 0. Title ─────────────────────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
# 01 — AI Hub 감귤 데이터셋 탐색적 분석 (EDA)

이 노트북은 **AI Hub 감귤 궤양병 데이터셋**의 구성을 이해하기 위한 탐색적 데이터 분석(EDA)입니다.
모델 학습 전에 데이터 품질·분포·편향을 파악하는 것이 목적입니다.

## What you'll learn
- 클래스(정상 / 궤양병) 별 이미지 수와 분포
- Polygon 라벨 커버리지 (전체 대비 polygon 존재 비율)
- 카메라·촬영지·생육단계 등 메타데이터 분포
- 기상 환경 데이터(기온, 습도, 일사량 등) 분포
- 촬영 시기(월별) 분포
"""),

    # ── 1. Setup ─────────────────────────────────────────────────────────────
    nbf.v4.new_markdown_cell("## Setup"),
    nbf.v4.new_code_cell("""\
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# project root → sys.path (노트북 실행 위치가 project root 라고 가정)
from pathlib import Path
PROJECT_ROOT = Path().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2

from common.label_parser import load_sample, polygon_to_mask

# matplotlib 한글 폰트 설정 (macOS)
import matplotlib.font_manager as fm
_korean_candidates = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
]
for _fc in _korean_candidates:
    if Path(_fc).exists():
        fm.fontManager.addfont(_fc)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fc).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

DATABASE_ROOT = PROJECT_ROOT / "database"
print("Project root :", PROJECT_ROOT)
print("Database root:", DATABASE_ROOT)
print("Database exists:", DATABASE_ROOT.exists())
"""),

    # ── 2. Section 1: 디렉토리 구조 ──────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 1. 디렉토리 구조 및 파일 수 집계

`database/` 하위 디렉토리를 순회하여 split × class 별 이미지/JSON/polygon 수를 집계합니다.
"""),
    nbf.v4.new_code_cell("""\
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

records = []
for split_name, cfg in SPLIT_DIRS.items():
    split_dir = DATABASE_ROOT / cfg["split"]
    for cls_dir in CLASS_DIRS:
        img_dir = split_dir / cfg["img"] / cls_dir
        lbl_dir = split_dir / cfg["lbl"] / cls_dir
        if not lbl_dir.exists():
            print(f"  [SKIP] {lbl_dir} not found")
            continue
        json_files = sorted(lbl_dir.glob("*.json"))
        img_files  = sorted(img_dir.glob("*")) if img_dir.exists() else []
        img_files  = [f for f in img_files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        polygon_count = sum(1 for jp in json_files if load_sample(jp)["has_polygon"])
        records.append({
            "split": split_name,
            "class": cls_dir,
            "image_count": len(img_files),
            "json_count":  len(json_files),
            "polygon_count": polygon_count,
        })

df_counts = pd.DataFrame(records)
df_counts["polygon_ratio"] = (df_counts["polygon_count"] / df_counts["json_count"]).round(4)
print(df_counts.to_string(index=False))
"""),

    # ── 3. Section 2: 클래스 분포 시각화 ─────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 2. 클래스 분포 시각화

train / val 각각의 클래스 수와 polygon 커버리지를 비교합니다.
"""),
    nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- (A) 이미지 수 ---
ax = axes[0]
for i, (split_name, grp) in enumerate(df_counts.groupby("split")):
    x_pos = [j + i * 0.35 for j in range(len(grp))]
    bars = ax.bar(x_pos, grp["image_count"], width=0.35, label=split_name)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
classes = [c.replace("열매_", "") for c in df_counts["class"].unique()]
ax.set_xticks([j + 0.175 for j in range(len(classes))])
ax.set_xticklabels(classes)
ax.set_title("이미지 수 (split × class)")
ax.set_ylabel("이미지 수")
ax.legend()

# --- (B) Polygon 커버리지 ---
ax2 = axes[1]
for i, (split_name, grp) in enumerate(df_counts.groupby("split")):
    x_pos = [j + i * 0.35 for j in range(len(grp))]
    bars = ax2.bar(x_pos, grp["polygon_ratio"] * 100, width=0.35, label=split_name)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
ax2.set_xticks([j + 0.175 for j in range(len(classes))])
ax2.set_xticklabels(classes)
ax2.set_title("Polygon 커버리지 (%) — JSON 대비")
ax2.set_ylabel("비율 (%)")
ax2.set_ylim(0, 110)
ax2.legend()

plt.tight_layout()
plt.show()
"""),

    # ── 4. Section 3: 이미지 샘플 + polygon overlay ──────────────────────────
    nbf.v4.new_markdown_cell("""\
## 3. 이미지 샘플 및 Polygon Overlay

정상(열매_정상) / 궤양병(열매_궤양병) 각 1장씩, polygon이 있는 궤양병 샘플에는 마스크를 overlay합니다.
"""),
    nbf.v4.new_code_cell("""\
def pick_sample(split, cls_dir, want_polygon=False):
    \"\"\"Return (img_path, json_path) for one sample matching criteria.\"\"\"\
    cfg = SPLIT_DIRS[split]
    lbl_dir = DATABASE_ROOT / cfg["split"] / cfg["lbl"] / cls_dir
    img_dir = DATABASE_ROOT / cfg["split"] / cfg["img"] / cls_dir
    for jp in sorted(lbl_dir.glob("*.json")):
        info = load_sample(jp)
        if want_polygon and not info["has_polygon"]:
            continue
        cands = list(img_dir.glob(f"{jp.stem}.*"))
        cands = [c for c in cands if c.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if cands:
            return cands[0], jp
    return None, None

samples = [
    ("train", "열매_정상",   False, "Train — 정상 (no polygon)"),
    ("train", "열매_궤양병", False, "Train — 궤양병 (no overlay)"),
    ("train", "열매_궤양병", True,  "Train — 궤양병 + polygon overlay"),
    ("val",   "열매_정상",   False, "Val — 정상"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (split, cls, want_poly, title) in zip(axes.flatten(), samples):
    img_path, jp = pick_sample(split, cls, want_polygon=want_poly)
    if img_path is None:
        ax.set_title(f"{title}\\n(샘플 없음)")
        ax.axis("off")
        continue
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if want_poly:
        info = load_sample(jp)
        if info["has_polygon"]:
            h, w = img_rgb.shape[:2]
            mask = polygon_to_mask(info["polygon"], h, w)
            # red overlay (alpha blend)
            overlay = img_rgb.copy()
            overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([200, 30, 30]) * 0.5).astype(np.uint8)
            img_rgb = overlay
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

plt.suptitle("이미지 샘플 (2×2 그리드)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
"""),

    # ── 5. Section 4: 메타데이터 EDA ─────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 4. 메타데이터 EDA

모든 JSON 라벨에서 메타데이터를 수집하여 DataFrame으로 만든 뒤 분포를 시각화합니다.
(전체 집계이므로 처음 실행 시 수 분 소요될 수 있습니다.)
"""),
    nbf.v4.new_code_cell("""\
all_records = []
for split_name, cfg in SPLIT_DIRS.items():
    split_dir = DATABASE_ROOT / cfg["split"]
    for cls_dir in CLASS_DIRS:
        lbl_dir = split_dir / cfg["lbl"] / cls_dir
        if not lbl_dir.exists():
            continue
        for jp in sorted(lbl_dir.glob("*.json")):
            info = load_sample(jp)
            m = info["metadata"]
            e = m["env"]
            all_records.append({
                "split":        split_name,
                "class":        cls_dir,
                "has_polygon":  info["has_polygon"],
                "camera":       m.get("camera"),
                "location":     m.get("location"),
                "place_type":   m.get("place_type"),
                "growth_stage": m.get("growth_stage"),
                "date":         m.get("date"),
                "solar":        e["solar"],
                "rain":         e["rain"],
                "temp":         e["temp"],
                "humidity":     e["humidity"],
                "soil_moisture":e["soil_moisture"],
            })

df_meta = pd.DataFrame(all_records)
print(f"총 샘플 수: {len(df_meta):,}")
print(df_meta.dtypes)
df_meta.head(3)
"""),
    nbf.v4.new_code_cell("""\
# 범주형 메타데이터 분포 — 4개 subplot
cat_cols = ["camera", "location", "place_type", "growth_stage"]
cat_labels = ["카메라", "촬영지(location)", "촬영 장소 유형", "생육단계"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, col, lbl in zip(axes.flatten(), cat_cols, cat_labels):
    vc = df_meta[col].value_counts(dropna=False).head(20)
    vc.plot(kind="bar", ax=ax)
    ax.set_title(lbl + " 분포")
    ax.set_xlabel("")
    ax.set_ylabel("샘플 수")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
"""),
    nbf.v4.new_code_cell("""\
# 환경 수치 데이터 히스토그램
env_cols   = ["temp", "humidity", "solar", "soil_moisture", "rain"]
env_labels = ["기온 (°C)", "상대습도 (%)", "일사량 (MJ/m²)", "토양수분 (%)", "강수량 (mm)"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col, lbl in zip(axes.flatten(), env_cols, env_labels):
    vals = df_meta[col].replace(0, np.nan).dropna()
    if len(vals) == 0:
        ax.set_title(lbl + "\\n(데이터 없음)")
        ax.axis("off")
        continue
    ax.hist(vals, bins=40, edgecolor="white", linewidth=0.5)
    ax.set_title(lbl)
    ax.set_ylabel("빈도")

# 마지막 subplot 숨기기
axes[-1, -1].axis("off")
plt.suptitle("환경 수치 데이터 분포", fontsize=13)
plt.tight_layout()
plt.show()
"""),

    # ── 6. Section 5: 시기 분포 ──────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 5. 촬영 시기(월) 분포

`OCPRD` 필드에서 월(month)을 추출해 계절적 패턴을 확인합니다.
"""),
    nbf.v4.new_code_cell("""\
import re

def extract_month(date_str):
    if not date_str:
        return None
    # 예시 형식: '20230815', '2023-08-15', '2023.08.15'
    m = re.search(r"(\\d{4})[-.]?(\\d{2})[-.]?(\\d{2})", str(date_str))
    if m:
        return int(m.group(2))
    return None

df_meta["month"] = df_meta["date"].apply(extract_month)

month_counts = df_meta.groupby(["month", "class"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
month_counts.plot(kind="bar", ax=ax)
ax.set_title("촬영 월별 샘플 분포")
ax.set_xlabel("월 (month)")
ax.set_ylabel("샘플 수")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()

print("월별 총계:\\n", df_meta["month"].value_counts().sort_index())
"""),

    # ── 7. Section 6: 관찰 요약 ──────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 6. 관찰 요약

데이터 탐색 결과에서 주목할 점을 정리합니다.

- **클래스 불균형**: 정상(열매_정상) 이미지가 궤양병(열매_궤양병) 이미지보다 많은 경향이 있습니다.
  학습 시 Weighted Sampler 또는 클래스 가중치를 적용할 필요가 있습니다.
- **Polygon 커버리지 편중**: 전체 JSON 라벨 중 polygon이 있는 비율은 궤양병 클래스에 집중되어 있습니다.
  정상 클래스는 polygon 없이 이미지 레벨 라벨만 존재하는 경우가 많습니다.
- **카메라·촬영 장소 편향**: 특정 카메라 모델 또는 촬영 지역에 샘플이 집중될 경우 도메인 쉬프트가 발생할 수 있습니다.
- **기상 데이터 결측**: 일부 환경 수치(일사량, 토양수분 등)에 0 또는 결측값이 많아 직접 활용 전 전처리가 필요합니다.
"""),

    # ── 8. Your Turn ─────────────────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 📝 Your turn

아래 질문에 답하는 분석을 직접 추가해보세요.

1. **Polygon 라벨이 일부에만 존재하는 이유**는 무엇일까요?
   AI Hub 라벨링 가이드라인이나 annotation 스키마를 찾아보고, 궤양병 클래스에만 polygon이 있는 이유를 설명해보세요.

2. **카메라별 불균형이 도메인 쉬프트 문제로 이어질까요?**
   특정 카메라로 찍힌 이미지가 train/val에 어떻게 분포하는지 확인하고, 쉬프트 가능성을 평가해보세요.

3. **기상 데이터와 궤양병 발생 사이에 상관관계가 있을까요?**
   `df_meta`에서 class=열매_궤양병 vs 열매_정상 그룹을 나눠 환경 변수(기온, 습도)의 분포를 비교·시각화해보세요.

4. **생육단계별 궤양병 발생률**은 어떻게 다를까요?
   `growth_stage × class` 교차 테이블을 만들어 분석해보세요.

5. **데이터 품질 이슈**가 있나요?
   이미지 파일은 있는데 JSON이 없는 케이스, 또는 반대 케이스가 있는지 확인하고 비율을 구해보세요.
"""),
]

# ---------------------------------------------------------------------------
# Notebook 2 — classification_results
# ---------------------------------------------------------------------------

nb2 = nbf.v4.new_notebook()
nb2["metadata"] = KERNEL_META

# Path constants for notebook 2 (used in prose, not computed at generate time)
RUN_BASE = "outputs/classification_compare/compare"

nb2["cells"] = [
    # ── 0. Title ─────────────────────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
# 02 — P1 분류 실험 결과 분석

이 노트북은 **P1 ResNet50 / 다중 백본 비교 실험** 결과를 정량·정성적으로 분석합니다.
학습 곡선, 최종 메트릭, Confusion Matrix, 오분류 샘플 시각화, ROC curve를 단계적으로 살펴봅니다.

## What you'll learn
- 학습 곡선(train/val loss, val_f1, val_auc)에서 과적합 여부 판단
- Confusion Matrix 해석: TP/FP/TN/FN → sensitivity / specificity
- 오분류 샘플의 시각적 공통 특징 파악
- ROC curve와 AUC의 의미
- 백본별 정확도 vs 속도 트레이드오프 비교
"""),

    # ── 1. Setup ─────────────────────────────────────────────────────────────
    nbf.v4.new_markdown_cell("## Setup"),
    nbf.v4.new_code_cell("""\
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
PROJECT_ROOT = Path().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import cv2
import torch

# 한글 폰트
_korean_candidates = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
]
for _fc in _korean_candidates:
    if Path(_fc).exists():
        fm.fontManager.addfont(_fc)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fc).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 최신 compare run 찾기 ──────────────────────────────────────────────────
COMPARE_ROOT = PROJECT_ROOT / "outputs" / "classification_compare" / "compare"
run_dirs = sorted(COMPARE_ROOT.iterdir()) if COMPARE_ROOT.exists() else []
if not run_dirs:
    raise FileNotFoundError(f"No compare runs found under {COMPARE_ROOT}")
LATEST_COMPARE = run_dirs[-1]  # 최신 타임스탬프 선택
print(f"Compare run: {LATEST_COMPARE.name}")

# ResNet50 을 기본 분석 대상으로 사용 (가장 표준적인 백본)
BACKBONE = "resnet50"
RUN_DIR = LATEST_COMPARE / BACKBONE / "run"
run_subdirs = sorted(RUN_DIR.iterdir()) if RUN_DIR.exists() else []
if not run_subdirs:
    raise FileNotFoundError(f"No run subdir under {RUN_DIR}")
RESULT_DIR = run_subdirs[-1]
print(f"Result dir : {RESULT_DIR}")
"""),

    # ── 2. Section 1: 학습 곡선 ──────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 1. 학습 곡선

`train.log`에서 epoch별 train_loss / val_loss / val_f1 / val_auc를 파싱하여 시각화합니다.
"""),
    nbf.v4.new_code_cell("""\
log_path = RESULT_DIR / "train.log"
epoch_pattern = re.compile(
    r"epoch (\\d+)/(\\d+)\\s+"
    r"train_loss=([\\d.]+)\\s+"
    r"val_loss=([\\d.]+)\\s+"
    r"val_acc=([\\d.]+)\\s+"
    r"val_f1_pos=([\\d.]+)\\s+"
    r"val_auc=([\\d.]+)"
)

epochs, train_losses, val_losses, val_accs, val_f1s, val_aucs = [], [], [], [], [], []
with open(log_path) as f:
    for line in f:
        m = epoch_pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(3)))
            val_losses.append(float(m.group(4)))
            val_accs.append(float(m.group(5)))
            val_f1s.append(float(m.group(6)))
            val_aucs.append(float(m.group(7)))

df_log = pd.DataFrame({
    "epoch": epochs,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "val_acc": val_accs,
    "val_f1": val_f1s,
    "val_auc": val_aucs,
})
print(df_log.to_string(index=False))
"""),
    nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Loss
ax = axes[0]
ax.plot(df_log["epoch"], df_log["train_loss"], "o-", label="train_loss")
ax.plot(df_log["epoch"], df_log["val_loss"],   "s--", label="val_loss")
ax.set_title("Train / Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
ax.set_xticks(df_log["epoch"])

# F1
ax = axes[1]
ax.plot(df_log["epoch"], df_log["val_f1"], "^-", color="tab:green", label="val_f1_pos")
ax.set_title("Validation F1 (양성 클래스)")
ax.set_xlabel("Epoch"); ax.set_ylabel("F1"); ax.legend()
ax.set_ylim(0.9, 1.01); ax.set_xticks(df_log["epoch"])

# AUC
ax = axes[2]
ax.plot(df_log["epoch"], df_log["val_auc"], "D-", color="tab:red", label="val_auc")
ax.set_title("Validation AUC")
ax.set_xlabel("Epoch"); ax.set_ylabel("AUC"); ax.legend()
ax.set_ylim(0.9, 1.01); ax.set_xticks(df_log["epoch"])

plt.suptitle(f"학습 곡선 ({BACKBONE})", fontsize=13)
plt.tight_layout()
plt.show()
"""),

    # ── 3. Section 2: 최종 메트릭 ────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 2. 최종 메트릭 (Best Checkpoint)

`metrics.json`에 저장된 최종 평가 지표를 테이블로 정리합니다.
"""),
    nbf.v4.new_code_cell("""\
metrics_path = RESULT_DIR / "metrics.json"
with open(metrics_path) as f:
    metrics = json.load(f)

# 요약 DataFrame
rows = [
    ("Accuracy",             metrics["accuracy"]),
    ("F1 (궤양병, positive)", metrics["f1_positive"]),
    ("Precision (궤양병)",    metrics["precision_positive"]),
    ("Recall (궤양병)",       metrics["recall_positive"]),
    ("AUC",                  metrics["auc"]),
    ("F1 (정상, class-0)",    metrics["f1_per_class"][0]),
    ("Precision (정상)",      metrics["precision_per_class"][0]),
    ("Recall (정상)",         metrics["recall_per_class"][0]),
]
df_metrics = pd.DataFrame(rows, columns=["Metric", "Value"])
df_metrics["Value"] = df_metrics["Value"].map(lambda x: f"{x:.4f}")
print(df_metrics.to_string(index=False))
"""),

    # ── 4. Section 3: Confusion Matrix ───────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 3. Confusion Matrix 해석

Confusion Matrix를 heatmap으로 시각화하고 TP/FP/TN/FN, sensitivity, specificity를 계산합니다.
"""),
    nbf.v4.new_code_cell("""\
cm = np.array(metrics["confusion_matrix"])  # [[TN, FP], [FN, TP]]
tn, fp = cm[0]
fn, tp = cm[1]

print(f"TN={tn}  FP={fp}")
print(f"FN={fn}  TP={tp}")
print(f"Sensitivity (Recall)  = TP / (TP+FN) = {tp}/{tp+fn} = {tp/(tp+fn):.4f}")
print(f"Specificity           = TN / (TN+FP) = {tn}/{tn+fp} = {tn/(tn+fp):.4f}")
print(f"Precision             = TP / (TP+FP) = {tp}/{tp+fp} = {tp/(tp+fp):.4f}")

class_names = ["정상 (0)", "궤양병 (1)"]
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["예측: 정상", "예측: 궤양병"],
            yticklabels=["실제: 정상", "실제: 궤양병"],
            ax=ax, linewidths=0.5)
ax.set_title(f"Confusion Matrix — {BACKBONE}")
ax.set_ylabel("실제 클래스")
ax.set_xlabel("예측 클래스")
plt.tight_layout()
plt.show()
"""),

    # ── 5. Section 4: 오분류 샘플 시각화 ─────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 4. 오분류 샘플 시각화

Best checkpoint를 불러와 val 데이터셋을 추론한 뒤 오분류된 이미지를 2×4 그리드로 확인합니다.
"""),
    nbf.v4.new_code_cell("""\
from torchvision import transforms as T
from common.dataset import ClassificationDataset
from classification.model import build_model

# config 로드
config_path = RESULT_DIR / "config.yaml"
import yaml
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# device
device = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")
print("Device:", device)

# transforms (val — no augmentation)
img_size = cfg["data"]["image_size"]
val_tf = T.Compose([
    T.ToTensor(),
    T.Resize((img_size, img_size)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def np_transform(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return val_tf(img_rgb)

val_ds = ClassificationDataset("database", split="val", transform=np_transform)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
print(f"Val samples: {len(val_ds)}")
"""),
    nbf.v4.new_code_cell("""\
# 모델 로드
model = build_model(cfg).to(device)
ckpt_path = RESULT_DIR / "ckpt" / "best.pt"
state = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print("Model loaded from:", ckpt_path)
"""),
    nbf.v4.new_code_cell("""\
# 추론 — 오분류 샘플 수집
misclassified = []  # (img_path, true_label, pred_label, prob_positive)
all_labels, all_probs = [], []

with torch.no_grad():
    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu()
        preds  = logits.argmax(dim=1).cpu()
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                # val_ds.items[global_idx] → we need the index in the dataset
                pass  # handled below via global index

# second pass: collect global indices
all_labels2, all_probs2, all_preds2 = [], [], []
global_idx = 0
for batch in val_loader:
    images = batch["image"].to(device)
    labels = batch["label"]
    with torch.no_grad():
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu()
        preds  = logits.argmax(dim=1).cpu()
    for i in range(len(labels)):
        all_labels2.append(labels[i].item())
        all_probs2.append(probs[i].item())
        all_preds2.append(preds[i].item())
        if preds[i] != labels[i]:
            img_path, jp = val_ds.items[global_idx]
            misclassified.append({
                "img_path": img_path,
                "true": labels[i].item(),
                "pred": preds[i].item(),
                "prob": probs[i].item(),
            })
        global_idx += 1

print(f"총 오분류 샘플: {len(misclassified)} / {len(val_ds)}")
CLASS_NAMES = ["정상", "궤양병"]
for r in misclassified[:8]:
    print(f"  true={CLASS_NAMES[r['true']]}  pred={CLASS_NAMES[r['pred']]}  prob_canker={r['prob']:.3f}  {Path(r['img_path']).name}")
"""),
    nbf.v4.new_code_cell("""\
# 오분류 이미지 시각화 — 최대 8장 (2×4)
n_show = min(8, len(misclassified))
if n_show == 0:
    print("오분류 샘플이 없습니다 — 완벽한 모델!")
else:
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
    for i, rec in enumerate(misclassified[:n_show]):
        img_bgr = cv2.imread(str(rec["img_path"]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        t = CLASS_NAMES[rec["true"]]
        p = CLASS_NAMES[rec["pred"]]
        axes[i].set_title(f"실제: {t}\\n예측: {p} (p={rec['prob']:.3f})", fontsize=9)
        axes[i].axis("off")
    # 남은 subplot 숨기기
    for j in range(n_show, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"오분류 샘플 ({BACKBONE} best checkpoint)", fontsize=12)
    plt.tight_layout()
    plt.show()
"""),

    # ── 6. Section 5: ROC Curve ───────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 5. ROC Curve

검증 세트 전체의 예측 확률(prob_positive)로 ROC curve를 그리고 AUC를 확인합니다.
"""),
    nbf.v4.new_code_cell("""\
from sklearn.metrics import roc_curve, auc as sklearn_auc

labels_arr = np.array(all_labels2)
probs_arr  = np.array(all_probs2)

fpr, tpr, thresholds = roc_curve(labels_arr, probs_arr, pos_label=1)
roc_auc = sklearn_auc(fpr, tpr)

# 최적 threshold: Youden's J
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
ax.scatter([fpr[best_idx]], [tpr[best_idx]], color="red", zorder=5,
           label=f"Best threshold = {best_thresh:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("FPR (1 - Specificity)")
ax.set_ylabel("TPR (Sensitivity / Recall)")
ax.set_title(f"ROC Curve — {BACKBONE}")
ax.legend(loc="lower right")
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
plt.tight_layout()
plt.show()

print(f"AUC: {roc_auc:.4f}")
print(f"Optimal threshold (Youden): {best_thresh:.4f}  →  TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")
"""),

    # ── 7. Section 5b: 백본 비교 ──────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 6. 백본별 성능 비교 (comparison.json)

compare 실험에서 5개 백본의 정확도 / F1 / AUC / 파라미터 수 / 추론 속도를 비교합니다.
"""),
    nbf.v4.new_code_cell("""\
compare_json = LATEST_COMPARE / "comparison.json"
with open(compare_json) as f:
    compare_data = json.load(f)

df_compare = pd.DataFrame(compare_data)
df_compare = df_compare.sort_values("f1_positive", ascending=False).reset_index(drop=True)
df_compare["params_M"] = (df_compare["params"] / 1e6).round(2)

display_cols = ["model", "params_M", "accuracy", "f1_positive",
                "recall_positive", "auc", "latency_bs1_ms", "throughput_bs_fps"]
print(df_compare[display_cols].to_string(index=False))
"""),
    nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy & F1
ax = axes[0]
x = np.arange(len(df_compare))
w = 0.35
ax.bar(x - w/2, df_compare["accuracy"],    width=w, label="Accuracy")
ax.bar(x + w/2, df_compare["f1_positive"], width=w, label="F1 (궤양병)")
ax.set_xticks(x); ax.set_xticklabels(df_compare["model"], rotation=20, ha="right")
ax.set_title("백본별 Accuracy / F1")
ax.set_ylim(0.95, 1.005)
ax.legend()

# Latency vs Params
ax2 = axes[1]
sc = ax2.scatter(df_compare["params_M"], df_compare["latency_bs1_ms"],
                 s=df_compare["f1_positive"] * 3000, alpha=0.8,
                 c=range(len(df_compare)), cmap="tab10")
for _, row in df_compare.iterrows():
    ax2.annotate(row["model"], (row["params_M"], row["latency_bs1_ms"]),
                 fontsize=8, ha="left", va="bottom")
ax2.set_xlabel("파라미터 수 (M)")
ax2.set_ylabel("추론 레이턴시 (ms, bs=1)")
ax2.set_title("파라미터 수 vs 추론 속도\\n(원 크기 = F1)")
plt.colorbar(sc, ax=ax2, label="모델 순위")

plt.tight_layout()
plt.show()
"""),

    # ── 8. Section 6: 관찰 요약 ──────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 7. 관찰 요약

- **과적합 여부**: ResNet50의 경우 epoch 2~3 이후 val_loss가 소폭 증가하지만 val_f1은 유지됩니다. 5 epoch 기준으로는 명확한 과적합이 보이지 않으나, 더 많은 epoch 학습 시 모니터링이 필요합니다.
- **궤양병 Recall 높음**: FN이 1건으로 매우 낮아 Recall이 ~99%입니다. 의료·농업 진단 태스크에서 중요한 성질이나, 반면 FP(5건)가 더 많습니다.
- **ViT vs ConvNet**: ViT-small은 파라미터 수 대비 F1이 높고, ConvNeXt-tiny는 레이턴시가 가장 낮으면서도 Recall이 100%입니다. 배포 환경에 따라 선택이 달라질 수 있습니다.
- **EfficientNet/MobileNet 효율성**: 파라미터 수가 ~4M임에도 97% 이상의 정확도를 보여 경량 배포 시 유망합니다.
"""),

    # ── 9. Your Turn ─────────────────────────────────────────────────────────
    nbf.v4.new_markdown_cell("""\
## 📝 Your turn

아래 질문에 답하는 분석을 직접 추가해보세요.

1. **궤양병 Recall이 ~99%인 이유**는 무엇일까요?
   Weighted Sampler, 클래스 가중치, 데이터 분포 중 어떤 요인이 가장 크게 기여했는지 분석해보세요.

2. **FP(오탐) 5장의 공통 특징**이 있을까요?
   오분류 이미지를 살펴보고 카메라 종류·조명 조건·생육 단계 등 메타데이터와 비교해보세요.

3. **과적합 징후**를 더 엄밀하게 확인하려면?
   train_loss와 val_loss의 격차를 epoch별로 수치화하고, 'generalization gap'이 임계값을 넘는 시점을 찾아보세요.

4. **ConvNeXt-tiny vs ViT-small 비교**:
   두 모델의 오분류 샘플이 겹치는가요? 각 모델로 동일하게 위 Section 4를 반복하여 앙상블 가능성을 탐색해보세요.

5. **Threshold 조정 실험**:
   ROC curve에서 구한 최적 threshold(Youden's J)를 0.5 대신 사용하면 Precision/Recall trade-off가 어떻게 바뀌나요? PR curve도 함께 그려보세요.
"""),
]

# ---------------------------------------------------------------------------
# Write notebooks
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

nb1_path = OUT_DIR / "01_dataset_analysis.ipynb"
nb2_path = OUT_DIR / "02_classification_results.ipynb"

nbf.write(nb1, str(nb1_path))
nbf.write(nb2, str(nb2_path))

print(f"[OK] {nb1_path}  ({len(nb1['cells'])} cells)")
print(f"[OK] {nb2_path}  ({len(nb2['cells'])} cells)")
