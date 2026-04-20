#!/bin/bash
# Overnight orchestrator: Phase 2 (A1 long-training) + Phase 3 (G quant, H distill)
#
# Launch (detached, persists across ssh disconnects):
#   cd /Users/khj/YBNML_macmini/disease_01
#   nohup caffeinate -i bash scripts/overnight_run.sh > /tmp/overnight_orchestrator.log 2>&1 & disown

# Don't set -e — we want to continue through failures and report each step's status
set -u

source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate disease_01
export KMP_DUPLICATE_LIB_OK=TRUE

cd /Users/khj/YBNML_macmini/disease_01
mkdir -p docs/results

log_header() {
    echo ""
    echo "=========================================="
    echo ">>> $1"
    echo "Start: $(date)"
    echo "=========================================="
}

log_footer() {
    echo "End: $(date)"
    echo "=========================================="
}

echo "#### OVERNIGHT ORCHESTRATOR START: $(date) ####"

# ---------- A1: yolov8m 30-epoch long training ----------
log_header "A1: yolov8m long training (30 epochs)"
if python -m detection.train --config detection/long_config.yaml 2>&1; then
    echo "A1 OK"
    # Ultralytics writes under <project>/<name>/ — resolve and archive results
    A1_RUN=$(ls -td outputs/detection_long/yolov8m_30ep*/ 2>/dev/null | head -1)
    if [ -n "$A1_RUN" ] && [ -f "$A1_RUN/results.csv" ]; then
        cp "$A1_RUN/results.csv" docs/results/2026-04-20-yolov8m-long-30ep.csv
        # Copy final-epoch summary row + headers
        { head -1 "$A1_RUN/results.csv"; tail -1 "$A1_RUN/results.csv"; } \
            > docs/results/2026-04-20-yolov8m-long-30ep-final.txt
        echo "A1 results → docs/results/2026-04-20-yolov8m-long-30ep.*"
    else
        echo "A1 results NOT FOUND under outputs/detection_long/"
    fi
else
    echo "A1 FAILED (non-zero exit)"
fi
log_footer

# ---------- A2: DeepLabV3+ 20-epoch long training ----------
log_header "A2: DeepLabV3+ long training (20 epochs)"
if python -m segmentation.train --config segmentation/long_config.yaml 2>&1; then
    echo "A2 train OK"
    A2_RUN=$(ls -td outputs/segmentation_long/run/*/ 2>/dev/null | head -1)
    if [ -n "$A2_RUN" ] && [ -f "$A2_RUN/ckpt/best.pt" ]; then
        cp "$A2_RUN/train.log" docs/results/2026-04-20-deeplabv3plus-long-20ep.log
        # Run eval to produce metrics.json
        if python -m segmentation.eval \
            --config segmentation/long_config.yaml \
            --ckpt "$A2_RUN/ckpt/best.pt" --samples 4 2>&1; then
            cp "$A2_RUN/metrics.json" docs/results/2026-04-20-deeplabv3plus-long-20ep.json 2>/dev/null \
                || echo "A2 metrics.json missing"
            echo "A2 results → docs/results/2026-04-20-deeplabv3plus-long-20ep.*"
        else
            echo "A2 eval FAILED"
        fi
    else
        echo "A2 best.pt NOT FOUND"
    fi
else
    echo "A2 FAILED"
fi
log_footer

# ---------- G: ViT-Small INT8 quantization ----------
log_header "G: ViT-Small INT8 quantization"
VIT_BEST=$(ls outputs/classification_compare/compare/2026-04-19_01-13-33/vit_small_patch16_224/run/*/ckpt/best.pt 2>/dev/null | head -1)
if [ -n "$VIT_BEST" ] && [ -f "$VIT_BEST" ]; then
    echo "Using VIT checkpoint: $VIT_BEST"
    if python scripts/quantize_and_benchmark.py \
        --ckpt "$VIT_BEST" \
        --config classification/config.yaml \
        --out docs/results/2026-04-20-quantization.md 2>&1; then
        echo "G OK → docs/results/2026-04-20-quantization.md"
    else
        echo "G FAILED"
    fi
else
    echo "G SKIPPED: ViT-Small best.pt not found at expected path"
fi
log_footer

# ---------- H: ViT-Small → MobileNetV3 knowledge distillation ----------
log_header "H: ViT-Small → MobileNetV3 knowledge distillation"
if [ -n "$VIT_BEST" ] && [ -f "$VIT_BEST" ]; then
    if python -m distillation.train \
        --config distillation/config.yaml \
        --teacher-ckpt "$VIT_BEST" 2>&1; then
        echo "H train OK"
        H_RUN=$(ls -td outputs/distillation/run/*/ 2>/dev/null | head -1)
        if [ -n "$H_RUN" ] && [ -f "$H_RUN/ckpt/best.pt" ]; then
            cp "$H_RUN/train.log" docs/results/2026-04-20-distillation.log
            # Capture last epoch's metrics from train log (last "epoch X/Y" line)
            grep "epoch " "$H_RUN/train.log" | tail -5 \
                > docs/results/2026-04-20-distillation-summary.txt
            echo "H results → docs/results/2026-04-20-distillation.*"
        fi
    else
        echo "H FAILED"
    fi
else
    echo "H SKIPPED: no teacher ckpt"
fi
log_footer

echo ""
echo "#### OVERNIGHT ORCHESTRATOR END: $(date) ####"
echo ""
echo "Result files produced:"
ls -la docs/results/2026-04-20-* 2>/dev/null | sed 's|^|  |'
