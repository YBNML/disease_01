#!/bin/bash
# Retry just G (quantization) + H (distillation) after torch.load weights_only=True fix.
#
# Launch:
#   nohup caffeinate -i bash scripts/overnight_gh_retry.sh > /tmp/overnight_gh_retry.log 2>&1 & disown

set -u
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate disease_01
export KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/khj/YBNML_macmini/disease_01
mkdir -p docs/results

VIT_BEST=$(ls outputs/classification_compare/compare/2026-04-19_01-13-33/vit_small_patch16_224/run/*/ckpt/best.pt 2>/dev/null | head -1)

echo "#### G+H RETRY START: $(date) ####"

# ---------- G ----------
echo ""
echo ">>> G: ViT-Small INT8 quantization"
echo "Start: $(date)"
if [ -n "$VIT_BEST" ]; then
    python scripts/quantize_and_benchmark.py \
        --ckpt "$VIT_BEST" \
        --config classification/config.yaml \
        --out docs/results/2026-04-20-quantization.md 2>&1 \
        && echo "G OK" || echo "G FAILED"
fi
echo "End: $(date)"

# ---------- H ----------
echo ""
echo ">>> H: ViT-Small → MobileNetV3 distillation"
echo "Start: $(date)"
if [ -n "$VIT_BEST" ]; then
    python -m distillation.train \
        --config distillation/config.yaml \
        --teacher-ckpt "$VIT_BEST" 2>&1 \
        && echo "H train OK" || echo "H FAILED"
    H_RUN=$(ls -td outputs/distillation/run/*/ 2>/dev/null | head -1)
    if [ -n "$H_RUN" ] && [ -f "$H_RUN/ckpt/best.pt" ]; then
        cp "$H_RUN/train.log" docs/results/2026-04-20-distillation.log
        grep "epoch " "$H_RUN/train.log" | tail -5 \
            > docs/results/2026-04-20-distillation-summary.txt
    fi
fi
echo "End: $(date)"

echo ""
echo "#### G+H RETRY END: $(date) ####"
ls -la docs/results/2026-04-20-{quantization,distillation}* 2>/dev/null
