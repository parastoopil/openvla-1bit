#!/usr/bin/env bash
# ============================================================
# OpenVLA 1-bit Quantization — End-to-End Pipeline
# ============================================================
# Usage:
#   bash run.sh                   # full pipeline (quantize + eval)
#   bash run.sh --skip_quant      # FP16 perplexity baseline only
#   bash run.sh --fast            # 32 calib samples, magnitude metric
#   bash run.sh --eval_action     # also measure action accuracy
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ── Parse convenience flags ─────────────────────────────────
SKIP_QUANT=""
EVAL_ACTION=""
NSAMPLES=128
METRIC="hessian"
DEVICE="cuda:0"

for arg in "$@"; do
  case $arg in
    --skip_quant)  SKIP_QUANT="--skip_quant" ;;
    --eval_action) EVAL_ACTION="--eval_action" ;;
    --fast)        NSAMPLES=32; METRIC="magnitude" ;;
    --device=*)    DEVICE="${arg#*=}" ;;
  esac
done

echo "============================================================"
echo " OpenVLA 1-bit Quantization Pipeline"
echo "  device      : $DEVICE"
echo "  nsamples    : $NSAMPLES"
echo "  metric      : $METRIC"
echo "  skip_quant  : ${SKIP_QUANT:-no}"
echo "  eval_action : ${EVAL_ACTION:-no}"
echo "============================================================"

# ── Step 1: Quantize OpenVLA LLM backbone and evaluate PPL ──
python quant_openvla.py \
  --device       "$DEVICE"      \
  --nsamples     "$NSAMPLES"    \
  --salient_metric "$METRIC"    \
  --blocksize    128            \
  --percdamp     0.01           \
  --save         output/openvla_1bit_llm \
  $SKIP_QUANT                  \
  $EVAL_ACTION                 \
  --n_action_samples 20

echo ""
echo "────────────────────────────────────────────────────────────"
echo " Step 1 complete.  Saved to output/openvla_1bit_llm/"
echo " Perplexity results in output/results.json"
echo "────────────────────────────────────────────────────────────"

# ── Step 2 (optional): Detailed action accuracy evaluation ───
if [[ -n "$EVAL_ACTION" && -z "$SKIP_QUANT" ]]; then
  echo ""
  echo "Running standalone action accuracy evaluation..."
  python eval_action_acc.py \
    --quant_llm  output/openvla_1bit_llm  \
    --device     "$DEVICE"                \
    --n_samples  30                       \
    --output     output/action_accuracy.json

  echo ""
  echo "────────────────────────────────────────────────────────────"
  echo " Action accuracy results in output/action_accuracy.json"
  echo "────────────────────────────────────────────────────────────"
fi

echo ""
echo "All results:"
ls -lh output/*.json 2>/dev/null || true
