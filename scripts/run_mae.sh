#!/bin/bash

FEATURE_DIR=""
OUTPUT_DIR=""

MASK_RATIO=0.75
EPOCHS=60
LR=2e-5
CURRICULUM_TRANSITION=0.05
GRADIENT_ACCUMULATION_STEPS=16
WARMUP_EPOCHS=1
WARMUP_LR=2e-6

EMBED_DIM=512
NUM_HEADS=8
LAYERS=6
PACK_SIZE=9
USE_FIXED_POS_EMBED=false

SEED=42
DEVICE="cuda:0"

mkdir -p ${OUTPUT_DIR}

cd "$(dirname "$0")/.."

FIXED_POS_EMBED_ARG=""
if [ "${USE_FIXED_POS_EMBED}" = "true" ]; then
    FIXED_POS_EMBED_ARG="--use-fixed-pos-embed"
    echo "Position Embedding: Fixed 2D sinusoidal"
else
    echo "Position Embedding: Learnable"
fi

echo "========================================"
echo "HSAN-MAE Self-Supervised Pretraining"
echo "========================================"
echo "Feature Dir:  ${FEATURE_DIR}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo "Epochs:       ${EPOCHS}"
echo "LR:           ${LR}"
echo "Mask Ratio:   ${MASK_RATIO}"
echo "Device:       ${DEVICE}"
TRANSITION_EPOCH=$(awk "BEGIN {printf \"%.0f\", ${EPOCHS} * ${CURRICULUM_TRANSITION}}")
echo "Phase 1.1 (Patch-level): Epoch 1 ~ ${TRANSITION_EPOCH}"
echo "Phase 1.2 (Pack-level):  Epoch $((TRANSITION_EPOCH + 1)) ~ ${EPOCHS}"
echo "========================================"
echo ""

python pretrain_mae.py \
    --feature-dir ${FEATURE_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --mask-ratio ${MASK_RATIO} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --curriculum-transition "${CURRICULUM_TRANSITION}" \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
    --warmup-epochs ${WARMUP_EPOCHS} \
    --warmup-lr ${WARMUP_LR} \
    --embed-dim ${EMBED_DIM} \
    --num-heads ${NUM_HEADS} \
    --layers ${LAYERS} \
    --pack-size ${PACK_SIZE} \
    --seed ${SEED} \
    --device ${DEVICE} \
    ${FIXED_POS_EMBED_ARG}

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Pretraining completed!"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Pretraining failed! Check the error messages above."
    echo "========================================"
    exit 1
fi

echo "Done!"

