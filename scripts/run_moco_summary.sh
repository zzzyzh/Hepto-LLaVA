#!/bin/bash

FEATURE_DIR=""
OUTPUT_DIR=""

PRETRAINED_CHECKPOINT=""

EPOCHS=30
ENCODER_LR=1e-6
PROJECTOR_LR=1e-4
DROP_RATIO=0.0
MAX_PACKS=900

PROJ_DIM=128
QUEUE_SIZE=8192
MOMENTUM=0.999
TEMPERATURE=0.07

POSITIVE_MODE="mix"
NOISE_STD=0.2
MAX_PAIRS_PER_SAMPLE=50
MIN_FILE_GAP=10

SEED=42
DEVICE="cuda:1"

mkdir -p ${OUTPUT_DIR}

cd "$(dirname "$0")/.."

echo "========================================"
echo "Summary Token Level MoCo Training"
echo "========================================"
echo "Feature Dir:  ${FEATURE_DIR}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo "Pretrained:   ${PRETRAINED_CHECKPOINT}"
echo ""
echo "Training Settings:"
echo "  Epochs:     ${EPOCHS}"
echo "  Encoder LR: ${ENCODER_LR}"
echo "  Proj LR:    ${PROJECTOR_LR}"
echo "  Drop Ratio: ${DROP_RATIO}"
echo "  Max Packs:  ${MAX_PACKS}"
echo ""
echo "Summary Token MoCo Settings:"
echo "  Positive Mode:    ${POSITIVE_MODE}"
if [ "${POSITIVE_MODE}" == "noise" ]; then
    echo "  Noise Std:        ${NOISE_STD}"
elif [ "${POSITIVE_MODE}" == "mix" ]; then
    echo "  Noise Std:        ${NOISE_STD} (for noise augmentation)"
    echo "  Strategy:         Randomly choose adjacent or noise per sample"
fi
echo "  Queue Size:       ${QUEUE_SIZE}"
echo "  Max Pairs/Sample: ${MAX_PAIRS_PER_SAMPLE}"
echo "  Min File Gap:     ${MIN_FILE_GAP}"
echo ""
echo "Device:       ${DEVICE}"
echo "========================================"
echo ""

CMD="python train_moco_summary.py \
    --feature-dir ${FEATURE_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} \
    --epochs ${EPOCHS} \
    --encoder-lr ${ENCODER_LR} \
    --projector-lr ${PROJECTOR_LR} \
    --drop-ratio ${DROP_RATIO} \
    --max-packs ${MAX_PACKS} \
    --proj-dim ${PROJ_DIM} \
    --queue-size ${QUEUE_SIZE} \
    --momentum ${MOMENTUM} \
    --temperature ${TEMPERATURE} \
    --positive-mode ${POSITIVE_MODE} \
    --noise-std ${NOISE_STD} \
    --max-pairs-per-sample ${MAX_PAIRS_PER_SAMPLE} \
    --min-file-gap ${MIN_FILE_GAP} \
    --seed ${SEED} \
    --device ${DEVICE}"

echo "Starting Summary Token Level MoCo training..."
eval ${CMD}

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Training completed!"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Training failed! Check the error messages above."
    echo "========================================"
    exit 1
fi

echo "Done!"

