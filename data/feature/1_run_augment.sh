#!/bin/bash

INPUT_PATH=""
OUTPUT_PATH=""

MODEL_PATH=""
BATCH_SIZE=32
PATCH_SIZE=512

CUDA_DEVICE=0 

SAVE_VIS=false
VIS_DIR="${OUTPUT_PATH}/thumbnails"  

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUGMENT_SCRIPT="${SCRIPT_DIR}/augment_feature.py"



if [ "$#" -ge 2 ]; then
    OUTPUT_PATH="$2"
    VIS_DIR="${OUTPUT_PATH}/thumbnails"
fi

if [ "$#" -ge 3 ]; then
    CUDA_DEVICE="$3"
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}
fi

if [ "$#" -ge 4 ]; then
    SAVE_VIS="$4"
fi

if [ ! -e "${INPUT_PATH}" ]; then
    echo "Error: Input path does not exist: ${INPUT_PATH}"
    exit 1
fi

if [ ! -f "${AUGMENT_SCRIPT}" ]; then
    echo "Error: Cannot find augment_feature.py: ${AUGMENT_SCRIPT}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "Warning: Model file does not exist: ${MODEL_PATH}"
    echo "Please confirm the model path is correct"
fi

mkdir -p "${OUTPUT_PATH}"

if [ "${SAVE_VIS}" == "true" ]; then
    mkdir -p "${VIS_DIR}"
fi

echo "========================================"
echo "WSI Data Augmentation Feature Extraction"
echo "========================================"
echo "Input path:   ${INPUT_PATH}"
echo "Output path:  ${OUTPUT_PATH}"
echo "Model path:   ${MODEL_PATH}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Patch:        ${PATCH_SIZE}"
echo "CUDA:         ${CUDA_DEVICE}"
echo "Save thumbnails: ${SAVE_VIS}"
if [ "${SAVE_VIS}" == "true" ]; then
    echo "Thumbnail dir:   ${VIS_DIR}"
fi
echo "========================================"
echo ""

CMD="python ${AUGMENT_SCRIPT} \
    --input ${INPUT_PATH} \
    --output ${OUTPUT_PATH} \
    --model ${MODEL_PATH} \
    --batch-size ${BATCH_SIZE} \
    --patch-size ${PATCH_SIZE}"

if [ "${SAVE_VIS}" == "true" ]; then
    CMD="${CMD} --save-vis --vis-dir ${VIS_DIR}"
fi

eval ${CMD}

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Processing completed!"
    echo "Output directory: ${OUTPUT_PATH}"
    
    if [ -d "${OUTPUT_PATH}" ]; then
        NUM_FILES=$(ls -1 "${OUTPUT_PATH}"/*.pt 2>/dev/null | wc -l)
        echo "Generated files: ${NUM_FILES}"
        
        DISK_USAGE=$(du -sh "${OUTPUT_PATH}" 2>/dev/null | cut -f1)
        echo "Disk usage: ${DISK_USAGE}"
    fi
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Processing failed, please check error messages"
    echo "========================================"
    exit 1
fi

