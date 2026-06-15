#!/bin/bash

INPUT_DIR=""
OUTPUT_DIR=""
MODEL_PATH=""
BATCH_SIZE=64
PATCH_SIZE=512


python pre_feature.py \
    --input ${INPUT_DIR} \
    --output ${OUTPUT_DIR} \
    --model ${MODEL_PATH} \
    --batch-size ${BATCH_SIZE} \
    --patch-size ${PATCH_SIZE}

echo "Feature extraction completed!"

