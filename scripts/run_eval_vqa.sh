#!/bin/bash

# VQA Evaluation Script
# Usage: bash run_eval_vqa.sh [--skip-inference]

set -e

# ============ Configuration ============

# Model settings
MODEL_PATH=""
MODEL_BASE=""  # Leave empty for full fine-tune, set path for LoRA mode
CONV_MODE="llava_v1"

# Data settings
DATA_FILE=""
IMAGE_FOLDER=""
OUTPUT_DIR=""

# Generation settings
TEMPERATURE=0.0
TOP_P=0.9
NUM_BEAMS=1
MAX_NEW_TOKENS=2048

# Custom connector settings (for models with feature padding)
# Set ACTUAL_MM_HIDDEN_SIZE to your feature dimension (e.g., 512) if using custom connector
# Leave empty for original WSI-LLaVA models
ACTUAL_MM_HIDDEN_SIZE=""  # e.g., "512" for 512-dim features
ENABLE_FEATURE_PADDING=""

# Environment settings
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

# ============ Parse Arguments ============
SKIP_INFERENCE=""
if [[ "$1" == "--skip-inference" ]]; then
    SKIP_INFERENCE="--skip-inference"
    echo "Skip inference mode enabled"
fi

# ============ Print Configuration ============
echo "============================================"
echo "VQA Evaluation Configuration"
echo "============================================"
echo "Model Path:      $MODEL_PATH"
echo "Data File:       $DATA_FILE"
echo "Image Folder:    $IMAGE_FOLDER"
echo "Output Dir:      $OUTPUT_DIR"
echo "Temperature:     $TEMPERATURE"
echo "Top-p:           $TOP_P"
echo "Num Beams:       $NUM_BEAMS"
echo "Max New Tokens:  $MAX_NEW_TOKENS"
echo "CUDA Devices:    $CUDA_VISIBLE_DEVICES"
if [ -n "$ACTUAL_MM_HIDDEN_SIZE" ]; then
    echo "Actual MM Size:  $ACTUAL_MM_HIDDEN_SIZE (custom connector mode)"
fi
echo "============================================"
echo ""

# ============ Validate Inputs ============
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Error: Image folder not found: $IMAGE_FOLDER"
    exit 1
fi

if [ -z "$SKIP_INFERENCE" ] && [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# ============ Run Evaluation ============
mkdir -p "$OUTPUT_DIR"

# Build command with optional parameters
CMD="python eval_vqa.py \
    --model-path \"$MODEL_PATH\" \
    --conv-mode \"$CONV_MODE\" \
    --data-file \"$DATA_FILE\" \
    --image-folder \"$IMAGE_FOLDER\" \
    --output-dir \"$OUTPUT_DIR\" \
    --temperature \"$TEMPERATURE\" \
    --top-p \"$TOP_P\" \
    --num-beams \"$NUM_BEAMS\" \
    --max-new-tokens \"$MAX_NEW_TOKENS\""

# Only add --model-base if it's non-empty (LoRA mode)
if [ -n "$MODEL_BASE" ]; then
    CMD="$CMD --model-base \"$MODEL_BASE\""
fi

# Add optional custom connector parameters
if [ -n "$ACTUAL_MM_HIDDEN_SIZE" ]; then
    CMD="$CMD --actual-mm-hidden-size $ACTUAL_MM_HIDDEN_SIZE"
fi

if [ "$ENABLE_FEATURE_PADDING" = "true" ]; then
    CMD="$CMD --enable-feature-padding"
fi

# Add skip inference flag if set
if [ -n "$SKIP_INFERENCE" ]; then
    CMD="$CMD $SKIP_INFERENCE"
fi

# Execute command
eval $CMD

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
