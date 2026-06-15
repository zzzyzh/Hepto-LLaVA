#!/bin/bash

# ================================================================
# WSI-LLaVA Finetune Script with Custom Connector
# ================================================================
# Three training modes:
#
# Mode A: Use Pretrained Projector (USE_PRETRAINED_PROJECTOR=true)
#   Use WSI-LLaVA-7b as model_name_or_path, keep original projector and weights,
#   only fine-tune LLaVA language model with LoRA.
#   Example config:
#     MODEL_NAME_OR_PATH="/path/to/WSI-LLaVA-7b"
#     USE_PRETRAINED_PROJECTOR=true
#     TRAIN_CONNECTOR=false    # Freeze projector
#     LORA_ENABLE=true         # LoRA fine-tune LLM
#
# Mode B: Custom Connector - Two-stage Training
#
#   Stage 1: Train Connector Only (USE_PRETRAINED_PROJECTOR=false)
#     Random init custom connector, freeze LLM, train connector only.
#     Output dir will save mm_projector.bin (connector weights only).
#     Example config:
#       USE_PRETRAINED_PROJECTOR=false
#       CONNECTOR_TYPE="mlp"
#       PRETRAIN_MM_MLP_ADAPTER=""           # No pretrained weights, random init
#       TRAIN_CONNECTOR=true
#       TRAIN_MLLM=false
#       LORA_ENABLE=false
#       OUTPUT_DIR="/path/to/stage1_output"
#
#   Stage 2: Joint Fine-tune Connector + LLM (USE_PRETRAINED_PROJECTOR=false)
#     Load stage 1 connector weights, joint LoRA fine-tune LLM.
#     Example config:
#       USE_PRETRAINED_PROJECTOR=false
#       CONNECTOR_TYPE="mlp"                 # Same as stage 1
#       PRETRAIN_MM_MLP_ADAPTER="/path/to/stage1_output/mm_projector.bin"
#       TRAIN_CONNECTOR=true
#       TRAIN_MLLM=true                      # Now train LLM too
#       LORA_ENABLE=true                     # Use LoRA for LLM
#       OUTPUT_DIR="/path/to/stage2_output"
#
#   Note: CONNECTOR_TYPE and parameters (MLP_*/QFORMER_*) must be consistent!
# ================================================================

set -e

# ================= Configuration =================

# GPU Settings
CUDA_DEVICES="0"
MASTER_PORT=29506

# Paths
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Model Paths
MODEL_NAME_OR_PATH=""
VISION_TOWER=""
OUTPUT_DIR=""

# Data Paths
DATA_PATH=""
IMAGE_FOLDER=""

# ================= Connector Configuration =================
# Pretrained Projector Mode
# true:  Use original WSI-LLaVA projector and weights
# false: Use custom connector configuration below
USE_PRETRAINED_PROJECTOR=false

# Pretrained Projector/Connector Weights Path (optional)
# - Pretrained mode (A): Load original projector weights from .bin file
# - Custom Connector Stage 2 (B): Load trained connector from stage 1 mm_projector.bin
# Leave empty for random init
PRETRAIN_MM_MLP_ADAPTER=""

# Connector Type (when USE_PRETRAINED_PROJECTOR=false)
# Options: "mlp", "qformer", or custom module path
CONNECTOR_TYPE="qformer"

# MLP Connector Parameters
MLP_NUM_LAYERS=2
MLP_ACTIVATION="gelu"
MLP_DROPOUT=0.0
MLP_USE_LAYER_NORM="false"

# Q-Former Connector Parameters
QFORMER_HIDDEN_SIZE=768               # QFormer internal dim (BLIP-2 default 768, much smaller than LLM 4096)
QFORMER_NUM_QUERY_TOKENS=64
QFORMER_NUM_LAYERS=2
QFORMER_NUM_HEADS=12                  # 768/12=64 head_dim (standard)
QFORMER_DROPOUT=0.1

# ================= Training Parameters =================
# Training Control
TRAIN_CONNECTOR=true             # Whether to train Connector/Projector
TRAIN_MLLM=true                  # Whether to train MLLM
LORA_ENABLE=true                 # Whether to use LoRA

# LoRA Parameters
LORA_R=128
LORA_ALPHA=256

# Training Hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=32
GRADIENT_ACCUMULATION=4
LEARNING_RATE=2e-4
MM_PROJECTOR_LR=2e-5
WARMUP_RATIO=0.03
MODEL_MAX_LENGTH=2048

# Other Parameters
BF16=true
GRADIENT_CHECKPOINTING=true
DATALOADER_WORKERS=4
SAVE_STEPS=50000

# DeepSpeed Config
DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/zero3.json"

# ===========================================

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Change to project root
cd "${PROJECT_ROOT}"

# Print configuration
echo "========================================"
echo "WSI-LLaVA Finetune with Custom Connector"
echo "========================================"
echo "Project Root:     ${PROJECT_ROOT}"
echo "Model Path:       ${MODEL_NAME_OR_PATH}"
echo "Vision Tower:     ${VISION_TOWER}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "Data Path:        ${DATA_PATH}"
echo "Image Folder:     ${IMAGE_FOLDER}"
echo "----------------------------------------"
echo "Projector Mode:"
echo "  Use Pretrained:   ${USE_PRETRAINED_PROJECTOR}"
if [ "${USE_PRETRAINED_PROJECTOR}" == "true" ]; then
    echo "  Pretrain Path:    ${PRETRAIN_MM_MLP_ADAPTER:-<none>}"
else
    echo "  Connector Type:   ${CONNECTOR_TYPE}"
    echo "  Pretrain Path:    ${PRETRAIN_MM_MLP_ADAPTER:-<none, random init>}"
    # Auto-detect stage
    if [ -n "${PRETRAIN_MM_MLP_ADAPTER}" ]; then
        echo "  [Stage 2] Load trained connector + joint fine-tune"
    else
        echo "  [Stage 1] Train connector from scratch"
    fi
fi
echo "----------------------------------------"
echo "Training Control:"
echo "  Train Connector:  ${TRAIN_CONNECTOR}"
echo "  Train MLLM:       ${TRAIN_MLLM}"
echo "  Use LoRA:         ${LORA_ENABLE}"
echo "----------------------------------------"
echo "LoRA R:           ${LORA_R}"
echo "Batch Size:       ${BATCH_SIZE}"
echo "Learning Rate:    ${LEARNING_RATE}"
echo "Epochs:           ${NUM_EPOCHS}"
echo "GPU Devices:      ${CUDA_DEVICES}"
echo "========================================"
echo ""

# Build Connector args JSON (only needed in non-pretrained mode)
if [ "${USE_PRETRAINED_PROJECTOR}" == "false" ]; then
    if [ "${CONNECTOR_TYPE}" == "mlp" ]; then
        CONNECTOR_ARGS='{"num_layers": '"${MLP_NUM_LAYERS}"', "activation": "'"${MLP_ACTIVATION}"'", "dropout": '"${MLP_DROPOUT}"', "use_layer_norm": '"${MLP_USE_LAYER_NORM}"'}'
    elif [ "${CONNECTOR_TYPE}" == "qformer" ]; then
        CONNECTOR_ARGS='{"qformer_hidden_size": '"${QFORMER_HIDDEN_SIZE}"', "num_query_tokens": '"${QFORMER_NUM_QUERY_TOKENS}"', "num_layers": '"${QFORMER_NUM_LAYERS}"', "num_heads": '"${QFORMER_NUM_HEADS}"', "dropout": '"${QFORMER_DROPOUT}"'}'
    else
        # Custom Connector, use empty or from env variable
        CONNECTOR_ARGS="${CUSTOM_CONNECTOR_ARGS:-{}}"
    fi
    
    echo "Connector Args: ${CONNECTOR_ARGS}"
    echo ""
fi

# Set environment variables
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Build training command
TRAIN_SCRIPT="${PROJECT_ROOT}/train_wsi_llava_v2.py"
CMD="deepspeed --include localhost:${CUDA_DEVICES} --master_port ${MASTER_PORT} ${TRAIN_SCRIPT}"

# Add arguments
CMD="${CMD} --model_name_or_path ${MODEL_NAME_OR_PATH}"
CMD="${CMD} --version v1"
CMD="${CMD} --data_path ${DATA_PATH}"
CMD="${CMD} --image_folder ${IMAGE_FOLDER}"
CMD="${CMD} --vision_tower ${VISION_TOWER}"
CMD="${CMD} --output_dir ${OUTPUT_DIR}"

# Projector/Connector arguments
CMD="${CMD} --use_pretrained_projector ${USE_PRETRAINED_PROJECTOR}"

if [ "${USE_PRETRAINED_PROJECTOR}" == "true" ]; then
    # Pretrained Projector mode
    # When not training connector, use native freeze_mm_mlp_adapter mechanism
    if [ "${TRAIN_CONNECTOR}" == "false" ]; then
        CMD="${CMD} --freeze_mm_mlp_adapter True"
    fi
else
    # Custom Connector mode
    CMD="${CMD} --mm_projector_path ${CONNECTOR_TYPE}"
    # Note: Use single quotes for JSON args to prevent shell from eating quotes
    CMD="${CMD} --mm_projector_args '${CONNECTOR_ARGS}'"
fi

# Load pretrained projector/connector weights (available in both modes)
# Stage 1: leave empty (random init), Stage 2: point to stage 1 mm_projector.bin
if [ -n "${PRETRAIN_MM_MLP_ADAPTER}" ]; then
    CMD="${CMD} --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER}"
fi

# Training control parameters
CMD="${CMD} --train_connector ${TRAIN_CONNECTOR}"
CMD="${CMD} --train_mllm ${TRAIN_MLLM}"

# Stage 1 auto-detect: Train Connector only, no LoRA
# Set tune_mm_mlp_adapter=True to output only mm_projector.bin
# (not full model), convenient for Stage 2 to load via PRETRAIN_MM_MLP_ADAPTER
if [ "${USE_PRETRAINED_PROJECTOR}" == "false" ] && \
   [ "${TRAIN_CONNECTOR}" == "true" ] && \
   [ "${TRAIN_MLLM}" == "false" ] && \
   [ "${LORA_ENABLE}" == "false" ]; then
    CMD="${CMD} --tune_mm_mlp_adapter True"
    echo "[Stage 1] tune_mm_mlp_adapter=True → output will contain mm_projector.bin"
fi

# Legacy parameters for compatibility (will be overridden by new parameters)
CMD="${CMD} --mm_projector_type mlp2x_gelu"
CMD="${CMD} --mm_vision_select_layer -2"
CMD="${CMD} --mm_use_im_start_end False"
CMD="${CMD} --mm_use_im_patch_token False"
CMD="${CMD} --image_aspect_ratio pad"
CMD="${CMD} --mm_patch_merge_type flat"

# LoRA parameters
if [ "${LORA_ENABLE}" == "true" ]; then
    CMD="${CMD} --lora_enable True"
    CMD="${CMD} --lora_r ${LORA_R}"
    CMD="${CMD} --lora_alpha ${LORA_ALPHA}"
    CMD="${CMD} --mm_projector_lr ${MM_PROJECTOR_LR}"
fi

# Training hyperparameters
CMD="${CMD} --num_train_epochs ${NUM_EPOCHS}"
CMD="${CMD} --per_device_train_batch_size ${BATCH_SIZE}"
CMD="${CMD} --per_device_eval_batch_size 1"
CMD="${CMD} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION}"
CMD="${CMD} --learning_rate ${LEARNING_RATE}"
CMD="${CMD} --warmup_ratio ${WARMUP_RATIO}"
CMD="${CMD} --lr_scheduler_type cosine"
CMD="${CMD} --model_max_length ${MODEL_MAX_LENGTH}"

# Other parameters
CMD="${CMD} --deepspeed ${DEEPSPEED_CONFIG}"
CMD="${CMD} --bf16 ${BF16}"
CMD="${CMD} --tf32 True"
CMD="${CMD} --gradient_checkpointing ${GRADIENT_CHECKPOINTING}"
CMD="${CMD} --dataloader_num_workers ${DATALOADER_WORKERS}"
CMD="${CMD} --lazy_preprocess True"
CMD="${CMD} --group_by_modality_length True"

# Save strategy
CMD="${CMD} --evaluation_strategy no"
CMD="${CMD} --save_strategy steps"
CMD="${CMD} --save_steps ${SAVE_STEPS}"
CMD="${CMD} --save_total_limit 1"
CMD="${CMD} --weight_decay 0."
CMD="${CMD} --logging_steps 1"
CMD="${CMD} --report_to wandb"

# Print command
echo "Running command:"
echo "${CMD}"
echo ""

# Execute training
eval ${CMD}

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Training completed successfully!"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Training failed! Check the error messages above."
    echo "========================================"
    exit 1
fi
