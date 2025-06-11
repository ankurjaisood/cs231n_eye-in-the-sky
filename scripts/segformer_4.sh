#!/usr/bin/env bash
set -e

# Determine this script’s directory (scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PROJECT_ROOT is one level up from scripts/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create logs directory at PROJECT_ROOT/logs
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TRAIN_SCRIPT="$PROJECT_ROOT/utils/finetune_segformer.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at: $TRAIN_SCRIPT"
    exit 1
fi

# Output folder for model checkpoints
CHECKPOINT_BASE="$PROJECT_ROOT/models/segformer_checkpoints"
mkdir -p "$CHECKPOINT_BASE"

# Hyperparameter grid
#LEARNING_RATES=(5e-5 1e-5 5e-4)
#BATCH_SIZES=(4 8 16)
#EPOCHS=(3 5 10)

MODEL_NAME=("nvidia/segformer-b4-finetuned-ade-512-512")
LEARNING_RATES=(5e-5)
BATCH_SIZES=(16)
EPOCHS=(20)
IGNORE_BACKGROUND=(0)

for model in "${MODEL_NAME[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for ep in "${EPOCHS[@]}"; do
                for ignore_bg in "${IGNORE_BACKGROUND[@]}"; do

                    base_model="${model##*/}"
                    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
                    RUN_NAME="model-${base_model}_num-classes-4_ignbg-${ignore_bg}_lr${lr}_bs${bs}_ep${ep}_${TIMESTAMP}"
                    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
                    OUTPUT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"
                    mkdir -p "$OUTPUT_DIR"

                    echo "================================================================="
                    echo "Starting run: $RUN_NAME"
                    echo "  Model:         $model"
                    echo "  Learning rate: $lr"
                    echo "  Batch size:    $bs"
                    echo "  Epochs:        $ep"
                    echo "  Ignore background: $ignore_bg"
                    echo "  Logging to:    $LOG_FILE"
                    echo "================================================================="

                    if [ "$ignore_bg" -eq 0 ]; then
                        python "$TRAIN_SCRIPT" \
                            --num_labels "4" \
                            --train_masks "/home/anksood/cs231n/cs231n_eye-in-the-sky/datasets/kaggle-image-segmentation/train/masks_4/" \
                            --valid_masks "/home/anksood/cs231n/cs231n_eye-in-the-sky/datasets/kaggle-image-segmentation/valid/masks_4/" \
                            --test_masks "/home/anksood/cs231n/cs231n_eye-in-the-sky/datasets/kaggle-image-segmentation/test/masks_4/" \
                            --pretrained "$model" \
                            --lr "$lr" \
                            --batch_size "$bs" \
                            --epochs "$ep" \
                            --output_dir "$OUTPUT_DIR" \
                            > "$LOG_FILE" 2>&1
                    else
                        python "$TRAIN_SCRIPT" \
                            --num_labels "4" \
                            --train_masks "/home/anksood/cs231n/cs231n_eye-in-the-sky/datasets/kaggle-image-segmentation/train/masks_4/" \
                            --valid_masks "/home/anksood/cs231n/cs231n_eye-in-the-sky/datasets/kaggle-image-segmentation/valid/masks_4/" \
                            --test_masks "/home/anksood/cs231n/cs231n_eye-in-the-sky/datasets/kaggle-image-segmentation/test/masks_4/" \
                            --pretrained "$model" \
                            --lr "$lr" \
                            --batch_size "$bs" \
                            --epochs "$ep" \
                            --output_dir "$OUTPUT_DIR" \
                            --ign_background \
                            > "$LOG_FILE" 2>&1
                    fi

                    echo "Finished run: $RUN_NAME"
                    echo "Checkpoint dir: $OUTPUT_DIR"
                    echo "Log saved at:    $LOG_FILE"
                    echo
                done
            done
        done
    done
done

echo "All experiments completed. Logs are in: $LOG_DIR"
