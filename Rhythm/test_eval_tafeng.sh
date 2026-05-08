#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/eval/ta-feng}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m Rhythm.main \
    --dataset "${SCRIPT_DIR}/data/ta-feng.csv" \
    --train_dir "eval_ta-feng" \
    --checkpoint "${SCRIPT_DIR}/model/ta-feng/best_model.pt" \
    --eval_only True \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --batch_size 256 \
    --lr 0.001 \
    --maxlen 50 \
    --hidden_units 256 \
    --num_blocks 2 \
    --num_heads 4 \
    --attn_dropout_rate 0.5 \
    --ff_dropout_rate 0.5 \
    --l2_emb 0.0 \
    --learnable_intent True \
    --loss_type full_ce \
    --n_negatives 256 \
    --time_type day_of_week \
    --use_cross False \
    --use_cross_withfnn True \
    --use_fixed False \
    --kmax 3 \
    --min_delta 0.2
