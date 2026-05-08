#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${DATASET:-${SCRIPT_DIR}/data/DHRD.csv}"
TRAIN_DIR="${TRAIN_DIR:-sasrec_fixedtimemask_cross}"

python -m Rhythm.main \
    --dataset "$DATASET" \
    --train_dir "$TRAIN_DIR" \
    --batch_size 512 \
    --lr 1e-3 \
    --maxlen 50 \
    --hidden_units 128 \
    --num_blocks 1 \
    --num_epochs 300 \
    --stop 10 \
    --num_heads 4 \
    --attn_dropout_rate 0.5 \
    --ff_dropout_rate 0.5 \
    --l2_emb 0.0 \
    --device auto \
    --eval_batch_size 256 \
    --num_workers 1 \
    --use_cross True \
    --learnable_intent False \
    --loss_type full_ce \
    --n_negatives 256 \
    --time_type hour \
    --use_fixed True \
    --kmax 4 \
    --min_delta 0.1
