#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
DATASET_NAME="${DATASET_NAME:-all}"
MAX_CASES="${MAX_CASES:-5}"
RANK_THRESHOLD="${RANK_THRESHOLD:-10}"
MAX_SCAN_USERS="${MAX_SCAN_USERS:-0}"
USER_SELECTION="${USER_SELECTION:-coverage}"
RANDOM_POOL_SIZE="${RANDOM_POOL_SIZE:-1000}"
MIN_SEGMENTS="${MIN_SEGMENTS:-1}"
SEED="${SEED:-2026}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/case_study_outputs}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m Rhythm.case_study \
    --dataset-name "${DATASET_NAME}" \
    --device "${DEVICE}" \
    --max-cases "${MAX_CASES}" \
    --rank-threshold "${RANK_THRESHOLD}" \
    --max-scan-users "${MAX_SCAN_USERS}" \
    --user-selection "${USER_SELECTION}" \
    --random-pool-size "${RANDOM_POOL_SIZE}" \
    --min-segments "${MIN_SEGMENTS}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}"
