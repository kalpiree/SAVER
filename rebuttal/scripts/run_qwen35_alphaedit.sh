#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-saver}"
LIMIT="${LIMIT:-2000}"
MIN_EDITS="${MIN_EDITS:-1500}"
CHECKPOINTS="${CHECKPOINTS:-100,500,1000,1500,2000}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/runs/local/qwen35-alphaedit-${RUN_STAMP}}"
CONFIG="${CONFIG:-rebuttal/configs/alphaedit_counterfact_qwen35_2000.json}"

PYTHON_BIN="${PYTHON_BIN}" \
CONFIG="${CONFIG}" \
MODE="${MODE}" \
LIMIT="${LIMIT}" \
MIN_EDITS="${MIN_EDITS}" \
CHECKPOINTS="${CHECKPOINTS}" \
RUN_STAMP="${RUN_STAMP}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash rebuttal/scripts/run_stream.sh "$@"
