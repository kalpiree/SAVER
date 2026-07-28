#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-saver}"
LIMIT="${LIMIT:-1000}"
MIN_EDITS="${MIN_EDITS:-500}"
CHECKPOINTS="${CHECKPOINTS:-100,500,1000}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/runs/local/qwen35-ultraedit-${RUN_STAMP}}"
CONFIG="${CONFIG:-rebuttal/configs/ultraedit_counterfact_qwen35_1000.json}"

PYTHON_BIN="${PYTHON_BIN}" \
CONFIG="${CONFIG}" \
MODE="${MODE}" \
LIMIT="${LIMIT}" \
MIN_EDITS="${MIN_EDITS}" \
CHECKPOINTS="${CHECKPOINTS}" \
RUN_STAMP="${RUN_STAMP}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash rebuttal/scripts/run_stream.sh "$@"
