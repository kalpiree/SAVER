#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-rebuttal/configs/ultraedit_counterfact_qwen35_1000.json}"
LIMIT="${LIMIT:-1000}"
MIN_EDITS="${MIN_EDITS:-500}"
CHECKPOINTS="${CHECKPOINTS:-100,500,1000}"
QUEUE_STAMP="${QUEUE_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/probe_quality/runs/local/probe-coverage-${QUEUE_STAMP}}"
MIN_LOCALITY_PROMPTS="${MIN_LOCALITY_PROMPTS:-2}"

for fraction in 0.25 0.50 0.75 1.00; do
  label="${fraction/./}"
  PYTHON_BIN="${PYTHON_BIN}" \
  CONFIG="${CONFIG}" \
  MODE="saver" \
  LIMIT="${LIMIT}" \
  MIN_EDITS="${MIN_EDITS}" \
  CHECKPOINTS="${CHECKPOINTS}" \
  RUN_STAMP="${QUEUE_STAMP}-coverage-${label}" \
  OUTPUT_DIR="${OUTPUT_DIR}/${label}" \
  PROBE_DESIGN="relevant" \
  PROBE_FRACTION="${fraction}" \
  LOCALITY_MONITOR_FRACTION="0.5" \
  MIN_LOCALITY_PROMPTS="${MIN_LOCALITY_PROMPTS}" \
  bash rebuttal/scripts/run_stream.sh "$@"
done
