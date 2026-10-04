#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/ablations/alphaedit_counterfact_qwen25_probe_quality_500.json}"
LIMIT="${LIMIT:-500}"
MIN_EDITS="${MIN_EDITS:-500}"
CHECKPOINTS="${CHECKPOINTS:-100,250,500}"
QUEUE_STAMP="${QUEUE_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/probe_quality/local/probe-coverage-${QUEUE_STAMP}}"
MIN_LOCALITY_PROMPTS="${MIN_LOCALITY_PROMPTS:-8}"

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
  bash scripts/run_stream.sh "$@"
done
