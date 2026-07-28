#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EDITOR="${EDITOR:-ultraedit}"
LIMIT="${LIMIT:-500}"
MIN_EDITS="${MIN_EDITS:-500}"
CHECKPOINTS="${CHECKPOINTS:-100,250,500}"
QUEUE_STAMP="${QUEUE_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
CONFIG="${CONFIG:-rebuttal/probe_quality/configs/${EDITOR}_counterfact_qwen25_probe_quality_500.json}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-rebuttal/probe_quality/runs/local/${EDITOR}-${QUEUE_STAMP}}"

for design in relevant weak; do
  PYTHON_BIN="${PYTHON_BIN}" \
  CONFIG="${CONFIG}" \
  MODE="saver" \
  LIMIT="${LIMIT}" \
  MIN_EDITS="${MIN_EDITS}" \
  CHECKPOINTS="${CHECKPOINTS}" \
  RUN_STAMP="${QUEUE_STAMP}-${design}" \
  OUTPUT_DIR="${OUTPUT_DIR_BASE}/${design}" \
  PROBE_DESIGN="${design}" \
  LOCALITY_MONITOR_FRACTION="0.5" \
  WEAK_BOTTOM_QUANTILE="0.25" \
  MIN_LOCALITY_PROMPTS="2" \
  bash rebuttal/scripts/run_stream.sh "$@"
done
