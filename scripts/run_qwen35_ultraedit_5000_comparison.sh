#!/usr/bin/env bash
set -euo pipefail

QUEUE_STAMP="${QUEUE_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/local/qwen35-ultraedit-5000-comparison-${QUEUE_STAMP}}"

for mode in unconstrained saver; do
  MODE="${mode}" \
  RUN_STAMP="${QUEUE_STAMP}-${mode}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash scripts/run_qwen35_ultraedit_5000.sh "$@"
done
