#!/usr/bin/env bash
set -euo pipefail

QUEUE_STAMP="${QUEUE_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/runs/local/qwen35-alphaedit-5000-comparison-${QUEUE_STAMP}}"

for mode in unconstrained saver; do
  MODE="${mode}" \
  RUN_STAMP="${QUEUE_STAMP}-${mode}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash rebuttal/scripts/run_qwen35_alphaedit_5000.sh "$@"
done
