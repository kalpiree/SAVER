#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
LIMIT="${LIMIT:-2000}"
MIN_EDITS="${MIN_EDITS:-1500}"
CHECKPOINTS="${CHECKPOINTS:-100,500,1000,1500,2000}"
QUEUE_STAMP="${QUEUE_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/runs/local/qwen25-ultraedit-${QUEUE_STAMP}}"
CONFIG="${CONFIG:-rebuttal/configs/ultraedit_counterfact_qwen25_2000.json}"

run_case() {
  local label="$1"
  local mode="$2"
  shift 2
  PYTHON_BIN="${PYTHON_BIN}" \
  CONFIG="${CONFIG}" \
  MODE="${mode}" \
  LIMIT="${LIMIT}" \
  MIN_EDITS="${MIN_EDITS}" \
  CHECKPOINTS="${CHECKPOINTS}" \
  RUN_STAMP="${QUEUE_STAMP}-${label}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash rebuttal/scripts/run_stream.sh "$@"
}

run_case saver-full saver --probe-fraction 1.0

SAVER_JSON="${OUTPUT_DIR}/json/${QUEUE_STAMP}-saver-full-saver.json"
TARGET_ACCEPTANCE="$("${PYTHON_BIN}" - "${SAVER_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["run_summary"]["acceptance_rate"])
PY
)"

run_case random-matched random_reject --target-acceptance "${TARGET_ACCEPTANCE}"
run_case probe-gate-matched probe_gate --target-acceptance "${TARGET_ACCEPTANCE}" --gate-threshold 1000000000
run_case kl-gate-matched kl_gate --target-acceptance "${TARGET_ACCEPTANCE}" --gate-threshold 1000000000
run_case saver-probe025 saver --probe-fraction 0.25
run_case saver-probe050 saver --probe-fraction 0.50
run_case saver-probe075 saver --probe-fraction 0.75
