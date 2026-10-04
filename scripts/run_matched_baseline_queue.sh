#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/main/alphaedit_counterfact_qwen25_500.json}"
LIMIT="${LIMIT:-500}"
MIN_EDITS="${MIN_EDITS:-500}"
CHECKPOINTS="${CHECKPOINTS:-100,250,500}"
QUEUE_STAMP="${QUEUE_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/local/matched-baselines-${QUEUE_STAMP}}"

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
  bash scripts/run_stream.sh "$@"
}

run_case saver saver

SAVER_JSON="${OUTPUT_DIR}/json/${QUEUE_STAMP}-saver-saver.json"
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
