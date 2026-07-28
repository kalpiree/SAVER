#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-rebuttal/configs/ultraedit_counterfact_qwen25_2000.json}"
MODE="${MODE:-saver}"
LIMIT="${LIMIT:-2000}"
MIN_EDITS="${MIN_EDITS:-1500}"
CHECKPOINTS="${CHECKPOINTS:-100,500,1000,1500,2000}"
PROBE_FRACTION="${PROBE_FRACTION:-1.0}"
PROBE_DESIGN="${PROBE_DESIGN:-standard}"
LOCALITY_MONITOR_FRACTION="${LOCALITY_MONITOR_FRACTION:-0.5}"
WEAK_BOTTOM_QUANTILE="${WEAK_BOTTOM_QUANTILE:-0.25}"
MIN_LOCALITY_PROMPTS="${MIN_LOCALITY_PROMPTS:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/runs/local}"
PREPARE_STREAM="${PREPARE_STREAM:-0}"
STREAM_LIMIT="${STREAM_LIMIT:-${LIMIT}}"
STREAM_SEED="${STREAM_SEED:-17}"

mkdir -p "${OUTPUT_DIR}/json" "${OUTPUT_DIR}/events"

if [[ "${PREPARE_STREAM}" == "1" ]]; then
  STREAM_PATH="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["dataset_path"])
PY
)"
  "${PYTHON_BIN}" scripts/prepare_hf_counterfact_stream.py \
    --output "${STREAM_PATH}" \
    --limit "${STREAM_LIMIT}" \
    --seed "${STREAM_SEED}"
fi

"${PYTHON_BIN}" scripts/run_rebuttal_stream.py \
  --config "${CONFIG}" \
  --mode "${MODE}" \
  --limit "${LIMIT}" \
  --min-edits "${MIN_EDITS}" \
  --checkpoints "${CHECKPOINTS}" \
  --output "${OUTPUT_DIR}/json/${RUN_STAMP}-${MODE}.json" \
  --events-output "${OUTPUT_DIR}/events/${RUN_STAMP}-${MODE}.jsonl" \
  --probe-fraction "${PROBE_FRACTION}" \
  --probe-design "${PROBE_DESIGN}" \
  --locality-monitor-fraction "${LOCALITY_MONITOR_FRACTION}" \
  --weak-bottom-quantile "${WEAK_BOTTOM_QUANTILE}" \
  --min-locality-prompts "${MIN_LOCALITY_PROMPTS}" \
  "$@"
