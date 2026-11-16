#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CONFIG="${CONFIG:-additional_results/configs/qwen35_alphaedit.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/additional_results/counterfact-$(date +%Y%m%d-%H%M%S)}"
DATA_DIR="${DATA_DIR:-${OUTPUT_DIR}/data}"
LIMIT="${LIMIT:-$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["limit"])' "${CONFIG}")}"
AUDIT_SIZE="${AUDIT_SIZE:-$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["audit_size"])' "${CONFIG}")}"
AUDIT_SEED="${AUDIT_SEED:-$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["audit_seed"])' "${CONFIG}")}"
read -r -a seeds <<< "${SEEDS:-$("${PYTHON_BIN}" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["seeds"])' "${CONFIG}")}"
read -r -a modes <<< "${MODES:-unconstrained saver}"
SOURCE="${COUNTERFACT_SOURCE:-${DATA_DIR}/source.jsonl}"

if [[ ! -f "${SOURCE}" ]]; then
  "${PYTHON_BIN}" scripts/prepare_hf_counterfact_stream.py \
    --output "${SOURCE}" --limit "${LIMIT}" --seed "${AUDIT_SEED}" --max-locality 16
fi
"${PYTHON_BIN}" additional_results/prepare_counterfact.py \
  --input "${SOURCE}" --output-dir "${DATA_DIR}/probes" \
  --limit "${LIMIT}" --audit-size "${AUDIT_SIZE}" --seed "${AUDIT_SEED}"

biography_args=()
if [[ -n "${BIOGRAPHY_TOPICS:-}" ]]; then
  "${PYTHON_BIN}" additional_results/prepare_datasets.py biographies \
    --input "${BIOGRAPHY_TOPICS}" --output "${DATA_DIR}/biographies.jsonl"
  biography_args=(--biographies "${DATA_DIR}/biographies.jsonl")
fi

runs=()
for seed in "${seeds[@]}"; do
  for mode in "${modes[@]}"; do
    run="${OUTPUT_DIR}/${mode}-${seed}"
    "${PYTHON_BIN}" additional_results/run_counterfact.py \
      --config "${CONFIG}" --stream "${DATA_DIR}/probes/stream.jsonl" \
      --audit-bank "${DATA_DIR}/probes/audit.jsonl" --output-dir "${run}" \
      --mode "${mode}" --seed "${seed}" --limit "${LIMIT}" ${biography_args[@]+"${biography_args[@]}"}
    runs+=("${run}/run.json")
  done
done
"${PYTHON_BIN}" additional_results/analyze.py --runs "${runs[@]}" --output-dir "${OUTPUT_DIR}/tables"
