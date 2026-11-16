#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CONFIG="${CONFIG:-additional_results/configs/qwen35_alphaedit.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/additional_results/mquake-$(date +%Y%m%d-%H%M%S)}"
DATA_DIR="${DATA_DIR:-${OUTPUT_DIR}/data}"
LOCALITY_BANK="${LOCALITY_BANK:?Set LOCALITY_BANK to a locality JSONL file}"
read -r -a seeds <<< "${SEEDS:-$("${PYTHON_BIN}" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["seeds"])' "${CONFIG}")}"
read -r -a modes <<< "${MODES:-unconstrained saver}"

dataset_args=(--download --revision "${MQUAKE_REVISION:-main}")
if [[ -n "${MQUAKE_SOURCE:-}" ]]; then
  dataset_args=(--input "${MQUAKE_SOURCE}")
fi
"${PYTHON_BIN}" additional_results/prepare_datasets.py mquake \
  "${dataset_args[@]}" --output "${DATA_DIR}/mquake.jsonl"

run_args=()
if [[ -n "${LIMIT:-}" ]]; then
  run_args+=(--limit "${LIMIT}")
fi
if [[ -n "${CASE_MODE:-}" ]]; then
  run_args+=(--case-mode "${CASE_MODE}")
fi
for seed in "${seeds[@]}"; do
  for mode in "${modes[@]}"; do
    "${PYTHON_BIN}" additional_results/run_mquake.py \
      --config "${CONFIG}" --dataset "${DATA_DIR}/mquake.jsonl" \
      --locality-bank "${LOCALITY_BANK}" --locality-size "${LOCALITY_SIZE:-16}" \
      --output-dir "${OUTPUT_DIR}/${mode}-${seed}" --mode "${mode}" --seed "${seed}" \
      ${run_args[@]+"${run_args[@]}"}
  done
done
