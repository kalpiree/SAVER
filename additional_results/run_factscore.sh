#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a completed CounterFact run with biographies}"
FACTSCORE_KEY_FILE="${FACTSCORE_KEY_FILE:?Set FACTSCORE_KEY_FILE to the scorer API key file}"
FACTSCORE_DATA_DIR="${FACTSCORE_DATA_DIR:-${HOME}/.cache/factscore}"

"${PYTHON_BIN}" additional_results/score_factscore.py \
  --input "${RUN_DIR}/biographies.jsonl" --input-metadata "${RUN_DIR}/run.json" \
  --output "${FACTSCORE_OUTPUT:-${RUN_DIR}/factscore.json}" \
  --api-key-file "${FACTSCORE_KEY_FILE}" \
  --data-dir "${FACTSCORE_DATA_DIR}" \
  --model-dir "${FACTSCORE_MODEL_DIR:-${FACTSCORE_DATA_DIR}}" \
  --cache-dir "${FACTSCORE_CACHE_DIR:-${FACTSCORE_DATA_DIR}/cache}" \
  --model-name "${FACTSCORE_MODEL:-retrieval+ChatGPT}" \
  --knowledge-source "${FACTSCORE_KNOWLEDGE_SOURCE:-enwiki-20230401}" \
  --gamma "${FACTSCORE_GAMMA:-10}" "$@"
