#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TARGET_DIR="${TARGET_DIR:-rebuttal/vendor/transformers_src}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3.5-9B}"

mkdir -p "${TARGET_DIR}"

"${PYTHON_BIN}" -m pip install \
  --upgrade \
  --no-deps \
  --target "${TARGET_DIR}" \
  'transformers@git+https://github.com/huggingface/transformers.git'

"${PYTHON_BIN}" -m pip install \
  --upgrade \
  --pre \
  --target "${TARGET_DIR}" \
  'huggingface-hub>=1.5.0,<2.0' \
  'tokenizers==0.23.0rc0' \
  'safetensors>=0.4.3' \
  'packaging>=24.0' \
  filelock \
  numpy \
  pyyaml \
  regex \
  requests \
  tqdm

PYTHONPATH="${TARGET_DIR}${PYTHONPATH:+:${PYTHONPATH}}" MODEL_DIR="${MODEL_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from transformers import AutoConfig

config = AutoConfig.from_pretrained(os.environ["MODEL_DIR"], trust_remote_code=True)
print(type(config).__name__)
print(config.model_type)
print(config.architectures)
PY
