#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-rebuttal/configs/alphaedit_counterfact_qwen35_5000.json}" \
LIMIT="${LIMIT:-5000}" \
MIN_EDITS="${MIN_EDITS:-5000}" \
CHECKPOINTS="${CHECKPOINTS:-500,1000,2500,5000}" \
bash rebuttal/scripts/run_qwen35_alphaedit.sh "$@"
