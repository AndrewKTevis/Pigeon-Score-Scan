#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
OUTPUT_DIR="${1:-${PROJECT_DIR}/training_data/models/deepscores-symbol-detector-e8-b2-20260727}"
export PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Capacity testing on the 8 GiB RTX 4060 established batch 2 as the largest
# reliable physical batch (batch 4 reached ~7.93 GiB before evaluator load).
# Accumulation 2 retains an effective batch of four without an OOM cliff.
exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/train_deepscores_symbol_detector.py" \
  --prepared-dir "${PROJECT_DIR}/training_data/prepared/deepscores_v2_expression_tiles_v3" \
  --images-dir "${PROJECT_DIR}/training_data/external/corpora/deepscores_v2_dense/ds2_dense/images" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 8 \
  --batch-size 2 \
  --accumulate 2 \
  --workers 2 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --eval-every 1 \
  --rare-class-sampling-power 0.35 \
  --rare-class-max-repeat 4 \
  --resume
