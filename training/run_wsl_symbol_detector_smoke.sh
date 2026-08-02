#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
OUTPUT_DIR="${1:-${PROJECT_DIR}/training_data/models/deepscores-symbol-detector-smoke-v3}"

exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/train_deepscores_symbol_detector.py" \
  --prepared-dir "${PROJECT_DIR}/training_data/prepared/deepscores_v2_expression_tiles_v3" \
  --images-dir "${PROJECT_DIR}/training_data/external/corpora/deepscores_v2_dense/ds2_dense/images" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 1 \
  --batch-size 1 \
  --accumulate 4 \
  --workers 2 \
  --max-train-tiles 32 \
  --max-test-tiles 16
