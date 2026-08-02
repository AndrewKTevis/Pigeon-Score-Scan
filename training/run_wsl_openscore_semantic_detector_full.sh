#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
PREPARED_DIR="${PROJECT_DIR}/training_data/prepared/openscore_string_quartets_svg_regions_normalized_v2"
OUTPUT_DIR="${1:-${PROJECT_DIR}/training_data/models/openscore-semantic-detector-e6-b2-20260728}"
INITIAL_BACKBONE_MODEL="${2:-${PROJECT_DIR}/training_data/models/deepscores-symbol-detector-e8-b2-20260727/model.best.pt}"
LOADER_WORKERS="${SCORESCAN_LOADER_WORKERS:-4}"
export PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -s "${PREPARED_DIR}/manifest.json" ||
      ! -s "${PREPARED_DIR}/train.jsonl" ||
      ! -s "${PREPARED_DIR}/test.jsonl" ]]; then
  echo "OpenScore semantic dataset is incomplete: ${PREPARED_DIR}" >&2
  exit 1
fi
if [[ ! -s "${INITIAL_BACKBONE_MODEL}" ]]; then
  echo "DeepScores detector initialization is incomplete: ${INITIAL_BACKBONE_MODEL}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/train_deepscores_symbol_detector.py" \
  --prepared-dir "${PREPARED_DIR}" \
  --images-dir "${PROJECT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --initial-backbone-model "${INITIAL_BACKBONE_MODEL}" \
  --epochs 6 \
  --batch-size 2 \
  --evaluation-batch-size 8 \
  --accumulate 2 \
  --workers "${LOADER_WORKERS}" \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --eval-every 1 \
  --rare-class-sampling-power 0.35 \
  --rare-class-max-repeat 4 \
  --resume
