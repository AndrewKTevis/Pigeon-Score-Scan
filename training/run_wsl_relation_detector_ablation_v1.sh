#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
PREPARED_DIR="${PROJECT_DIR}/training_data/prepared/muse_omr_relation_detector_subset_v1"
IMAGES_DIR="${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_stratified_complete_page_v6"
OUTPUT_DIR="${PROJECT_DIR}/training_data/models/muse-omr-relation-detector-v1-e4-stage1-20260801"
INITIAL_BACKBONE="${PROJECT_DIR}/training_data/models/muse-omr-scan-semantic-detector-v7-final-refit-e2-b2-20260730/model.best.pt"
IMAGE_CACHE_DIR="${SCORESCAN_MUSE_IMAGE_CACHE:-${HOME}/.cache/scorescan/detector-images/muse-scan-semantic-v1}"
TRAIN_DEVICE="${SCORESCAN_TRAIN_DEVICE:-cuda}"
LOADER_WORKERS="${SCORESCAN_LOADER_WORKERS:-2}"
STOP_AFTER_EPOCH="${SCORESCAN_RELATION_STOP_AFTER_EPOCH:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

for path in \
  "${PREPARED_DIR}/manifest.json" \
  "${PREPARED_DIR}/prepare-report.json" \
  "${PREPARED_DIR}/categories.json" \
  "${PREPARED_DIR}/train.jsonl" \
  "${PREPARED_DIR}/test.jsonl" \
  "${INITIAL_BACKBONE}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Relation-detector input is missing: ${path}" >&2
    exit 1
  fi
done

# The four-epoch plan and all gates are frozen before the first stage.  Runtime
# truncation at epoch one is an efficiency decision only; the same optimizer and RNG
# state can resume if, and only if, the first independent test justifies it.
exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/train_deepscores_symbol_detector.py" \
  --prepared-dir "${PREPARED_DIR}" \
  --images-dir "${IMAGES_DIR}" \
  --require-complete-page-targets \
  --image-cache-dir "${IMAGE_CACHE_DIR}" \
  --populate-image-cache \
  --output-dir "${OUTPUT_DIR}" \
  --initial-backbone-model "${INITIAL_BACKBONE}" \
  --epochs 4 \
  --batch-size 2 \
  --microbatch-size 1 \
  --adaptive-full-batch-object-limit 80 \
  --adaptive-full-batch-min-free-mib 4608 \
  --evaluation-batch-size 8 \
  --accumulate 2 \
  --workers "${LOADER_WORKERS}" \
  --learning-rate 0.00005 \
  --weight-decay 0.0001 \
  --eval-every 1 \
  --external-acceptance-pending \
  --rare-class-sampling-power 0.35 \
  --rare-class-max-repeat 4 \
  --minimum-best-map-50 0.90 \
  --minimum-best-map-75 0.80 \
  --minimum-best-priority-map 0.85 \
  --minimum-required-class-test-objects 25 \
  --required-class-map slur=0.85 \
  --required-class-map tie=0.85 \
  --required-class-map hairpin=0.85 \
  --runtime-stop-after-epoch "${STOP_AFTER_EPOCH}" \
  --runtime-stop-reason relation-detector-stage-gate \
  --device "${TRAIN_DEVICE}" \
  --resumable-augmentation-v3 \
  --resume
