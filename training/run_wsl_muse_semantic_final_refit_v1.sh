#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
PREPARED_DIR="${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_final_refit_v1"
IMAGES_DIR="${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_stratified_complete_page_v6"
OUTPUT_DIR="${PROJECT_DIR}/training_data/models/muse-omr-scan-semantic-detector-v7-final-refit-e2-b2-20260730"
INITIAL_MODEL="${PROJECT_DIR}/training_data/models/muse-omr-scan-semantic-detector-v5-matcher35-25-ablation-e12-runtime2-20260730/model.best.pt"
INITIAL_CATEGORIES="${PREPARED_DIR}/categories.json"
REPLAY_PREPARED_DIR="${PROJECT_DIR}/training_data/prepared/openscore_lieder_train_1091_svg_regions_complete_page_overlap_consistent_deduplicated_v4"
REPLAY_IMAGES_DIR="${PROJECT_DIR}/training_data/prepared/openscore_lieder_train_1091_svg_regions_complete_page_v2"
TRAIN_DEVICE="${SCORESCAN_TRAIN_DEVICE:-cuda}"
LOADER_WORKERS="${SCORESCAN_LOADER_WORKERS:-2}"
ADAPTIVE_FULL_BATCH_OBJECT_LIMIT="${SCORESCAN_ADAPTIVE_FULL_BATCH_OBJECT_LIMIT:-80}"
ADAPTIVE_FULL_BATCH_MIN_FREE_MIB="${SCORESCAN_ADAPTIVE_FULL_BATCH_MIN_FREE_MIB:-4608}"
IMAGE_CACHE_DIR="${SCORESCAN_MUSE_IMAGE_CACHE:-${HOME}/.cache/scorescan/detector-images/muse-scan-semantic-v1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

for path in \
  "${PREPARED_DIR}/manifest.json" \
  "${PREPARED_DIR}/prepare-report.json" \
  "${PREPARED_DIR}/train.jsonl" \
  "${PREPARED_DIR}/test.jsonl" \
  "${INITIAL_MODEL}" \
  "${INITIAL_CATEGORIES}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Final-refit input is missing: ${path}" >&2
    exit 1
  fi
done

# Architecture, matcher, sampling, and the two-epoch refit window were frozen
# before this partition was materialized. The prior internal test fold may now
# join training; the prior calibration fold is diagnostic only. This run is
# never release evidence, and the untouched forbidden external holdout remains
# the only detector-candidate acceptance set.
# The two-epoch endpoint was frozen in advance. Evaluating only that endpoint
# prevents the calibration fold from selecting between epoch 1 and epoch 2.
exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/train_deepscores_symbol_detector.py" \
  --prepared-dir "${PREPARED_DIR}" \
  --images-dir "${IMAGES_DIR}" \
  --require-complete-page-targets \
  --image-cache-dir "${IMAGE_CACHE_DIR}" \
  --populate-image-cache \
  --output-dir "${OUTPUT_DIR}" \
  --initial-model "${INITIAL_MODEL}" \
  --initial-categories "${INITIAL_CATEGORIES}" \
  --replay-prepared-dir "${REPLAY_PREPARED_DIR}" \
  --replay-images-dir "${REPLAY_IMAGES_DIR}" \
  --replay-fraction 0.35 \
  --replay-max-train-tiles 60000 \
  --replay-rare-class-max-repeat 20 \
  --epochs 2 \
  --batch-size 2 \
  --microbatch-size 1 \
  --adaptive-full-batch-object-limit "${ADAPTIVE_FULL_BATCH_OBJECT_LIMIT}" \
  --adaptive-full-batch-min-free-mib "${ADAPTIVE_FULL_BATCH_MIN_FREE_MIB}" \
  --evaluation-batch-size 8 \
  --accumulate 2 \
  --workers "${LOADER_WORKERS}" \
  --learning-rate 0.00002 \
  --weight-decay 0.0001 \
  --eval-every 2 \
  --external-acceptance-pending \
  --rare-class-sampling-power 0.4 \
  --rare-class-max-repeat 5 \
  --minimum-best-map-50 0.95 \
  --minimum-best-map-75 0.90 \
  --minimum-best-priority-map 0.85 \
  --minimum-required-class-test-objects 25 \
  --device "${TRAIN_DEVICE}" \
  --resumable-augmentation-v3 \
  --resume
