#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
PREPARED_DIR="${PROJECT_DIR}/training_data/prepared/openscore_lieder_train_1091_svg_regions_overlap_consistent_deduplicated_v3"
TRAIN_IMAGES_DIR="${PROJECT_DIR}/training_data/prepared/openscore_lieder_train_1091_svg_regions_v1"
EVALUATION_DIR="${PROJECT_DIR}/training_data/prepared/openscore_quartet_lieder_semantic_overlap_consistent_deduplicated_v3"
REPLAY_DIR="${PROJECT_DIR}/training_data/prepared/openscore_string_quartets_svg_regions_overlap_consistent_deduplicated_v4"
PREPARATION_FAILURE="${PREPARED_DIR}/preparation.failed.txt"
OUTPUT_DIR="${1:-${PROJECT_DIR}/training_data/models/openscore-lieder-semantic-detector-e6-b2-20260728}"
RECOVERY_OUTPUT_DIR="${PROJECT_DIR}/training_data/models/openscore-semantic-detector-overlap-recovery-v3-e4-b2-20260729"
INITIAL_MODEL="${2:-${RECOVERY_OUTPUT_DIR}/model.best.pt}"
INITIAL_CATEGORIES="${REPLAY_DIR}/categories.json"
TRAIN_DEVICE="${SCORESCAN_TRAIN_DEVICE:-cuda}"
CPU_THREADS="${SCORESCAN_CPU_THREADS:-0}"
# Keep host-memory use bounded while the 8 GiB GPU remains the bottleneck.
LOADER_WORKERS="${SCORESCAN_LOADER_WORKERS:-2}"
ADAPTIVE_FULL_BATCH_OBJECT_LIMIT="${SCORESCAN_ADAPTIVE_FULL_BATCH_OBJECT_LIMIT:-80}"
ADAPTIVE_FULL_BATCH_MIN_FREE_MIB="${SCORESCAN_ADAPTIVE_FULL_BATCH_MIN_FREE_MIB:-4608}"
IMAGE_CACHE_DIR="${SCORESCAN_LIEDER_IMAGE_CACHE:-${HOME}/.cache/scorescan/detector-images/openscore-lieder-semantic-v1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

while [[ ! -s "${PREPARED_DIR}/manifest.json" ||
         ! -s "${PREPARED_DIR}/train.jsonl" ||
         ! -s "${PREPARED_DIR}/prepare-report.json" ||
         ! -s "${EVALUATION_DIR}/manifest.json" ||
         ! -s "${EVALUATION_DIR}/test.jsonl" ||
         ! -s "${EVALUATION_DIR}/prepare-report.json" ||
         ! -s "${REPLAY_DIR}/manifest.json" ||
         ! -s "${REPLAY_DIR}/train.jsonl" ||
         ! -s "${REPLAY_DIR}/prepare-report.json" ]]; do
  if [[ -s "${PREPARATION_FAILURE}" ]]; then
    echo "OpenScore Lieder semantic preparation failed:" >&2
    cat "${PREPARATION_FAILURE}" >&2
    exit 1
  fi
  sleep 20
done
if [[ $# -lt 2 ]]; then
  # The former quartet run saw contradictory background labels in overlapping
  # tiles. Recover on the immutable overlap-consistent targets before Lieder
  # fine-tuning. Keeping this inside the serial entry point also upgrades an
  # already-running Windows queue when it reaches this script.
  bash "${SCRIPT_DIR}/run_wsl_openscore_overlap_recovery.sh" \
    "${RECOVERY_OUTPUT_DIR}" \
    "${PROJECT_DIR}/training_data/models/openscore-semantic-detector-e6-b2-20260728/model.best.pt"
fi
if [[ ! -s "${INITIAL_MODEL}" || ! -s "${INITIAL_CATEGORIES}" ]]; then
  echo "Compatible quartet initialization is incomplete: ${INITIAL_MODEL}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/train_deepscores_symbol_detector.py" \
  --prepared-dir "${PREPARED_DIR}" \
  --images-dir "${TRAIN_IMAGES_DIR}" \
  --image-cache-dir "${IMAGE_CACHE_DIR}" \
  --populate-image-cache \
  --evaluation-prepared-dir "${EVALUATION_DIR}" \
  --evaluation-images-dir "${PROJECT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --initial-model "${INITIAL_MODEL}" \
  --initial-categories "${INITIAL_CATEGORIES}" \
  --replay-prepared-dir "${REPLAY_DIR}" \
  --replay-images-dir "${PROJECT_DIR}" \
  --replay-fraction 0.20 \
  --replay-max-train-tiles 60000 \
  --replay-rare-class-max-repeat 20 \
  --epochs 8 \
  --batch-size 2 \
  --microbatch-size 1 \
  --adaptive-full-batch-object-limit "${ADAPTIVE_FULL_BATCH_OBJECT_LIMIT}" \
  --adaptive-full-batch-min-free-mib "${ADAPTIVE_FULL_BATCH_MIN_FREE_MIB}" \
  --evaluation-batch-size 8 \
  --accumulate 2 \
  --workers "${LOADER_WORKERS}" \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --eval-every 2 \
  --stop-when-accepted \
  --rare-class-sampling-power 0.35 \
  --rare-class-max-repeat 4 \
  --minimum-best-map-50 0.95 \
  --minimum-best-map-75 0.90 \
  --minimum-best-priority-map 0.85 \
  --device "${TRAIN_DEVICE}" \
  --cpu-threads "${CPU_THREADS}" \
  --resumable-augmentation-v3 \
  --resume
