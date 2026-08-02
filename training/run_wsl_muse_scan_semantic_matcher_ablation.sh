#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
PREPARED_DIR="${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_stratified_complete_page_overlap_consistent_deduplicated_v7"
IMAGES_DIR="${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_stratified_complete_page_v6"
OUTPUT_DIR="${PROJECT_DIR}/training_data/models/muse-omr-scan-semantic-detector-v5-matcher35-25-ablation-e12-runtime2-20260730"
INITIAL_MODEL="${PROJECT_DIR}/training_data/models/muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730/model.best.pt"
INITIAL_CATEGORIES="${PREPARED_DIR}/categories.json"
REPLAY_PREPARED_DIR="${PROJECT_DIR}/training_data/prepared/openscore_lieder_train_1091_svg_regions_complete_page_overlap_consistent_deduplicated_v4"
REPLAY_IMAGES_DIR="${PROJECT_DIR}/training_data/prepared/openscore_lieder_train_1091_svg_regions_complete_page_v2"
TRAIN_DEVICE="${SCORESCAN_TRAIN_DEVICE:-cuda}"
CPU_THREADS="${SCORESCAN_CPU_THREADS:-0}"
LOADER_WORKERS="${SCORESCAN_LOADER_WORKERS:-2}"
ADAPTIVE_FULL_BATCH_OBJECT_LIMIT="${SCORESCAN_ADAPTIVE_FULL_BATCH_OBJECT_LIMIT:-80}"
ADAPTIVE_FULL_BATCH_MIN_FREE_MIB="${SCORESCAN_ADAPTIVE_FULL_BATCH_MIN_FREE_MIB:-4608}"
RUNTIME_STOP_AFTER_EPOCH="${SCORESCAN_RUNTIME_STOP_AFTER_EPOCH:-2}"
RUNTIME_STOP_REASON="${SCORESCAN_RUNTIME_STOP_REASON:-matcher35-25-structural-ablation-before-full-run}"
IMAGE_CACHE_DIR="${SCORESCAN_MUSE_IMAGE_CACHE:-${HOME}/.cache/scorescan/detector-images/muse-scan-semantic-v1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! "${RUNTIME_STOP_AFTER_EPOCH}" =~ ^[0-9]+$ ]] ||
   (( RUNTIME_STOP_AFTER_EPOCH < 2 || RUNTIME_STOP_AFTER_EPOCH > 12 )); then
  echo "Runtime stop epoch must be an integer in [2, 12]." >&2
  exit 1
fi
if [[ -z "${RUNTIME_STOP_REASON}" ]]; then
  echo "Runtime stop reason must not be empty." >&2
  exit 1
fi

if [[ ! -s "${PREPARED_DIR}/manifest.json" ||
      ! -s "${PREPARED_DIR}/train.jsonl" ||
      ! -s "${PREPARED_DIR}/test.jsonl" ]]; then
  echo "Registered complete-page Muse dataset is incomplete: ${PREPARED_DIR}" >&2
  exit 1
fi
if [[ ! -s "${INITIAL_MODEL}" || ! -s "${INITIAL_CATEGORIES}" ]]; then
  echo "Matcher ablation initialization is incomplete: ${INITIAL_MODEL}" >&2
  exit 1
fi

# Preserve the intended twelve-epoch release schedule in the immutable run
# configuration, but stop after the first evaluated two-epoch window.  A full
# run is authorized only when this matched baseline comparison improves the
# weak positioned-mark classes enough to justify additional GPU time.
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
  --epochs 12 \
  --runtime-stop-after-epoch "${RUNTIME_STOP_AFTER_EPOCH}" \
  --runtime-stop-reason "${RUNTIME_STOP_REASON}" \
  --batch-size 2 \
  --microbatch-size 1 \
  --adaptive-full-batch-object-limit "${ADAPTIVE_FULL_BATCH_OBJECT_LIMIT}" \
  --adaptive-full-batch-min-free-mib "${ADAPTIVE_FULL_BATCH_MIN_FREE_MIB}" \
  --evaluation-batch-size 8 \
  --accumulate 2 \
  --workers "${LOADER_WORKERS}" \
  --learning-rate 0.00005 \
  --weight-decay 0.0001 \
  --eval-every 2 \
  --stop-when-accepted \
  --external-acceptance-pending \
  --rare-class-sampling-power 0.4 \
  --rare-class-max-repeat 5 \
  --minimum-best-map-50 0.95 \
  --minimum-best-map-75 0.90 \
  --minimum-best-priority-map 0.85 \
  --minimum-required-class-test-objects 25 \
  --required-class-map arpeggio=0.75 \
  --required-class-map augmentationDot=0.85 \
  --required-class-map beam=0.90 \
  --required-class-map bracket=0.80 \
  --required-class-map breathMark=0.70 \
  --required-class-map fermata=0.80 \
  --required-class-map fingeringText=0.80 \
  --required-class-map flag=0.90 \
  --required-class-map genericAccidental=0.90 \
  --required-class-map genericArticulation=0.85 \
  --required-class-map genericBarline=0.92 \
  --required-class-map genericClef=0.95 \
  --required-class-map genericDynamic=0.85 \
  --required-class-map genericKeySignature=0.92 \
  --required-class-map genericRest=0.92 \
  --required-class-map genericTimeSignature=0.92 \
  --required-class-map graceSlash=0.75 \
  --required-class-map hairpin=0.85 \
  --required-class-map instrumentNameText=0.80 \
  --required-class-map measureNumberText=0.85 \
  --required-class-map ottava=0.80 \
  --required-class-map pedal=0.80 \
  --required-class-map slur=0.85 \
  --required-class-map tempoText=0.85 \
  --required-class-map tie=0.85 \
  --required-class-map tuplet=0.85 \
  --required-class-map volta=0.80 \
  --device "${TRAIN_DEVICE}" \
  --cpu-threads "${CPU_THREADS}" \
  --resumable-augmentation-v3 \
  --resume
