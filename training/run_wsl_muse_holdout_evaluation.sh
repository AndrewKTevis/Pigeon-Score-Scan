#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
HOLDOUT_DIR="${SCORESCAN_HOLDOUT_PREPARED_DIR:-${PROJECT_DIR}/training_data/prepared/muse_omr_scan_holdout_regions_stratified_complete_page_overlap_consistent_deduplicated_v7}"
HOLDOUT_IMAGES_DIR="${SCORESCAN_HOLDOUT_IMAGES_DIR:-${PROJECT_DIR}/training_data/prepared/muse_omr_scan_holdout_regions_stratified_complete_page_v6}"
CALIBRATION_DIR="${SCORESCAN_CALIBRATION_PREPARED_DIR:-${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_stratified_complete_page_overlap_consistent_deduplicated_v7}"
CALIBRATION_IMAGES_DIR="${SCORESCAN_CALIBRATION_IMAGES_DIR:-${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_stratified_complete_page_v6}"
HOLDOUT_LAYOUT_EVIDENCE="${SCORESCAN_HOLDOUT_LAYOUT_EVIDENCE:-${PROJECT_DIR}/training_data/benchmarks/muse-holdout-semantic-page-layout-evidence-v1.json}"
CALIBRATION_LAYOUT_EVIDENCE="${SCORESCAN_CALIBRATION_LAYOUT_EVIDENCE:-${PROJECT_DIR}/training_data/benchmarks/muse-training-semantic-page-layout-evidence-v1.json}"
MODEL_DIR="${SCORESCAN_DETECTOR_MODEL_DIR:-${PROJECT_DIR}/training_data/models/muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730}"
MODEL="${SCORESCAN_DETECTOR_MODEL:-${MODEL_DIR}/model.best.pt}"
MODEL_CATEGORIES="${SCORESCAN_DETECTOR_CATEGORIES:-${CALIBRATION_DIR}/categories.json}"
OUTPUT_REPORT="${SCORESCAN_HOLDOUT_OUTPUT_REPORT:-${MODEL_DIR}/evaluation.independent-muse-holdout.json}"
EVAL_DEVICE="${SCORESCAN_TRAIN_DEVICE:-cuda}"
CPU_THREADS="${SCORESCAN_CPU_THREADS:-0}"
LOADER_WORKERS="${SCORESCAN_LOADER_WORKERS:-4}"
CALIBRATION_SPLIT="${SCORESCAN_CALIBRATION_SPLIT:-}"
CALIBRATION_SPLIT_ARGS=()
if [[ -n "${CALIBRATION_SPLIT}" ]]; then
  case "${CALIBRATION_SPLIT}" in
    train|calibration|test)
      CALIBRATION_SPLIT_ARGS=(
        --operating-point-calibration-split "${CALIBRATION_SPLIT}"
      )
      ;;
    *)
      echo "Unsupported threshold-calibration split: ${CALIBRATION_SPLIT}" >&2
      exit 1
      ;;
  esac
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -s "${HOLDOUT_DIR}/prepare-report.json" ||
      ! -s "${HOLDOUT_DIR}/test.jsonl" ||
      ! -s "${CALIBRATION_DIR}/prepare-report.json" ||
      ! -s "${CALIBRATION_DIR}/train.jsonl" ||
      ! -s "${CALIBRATION_DIR}/calibration.jsonl" ||
      ! -s "${CALIBRATION_DIR}/test.jsonl" ||
      ! -s "${HOLDOUT_LAYOUT_EVIDENCE}" ||
      ! -s "${CALIBRATION_LAYOUT_EVIDENCE}" ||
      ! -s "${MODEL}" ||
      ! -s "${MODEL_CATEGORIES}" ]]; then
  echo "Independent Muse OMR holdout or trained detector is incomplete" >&2
  exit 1
fi
if [[ -s "${OUTPUT_REPORT}" ]]; then
  exec "${PYTHON_BIN}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("acceptance",{}).get("passed") is True else 1)' \
    "${OUTPUT_REPORT}"
fi

exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/evaluate_semantic_detector_holdout.py" \
  --prepared-dir "${HOLDOUT_DIR}" \
  --images-dir "${HOLDOUT_IMAGES_DIR}" \
  --operating-point-calibration-prepared-dir "${CALIBRATION_DIR}" \
  --operating-point-calibration-images-dir "${CALIBRATION_IMAGES_DIR}" \
  "${CALIBRATION_SPLIT_ARGS[@]}" \
  --page-layout-evidence "${HOLDOUT_LAYOUT_EVIDENCE}" \
  --operating-point-calibration-page-layout-evidence "${CALIBRATION_LAYOUT_EVIDENCE}" \
  --model "${MODEL}" \
  --model-categories "${MODEL_CATEGORIES}" \
  --output-report "${OUTPUT_REPORT}" \
  --batch-size 8 \
  --workers "${LOADER_WORKERS}" \
  --minimum-map-50 0.95 \
  --minimum-map-75 0.90 \
  --minimum-priority-map 0.85 \
  --minimum-independent-works 200 \
  --minimum-required-class-test-objects 25 \
  --minimum-operating-point-recall 0.98 \
  --minimum-high-recall-mark-recall 0.99 \
  --minimum-operating-point-calibration-true-positives 10 \
  --required-class-map arpeggio=0.75 \
  --required-class-map augmentationDot=0.85 \
  --required-class-map beam=0.90 \
  --required-class-map bracket=0.80 \
  --required-class-map breathMark=0.70 \
  --required-class-map expressionText=0.80 \
  --required-class-map fermata=0.80 \
  --required-class-map fingeringText=0.80 \
  --required-class-map flag=0.90 \
  --required-class-map genericAccidental=0.90 \
  --required-class-map genericArticulation=0.85 \
  --required-class-map genericBarline=0.92 \
  --required-class-map genericClef=0.95 \
  --required-class-map genericDynamic=0.85 \
  --required-class-map genericKeySignature=0.92 \
  --required-class-map genericOrnament=0.80 \
  --required-class-map genericRest=0.92 \
  --required-class-map genericTimeSignature=0.92 \
  --required-class-map glissando=0.70 \
  --required-class-map graceSlash=0.75 \
  --required-class-map hairpin=0.85 \
  --required-class-map instrumentNameText=0.80 \
  --required-class-map jumpText=0.75 \
  --required-class-map markerText=0.80 \
  --required-class-map measureNumberText=0.85 \
  --required-class-map ottava=0.80 \
  --required-class-map parenthesis=0.75 \
  --required-class-map pedal=0.80 \
  --required-class-map rehearsalMarkText=0.85 \
  --required-class-map scoreText=0.80 \
  --required-class-map slur=0.85 \
  --required-class-map staffText=0.80 \
  --required-class-map systemText=0.80 \
  --required-class-map techniqueText=0.80 \
  --required-class-map tempoText=0.85 \
  --required-class-map textLine=0.75 \
  --required-class-map tie=0.85 \
  --required-class-map tremoloBetweenNotes=0.75 \
  --required-class-map tremoloSingle=0.80 \
  --required-class-map trillExtension=0.75 \
  --required-class-map tuplet=0.85 \
  --required-class-map volta=0.80 \
  --device "${EVAL_DEVICE}" \
  --cpu-threads "${CPU_THREADS}"
