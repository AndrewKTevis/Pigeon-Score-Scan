#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
MODEL_DIR="${PROJECT_DIR}/training_data/models/muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730"
CANDIDATE_DIR="${PROJECT_DIR}/training_data/release_candidates/semantic-detector-muse-v4-complete-page-e12-20260730"
MODEL="${MODEL_DIR}/model.best.pt"
CATEGORIES="${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_stratified_complete_page_overlap_consistent_deduplicated_v7/categories.json"
SAMPLE_REPORT="${CANDIDATE_DIR}/parity-sample.json"
ONNX_MODEL="${CANDIDATE_DIR}/semantic_detector.onnx"
PARITY_REPORT="${CANDIDATE_DIR}/semantic_detector_onnx_parity.json"

for path in \
  "${MODEL}" \
  "${CATEGORIES}" \
  "${MODEL_DIR}/training_report.json" \
  "${MODEL_DIR}/evaluation.independent-muse-holdout.json" \
  "${SAMPLE_REPORT}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Semantic release-candidate input is missing: ${path}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" -c \
  'import onnx,onnxruntime; assert onnx.__version__=="1.19.1"; assert onnxruntime.__version__=="1.23.2"'

if [[ -s "${ONNX_MODEL}" && -s "${PARITY_REPORT}" ]]; then
  exec "${PYTHON_BIN}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("passed") is True else 1)' \
    "${PARITY_REPORT}"
fi
if [[ -e "${ONNX_MODEL}" || -e "${PARITY_REPORT}" ]]; then
  echo "Refusing partial/stale semantic release-candidate export" >&2
  exit 1
fi

readarray -t SAMPLE < <(
  "${PYTHON_BIN}" - "${SAMPLE_REPORT}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["image"])
print(",".join(str(value) for value in payload["crop_xyxy"]))
PY
)
if [[ "${#SAMPLE[@]}" -ne 2 ]]; then
  echo "Semantic parity sample report is invalid" >&2
  exit 1
fi

cd -- "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/export_semantic_detector_onnx.py" \
  --model "${MODEL}" \
  --categories "${CATEGORIES}" \
  --parity-image "${SAMPLE[0]}" \
  --parity-crop "${SAMPLE[1]}" \
  --output "${ONNX_MODEL}" \
  --output-report "${PARITY_REPORT}" \
  --minimum-parity-detections 1 \
  --maximum-box-error 0.10 \
  --maximum-score-error 0.0001 \
  --device cpu
