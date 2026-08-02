#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PADDLEOCR_DIR="${PROJECT_DIR}/training_data/external/corpora/paddleocr_2661c7c0"
PYTHON_BIN="${SCORESCAN_PADDLEOCR_PYTHON:-${HOME}/.cache/scorescan/paddleocr-3.2-cu126-py311/bin/python}"
CONFIG="${SCRIPT_DIR}/ppocrv6_scorescan_rec.yml"
LABEL_DIR="${PROJECT_DIR}/training_data/prepared/scorescan_ocr_training_stratified_v2"
OUTPUT_DIR="${PROJECT_DIR}/training_data/models/ppocrv6-scorescan-rec-stratified-e18-b8-20260729"
BEST_MODEL="${OUTPUT_DIR}/best_accuracy.pdparams"
LATEST_MODEL="${OUTPUT_DIR}/latest.pdparams"
LATEST_STATE="${OUTPUT_DIR}/latest.states"
CALIBRATION_SCAN_LOG="${OUTPUT_DIR}/evaluation.calibration-scan.log"
CALIBRATION_CLEAN_LOG="${OUTPUT_DIR}/evaluation.calibration-clean.log"
CALIBRATION_REPORT="${OUTPUT_DIR}/calibration-early-stop.json"
SCAN_LOG="${OUTPUT_DIR}/evaluation.scan-test.log"
CLEAN_LOG="${OUTPUT_DIR}/evaluation.clean-test.log"
GATE_REPORT="${OUTPUT_DIR}/release-gate.json"
INFERENCE_DIR="${OUTPUT_DIR}/inference"
ONNX_MODEL="${OUTPUT_DIR}/scorescan-ppocrv6-rec.onnx"
TOTAL_EPOCHS=18
EPOCHS_PER_INVOCATION=2
LIBGOMP_LIBRARY="$(
  find "${HOME}/.cache/scorescan/syslibs/libgomp/root" \
    -name "libgomp.so.1" -print -quit 2>/dev/null || true
)"

if ! ldconfig -p 2>/dev/null | grep -q "libgomp.so.1"; then
  if [[ -z "${LIBGOMP_LIBRARY}" ]]; then
    echo "User-space libgomp.so.1 is missing; rerun the Paddle bootstrap" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="$(
    dirname "${LIBGOMP_LIBRARY}"
  )${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

for required in \
  "${PYTHON_BIN}" \
  "${PADDLEOCR_DIR}/tools/train.py" \
  "${CONFIG}" \
  "${LABEL_DIR}/merge-report.json" \
  "${LABEL_DIR}/train.balanced.paddle.txt" \
  "${LABEL_DIR}/calibration.scan.paddle.txt" \
  "${LABEL_DIR}/test.scan.paddle.txt" \
  "${LABEL_DIR}/test.clean.paddle.txt"; do
  if [[ ! -s "${required}" ]]; then
    echo "Required PaddleOCR training input is missing: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${PADDLEOCR_DIR}"

completed_epoch() {
  if [[ ! -s "${LATEST_STATE}" ]]; then
    printf '0\n'
    return
  fi
  "${PYTHON_BIN}" - "${LATEST_STATE}" <<'PY'
import pickle
import pathlib
import sys

state = pickle.loads(pathlib.Path(sys.argv[1]).read_bytes())
print(int(state.get("epoch", 0)))
PY
}

prune_duplicate_best_training_state() {
  # Paddle writes a generic best_model copy in addition to best_accuracy and
  # stores an optimizer beside the inference-selected best weights. Resumption
  # is bound exclusively to latest.{pdparams,pdopt,states}; evaluation/export
  # uses best_accuracy.pdparams. Keep those two roles and discard only the
  # duplicated, non-resumable state after train.py has closed its writers.
  rm -rf -- "${OUTPUT_DIR}/best_model"
  rm -f -- "${OUTPUT_DIR}/best_accuracy.pdopt"
  rm -f -- \
    "${OUTPUT_DIR}"/iter_epoch_*.pdparams \
    "${OUTPUT_DIR}"/iter_epoch_*.pdopt \
    "${OUTPUT_DIR}"/iter_epoch_*.states
}

while (( "$(completed_epoch)" < TOTAL_EPOCHS )); do
  before_epoch="$(completed_epoch)"
  train_options=()
  if [[ -s "${LATEST_MODEL}" && -s "${LATEST_STATE}" ]]; then
    train_options+=("-o" "Global.checkpoints=${OUTPUT_DIR}/latest")
  fi
  export SCORESCAN_MAX_EPOCHS_THIS_RUN="${EPOCHS_PER_INVOCATION}"
  "${PYTHON_BIN}" tools/train.py -c "${CONFIG}" "${train_options[@]}" 2>&1 |
    tee -a "${OUTPUT_DIR}/training.log"
  unset SCORESCAN_MAX_EPOCHS_THIS_RUN
  after_epoch="$(completed_epoch)"
  if (( after_epoch <= before_epoch )); then
    echo "PaddleOCR recognition checkpoint did not advance" >&2
    exit 1
  fi
  if [[ ! -s "${BEST_MODEL}" ]]; then
    echo "PaddleOCR recognition training did not produce ${BEST_MODEL}" >&2
    exit 1
  fi
  prune_duplicate_best_training_state

  "${PYTHON_BIN}" tools/eval.py -c "${CONFIG}" \
    -o "Global.pretrained_model=${OUTPUT_DIR}/best_accuracy" \
       "Eval.dataset.label_file_list=[${LABEL_DIR}/calibration.scan.paddle.txt]" \
    2>&1 | tee "${CALIBRATION_SCAN_LOG}"
  "${PYTHON_BIN}" tools/eval.py -c "${CONFIG}" \
    -o "Global.pretrained_model=${OUTPUT_DIR}/best_accuracy" \
       "Eval.dataset.label_file_list=[${LABEL_DIR}/calibration.clean.paddle.txt]" \
    2>&1 | tee "${CALIBRATION_CLEAN_LOG}"

  cd "${PROJECT_DIR}"
  if "${PYTHON_BIN}" -m app.tools.check_paddleocr_recognition_calibration \
      --scan-log "${CALIBRATION_SCAN_LOG}" \
      --clean-log "${CALIBRATION_CLEAN_LOG}" \
      --output-report "${CALIBRATION_REPORT}" \
      --model "${BEST_MODEL}" \
      --dataset-report "${LABEL_DIR}/merge-report.json" \
      --minimum-accuracy 0.999 \
      --minimum-normalized-edit 0.9997 \
      --completed-epoch "${after_epoch}"; then
    cd "${PADDLEOCR_DIR}"
    break
  fi
  cd "${PADDLEOCR_DIR}"
done

if [[ ! -s "${BEST_MODEL}" ]]; then
  echo "PaddleOCR training did not produce ${BEST_MODEL}" >&2
  exit 1
fi

"${PYTHON_BIN}" tools/eval.py -c "${CONFIG}" \
  -o "Global.pretrained_model=${OUTPUT_DIR}/best_accuracy" \
     "Eval.dataset.label_file_list=[${LABEL_DIR}/test.scan.paddle.txt]" \
  2>&1 | tee "${SCAN_LOG}"

"${PYTHON_BIN}" tools/eval.py -c "${CONFIG}" \
  -o "Global.pretrained_model=${OUTPUT_DIR}/best_accuracy" \
     "Eval.dataset.label_file_list=[${LABEL_DIR}/test.clean.paddle.txt]" \
  2>&1 | tee "${CLEAN_LOG}"

cd "${PROJECT_DIR}"
if "${PYTHON_BIN}" -m app.tools.gate_paddleocr_evaluation \
    --scan-log "${SCAN_LOG}" \
    --clean-log "${CLEAN_LOG}" \
    --output-report "${GATE_REPORT}" \
    --model "${BEST_MODEL}" \
    --dataset-report "${LABEL_DIR}/merge-report.json" \
    --minimum-scan-accuracy 0.998 \
    --minimum-scan-normalized-edit 0.9995 \
    --minimum-clean-accuracy 0.998 \
    --minimum-clean-normalized-edit 0.9995; then
  echo "Paddle in-framework recognition diagnostic gate passed"
else
  echo \
    "Paddle recognition diagnostic gate did not pass; exporting the non-promotable candidate for independent runtime evaluation" \
    >&2
fi

cd "${PADDLEOCR_DIR}"
"${PYTHON_BIN}" tools/export_model.py -c "${CONFIG}" \
  -o "Global.pretrained_model=${OUTPUT_DIR}/best_accuracy" \
     "Global.save_inference_dir=${INFERENCE_DIR}"

if [[ -s "${INFERENCE_DIR}/inference.json" ]]; then
  MODEL_FILENAME="inference.json"
elif [[ -s "${INFERENCE_DIR}/inference.pdmodel" ]]; then
  MODEL_FILENAME="inference.pdmodel"
else
  echo "PaddleOCR export did not produce an inference graph" >&2
  exit 1
fi
if [[ ! -s "${INFERENCE_DIR}/inference.pdiparams" ]]; then
  echo "PaddleOCR export did not produce inference parameters" >&2
  exit 1
fi

"${VENV_DIR:-${HOME}/.cache/scorescan/paddleocr-3.2-cu126-py311}/bin/paddle2onnx" \
  --model_dir "${INFERENCE_DIR}" \
  --model_filename "${MODEL_FILENAME}" \
  --params_filename "inference.pdiparams" \
  --save_file "${ONNX_MODEL}" \
  --opset_version 17 \
  --enable_onnx_checker True

if [[ ! -s "${ONNX_MODEL}" ]]; then
  echo "Paddle2ONNX did not produce ${ONNX_MODEL}" >&2
  exit 1
fi
