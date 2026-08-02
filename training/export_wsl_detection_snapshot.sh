#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 MODEL_PREFIX OUTPUT_DIRECTORY" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PADDLEOCR_DIR="${PROJECT_DIR}/training_data/external/corpora/paddleocr_2661c7c0"
PYTHON_BIN="${SCORESCAN_PADDLEOCR_PYTHON:-${HOME}/.cache/scorescan/paddleocr-3.2-cu126-py311/bin/python}"
PADDLE2ONNX_BIN="${HOME}/.cache/scorescan/paddleocr-3.2-cu126-py311/bin/paddle2onnx"
CONFIG="${SCRIPT_DIR}/ppocrv6_scorescan_det.yml"
MODEL_PREFIX="$(realpath -m -- "$1")"
OUTPUT_DIR="$(realpath -m -- "$2")"
INFERENCE_DIR="${OUTPUT_DIR}/inference"
ONNX_MODEL="${OUTPUT_DIR}/scorescan-ppocrv6-det.onnx"
LIBGOMP_LIBRARY="$(
  find "${HOME}/.cache/scorescan/syslibs/libgomp/root" \
    -name "libgomp.so.1" -print -quit 2>/dev/null || true
)"

case "${MODEL_PREFIX}" in
  "${PROJECT_DIR}"/training_data/models/*) ;;
  *)
    echo "Model prefix must stay inside training_data/models" >&2
    exit 2
    ;;
esac
case "${OUTPUT_DIR}" in
  "${PROJECT_DIR}"/training_data/models/*) ;;
  *)
    echo "Output directory must stay inside training_data/models" >&2
    exit 2
    ;;
esac

for required in \
  "${PYTHON_BIN}" \
  "${PADDLE2ONNX_BIN}" \
  "${PADDLEOCR_DIR}/tools/export_model.py" \
  "${CONFIG}" \
  "${MODEL_PREFIX}.pdparams"; do
  if [[ ! -s "${required}" ]]; then
    echo "Required detection export input is missing: ${required}" >&2
    exit 1
  fi
done
if [[ -e "${ONNX_MODEL}" ]]; then
  echo "Refusing to overwrite an existing ONNX snapshot: ${ONNX_MODEL}" >&2
  exit 1
fi

if ! ldconfig -p 2>/dev/null | grep -q "libgomp.so.1"; then
  if [[ -z "${LIBGOMP_LIBRARY}" ]]; then
    echo "User-space libgomp.so.1 is missing; rerun the Paddle bootstrap" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="$(
    dirname "${LIBGOMP_LIBRARY}"
  )${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mkdir -p "${INFERENCE_DIR}"
cd "${PADDLEOCR_DIR}"
"${PYTHON_BIN}" tools/export_model.py -c "${CONFIG}" \
  -o "Global.use_gpu=False" \
     "Global.pretrained_model=${MODEL_PREFIX}" \
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

"${PADDLE2ONNX_BIN}" \
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
sha256sum "${MODEL_PREFIX}.pdparams" "${ONNX_MODEL}"
