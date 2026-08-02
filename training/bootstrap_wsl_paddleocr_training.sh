#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_UV="${HOME}/.cache/scorescan/zeus-bootstrap/bin/uv"
VENV_DIR="${SCORESCAN_PADDLEOCR_VENV:-${HOME}/.cache/scorescan/paddleocr-3.2-cu126-py311}"
PADDLEOCR_DIR="${SCORESCAN_PADDLEOCR_DIR:-/workspace/pigeon-score-scan/training_data/external/corpora/paddleocr_2661c7c0}"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
LIBGOMP_ROOT="${HOME}/.cache/scorescan/syslibs/libgomp"

if [[ ! -x "${BOOTSTRAP_UV}" ]]; then
  echo "uv bootstrap is missing: ${BOOTSTRAP_UV}" >&2
  exit 1
fi
if [[ ! -f "${PADDLEOCR_DIR}/tools/train.py" ]]; then
  echo "Pinned PaddleOCR source is missing: ${PADDLEOCR_DIR}" >&2
  exit 1
fi

if ! ldconfig -p 2>/dev/null | grep -q "libgomp.so.1"; then
  mkdir -p "${LIBGOMP_ROOT}"
  if ! find "${LIBGOMP_ROOT}/root" -name "libgomp.so.1" -print -quit \
    2>/dev/null | grep -q .; then
    (
      cd "${LIBGOMP_ROOT}"
      apt-get download libgomp1
      dpkg-deb -x libgomp1_*.deb root
    )
  fi
  LIBGOMP_LIBRARY="$(
    find "${LIBGOMP_ROOT}/root" -name "libgomp.so.1" -print -quit
  )"
  if [[ -z "${LIBGOMP_LIBRARY}" ]]; then
    echo "Unable to provide the required libgomp.so.1 runtime" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="$(
    dirname "${LIBGOMP_LIBRARY}"
  )${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${BOOTSTRAP_UV}" venv --python "3.11.12" "${VENV_DIR}"
fi
"${BOOTSTRAP_UV}" pip install \
  --python "${VENV_DIR}/bin/python" \
  --index-url "${PADDLE_INDEX}" \
  "paddlepaddle-gpu==3.2.0"
"${BOOTSTRAP_UV}" pip install \
  --python "${VENV_DIR}/bin/python" \
  "setuptools==80.10.2" \
  "paddle2onnx==2.1.0" \
  -r "${PADDLEOCR_DIR}/requirements.txt"

"${VENV_DIR}/bin/python" - <<'PY'
import json
import paddle

compiled = bool(paddle.device.is_compiled_with_cuda())
report = {
    "paddle": paddle.__version__,
    "compiled_with_cuda": compiled,
    "device": paddle.device.get_device(),
    "cuda_device_count": paddle.device.cuda.device_count() if compiled else 0,
}
print(json.dumps(report, sort_keys=True))
if paddle.__version__ != "3.2.0":
    raise SystemExit("unexpected PaddlePaddle version")
if not compiled:
    raise SystemExit("PaddlePaddle was not compiled with CUDA")
PY
"${BOOTSTRAP_UV}" pip check --python "${VENV_DIR}/bin/python"
