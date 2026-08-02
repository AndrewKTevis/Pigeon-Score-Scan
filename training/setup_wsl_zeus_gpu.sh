#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${SCORESCAN_ZEUS_VENV:-${HOME}/.cache/scorescan/zeus-tf215-py310}"
BOOTSTRAP_DIR="${HOME}/.cache/scorescan/zeus-bootstrap"
PYTHON_BIN="${SCORESCAN_ZEUS_PYTHON:-python3}"

"${PYTHON_BIN}" -m venv "${BOOTSTRAP_DIR}"
"${BOOTSTRAP_DIR}/bin/python" -m pip install --upgrade \
  "pip==25.1.1" \
  "setuptools==80.9.0" \
  "wheel==0.45.1"
"${BOOTSTRAP_DIR}/bin/python" -m pip install "uv==0.11.32"
"${BOOTSTRAP_DIR}/bin/uv" venv --python "3.10.19" "${VENV_DIR}"

"${BOOTSTRAP_DIR}/bin/uv" pip install --python "${VENV_DIR}/bin/python" \
  "tensorflow[and-cuda]==2.15.1" \
  "Levenshtein==0.27.1" \
  "zss==1.2.0" \
  "typing_extensions==4.12.2"

"${VENV_DIR}/bin/python" - <<'PY'
import json
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
report = {
    "tensorflow": tf.__version__,
    "cuda_built": tf.test.is_built_with_cuda(),
    "gpus": [gpu.name for gpu in gpus],
}
print(json.dumps(report, sort_keys=True))
if not gpus:
    raise SystemExit("TensorFlow did not expose a GPU; refusing to mark the runtime ready")
PY
