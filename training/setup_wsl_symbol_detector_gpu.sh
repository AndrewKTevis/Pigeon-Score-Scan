#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_UV="${HOME}/.cache/scorescan/zeus-bootstrap/bin/uv"
VENV_DIR="${SCORESCAN_SYMBOL_VENV:-${HOME}/.cache/scorescan/symbol-torch260-py311}"

if [[ ! -x "${BOOTSTRAP_UV}" ]]; then
  echo "Run training/setup_wsl_zeus_gpu.sh first (uv bootstrap missing)." >&2
  exit 1
fi

"${BOOTSTRAP_UV}" venv --python "3.11.12" "${VENV_DIR}"
"${BOOTSTRAP_UV}" pip install \
  --python "${VENV_DIR}/bin/python" \
  --index-url "https://download.pytorch.org/whl/cu124" \
  "torch==2.6.0" \
  "torchvision==0.21.0"
"${BOOTSTRAP_UV}" pip install \
  --python "${VENV_DIR}/bin/python" \
  "Pillow==11.1.0" \
  "onnx==1.19.1" \
  "onnxruntime==1.23.2" \
  "pycocotools==2.0.8" \
  "torchmetrics==1.6.2"

"${VENV_DIR}/bin/python" - <<'PY'
import json
import onnx
import onnxruntime
import torch
import torchvision

report = {
    "onnx": onnx.__version__,
    "onnxruntime": onnxruntime.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(report, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit("PyTorch did not expose a GPU")
PY
