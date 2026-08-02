#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 PID [OUTPUT_DIR]" >&2
  exit 2
fi

WAIT_PID="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${2:-${PROJECT_DIR}/training_data/models/deepscores-symbol-detector-e8-b2-20260727}"

while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 20
done

# CUDA contexts can linger briefly after TensorFlow exits.  Wait for the
# process table to clear rather than risking two frameworks contending for the
# 8 GiB device.  An empty query is success.
for _attempt in $(seq 1 30); do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      | grep -q '[0-9]'; then
    break
  fi
  sleep 10
done

exec bash "${SCRIPT_DIR}/run_wsl_symbol_detector_full.sh" "${OUTPUT_DIR}"
