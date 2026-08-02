#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 WAIT_PID" >&2
  exit 2
fi

WAIT_PID="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEEPSCORES_OUTPUT="${PROJECT_DIR}/training_data/models/deepscores-symbol-detector-e8-b2-20260727"
OPENSCORE_OUTPUT="${PROJECT_DIR}/training_data/models/openscore-semantic-detector-e6-b2-20260728"
OPENSCORE_MANIFEST="${PROJECT_DIR}/training_data/prepared/openscore_string_quartets_svg_regions_normalized_v2/manifest.json"

while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 20
done

for _attempt in $(seq 1 30); do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      | grep -q '[0-9]'; then
    break
  fi
  sleep 10
done

bash "${SCRIPT_DIR}/run_wsl_symbol_detector_full.sh" "${DEEPSCORES_OUTPUT}"

# CPU rendering and GPU training are intentionally independent.  If rendering
# is still running after the fine-grained detector finishes, wait for its
# atomic completion manifest instead of consuming a partial data tree.
while [[ ! -s "${OPENSCORE_MANIFEST}" ]]; do
  sleep 20
done

exec bash "${SCRIPT_DIR}/run_wsl_openscore_semantic_detector_full.sh" \
  "${OPENSCORE_OUTPUT}"
