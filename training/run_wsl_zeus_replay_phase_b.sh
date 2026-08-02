#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PUBLISHED_BASE="${PROJECT_DIR}/training_data/external/corpora/olimpic_zeus_model/zeus-olimpic-1.0-2024-02-12.model"

if [[ ! -f "${PUBLISHED_BASE}/weights.h5" ]]; then
  echo "Published OLiMPiC baseline is incomplete: ${PUBLISHED_BASE}" >&2
  exit 1
fi

exec "${SCRIPT_DIR}/run_wsl_zeus_training.sh" \
  --base-model "${PUBLISHED_BASE}" \
  --prepared-dir "${PROJECT_DIR}/training_data/prepared/olimpic_real_plus_replay_v2" \
  --output-dir "${PROJECT_DIR}/training_data/models/zeus-olimpic-real-replay-e2-b8-lr1e5-20260727" \
  --epochs 2 \
  --batch-size 8 \
  --learning-rate 0.00001 \
  --minimum-ser-improvement 0.10 \
  --precision float32
