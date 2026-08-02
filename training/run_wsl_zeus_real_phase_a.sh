#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

exec "${SCRIPT_DIR}/run_wsl_zeus_training.sh" \
  --prepared-dir "${PROJECT_DIR}/training_data/prepared/olimpic_real_v1" \
  --output-dir "${PROJECT_DIR}/training_data/models/zeus-olimpic-real-phase-a-e4-lr3e5-20260727" \
  --epochs 4 \
  --batch-size 16 \
  --learning-rate 0.00003 \
  --precision float32
