#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_ZEUS_PYTHON:-${HOME}/.cache/scorescan/zeus-tf215-py310/bin/python}"

unset TF_USE_LEGACY_KERAS
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/finetune_zeus_olimpic.py" \
  --upstream-dir "${PROJECT_DIR}/development_reports/source_audits/external-dataset-audit-20260727/olimpic" \
  --base-model "${PROJECT_DIR}/training_data/external/corpora/olimpic_zeus_model/zeus-olimpic-1.0-2024-02-12.model" \
  --prepared-dir "${PROJECT_DIR}/training_data/prepared/olimpic_real_v1" \
  "$@"
