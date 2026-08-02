#!/usr/bin/env bash
set -euo pipefail

# Low-impact resumable training mode used while the GPU is reserved for the user.
# nice lowers scheduler priority; thread caps keep the detector from monopolising
# the desktop CPU. The statistical run configuration remains identical.
PROJECT_DIR="/workspace/pigeon-score-scan"
PYTHON="/workspace/.cache/scorescan/symbol-torch260-py311/bin/python"
OUTPUT_DIR="${1:-${PROJECT_DIR}/training_data/models/openscore-semantic-detector-e6-b2-20260728}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

exec taskset -c 0-3 nice -n 12 "${PYTHON}" \
  "${PROJECT_DIR}/app/tools/train_deepscores_symbol_detector.py" \
  --prepared-dir \
  "${PROJECT_DIR}/training_data/prepared/openscore_string_quartets_svg_regions_normalized_v2" \
  --images-dir "${PROJECT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --initial-backbone-model \
  "${PROJECT_DIR}/training_data/models/deepscores-symbol-detector-e8-b2-20260727/model.best.pt" \
  --epochs 6 \
  --batch-size 2 \
  --accumulate 2 \
  --workers 2 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --eval-every 1 \
  --rare-class-sampling-power 0.35 \
  --rare-class-max-repeat 4 \
  --checkpoint-every-steps 500 \
  --device cpu \
  --cpu-threads 2 \
  --resume
