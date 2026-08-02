#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
PREPARED_DIR="${PROJECT_DIR}/training_data/prepared/openscore_string_quartets_svg_regions_overlap_consistent_deduplicated_v4"
OUTPUT_DIR="${1:-${PROJECT_DIR}/training_data/models/openscore-semantic-detector-overlap-recovery-v3-e4-b2-20260729}"
INITIAL_MODEL="${2:-${PROJECT_DIR}/training_data/models/openscore-semantic-detector-e6-b2-20260728/model.best.pt}"
INITIAL_CATEGORIES="${PREPARED_DIR}/categories.json"
TRAIN_DEVICE="${SCORESCAN_TRAIN_DEVICE:-cuda}"
CPU_THREADS="${SCORESCAN_CPU_THREADS:-0}"
# Two workers keep the native verified image cache ahead of the GPU without
# duplicating the large Python target graph four ways. The logical batch remains
# two. Sparse batches use one two-image forward for throughput, while batches
# above the audited target-density limit fall back to one-image forwards to
# avoid dense-score unified-memory spill on 8 GiB cards.
LOADER_WORKERS="${SCORESCAN_LOADER_WORKERS:-2}"
ADAPTIVE_FULL_BATCH_OBJECT_LIMIT="${SCORESCAN_ADAPTIVE_FULL_BATCH_OBJECT_LIMIT:-80}"
ADAPTIVE_FULL_BATCH_MIN_FREE_MIB="${SCORESCAN_ADAPTIVE_FULL_BATCH_MIN_FREE_MIB:-4608}"
IMAGE_CACHE_DIR="${SCORESCAN_DETECTOR_IMAGE_CACHE:-${HOME}/.cache/scorescan/detector-images/openscore-quartet-semantic-v1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

for required in \
  "${PREPARED_DIR}/manifest.json" \
  "${PREPARED_DIR}/prepare-report.json" \
  "${PREPARED_DIR}/train.jsonl" \
  "${PREPARED_DIR}/test.jsonl" \
  "${INITIAL_MODEL}" \
  "${INITIAL_CATEGORIES}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Overlap-consistent quartet recovery input is incomplete: ${required}" >&2
    exit 1
  fi
done

if [[ -s "${OUTPUT_DIR}/training_report.json" ]]; then
  "${PYTHON_BIN}" - "${OUTPUT_DIR}/training_report.json" "${PREPARED_DIR}/manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
report = json.loads(report_path.read_text(encoding="utf-8"))
expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
actual_hash = report.get("data", {}).get("prepared_manifest_sha256")
if actual_hash != expected_hash:
    raise SystemExit(
        "completed overlap recovery used a different prepared manifest: "
        f"{actual_hash!r} != {expected_hash!r}"
    )
if report.get("acceptance", {}).get("passed") is not True:
    raise SystemExit("completed overlap recovery did not pass its configured gates")
required_metrics = {
    "best_map_50": 0.75,
    "best_map_75": 0.70,
    "best_priority_mark_map": 0.55,
}
for field, floor in required_metrics.items():
    actual = float(report.get(field, -1.0))
    if actual < floor:
        raise SystemExit(
            f"completed overlap recovery {field} is below its handoff floor: "
            f"{actual} < {floor}"
        )
PY
  exit 0
fi

# Recovery was planned as a four-epoch continuation, but its only role is to
# initialize the stricter Lieder and registered-scan stages. Two complete
# full-corpus epochs plus an independent evaluation retain an audited handoff
# while skipping the low-yield third and fourth synthetic passes.
exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/app/tools/train_deepscores_symbol_detector.py" \
  --prepared-dir "${PREPARED_DIR}" \
  --images-dir "${PROJECT_DIR}" \
  --image-cache-dir "${IMAGE_CACHE_DIR}" \
  --populate-image-cache \
  --output-dir "${OUTPUT_DIR}" \
  --initial-model "${INITIAL_MODEL}" \
  --initial-categories "${INITIAL_CATEGORIES}" \
  --epochs 4 \
  --batch-size 2 \
  --microbatch-size 1 \
  --adaptive-full-batch-object-limit "${ADAPTIVE_FULL_BATCH_OBJECT_LIMIT}" \
  --adaptive-full-batch-min-free-mib "${ADAPTIVE_FULL_BATCH_MIN_FREE_MIB}" \
  --evaluation-batch-size 8 \
  --accumulate 2 \
  --workers "${LOADER_WORKERS}" \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --eval-every 2 \
  --minimum-best-map-50 0.75 \
  --minimum-best-map-75 0.70 \
  --minimum-best-priority-map 0.55 \
  --rare-class-sampling-power 0.35 \
  --rare-class-max-repeat 4 \
  --runtime-stop-after-epoch 2 \
  --runtime-stop-reason "superseded_by_bounded_lieder_replay_and_registered_scan_finetuning" \
  --device "${TRAIN_DEVICE}" \
  --cpu-threads "${CPU_THREADS}" \
  --allow-resume-worker-change \
  --resume
