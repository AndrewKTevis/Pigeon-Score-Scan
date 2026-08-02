#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${SCORESCAN_SYMBOL_PYTHON:-${HOME}/.cache/scorescan/symbol-torch260-py311/bin/python}"
MODEL_DIR="${PROJECT_DIR}/training_data/models/muse-omr-scan-semantic-detector-v7-final-refit-e2-b2-20260730"
MODEL="${MODEL_DIR}/model.best.pt"
TRAINING_REPORT="${MODEL_DIR}/training_report.json"
FINAL_REFIT_DIR="${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_final_refit_v1"
DEVELOPMENT_DIR="${PROJECT_DIR}/training_data/prepared/muse_omr_scan_regions_stratified_complete_page_overlap_consistent_deduplicated_v8_page_shape_refreeze"
HOLDOUT_DIR="${PROJECT_DIR}/training_data/prepared/muse_omr_scan_holdout_regions_stratified_complete_page_overlap_consistent_deduplicated_v8_page_shape_refreeze"
CALIBRATION_LAYOUT_EVIDENCE="${PROJECT_DIR}/training_data/benchmarks/muse-training-calibration-only-semantic-page-layout-evidence-v2-page-shape-refreeze.json"
HOLDOUT_LAYOUT_EVIDENCE="${PROJECT_DIR}/training_data/benchmarks/muse-holdout-semantic-page-layout-evidence-v2-page-shape-refreeze.json"
OUTPUT_REPORT="${MODEL_DIR}/evaluation.independent-muse-holdout.page-shape-refreeze-v2.json"
export PYTHONUNBUFFERED=1

for path in \
  "${MODEL}" \
  "${TRAINING_REPORT}" \
  "${FINAL_REFIT_DIR}/train.jsonl" \
  "${DEVELOPMENT_DIR}/calibration.jsonl" \
  "${HOLDOUT_DIR}/test.jsonl" \
  "${DEVELOPMENT_DIR}/manifest.json" \
  "${DEVELOPMENT_DIR}/prepare-report.json" \
  "${HOLDOUT_DIR}/manifest.json" \
  "${HOLDOUT_DIR}/prepare-report.json" \
  "${CALIBRATION_LAYOUT_EVIDENCE}" \
  "${HOLDOUT_LAYOUT_EVIDENCE}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Final-refit holdout input is missing: ${path}" >&2
    exit 1
  fi
done

# Fail closed before the only external candidate evaluation. The threshold
# calibration pages must be absent from gradient training, and neither
# development set may overlap the forbidden external holdout.
"${PYTHON_BIN}" - \
  "${TRAINING_REPORT}" \
  "${MODEL}" \
  "${FINAL_REFIT_DIR}/train.jsonl" \
  "${DEVELOPMENT_DIR}/calibration.jsonl" \
  "${HOLDOUT_DIR}/test.jsonl" \
  "${DEVELOPMENT_DIR}/manifest.json" \
  "${DEVELOPMENT_DIR}/prepare-report.json" \
  "${HOLDOUT_DIR}/manifest.json" \
  "${HOLDOUT_DIR}/prepare-report.json" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sources(path: pathlib.Path) -> set[str]:
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            source = str(payload.get("source_key") or "").strip()
            if not source:
                raise ValueError(f"{path}:{line_number}: source_key is missing")
            result.add(source)
    return result


report_path, model_path, train_path, calibration_path, holdout_path = map(
    pathlib.Path,
    sys.argv[1:6],
)
metadata_paths = list(map(pathlib.Path, sys.argv[6:10]))
report = json.loads(report_path.read_text(encoding="utf-8"))
if (
    report.get("completed_epochs") != 2
    or report.get("planned_epochs") != 2
    or report.get("runtime_truncated") is not False
    or report.get("configuration", {}).get("external_acceptance_pending")
    is not True
    or report.get("best_model_sha256") != sha256(model_path)
):
    raise ValueError("final-refit training report is incomplete or stale")

for metadata_path in metadata_paths:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("page_shape_refreeze_contract")
        != "pre-inference-model-independent-source-page-shape-refreeze@1"
        or metadata.get("scan_page_shape_contract")
        != "ordinary-single-page-or-two-page-spread-aspect-ratio@1"
        or metadata.get("maximum_scan_page_aspect_ratio") != 3.0
        or metadata.get("selection_rule_model_independent") is not True
        or metadata.get("model_predictions_observed_for_refreeze") is not False
    ):
        raise ValueError(
            f"page-shape refreeze evidence is incomplete: {metadata_path}"
        )

train = sources(train_path)
calibration = sources(calibration_path)
holdout = sources(holdout_path)
if train & calibration:
    raise ValueError("threshold calibration overlaps final-refit training")
if train & holdout:
    raise ValueError("final-refit training overlaps forbidden holdout")
if calibration & holdout:
    raise ValueError("threshold calibration overlaps forbidden holdout")
print(
    json.dumps(
        {
            "final_refit_train_sources": len(train),
            "threshold_calibration_sources": len(calibration),
            "forbidden_holdout_sources": len(holdout),
            "source_intersections": {
                "train_calibration": 0,
                "train_holdout": 0,
                "calibration_holdout": 0,
            },
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

SCORESCAN_DETECTOR_MODEL_DIR="${MODEL_DIR}" \
SCORESCAN_DETECTOR_MODEL="${MODEL}" \
SCORESCAN_DETECTOR_CATEGORIES="${FINAL_REFIT_DIR}/categories.json" \
SCORESCAN_HOLDOUT_PREPARED_DIR="${HOLDOUT_DIR}" \
SCORESCAN_HOLDOUT_LAYOUT_EVIDENCE="${HOLDOUT_LAYOUT_EVIDENCE}" \
SCORESCAN_CALIBRATION_PREPARED_DIR="${DEVELOPMENT_DIR}" \
SCORESCAN_CALIBRATION_SPLIT="calibration" \
SCORESCAN_CALIBRATION_LAYOUT_EVIDENCE="${CALIBRATION_LAYOUT_EVIDENCE}" \
SCORESCAN_HOLDOUT_OUTPUT_REPORT="${OUTPUT_REPORT}" \
exec "${SCRIPT_DIR}/run_wsl_muse_holdout_evaluation.sh"
