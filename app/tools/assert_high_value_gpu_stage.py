#!/usr/bin/env python3
"""Fail closed unless a high-value GPU stage produced its accepted artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_SRC = PROJECT_ROOT / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso


STAGES = (
    "semantic_detector_and_holdout",
    "ocr_recognition",
    "ocr_detection",
    "ocr_detection_exhaustive",
    "semantic_family_priority",
)


def _required_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    resolved = _required_file(path)
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {resolved}")
    return payload


def _file_record(path: Path) -> dict[str, Any]:
    resolved = _required_file(path)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _assert_model_gate(
    gate_path: Path,
    model_path: Path,
    *,
    expected_name: str,
) -> list[dict[str, Any]]:
    gate = _load_json(gate_path)
    model_record = _file_record(model_path)
    if (
        gate.get("name") != expected_name
        or gate.get("passed") is not True
        or gate.get("integration_authorized") is not True
        or gate.get("model_sha256") != model_record["sha256"]
    ):
        raise ValueError(f"model release gate is not accepted: {gate_path}")
    return [_file_record(gate_path), model_record]


def _semantic(project: Path) -> list[dict[str, Any]]:
    model_dir = (
        project
        / "training_data/models/"
        "muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730"
    )
    candidate_dir = (
        project
        / "training_data/release_candidates/"
        "semantic-detector-muse-v4-complete-page-e12-20260730"
    )
    candidate_path = candidate_dir / "release-candidate.json"
    candidate = _load_json(candidate_path)
    model = model_dir / "model.best.pt"
    model_record = _file_record(model)
    onnx = Path(str(candidate.get("onnx", "")))
    onnx_record = _file_record(onnx)
    training = _load_json(model_dir / "training_report.json")
    holdout = _load_json(model_dir / "evaluation.independent-muse-holdout.json")
    cpu_parity_path = Path(str(candidate.get("cpu_parity", "")))
    gpu_parity_path = Path(str(candidate.get("gpu_parity", "")))
    cpu_parity = _load_json(cpu_parity_path)
    gpu_parity = _load_json(gpu_parity_path)
    evaluation_resources = Path(
        str(candidate.get("isolated_product_evaluation_resources", ""))
    )
    semantic_manifest = evaluation_resources / "semantic_detector.json"
    semantic_manifest_record = _file_record(semantic_manifest)
    if (
        candidate.get("format") != 2
        or candidate.get("canonical_resources_authorized") is not False
        or candidate.get("physical_scan_release_evidence") is not False
        or candidate.get("isolated_product_evaluation_resources_verified")
        is not True
        or candidate.get("source_model_sha256") != model_record["sha256"]
        or candidate.get("onnx_sha256") != onnx_record["sha256"]
        or candidate.get("semantic_manifest_sha256")
        != semantic_manifest_record["sha256"]
        or training.get("acceptance", {}).get("passed") is not True
        or holdout.get("acceptance", {}).get("passed") is not True
        or cpu_parity.get("passed") is not True
        or gpu_parity.get("passed") is not True
    ):
        raise ValueError("semantic detector candidate is not accepted")
    return [
        _file_record(candidate_path),
        model_record,
        onnx_record,
        _file_record(model_dir / "training_report.json"),
        _file_record(model_dir / "evaluation.independent-muse-holdout.json"),
        _file_record(cpu_parity_path),
        _file_record(gpu_parity_path),
        semantic_manifest_record,
    ]


def _recognition(project: Path) -> list[dict[str, Any]]:
    model_dir = (
        project
        / "training_data/models/"
        "ppocrv6-scorescan-rec-stratified-e18-b8-20260729"
    )
    return _assert_model_gate(
        model_dir / "onnx-release-gate.json",
        model_dir / "scorescan-ppocrv6-rec.onnx",
        expected_name="scorescan-ppocrv6-domain-release-gate-v1",
    )


def _assert_ocr_resource_manifest(
    project: Path,
    *,
    recognition_gate: Path,
    detection_gate: Path,
    holdout_gate: Path,
) -> list[dict[str, Any]]:
    resource_dir = project / "app/src/scorescan/resources/ocr"
    manifest_path = resource_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("format") != 1
        or manifest.get("integration_authorized") is not True
    ):
        raise ValueError("packaged OCR resource manifest is not authorized")
    records = [_file_record(manifest_path)]
    files = manifest.get("files")
    gates = manifest.get("gates")
    if not isinstance(files, dict) or not isinstance(gates, dict):
        raise ValueError("packaged OCR provenance is incomplete")
    for entry in files.values():
        if not isinstance(entry, dict):
            raise ValueError("packaged OCR file record is invalid")
        artifact = resource_dir / str(entry.get("file", ""))
        record = _file_record(artifact)
        if (
            entry.get("sha256") != record["sha256"]
            or int(entry.get("bytes", -1)) != record["bytes"]
        ):
            raise ValueError(f"packaged OCR file hash mismatch: {artifact}")
        records.append(record)
    for name, gate_path in (
        ("recognition", recognition_gate),
        ("detection", detection_gate),
        ("independent_holdout", holdout_gate),
    ):
        entry = gates.get(name)
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != sha256_file(_required_file(gate_path))
        ):
            raise ValueError(f"packaged OCR gate hash mismatch: {name}")
        records.append(_file_record(gate_path))
    return records


def _detection(project: Path) -> list[dict[str, Any]]:
    recognition_dir = (
        project
        / "training_data/models/"
        "ppocrv6-scorescan-rec-stratified-e18-b8-20260729"
    )
    model_dir = (
        project
        / "training_data/models/"
        "ppocrv6-scorescan-det-stratified-e36-b2-20260729"
    )
    recognition_model = recognition_dir / "scorescan-ppocrv6-rec.onnx"
    detection_model = model_dir / "scorescan-ppocrv6-det.onnx"
    detection_gate = model_dir / "onnx-release-gate.json"
    holdout_gate = model_dir / "onnx-independent-holdout-gate.json"
    records = _assert_model_gate(
        detection_gate,
        detection_model,
        expected_name="scorescan-ppocrv6-domain-detection-release-gate-v1",
    )
    holdout = _load_json(holdout_gate)
    recognition_record = _file_record(recognition_model)
    detection_record = _file_record(detection_model)
    if (
        holdout.get("name")
        != "scorescan-independent-scan-ocr-release-gate-v1"
        or holdout.get("passed") is not True
        or holdout.get("integration_authorized") is not True
        or holdout.get("recognition_model_sha256")
        != recognition_record["sha256"]
        or holdout.get("detection_model_sha256") != detection_record["sha256"]
    ):
        raise ValueError("independent OCR holdout is not accepted")
    records.extend(
        [
            _file_record(holdout_gate),
            recognition_record,
            *_assert_ocr_resource_manifest(
                project,
                recognition_gate=recognition_dir / "onnx-release-gate.json",
                detection_gate=detection_gate,
                holdout_gate=holdout_gate,
            ),
        ]
    )
    return records


def _exhaustive(project: Path) -> list[dict[str, Any]]:
    model_dir = (
        project
        / "training_data/models/"
        "ppocrv6-scorescan-det-exhaustive-stratified-e24-b2-20260729"
    )
    return _assert_model_gate(
        model_dir / "onnx-release-gate.iou075.json",
        model_dir / "scorescan-ppocrv6-det.onnx",
        expected_name="scorescan-ppocrv6-domain-detection-release-gate-v1",
    )


def _zeus(project: Path) -> list[dict[str, Any]]:
    model_dir = (
        project
        / "training_data/models/"
        "zeus-olimpic-real-family-priority-source-doc-safe-mp-e6-b16-lr5e6-"
        "20260729"
    )
    evaluation_dir = (
        project
        / "training_data/models/"
        "zeus-olimpic-real-family-priority-source-doc-safe-upstream-test-"
        "20260729"
    )
    training_report_path = model_dir / "training_report.json"
    evaluation_report_path = evaluation_dir / "training_report.json"
    completion_path = (
        project
        / "training_data/logs/"
        "zeus-family-priority-source-doc-safe-queue.completion.json"
    )
    calibration_path = model_dir / "family-priority-calibration-gate.json"
    upstream_path = evaluation_dir / "upstream-candidate-test-gate.json"
    completion = _load_json(completion_path)
    calibration = _load_json(calibration_path)
    upstream = _load_json(upstream_path)
    if (
        completion.get("state") != "completed"
        or completion.get("calibration_passed") is not True
        or completion.get("upstream_candidate_test_opened") is not True
        or completion.get("upstream_candidate_passed") is not True
        or calibration.get("passed") is not True
        or calibration.get("candidate_test_evaluation_authorized") is not True
        or upstream.get("passed") is not True
        or upstream.get("observer_candidate_authorized") is not True
    ):
        raise ValueError("Zeus family-priority candidate is not accepted")
    training_record = _file_record(training_report_path)
    evaluation_record = _file_record(evaluation_report_path)
    calibration_input = calibration.get("input", {}).get("training_report", {})
    upstream_input = upstream.get("input", {}).get("evaluation_report", {})
    if (
        calibration_input.get("sha256") != training_record["sha256"]
        or int(calibration_input.get("bytes", -1)) != training_record["bytes"]
        or upstream_input.get("sha256") != evaluation_record["sha256"]
        or int(upstream_input.get("bytes", -1)) != evaluation_record["bytes"]
    ):
        raise ValueError("Zeus family-priority candidate is not accepted")
    return [
        _file_record(completion_path),
        _file_record(calibration_path),
        _file_record(upstream_path),
        training_record,
        evaluation_record,
    ]


def verify_stage(project_root: Path, stage: str) -> dict[str, Any]:
    project = project_root.resolve()
    if stage not in STAGES:
        raise ValueError(f"unsupported GPU stage: {stage}")
    artifacts = {
        "semantic_detector_and_holdout": _semantic,
        "ocr_recognition": _recognition,
        "ocr_detection": _detection,
        "ocr_detection_exhaustive": _exhaustive,
        "semantic_family_priority": _zeus,
    }[stage](project)
    return {
        "format": 1,
        "name": "scorescan-high-value-gpu-stage-gate-v1",
        "verified_at_utc": utc_now_iso(),
        "stage": stage,
        "passed": True,
        "artifacts": artifacts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_stage(args.project_root, args.stage)
    except Exception as exc:
        atomic_write_json(
            args.output_report,
            {
                "format": 1,
                "name": "scorescan-high-value-gpu-stage-gate-v1",
                "verified_at_utc": utc_now_iso(),
                "stage": args.stage,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    atomic_write_json(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
