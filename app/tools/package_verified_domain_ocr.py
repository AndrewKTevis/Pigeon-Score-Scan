#!/usr/bin/env python3
"""Package recognition and detection ONNX assets only from passed runtime gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.tools.gate_ocr_independent_holdout import (
    DETECTION_DEPLOY_PARAMETER_NAMES,
    DETECTION_RUNTIME_FIXED_PARAMETERS,
)
from app.tools.merge_ocr_training_labels import sha256_file


RUNTIME_PROFILE = "ppocrv6-imagenet-db-scorescan-calibrated-v2"
RECOGNITION_FLOORS = {
    "acc": 0.998,
    "norm_edit_dis": 0.9995,
}
DETECTION_FLOORS = {
    "precision": 0.995,
    "recall": 0.995,
    "hmean": 0.995,
}
MINIMUM_HOLDOUT_SOURCES = 200
MINIMUM_HOLDOUT_WORDS = 1000
MINIMUM_HOLDOUT_PAGES = 100
MINIMUM_HOLDOUT_IOU = 0.75


def _load_gate(
    path: Path,
    *,
    model: Path,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    if not model.is_file() or model.stat().st_size <= 0:
        raise FileNotFoundError(model)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(gate, dict)
        or gate.get("passed") is not True
        or gate.get("integration_authorized") is not True
        or gate.get("model_sha256") != sha256_file(model)
    ):
        raise ValueError(f"gate does not authorize the supplied model: {path}")
    evaluations = gate.get("evaluations")
    if not isinstance(evaluations, dict):
        raise ValueError(f"gate evaluations are missing: {path}")
    return gate


def _combined_evaluations(
    recognition_gate: dict[str, Any],
    detection_gate: dict[str, Any],
) -> dict[str, dict[str, float]]:
    combined = {}
    for domain in ("registered_scan_test", "clean_render_test"):
        recognition = recognition_gate["evaluations"].get(domain)
        detection = detection_gate["evaluations"].get(domain)
        if not isinstance(recognition, dict) or not isinstance(detection, dict):
            raise ValueError(f"missing {domain} gate metrics")
        metrics: dict[str, float] = {}
        try:
            for name, floor in RECOGNITION_FLOORS.items():
                metrics[name] = float(recognition[name])
                if (
                    not math.isfinite(metrics[name])
                    or not floor <= metrics[name] <= 1.0
                ):
                    raise ValueError(f"{domain} {name} is below packaging floor")
            for name, floor in DETECTION_FLOORS.items():
                metrics[name] = float(detection[name])
                if (
                    not math.isfinite(metrics[name])
                    or not floor <= metrics[name] <= 1.0
                ):
                    raise ValueError(f"{domain} {name} is below packaging floor")
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(f"invalid {domain} gate metrics") from error
        combined[domain] = metrics
    return combined


def _load_independent_holdout_gate(
    path: Path,
    *,
    recognition_model: Path,
    recognition_keys: Path,
    detection_model: Path,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(gate, dict)
        or gate.get("name")
        != "scorescan-independent-scan-ocr-release-gate-v1"
        or gate.get("passed") is not True
        or gate.get("integration_authorized") is not True
        or gate.get("recognition_model_sha256")
        != sha256_file(recognition_model)
        or gate.get("recognition_keys_sha256") != sha256_file(recognition_keys)
        or gate.get("detection_model_sha256") != sha256_file(detection_model)
    ):
        raise ValueError(
            "independent holdout gate does not authorize both supplied models"
        )
    coverage = gate.get("coverage")
    evaluations = gate.get("evaluations")
    if not isinstance(coverage, dict) or not isinstance(evaluations, dict):
        raise ValueError("independent holdout coverage/evaluations are missing")
    if (
        int(coverage.get("sources", -1)) < MINIMUM_HOLDOUT_SOURCES
        or int(coverage.get("words", -1)) < MINIMUM_HOLDOUT_WORDS
        or int(coverage.get("pages", -1)) < MINIMUM_HOLDOUT_PAGES
        or float(coverage.get("minimum_iou", -1)) < MINIMUM_HOLDOUT_IOU
    ):
        raise ValueError("independent holdout coverage is below packaging floor")
    evaluation = evaluations.get("independent_registered_scan_holdout")
    if not isinstance(evaluation, dict):
        raise ValueError("independent holdout metrics are missing")
    recognition = evaluation.get("recognition")
    detection = evaluation.get("detection")
    if not isinstance(recognition, dict) or not isinstance(detection, dict):
        raise ValueError("independent holdout metric families are missing")
    combined: dict[str, float] = {}
    try:
        for name, floor in RECOGNITION_FLOORS.items():
            combined[name] = float(recognition[name])
            if (
                not math.isfinite(combined[name])
                or not floor <= combined[name] <= 1.0
            ):
                raise ValueError(
                    f"independent holdout {name} is below packaging floor"
                )
        for name, floor in DETECTION_FLOORS.items():
            combined[name] = float(detection[name])
            if (
                not math.isfinite(combined[name])
                or not floor <= combined[name] <= 1.0
            ):
                raise ValueError(
                    f"independent holdout {name} is below packaging floor"
                )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("invalid independent holdout metrics") from error
    runtime = gate.get("detection_runtime_parameters")
    expected_names = set(DETECTION_DEPLOY_PARAMETER_NAMES)
    fixed_deploy = {
        name: value
        for name, value in DETECTION_RUNTIME_FIXED_PARAMETERS.items()
        if name in expected_names
    }
    if (
        not isinstance(runtime, dict)
        or set(runtime) != expected_names
        or any(
            runtime.get(name) != expected
            for name, expected in fixed_deploy.items()
        )
    ):
        raise ValueError("independent holdout runtime profile is invalid")
    try:
        threshold = float(runtime["Det.thresh"])
        box_threshold = float(runtime["Det.box_thresh"])
        unclip_ratio = float(runtime["Det.unclip_ratio"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("independent holdout thresholds are invalid") from error
    if (
        not 0 < threshold < 1
        or not 0 < box_threshold < 1
        or not 0.5 <= unclip_ratio <= 3.0
    ):
        raise ValueError("independent holdout thresholds are invalid")
    normalized_runtime = dict(runtime)
    normalized_runtime["Det.thresh"] = threshold
    normalized_runtime["Det.box_thresh"] = box_threshold
    normalized_runtime["Det.unclip_ratio"] = unclip_ratio
    return gate, combined, normalized_runtime


def _file_record(path: Path, filename: str) -> dict[str, Any]:
    return {
        "file": filename,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition-model", type=Path, required=True)
    parser.add_argument("--recognition-keys", type=Path, required=True)
    parser.add_argument("--recognition-gate", type=Path, required=True)
    parser.add_argument("--detection-model", type=Path, required=True)
    parser.add_argument("--detection-gate", type=Path, required=True)
    parser.add_argument("--independent-holdout-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    version = args.model_version.strip()
    if not version or any(character in version for character in "\t\r\n"):
        raise ValueError("model version is empty or unsafe")
    if (
        not args.recognition_keys.is_file()
        or args.recognition_keys.stat().st_size <= 0
    ):
        raise FileNotFoundError(args.recognition_keys)
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to replace existing OCR assets: {args.output_dir}"
        )
    recognition_gate = _load_gate(
        args.recognition_gate,
        model=args.recognition_model,
    )
    detection_gate = _load_gate(
        args.detection_gate,
        model=args.detection_model,
    )
    evaluations = _combined_evaluations(
        recognition_gate,
        detection_gate,
    )
    (
        holdout_gate,
        holdout_metrics,
        detection_runtime_parameters,
    ) = _load_independent_holdout_gate(
        args.independent_holdout_gate,
        recognition_model=args.recognition_model,
        recognition_keys=args.recognition_keys,
        detection_model=args.detection_model,
    )
    evaluations["independent_registered_scan_holdout"] = holdout_metrics

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}-",
            dir=args.output_dir.parent,
        )
    )
    try:
        recognition_name = "scorescan-ppocrv6-rec.onnx"
        detection_name = "scorescan-ppocrv6-det.onnx"
        keys_name = "ppocrv6_dict.txt"
        recognition_destination = staging / recognition_name
        detection_destination = staging / detection_name
        keys_destination = staging / keys_name
        shutil.copy2(args.recognition_model, recognition_destination)
        shutil.copy2(args.detection_model, detection_destination)
        shutil.copy2(args.recognition_keys, keys_destination)
        manifest = {
            "format": 1,
            "model_version": version,
            "integration_authorized": True,
            "detection_runtime_profile": RUNTIME_PROFILE,
            "detection_runtime_parameters": (
                detection_runtime_parameters
            ),
            "evaluations": evaluations,
            "independent_holdout_coverage": holdout_gate.get("coverage"),
            "files": {
                "recognition_model": _file_record(
                    recognition_destination,
                    recognition_name,
                ),
                "recognition_keys": _file_record(
                    keys_destination,
                    keys_name,
                ),
                "detection_model": _file_record(
                    detection_destination,
                    detection_name,
                ),
            },
            "gates": {
                "recognition": {
                    "sha256": sha256_file(args.recognition_gate),
                    "name": recognition_gate.get("name"),
                },
                "detection": {
                    "sha256": sha256_file(args.detection_gate),
                    "name": detection_gate.get("name"),
                },
                "independent_holdout": {
                    "sha256": sha256_file(args.independent_holdout_gate),
                    "name": holdout_gate.get("name"),
                },
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "model_version": version,
                "integration_authorized": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
