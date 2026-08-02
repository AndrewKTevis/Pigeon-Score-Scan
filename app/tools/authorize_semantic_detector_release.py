#!/usr/bin/env python3
from __future__ import annotations

"""Package a semantic ONNX detector only after every production gate passes."""

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from app.tools.muse_omr_contract import TRAINING_REGION_ROLE
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
)
from scorescan.semantic_detector_contract import (
    CALIBRATED_OPERATING_POINT_SELECTION_METHOD,
    FIXED_RARE_CLASS_OPERATING_POINT_THRESHOLD,
    FIXED_RARE_CLASS_SELECTION_METHOD,
    HIGH_RECALL_MARK_CLASSES,
    MINIMUM_INDEPENDENT_WORKS,
    MINIMUM_HIGH_RECALL_MARK_RECALL,
    MINIMUM_OPERATING_POINT_PRECISION,
    MINIMUM_OPERATING_POINT_RECALL,
    MINIMUM_OPERATING_POINT_TRUE_POSITIVES,
    SEMANTIC_DETECTOR_INPUT_SIZE,
    SEMANTIC_DETECTOR_MAXIMUM_SCALE,
    SEMANTIC_DETECTOR_MAXIMUM_TILES,
    SEMANTIC_DETECTOR_MINIMUM_SCALE,
    SEMANTIC_DETECTOR_TARGET_STAFF_SPACING,
    SEMANTIC_DETECTOR_TILE_OVERLAP,
    SEMANTIC_PAGE_NMS_IOU,
    SEMANTIC_DETECTOR_MANIFEST_NAME,
    SUPPORTED_RUNTIME_CLASSES,
    TILE_FRAGMENT_FUSION_VERSION,
)
from scorescan.semantic_detector import (
    load_semantic_detector_assets,
)
from scorescan.semantic_tile_fusion import PAGE_LAYOUT_EVIDENCE_VERSION
from scorescan.util import atomic_write_json, sha256_file
from app.tools.evaluate_semantic_detector_holdout import (
    DENSE_MAP_VERSION,
    PAGE_STITCHING_VERSION,
    RUNTIME_PAGE_TILING_VERSION,
)
from app.tools.train_deepscores_symbol_detector import (
    DETECTOR_NMS_IOU,
    PRIORITY_SELECTION_PROTOCOL,
    detector_selection_evidence_failures,
    detector_model_contract,
)

MODEL_NAME = "semantic_detector.onnx"
CATEGORIES_NAME = "semantic_detector_categories.json"
HOLDOUT_NAME = "semantic_detector_holdout.json"
PARITY_NAME = "semantic_detector_onnx_parity.json"
GPU_PARITY_NAME = "semantic_detector_onnx_gpu_parity.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--holdout-report", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--gpu-parity-report", type=Path, required=True)
    parser.add_argument("--output-resources", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    return parser


def _read_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report is not a JSON object: {path}")
    return payload


def _copy_exclusive(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _named_class_support(
    metrics: dict[str, Any],
    raw_counts: Any,
    class_name_by_label: dict[int, str],
) -> dict[str, int] | None:
    named = metrics.get("map_per_class_named")
    labels = metrics.get("classes")
    if (
        not isinstance(named, dict)
        or not isinstance(labels, list)
        or len(named) != len(labels)
        or not isinstance(raw_counts, dict)
    ):
        return None
    try:
        counts = {int(label): int(count) for label, count in raw_counts.items()}
        expected_names = [
            class_name_by_label[int(label)] for label in labels
        ]
        if expected_names != list(named):
            return None
        return {
            name: counts.get(int(label), 0)
            for label, name in zip(labels, expected_names)
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_version = str(args.model_version).strip()
    if not model_version:
        raise ValueError("model version must not be empty")
    for path in (
        args.source_model,
        args.onnx,
        args.categories,
        args.training_report,
        args.holdout_report,
        args.parity_report,
        args.gpu_parity_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_resources.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = (
        output / MODEL_NAME,
        output / CATEGORIES_NAME,
        output / HOLDOUT_NAME,
        output / PARITY_NAME,
        output / GPU_PARITY_NAME,
        output / SEMANTIC_DETECTOR_MANIFEST_NAME,
    )
    if any(path.exists() for path in targets):
        raise FileExistsError(
            "refusing to overwrite an existing authorized semantic detector"
        )

    source_model_hash = sha256_file(args.source_model)
    categories_hash = sha256_file(args.categories)
    onnx_hash = sha256_file(args.onnx)
    categories = _read_report(args.categories)
    training = _read_report(args.training_report)
    holdout = _read_report(args.holdout_report)
    parity = _read_report(args.parity_report)
    gpu_parity = _read_report(args.gpu_parity_report)

    failures: list[str] = []
    try:
        class_name_by_label = {
            int(item["label"]): str(item["name"])
            for item in categories["classes"]
        }
        if (
            len(class_name_by_label) != len(categories["classes"])
            or not class_name_by_label
        ):
            raise ValueError("duplicate or empty detector categories")
    except (KeyError, TypeError, ValueError, OverflowError):
        class_name_by_label = {}
        failures.append("categories_class_mapping")
    if training.get("acceptance", {}).get("passed") is not True:
        failures.append("training_acceptance")
    if str(training.get("best_model_sha256") or "") != source_model_hash:
        failures.append("training_model_hash")
    if holdout.get("acceptance", {}).get("passed") is not True:
        failures.append("holdout_acceptance")
    if int(holdout.get("format", 0)) < 3:
        failures.append("holdout_frozen_threshold_contract")
    if holdout.get("detection_metric_protocol") != DENSE_MAP_VERSION:
        failures.append("holdout_dense_metric_protocol")
    if (
        training.get("priority_selection_protocol")
        != PRIORITY_SELECTION_PROTOCOL
    ):
        failures.append("training_priority_selection_protocol")
    if (
        holdout.get("priority_selection_protocol")
        != PRIORITY_SELECTION_PROTOCOL
    ):
        failures.append("holdout_priority_selection_protocol")
    training_best_epoch = int(training.get("best_epoch", -1))
    training_best_test = next(
        (
            record.get("test")
            for record in training.get("metrics", {}).get("epochs", [])
            if isinstance(record, dict)
            and int(record.get("epoch", -1)) == training_best_epoch
        ),
        None,
    )
    training_minimum_support = int(
        training.get("configuration", {}).get(
            "minimum_required_class_test_objects",
            -1,
        )
    )
    training_class_support = (
        _named_class_support(
            training_best_test,
            training.get("data", {}).get("test_class_counts"),
            class_name_by_label,
        )
        if isinstance(training_best_test, dict)
        else None
    )
    if training_minimum_support <= 0 or training_class_support is None:
        failures.append("training_selection_evidence")
    else:
        failures.extend(
            "training_selection_evidence:" + failure
            for failure in detector_selection_evidence_failures(
                training_best_test,
                class_support=training_class_support,
                minimum_support=training_minimum_support,
            )
        )
    holdout_metrics = holdout.get("metrics")
    holdout_minimum_support = int(
        holdout.get("acceptance", {}).get(
            "minimum_required_class_test_objects",
            -1,
        )
    )
    holdout_class_support = (
        _named_class_support(
            holdout_metrics,
            holdout.get("test_class_counts"),
            class_name_by_label,
        )
        if isinstance(holdout_metrics, dict)
        else None
    )
    if holdout_minimum_support <= 0 or holdout_class_support is None:
        failures.append("holdout_selection_evidence")
    else:
        failures.extend(
            "holdout_selection_evidence:" + failure
            for failure in detector_selection_evidence_failures(
                holdout_metrics,
                class_support=holdout_class_support,
                minimum_support=holdout_minimum_support,
            )
        )
    expected_model_contract = detector_model_contract()
    if training.get("model_contract") != expected_model_contract:
        failures.append("training_model_contract")
    if holdout.get("model_contract") != expected_model_contract:
        failures.append("holdout_model_contract")
    holdout_stitching = holdout.get("page_stitching")
    if (
        not isinstance(holdout_stitching, dict)
        or holdout_stitching.get("version") != PAGE_STITCHING_VERSION
        or holdout_stitching.get("tile_fragment_fusion_version")
        != TILE_FRAGMENT_FUSION_VERSION
        or holdout_stitching.get("runtime_layout_assignment") is not True
        or holdout_stitching.get("runtime_page_retiling_version")
        != RUNTIME_PAGE_TILING_VERSION
        or int(holdout_stitching.get("pages", 0)) <= 0
        or int(holdout_stitching.get("layout_pages", 0))
        != int(holdout_stitching.get("pages", 0))
        or int(holdout_stitching.get("unique_source_targets", 0)) <= 0
        or int(holdout_stitching.get("tile_target_instances", 0))
        < int(holdout_stitching.get("unique_source_targets", 0))
    ):
        failures.append("holdout_page_stitching")
    holdout_runtime_tiling = holdout.get("runtime_page_tiling")
    if (
        not isinstance(holdout_runtime_tiling, dict)
        or holdout_runtime_tiling.get("version")
        != RUNTIME_PAGE_TILING_VERSION
        or int(holdout_runtime_tiling.get("input_size", 0))
        != SEMANTIC_DETECTOR_INPUT_SIZE
        or float(holdout_runtime_tiling.get("target_staff_spacing", 0.0))
        != SEMANTIC_DETECTOR_TARGET_STAFF_SPACING
        or int(holdout_runtime_tiling.get("overlap", -1))
        != SEMANTIC_DETECTOR_TILE_OVERLAP
        or holdout_runtime_tiling.get(
            "oversized_fragment_visibility_version"
        )
        != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        or float(holdout_runtime_tiling.get("minimum_scale", 0.0))
        != SEMANTIC_DETECTOR_MINIMUM_SCALE
        or float(holdout_runtime_tiling.get("maximum_scale", 0.0))
        != SEMANTIC_DETECTOR_MAXIMUM_SCALE
        or int(holdout_runtime_tiling.get("maximum_tiles", 0))
        != SEMANTIC_DETECTOR_MAXIMUM_TILES
        or int(holdout_runtime_tiling.get("pages", 0))
        != int((holdout_stitching or {}).get("pages", 0))
        or int(holdout_runtime_tiling.get("tiles", 0))
        != int((holdout_stitching or {}).get("tiles", 0))
    ):
        failures.append("holdout_runtime_page_tiling")
    holdout_layout_evidence = holdout.get("page_layout_evidence")
    if (
        not isinstance(holdout_layout_evidence, dict)
        or holdout_layout_evidence.get("version")
        != PAGE_LAYOUT_EVIDENCE_VERSION
        or len(str(holdout_layout_evidence.get("sha256") or "")) != 64
        or int(holdout_layout_evidence.get("pages", 0))
        != int((holdout_stitching or {}).get("pages", 0))
    ):
        failures.append("holdout_page_layout_evidence")
    if holdout.get("integration_authorized") is not False:
        failures.append("holdout_integration_contract")
    if str(holdout.get("model_sha256") or "") != source_model_hash:
        failures.append("holdout_model_hash")
    if str(holdout.get("model_categories_sha256") or "") != categories_hash:
        failures.append("holdout_categories_hash")
    independent_works = int(holdout.get("independent_works", 0))
    if independent_works < MINIMUM_INDEPENDENT_WORKS:
        failures.append("holdout_independent_works")
    calibration = holdout.get("operating_point_calibration")
    calibration_minimum_true_positives = 0
    if (
        not isinstance(calibration, dict)
        or calibration.get("selection_dataset_role") != TRAINING_REGION_ROLE
        or calibration.get("selected_splits") != ["calibration"]
        or calibration.get("holdout_reused_for_selection") is not False
        or calibration.get("source_overlap_with_holdout") != []
        or not isinstance(calibration.get("selected_points"), dict)
        or not isinstance(calibration.get("page_stitching"), dict)
        or calibration["page_stitching"].get("version")
        != PAGE_STITCHING_VERSION
        or calibration["page_stitching"].get(
            "tile_fragment_fusion_version"
        )
        != TILE_FRAGMENT_FUSION_VERSION
        or calibration["page_stitching"].get("runtime_layout_assignment")
        is not True
        or calibration["page_stitching"].get(
            "runtime_page_retiling_version"
        )
        != RUNTIME_PAGE_TILING_VERSION
        or int(calibration["page_stitching"].get("pages", 0)) <= 0
        or int(calibration["page_stitching"].get("layout_pages", 0))
        != int(calibration["page_stitching"].get("pages", 0))
        or not isinstance(calibration.get("page_layout_evidence"), dict)
        or calibration["page_layout_evidence"].get("version")
        != PAGE_LAYOUT_EVIDENCE_VERSION
        or len(
            str(
                calibration["page_layout_evidence"].get("sha256")
                or ""
            )
        )
        != 64
        or int(calibration["page_layout_evidence"].get("pages", 0))
        != int(calibration["page_stitching"].get("pages", 0))
        or not isinstance(calibration.get("runtime_page_tiling"), dict)
        or calibration["runtime_page_tiling"].get("version")
        != RUNTIME_PAGE_TILING_VERSION
        or int(calibration["runtime_page_tiling"].get("input_size", 0))
        != SEMANTIC_DETECTOR_INPUT_SIZE
        or float(
            calibration["runtime_page_tiling"].get(
                "target_staff_spacing",
                0.0,
            )
        )
        != SEMANTIC_DETECTOR_TARGET_STAFF_SPACING
        or int(calibration["runtime_page_tiling"].get("overlap", -1))
        != SEMANTIC_DETECTOR_TILE_OVERLAP
        or calibration["runtime_page_tiling"].get(
            "oversized_fragment_visibility_version"
        )
        != OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        or float(
            calibration["runtime_page_tiling"].get("minimum_scale", 0.0)
        )
        != SEMANTIC_DETECTOR_MINIMUM_SCALE
        or float(
            calibration["runtime_page_tiling"].get("maximum_scale", 0.0)
        )
        != SEMANTIC_DETECTOR_MAXIMUM_SCALE
        or int(
            calibration["runtime_page_tiling"].get("maximum_tiles", 0)
        )
        != SEMANTIC_DETECTOR_MAXIMUM_TILES
        or int(calibration["runtime_page_tiling"].get("pages", 0))
        != int(calibration["page_stitching"].get("pages", 0))
        or int(calibration["runtime_page_tiling"].get("tiles", 0))
        != int(calibration["page_stitching"].get("tiles", 0))
    ):
        failures.append("holdout_operating_point_calibration")
    else:
        try:
            calibration_minimum_true_positives = int(
                calibration.get("minimum_true_positives", 0)
            )
        except (TypeError, ValueError, OverflowError):
            calibration_minimum_true_positives = 0
        if calibration_minimum_true_positives < 10:
            failures.append("holdout_operating_point_calibration_support")
    if parity.get("passed") is not True:
        failures.append("onnx_parity")
    if parity.get("model_contract") != expected_model_contract:
        failures.append("onnx_model_contract")
    if str(parity.get("source_model_sha256") or "") != source_model_hash:
        failures.append("parity_source_model_hash")
    if str(parity.get("categories_sha256") or "") != categories_hash:
        failures.append("parity_categories_hash")
    if str(parity.get("onnx_sha256") or "") != onnx_hash:
        failures.append("parity_onnx_hash")
    if gpu_parity.get("passed") is not True:
        failures.append("onnx_gpu_parity")
    if str(gpu_parity.get("onnx_sha256") or "") != onnx_hash:
        failures.append("gpu_parity_onnx_hash")
    gpu_runtime = gpu_parity.get("runtime")
    gpu_metrics = gpu_parity.get("parity")
    if (
        not isinstance(gpu_runtime, dict)
        or str(gpu_runtime.get("onnxruntime") or "") != "1.26.0"
        or not isinstance(gpu_runtime.get("cuda_session_providers"), list)
        or not gpu_runtime["cuda_session_providers"]
        or gpu_runtime["cuda_session_providers"][0] != "CUDAExecutionProvider"
    ):
        failures.append("gpu_parity_runtime")
    if (
        not isinstance(gpu_metrics, dict)
        or gpu_metrics.get("passed") is not True
        or int(gpu_metrics.get("cpu_detections", 0)) <= 0
        or int(gpu_metrics.get("cuda_detections", 0)) <= 0
    ):
        failures.append("gpu_parity_metrics")
    try:
        gpu_comparison_floor = float(
            gpu_metrics.get("comparison_score_floor", -1.0)
        )
    except (TypeError, ValueError, OverflowError):
        gpu_comparison_floor = -1.0

    metrics = holdout.get("metrics")
    operating_points = (
        metrics.get("operating_points")
        if isinstance(metrics, dict)
        else None
    )
    if not isinstance(operating_points, dict):
        failures.append("holdout_operating_points")
        operating_points = {}
    selected_points: dict[str, dict[str, object]] = {}
    for class_name in sorted(SUPPORTED_RUNTIME_CLASSES):
        point = operating_points.get(class_name)
        calibration_point = (
            calibration.get("selected_points", {}).get(class_name)
            if isinstance(calibration, dict)
            and isinstance(calibration.get("selected_points"), dict)
            else None
        )
        if not isinstance(point, dict):
            failures.append(f"operating_point:{class_name}:missing")
            continue
        try:
            threshold = float(point.get("threshold", -1.0))
            precision = float(point.get("precision", -1.0))
            recall = float(point.get("recall", -1.0))
            true_positives = int(point.get("true_positives", 0))
            calibration_true_positives = int(
                point.get("calibration_true_positives", 0)
            )
            calibration_point_threshold = float(
                calibration_point.get("threshold", -1.0)
                if isinstance(calibration_point, dict)
                else -1.0
            )
            calibration_point_true_positives = int(
                calibration_point.get("true_positives", -1)
                if isinstance(calibration_point, dict)
                else -1
            )
            calibration_point_target_objects = int(
                calibration_point.get("target_objects", -1)
                if isinstance(calibration_point, dict)
                else -1
            )
            calibration_method = str(
                calibration_point.get("selection_method", "")
                if isinstance(calibration_point, dict)
                else ""
            )
            reported_calibration_method = str(
                point.get("calibration_selection_method", "")
            )
        except (TypeError, ValueError, OverflowError):
            failures.append(f"operating_point:{class_name}:invalid")
            continue
        calibrated_selection = (
            calibration_method
            == CALIBRATED_OPERATING_POINT_SELECTION_METHOD
            and calibration_true_positives
            >= calibration_minimum_true_positives
        )
        fixed_rare_selection = (
            calibration_method == FIXED_RARE_CLASS_SELECTION_METHOD
            and calibration_point_target_objects
            < calibration_minimum_true_positives
            and calibration_true_positives == 0
            and calibration_point_true_positives == 0
            and calibration_point_threshold
            == FIXED_RARE_CLASS_OPERATING_POINT_THRESHOLD
        )
        if (
            point.get("passed") is not True
            or point.get("calibration_passed") is not True
            or not isinstance(calibration_point, dict)
            or calibration_point.get("passed") is not True
            or reported_calibration_method != calibration_method
            or not (calibrated_selection or fixed_rare_selection)
            or calibration_true_positives
            != calibration_point_true_positives
            or calibration_point_threshold != threshold
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
            or not math.isfinite(precision)
            or not MINIMUM_OPERATING_POINT_PRECISION <= precision <= 1.0
            or not math.isfinite(recall)
            or not MINIMUM_OPERATING_POINT_RECALL <= recall <= 1.0
            or true_positives < MINIMUM_OPERATING_POINT_TRUE_POSITIVES
        ):
            failures.append(f"operating_point:{class_name}:unsafe")
            continue
        if (
            class_name in HIGH_RECALL_MARK_CLASSES
            and recall < MINIMUM_HIGH_RECALL_MARK_RECALL
        ):
            failures.append(
                f"operating_point:{class_name}:high_recall"
            )
            continue
        selected_points[class_name] = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "true_positives": true_positives,
            "false_positives": int(point.get("false_positives", 0)),
            "target_objects": int(point.get("target_objects", 0)),
        }
    if (
        not 0.0 <= gpu_comparison_floor <= 1.0
        or (
            selected_points
            and gpu_comparison_floor
            > min(float(point["threshold"]) for point in selected_points.values())
        )
    ):
        failures.append("gpu_parity_comparison_floor")
    tensor_contract = parity.get("tensor_contract")
    if not isinstance(tensor_contract, dict):
        failures.append("parity_tensor_contract")
        tensor_contract = {}
    if (
        tensor_contract.get("input_name") != "image"
        or tensor_contract.get("input_shape") != [1, 3, 1024, 1024]
        or tensor_contract.get("output_names") != ["boxes", "scores", "labels"]
        or float(tensor_contract.get("score_threshold", -1.0)) != 0.05
        or int(tensor_contract.get("detections_per_tile", 0)) != 300
        or float(tensor_contract.get("nms_iou", -1.0))
        != DETECTOR_NMS_IOU
    ):
        failures.append("parity_tensor_contract")
    gpu_tensor_contract = gpu_parity.get("tensor_contract")
    if (
        not isinstance(gpu_tensor_contract, dict)
        or gpu_tensor_contract.get("input_name") != "image"
        or gpu_tensor_contract.get("input_shape") != [1, 3, 1024, 1024]
        or gpu_tensor_contract.get("output_names")
        != ["boxes", "scores", "labels"]
    ):
        failures.append("gpu_parity_tensor_contract")
    if failures:
        raise RuntimeError(
            "semantic detector release authorization failed: "
            + "; ".join(dict.fromkeys(failures))
        )

    (
        model_target,
        categories_target,
        holdout_target,
        parity_target,
        gpu_parity_target,
        manifest_target,
    ) = targets
    _copy_exclusive(args.onnx, model_target)
    _copy_exclusive(args.categories, categories_target)
    _copy_exclusive(args.holdout_report, holdout_target)
    _copy_exclusive(args.parity_report, parity_target)
    _copy_exclusive(args.gpu_parity_report, gpu_parity_target)
    manifest = {
        "format": 1,
        "model_version": model_version,
        "purpose": (
            "release-gated corroboration of independently fitted source geometry; "
            "never a sole MusicXML writer"
        ),
        "integration_authorized": True,
        "model": _artifact(model_target),
        "categories": _artifact(categories_target),
        "release_gate": {
            "independent_holdout": {
                "passed": True,
                "independent_works": independent_works,
                **_artifact(holdout_target),
            },
            "onnx_parity": {
                "passed": True,
                **_artifact(parity_target),
            },
            "onnx_gpu_parity": {
                "passed": True,
                "runtime": "onnxruntime-gpu==1.26.0",
                **_artifact(gpu_parity_target),
            },
        },
        "operating_points": selected_points,
        "input": {
            "name": "image",
            "size": SEMANTIC_DETECTOR_INPUT_SIZE,
            "target_staff_spacing": (
                SEMANTIC_DETECTOR_TARGET_STAFF_SPACING
            ),
            "overlap": SEMANTIC_DETECTOR_TILE_OVERLAP,
            "page_nms_iou": SEMANTIC_PAGE_NMS_IOU,
            "tile_fragment_fusion_version": TILE_FRAGMENT_FUSION_VERSION,
            "maximum_tiles": SEMANTIC_DETECTOR_MAXIMUM_TILES,
            "minimum_scale": SEMANTIC_DETECTOR_MINIMUM_SCALE,
            "maximum_scale": SEMANTIC_DETECTOR_MAXIMUM_SCALE,
        },
        "outputs": {"names": ["boxes", "scores", "labels"]},
        "provenance": {
            "source_model_sha256": source_model_hash,
            "training_report_sha256": sha256_file(args.training_report),
            "holdout_report_sha256": sha256_file(args.holdout_report),
            "parity_report_sha256": sha256_file(args.parity_report),
            "gpu_parity_report_sha256": sha256_file(args.gpu_parity_report),
        },
    }
    atomic_write_json(manifest_target, manifest)
    # Re-open and re-hash the exact packaged files through the runtime verifier.
    verified = load_semantic_detector_assets(output)
    print(
        json.dumps(
            {
                "authorized": True,
                "model_version": verified.model_version,
                "manifest_sha256": verified.manifest_sha256,
                "model_sha256": sha256_file(verified.model_path),
                "independent_works": independent_works,
                "operating_points": selected_points,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
