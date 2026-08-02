from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.tools.authorize_semantic_detector_release import main
from app.tools.evaluate_semantic_detector_holdout import (
    DENSE_MAP_VERSION,
    PAGE_STITCHING_VERSION,
    RUNTIME_PAGE_TILING_VERSION,
)
from app.tools.muse_omr_contract import TRAINING_REGION_ROLE
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
)
from app.tools.train_deepscores_symbol_detector import (
    DETECTOR_NMS_IOU,
    PRIORITY_SELECTION_PROTOCOL,
    detector_model_contract,
    is_priority_mark_class,
)
from scorescan.semantic_detector import (
    SUPPORTED_RUNTIME_CLASSES,
    load_semantic_detector_assets,
)
from scorescan.semantic_detector_contract import (
    SEMANTIC_DETECTOR_INPUT_SIZE,
    SEMANTIC_DETECTOR_MAXIMUM_SCALE,
    SEMANTIC_DETECTOR_MAXIMUM_TILES,
    SEMANTIC_DETECTOR_MINIMUM_SCALE,
    SEMANTIC_DETECTOR_TARGET_STAFF_SPACING,
    SEMANTIC_DETECTOR_TILE_OVERLAP,
    TILE_FRAGMENT_FUSION_VERSION,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_metrics(class_names: list[str]) -> dict[str, object]:
    return {
        "classes": list(range(1, len(class_names) + 1)),
        "map": 1.0,
        "map_per_class_named": {name: 1.0 for name in class_names},
        "selection_score": 1.0,
        "selection_support_filtered_map": 1.0,
        "priority_mark_map": 1.0,
        "selection_minimum_class_support": 25,
        "priority_mark_minimum_class_support": 25,
        "selection_supported_classes": class_names,
        "priority_mark_supported_classes": [
            name for name in class_names if is_priority_mark_class(name)
        ],
    }


def _inputs(tmp_path: Path) -> dict[str, Path]:
    model = tmp_path / "model.pt"
    onnx = tmp_path / "model.onnx"
    categories = tmp_path / "categories.json"
    training = tmp_path / "training.json"
    holdout = tmp_path / "holdout.json"
    parity = tmp_path / "parity.json"
    gpu_parity = tmp_path / "gpu-parity.json"
    class_names = sorted(SUPPORTED_RUNTIME_CLASSES)
    test_class_counts = {
        str(label): 25 for label in range(1, len(class_names) + 1)
    }
    model.write_bytes(b"source-model")
    onnx.write_bytes(b"onnx-model")
    categories.write_text(
        json.dumps(
            {
                "format": 1,
                "classes": [
                    {"label": index, "name": name}
                    for index, name in enumerate(class_names, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    training.write_text(
        json.dumps(
            {
                "acceptance": {"passed": True},
                "best_model_sha256": _sha(model),
                "best_epoch": 1,
                "priority_selection_protocol": PRIORITY_SELECTION_PROTOCOL,
                "model_contract": detector_model_contract(),
                "configuration": {
                    "minimum_required_class_test_objects": 25
                },
                "data": {"test_class_counts": test_class_counts},
                "metrics": {
                    "epochs": [
                        {
                            "epoch": 1,
                            "test": _selection_metrics(class_names),
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    points = {
        name: {
            "passed": True,
            "calibration_passed": True,
            "calibration_selection_method": "development_calibrated",
            "calibration_true_positives": 40,
            "calibration_minimum_true_positives": 10,
            "threshold": 0.998,
            "precision": 0.999,
            "recall": 0.99,
            "true_positives": 40,
            "false_positives": 0,
            "target_objects": 50,
        }
        for name in SUPPORTED_RUNTIME_CLASSES
    }
    holdout.write_text(
        json.dumps(
            {
                "format": 3,
                "acceptance": {
                    "passed": True,
                    "minimum_required_class_test_objects": 25,
                },
                "integration_authorized": False,
                "model_sha256": _sha(model),
                "model_categories_sha256": _sha(categories),
                "priority_selection_protocol": PRIORITY_SELECTION_PROTOCOL,
                "model_contract": detector_model_contract(),
                "independent_works": 200,
                "test_class_counts": test_class_counts,
                "detection_metric_protocol": DENSE_MAP_VERSION,
                "page_stitching": {
                    "version": PAGE_STITCHING_VERSION,
                    "tile_fragment_fusion_version": (
                        TILE_FRAGMENT_FUSION_VERSION
                    ),
                    "runtime_layout_assignment": True,
                    "runtime_page_retiling_version": (
                        RUNTIME_PAGE_TILING_VERSION
                    ),
                    "layout_pages": 200,
                    "pages": 200,
                    "tiles": 900,
                    "tile_target_instances": 1000,
                    "unique_source_targets": 800,
                },
                "page_layout_evidence": {
                    "version": "scorescan-semantic-page-layout-evidence@1",
                    "sha256": "a" * 64,
                    "pages": 200,
                },
                "runtime_page_tiling": {
                    "version": RUNTIME_PAGE_TILING_VERSION,
                    "input_size": SEMANTIC_DETECTOR_INPUT_SIZE,
                    "target_staff_spacing": (
                        SEMANTIC_DETECTOR_TARGET_STAFF_SPACING
                    ),
                    "overlap": SEMANTIC_DETECTOR_TILE_OVERLAP,
                    "oversized_fragment_visibility_version": (
                        OVERSIZED_FRAGMENT_VISIBILITY_VERSION
                    ),
                    "minimum_scale": SEMANTIC_DETECTOR_MINIMUM_SCALE,
                    "maximum_scale": SEMANTIC_DETECTOR_MAXIMUM_SCALE,
                    "maximum_tiles": SEMANTIC_DETECTOR_MAXIMUM_TILES,
                    "pages": 200,
                    "tiles": 900,
                },
                "operating_point_calibration": {
                    "selection_dataset_role": TRAINING_REGION_ROLE,
                    "selected_splits": ["calibration"],
                    "holdout_reused_for_selection": False,
                    "source_overlap_with_holdout": [],
                    "minimum_true_positives": 10,
                    "page_stitching": {
                        "version": PAGE_STITCHING_VERSION,
                        "tile_fragment_fusion_version": (
                            TILE_FRAGMENT_FUSION_VERSION
                        ),
                        "runtime_layout_assignment": True,
                        "runtime_page_retiling_version": (
                            RUNTIME_PAGE_TILING_VERSION
                        ),
                        "layout_pages": 170,
                        "pages": 170,
                        "tiles": 700,
                    },
                    "page_layout_evidence": {
                        "version": (
                            "scorescan-semantic-page-layout-evidence@1"
                        ),
                        "sha256": "b" * 64,
                        "pages": 170,
                    },
                    "runtime_page_tiling": {
                        "version": RUNTIME_PAGE_TILING_VERSION,
                        "input_size": SEMANTIC_DETECTOR_INPUT_SIZE,
                        "target_staff_spacing": (
                            SEMANTIC_DETECTOR_TARGET_STAFF_SPACING
                        ),
                        "overlap": SEMANTIC_DETECTOR_TILE_OVERLAP,
                        "oversized_fragment_visibility_version": (
                            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
                        ),
                        "minimum_scale": SEMANTIC_DETECTOR_MINIMUM_SCALE,
                        "maximum_scale": SEMANTIC_DETECTOR_MAXIMUM_SCALE,
                        "maximum_tiles": SEMANTIC_DETECTOR_MAXIMUM_TILES,
                        "pages": 170,
                        "tiles": 700,
                    },
                    "selected_points": {
                        name: {
                            "passed": True,
                            "selection_method": "development_calibrated",
                            "threshold": 0.998,
                            "true_positives": 40,
                            "target_objects": 50,
                        }
                        for name in SUPPORTED_RUNTIME_CLASSES
                    },
                },
                "metrics": {
                    **_selection_metrics(class_names),
                    "operating_points": points,
                },
            }
        ),
        encoding="utf-8",
    )
    parity.write_text(
        json.dumps(
            {
                "passed": True,
                "source_model_sha256": _sha(model),
                "categories_sha256": _sha(categories),
                "onnx_sha256": _sha(onnx),
                "model_contract": detector_model_contract(),
                "tensor_contract": {
                    "input_name": "image",
                    "input_shape": [1, 3, 1024, 1024],
                    "output_names": ["boxes", "scores", "labels"],
                    "score_threshold": 0.05,
                    "detections_per_tile": 300,
                    "nms_iou": DETECTOR_NMS_IOU,
                },
            }
        ),
        encoding="utf-8",
    )
    gpu_parity.write_text(
        json.dumps(
            {
                "passed": True,
                "onnx_sha256": _sha(onnx),
                "tensor_contract": {
                    "input_name": "image",
                    "input_shape": [1, 3, 1024, 1024],
                    "output_names": ["boxes", "scores", "labels"],
                },
                "runtime": {
                    "onnxruntime": "1.26.0",
                    "cuda_session_providers": [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                },
                "parity": {
                    "passed": True,
                    "cpu_detections": 4,
                    "cuda_detections": 4,
                    "comparison_score_floor": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "onnx": onnx,
        "categories": categories,
        "training": training,
        "holdout": holdout,
        "parity": parity,
        "gpu_parity": gpu_parity,
    }


def _arguments(inputs: dict[str, Path], output: Path) -> list[str]:
    return [
        "--source-model",
        str(inputs["model"]),
        "--onnx",
        str(inputs["onnx"]),
        "--categories",
        str(inputs["categories"]),
        "--training-report",
        str(inputs["training"]),
        "--holdout-report",
        str(inputs["holdout"]),
        "--parity-report",
        str(inputs["parity"]),
        "--gpu-parity-report",
        str(inputs["gpu_parity"]),
        "--output-resources",
        str(output),
        "--model-version",
        "semantic-release-test-1",
    ]


def test_authorizer_packages_only_a_self_verifying_release(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "resources"

    assert main(_arguments(inputs, output)) == 0

    assets = load_semantic_detector_assets(output)
    assert assets.model_version == "semantic-release-test-1"
    assert assets.thresholds == {
        name: 0.998 for name in SUPPORTED_RUNTIME_CLASSES
    }
    assert (output / "semantic_detector_holdout.json").is_file()
    assert (output / "semantic_detector_onnx_parity.json").is_file()
    assert (output / "semantic_detector_onnx_gpu_parity.json").is_file()
    manifest = json.loads(
        (output / "semantic_detector.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["input"]["tile_fragment_fusion_version"]
        == TILE_FRAGMENT_FUSION_VERSION
    )


def test_authorizer_accepts_only_the_code_fixed_rare_class_threshold(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    point = holdout["metrics"]["operating_points"]["jumpText"]
    point["threshold"] = 0.995
    point["calibration_selection_method"] = "fixed_contract_rare_class"
    point["calibration_true_positives"] = 0
    calibration = holdout["operating_point_calibration"]["selected_points"][
        "jumpText"
    ]
    calibration.update(
        {
            "selection_method": "fixed_contract_rare_class",
            "threshold": 0.995,
            "true_positives": 0,
            "target_objects": 7,
        }
    )
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")
    output = tmp_path / "resources"

    assert main(_arguments(inputs, output)) == 0
    assert load_semantic_detector_assets(output).thresholds["jumpText"] == 0.995


def test_authorizer_rejects_changed_rare_class_fallback_threshold(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    point = holdout["metrics"]["operating_points"]["jumpText"]
    point["threshold"] = 0.994
    point["calibration_selection_method"] = "fixed_contract_rare_class"
    point["calibration_true_positives"] = 0
    calibration = holdout["operating_point_calibration"]["selected_points"][
        "jumpText"
    ]
    calibration.update(
        {
            "selection_method": "fixed_contract_rare_class",
            "threshold": 0.994,
            "true_positives": 0,
            "target_objects": 7,
        }
    )
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(RuntimeError, match="operating_point:jumpText:unsafe"):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_below_precision_operating_point(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    holdout["metrics"]["operating_points"]["tie"]["precision"] = 0.994
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(RuntimeError, match="operating_point:tie:unsafe"):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_holdout_selected_thresholds(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    holdout["operating_point_calibration"]["holdout_reused_for_selection"] = True
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="holdout_operating_point_calibration",
    ):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_training_split_threshold_selection(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    holdout["operating_point_calibration"]["selected_splits"] = [
        "train",
        "calibration",
    ]
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="holdout_operating_point_calibration",
    ):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_legacy_100_detection_metric(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    holdout["detection_metric_protocol"] = "torchmetrics-maxdets100"
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(RuntimeError, match="holdout_dense_metric_protocol"):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_changed_detector_nms_contract(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    holdout["model_contract"]["nms_iou"] = 0.5
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(RuntimeError, match="holdout_model_contract"):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_changed_training_matcher_contract(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    training = json.loads(inputs["training"].read_text(encoding="utf-8"))
    training["model_contract"]["foreground_iou_threshold"] = 0.5
    inputs["training"].write_text(json.dumps(training), encoding="utf-8")

    with pytest.raises(RuntimeError, match="training_model_contract"):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_unbound_calibration_threshold(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    holdout["operating_point_calibration"]["selected_points"]["tie"][
        "threshold"
    ] = 0.997
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(RuntimeError, match="operating_point:tie:unsafe"):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_below_recall_operating_point(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    holdout["metrics"]["operating_points"]["tie"]["recall"] = 0.949
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(RuntimeError, match="operating_point:tie:unsafe"):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_weak_high_recall_mark(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    holdout["metrics"]["operating_points"]["genericAccidental"]["recall"] = 0.985
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="operating_point:genericAccidental:high_recall",
    ):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_applies_high_recall_gate_to_relations(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    holdout = json.loads(inputs["holdout"].read_text(encoding="utf-8"))
    holdout["metrics"]["operating_points"]["tie"]["recall"] = 0.985
    inputs["holdout"].write_text(json.dumps(holdout), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="operating_point:tie:high_recall",
    ):
        main(_arguments(inputs, tmp_path / "resources"))


def test_authorizer_rejects_silent_gpu_fallback(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    report = json.loads(inputs["gpu_parity"].read_text(encoding="utf-8"))
    report["runtime"]["cuda_session_providers"] = ["CPUExecutionProvider"]
    inputs["gpu_parity"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="gpu_parity_runtime"):
        main(_arguments(inputs, tmp_path / "resources"))
