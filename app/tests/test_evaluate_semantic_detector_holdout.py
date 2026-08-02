from __future__ import annotations

import pytest

from app.tools.evaluate_semantic_detector_holdout import (
    acceptance_failures,
    build_parser,
    calibration_isolation_failures,
    compute_dense_detection_metrics,
    evaluate_fixed_operating_point,
    fixed_rare_class_operating_point,
    holdout_isolation_failures,
    minimum_recall_for_class,
    retile_complete_page_rows_for_runtime,
    select_high_precision_operating_point,
    stitch_tiled_detections,
    unique_class_counts,
)
from scorescan.semantic_detector_contract import HIGH_RECALL_MARK_CLASSES
from app.tools.expand_overlapping_semantic_targets import TRANSFORMATION_VERSION
from app.tools.muse_omr_contract import (
    BENCHMARK_SELECTION_ROLE,
    MAXIMUM_SCAN_PAGE_ASPECT_RATIO,
    SCAN_PAGE_SHAPE_CONTRACT,
    TRAINING_REGION_ROLE,
)
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_PROVENANCE,
)
from app.tools.semantic_target_visibility import (
    OVERSIZED_FRAGMENT_VISIBILITY_VERSION,
)


def test_holdout_acceptance_checks_localization_and_every_required_class() -> None:
    metrics = {
        "map_50": 0.96,
        "map_75": 0.91,
        "priority_mark_map": 0.86,
        "map_per_class_named": {
            "hairpin": 0.84,
            "slur": 0.90,
        },
    }
    assert acceptance_failures(
        metrics,
        minimum_map_50=0.95,
        minimum_map_75=0.90,
        minimum_priority_map=0.85,
        required_class_maps={"hairpin": 0.85, "slur": 0.85},
    ) == ["class:hairpin=0.840000<0.850000"]


def test_holdout_acceptance_fails_closed_on_missing_metrics() -> None:
    failures = acceptance_failures(
        {},
        minimum_map_50=0.95,
        minimum_map_75=0.90,
        minimum_priority_map=0.85,
        required_class_maps={"tie": 0.85},
    )
    assert len(failures) == 4


def test_high_cost_mark_classes_get_stronger_recall_floor() -> None:
    for class_name in HIGH_RECALL_MARK_CLASSES:
        assert minimum_recall_for_class(
            class_name,
            minimum_recall=0.98,
            minimum_high_recall_mark_recall=0.99,
        ) == 0.99
    assert minimum_recall_for_class(
        "tempoText",
        minimum_recall=0.98,
        minimum_high_recall_mark_recall=0.99,
    ) == 0.98
    assert minimum_recall_for_class(
        "tie",
        minimum_recall=0.995,
        minimum_high_recall_mark_recall=0.99,
    ) == 0.995


def test_operating_point_is_grouped_reproducible_and_high_precision() -> None:
    outputs = [
        {
            "boxes": [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ],
            "scores": [0.999, 0.998, 0.50],
            "labels": [7, 7, 7],
        }
    ]
    targets = [
        {
            "boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "labels": [7, 7],
        }
    ]

    point = select_high_precision_operating_point(
        outputs,
        targets,
        label=7,
        class_name="hairpin",
        minimum_precision=0.995,
        minimum_recall=1.0,
        minimum_true_positives=2,
    )

    assert point["passed"]
    assert point["threshold"] == 0.998
    assert point["precision"] == 1.0
    assert point["recall"] == 1.0
    assert point["true_positives"] == 2
    assert point["false_positives"] == 0


def test_operating_point_and_acceptance_fail_when_precision_is_insufficient() -> None:
    point = select_high_precision_operating_point(
        [
            {
                "boxes": [[40, 40, 50, 50], [0, 0, 10, 10]],
                "scores": [0.999, 0.998],
                "labels": [3, 3],
            }
        ],
        [{"boxes": [[0, 0, 10, 10]], "labels": [3]}],
        label=3,
        class_name="tie",
        minimum_precision=0.995,
        minimum_recall=1.0,
        minimum_true_positives=1,
    )

    assert not point["passed"]
    failures = acceptance_failures(
        {
            "map_50": 1.0,
            "map_75": 1.0,
            "priority_mark_map": 1.0,
            "map_per_class_named": {"tie": 1.0},
        },
        minimum_map_50=0.95,
        minimum_map_75=0.90,
        minimum_priority_map=0.85,
        required_class_maps={"tie": 0.85},
        operating_points={"tie": point},
    )
    assert failures == ["operating_point:tie"]


def test_fixed_operating_point_does_not_adapt_to_holdout_scores() -> None:
    outputs = [
        {
            "boxes": [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ],
            "scores": [0.999, 0.80, 0.79],
            "labels": [7, 7, 7],
        }
    ]
    targets = [
        {
            "boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "labels": [7, 7],
        }
    ]

    point = evaluate_fixed_operating_point(
        outputs,
        targets,
        label=7,
        class_name="hairpin",
        threshold=0.90,
        minimum_precision=0.995,
        minimum_recall=0.95,
        minimum_true_positives=1,
    )

    assert point["threshold"] == 0.90
    assert point["precision"] == 1.0
    assert point["recall"] == 0.5
    assert point["true_positives"] == 1
    assert point["false_positives"] == 0
    assert not point["passed"]


def test_rare_class_fallback_is_fixed_without_development_predictions() -> None:
    point = fixed_rare_class_operating_point(
        label=11,
        class_name="jumpText",
        target_objects=7,
        minimum_precision=0.995,
        minimum_recall=0.98,
        minimum_true_positives=10,
    )

    assert point["selection_method"] == "fixed_contract_rare_class"
    assert point["threshold"] == 0.995
    assert point["target_objects"] == 7
    assert point["true_positives"] == 0
    assert point["precision"] is None
    assert point["recall"] is None
    assert point["passed"]


def test_holdout_isolation_requires_distinct_work_coverage() -> None:
    manifest = {
        "role": BENCHMARK_SELECTION_ROLE,
        "source_split_overlap": 0,
        "forbidden_selection_overlap": [],
        "forbidden_work_overlap": [],
        "accepted_works": 205,
        "target_assignment_version": TRANSFORMATION_VERSION,
        "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
        "maximum_scan_page_aspect_ratio": (
            MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        ),
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
    }
    preparation = {
        "role": BENCHMARK_SELECTION_ROLE,
        "forbidden_selection_overlap": [],
        "forbidden_work_overlap": [],
        "selected_works": 231,
        "accepted_works": 205,
        "source_count_by_split": {"test": 205},
        "transformation_version": TRANSFORMATION_VERSION,
        "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
        "maximum_scan_page_aspect_ratio": (
            MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        ),
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
    }

    assert holdout_isolation_failures(
        manifest,
        preparation,
        minimum_independent_works=200,
    ) == []

    preparation["accepted_works"] = 199
    assert holdout_isolation_failures(
        manifest,
        preparation,
        minimum_independent_works=200,
    ) == [
        "accepted_works=199<200",
        "manifest_accepted_works=205!=199",
        "test_sources=205!=199",
    ]


def test_operating_point_calibration_requires_disjoint_training_role() -> None:
    manifest = {
        "role": TRAINING_REGION_ROLE,
        "source_split_overlap": 0,
        "forbidden_selection_overlap": [],
        "forbidden_work_overlap": [],
        "target_assignment_version": TRANSFORMATION_VERSION,
        "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
        "maximum_scan_page_aspect_ratio": (
            MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        ),
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
    }
    preparation = {
        "role": TRAINING_REGION_ROLE,
        "forbidden_selection_overlap": [],
        "forbidden_work_overlap": [],
        "transformation_version": TRANSFORMATION_VERSION,
        "scan_page_shape_contract": SCAN_PAGE_SHAPE_CONTRACT,
        "maximum_scan_page_aspect_ratio": (
            MAXIMUM_SCAN_PAGE_ASPECT_RATIO
        ),
        "oversized_fragment_visibility_version": (
            OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
    }
    assert calibration_isolation_failures(manifest, preparation) == []

    preparation["role"] = BENCHMARK_SELECTION_ROLE
    assert calibration_isolation_failures(manifest, preparation) == [
        "preparation_role"
    ]


def test_holdout_parser_can_limit_threshold_selection_to_unseen_calibration() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--prepared-dir",
            "holdout",
            "--images-dir",
            "holdout-images",
            "--operating-point-calibration-prepared-dir",
            "development",
            "--operating-point-calibration-images-dir",
            "development-images",
            "--operating-point-calibration-split",
            "calibration",
            "--page-layout-evidence",
            "holdout-layout.json",
            "--operating-point-calibration-page-layout-evidence",
            "calibration-layout.json",
            "--model",
            "model.pt",
            "--model-categories",
            "categories.json",
            "--output-report",
            "evaluation.json",
        ]
    )

    assert args.operating_point_calibration_split == ["calibration"]


def test_holdout_parser_keeps_legacy_calibration_default_explicitly_unset() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--prepared-dir",
            "holdout",
            "--images-dir",
            "holdout-images",
            "--operating-point-calibration-prepared-dir",
            "development",
            "--operating-point-calibration-images-dir",
            "development-images",
            "--page-layout-evidence",
            "holdout-layout.json",
            "--operating-point-calibration-page-layout-evidence",
            "calibration-layout.json",
            "--model",
            "model.pt",
            "--model-categories",
            "categories.json",
            "--output-report",
            "evaluation.json",
        ]
    )

    assert args.operating_point_calibration_split is None


def test_page_stitching_unions_targets_and_suppresses_tile_duplicates() -> None:
    rows = [
        {
            "source_key": "work-a",
            "image": "page.png",
            "image_id": "page-a",
            "crop_xyxy": [0, 0, 10, 10],
            "objects": [
                {
                    "source_object_id": "object-1",
                    "box_xyxy": [8, 2, 10, 4],
                    "page_box_xyxy": [8, 2, 12, 4],
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                    "label": 37,
                }
            ],
        },
        {
            "source_key": "work-a",
            "image": "page.png",
            "image_id": "page-a",
            "crop_xyxy": [5, 0, 15, 10],
            "objects": [
                {
                    "source_object_id": "object-1",
                    "box_xyxy": [3, 2, 7, 4],
                    "page_box_xyxy": [8, 2, 12, 4],
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                    "label": 37,
                }
            ],
        },
    ]
    targets = [
        {"boxes": [[8, 2, 10, 4]], "labels": [37]},
        {"boxes": [[3, 2, 7, 4]], "labels": [37]},
    ]
    outputs = [
        {
            "boxes": [[8, 2, 10, 4]],
            "labels": [37],
            "scores": [0.99],
        },
        {
            "boxes": [[3, 2, 5, 4], [7, 7, 9, 9]],
            "labels": [37, 31],
            "scores": [0.98, 0.90],
        },
    ]

    stitched_outputs, stitched_targets, audit = stitch_tiled_detections(
        rows,
        outputs,
        targets,
    )

    assert len(stitched_outputs) == len(stitched_targets) == 1
    assert stitched_targets[0]["labels"].tolist() == [37]
    assert stitched_targets[0]["boxes"].tolist() == [
        [8.0, 2.0, 12.0, 4.0]
    ]
    assert sorted(stitched_outputs[0]["labels"].tolist()) == [31, 37]
    assert audit["tile_target_instances"] == 2
    assert audit["unique_source_targets"] == 1
    assert audit["duplicate_target_instances"] == 1
    assert audit["tile_predictions"] == 3
    assert audit["page_predictions"] == 2
    assert audit["nms_removed_predictions"] == 1
    assert audit["maximum_page_predictions"] == 2
    assert audit["maximum_page_targets"] == 1
    assert unique_class_counts(rows) == {37: 1}


def test_unique_class_counts_scope_source_object_ids_to_a_page() -> None:
    rows = [
        {
            "source_key": "work-a",
            "image": f"page-{page}.png",
            "image_id": f"page-{page}",
            "objects": [
                {
                    "source_object_id": "locally-stable-id",
                    "label": 14,
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        }
        for page in (1, 2)
    ]

    assert unique_class_counts(rows) == {14: 2}


def test_runtime_page_stitching_fuses_opposing_text_fragments() -> None:
    page_box = [5, 2, 10, 4]
    base_object = {
        "source_object_id": "text-object-1",
        "page_box_xyxy": page_box,
        "target_geometry_provenance": COMPLETE_PAGE_TARGET_PROVENANCE,
        "label": 41,
    }
    rows = [
        {
            "source_key": "work-a",
            "image": "page.png",
            "image_id": "page-a",
            "crop_xyxy": [0, 0, 10, 10],
            "objects": [base_object | {"box_xyxy": [5, 2, 10, 4]}],
        },
        {
            "source_key": "work-a",
            "image": "page.png",
            "image_id": "page-a",
            "crop_xyxy": [5, 0, 15, 10],
            "objects": [base_object | {"box_xyxy": [0, 2, 5, 4]}],
        },
    ]
    targets = [
        {"boxes": [[5, 2, 10, 4]], "labels": [41]},
        {"boxes": [[0, 2, 5, 4]], "labels": [41]},
    ]
    outputs = [
        {
            "boxes": [[7, 2, 10, 4]],
            "labels": [41],
            "scores": [0.99],
        },
        {
            "boxes": [[0, 2, 4, 4]],
            "labels": [41],
            "scores": [0.98],
        },
    ]
    key = ("work-a", "page.png", "page-a")
    layouts = {
        key: {
            "systems": [
                {
                    "index": 1,
                    "line_y": [2, 3, 4, 5, 6],
                    "spacing": 1,
                }
            ]
        }
    }

    stitched_outputs, stitched_targets, audit = stitch_tiled_detections(
        rows,
        outputs,
        targets,
        class_name_by_label={41: "scoreText"},
        layouts_by_page=layouts,
    )

    assert stitched_outputs[0]["labels"].tolist() == [41]
    assert stitched_outputs[0]["boxes"].tolist() == [
        [5.0, 2.0, 10.0, 4.0]
    ]
    assert stitched_targets[0]["boxes"].tolist() == [
        [5.0, 2.0, 10.0, 4.0]
    ]
    assert audit["runtime_layout_assignment"] is True
    assert audit["mapped_tile_predictions"] == 2
    assert audit["fused_fragment_predictions"] == 1
    assert audit["page_predictions"] == 1


def test_complete_page_targets_are_retiled_with_the_runtime_contract() -> None:
    rows = [
        {
            "split": "test",
            "source_key": "work-a",
            "image": "page.png",
            "image_id": "page-a",
            "crop_xyxy": [0, 0, 1024, 512],
            "objects": [
                {
                    "source_object_id": "hairpin-1",
                    "category_id": "hairpin",
                    "label": 17,
                    "box_xyxy": [800, 100, 1024, 130],
                    "page_box_xyxy": [800, 100, 1200, 130],
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        },
        {
            "split": "test",
            "source_key": "work-a",
            "image": "page.png",
            "image_id": "page-a",
            "crop_xyxy": [512, 0, 1536, 512],
            "objects": [
                {
                    "source_object_id": "hairpin-1",
                    "category_id": "hairpin",
                    "label": 17,
                    "box_xyxy": [288, 100, 688, 130],
                    "page_box_xyxy": [800, 100, 1200, 130],
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        },
    ]
    key = ("work-a", "page.png", "page-a")
    layouts = {
        key: {
            "width": 1536,
            "height": 512,
            "systems": [
                {
                    "index": 1,
                    "line_y": [100, 121, 142, 163, 184],
                    "spacing": 21,
                }
            ],
        }
    }

    runtime_rows, audit = retile_complete_page_rows_for_runtime(
        rows,
        layouts,
        minimum_visible_fraction=0.8,
        long_span_minimum_visible_fraction=0.25,
    )

    assert [row["crop_xyxy"] for row in runtime_rows] == [
        [0, 0, 1024, 512],
        [512, 0, 1536, 512],
    ]
    assert runtime_rows[0]["objects"][0]["box_xyxy"] == [
        800.0,
        100.0,
        1024.0,
        130.0,
    ]
    assert runtime_rows[1]["objects"][0]["box_xyxy"] == [
        288.0,
        100.0,
        688.0,
        130.0,
    ]
    assert audit["pages"] == 1
    assert audit["tiles"] == 2
    assert audit["unique_source_targets"] == 1
    assert audit["tile_target_instances"] == 2


def test_runtime_retiling_rejects_stitched_page_with_identity() -> None:
    rows = [
        {
            "split": "test",
            "source_key": "work-panorama",
            "image": "stitched.png",
            "image_id": "page-panorama",
            "crop_xyxy": [0, 0, 301, 100],
            "objects": [],
        }
    ]
    layouts = {
        ("work-panorama", "stitched.png", "page-panorama"): {
            "width": 301,
            "height": 100,
            "systems": [
                {
                    "index": 1,
                    "line_y": [20, 25, 30, 35, 40],
                    "spacing": 5,
                }
            ],
        }
    }

    with pytest.raises(
        ValueError,
        match=r"work-panorama.*301x100.*maximum=3\.000000",
    ):
        retile_complete_page_rows_for_runtime(
            rows,
            layouts,
            minimum_visible_fraction=0.8,
            long_span_minimum_visible_fraction=0.25,
        )


def test_runtime_retiling_fails_if_scaling_drops_a_complete_page_target() -> None:
    key = ("work-a", "page.png", "page-a")
    rows = [
        {
            "split": "test",
            "source_key": key[0],
            "image": key[1],
            "image_id": key[2],
            "crop_xyxy": [0, 0, 1024, 1024],
            "objects": [
                {
                    "source_object_id": "dynamic-wide",
                    "category_id": "genericDynamic",
                    "label": 14,
                    "box_xyxy": [776.449, 738.019, 1024, 786.894],
                    "page_box_xyxy": [
                        776.449,
                        738.019,
                        1409.07,
                        786.894,
                    ],
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        }
    ]
    layouts = {
        key: {
            "width": 2977,
            "height": 4208,
            "systems": [
                {
                    "index": 1,
                    "line_y": [700, 725, 750, 775, 800],
                    "spacing": 25,
                }
            ],
        }
    }

    with pytest.raises(
        ValueError,
        match=r"dropped complete-page target.*dynamic-wide",
    ):
        retile_complete_page_rows_for_runtime(
            rows,
            layouts,
            minimum_visible_fraction=0.8,
            long_span_minimum_visible_fraction=0.25,
        )


def test_runtime_retiling_and_stitching_round_trip_scaled_page_geometry() -> None:
    rows = [
        {
            "split": "test",
            "source_key": "work-scale",
            "image": "page.png",
            "image_id": "page-scale",
            "crop_xyxy": [0, 0, 1024, 1024],
            "objects": [
                {
                    "source_object_id": "slur-1",
                    "category_id": "slur",
                    "label": 2,
                    "box_xyxy": [100, 100, 300, 130],
                    "page_box_xyxy": [100, 100, 300, 130],
                    "target_geometry_provenance": (
                        COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        }
    ]
    key = ("work-scale", "page.png", "page-scale")
    layouts = {
        key: {
            "width": 2048,
            "height": 1024,
            "systems": [
                {
                    "index": 1,
                    "line_y": [100, 142, 184, 226, 268],
                    "spacing": 42,
                }
            ],
        }
    }
    runtime_rows, _audit = retile_complete_page_rows_for_runtime(
        rows,
        layouts,
        minimum_visible_fraction=0.8,
        long_span_minimum_visible_fraction=0.25,
    )
    assert len(runtime_rows) == 1
    assert runtime_rows[0]["runtime_page_scale"] == 0.5
    assert runtime_rows[0]["objects"][0]["box_xyxy"] == [
        50.0,
        50.0,
        150.0,
        65.0,
    ]

    stitched_outputs, stitched_targets, audit = stitch_tiled_detections(
        runtime_rows,
        [
            {
                "boxes": [[50, 50, 150, 65]],
                "labels": [2],
                "scores": [0.999],
            }
        ],
        [{"boxes": [[50, 50, 150, 65]], "labels": [2]}],
        class_name_by_label={2: "slur"},
        layouts_by_page=layouts,
    )

    assert stitched_outputs[0]["boxes"].tolist() == [
        [100.0, 100.0, 300.0, 130.0]
    ]
    assert stitched_targets[0]["boxes"].tolist() == [
        [100.0, 100.0, 300.0, 130.0]
    ]
    assert audit["runtime_page_retiling_version"] is not None


def test_dense_map_retains_more_than_100_correct_page_detections() -> None:
    boxes = [
        [float(index * 3), 0.0, float(index * 3 + 2), 2.0]
        for index in range(150)
    ]
    metrics = compute_dense_detection_metrics(
        [
            {
                "boxes": boxes,
                "scores": [
                    1.0 - index / 1000.0 for index in range(len(boxes))
                ],
                "labels": [7] * len(boxes),
            }
        ],
        [{"boxes": boxes, "labels": [7] * len(boxes)}],
    )

    assert metrics["unbounded_detections_per_image"] is True
    assert metrics["evaluated_predictions"] == 150
    assert metrics["evaluated_targets"] == 150
    assert metrics["map"] == 1.0
    assert metrics["map_50"] == 1.0
    assert metrics["map_75"] == 1.0
    assert metrics["map_per_class"] == [1.0]
    assert metrics["classes"] == [7]


def test_dense_map_penalizes_false_positive_and_localization_error() -> None:
    metrics = compute_dense_detection_metrics(
        [
            {
                "boxes": [
                    [30.0, 30.0, 40.0, 40.0],
                    [0.0, 0.0, 8.0, 10.0],
                ],
                "scores": [0.99, 0.98],
                "labels": [3, 3],
            }
        ],
        [{"boxes": [[0.0, 0.0, 10.0, 10.0]], "labels": [3]}],
    )

    assert metrics["map_50"] == 0.5
    assert metrics["map_75"] == 0.5
    assert 0.0 < metrics["map"] < 0.5
