from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.tools.select_semantic_detector_parity_sample import (
    main as select_main,
    select_parity_row,
)
from app.tools.verify_semantic_detector_onnx_runtime import (
    compare_runtime_outputs,
)


def test_parity_selector_prefers_geometry_and_text_coverage() -> None:
    rows = [
        {
            "split": "test",
            "image": "one.png",
            "crop_xyxy": [0, 0, 1024, 1024],
            "objects": [{"category_id": "slur"}],
        },
        {
            "split": "test",
            "image": "two.png",
            "crop_xyxy": [0, 0, 1024, 1024],
            "objects": [
                {"category_id": "slur"},
                {"category_id": "tie"},
                {"category_id": "hairpin"},
                {"category_id": "tempoText"},
            ],
        },
    ]

    assert select_parity_row(rows)["image"] == "two.png"


def test_parity_selector_resolves_and_hashes_real_image(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    Image.new("L", (1200, 1200), 255).save(image)
    test_jsonl = tmp_path / "test.jsonl"
    test_jsonl.write_text(
        json.dumps(
            {
                "split": "test",
                "image": "page.png",
                "source_key": "work/test",
                "image_id": "page-1",
                "crop_xyxy": [0, 0, 1024, 1024],
                "objects": [
                    {"category_id": "hairpin"},
                    {"category_id": "tempoText"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "selection.json"

    assert (
        select_main(
            [
                "--test-jsonl",
                str(test_jsonl),
                "--project-root",
                str(tmp_path),
                "--output-report",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["image"] == str(image.resolve())
    assert report["runtime_classes"] == ["hairpin", "tempoText"]
    assert report["runtime_object_count"] == 2


def test_cpu_cuda_comparison_rejects_numeric_drift() -> None:
    boxes = np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    scores = np.asarray([0.999], dtype=np.float32)
    labels = np.asarray([3], dtype=np.int64)
    passed = compare_runtime_outputs(
        (boxes, scores, labels),
        (boxes + np.float32(0.01), scores, labels),
        minimum_detections=1,
        maximum_box_error=0.1,
        maximum_score_error=1e-4,
        comparison_score_floor=0.5,
    )
    failed = compare_runtime_outputs(
        (boxes, scores, labels),
        (boxes + np.float32(0.2), scores, labels),
        minimum_detections=1,
        maximum_box_error=0.1,
        maximum_score_error=1e-4,
        comparison_score_floor=0.5,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert any("maximum_box_error" in item for item in failed["failures"])


def test_cpu_cuda_comparison_matches_equal_class_detections_spatially() -> None:
    reference = (
        np.asarray(
            [[100.0, 20.0, 130.0, 40.0], [10.0, 20.0, 40.0, 40.0]],
            dtype=np.float32,
        ),
        np.asarray([0.90000, 0.90001], dtype=np.float32),
        np.asarray([3, 3], dtype=np.int64),
    )
    candidate = (
        np.asarray(
            [[10.01, 20.0, 40.01, 40.0], [100.01, 20.0, 130.01, 40.0]],
            dtype=np.float32,
        ),
        np.asarray([0.89999, 0.90002], dtype=np.float32),
        np.asarray([3, 3], dtype=np.int64),
    )

    result = compare_runtime_outputs(
        reference,
        candidate,
        minimum_detections=1,
        maximum_box_error=0.1,
        maximum_score_error=1e-4,
        comparison_score_floor=0.5,
    )

    assert result["passed"] is True
