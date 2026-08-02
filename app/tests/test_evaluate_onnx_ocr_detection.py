from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.tools import evaluate_onnx_ocr_detection as module


def _box(left: float, top: float, right: float, bottom: float):
    return module._polygon(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ]
    )


def test_polygon_iou_and_maximum_cardinality_matching() -> None:
    truth = [_box(0, 0, 10, 10), _box(20, 0, 30, 10)]
    predicted = [_box(1, 0, 11, 10), _box(20, 0, 30, 10)]
    assert round(module.polygon_iou(truth[0], predicted[0]), 6) == 0.818182
    count, matches = module.maximum_matches(truth, predicted)
    assert count == 2
    assert len(matches) == 2


def test_polygon_points_are_compact_and_do_not_repeat_closure() -> None:
    assert module.polygon_points(_box(0.12345, 1, 10, 20)) == [
        [0.123, 1.0],
        [10.0, 1.0],
        [10.0, 20.0],
        [0.123, 20.0],
    ]


def test_aggregate_metrics_penalizes_missed_and_extra_boxes() -> None:
    truth = [_box(0, 0, 10, 10), _box(20, 0, 30, 10)]
    predicted = [_box(0, 0, 10, 10), _box(40, 0, 50, 10)]
    metrics = module.aggregate_metrics([(truth, predicted)])
    assert metrics["true_positive_boxes"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["hmean"] == 0.5


def test_runtime_parameters_serialize_enum_values() -> None:
    from rapidocr.utils.typings import EngineType

    assert module.serializable_runtime_parameters(
        {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.limit_side_len": 736,
        }
    ) == {
        "Det.engine_type": "onnxruntime",
        "Det.limit_side_len": 736,
    }


def test_load_labels_validates_project_relative_page_and_polygons(
    tmp_path: Path,
) -> None:
    page = tmp_path / "pages" / "page.png"
    page.parent.mkdir()
    Image.new("RGB", (100, 100), "white").save(page)
    labels = tmp_path / "test.det.txt"
    labels.write_text(
        "pages/page.png\t"
        + json.dumps(
            [
                {
                    "transcription": "Allegro",
                    "points": [[5, 5], [50, 5], [50, 20], [5, 20]],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = module.load_labels(labels, project_root=tmp_path)
    assert len(rows) == 1
    assert rows[0][0] == page
    assert rows[0][1][0].area == 675
