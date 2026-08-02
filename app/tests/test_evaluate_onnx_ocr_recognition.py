from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.tools import evaluate_onnx_ocr_recognition as module


def test_edit_distance_and_metrics_are_exact() -> None:
    assert module.edit_distance("Allegro", "Allegro") == 0
    assert module.edit_distance("Allegro", "Alegro") == 1
    metrics = module.recognition_metrics(
        ["Allegro", "mf"],
        ["Alegro", "mf"],
    )
    assert metrics["samples"] == 2
    assert metrics["acc"] == 0.5
    assert metrics["norm_edit_dis"] == pytest.approx(
        (6 / 7 + 1.0) / 2
    )


def test_label_loader_rejects_project_escape(tmp_path: Path) -> None:
    labels = tmp_path / "labels.txt"
    labels.write_text("../outside.png\tAllegro\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        module.load_label_rows(labels, project_root=tmp_path)


def test_evaluate_rows_batches_and_records_failures(tmp_path: Path) -> None:
    rows = []
    for index, label in enumerate(("Allegro", "mf", "dolce")):
        image = tmp_path / f"{index}.png"
        image.write_bytes(b"placeholder")
        rows.append((image, label))
    observed_batches: list[int] = []

    def fake_imread(_path, _mode):
        return np.full((16, 32, 3), 255, np.uint8)

    original = module.cv2.imread
    module.cv2.imread = fake_imread
    try:
        def recognize(images):
            observed_batches.append(len(images))
            values = ["Allegro", "mF", "dolce"]
            offset = sum(observed_batches[:-1])
            return values[offset : offset + len(images)]

        metrics, failures = module.evaluate_rows(
            rows,
            recognize_batch=recognize,
            batch_size=2,
        )
    finally:
        module.cv2.imread = original

    assert observed_batches == [2, 1]
    assert metrics["acc"] == pytest.approx(2 / 3)
    assert failures[0]["reference"] == "mf"
    assert failures[0]["prediction"] == "mF"
