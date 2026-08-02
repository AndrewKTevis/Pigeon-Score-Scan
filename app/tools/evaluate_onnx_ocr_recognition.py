#!/usr/bin/env python3
"""Evaluate an exported OCR recognition ONNX with the actual RapidOCR runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_label_rows(
    label_path: Path,
    *,
    project_root: Path,
    maximum_rows: int | None = None,
) -> list[tuple[Path, str]]:
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    project_root = project_root.resolve()
    rows: list[tuple[Path, str]] = []
    with label_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            relative_path, separator, label = line.partition("\t")
            if not separator or not relative_path or not label:
                raise ValueError(
                    f"{label_path}:{line_number}: malformed OCR label"
                )
            image_path = (project_root / relative_path).resolve()
            if not _path_within(image_path, project_root):
                raise ValueError(
                    f"{label_path}:{line_number}: image escapes project root"
                )
            if not image_path.is_file() or image_path.stat().st_size <= 0:
                raise FileNotFoundError(image_path)
            rows.append((image_path, label))
            if maximum_rows is not None and len(rows) >= maximum_rows:
                break
    if not rows:
        raise ValueError("OCR evaluation label file is empty")
    return rows


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def recognition_metrics(
    references: list[str],
    predictions: list[str],
) -> dict[str, float | int]:
    if not references or len(references) != len(predictions):
        raise ValueError("reference and prediction rows must be nonempty and equal")
    exact = sum(
        reference == prediction
        for reference, prediction in zip(references, predictions, strict=True)
    )
    similarities = [
        1.0
        - edit_distance(reference, prediction)
        / max(len(reference), len(prediction), 1)
        for reference, prediction in zip(references, predictions, strict=True)
    ]
    return {
        "samples": len(references),
        "correct": exact,
        "acc": exact / len(references),
        "norm_edit_dis": sum(similarities) / len(similarities),
    }


def evaluate_rows(
    rows: list[tuple[Path, str]],
    *,
    recognize_batch: Callable[[list[np.ndarray]], list[str]],
    batch_size: int,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    references: list[str] = []
    predictions: list[str] = []
    failures: list[dict[str, object]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = []
        for path, _label in batch:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unable to decode OCR crop: {path}")
            images.append(image)
        batch_predictions = recognize_batch(images)
        if len(batch_predictions) != len(batch):
            raise RuntimeError("recognizer returned the wrong number of rows")
        for (path, reference), prediction in zip(
            batch,
            batch_predictions,
            strict=True,
        ):
            prediction = str(prediction)
            references.append(reference)
            predictions.append(prediction)
            if prediction != reference and len(failures) < 200:
                failures.append(
                    {
                        "image": str(path),
                        "reference": reference,
                        "prediction": prediction,
                        "edit_distance": edit_distance(reference, prediction),
                    }
                )
    return recognition_metrics(references, predictions), failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--maximum-rows", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.model, args.keys):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    rows = load_label_rows(
        args.labels,
        project_root=args.project_root,
        maximum_rows=args.maximum_rows,
    )

    from rapidocr import RapidOCR
    from rapidocr.ch_ppocr_rec.typings import TextRecInput
    from rapidocr.utils.typings import EngineType

    engine = RapidOCR(
        params={
            "Global.log_level": "warning",
            "Global.use_det": False,
            "Global.use_cls": False,
            "Global.use_rec": True,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.model_path": str(args.model.resolve()),
            "Rec.rec_keys_path": str(args.keys.resolve()),
            "Rec.rec_batch_num": min(64, args.batch_size),
        }
    )

    def recognize(images: list[np.ndarray]) -> list[str]:
        output = engine.text_rec(
            TextRecInput(img=images, return_word_box=False)
        )
        if output.txts is None:
            raise RuntimeError("RapidOCR recognition returned no texts")
        return [str(value) for value in output.txts]

    metrics, failures = evaluate_rows(
        rows,
        recognize_batch=recognize,
        batch_size=args.batch_size,
    )
    report = {
        "schema_version": 1,
        "name": "scorescan-domain-ocr-onnx-runtime-evaluation-v1",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "keys": str(args.keys.resolve()),
        "keys_sha256": sha256_file(args.keys),
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256_file(args.labels),
        "metrics": metrics,
        "failures_truncated": len(failures) >= 200,
        "failures": failures,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(
        args.output_report.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output_report)
    # Keep a short Paddle-compatible metric footer for the shared release gate.
    print(f"acc:{float(metrics['acc']):.12f}")
    print(f"norm_edit_dis:{float(metrics['norm_edit_dis']):.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
