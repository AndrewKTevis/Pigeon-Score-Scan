#!/usr/bin/env python3
from __future__ import annotations

"""Verify CPU/CUDA ONNX Runtime parity for the release semantic detector."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from scorescan.util import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--parity-image", type=Path, required=True)
    parser.add_argument("--parity-crop", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--maximum-box-error", type=float, default=0.10)
    parser.add_argument("--maximum-score-error", type=float, default=1e-4)
    parser.add_argument("--minimum-detections", type=int, default=1)
    parser.add_argument(
        "--comparison-score-floor",
        type=float,
        required=True,
        help=(
            "lowest independently selected deployment threshold; detections "
            "below this score cannot affect runtime output"
        ),
    )
    return parser


def _canonical_rows(
    outputs: tuple[Any, Any, Any],
    *,
    minimum_score: float,
) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    import numpy as np

    boxes = np.asarray(outputs[0], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(outputs[1], dtype=np.float64).reshape(-1)
    labels = np.asarray(outputs[2], dtype=np.int64).reshape(-1)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("semantic detector ONNX output lengths disagree")
    rows = [
        (
            int(label),
            float(score),
            tuple(float(value) for value in box),
        )
        for box, score, label in zip(boxes, scores, labels, strict=True)
        if float(score) >= minimum_score
    ]
    # Match spatially rather than by score.  CPU and CUDA kernels may perturb
    # two nearly equal scores without changing either detection; ordering by
    # score would turn that harmless permutation into a false parity failure.
    return sorted(
        rows,
        key=lambda item: (
            item[0],
            0.5 * (item[2][0] + item[2][2]),
            0.5 * (item[2][1] + item[2][3]),
            item[2],
            -item[1],
        ),
    )


def compare_runtime_outputs(
    reference: tuple[Any, Any, Any],
    candidate: tuple[Any, Any, Any],
    *,
    minimum_detections: int,
    maximum_box_error: float,
    maximum_score_error: float,
    comparison_score_floor: float = 0.0,
) -> dict[str, Any]:
    reference_rows = _canonical_rows(
        reference,
        minimum_score=comparison_score_floor,
    )
    candidate_rows = _canonical_rows(
        candidate,
        minimum_score=comparison_score_floor,
    )
    failures: list[str] = []
    if len(reference_rows) < minimum_detections:
        failures.append(
            f"cpu_detections={len(reference_rows)}<{minimum_detections}"
        )
    if len(reference_rows) != len(candidate_rows):
        failures.append(
            f"cuda_detection_count={len(candidate_rows)}!={len(reference_rows)}"
        )
    label_mismatches = 0
    maximum_observed_box_error = 0.0
    maximum_observed_score_error = 0.0
    for left, right in zip(reference_rows, candidate_rows, strict=False):
        if left[0] != right[0]:
            label_mismatches += 1
        maximum_observed_score_error = max(
            maximum_observed_score_error,
            abs(left[1] - right[1]),
        )
        maximum_observed_box_error = max(
            maximum_observed_box_error,
            max(
                abs(a - b)
                for a, b in zip(left[2], right[2], strict=True)
            ),
        )
    if label_mismatches:
        failures.append(f"label_mismatches={label_mismatches}")
    if (
        not math.isfinite(maximum_observed_box_error)
        or maximum_observed_box_error > maximum_box_error
    ):
        failures.append(
            "maximum_box_error="
            f"{maximum_observed_box_error:.8f}>{maximum_box_error:.8f}"
        )
    if (
        not math.isfinite(maximum_observed_score_error)
        or maximum_observed_score_error > maximum_score_error
    ):
        failures.append(
            "maximum_score_error="
            f"{maximum_observed_score_error:.8f}>{maximum_score_error:.8f}"
        )
    return {
        "passed": not failures,
        "cpu_detections": len(reference_rows),
        "cuda_detections": len(candidate_rows),
        "label_mismatches": label_mismatches,
        "maximum_box_error": maximum_observed_box_error,
        "maximum_score_error": maximum_observed_score_error,
        "box_error_limit": maximum_box_error,
        "score_error_limit": maximum_score_error,
        "minimum_detections": minimum_detections,
        "comparison_score_floor": comparison_score_floor,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    onnx_path = args.onnx.resolve()
    parity_image = args.parity_image.resolve()
    output_report = args.output_report.resolve()
    for path in (onnx_path, parity_image):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_report.exists():
        raise FileExistsError(output_report)
    if (
        args.minimum_detections <= 0
        or args.maximum_box_error < 0
        or args.maximum_score_error < 0
        or not math.isfinite(args.comparison_score_floor)
        or not 0.0 <= args.comparison_score_floor <= 1.0
    ):
        raise ValueError("invalid semantic ONNX runtime parity gate")
    try:
        crop = tuple(int(value.strip()) for value in args.parity_crop.split(","))
    except ValueError as exc:
        raise ValueError("parity crop must contain four integers") from exc
    if (
        len(crop) != 4
        or crop[2] - crop[0] != 1024
        or crop[3] - crop[1] != 1024
        or crop[0] < 0
        or crop[1] < 0
    ):
        raise ValueError("parity crop must be one positive 1024x1024 region")

    import numpy as np
    import onnxruntime as ort
    from PIL import Image, ImageOps

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload(cuda=True, cudnn=True, msvc=True, directory="")
    with Image.open(parity_image) as source:
        if crop[2] > source.width or crop[3] > source.height:
            raise ValueError("parity crop exceeds the source image")
        image = ImageOps.grayscale(source).convert("RGB").crop(crop)
        tensor = (
            np.asarray(image, dtype=np.float32)
            .transpose(2, 0, 1)[None, ...]
            / np.float32(255.0)
        )
    cpu_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    cuda_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    cpu_providers = [str(value) for value in cpu_session.get_providers()]
    cuda_providers = [str(value) for value in cuda_session.get_providers()]
    if not cuda_providers or cuda_providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "semantic detector silently fell back from CUDA: "
            f"{cuda_providers}"
        )
    output_names = ["boxes", "scores", "labels"]
    reference = tuple(cpu_session.run(output_names, {"image": tensor}))
    candidate = tuple(cuda_session.run(output_names, {"image": tensor}))
    parity = compare_runtime_outputs(
        reference,
        candidate,
        minimum_detections=args.minimum_detections,
        maximum_box_error=args.maximum_box_error,
        maximum_score_error=args.maximum_score_error,
        comparison_score_floor=args.comparison_score_floor,
    )
    report = {
        "format": 1,
        "name": "scorescan-semantic-detector-onnx-cpu-cuda-parity-v1",
        "passed": parity["passed"],
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "parity_image_sha256": sha256_file(parity_image),
        "parity_crop_xyxy": list(crop),
        "tensor_contract": {
            "input_name": "image",
            "input_shape": [1, 3, 1024, 1024],
            "output_names": output_names,
        },
        "runtime": {
            "onnxruntime": ort.__version__,
            "available_providers": list(ort.get_available_providers()),
            "cpu_session_providers": cpu_providers,
            "cuda_session_providers": cuda_providers,
        },
        "parity": parity,
        "elapsed_seconds": time.time() - started,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_report.with_suffix(output_report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_report)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    if not parity["passed"]:
        raise RuntimeError(
            "semantic detector CPU/CUDA parity gate failed: "
            + "; ".join(parity["failures"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
