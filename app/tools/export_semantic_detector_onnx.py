#!/usr/bin/env python3
from __future__ import annotations

"""Export the canonical RetinaNet semantic verifier to fixed-contract ONNX.

The output report is also the PyTorch-versus-ONNX parity gate consumed by the
release authorizer.  A real score tile with at least one retained detection is
required; a blank-tensor smoke test is not accepted as deployment parity.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from app.tools.train_deepscores_symbol_detector import (
    DETECTOR_NMS_IOU,
    build_detector_model,
    category_label_name_map,
    detector_model_contract,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--parity-image", type=Path, required=True)
    parser.add_argument(
        "--parity-crop",
        default="0,0,1024,1024",
        help="x1,y1,x2,y2 crop in the source parity image",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--opset-version", type=int, default=18)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--detections-per-tile", type=int, default=300)
    parser.add_argument("--minimum-parity-detections", type=int, default=1)
    parser.add_argument("--maximum-box-error", type=float, default=0.10)
    parser.add_argument("--maximum-score-error", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def _canonical_rows(
    boxes: Any,
    scores: Any,
    labels: Any,
) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    import numpy as np

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("detector parity output lengths disagree")
    rows = [
        (
            int(label),
            float(score),
            tuple(float(value) for value in box),
        )
        for box, score, label in zip(boxes, scores, labels, strict=True)
    ]
    return sorted(
        rows,
        key=lambda item: (
            -item[1],
            item[0],
            item[2],
        ),
    )


def compare_detector_outputs(
    reference: tuple[Any, Any, Any],
    candidate: tuple[Any, Any, Any],
    *,
    minimum_detections: int,
    maximum_box_error: float,
    maximum_score_error: float,
) -> dict[str, Any]:
    reference_rows = _canonical_rows(*reference)
    candidate_rows = _canonical_rows(*candidate)
    failures: list[str] = []
    if len(reference_rows) < minimum_detections:
        failures.append(
            f"reference_detections={len(reference_rows)}<{minimum_detections}"
        )
    if len(reference_rows) != len(candidate_rows):
        failures.append(
            f"detection_count={len(candidate_rows)}!={len(reference_rows)}"
        )
    maximum_observed_box_error = 0.0
    maximum_observed_score_error = 0.0
    label_mismatches = 0
    for reference_row, candidate_row in zip(
        reference_rows,
        candidate_rows,
        strict=False,
    ):
        if reference_row[0] != candidate_row[0]:
            label_mismatches += 1
        maximum_observed_score_error = max(
            maximum_observed_score_error,
            abs(reference_row[1] - candidate_row[1]),
        )
        maximum_observed_box_error = max(
            maximum_observed_box_error,
            max(
                abs(left - right)
                for left, right in zip(
                    reference_row[2],
                    candidate_row[2],
                    strict=True,
                )
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
        "reference_detections": len(reference_rows),
        "onnx_detections": len(candidate_rows),
        "label_mismatches": label_mismatches,
        "maximum_box_error": maximum_observed_box_error,
        "maximum_score_error": maximum_observed_score_error,
        "box_error_limit": maximum_box_error,
        "score_error_limit": maximum_score_error,
        "minimum_detections": minimum_detections,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    for path in (args.model, args.categories, args.parity_image):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite semantic detector export artifacts")
    if (
        args.minimum_parity_detections <= 0
        or args.maximum_box_error < 0
        or args.maximum_score_error < 0
        or not 0 <= args.score_threshold <= 1
        or args.detections_per_tile <= 0
    ):
        raise ValueError("invalid semantic detector export gate")
    try:
        parity_crop = tuple(
            int(value.strip()) for value in args.parity_crop.split(",")
        )
    except ValueError as exc:
        raise ValueError("parity crop must contain four integers") from exc
    if (
        len(parity_crop) != 4
        or parity_crop[2] - parity_crop[0] != 1024
        or parity_crop[3] - parity_crop[1] != 1024
        or parity_crop[0] < 0
        or parity_crop[1] < 0
    ):
        raise ValueError("parity crop must be one positive 1024x1024 region")

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from PIL import Image, ImageOps
    from torchvision.transforms import functional as vision_f

    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    class_name_by_label = category_label_name_map(categories)
    number_of_classes = max(class_name_by_label) + 1
    model = build_detector_model(
        number_of_classes=number_of_classes,
        score_threshold=args.score_threshold,
        detections_per_tile=args.detections_per_tile,
        pretrained_backbone=False,
        class_name_by_label=class_name_by_label,
    )
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("semantic detector model is not a state dictionary")
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export was requested but CUDA is unavailable")
    model = model.to(device).eval()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, detector: torch.nn.Module) -> None:
            super().__init__()
            self.detector = detector

        def forward(self, image):
            output = self.detector([image[0]])[0]
            return output["boxes"], output["scores"], output["labels"]

    wrapper = ExportWrapper(model).to(device).eval()
    with Image.open(args.parity_image) as source:
        if parity_crop[2] > source.width or parity_crop[3] > source.height:
            raise ValueError("parity crop exceeds the source image")
        image = ImageOps.grayscale(source).convert("RGB").crop(parity_crop)
        tensor = vision_f.pil_to_tensor(image).float().div_(255.0).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.inference_mode():
        reference_tensors = wrapper(tensor)
    reference = tuple(value.detach().cpu().numpy() for value in reference_tensors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.onnx.export(
        wrapper,
        tensor,
        temporary,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={
            "boxes": {0: "detections"},
            "scores": {0: "detections"},
            "labels": {0: "detections"},
        },
    )
    onnx_model = onnx.load(str(temporary), load_external_data=True)
    onnx.checker.check_model(onnx_model, full_check=True)
    os.replace(temporary, args.output)

    session = ort.InferenceSession(
        str(args.output),
        providers=["CPUExecutionProvider"],
    )
    input_array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    candidate = tuple(
        session.run(["boxes", "scores", "labels"], {"image": input_array})
    )
    parity = compare_detector_outputs(
        reference,
        candidate,
        minimum_detections=args.minimum_parity_detections,
        maximum_box_error=args.maximum_box_error,
        maximum_score_error=args.maximum_score_error,
    )
    report = {
        "format": 1,
        "name": "scorescan-semantic-detector-onnx-parity-v1",
        "passed": parity["passed"],
        "source_model_sha256": sha256_file(args.model),
        "categories_sha256": sha256_file(args.categories),
        "model_contract": detector_model_contract(),
        "onnx_sha256": sha256_file(args.output),
        "onnx_bytes": args.output.stat().st_size,
        "parity_image_sha256": sha256_file(args.parity_image),
        "parity_crop_xyxy": list(parity_crop),
        "tensor_contract": {
            "input_name": "image",
            "input_shape": [1, 3, 1024, 1024],
            "output_names": ["boxes", "scores", "labels"],
            "opset_version": args.opset_version,
            "score_threshold": args.score_threshold,
            "detections_per_tile": args.detections_per_tile,
            "nms_iou": DETECTOR_NMS_IOU,
        },
        "runtime": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "export_device": str(device),
            "parity_provider": session.get_providers(),
        },
        "parity": parity,
        "elapsed_seconds": time.time() - started,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.output_report.with_suffix(
        args.output_report.suffix + ".tmp"
    )
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, args.output_report)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    if not parity["passed"]:
        raise RuntimeError(
            "semantic detector ONNX parity gate failed: "
            + "; ".join(parity["failures"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
