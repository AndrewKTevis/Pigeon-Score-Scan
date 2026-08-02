#!/usr/bin/env python3
"""Benchmark RapidOCR recognizers on frozen, degradation-stratified score text."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


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
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def text_family(value: str) -> str:
    normalized = normalized_text(value)
    has_alpha = any(character.isalpha() for character in normalized)
    has_digit = any(character.isdigit() for character in normalized)
    if has_digit and not has_alpha:
        return "numeric"
    if has_alpha and not has_digit and len(normalized.strip(" .,:;()[]{}")) <= 12:
        return "music_word"
    return "metadata_mixed"


def stable_subset(
    rows: list[dict[str, Any]], maximum: int | None, seed: int
) -> list[dict[str, Any]]:
    if maximum is None or maximum >= len(rows):
        return rows
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (
                f"{seed}\0{row['source_key']}\0{row['page']}\0"
                f"{row['word_index']}\0{row['text']}"
            ).encode("utf-8")
        ).hexdigest(),
    )[:maximum]


def summarize_predictions(
    predictions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(predictions)
    truth_characters = sum(len(normalized_text(row["truth"])) for row in rows)
    total_edits = sum(
        edit_distance(normalized_text(row["truth"]), normalized_text(row["prediction"]))
        for row in rows
    )
    exact = sum(
        normalized_text(row["truth"]) == normalized_text(row["prediction"])
        for row in rows
    )
    confidences = [float(row["confidence"]) for row in rows]
    return {
        "samples": len(rows),
        "exact_match": exact / max(1, len(rows)),
        "character_error_rate": total_edits / max(1, truth_characters),
        "mean_confidence": sum(confidences) / max(1, len(confidences)),
    }


def select_recognizer(
    *,
    small: dict[str, Any],
    medium: dict[str, Any],
    profiles: list[str],
    small_samples_per_second: float,
    medium_samples_per_second: float,
) -> dict[str, Any]:
    regressions = []
    for profile in profiles:
        for family in ("overall", "numeric", "music_word", "metadata_mixed"):
            small_exact = float(small[profile][family]["exact_match"])
            medium_exact = float(medium[profile][family]["exact_match"])
            if medium_exact + 0.005 < small_exact:
                regressions.append(
                    {
                        "profile": profile,
                        "family": family,
                        "small_exact": small_exact,
                        "medium_exact": medium_exact,
                    }
                )
    small_cer = sum(
        float(small[profile]["overall"]["character_error_rate"])
        for profile in profiles
    ) / len(profiles)
    medium_cer = sum(
        float(medium[profile]["overall"]["character_error_rate"])
        for profile in profiles
    ) / len(profiles)
    hard_profile = "scan_hard" if "scan_hard" in profiles else profiles[-1]
    hard_exact_gain = (
        float(medium[hard_profile]["overall"]["exact_match"])
        - float(small[hard_profile]["overall"]["exact_match"])
    )
    absolute_cer_gain = small_cer - medium_cer
    accuracy_selected = (
        "ppocrv6_medium"
        if not regressions
        and (absolute_cer_gain >= 0.0005 or hard_exact_gain >= 0.005)
        else "ppocrv6_small"
    )
    speed_ratio = medium_samples_per_second / max(
        small_samples_per_second, 1e-9
    )
    return {
        "small_mean_cer": small_cer,
        "medium_mean_cer": medium_cer,
        "absolute_cer_gain": absolute_cer_gain,
        "hard_profile": hard_profile,
        "hard_exact_match_gain": hard_exact_gain,
        "maximum_allowed_exact_regression": 0.005,
        "minimum_cer_gain": 0.0005,
        "minimum_hard_exact_gain": 0.005,
        "regressions": regressions,
        "accuracy_selected": accuracy_selected,
        "medium_to_small_speed_ratio": speed_ratio,
        "requires_accelerated_runtime_benchmark": (
            accuracy_selected == "ppocrv6_medium" and speed_ratio < 0.25
        ),
    }


def _degrade(image: Any, profile: str, seed: int) -> Any:
    import cv2
    import numpy as np

    if profile == "clean":
        return image
    rng = random.Random(seed)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if profile == "scan_light":
        scale = rng.uniform(0.72, 0.88)
        reduced = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.resize(
            reduced,
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        gray = cv2.GaussianBlur(gray, (3, 3), rng.uniform(0.2, 0.55))
        alpha = rng.uniform(0.88, 1.06)
        beta = rng.uniform(-7, 5)
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    elif profile == "scan_hard":
        scale = rng.uniform(0.55, 0.72)
        reduced = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.resize(
            reduced,
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        gray = cv2.GaussianBlur(gray, (3, 3), rng.uniform(0.55, 0.9))
        height, width = gray.shape
        ramp = np.linspace(
            rng.uniform(0.86, 0.98),
            rng.uniform(0.86, 0.98),
            width,
            dtype=np.float32,
        )
        noisy = gray.astype(np.float32) * ramp.reshape(1, -1)
        noise = np.random.default_rng(seed).normal(
            0.0, rng.uniform(2.0, 5.0), (height, width)
        )
        gray = np.clip(noisy + noise, 0, 255).astype(np.uint8)
        encode_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(58, 76)]
        success, encoded = cv2.imencode(".jpg", gray, encode_parameters)
        if not success:
            raise RuntimeError("JPEG degradation failed")
        gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError(f"unknown degradation profile: {profile}")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _load_samples(
    rows: list[dict[str, Any]],
    *,
    images_dir: Path,
    profiles: list[str],
    seed: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    import cv2

    page_cache: dict[str, Any] = {}
    images = []
    samples = []
    for row in rows:
        image_key = str(row["image"])
        if image_key not in page_cache:
            page = cv2.imread(str(images_dir / image_key), cv2.IMREAD_COLOR)
            if page is None:
                raise FileNotFoundError(images_dir / image_key)
            page_cache[image_key] = page
        page = page_cache[image_key]
        left, top, right, bottom = (round(value) for value in row["box_xyxy"])
        crop = page[top:bottom, left:right]
        if crop.size == 0:
            raise ValueError(f"empty text crop: {row}")
        sample_key = (
            f"{seed}\0{row['source_key']}\0{row['page']}\0"
            f"{row['word_index']}\0{row['text']}"
        )
        sample_seed = int(hashlib.sha256(sample_key.encode()).hexdigest()[:16], 16)
        for profile in profiles:
            images.append(_degrade(crop, profile, sample_seed))
            samples.append(
                {
                    "truth": row["text"],
                    "family": text_family(row["text"]),
                    "profile": profile,
                    "source_key": row["source_key"],
                    "page": row["page"],
                    "word_index": row["word_index"],
                }
            )
    return images, samples


def _recognize(
    images: list[Any],
    *,
    model_path: Path | None,
    batch_size: int,
    use_cuda: bool,
) -> tuple[list[str], list[float], float, dict[str, Any]]:
    from rapidocr import RapidOCR
    from rapidocr.ch_ppocr_rec import TextRecInput

    params: dict[str, Any] = {
        "Global.log_level": "error",
        "Rec.rec_batch_num": batch_size,
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
    }
    if model_path is not None:
        params["Rec.model_path"] = str(model_path.resolve())
    started = time.perf_counter()
    engine = RapidOCR(params=params)
    initialized = time.perf_counter()
    output = engine.text_rec(TextRecInput(img=images, return_word_box=False))
    elapsed = time.perf_counter() - started
    if output.txts is None:
        raise RuntimeError("recognizer returned no text")
    return (
        list(output.txts),
        [float(value) for value in output.scores],
        elapsed,
        {
            "initialization_seconds": initialized - started,
            "recognition_seconds": elapsed - (initialized - started),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--medium-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="require ONNX Runtime CUDAExecutionProvider",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for directory in (args.dataset_dir, args.images_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if not args.medium_model.is_file():
        raise FileNotFoundError(args.medium_model)
    if args.output.exists():
        raise FileExistsError(args.output)
    import onnxruntime

    # Listing CUDAExecutionProvider does not prove that its CUDA/cuDNN
    # dependencies can be loaded. The portable runtime installs NVIDIA's pip
    # packages outside the ordinary DLL search path, so preload them before
    # RapidOCR constructs any sessions.
    if args.use_cuda:
        from scorescan.accelerator import (
            preload_onnxruntime_cuda_dlls,
            probe_accelerator,
        )

        preload_onnxruntime_cuda_dlls()
        accelerator = probe_accelerator("cuda")
        if accelerator.selected != "cuda":
            raise RuntimeError(
                "CUDA was requested but the native runtime probe failed: "
                + str(accelerator.fallback_reason or accelerator.cuda_probe_error)
            )
    else:
        accelerator = None

    available_providers = onnxruntime.get_available_providers()
    if args.use_cuda and "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(
            "CUDA was requested but ONNX Runtime has no CUDAExecutionProvider"
        )
    rows = [
        json.loads(line)
        for line in (args.dataset_dir / f"{args.split}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    rows = stable_subset(rows, args.max_samples, args.seed)
    profiles = ["clean", "scan_light", "scan_hard"]
    images, sample_metadata = _load_samples(
        rows,
        images_dir=args.images_dir,
        profiles=profiles,
        seed=args.seed,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "split": args.split,
        "base_samples": len(rows),
        "profiles": profiles,
        "synthetic_degraded_samples": len(images),
        "runtime": {
            "onnxruntime": onnxruntime.__version__,
            "available_providers": available_providers,
            "use_cuda": args.use_cuda,
            "accelerator": (
                accelerator.to_dict() if accelerator is not None else None
            ),
        },
        "models": {},
    }
    all_predictions: dict[str, list[dict[str, Any]]] = {}
    for model_name, model_path in (
        ("ppocrv6_small", None),
        ("ppocrv6_medium", args.medium_model),
    ):
        predictions, scores, elapsed, timing = _recognize(
            images,
            model_path=model_path,
            batch_size=args.batch_size,
            use_cuda=args.use_cuda,
        )
        prediction_rows = [
            metadata
            | {
                "prediction": prediction,
                "confidence": score,
            }
            for metadata, prediction, score in zip(
                sample_metadata, predictions, scores, strict=True
            )
        ]
        all_predictions[model_name] = prediction_rows
        stratified = {}
        for profile in profiles:
            profile_rows = [
                row for row in prediction_rows if row["profile"] == profile
            ]
            stratified[profile] = {"overall": summarize_predictions(profile_rows)}
            for family in ("numeric", "music_word", "metadata_mixed"):
                stratified[profile][family] = summarize_predictions(
                    row for row in profile_rows if row["family"] == family
                )
        failures = sorted(
            (
                row
                | {
                    "edit_distance": edit_distance(
                        normalized_text(row["truth"]),
                        normalized_text(row["prediction"]),
                    )
                }
                for row in prediction_rows
                if normalized_text(row["truth"])
                != normalized_text(row["prediction"])
            ),
            key=lambda row: (
                -row["edit_distance"],
                row["profile"],
                row["truth"],
            ),
        )[:100]
        report["models"][model_name] = {
            "model_path": str(model_path.resolve()) if model_path else "bundled-small",
            "elapsed_seconds": elapsed,
            "timing": timing,
            "samples_per_second": len(images) / max(elapsed, 1e-9),
            "stratified": stratified,
            "failure_examples": failures,
        }

    small = report["models"]["ppocrv6_small"]["stratified"]
    medium = report["models"]["ppocrv6_medium"]["stratified"]
    report["selection"] = select_recognizer(
        small=small,
        medium=medium,
        profiles=profiles,
        small_samples_per_second=float(
            report["models"]["ppocrv6_small"]["samples_per_second"]
        ),
        medium_samples_per_second=float(
            report["models"]["ppocrv6_medium"]["samples_per_second"]
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["selection"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
