from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.primus_evaluation import (  # noqa: E402
    aggregate_primus_reports,
    compare_primus_semantics,
    parse_primus_semantic,
    parse_primus_semantic_text,
)
from scorescan.primus_onnx import PrimusOnnxModel  # noqa: E402
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


def _stable_sample(names: Sequence[str], limit: int, seed: str) -> list[str]:
    ranked = sorted(
        set(names),
        key=lambda name: hashlib.sha256(f"{seed}\0{name}".encode()).digest(),
    )
    return ranked if limit == 0 else ranked[:limit]


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_item in enumerate(right, start=1):
        current = [row]
        for column, left_item in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _dataset_index(dataset: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in dataset.glob("package_*/*/*.semantic")
        if not path.name.startswith("._")
    }


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 9) if values else None


def _token_aggregate(reports: Sequence[dict[str, object]]) -> dict[str, object]:
    successful = [item for item in reports if not item.get("error")]
    reference_count = sum(
        int(item["reference_token_count"]) for item in successful
    )
    edit_count = sum(int(item["token_edit_count"]) for item in successful)
    return {
        "total_reference_token_count": reference_count,
        "total_token_edit_count": edit_count,
        "micro_token_error_rate": (
            round(edit_count / reference_count, 9)
            if reference_count
            else (0.0 if successful else None)
        ),
        "mean_token_error_rate": _mean(
            [float(item["token_error_rate"]) for item in successful]
        ),
    }


def run_benchmark(
    dataset: Path,
    model_path: Path,
    vocabulary_path: Path,
    test_list: Path,
    output: Path,
    limit: int,
    seed: str,
    checkpoint_every: int,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    all_test_names = [
        line.strip()
        for line in test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = _stable_sample(all_test_names, limit, seed)
    semantic_index = _dataset_index(dataset)
    model_started = time.monotonic()
    model = PrimusOnnxModel(model_path, vocabulary_path)
    model_load_seconds = round(time.monotonic() - model_started, 3)
    reports: list[dict[str, object]] = []
    started = time.monotonic()

    metadata: dict[str, object] = {
        "dataset": str(dataset.resolve()),
        "dataset_name": "PrIMuS",
        "dataset_scope": "official held-out fold; synthetic monophonic printed incipits",
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "test_list": str(test_list.resolve()),
        "test_list_sha256": sha256_file(test_list),
        "official_test_case_count": len(set(all_test_names)),
        "seed": seed,
        "requested_cases": limit if limit else "all",
        "selected_cases": len(selected),
        "model_load_seconds": model_load_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": utc_now_iso(),
    }

    for index, case_id in enumerate(selected, start=1):
        case_started = time.monotonic()
        report: dict[str, object] = {"case": case_id}
        try:
            semantic_path = semantic_index.get(case_id)
            if semantic_path is None:
                raise FileNotFoundError(f"case is absent from extracted dataset: {case_id}")
            image_path = semantic_path.with_suffix(".png")
            reference_tokens = semantic_path.read_text(encoding="utf-8").split()
            predicted_tokens = model.predict_tokens(image_path)
            prediction_text = " ".join(predicted_tokens)
            reference = parse_primus_semantic(semantic_path)
            candidate = parse_primus_semantic_text(prediction_text)
            report.update(compare_primus_semantics(reference, candidate))
            token_edits = _levenshtein(reference_tokens, predicted_tokens)
            report.update(
                {
                    "semantic_path": str(semantic_path.relative_to(dataset)),
                    "source_sha256": sha256_file(image_path),
                    "reference_token_count": len(reference_tokens),
                    "predicted_token_count": len(predicted_tokens),
                    "token_edit_count": token_edits,
                    "token_error_rate": round(
                        token_edits / len(reference_tokens), 9
                    )
                    if reference_tokens
                    else 0.0,
                    "prediction": prediction_text,
                }
            )
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds"] = round(time.monotonic() - case_started, 3)
        reports.append(report)

        successful = [item for item in reports if not item.get("error")]
        if index % checkpoint_every == 0 or index == len(selected):
            partial = {
                "metadata": metadata,
                "aggregate": {
                    **aggregate_primus_reports(reports),
                    **_token_aggregate(reports),
                },
                "cases": reports,
            }
            atomic_write_json(output / "report.json", partial)
        status = "ERROR" if report.get("error") else (
            "EXACT" if report.get("semantic_exact") else (
                f"tokenER={float(report['token_error_rate']):.3f} "
                f"eventER={float(report['event_error_rate']):.3f}"
            )
        )
        print(
            f"[{index:04d}/{len(selected):04d}] {case_id}: "
            f"{status} ({report['elapsed_seconds']}s)",
            flush=True,
        )

    metadata["completed_at"] = utc_now_iso()
    metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
    successful = [item for item in reports if not item.get("error")]
    final = {
        "metadata": metadata,
        "aggregate": {
            **aggregate_primus_reports(reports),
            **_token_aggregate(reports),
        },
        "cases": reports,
    }
    atomic_write_json(output / "report.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the published PrIMuS semantic ONNX model on its official test fold."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Deterministic held-out cases to test; use 0 for all.",
    )
    parser.add_argument("--seed", default="scorescan-official-test-v1")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write the growing report after this many cases.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    result = run_benchmark(
        args.dataset,
        args.model,
        args.vocabulary,
        args.test_list,
        args.output,
        args.limit,
        args.seed,
        args.checkpoint_every,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
