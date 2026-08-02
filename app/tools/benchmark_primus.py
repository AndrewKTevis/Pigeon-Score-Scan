from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.config import APP_VERSION, WORKFLOW_VERSION  # noqa: E402
from scorescan.omr import HomrRunner  # noqa: E402
from scorescan.primus_evaluation import (  # noqa: E402
    aggregate_primus_reports,
    compare_primus_semantics,
    parse_musicxml_semantics,
    parse_primus_semantic,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


def _stable_sample(
    dataset: Path,
    limit: int,
    seed: str,
    allowed_names: set[str] | None = None,
) -> list[Path]:
    semantic_files = [
        path
        for path in dataset.glob("package_*/*/*.semantic")
        if not path.name.startswith("._")
        and (allowed_names is None or path.stem in allowed_names)
    ]
    ranked = sorted(
        semantic_files,
        key=lambda path: hashlib.sha256(
            f"{seed}\0{path.relative_to(dataset).as_posix()}".encode()
        ).digest(),
    )
    return ranked[:limit]


def _case_id(semantic_path: Path) -> str:
    return semantic_path.stem


def run_benchmark(
    dataset: Path,
    output: Path,
    limit: int,
    seed: str,
    case_list: Path | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    cases_root = output / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    runner_logs: deque[str] = deque(maxlen=80)
    runner = HomrRunner(runner_logs.append)
    allowed_names = None
    if case_list is not None:
        allowed_names = {
            line.strip()
            for line in case_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    selected = _stable_sample(dataset, limit, seed, allowed_names)
    started = time.monotonic()

    metadata: dict[str, object] = {
        "application_version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "dataset": str(dataset.resolve()),
        "dataset_name": "PrIMuS",
        "dataset_scope": "synthetic monophonic printed incipits",
        "seed": seed,
        "requested_cases": limit,
        "selected_cases": len(selected),
        "case_list": str(case_list.resolve()) if case_list is not None else None,
        "case_list_sha256": sha256_file(case_list) if case_list is not None else None,
        "homr_version": importlib.metadata.version("homr"),
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": utc_now_iso(),
    }

    for index, semantic_path in enumerate(selected, start=1):
        case_started = time.monotonic()
        case_id = _case_id(semantic_path)
        case_root = cases_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        source_image = semantic_path.with_suffix(".png")
        work_image = case_root / "source.png"
        report: dict[str, object] = {
            "case": case_id,
            "semantic_path": str(semantic_path.relative_to(dataset)),
            "source_sha256": sha256_file(source_image) if source_image.exists() else None,
        }
        try:
            if not source_image.exists():
                raise FileNotFoundError(f"missing source image: {source_image}")
            if not work_image.exists() or sha256_file(work_image) != report["source_sha256"]:
                shutil.copy2(source_image, work_image)
            result = runner.run_page(work_image, threading.Event())
            report["engine_seconds"] = round(result.elapsed_seconds, 3)
            report["engine_return_code"] = result.return_code
            if result.return_code != 0 or result.xml_path is None:
                raise RuntimeError(result.error or f"homr returned {result.return_code}")
            reference = parse_primus_semantic(semantic_path)
            candidate = parse_musicxml_semantics(result.xml_path)
            report.update(compare_primus_semantics(reference, candidate))
            report["candidate_sha256"] = sha256_file(result.xml_path)
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            report["engine_log_tail"] = list(runner_logs)[-20:]
        report["elapsed_seconds"] = round(time.monotonic() - case_started, 3)
        reports.append(report)
        partial = {
            "metadata": metadata,
            "aggregate": aggregate_primus_reports(reports),
            "cases": reports,
        }
        atomic_write_json(output / "report.json", partial)
        status = "ERROR" if report.get("error") else (
            "EXACT" if report.get("semantic_exact") else f"EER={float(report['event_error_rate']):.3f}"
        )
        print(f"[{index:03d}/{len(selected):03d}] {case_id}: {status} ({report['elapsed_seconds']}s)", flush=True)

    metadata["completed_at"] = utc_now_iso()
    metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
    final = {
        "metadata": metadata,
        "aggregate": aggregate_primus_reports(reports),
        "cases": reports,
    }
    atomic_write_json(output / "report.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic homr-vs-PrIMuS semantic accuracy benchmark."
    )
    parser.add_argument("dataset", type=Path, help="Extracted PrIMuS root containing package_* folders")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", default="scorescan-public-baseline-v1")
    parser.add_argument(
        "--case-list",
        type=Path,
        help="Optional newline-delimited allowlist, such as the official held-out test fold.",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    result = run_benchmark(
        args.dataset,
        args.output,
        args.limit,
        args.seed,
        args.case_list,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
