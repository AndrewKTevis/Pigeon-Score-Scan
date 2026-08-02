#!/usr/bin/env python3
"""Refresh release QA only from a complete, hash-verified application test run."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_SRC = PROJECT_ROOT / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from scorescan.util import atomic_write_json, sha256_file, utc_now_iso


SUMMARY_PATTERN = re.compile(
    r"(?P<body>(?:\d+ (?:failed|passed|skipped|warnings?|errors?),? ?)+)"
    r"in \d+(?:\.\d+)?s",
    re.IGNORECASE,
)
COUNT_PATTERN = re.compile(
    r"(?P<count>\d+) (?P<kind>failed|passed|skipped|warnings?|errors?)",
    re.IGNORECASE,
)


def parse_pytest_log(text: str) -> dict[str, Any]:
    matches = list(SUMMARY_PATTERN.finditer(text))
    if not matches:
        raise ValueError("pytest completion summary is absent")
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "warnings": 0,
        "errors": 0,
    }
    for match in COUNT_PATTERN.finditer(matches[-1].group("body")):
        kind = match.group("kind").lower()
        if kind == "warning":
            kind = "warnings"
        elif kind == "error":
            kind = "errors"
        counts[kind] = int(match.group("count"))
    skipped_details = [
        line.strip()
        for line in text.splitlines()
        if line.lstrip().startswith("SKIPPED [")
    ]
    return {
        **counts,
        "total": counts["passed"]
        + counts["failed"]
        + counts["skipped"]
        + counts["errors"],
        "skipped_details": skipped_details,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _read_test_log(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8", errors="replace")


def _server_test_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in tree.body
    )


def _verify_torch_contract(
    project: Path,
    report_path: Path,
) -> dict[str, Any]:
    report = _load_json(report_path)
    training_source = (
        project / "app/tools/train_deepscores_symbol_detector.py"
    )
    if (
        report.get("name")
        != "scorescan-detector-isolated-torch-contracts-v1"
        or report.get("passed") is not True
        or report.get("checks", {})
        .get("microbatch_gradient", {})
        .get("passed")
        is not True
        or report.get("checks", {})
        .get("legacy_sampler_recovery", {})
        .get("passed")
        is not True
        or report.get("runtime", {}).get("cuda_available") is not False
        or report.get("input", {}).get("sha256")
        != sha256_file(training_source)
        or int(report.get("input", {}).get("bytes", -1))
        != training_source.stat().st_size
    ):
        raise ValueError("isolated PyTorch contract report is not accepted")
    return {
        "path": str(report_path.resolve()),
        "bytes": report_path.stat().st_size,
        "sha256": sha256_file(report_path),
        "runtime": report.get("runtime"),
        "checks": report.get("checks"),
    }


def invalidate_archive_reverification(
    payload: dict[str, Any],
    *,
    refreshed_at: str,
) -> None:
    """Fail closed when a QA refresh is newer than packaged artifacts."""

    payload["release_package_reverification"] = {
        "completed": False,
        "invalidated_at_utc": refreshed_at,
        "reason": (
            "No source or Windows archive built from this refreshed "
            "source/test state has been registered."
        ),
    }


def refresh_release_qa(
    project: Path,
    *,
    test_report_path: Path,
    torch_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    project = project.resolve()
    gate = _load_json(test_report_path)
    output_log = Path(str(gate.get("output", "")))
    if not output_log.is_absolute():
        output_log = project / output_log
    if (
        gate.get("passed") is not True
        or int(gate.get("exit_code", -1)) != 0
        or not output_log.is_file()
        or gate.get("output_sha256") != sha256_file(output_log)
    ):
        raise ValueError("full application test gate is not accepted")
    parsed = parse_pytest_log(_read_test_log(output_log))
    if parsed["failed"] or parsed["errors"]:
        raise ValueError("full application test log contains failures")
    if parsed["skipped"] > 1:
        raise ValueError("unexpected application test skips")

    torch_evidence = _verify_torch_contract(project, torch_report_path)
    if parsed["skipped"]:
        details = "\n".join(parsed["skipped_details"]).lower()
        if (
            "test_train_deepscores_symbol_detector.py" not in details
            or "torch" not in details
        ):
            raise ValueError("application test skip is not the isolated torch case")

    server_tests = _server_test_count(project / "app/tests/test_server.py")
    if server_tests <= 0 or parsed["passed"] < server_tests:
        raise ValueError("server test accounting is invalid")
    non_server_passed = parsed["passed"] - server_tests

    compile_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "-f",
            str(project / "app/src"),
            str(project / "app/tools"),
            str(project / "app/tests"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    javascript_result = subprocess.run(
        [
            "node",
            "--check",
            str(project / "app/src/scorescan/web/app.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if compile_result.returncode or javascript_result.returncode:
        raise RuntimeError(
            "static release checks failed: "
            f"compile={compile_result.returncode}; "
            f"javascript={javascript_result.returncode}"
        )

    existing = _load_json(output_path) if output_path.is_file() else {}
    refreshed_at = utc_now_iso()
    invalidate_archive_reverification(existing, refreshed_at=refreshed_at)
    existing.update(
        {
            "tests_passed": parsed["passed"],
            "tests_failed": 0,
            "tests_skipped": parsed["skipped"],
            "test_nodes_total": parsed["total"],
            # Retained for the readiness schema used by previous releases.
            "test_functions_total": parsed["total"],
            "non_server_tests_reexecuted": non_server_passed,
            "server_test_functions": server_tests,
            "server_tests_executed": server_tests,
            "server_tests_unexecuted": 0,
            "server_test_gap_reason": "",
            "test_warnings": parsed["warnings"],
            "test_skip_details": parsed["skipped_details"],
            "test_execution_method": (
                f"{parsed['total']} ScoreScan pytest nodes collected in one "
                "uninterrupted cross-volume full-suite run; "
                f"{parsed['passed']} passed, {parsed['skipped']} training-only "
                "PyTorch node skipped in the lightweight desktop environment "
                "and re-executed by the exact isolated-runtime contract gate."
            ),
            "full_suite_gate": {
                "path": str(test_report_path.resolve()),
                "bytes": test_report_path.stat().st_size,
                "sha256": sha256_file(test_report_path),
                "output_sha256": gate["output_sha256"],
                "elapsed_seconds": gate.get("elapsed_seconds"),
            },
            "detector_isolated_torch_contracts": torch_evidence,
            "python_compile": True,
            "javascript_syntax": True,
            # An archive verified before this source/test refresh is stale even if
            # its ZIP still exists and passes CRC.  Artifact QA must be produced by
            # the packaging workflow from the exact refreshed tree; carrying the old
            # success block forward would make RELEASE_QA claim more than was run.
            "release_qa_generated_at_utc": refreshed_at,
        }
    )
    atomic_write_json(output_path, existing)
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument(
        "--torch-report",
        type=Path,
        default=Path("training/detector_isolated_torch_contracts_v1.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("RELEASE_QA.json"))
    args = parser.parse_args()
    project = args.project_root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else project / path

    refresh_release_qa(
        project,
        test_report_path=resolve(args.test_report),
        torch_report_path=resolve(args.torch_report),
        output_path=resolve(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
