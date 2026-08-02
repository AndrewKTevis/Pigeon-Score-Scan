from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# The audit is scalar and gains nothing from a process-wide BLAS worker pool.
# Bound implicit workers before importing MusicXML/layout modules so the CLI
# remains usable while the serial GPU training process owns most system RAM.
for _thread_limit_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_limit_variable, "1")

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scorescan.config import APP_VERSION, WORKFLOW_VERSION
from scorescan.measure_localized import (
    splice_measure_candidate,
    validate_measure_localized_context,
)
from scorescan.musicxml import MUSICXML_DOCTYPE
from scorescan.policy import DEFAULT_POLICY
from scorescan.util import atomic_write_json


def _write_score(
    path: Path,
    *,
    step: str = "C",
    clef: tuple[str, int] | None = None,
    fifths: int | None = None,
    time: tuple[int, int] | None = (4, 4),
    transpose: tuple[int, int, int, bool] | None = None,
    context_after_note: bool = False,
) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    leading = etree.SubElement(measure, "attributes")
    etree.SubElement(leading, "divisions").text = "1"

    def add_context(attributes: etree._Element) -> None:
        if clef is not None:
            node = etree.SubElement(attributes, "clef")
            etree.SubElement(node, "sign").text = clef[0]
            etree.SubElement(node, "line").text = str(clef[1])
        if fifths is not None:
            node = etree.SubElement(attributes, "key")
            etree.SubElement(node, "fifths").text = str(fifths)
        if time is not None:
            node = etree.SubElement(attributes, "time")
            etree.SubElement(node, "beats").text = str(time[0])
            etree.SubElement(node, "beat-type").text = str(time[1])
        if transpose is not None:
            diatonic, chromatic, octave_change, doubled = transpose
            node = etree.SubElement(attributes, "transpose")
            etree.SubElement(node, "diatonic").text = str(diatonic)
            etree.SubElement(node, "chromatic").text = str(chromatic)
            etree.SubElement(node, "octave-change").text = str(octave_change)
            if doubled:
                etree.SubElement(node, "double")

    if not context_after_note:
        add_context(leading)
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = step
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = "4"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "whole"
    if context_after_note:
        later = etree.SubElement(measure, "attributes")
        add_context(later)
    etree.ElementTree(root).write(
        str(path),
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
    )


def _scenario(
    root: Path,
    name: str,
    *,
    template: dict[str, object],
    localized: dict[str, object],
    expected: bool,
    error_contains: str | None = None,
    splice: bool = False,
) -> dict[str, object]:
    template_path = root / f"{name}_template.musicxml"
    local_path = root / f"{name}_local.musicxml"
    output_path = root / f"{name}_candidate.musicxml"
    _write_score(template_path, **template)
    _write_score(local_path, step="E", **localized)
    valid, error = validate_measure_localized_context(local_path, template_path, 1)
    splice_ok: bool | None = None
    splice_error: str | None = None
    if splice:
        try:
            splice_measure_candidate(template_path, local_path, 1, output_path)
            splice_ok = output_path.is_file()
        except Exception as exc:  # audit records failure instead of hiding it
            splice_ok = False
            splice_error = str(exc)
    passed = valid is expected
    if error_contains is not None:
        passed = passed and error_contains in str(error)
    if splice:
        passed = passed and splice_ok is expected
    return {
        "name": name,
        "expected_valid": expected,
        "valid": valid,
        "error": error,
        "error_contains": error_contains,
        "splice_checked": splice,
        "splice_ok": splice_ok,
        "splice_error": splice_error,
        "passed": passed,
    }


def evaluate() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="scorescan-context-audit-") as directory:
        root = Path(directory)
        nondefault = {
            "clef": ("F", 4),
            "fifths": -2,
            "time": (3, 4),
            "transpose": (-1, -2, 0, False),
        }
        scenarios = [
            _scenario(
                root,
                "implicit-default",
                template={},
                localized={},
                expected=True,
                splice=True,
            ),
            _scenario(
                root,
                "explicit-default-local",
                template={},
                localized={
                    "clef": ("G", 2),
                    "fifths": 0,
                    "transpose": (0, 0, 0, False),
                },
                expected=True,
            ),
            _scenario(
                root,
                "matching-nondefault",
                template=nondefault,
                localized=nondefault,
                expected=True,
                splice=True,
            ),
            _scenario(
                root,
                "conflicting-clef",
                template={},
                localized={"clef": ("F", 4)},
                expected=False,
                error_contains="clef context",
            ),
            _scenario(
                root,
                "conflicting-key",
                template={},
                localized={"fifths": 4},
                expected=False,
                error_contains="key context",
            ),
            _scenario(
                root,
                "conflicting-time",
                template={},
                localized={"time": (6, 8)},
                expected=False,
                error_contains="time context",
            ),
            _scenario(
                root,
                "conflicting-transpose",
                template={},
                localized={"transpose": (1, 2, 0, False)},
                expected=False,
                error_contains="transpose context",
            ),
            _scenario(
                root,
                "missing-nondefault-clef",
                template={"clef": ("F", 4)},
                localized={},
                expected=False,
                error_contains="clef context is missing",
            ),
            _scenario(
                root,
                "local-mid-measure-context",
                template={},
                localized={"clef": ("G", 2), "context_after_note": True},
                expected=False,
                error_contains="after performed content",
            ),
            _scenario(
                root,
                "template-mid-measure-context",
                template={"clef": ("G", 2), "context_after_note": True},
                localized={"clef": ("G", 2)},
                expected=False,
                error_contains="mid-measure attributes",
            ),
        ]
    return {
        "version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "policy_version": DEFAULT_POLICY.version,
        "contract": "measure-localized-notation-context@1",
        "scenario_count": len(scenarios),
        "passed_count": sum(bool(item["passed"]) for item in scenarios),
        "passed": all(bool(item["passed"]) for item in scenarios),
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate()
    atomic_write_json(args.output, report)
    print(args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
