from __future__ import annotations

"""Deterministic audit of exact splice-content consensus for local measure rescue."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scorescan.measure_localized import (  # noqa: E402
    measure_localized_content_signature,
    measure_localized_semantic_signature,
)
from scorescan.musicxml import MUSICXML_DOCTYPE  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402


def _write(path: Path, *, divisions: int = 1, duration: int = 1) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "E"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = str(duration)
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "quarter"
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE
    )


def _mutate(path: Path, kind: str) -> None:
    tree = etree.parse(str(path))
    note = tree.find("./part/measure/note")
    measure = tree.find("./part/measure")
    assert note is not None and measure is not None
    if kind == "beam-begin":
        etree.SubElement(note, "beam", number="1").text = "begin"
    elif kind == "beam-end":
        etree.SubElement(note, "beam", number="1").text = "end"
    elif kind == "stem-up":
        etree.SubElement(note, "stem").text = "up"
    elif kind == "fermata":
        notations = etree.SubElement(note, "notations")
        etree.SubElement(notations, "fermata").text = "normal"
    elif kind == "technical":
        notations = etree.SubElement(note, "notations")
        technical = etree.SubElement(notations, "technical")
        etree.SubElement(technical, "fingering").text = "2"
    elif kind == "layout-left":
        note.set("default-x", "17")
        note.set("relative-y", "9")
        note.set("color", "#000000")
    elif kind == "layout-right":
        note.set("default-x", "91")
        note.set("relative-y", "33")
        note.set("color", "#123456")
    elif kind == "mid-divisions":
        later = etree.Element("attributes")
        etree.SubElement(later, "divisions").text = "4"
        measure.insert(measure.index(note) + 1, later)
    else:
        raise ValueError(kind)
    tree.write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def audit() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="scorescan-local-exact-") as raw:
        root = Path(raw)

        def pair(name: str, left_kind: str | None, right_kind: str | None, *, equivalent: bool) -> None:
            left = root / f"{name}_left.musicxml"
            right = root / f"{name}_right.musicxml"
            _write(left)
            _write(right)
            if left_kind:
                _mutate(left, left_kind)
            if right_kind:
                _mutate(right, right_kind)
            semantic_equal = (
                measure_localized_semantic_signature(left)
                == measure_localized_semantic_signature(right)
            )
            exact_equal = (
                measure_localized_content_signature(left)
                == measure_localized_content_signature(right)
            )
            rows.append(
                {
                    "name": name,
                    "expected_exact_equal": equivalent,
                    "semantic_equal": semantic_equal,
                    "exact_equal": exact_equal,
                    "passed": exact_equal is equivalent,
                }
            )

        pair("layout-only", "layout-left", "layout-right", equivalent=True)
        pair("beam-topology", "beam-begin", "beam-end", equivalent=False)
        pair("stem-topology", "stem-up", None, equivalent=False)
        pair("unmodelled-notation", "fermata", "technical", equivalent=False)

        div1 = root / "div1.musicxml"
        div4 = root / "div4.musicxml"
        _write(div1, divisions=1, duration=1)
        _write(div4, divisions=4, duration=4)
        rows.append(
            {
                "name": "equivalent-divisions",
                "expected_exact_equal": True,
                "semantic_equal": measure_localized_semantic_signature(div1)
                == measure_localized_semantic_signature(div4),
                "exact_equal": measure_localized_content_signature(div1)
                == measure_localized_content_signature(div4),
                "passed": measure_localized_content_signature(div1)
                == measure_localized_content_signature(div4),
            }
        )

        invalid = root / "mid_divisions.musicxml"
        _write(invalid, divisions=2, duration=2)
        _mutate(invalid, "mid-divisions")
        rejected = False
        error = None
        try:
            measure_localized_content_signature(invalid)
        except ValueError as exc:
            rejected = True
            error = str(exc)
        rows.append(
            {
                "name": "mid-measure-divisions",
                "expected_rejected": True,
                "rejected": rejected,
                "error": error,
                "passed": rejected,
            }
        )

    passed = sum(bool(row["passed"]) for row in rows)
    return {
        "format": 2,
        "policy_version": DEFAULT_POLICY.version,
        "scenario_count": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "permission_signature": "normalized-splice-content-c14n-v1",
        "diagnostic_signature": "score-ir-semantic-v1",
        "scenarios": rows,
        "accepted_capabilities": {
            "equivalent_divisions_normalized": True,
            "layout_coordinates_ignored": True,
            "beam_and_stem_disagreement_detected": True,
            "unmodelled_notation_disagreement_detected": True,
            "mid_measure_divisions_rejected": True,
        },
        "programmatic_audit": True,
        "end_to_end_accuracy_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    if int(report["failed"]) != 0:
        raise SystemExit("measure-localised exact content audit failed")
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
