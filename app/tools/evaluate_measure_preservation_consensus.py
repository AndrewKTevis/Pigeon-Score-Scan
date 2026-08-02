from __future__ import annotations

"""Audit full MusicXML preservation agreement for whole-measure consensus.

The historical exact signature covered the core note timeline but not every object copied
when a complete measure replaced the template.  This audit keeps that old contract as a
fixed baseline and verifies that the canonical preservation signature:

* keeps representational/layout equivalence;
* separates beam, notehead, fermata, lyric and unknown-notation conflicts;
* prevents three independent families with three different write-back surfaces from
  manufacturing an exact majority;
* still permits semantic consensus when two families agree on the full selected surface.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.config import APP_VERSION, WORKFLOW_VERSION  # noqa: E402
from scorescan.musicxml_signature import measure_preservation_signature  # noqa: E402
from scorescan.policy import DEFAULT_POLICY  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402


def _text(value: str | None) -> str:
    return (value or "").strip()


def _legacy_signature(measure: etree._Element) -> str:
    """Reproduce the pre-0.27 simplified exact signature."""

    def pitch(note: etree._Element) -> tuple[object, ...]:
        if note.find("rest") is not None:
            return ("rest",)
        item = note.find("pitch")
        if item is None:
            return ("unknown",)
        return (
            "pitch",
            _text(item.findtext("step")),
            _text(item.findtext("alter")) or "0",
            _text(item.findtext("octave")),
        )

    notes: list[tuple[object, ...]] = []
    for note in measure.findall("note"):
        notes.append(
            (
                pitch(note),
                _text(note.findtext("duration")),
                _text(note.findtext("voice")) or "1",
                _text(note.findtext("type")),
                len(note.findall("dot")),
                note.find("chord") is not None,
                note.find("grace") is not None,
                _text(note.findtext("accidental")),
                _text(note.findtext("stem")),
                tuple(sorted(_text(item.get("type")) for item in note.findall("tie"))),
                tuple(sorted(child.tag for child in note.findall("./notations/articulations/*"))),
                tuple(sorted(child.tag for child in note.findall("./notations/ornaments/*"))),
                tuple(sorted(child.tag for child in note.findall("./notations/technical/*"))),
            )
        )
    payload = repr(tuple(notes)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _measure(*, divisions: int = 1, duration: int = 1) -> etree._Element:
    measure = etree.Element("measure", number="1", width="800")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = str(divisions)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = "1"
    etree.SubElement(time, "beat-type").text = "4"
    note = etree.SubElement(measure, "note", **{"default-x": "25"})
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "C"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = str(duration)
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "quarter"
    return measure


def _clone(measure: etree._Element) -> etree._Element:
    return etree.fromstring(etree.tostring(measure))


def _note(measure: etree._Element) -> etree._Element:
    note = measure.find("note")
    assert note is not None
    return note


def _case(name: str, left: etree._Element, right: etree._Element, expected_equal: bool) -> dict[str, object]:
    legacy_equal = _legacy_signature(left) == _legacy_signature(right)
    preservation_equal = measure_preservation_signature(left) == measure_preservation_signature(right)
    return {
        "name": name,
        "expected_equal": expected_equal,
        "legacy_equal": legacy_equal,
        "preservation_equal": preservation_equal,
        "passed": preservation_equal == expected_equal,
        "legacy_false_collapse": bool(not expected_equal and legacy_equal),
    }


def run_audit() -> dict[str, object]:
    base = _measure()
    cases: list[dict[str, object]] = []

    divisions = _measure(divisions=4, duration=4)
    cases.append(_case("equivalent_divisions", base, divisions, True))

    layout = _clone(base)
    layout.set("number", "999")
    layout.set("width", "250")
    _note(layout).set("default-x", "900")
    cases.append(_case("layout_coordinates", base, layout, True))

    beam = _clone(base)
    etree.SubElement(_note(beam), "beam", number="1").text = "begin"
    cases.append(_case("beam_topology", base, beam, False))

    notehead = _clone(base)
    etree.SubElement(_note(notehead), "notehead", filled="no").text = "diamond"
    cases.append(_case("notehead_shape", base, notehead, False))

    fermata = _clone(base)
    notations = etree.SubElement(_note(fermata), "notations")
    etree.SubElement(notations, "fermata", type="upright").text = "normal"
    cases.append(_case("fermata", base, fermata, False))

    lyric = _clone(base)
    lyric_node = etree.SubElement(_note(lyric), "lyric", number="1")
    etree.SubElement(lyric_node, "text").text = "La"
    cases.append(_case("lyric_text", base, lyric, False))

    technical_a = _clone(base)
    technical_b = _clone(base)
    for measure, value in ((technical_a, "1"), (technical_b, "3")):
        notations = etree.SubElement(_note(measure), "notations")
        technical = etree.SubElement(notations, "technical")
        etree.SubElement(technical, "fingering").text = value
    cases.append(_case("technical_value", technical_a, technical_b, False))

    unknown = _clone(base)
    notations = etree.SubElement(_note(unknown), "notations")
    other = etree.SubElement(notations, "other-notation", type="single")
    other.text = "custom-mark"
    cases.append(_case("unknown_notation", base, unknown, False))

    three_way = []
    for kind in ("beam", "notehead", "fermata"):
        item = _clone(base)
        if kind == "beam":
            etree.SubElement(_note(item), "beam", number="1").text = "begin"
        elif kind == "notehead":
            etree.SubElement(_note(item), "notehead").text = "diamond"
        else:
            notations = etree.SubElement(_note(item), "notations")
            etree.SubElement(notations, "fermata").text = "normal"
        three_way.append(item)
    legacy_groups = len({_legacy_signature(item) for item in three_way})
    preservation_groups = len({measure_preservation_signature(item) for item in three_way})

    duplicate = _clone(three_way[0])
    two_family_support = max(
        [
            [measure_preservation_signature(item) for item in (*three_way, duplicate)].count(signature)
            for signature in {measure_preservation_signature(item) for item in (*three_way, duplicate)}
        ]
    )

    passed = (
        all(bool(item["passed"]) for item in cases)
        and legacy_groups == 1
        and preservation_groups == 3
        and two_family_support == DEFAULT_POLICY.selection_semantic_preservation_minimum_families
    )
    return {
        "format": 1,
        "version": APP_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "policy_version": DEFAULT_POLICY.version,
        "cases": cases,
        "legacy_false_collapse_count": sum(bool(item["legacy_false_collapse"]) for item in cases),
        "three_way_conflict": {
            "legacy_distinct_signatures": legacy_groups,
            "preservation_distinct_signatures": preservation_groups,
            "old_exact_majority_would_form": legacy_groups == 1,
            "new_exact_majority_would_form": preservation_groups == 1,
        },
        "two_family_preservation_support": {
            "support": two_family_support,
            "required": DEFAULT_POLICY.selection_semantic_preservation_minimum_families,
            "accepted": two_family_support
            >= DEFAULT_POLICY.selection_semantic_preservation_minimum_families,
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit()
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
