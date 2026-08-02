#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate geometric hairpin detection and source-to-MusicXML restoration.

MuseScore labels both real ``<``/``>`` wedges and textual ``cresc. - - -``
continuations as ``HairpinSegment``.  This evaluator deliberately treats only
undashed three-point wedges, plus the older two-line wedge representation, as
geometric oracle objects.  It then removes every MusicXML wedge transaction,
runs the production detector/resolver, and compares part/staff/measure/kind
events while proving that non-wedge semantics are unchanged.
"""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Iterable

from lxml import etree

from scorescan.layout import analyze_layout
from scorescan.musicxml import MUSICXML_DOCTYPE
from scorescan.notation_coverage import (
    DETECTOR_VERSION,
    VisualNotationCandidate,
    detect_notation_candidates,
    wedge_source_specificity_gate,
)
from scorescan.util import atomic_write_bytes, atomic_write_json, sha256_file
from scorescan.wedge_enrichment import (
    WEDGE_ENRICHMENT_VERSION,
    enrich_musicxml_with_wedges,
)
from app.tools.evaluate_source_beam_restoration import _export_reference


DEFAULT_PAIR_IDS = (34, 117, 187, 231, 261, 263, 266, 268, 890)
MINIMUM_ORACLE_PRECISION = 0.995
MINIMUM_ORACLE_RECALL = 0.995
MINIMUM_EVENT_PRECISION = 0.995
MINIMUM_EVENT_RECALL = 0.995
_PARSER = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
    recover=False,
    huge_tree=False,
)


@dataclass(frozen=True)
class OracleWedge:
    kind: str
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "bbox": list(self.bbox)}


def _parse_pair_ids(values: Iterable[str]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            pair_id = int(token)
            if pair_id < 0:
                raise ValueError("pair ids must be non-negative")
            if pair_id not in result:
                result.append(pair_id)
    if not result:
        raise ValueError("at least one pair id is required")
    return tuple(result)


def _points(value: str) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for token in value.replace("\n", " ").split():
        fields = token.split(",")
        if len(fields) != 2:
            raise ValueError(f"invalid SVG polyline point {token!r}")
        result.append((float(fields[0]), float(fields[1])))
    return tuple(result)


def _bbox(
    points: tuple[tuple[float, float], ...],
    stroke_width: float,
) -> tuple[int, int, int, int]:
    padding = max(2.0, stroke_width * 0.75)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        round(min(xs) - padding),
        round(min(ys) - padding),
        round(max(xs) + padding),
        round(max(ys) + padding),
    )


def _segment_y(
    segment: tuple[tuple[float, float], tuple[float, float]],
    x: float,
) -> float:
    (x1, y1), (x2, y2) = segment
    return y1 + (y2 - y1) * (x - x1) / max(x2 - x1, 1e-6)


def oracle_svg_wedges(svg_path: Path) -> tuple[OracleWedge, ...]:
    root = ET.parse(svg_path).getroot()
    wedges: list[OracleWedge] = []
    line_segments: list[
        tuple[
            tuple[tuple[float, float], tuple[float, float]],
            float,
        ]
    ] = []
    for element in root.iter():
        if (
            element.tag.rsplit("}", 1)[-1] != "polyline"
            or element.attrib.get("class") != "HairpinSegment"
            or element.attrib.get("stroke-dasharray")
        ):
            continue
        if element.attrib.get("transform"):
            raise ValueError(
                f"transformed hairpin polyline is unsupported: {svg_path}"
            )
        points = _points(element.attrib.get("points", ""))
        stroke_width = float(element.attrib.get("stroke-width", "1"))
        if len(points) == 3:
            open_x = 0.5 * (points[0][0] + points[2][0])
            apex_x = points[1][0]
            if abs(open_x - apex_x) < 1.0:
                continue
            kind = "crescendo" if open_x > apex_x else "diminuendo"
            wedges.append(OracleWedge(kind, _bbox(points, stroke_width)))
        elif len(points) == 2:
            left, right = points
            if right[0] < left[0]:
                left, right = right, left
            if right[0] - left[0] > 0:
                line_segments.append(((left, right), stroke_width))

    used: set[int] = set()
    for left_index, (first, first_width) in enumerate(line_segments):
        if left_index in used:
            continue
        first_slope = (
            first[1][1] - first[0][1]
        ) / max(first[1][0] - first[0][0], 1e-6)
        best: tuple[float, int, str, tuple[int, int, int, int]] | None = None
        for right_index in range(left_index + 1, len(line_segments)):
            if right_index in used:
                continue
            second, second_width = line_segments[right_index]
            second_slope = (
                second[1][1] - second[0][1]
            ) / max(second[1][0] - second[0][0], 1e-6)
            if first_slope * second_slope >= 0:
                continue
            overlap_left = max(first[0][0], second[0][0])
            overlap_right = min(first[1][0], second[1][0])
            overlap = overlap_right - overlap_left
            shorter = min(
                first[1][0] - first[0][0],
                second[1][0] - second[0][0],
            )
            if overlap <= 0 or overlap / max(shorter, 1.0) < 0.8:
                continue
            separation_left = abs(
                _segment_y(first, overlap_left)
                - _segment_y(second, overlap_left)
            )
            separation_right = abs(
                _segment_y(first, overlap_right)
                - _segment_y(second, overlap_right)
            )
            if abs(separation_left - separation_right) < 2.0:
                continue
            kind = (
                "crescendo"
                if separation_left < separation_right
                else "diminuendo"
            )
            box = _bbox(
                (
                    (overlap_left, _segment_y(first, overlap_left)),
                    (overlap_right, _segment_y(first, overlap_right)),
                    (overlap_left, _segment_y(second, overlap_left)),
                    (overlap_right, _segment_y(second, overlap_right)),
                ),
                max(first_width, second_width),
            )
            score = abs(separation_left - separation_right)
            candidate = (score, right_index, kind, box)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            continue
        _score, right_index, kind, box = best
        used.update((left_index, right_index))
        wedges.append(OracleWedge(kind, box))
    return tuple(
        sorted(wedges, key=lambda item: (item.bbox[1], item.bbox[0], item.kind))
    )


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / max(1, left_area + right_area - intersection)


def _oracle_matches(
    oracle: tuple[OracleWedge, ...],
    candidates: tuple[VisualNotationCandidate, ...],
) -> int:
    ranked = sorted(
        (
            (_bbox_iou(reference.bbox, candidate.bbox), left, right)
            for left, reference in enumerate(oracle)
            for right, candidate in enumerate(candidates)
            if reference.kind == candidate.kind
        ),
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches = 0
    for score, left, right in ranked:
        if score < 0.25:
            break
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        matches += 1
    return matches


def _strip_wedges(reference: Path, output: Path) -> None:
    tree = etree.parse(str(reference), _PARSER)
    root = tree.getroot()
    for direction in list(root.findall("./part/measure/direction")):
        direction_type = direction.find("direction-type")
        if direction_type is None or direction_type.find("wedge") is None:
            continue
        for wedge in list(direction_type.findall("wedge")):
            direction_type.remove(wedge)
        if len(direction_type) == 0:
            direction.getparent().remove(direction)
    atomic_write_bytes(
        output,
        etree.tostring(
            tree,
            encoding="UTF-8",
            xml_declaration=True,
            doctype=MUSICXML_DOCTYPE,
        ),
    )


def _wedge_free_c14n(path: Path) -> bytes:
    tree = etree.parse(str(path), _PARSER)
    root = deepcopy(tree.getroot())
    for direction in list(root.findall("./part/measure/direction")):
        direction_type = direction.find("direction-type")
        if direction_type is None or direction_type.find("wedge") is None:
            continue
        direction.getparent().remove(direction)
    return etree.tostring(root, method="c14n")


def _wedge_events(path: Path) -> Counter[tuple[int, int, int, str]]:
    root = etree.parse(str(path), _PARSER).getroot()
    result: Counter[tuple[int, int, int, str]] = Counter()
    for part_index, part in enumerate(root.findall("part")):
        for measure_index, measure in enumerate(part.findall("measure")):
            for direction in measure.findall("direction"):
                staff = int(direction.findtext("staff") or "1")
                for wedge in direction.findall("direction-type/wedge"):
                    result[
                        (
                            part_index,
                            measure_index,
                            staff,
                            str(wedge.get("type") or ""),
                        )
                    ] += 1
    return result


def _counter_matches(
    reference: Counter[tuple[int, int, int, str]],
    candidate: Counter[tuple[int, int, int, str]],
) -> int:
    return sum(
        min(count, candidate.get(key, 0))
        for key, count in reference.items()
    )


def evaluate_source_wedges(
    *,
    dataset_dir: Path,
    prepared_dir: Path,
    output_dir: Path,
    musescore: Path,
    pair_ids: tuple[int, ...],
    timeout_seconds: int = 180,
) -> dict[str, object]:
    catalog = json.loads(
        (dataset_dir / "benchmark_dataset.json").read_text(encoding="utf-8")
    )
    if not isinstance(catalog, dict):
        raise ValueError("benchmark dataset catalog is invalid")
    musescore_sha256 = sha256_file(musescore)
    references = output_dir / "references"
    candidates_dir = output_dir / "candidates"
    references.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []

    for pair_id in pair_ids:
        pair_name = f"pair-{pair_id:04d}"
        entry = catalog.get(str(pair_id))
        if not isinstance(entry, dict) or not isinstance(entry.get("score"), str):
            raise ValueError(f"{pair_name} is absent from the dataset catalog")
        source = dataset_dir / entry["score"]
        image = prepared_dir / "pages" / pair_name / "page-1.jpg"
        svg = prepared_dir / "reference_pages" / pair_name / "page-1.svg"
        if not source.is_file() or not image.is_file() or not svg.is_file():
            raise FileNotFoundError(f"{pair_name} source, scan, or SVG is missing")
        reference = references / f"{pair_name}.musicxml"
        candidate = candidates_dir / f"{pair_name}.musicxml"
        _export_reference(
            source,
            reference,
            musescore=musescore,
            musescore_sha256=musescore_sha256,
            timeout_seconds=timeout_seconds,
        )
        _strip_wedges(reference, candidate)
        before_non_wedge = _wedge_free_c14n(candidate)
        layout = analyze_layout(image)
        detected = detect_notation_candidates(image, layout)
        eligible = tuple(
            item
            for item in detected
            if item.kind in {"crescendo", "diminuendo"}
            and wedge_source_specificity_gate(item, detected)[0]
        )
        oracle = oracle_svg_wedges(svg)
        oracle_matches = _oracle_matches(oracle, eligible)
        report = enrich_musicxml_with_wedges(
            image,
            candidate,
            layout,
            candidates=detected,
        )
        non_wedge_exact = before_non_wedge == _wedge_free_c14n(candidate)
        reference_events = _wedge_events(reference)
        candidate_events = _wedge_events(candidate)
        event_matches = _counter_matches(reference_events, candidate_events)
        reference_start_count = sum(
            count
            for key, count in reference_events.items()
            if key[3] != "stop"
        )
        reference_stop_count = sum(
            count
            for key, count in reference_events.items()
            if key[3] == "stop"
        )
        candidate_start_count = sum(
            count
            for key, count in candidate_events.items()
            if key[3] != "stop"
        )
        candidate_stop_count = sum(
            count
            for key, count in candidate_events.items()
            if key[3] == "stop"
        )
        cases.append(
            {
                "pair_id": pair_id,
                "source_sha256": sha256_file(source),
                "scan_sha256": sha256_file(image),
                "reference_sha256": sha256_file(reference),
                "candidate_sha256": sha256_file(candidate),
                "oracle_wedges": [item.to_dict() for item in oracle],
                "oracle_wedge_count": len(oracle),
                "eligible_candidate_count": len(eligible),
                "oracle_match_count": oracle_matches,
                "reference_event_count": sum(reference_events.values()),
                "candidate_event_count": sum(candidate_events.values()),
                "event_match_count": event_matches,
                "reference_transaction_balance": (
                    reference_start_count == reference_stop_count
                ),
                "candidate_transaction_balance": (
                    candidate_start_count == candidate_stop_count
                ),
                "non_wedge_preservation_exact": non_wedge_exact,
                "transaction_committed": report.transaction_committed,
                "injected_wedge_count": report.injected_count,
                "abstention_reasons": dict(
                    Counter(
                        proposal.reason
                        for proposal in report.proposals
                        if not proposal.injected
                    )
                ),
            }
        )

    oracle_total = sum(int(case["oracle_wedge_count"]) for case in cases)
    candidate_total = sum(
        int(case["eligible_candidate_count"]) for case in cases
    )
    oracle_matches = sum(int(case["oracle_match_count"]) for case in cases)
    reference_events = sum(
        int(case["reference_event_count"]) for case in cases
    )
    candidate_events = sum(
        int(case["candidate_event_count"]) for case in cases
    )
    event_matches = sum(int(case["event_match_count"]) for case in cases)

    def rate(numerator: int, denominator: int) -> float:
        return 1.0 if denominator == 0 else numerator / denominator

    oracle_precision = rate(oracle_matches, candidate_total)
    oracle_recall = rate(oracle_matches, oracle_total)
    event_precision = rate(event_matches, candidate_events)
    event_recall = rate(event_matches, reference_events)
    gates = {
        "minimum_oracle_precision": (
            oracle_precision >= MINIMUM_ORACLE_PRECISION
        ),
        "minimum_oracle_recall": oracle_recall >= MINIMUM_ORACLE_RECALL,
        "minimum_event_precision": event_precision >= MINIMUM_EVENT_PRECISION,
        "minimum_event_recall": event_recall >= MINIMUM_EVENT_RECALL,
        "all_non_wedge_semantics_preserved": all(
            bool(case["non_wedge_preservation_exact"])
            for case in cases
        ),
        "all_wedge_transactions_balanced": all(
            bool(case["reference_transaction_balance"])
            and bool(case["candidate_transaction_balance"])
            for case in cases
        ),
    }
    payload: dict[str, object] = {
        "format": 1,
        "evaluation_kind": "oracle-source-wedge-restoration",
        "detector_version": DETECTOR_VERSION,
        "enrichment_version": WEDGE_ENRICHMENT_VERSION,
        "pair_ids": list(pair_ids),
        "aggregate": {
            "case_count": len(cases),
            "oracle_wedge_count": oracle_total,
            "eligible_candidate_count": candidate_total,
            "oracle_match_count": oracle_matches,
            "oracle_precision": oracle_precision,
            "oracle_recall": oracle_recall,
            "reference_event_count": reference_events,
            "candidate_event_count": candidate_events,
            "event_match_count": event_matches,
            "event_precision": event_precision,
            "event_recall": event_recall,
        },
        "gates": gates,
        "gate_passed": all(gates.values()),
        "cases": cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "source-wedge-restoration-report.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--musescore-exe", type=Path, required=True)
    parser.add_argument("--pairs", action="append")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    pair_ids = (
        _parse_pair_ids(args.pairs)
        if args.pairs
        else DEFAULT_PAIR_IDS
    )
    payload = evaluate_source_wedges(
        dataset_dir=args.dataset_dir.resolve(),
        prepared_dir=args.prepared_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        musescore=args.musescore_exe.resolve(),
        pair_ids=pair_ids,
        timeout_seconds=max(1, args.timeout_seconds),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not args.gate or bool(payload["gate_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
