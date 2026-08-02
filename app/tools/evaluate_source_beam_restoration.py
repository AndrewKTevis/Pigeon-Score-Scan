from __future__ import annotations

"""Evaluate source-to-MusicXML beam restoration with exact registered boxes.

This is a resolver/layout test, not a detector benchmark.  Strictly registered
Muse OMR scan labels provide oracle beam boxes on the source scan.  Every beam is
removed from the matching MuseScore MusicXML, restored by the production resolver,
and compared at the original note event.  The resulting precision and recall expose
mapping/assignment errors without conflating them with detector misses.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

from lxml import etree

from scorescan.beam_enrichment import (
    BEAM_ENRICHMENT_VERSION,
    enrich_musicxml_with_source_beams,
)
from scorescan.evaluation import compare_musicxml
from scorescan.layout import analyze_layout
from scorescan.musicxml import MUSICXML_DOCTYPE, validate_musicxml
from scorescan.semantic_detector import (
    SemanticDetection,
    assign_bbox_to_staff,
)
from scorescan.util import atomic_write_bytes, atomic_write_json, sha256_file
from scorescan.wedge_enrichment import _build_topology
from app.tools.prepare_muse_omr_benchmark import analyze_reference_boundary
from app.tools.prepare_openscore_svg_regions import (
    _render_score,
    semantic_svg_objects,
)


DEFAULT_PAIR_IDS = (
    34,
    51,
    84,
    117,
    149,
    187,
    188,
    189,
    230,
    231,
    261,
    263,
    266,
    268,
    274,
    306,
)
MINIMUM_ORACLE_PRECISION = 0.995
MINIMUM_ORACLE_RECALL = 0.95
MINIMUM_REFERENCE_MARKERS = 500
MINIMUM_CASES = 12
CROSS_STAFF_MINIMUM_REFERENCE_MARKERS = 50
CROSS_STAFF_MINIMUM_CASES = 2
_PARSER = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
    recover=False,
    huge_tree=False,
)


def _parse_pair_ids(values: Iterable[str]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        for token in value.split(","):
            stripped = token.strip()
            if not stripped:
                continue
            pair_id = int(stripped)
            if pair_id < 0:
                raise ValueError("pair ids must be non-negative")
            if pair_id not in result:
                result.append(pair_id)
    if not result:
        raise ValueError("at least one pair id is required")
    return tuple(result)


def _parse_page_cases(values: Iterable[str]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for value in values:
        for token in value.split(","):
            stripped = token.strip()
            if not stripped:
                continue
            pair_text, separator, page_text = stripped.partition(":")
            if not separator:
                raise ValueError("page cases must use PAIR:PAGE syntax")
            pair_id = int(pair_text)
            page_number = int(page_text)
            if pair_id < 0 or page_number <= 0:
                raise ValueError("pair ids must be non-negative and pages positive")
            case = (pair_id, page_number)
            if case not in result:
                result.append(case)
    if not result:
        raise ValueError("at least one page case is required")
    return tuple(result)


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


def oracle_category_boxes(
    acceptance: dict[str, Any],
    category_id: str,
    *,
    page_number: int | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    """Reassemble registered full-page semantic boxes from overlapping tiles."""

    boxes: list[tuple[int, int, int, int]] = []
    rows = acceptance.get("rows")
    if not isinstance(rows, list):
        raise ValueError("acceptance rows are missing")
    for row in rows:
        if page_number is not None:
            image = Path(str(row.get("image") or "")).name
            if image != f"page-{page_number}.jpg":
                continue
        crop = row.get("crop_xyxy")
        objects = row.get("objects")
        if (
            not isinstance(crop, list)
            or len(crop) != 4
            or not isinstance(objects, list)
        ):
            raise ValueError("acceptance tile geometry is invalid")
        offset_x, offset_y = float(crop[0]), float(crop[1])
        for item in objects:
            if not isinstance(item, dict) or item.get("category_id") != category_id:
                continue
            local = item.get("box_xyxy")
            if not isinstance(local, list) or len(local) != 4:
                raise ValueError("beam box geometry is invalid")
            x1, y1, x2, y2 = (
                float(local[0]) + offset_x,
                float(local[1]) + offset_y,
                float(local[2]) + offset_x,
                float(local[3]) + offset_y,
            )
            box = tuple(round(value) for value in (x1, y1, x2, y2))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("beam box is empty")
            # Overlapping semantic levels are vertically distinct thin boxes.
            # Only near-identical boxes duplicated by overlapping tiles are merged.
            if any(_bbox_iou(box, existing) >= 0.90 for existing in boxes):
                continue
            boxes.append(box)
    boxes.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    return tuple(boxes)


def oracle_beam_boxes(
    acceptance: dict[str, Any],
    *,
    page_number: int | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    return oracle_category_boxes(
        acceptance,
        "beam",
        page_number=page_number,
    )


def oracle_svg_category_boxes(
    svg_path: Path,
    category_id: str,
) -> tuple[tuple[int, int, int, int], ...]:
    """Read complete page-level oracle geometry before training-tile clipping."""

    _view_box, objects, _description, _excluded = semantic_svg_objects(svg_path)
    boxes = {
        tuple(round(float(value)) for value in item["box_xyxy"])
        for item in objects
        if item.get("category") == category_id
    }
    result = sorted(
        (
            box
            for box in boxes
            if box[2] > box[0] and box[3] > box[1]
        ),
        key=lambda box: (box[1], box[0], box[3], box[2]),
    )
    return tuple(result)


def _reference_cache_is_current(
    reference: Path,
    metadata: Path,
    *,
    source_sha256: str,
    musescore_sha256: str,
) -> bool:
    if not reference.is_file() or not metadata.is_file():
        return False
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        payload.get("source_sha256") == source_sha256
        and payload.get("musescore_sha256") == musescore_sha256
        and payload.get("reference_sha256") == sha256_file(reference)
    )


def _export_reference(
    source: Path,
    destination: Path,
    *,
    musescore: Path,
    musescore_sha256: str,
    timeout_seconds: int,
) -> None:
    source_sha256 = sha256_file(source)
    metadata = destination.with_suffix(destination.suffix + ".source.json")
    if _reference_cache_is_current(
        destination,
        metadata,
        source_sha256=source_sha256,
        musescore_sha256=musescore_sha256,
    ):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scorescan-beam-oracle-export-") as name:
        staged = Path(name) / destination.name
        completed = subprocess.run(
            [str(musescore), "-o", str(staged), str(source)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(30, timeout_seconds),
            check=False,
        )
        if completed.returncode != 0 or not staged.is_file():
            tail = "\n".join((completed.stdout or "").splitlines()[-12:])
            raise RuntimeError(
                f"MuseScore export failed for {source.name}"
                + (f":\n{tail}" if tail else f" (exit {completed.returncode})")
            )
        validation_errors = validate_musicxml(staged)
        if validation_errors:
            raise RuntimeError(
                f"MuseScore reference is invalid for {source.name}: "
                + "; ".join(validation_errors)
            )
        atomic_write_bytes(destination, staged.read_bytes())
    atomic_write_json(
        metadata,
        {
            "format": 1,
            "source": str(source),
            "source_sha256": source_sha256,
            "musescore": str(musescore),
            "musescore_sha256": musescore_sha256,
            "reference_sha256": sha256_file(destination),
        },
    )


def _strip_beams(reference: Path, destination: Path) -> None:
    tree = etree.parse(str(reference), _PARSER)
    root = tree.getroot()
    for beam in list(root.xpath(".//*[local-name()='note']/*[local-name()='beam']")):
        parent = beam.getparent()
        if parent is not None:
            parent.remove(beam)
    payload = etree.tostring(
        tree,
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
        pretty_print=True,
    )
    atomic_write_bytes(destination, payload)
    validation_errors = validate_musicxml(destination)
    if validation_errors:
        raise RuntimeError(
            f"beam-stripped reference is invalid: {'; '.join(validation_errors)}"
        )


def _extract_musicxml_page(
    source: Path,
    destination: Path,
    page_number: int,
) -> int:
    """Materialize one engraved page without relying on cross-page x alignment."""

    if page_number <= 0:
        raise ValueError("page number must be positive")
    tree = etree.parse(str(source), _PARSER)
    root = tree.getroot()
    parts = root.findall("part")
    if not parts:
        raise ValueError("MusicXML has no parts")
    first_measures = list(parts[0].findall("measure"))
    page_by_measure: list[int] = []
    current_page = 1
    for index, measure in enumerate(first_measures):
        print_node = measure.find("print")
        if (
            index
            and print_node is not None
            and print_node.get("new-page") == "yes"
        ):
            current_page += 1
        page_by_measure.append(current_page)
    selected = {
        index
        for index, value in enumerate(page_by_measure)
        if value == page_number
    }
    if not selected:
        raise ValueError(f"MusicXML page {page_number} does not exist")
    for part in parts:
        measures = list(part.findall("measure"))
        if len(measures) != len(first_measures):
            raise ValueError("MusicXML part measure counts differ across the page")
        for index, measure in enumerate(measures):
            if index not in selected:
                part.remove(measure)
    payload = etree.tostring(
        tree,
        encoding="UTF-8",
        xml_declaration=True,
        doctype=MUSICXML_DOCTYPE,
        pretty_print=True,
    )
    atomic_write_bytes(destination, payload)
    validation_errors = validate_musicxml(destination)
    if validation_errors:
        raise RuntimeError(
            "page-sliced reference is invalid: "
            + "; ".join(validation_errors)
        )
    return len(selected)


def _beam_free_c14n(path: Path) -> bytes:
    tree = etree.parse(str(path), _PARSER)
    root = deepcopy(tree.getroot())
    for beam in list(root.xpath(".//*[local-name()='note']/*[local-name()='beam']")):
        parent = beam.getparent()
        if parent is not None:
            parent.remove(beam)
    return etree.tostring(root, method="c14n", with_comments=False)


def _beam_marker_counts(
    reference: Path,
    candidate: Path,
) -> tuple[int, int, int, int]:
    reference_root = etree.parse(str(reference), _PARSER).getroot()
    candidate_root = etree.parse(str(candidate), _PARSER).getroot()
    reference_notes = reference_root.xpath(".//*[local-name()='part']/*[local-name()='measure']/*[local-name()='note']")
    candidate_notes = candidate_root.xpath(".//*[local-name()='part']/*[local-name()='measure']/*[local-name()='note']")
    if len(reference_notes) != len(candidate_notes):
        raise RuntimeError("beam restoration changed the note event count")

    reference_total = 0
    candidate_total = 0
    matches = 0
    exact_topologies = 0
    for reference_note, candidate_note in zip(
        reference_notes,
        candidate_notes,
        strict=True,
    ):
        reference_markers = Counter(
            (
                str(beam.get("number") or "1").strip() or "1",
                " ".join((beam.text or "").split()).casefold(),
            )
            for beam in reference_note.xpath("./*[local-name()='beam']")
        )
        candidate_markers = Counter(
            (
                str(beam.get("number") or "1").strip() or "1",
                " ".join((beam.text or "").split()).casefold(),
            )
            for beam in candidate_note.xpath("./*[local-name()='beam']")
        )
        reference_total += sum(reference_markers.values())
        candidate_total += sum(candidate_markers.values())
        matches += sum((reference_markers & candidate_markers).values())
        exact_topologies += int(reference_markers == candidate_markers)
    return reference_total, candidate_total, matches, exact_topologies


def _safe_rate(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def evaluate_source_beams(
    *,
    dataset_dir: Path,
    prepared_dir: Path,
    output_dir: Path,
    musescore: Path,
    pair_ids: tuple[int, ...],
    page_cases: tuple[tuple[int, int], ...] = (),
    gate_profile: str = "standard",
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    if not musescore.is_file():
        raise FileNotFoundError(f"MuseScore executable not found: {musescore}")
    catalog_path = dataset_dir / "benchmark_dataset.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("benchmark dataset catalog is invalid")
    musescore_sha256 = sha256_file(musescore)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = output_dir / "references"
    candidates = output_dir / "candidates"
    cases: list[dict[str, Any]] = []

    case_specs = tuple((pair_id, None) for pair_id in pair_ids) + tuple(
        (pair_id, page_number)
        for pair_id, page_number in page_cases
    )
    if not case_specs:
        raise ValueError("at least one beam evaluation case is required")
    for pair_id, page_number in case_specs:
        pair_name = f"pair-{pair_id:04d}"
        case_name = (
            pair_name
            if page_number is None
            else f"{pair_name}-page-{page_number:03d}"
        )
        acceptance_path = prepared_dir / "acceptances" / f"{pair_name}.json"
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        accepted = acceptance.get("accepted")
        if not isinstance(accepted, dict):
            raise ValueError(f"{pair_name} has no accepted registered pages")
        if page_number is None and int(accepted.get("pages", 0)) != 1:
            raise ValueError(f"{pair_name} is not a strict one-page acceptance")
        entry = catalog.get(str(pair_id))
        if not isinstance(entry, dict) or not isinstance(entry.get("score"), str):
            raise ValueError(f"{pair_name} is absent from the dataset catalog")
        source = dataset_dir / entry["score"]
        selected_page = page_number or 1
        image = (
            prepared_dir
            / "pages"
            / pair_name
            / f"page-{selected_page}.jpg"
        )
        reference_svg = (
            prepared_dir
            / "reference_pages"
            / pair_name
            / f"page-{selected_page}.svg"
        )
        if not source.is_file() or not image.is_file():
            raise FileNotFoundError(f"{pair_name} source or registered scan is missing")
        if not reference_svg.is_file():
            rendered = _render_score(
                source,
                musescore_exe=musescore,
                piece_dir=output_dir / "reference_pages" / pair_name,
                timeout_seconds=timeout_seconds,
            )
            rendered_by_page = {
                int(svg.stem.rsplit("-", 1)[-1]): svg
                for svg, _png in rendered
            }
            reference_svg = rendered_by_page.get(selected_page)
            if reference_svg is None:
                raise FileNotFoundError(
                    f"{pair_name} rendered page {selected_page} is missing"
                )

        full_reference = references / f"{pair_name}.full.musicxml"
        _export_reference(
            source,
            full_reference,
            musescore=musescore,
            musescore_sha256=musescore_sha256,
            timeout_seconds=timeout_seconds,
        )
        reference = references / f"{case_name}.musicxml"
        selected_measure_count: int | None = None
        if page_number is None:
            shutil.copyfile(full_reference, reference)
        else:
            selected_measure_count = _extract_musicxml_page(
                full_reference,
                reference,
                page_number,
            )
        candidate = candidates / f"{case_name}.musicxml"
        _strip_beams(reference, candidate)
        before_non_beam = _beam_free_c14n(candidate)

        layout = analyze_layout(image)
        accepted_tile_beam_boxes = oracle_beam_boxes(
            acceptance,
            page_number=page_number,
        )
        boxes = oracle_svg_category_boxes(reference_svg, "beam")
        barline_boxes = oracle_svg_category_boxes(
            reference_svg,
            "genericBarline",
        )
        detections: list[SemanticDetection] = []
        unassigned_boxes = 0
        for box in boxes:
            owner = assign_bbox_to_staff(layout, box)
            if owner is None:
                unassigned_boxes += 1
                continue
            detections.append(
                SemanticDetection(
                    class_name="beam",
                    label=3,
                    bbox=box,
                    confidence=1.0,
                    staff_index=owner[0],
                    placement=owner[1],
                )
            )
        assigned_barline_count = 0
        for box in barline_boxes:
            owner = assign_bbox_to_staff(layout, box)
            if owner is None:
                continue
            detections.append(
                SemanticDetection(
                    class_name="genericBarline",
                    label=12,
                    bbox=box,
                    confidence=1.0,
                    staff_index=owner[0],
                    placement=owner[1],
                )
            )
            assigned_barline_count += 1

        report = enrich_musicxml_with_source_beams(
            candidate,
            layout,
            tuple(detections),
        )
        enrichment_report_path = (
            output_dir / "enrichment_reports" / f"{case_name}.json"
        )
        atomic_write_json(enrichment_report_path, report.to_dict())
        after_non_beam = _beam_free_c14n(candidate)
        if before_non_beam != after_non_beam:
            raise RuntimeError(f"{pair_name} changed non-beam MusicXML semantics")
        reference_count, candidate_count, matches, exact_topologies = _beam_marker_counts(
            reference,
            candidate,
        )
        boundary = analyze_reference_boundary(reference)
        comparison = compare_musicxml(reference, candidate)
        topology, topology_error = _build_topology(
            etree.parse(str(reference), _PARSER).getroot(),
            layout,
        )
        reasons = Counter(
            proposal.reason
            for proposal in report.proposals
            if not proposal.injected
        )
        cases.append(
            {
                "pair_id": pair_id,
                "page_number": selected_page,
                "selected_measure_count": selected_measure_count,
                "boundary": boundary,
                "source_sha256": sha256_file(source),
                "scan_sha256": sha256_file(image),
                "acceptance_sha256": sha256_file(acceptance_path),
                "reference_sha256": sha256_file(reference),
                "candidate_sha256": sha256_file(candidate),
                "oracle_box_count": len(boxes),
                "accepted_tile_beam_box_count": len(
                    accepted_tile_beam_boxes
                ),
                "tile_dropped_beam_box_count": max(
                    0,
                    len(boxes) - len(accepted_tile_beam_boxes),
                ),
                "assigned_box_count": len(boxes) - unassigned_boxes,
                "unassigned_box_count": unassigned_boxes,
                "oracle_barline_box_count": len(barline_boxes),
                "assigned_barline_box_count": assigned_barline_count,
                "raw_staff_appearance_count": len(layout.systems),
                "conditioned_staff_appearance_count": (
                    len(topology.ordered_appearances)
                    if topology is not None
                    else None
                ),
                "topology_error": topology_error,
                "transaction_committed": report.transaction_committed,
                "injected_segment_count": report.injected_segment_count,
                "reference_beam_marker_count": reference_count,
                "candidate_beam_marker_count": candidate_count,
                "beam_marker_matches": matches,
                "beam_marker_precision": _safe_rate(matches, candidate_count),
                "beam_marker_recall": _safe_rate(matches, reference_count),
                "exact_note_beam_topology_rate": _safe_rate(
                    exact_topologies,
                    len(
                        etree.parse(str(reference), _PARSER).getroot().xpath(
                            ".//*[local-name()='part']/*[local-name()='measure']/*[local-name()='note']"
                        )
                    ),
                ),
                "comparison_beam_topology_accuracy_aligned": comparison[
                    "beam_topology_accuracy_aligned"
                ],
                "abstention_reasons": dict(sorted(reasons.items())),
                "enrichment_error": report.error,
                "enrichment_report": str(enrichment_report_path),
                "non_beam_preservation_exact": before_non_beam == after_non_beam,
            }
        )

    reference_total = sum(item["reference_beam_marker_count"] for item in cases)
    candidate_total = sum(item["candidate_beam_marker_count"] for item in cases)
    matches = sum(item["beam_marker_matches"] for item in cases)
    precision = _safe_rate(matches, candidate_total)
    recall = _safe_rate(matches, reference_total)
    shape_counts = Counter(
        str(item["boundary"]["score_shape"])
        for item in cases
    )
    in_boundary_shape_counts = Counter(
        str(item["boundary"]["score_shape"])
        for item in cases
        if bool(item["boundary"]["accepted"])
    )
    rejection_reasons = Counter(
        str(reason)
        for item in cases
        for reason in item["boundary"]["reasons"]
    )
    cross_staff_groups = sum(
        int(item["boundary"]["counts"]["cross_staff_beam_groups"])
        for item in cases
    )
    if gate_profile == "cross-staff":
        minimum_cases = CROSS_STAFF_MINIMUM_CASES
        minimum_reference_markers = CROSS_STAFF_MINIMUM_REFERENCE_MARKERS
    elif gate_profile == "standard":
        minimum_cases = MINIMUM_CASES
        minimum_reference_markers = MINIMUM_REFERENCE_MARKERS
    else:
        raise ValueError(f"unsupported gate profile: {gate_profile!r}")
    gate_checks = {
        "minimum_cases": len(cases) >= minimum_cases,
        "minimum_reference_markers": reference_total >= minimum_reference_markers,
        "minimum_oracle_precision": precision >= MINIMUM_ORACLE_PRECISION,
        "minimum_per_case_oracle_precision": all(
            item["beam_marker_precision"] >= MINIMUM_ORACLE_PRECISION
            for item in cases
        ),
        "minimum_oracle_recall": recall >= MINIMUM_ORACLE_RECALL,
        "non_beam_preservation": all(
            item["non_beam_preservation_exact"] for item in cases
        ),
        "all_oracle_boxes_assigned": all(
            item["unassigned_box_count"] == 0 for item in cases
        ),
    }
    if gate_profile == "cross-staff":
        gate_checks["all_cases_contain_cross_staff_beams"] = all(
            int(item["boundary"]["counts"]["cross_staff_beam_groups"]) > 0
            for item in cases
        )
        gate_checks["all_cases_are_in_product_boundary"] = all(
            bool(item["boundary"]["accepted"])
            for item in cases
        )
    return {
        "format": 1,
        "evaluation_kind": "oracle-registered-source-beam-resolver",
        "not_a_detector_benchmark": True,
        "limitations": [
            "uses complete registered page-SVG ground-truth beam boxes, not model predictions",
            "evaluates registered pages independently; it does not infer cross-page x alignment",
            "proves source mapping and beam assignment independently of detector recall",
        ],
        "source_box_provenance": "complete-reference-page-svg-before-tile-clipping",
        "dataset_dir": str(dataset_dir),
        "prepared_dir": str(prepared_dir),
        "dataset_catalog_sha256": sha256_file(catalog_path),
        "musescore_sha256": musescore_sha256,
        "enrichment_version": BEAM_ENRICHMENT_VERSION,
        "pair_ids": list(pair_ids),
        "page_cases": [
            {"pair_id": pair_id, "page_number": page_number}
            for pair_id, page_number in page_cases
        ],
        "gate_profile": gate_profile,
        "aggregate": {
            "case_count": len(cases),
            "reference_beam_marker_count": reference_total,
            "candidate_beam_marker_count": candidate_total,
            "beam_marker_matches": matches,
            "beam_marker_precision": precision,
            "beam_marker_recall": recall,
            "beam_marker_f1": _safe_rate(
                2 * matches,
                reference_total + candidate_total,
            ),
            "cross_staff_beam_group_count": cross_staff_groups,
            "score_shape_case_counts": dict(sorted(shape_counts.items())),
            "in_boundary_score_shape_case_counts": dict(
                sorted(in_boundary_shape_counts.items())
            ),
            "boundary_accepted_case_count": sum(
                bool(item["boundary"]["accepted"])
                for item in cases
            ),
            "boundary_rejected_case_count": sum(
                not bool(item["boundary"]["accepted"])
                for item in cases
            ),
            "boundary_rejection_reasons": dict(
                sorted(rejection_reasons.items())
            ),
        },
        "gate": {
            "thresholds": {
                "minimum_cases": minimum_cases,
                "minimum_reference_markers": minimum_reference_markers,
                "minimum_oracle_precision": MINIMUM_ORACLE_PRECISION,
                "minimum_oracle_recall": MINIMUM_ORACLE_RECALL,
            },
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate source beam mapping with registered oracle boxes."
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--musescore-exe", required=True, type=Path)
    parser.add_argument(
        "--pairs",
        action="append",
        default=[],
        help="Comma-separated pair ids; defaults to the fixed development panel.",
    )
    parser.add_argument(
        "--page-cases",
        action="append",
        default=[],
        help="Comma-separated PAIR:PAGE registered-page cases.",
    )
    parser.add_argument(
        "--gate-profile",
        choices=("standard", "cross-staff"),
        default="standard",
    )
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    page_cases = (
        _parse_page_cases(args.page_cases)
        if args.page_cases
        else ()
    )
    pair_ids = (
        _parse_pair_ids(args.pairs)
        if args.pairs
        else (() if page_cases else DEFAULT_PAIR_IDS)
    )
    report = evaluate_source_beams(
        dataset_dir=args.dataset_dir.resolve(),
        prepared_dir=args.prepared_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        musescore=args.musescore_exe.resolve(),
        pair_ids=pair_ids,
        page_cases=page_cases,
        gate_profile=args.gate_profile,
        timeout_seconds=args.timeout_seconds,
    )
    report_path = args.output_dir.resolve() / "source-beam-restoration-report.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.gate and not report["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
