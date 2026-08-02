from __future__ import annotations

"""Prepare pinned Muse OMR rendered degradations as a development benchmark.

The source PDFs are intentionally never copied into a training directory. MuseScore
files are exported to MusicXML only to provide independent semantic ground truth.
Every case is classified against ScoreScan's declared instrumental-score boundary
before it can enter development evaluation. Generated-and-degraded pages can never
satisfy the physical-scan production-v2 evidence contract.
"""

import argparse
import concurrent.futures
import functools
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from lxml import etree
import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import atomic_write_bytes, atomic_write_json, sha256_file, utc_now_iso
from scorescan.product_scope import (
    MAXIMUM_KEYBOARD_PARTS,
    MAXIMUM_KEYBOARD_STAVES,
    MAXIMUM_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE,
    MAXIMUM_NON_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE,
    MAXIMUM_PHYSICAL_STAVES_PER_SYSTEM,
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from app.tools.evaluate_release_dataset import (
    PRODUCTION_RELEASE_GATES_V2,
    PRODUCTION_SCORE_CONFIGURATIONS,
    SCORE_CONFIGURATION_BY_SHAPE,
)
from app.tools.muse_omr_contract import (
    BENCHMARK_SELECTION_ROLE,
    SCAN_DEGRADED_IMAGE_ORIGIN,
    TRAINING_SELECTION_ROLE,
)


WORK_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
TRAINING_BOUNDARY_CLASSIFICATION_ROLE = (
    "training_boundary_classification_only_not_evaluation"
)


def _boundary_output_role(
    selection_role: object,
    *,
    allow_training_classification: bool,
) -> str:
    if selection_role == BENCHMARK_SELECTION_ROLE:
        return BENCHMARK_SELECTION_ROLE
    if (
        allow_training_classification
        and selection_role == TRAINING_SELECTION_ROLE
    ):
        return TRAINING_BOUNDARY_CLASSIFICATION_ROLE
    raise ValueError(
        "source selection role is not authorized for boundary benchmark "
        "preparation"
    )


def _selection_work_map(
    selection: dict[str, object],
    selected_ids: list[int],
) -> dict[int, str]:
    rows = selection.get("pair_work_fingerprints")
    selected_works = selection.get("selected_work_fingerprints")
    if not isinstance(rows, list) or not isinstance(selected_works, list):
        raise ValueError("benchmark selection has no work-level provenance")
    mapping: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid pair/work provenance row")
        pair_id = int(row.get("pair_id", -1))
        fingerprint = str(row.get("work_fingerprint", ""))
        if (
            pair_id in mapping
            or WORK_FINGERPRINT_PATTERN.fullmatch(fingerprint) is None
        ):
            raise ValueError("invalid pair/work provenance")
        mapping[pair_id] = fingerprint
    expected_works = {str(value) for value in selected_works}
    if (
        set(mapping) != set(selected_ids)
        or set(mapping.values()) != expected_works
        or int(selection.get("selected_work_count", -1))
        != len(expected_works)
    ):
        raise ValueError("benchmark work-level provenance is inconsistent")
    return mapping


def unique_work_cases(
    cases: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep one deterministic submitted document for each independent work."""

    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for case in cases:
        fingerprint = str(case.get("work_fingerprint", ""))
        if WORK_FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
            raise ValueError(
                f"case {case.get('id')} has no valid work fingerprint"
            )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(case)
    return unique


def production_page_coverage(
    cases: list[dict[str, object]],
) -> tuple[int, dict[str, int]]:
    """Count submitted pages once, using the closed production configurations."""

    pages_by_configuration = {
        name: 0 for name in PRODUCTION_SCORE_CONFIGURATIONS
    }
    total_pages = 0
    for case in cases:
        raw_pages = case.get("input_pdf_pages")
        if (
            isinstance(raw_pages, bool)
            or not isinstance(raw_pages, int)
            or raw_pages <= 0
        ):
            raise ValueError(
                f"case {case.get('id')} has no positive input PDF page count"
            )
        boundary = case.get("boundary")
        if not isinstance(boundary, dict):
            raise ValueError(f"case {case.get('id')} has no boundary record")
        score_shape = str(boundary.get("score_shape", ""))
        try:
            configuration = SCORE_CONFIGURATION_BY_SHAPE[score_shape]
        except KeyError as exc:
            raise ValueError(
                f"case {case.get('id')} has unsupported score shape: "
                f"{score_shape or '<missing>'}"
            ) from exc
        total_pages += raw_pages
        pages_by_configuration[configuration] += raw_pages
    return total_pages, pages_by_configuration


def _pdf_page_count(path: Path) -> int:
    with fitz.open(path) as document:
        count = int(document.page_count)
    if count <= 0:
        raise ValueError(f"benchmark PDF has no pages: {path}")
    return count


def _integer(value: str | None, default: int = 1) -> int:
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def _cross_staff_beam_count(root: etree._Element) -> int:
    count = 0
    for measure in root.findall("./part/measure"):
        active: dict[tuple[str, str], set[int]] = {}
        for note in measure.findall("note"):
            voice = note.findtext("voice") or "1"
            staff = _integer(note.findtext("staff"), 1)
            for beam in note.findall("beam"):
                number = beam.get("number") or "1"
                key = (voice, number)
                value = (beam.text or "").strip().casefold()
                if value == "begin":
                    active[key] = {staff}
                elif value in {"continue", "forward hook", "backward hook"}:
                    active.setdefault(key, set()).add(staff)
                elif value == "end":
                    group = active.pop(key, set())
                    group.add(staff)
                    if len(group) > 1:
                        count += 1
        count += sum(len(staffs) > 1 for staffs in active.values())
    return count


def _simultaneous_voices_by_staff(
    measure: etree._Element,
) -> dict[int, int]:
    """Count concurrent MusicXML voice timelines, not reused voice labels."""
    cursor = 0
    previous_note_onset: int | None = None
    intervals: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for child in measure:
        name = etree.QName(child).localname
        if name == "backup":
            cursor = max(0, cursor - _integer(child.findtext("duration"), 0))
            previous_note_onset = None
            continue
        if name == "forward":
            cursor += _integer(child.findtext("duration"), 0)
            previous_note_onset = None
            continue
        if name != "note":
            previous_note_onset = None
            continue
        duration = max(0, _integer(child.findtext("duration"), 0))
        is_chord = child.find("chord") is not None
        onset = (
            previous_note_onset
            if is_chord and previous_note_onset is not None
            else cursor
        )
        staff = _integer(child.findtext("staff"), 1)
        voice = child.findtext("voice") or "1"
        if duration > 0:
            intervals[staff][voice].append((onset, onset + duration))
        if not is_chord:
            cursor += duration
            previous_note_onset = onset

    result = {}
    for staff, by_voice in intervals.items():
        boundaries = sorted(
            {
                boundary
                for voice_intervals in by_voice.values()
                for interval in voice_intervals
                for boundary in interval
            }
        )
        result[staff] = max(
            (
                sum(
                    any(start <= point < end for start, end in voice_intervals)
                    for voice_intervals in by_voice.values()
                )
                for point in boundaries[:-1]
            ),
            default=0,
        )
    return result


def analyze_reference_boundary(path: Path) -> dict[str, object]:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    root = etree.parse(str(path), parser).getroot()
    parts = root.findall("part")
    part_staff_counts: list[int] = []
    simultaneous_voices_by_part_staff_measure: dict[
        tuple[int, int, int],
        int,
    ] = {}
    note_count = 0
    measure_count = 0
    for part_index, part in enumerate(parts):
        staff_count = 1
        measures = part.findall("measure")
        measure_count = max(measure_count, len(measures))
        for staves in part.findall("./measure/attributes/staves"):
            staff_count = max(staff_count, _integer(staves.text, 1))
        for measure_index, measure in enumerate(measures):
            for staff, voice_count in _simultaneous_voices_by_staff(
                measure
            ).items():
                simultaneous_voices_by_part_staff_measure[
                    (part_index, measure_index, staff)
                ] = voice_count
            for note in measure.findall("note"):
                note_count += 1
                staff = _integer(note.findtext("staff"), 1)
                staff_count = max(staff_count, staff)
        part_staff_counts.append(staff_count)

    multi_staff_parts = sum(value > 1 for value in part_staff_counts)
    maximum_voices = max(
        (
            value
            for value in simultaneous_voices_by_part_staff_measure.values()
        ),
        default=0,
    )
    maximum_keyboard_voices = max(
        (
            voice_count
            for (voice_part_index, _measure_index, _staff), voice_count
            in simultaneous_voices_by_part_staff_measure.items()
            for boundary_part_index, staff_count
            in enumerate(part_staff_counts)
            if staff_count > 1
            if voice_part_index == boundary_part_index
        ),
        default=0,
    )
    maximum_non_keyboard_voices = max(
        (
            voice_count
            for (voice_part_index, _measure_index, _staff), voice_count
            in simultaneous_voices_by_part_staff_measure.items()
            for part_index, staff_count in enumerate(part_staff_counts)
            if staff_count == 1
            if voice_part_index == part_index
        ),
        default=0,
    )
    cross_staff_beams = _cross_staff_beam_count(root)
    lyric_count = len(root.findall(".//lyric"))
    harmony_count = len(root.findall(".//harmony"))
    unpitched_count = len(root.findall(".//unpitched"))
    reasons: list[str] = []
    if not parts or not measure_count or not note_count:
        reasons.append("empty_or_non_notated_score")
    if len(parts) > 24:
        reasons.append("more_than_24_parts")
    if sum(part_staff_counts) > MAXIMUM_PHYSICAL_STAVES_PER_SYSTEM:
        reasons.append("more_than_16_physical_staves")
    # Common keyboard engraving includes temporary third/fourth staves and
    # ossias. The product scope explicitly includes those forms. A single
    # multi-staff keyboard part may therefore use up to four physical staves.
    if any(value > MAXIMUM_KEYBOARD_STAVES for value in part_staff_counts):
        reasons.append("keyboard_part_with_more_than_four_staves")
    if multi_staff_parts > MAXIMUM_KEYBOARD_PARTS:
        reasons.append("more_than_one_keyboard_part")
    # Complex piano engraving can use more than four simultaneous voices on
    # one staff, especially with cross-staff writing and temporary third
    # staves. The frozen product scope promises those common forms. Counts
    # above eight are treated as condensed/cue-heavy notation.
    if (
        maximum_keyboard_voices
        > MAXIMUM_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE
    ):
        reasons.append(
            "more_than_eight_independent_voices_per_keyboard_staff"
        )
    # A non-keyboard staff in the frozen product contract has one independent
    # rhythmic timeline. Chords remain one voice; overlapping second voices
    # are condensed/divisi notation and therefore outside the boundary.
    if (
        maximum_non_keyboard_voices
        > MAXIMUM_NON_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE
    ):
        reasons.append(
            "more_than_one_independent_voice_per_non_keyboard_staff"
        )
    # Cross-staff notes and beams are explicitly inside the common-piano
    # product contract.  Keep the count for stratified evaluation, but never
    # define these difficult pages out of the frozen benchmark.
    if lyric_count:
        reasons.append("lyrics")
    if harmony_count:
        reasons.append("harmony_symbols")
    if unpitched_count:
        reasons.append("unpitched_or_percussion_notation")

    if multi_staff_parts and len(parts) > 1:
        score_shape = "keyboard_plus_single_staff_ensemble"
    elif multi_staff_parts:
        score_shape = "keyboard"
    elif len(parts) > 1:
        score_shape = "single_staff_ensemble"
    else:
        score_shape = "single_staff_solo"

    counts = {
        "parts": len(parts),
        "measures": measure_count,
        "notes": note_count,
        "maximum_voices_per_staff": maximum_voices,
        "maximum_voices_per_keyboard_staff": maximum_keyboard_voices,
        "maximum_voices_per_non_keyboard_staff": (
            maximum_non_keyboard_voices
        ),
        "ties": len(root.findall(".//tie")),
        "slurs": len(root.findall(".//slur")),
        "beams": len(root.findall(".//beam")),
        "tuplets": len(root.findall(".//tuplet")),
        "grace_notes": len(root.findall(".//grace")),
        "articulations": sum(len(node) for node in root.findall(".//articulations")),
        "ornaments": sum(len(node) for node in root.findall(".//ornaments")),
        "directions": len(root.findall(".//direction")),
        "words": len(root.findall(".//direction-type/words")),
        "dynamics": sum(len(node) for node in root.findall(".//direction-type/dynamics")),
        "wedges": len(root.findall(".//wedge")),
        "lyrics": lyric_count,
        "harmony_symbols": harmony_count,
        "unpitched_notes": unpitched_count,
        "cross_staff_beam_groups": cross_staff_beams,
    }
    return {
        "contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "accepted": not reasons,
        "reasons": reasons,
        "score_shape": score_shape,
        "part_staff_counts": part_staff_counts,
        "counts": counts,
    }


def _export_musicxml(
    source: Path,
    destination: Path,
    musescore: Path,
    *,
    timeout_seconds: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for attempt in range(1, 4):
        with tempfile.TemporaryDirectory(
            prefix="scorescan-muse-reference-"
        ) as temp_dir:
            staged = Path(temp_dir) / destination.name
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
            if completed.returncode == 0 and staged.is_file():
                try:
                    analyze_reference_boundary(staged)
                    atomic_write_bytes(destination, staged.read_bytes())
                    return
                except (OSError, ValueError, etree.XMLSyntaxError) as exc:
                    failures.append(f"attempt {attempt}: {exc}")
            else:
                tail = "\n".join(
                    (completed.stdout or "").splitlines()[-12:]
                )
                failures.append(
                    f"attempt {attempt}, exit {completed.returncode}"
                    + (f":\n{tail}" if tail else "")
                )
        if attempt < 3:
            time.sleep(attempt)
    raise RuntimeError(
        f"MuseScore export failed for {source.name} after 3 attempts:\n"
        + "\n".join(failures)
    )


def _prepare_boundary_case(
    item: tuple[int, int, dict[str, object], str],
    *,
    benchmark_root: Path,
    output_dir: Path,
    musescore: Path,
    timeout_seconds: int,
    force: bool,
    output_role: str,
    total: int,
) -> dict[str, object]:
    position, pair_id, source_info, work_fingerprint = item
    score_path = (benchmark_root / str(source_info["score"])).resolve()
    pdf_path = (benchmark_root / str(source_info["pdf_image"])).resolve()
    if not score_path.is_file() or not pdf_path.is_file():
        raise FileNotFoundError(f"benchmark pair {pair_id} is incomplete")
    reference_path = (
        output_dir / "references" / f"work_{work_fingerprint}.musicxml"
    )
    print(f"[{position}/{total}] reference {pair_id}", flush=True)
    if force or not reference_path.is_file():
        _export_musicxml(
            score_path,
            reference_path,
            musescore,
            timeout_seconds=timeout_seconds,
        )
    boundary = analyze_reference_boundary(reference_path)
    return {
        "id": f"muse-{pair_id}",
        "pair_id": pair_id,
        "variant_key": f"muse-omr/{pair_id}",
        "work_fingerprint": work_fingerprint,
        "role": (
            "external_test_only"
            if output_role == BENCHMARK_SELECTION_ROLE
            else TRAINING_BOUNDARY_CLASSIFICATION_ROLE
        ),
        "input_pdf": str(pdf_path),
        "input_pdf_pages": _pdf_page_count(pdf_path),
        "reference": str(reference_path.relative_to(output_dir)),
        "source_mscz_sha256": sha256_file(score_path),
        "input_pdf_sha256": sha256_file(pdf_path),
        "boundary": boundary,
    }


def _capture_boundary_case(
    item: tuple[int, int, dict[str, object], str],
    *,
    prepare_case: object,
) -> tuple[
    tuple[int, int, dict[str, object], str],
    dict[str, object] | None,
    str | None,
]:
    try:
        return item, prepare_case(item), None  # type: ignore[operator]
    except RuntimeError as exc:
        return item, None, str(exc)


def _unclassifiable_boundary_case(
    item: tuple[int, int, dict[str, object], str],
    *,
    benchmark_root: Path,
    output_dir: Path,
    output_role: str,
    error: str,
) -> dict[str, object]:
    _position, pair_id, source_info, work_fingerprint = item
    score_path = (benchmark_root / str(source_info["score"])).resolve()
    pdf_path = (benchmark_root / str(source_info["pdf_image"])).resolve()
    return {
        "id": f"muse-{pair_id}",
        "pair_id": pair_id,
        "variant_key": f"muse-omr/{pair_id}",
        "work_fingerprint": work_fingerprint,
        "role": (
            "external_test_only"
            if output_role == BENCHMARK_SELECTION_ROLE
            else TRAINING_BOUNDARY_CLASSIFICATION_ROLE
        ),
        "input_pdf": str(pdf_path),
        "input_pdf_pages": _pdf_page_count(pdf_path),
        "reference": None,
        "source_mscz_sha256": sha256_file(score_path),
        "input_pdf_sha256": sha256_file(pdf_path),
        "classification_error": error,
        "boundary": {
            "contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
            "accepted": False,
            "reasons": ["reference_export_or_parse_failed"],
            "score_shape": "unclassified_source_failure",
            "part_staff_counts": [],
            "counts": {},
        },
    }


def prepare_benchmark(
    benchmark_root: Path,
    output_dir: Path,
    musescore: Path,
    *,
    limit: int | None = None,
    timeout_seconds: int = 180,
    force: bool = False,
    allow_training_classification: bool = False,
    workers: int = 1,
) -> dict[str, object]:
    dataset_path = benchmark_root / "benchmark_dataset.json"
    selection_path = benchmark_root / "selection.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    output_role = _boundary_output_role(
        selection.get("role"),
        allow_training_classification=allow_training_classification,
    )
    if (
        selection.get("source_image_origin")
        != SCAN_DEGRADED_IMAGE_ORIGIN
        or selection.get("production_evidence_eligible") is not False
    ):
        raise ValueError(
            "Muse OMR selection has no explicit scan-degraded origin"
        )
    selected_ids = [int(value) for value in selection.get("selected_pair_ids", [])]
    work_by_pair = _selection_work_map(selection, selected_ids)
    if limit is not None:
        selected_ids = selected_ids[: max(0, int(limit))]
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    work_items: list[tuple[int, int, dict[str, object], str]] = []
    for position, pair_id in enumerate(selected_ids, start=1):
        source_info = dataset.get(str(pair_id))
        if not isinstance(source_info, dict):
            raise RuntimeError(f"dataset entry {pair_id} is missing")
        work_items.append(
            (
                position,
                pair_id,
                source_info,
                work_by_pair[pair_id],
            )
        )
    prepare_case = functools.partial(
        _prepare_boundary_case,
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        musescore=musescore,
        timeout_seconds=timeout_seconds,
        force=force,
        output_role=output_role,
        total=len(selected_ids),
    )
    if workers == 1:
        captured = [
            _capture_boundary_case(item, prepare_case=prepare_case)
            for item in work_items
        ]
    else:
        capture_case = functools.partial(
            _capture_boundary_case,
            prepare_case=prepare_case,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="muse-boundary",
        ) as executor:
            captured = list(executor.map(capture_case, work_items))
    cases: list[dict[str, object]] = []
    for item, case, concurrent_error in captured:
        if case is not None:
            cases.append(case)
            continue
        # MuseScore is not fully multi-instance safe.  Retry after every
        # concurrent export has ended before quarantining the source.
        try:
            cases.append(prepare_case(item))
        except RuntimeError as serial_error:
            cases.append(
                _unclassifiable_boundary_case(
                    item,
                    benchmark_root=benchmark_root,
                    output_dir=output_dir,
                    output_role=output_role,
                    error=(
                        f"concurrent: {concurrent_error}\n"
                        f"serial: {serial_error}"
                    ),
                )
            )

    accepted_cases = [
        case for case in cases if bool(case["boundary"]["accepted"])
    ]
    accepted = len(accepted_cases)
    accepted_work_cases = unique_work_cases(accepted_cases)
    accepted_work_fingerprints = sorted(
        str(case["work_fingerprint"]) for case in accepted_work_cases
    )
    (
        accepted_input_page_count,
        pages_by_score_configuration,
    ) = production_page_coverage(accepted_work_cases)
    production_minimum = PRODUCTION_RELEASE_GATES_V2["minimum"]
    coverage_gaps = {
        name: max(
            0,
            int(production_minimum[f"{name}_page_count"])
            - pages_by_score_configuration[name],
        )
        for name in PRODUCTION_SCORE_CONFIGURATIONS
    }
    coverage_gaps["submitted_scan_page_count"] = max(
        0,
        int(production_minimum["submitted_scan_page_count"])
        - accepted_input_page_count,
    )
    coverage_gaps["source_group_count"] = max(
        0,
        int(production_minimum["source_group_count"])
        - len(accepted_work_fingerprints),
    )
    report = {
        "format": 1,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "created_at": utc_now_iso(),
        "name": (
            "Muse OMR pinned external ScoreScan boundary benchmark"
            if output_role == BENCHMARK_SELECTION_ROLE
            else "Muse OMR training-partition boundary classification"
        ),
        "role": output_role,
        "source_image_origin": SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "production_evidence_blockers": [
            "source images are generated renders with simulated scan degradation",
            "production-v2 requires uniquely identified physical scan pages",
            "development references are not double-annotated frozen release truth",
        ],
        "source_selection_role": selection.get("role"),
        "license": selection.get("license"),
        "repository": selection.get("repository"),
        "revision": selection.get("revision"),
        "source_selection_sha256": sha256_file(selection_path),
        "source_dataset_sha256": sha256_file(dataset_path),
        "case_count": len(cases),
        "work_count": len({str(case["work_fingerprint"]) for case in cases}),
        "accepted_case_count": accepted,
        "accepted_submitted_document_count": len(accepted_work_cases),
        "accepted_work_count": len(accepted_work_fingerprints),
        "accepted_work_fingerprints": accepted_work_fingerprints,
        "accepted_input_page_count": accepted_input_page_count,
        "accepted_input_pages_by_score_configuration": (
            pages_by_score_configuration
        ),
        "development_coverage_against_production_shape_minimum": (
            coverage_gaps
        ),
        "development_shape_coverage_complete": all(
            gap == 0 for gap in coverage_gaps.values()
        ),
        "production_scope_coverage_complete": False,
        "rejected_case_count": len(cases) - accepted,
        "cases": cases,
    }
    atomic_write_json(output_dir / "boundary_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--musescore", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-training-classification",
        action="store_true",
        help=(
            "classify a training selection without authorizing its output "
            "for release evaluation"
        ),
    )
    args = parser.parse_args()
    report = prepare_benchmark(
        args.benchmark_root.resolve(),
        args.output_dir.resolve(),
        args.musescore.resolve(),
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        force=args.force,
        allow_training_classification=args.allow_training_classification,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "accepted_case_count": report["accepted_case_count"],
                "rejected_case_count": report["rejected_case_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
