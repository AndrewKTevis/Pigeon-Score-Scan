from __future__ import annotations

"""Independent semantic evaluation for the PrIMuS monophonic-score corpus.

The product writes MusicXML while PrIMuS supplies a compact semantic token stream.
This module compares those two representations directly, deliberately ignoring
titles and engraving metadata that are not part of musical recognition accuracy.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

from lxml import etree


_PITCH_RE = re.compile(r"^(?P<step>[A-G])(?P<accidental>[#b]*)(?P<octave>-?\d+)$")
_FIFTHS = {
    "CbM": -7,
    "GbM": -6,
    "DbM": -5,
    "AbM": -4,
    "EbM": -3,
    "BbM": -2,
    "FM": -1,
    "CM": 0,
    "GM": 1,
    "DM": 2,
    "AM": 3,
    "EM": 4,
    "BM": 5,
    "F#M": 6,
    "C#M": 7,
}
_TYPE_NAMES = {
    "quadruple_whole": "long",
    "double_whole": "breve",
    "whole": "whole",
    "half": "half",
    "quarter": "quarter",
    "eighth": "eighth",
    "sixteenth": "16th",
    "thirty_second": "32nd",
    "sixty_fourth": "64th",
    "hundred_twenty_eighth": "128th",
}


@dataclass(frozen=True)
class SemanticEvent:
    kind: str
    pitch: tuple[str, int, int] | None = None
    rhythm: tuple[str, int] | None = None
    fermata: bool = False
    value: int | None = None


@dataclass(frozen=True)
class SemanticScore:
    events: tuple[SemanticEvent, ...]
    measure_count: int
    clef: str | None
    fifths: int | None
    time_signature: tuple[int, int] | None
    tie_count: int
    token_features: tuple[str, ...]


def _parse_pitch(value: str) -> tuple[str, int, int] | None:
    match = _PITCH_RE.fullmatch(value)
    if match is None:
        return None
    accidental = match.group("accidental")
    alter = accidental.count("#") - accidental.count("b")
    return match.group("step"), alter, int(match.group("octave"))


def _parse_rhythm(value: str) -> tuple[tuple[str, int] | None, bool]:
    fermata = value.endswith("_fermata")
    if fermata:
        value = value.removesuffix("_fermata")
    dots = len(value) - len(value.rstrip("."))
    type_name = _TYPE_NAMES.get(value.rstrip("."))
    return ((type_name, dots) if type_name else None), fermata


def parse_primus_semantic_text(text: str) -> SemanticScore:
    events: list[SemanticEvent] = []
    clef: str | None = None
    fifths: int | None = None
    time_signature: tuple[int, int] | None = None
    tie_count = 0
    barlines = 0
    features: set[str] = set()

    for token in text.split():
        if token.startswith("clef-"):
            clef = token.removeprefix("clef-")
        elif token.startswith("keySignature-"):
            fifths = _FIFTHS.get(token.removeprefix("keySignature-"))
        elif token.startswith("timeSignature-"):
            value = token.removeprefix("timeSignature-")
            if value == "C":
                time_signature = (4, 4)
            elif value == "C/":
                time_signature = (2, 2)
            elif "/" in value:
                beats, beat_type = value.split("/", 1)
                try:
                    time_signature = int(beats), int(beat_type)
                except ValueError:
                    features.add("unparsed_time_signature")
        elif token == "barline":
            barlines += 1
        elif token == "tie":
            tie_count += 1
            features.add("tie")
        elif token.startswith("multirest-"):
            try:
                value = int(token.removeprefix("multirest-"))
            except ValueError:
                value = None
            events.append(SemanticEvent("multirest", value=value))
            features.add("multirest")
        elif token.startswith(("note-", "gracenote-")):
            prefix, value = token.split("-", 1)
            if "_" not in value:
                features.add("unparsed_event")
                continue
            pitch_text, rhythm_text = value.split("_", 1)
            rhythm, fermata = _parse_rhythm(rhythm_text)
            kind = "grace" if prefix == "gracenote" else "note"
            events.append(SemanticEvent(kind, _parse_pitch(pitch_text), rhythm, fermata))
            if kind == "grace":
                features.add("grace")
            if fermata:
                features.add("fermata")
        elif token.startswith("rest-"):
            rhythm, fermata = _parse_rhythm(token.removeprefix("rest-"))
            events.append(SemanticEvent("rest", rhythm=rhythm, fermata=fermata))
            if fermata:
                features.add("fermata")
        else:
            features.add(f"unknown:{token.split('-', 1)[0]}")

    return SemanticScore(
        events=tuple(events),
        measure_count=barlines,
        clef=clef,
        fifths=fifths,
        time_signature=time_signature,
        tie_count=tie_count,
        token_features=tuple(sorted(features)),
    )


def parse_primus_semantic(path: Path) -> SemanticScore:
    return parse_primus_semantic_text(path.read_text(encoding="utf-8"))


def _int_text(node: etree._Element | None, name: str) -> int | None:
    if node is None:
        return None
    value = node.findtext(name)
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def _musicxml_pitch(note: etree._Element) -> tuple[str, int, int] | None:
    pitch = note.find("pitch")
    if pitch is None:
        return None
    step = pitch.findtext("step")
    octave = _int_text(pitch, "octave")
    alter = _int_text(pitch, "alter") or 0
    if step not in set("ABCDEFG") or octave is None:
        return None
    return step, alter, octave


def parse_musicxml_semantics(path: Path) -> SemanticScore:
    tree = etree.parse(str(path), etree.XMLParser(resolve_entities=False, no_network=True))
    part = tree.getroot().find("part")
    if part is None:
        return SemanticScore((), 0, None, None, None, 0, ("missing_part",))

    events: list[SemanticEvent] = []
    clef: str | None = None
    fifths: int | None = None
    time_signature: tuple[int, int] | None = None
    tie_count = 0
    features: set[str] = set()
    measures = part.findall("measure")

    for measure in measures:
        for attributes in measure.findall("attributes"):
            if clef is None:
                clef_node = attributes.find("clef")
                if clef_node is not None:
                    sign, line = clef_node.findtext("sign"), clef_node.findtext("line")
                    clef = f"{sign}{line}" if sign and line else sign
            if fifths is None:
                fifths = _int_text(attributes.find("key"), "fifths")
            if time_signature is None:
                time = attributes.find("time")
                beats = _int_text(time, "beats")
                beat_type = _int_text(time, "beat-type")
                if beats is not None and beat_type is not None:
                    time_signature = beats, beat_type
            multiple_rest = _int_text(attributes.find("measure-style"), "multiple-rest")
            if multiple_rest is not None:
                events.append(SemanticEvent("multirest", value=multiple_rest))
                features.add("multirest")

        for note in measure.findall("note"):
            if note.find("chord") is not None:
                features.add("chord")
            kind = "rest" if note.find("rest") is not None else ("grace" if note.find("grace") is not None else "note")
            type_name = note.findtext("type")
            rhythm = (type_name, len(note.findall("dot"))) if type_name else None
            fermata = bool(note.findall("./notations/fermata"))
            events.append(
                SemanticEvent(
                    kind=kind,
                    pitch=_musicxml_pitch(note) if kind != "rest" else None,
                    rhythm=rhythm,
                    fermata=fermata,
                )
            )
            if kind == "grace":
                features.add("grace")
            if fermata:
                features.add("fermata")
            tie_count += sum(tie.get("type") == "start" for tie in note.findall("tie"))

    return SemanticScore(
        events=tuple(events),
        measure_count=len(measures),
        clef=clef,
        fifths=fifths,
        time_signature=time_signature,
        tie_count=tie_count,
        token_features=tuple(sorted(features)),
    )


def _alignment(
    reference: Sequence[SemanticEvent],
    candidate: Sequence[SemanticEvent],
) -> tuple[int, tuple[tuple[int | None, int | None], ...]]:
    rows, columns = len(reference) + 1, len(candidate) + 1
    costs = [[0] * columns for _ in range(rows)]
    back: list[list[str | None]] = [[None] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], back[i][0] = i, "delete"
    for j in range(1, columns):
        costs[0][j], back[0][j] = j, "insert"
    for i in range(1, rows):
        for j in range(1, columns):
            substitution = costs[i - 1][j - 1] + int(reference[i - 1] != candidate[j - 1])
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            best = min(substitution, deletion, insertion)
            costs[i][j] = best
            # Prefer aligned substitutions over delete+insert when costs tie.
            back[i][j] = "align" if substitution == best else ("delete" if deletion == best else "insert")

    aligned: list[tuple[int | None, int | None]] = []
    i, j = len(reference), len(candidate)
    while i or j:
        operation = back[i][j]
        if operation == "align":
            i -= 1
            j -= 1
            aligned.append((i, j))
        elif operation == "delete":
            i -= 1
            aligned.append((i, None))
        else:
            j -= 1
            aligned.append((None, j))
    aligned.reverse()
    return costs[-1][-1], tuple(aligned)


def compare_primus_semantics(reference: SemanticScore, candidate: SemanticScore) -> dict[str, object]:
    edits, alignment = _alignment(reference.events, candidate.events)
    aligned_pairs = [(reference.events[i], candidate.events[j]) for i, j in alignment if i is not None and j is not None]
    matched = len(aligned_pairs)
    deleted = sum(j is None for _, j in alignment)
    inserted = sum(i is None for i, _ in alignment)
    pitched = [(left, right) for left, right in aligned_pairs if left.kind in {"note", "grace"}]
    rhythmic = [(left, right) for left, right in aligned_pairs if left.rhythm is not None]

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 9) if denominator else 1.0

    header_items = (
        (reference.clef, candidate.clef),
        (reference.fifths, candidate.fifths),
        (reference.time_signature, candidate.time_signature),
    )
    available_headers = [(left, right) for left, right in header_items if left is not None]
    report = {
        "reference_event_count": len(reference.events),
        "candidate_event_count": len(candidate.events),
        "event_edit_count": edits,
        "event_error_rate": ratio(edits, len(reference.events)),
        "event_presence_precision": ratio(matched, matched + inserted),
        "event_presence_recall": ratio(matched, matched + deleted),
        "event_kind_accuracy": ratio(sum(left.kind == right.kind for left, right in aligned_pairs), len(reference.events)),
        "pitch_accuracy": ratio(sum(left.pitch == right.pitch for left, right in pitched), len(pitched)),
        "rhythm_accuracy": ratio(sum(left.rhythm == right.rhythm for left, right in rhythmic), len(rhythmic)),
        "fermata_accuracy": ratio(sum(left.fermata == right.fermata for left, right in aligned_pairs), len(aligned_pairs)),
        "tie_count_exact": reference.tie_count == candidate.tie_count,
        "measure_count_exact": reference.measure_count == candidate.measure_count,
        "clef_exact": reference.clef is None or reference.clef == candidate.clef,
        "key_signature_exact": reference.fifths is None or reference.fifths == candidate.fifths,
        "time_signature_exact": reference.time_signature is None or reference.time_signature == candidate.time_signature,
        "header_accuracy": ratio(sum(left == right for left, right in available_headers), len(available_headers)),
        "semantic_exact": (
            reference.events == candidate.events
            and reference.measure_count == candidate.measure_count
            and all(left is None or left == right for left, right in header_items)
            and reference.tie_count == candidate.tie_count
        ),
        "reference_features": list(reference.token_features),
        "candidate_features": list(candidate.token_features),
    }
    return report


def aggregate_primus_reports(reports: Iterable[dict[str, object]]) -> dict[str, object]:
    items = list(reports)
    successful = [item for item in items if not item.get("error")]
    numeric_rates = (
        "event_error_rate",
        "event_presence_precision",
        "event_presence_recall",
        "event_kind_accuracy",
        "pitch_accuracy",
        "rhythm_accuracy",
        "fermata_accuracy",
        "header_accuracy",
    )
    exact_fields = (
        "semantic_exact",
        "measure_count_exact",
        "clef_exact",
        "key_signature_exact",
        "time_signature_exact",
        "tie_count_exact",
    )
    aggregate: dict[str, object] = {
        "case_count": len(items),
        "successful_case_count": len(successful),
        "failed_case_count": len(items) - len(successful),
        "total_reference_event_count": sum(
            int(item["reference_event_count"]) for item in successful
        ),
        "total_event_edit_count": sum(
            int(item["event_edit_count"]) for item in successful
        ),
    }
    total_reference_events = int(aggregate["total_reference_event_count"])
    aggregate["micro_event_error_rate"] = (
        round(int(aggregate["total_event_edit_count"]) / total_reference_events, 9)
        if total_reference_events
        else (0.0 if successful else None)
    )
    for field in numeric_rates:
        aggregate[f"mean_{field}"] = (
            round(sum(float(item[field]) for item in successful) / len(successful), 9)
            if successful
            else None
        )
    for field in exact_fields:
        aggregate[f"{field}_rate"] = (
            round(sum(bool(item[field]) for item in successful) / len(successful), 9)
            if successful
            else None
        )
    return aggregate
