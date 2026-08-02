from __future__ import annotations

"""Immutable semantic representation used by validation and ensemble selection.

ScoreScan deliberately keeps MusicXML as an interchange format rather than using the
XML tree as its internal truth.  This module extracts the subset of common Western
notation that the first product profile supports and normalises timing to quarter-note
fractions, allowing candidates with different ``divisions`` values to be compared.
"""

import hashlib
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from lxml import etree

from .product_scope import (
    MAXIMUM_KEYBOARD_PARTS,
    MAXIMUM_KEYBOARD_STAVES,
    MAXIMUM_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE,
    MAXIMUM_NON_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE,
    MAXIMUM_PHYSICAL_STAVES_PER_SYSTEM,
)

_STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_TYPE_QUARTERS: dict[str, Fraction] = {
    "maxima": Fraction(32, 1),
    "long": Fraction(16, 1),
    "breve": Fraction(8, 1),
    "whole": Fraction(4, 1),
    "half": Fraction(2, 1),
    "quarter": Fraction(1, 1),
    "eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
    "128th": Fraction(1, 32),
    "256th": Fraction(1, 64),
    "512th": Fraction(1, 128),
    "1024th": Fraction(1, 256),
}
_WORD_NORMALIZER = re.compile(r"\s+")


def _text(value: str | None) -> str:
    return (value or "").strip()


def _integer(value: str | None, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def normalize_words(value: str) -> str:
    value = value.strip().replace("’", "'").replace("`", "'")
    value = value.replace("—", "-").replace("–", "-")
    return _WORD_NORMALIZER.sub(" ", value).casefold().strip(" ,;:()[]{}")


@dataclass(frozen=True, order=True)
class PitchIR:
    step: str
    alter: Fraction
    octave: int

    @property
    def midi_cents(self) -> int:
        semitone = _STEP_TO_SEMITONE.get(self.step.upper(), 0) + 12 * (self.octave + 1)
        return int(round((Fraction(semitone, 1) + self.alter) * 100))

    def stable_tuple(self) -> tuple[object, ...]:
        return self.step.upper(), _fraction_text(self.alter), self.octave


@dataclass(frozen=True)
class NoteIR:
    onset: Fraction
    duration: Fraction
    voice: str
    pitch: PitchIR | None
    rest: bool
    chord: bool
    grace: bool
    note_type: str
    dots: int
    accidental: str
    ties: tuple[str, ...]
    slurs: tuple[tuple[str, str], ...]
    articulations: tuple[str, ...]
    ornaments: tuple[str, ...]
    tuple_ratio: tuple[int, int] | None
    lyrics: tuple[tuple[str, str, str], ...] = ()
    staff: int = 1

    def stable_tuple(self) -> tuple[object, ...]:
        # Semantic lyrics are deliberately retained on the IR for offline
        # diagnostics, but they are outside the supported product boundary.
        # Excluding them here prevents lyric-only differences from splitting
        # consensus groups or blocking an otherwise safe in-boundary patch.
        return (
            _fraction_text(self.onset),
            _fraction_text(self.duration),
            self.voice,
            self.pitch.stable_tuple() if self.pitch else None,
            self.rest,
            self.chord,
            self.grace,
            self.note_type,
            self.dots,
            self.accidental,
            self.ties,
            self.slurs,
            self.articulations,
            self.ornaments,
            self.tuple_ratio,
            self.staff,
        )


@dataclass(frozen=True)
class DirectionIR:
    onset: Fraction
    placement: str
    kind: str
    value: str
    end_value: str = ""
    staff: int = 1
    voice: str = ""

    def stable_tuple(self) -> tuple[object, ...]:
        return (
            _fraction_text(self.onset),
            self.placement,
            self.kind,
            normalize_words(self.value),
            normalize_words(self.end_value),
            self.staff,
            self.voice,
        )


@dataclass(frozen=True)
class MeasureIR:
    divisions: int
    time_signature: tuple[int, int] | None
    key_signature: tuple[int, str] | None
    clef: tuple[str, int, int] | None
    notes: tuple[NoteIR, ...]
    directions: tuple[DirectionIR, ...]
    barlines: tuple[tuple[str, str, str, str], ...]
    number: str = ""
    staff_count: int = 1
    staff_clefs: tuple[tuple[int, str, int, int], ...] = ()
    transpose: tuple[int, int, int] | None = None

    @property
    def voice_count(self) -> int:
        return len({note.voice for note in self.notes if not note.grace})

    def voice_count_for_staff(self, staff: int) -> int:
        return len({note.voice for note in self.notes if note.staff == staff and not note.grace})

    def maximum_simultaneous_voices_for_staff(self, staff: int) -> int:
        """Count overlapping voice timelines, not sequentially reused labels."""

        intervals_by_voice: dict[str, list[tuple[Fraction, Fraction]]] = {}
        for note in self.notes:
            if (
                note.staff != staff
                or note.grace
                or note.duration <= 0
            ):
                continue
            intervals_by_voice.setdefault(note.voice, []).append(
                (note.onset, note.onset + note.duration)
            )
        boundaries = sorted(
            {
                boundary
                for intervals in intervals_by_voice.values()
                for interval in intervals
                for boundary in interval
            }
        )
        return max(
            (
                sum(
                    any(start <= point < end for start, end in intervals)
                    for intervals in intervals_by_voice.values()
                )
                for point in boundaries[:-1]
            ),
            default=0,
        )

    @property
    def staff_numbers(self) -> tuple[int, ...]:
        return tuple(sorted({note.staff for note in self.notes} | {direction.staff for direction in self.directions}))

    @property
    def expected_duration(self) -> Fraction | None:
        if not self.time_signature:
            return None
        beats, beat_type = self.time_signature
        if beat_type <= 0:
            return None
        return Fraction(beats * 4, beat_type)

    @property
    def fingerprint(self) -> str:
        payload = repr(
            (
                self.time_signature,
                self.key_signature,
                self.clef,
                tuple(note.stable_tuple() for note in self.notes),
                tuple(direction.stable_tuple() for direction in self.directions),
                self.barlines,
                self.number,
                self.staff_count,
                self.staff_clefs,
                self.transpose,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]


@dataclass(frozen=True)
class PartIR:
    id: str
    name: str
    abbreviation: str
    measures: tuple[MeasureIR, ...]
    staff_count: int = 1
    transposition: tuple[int, int, int] | None = None

    @property
    def is_keyboard_part(self) -> bool:
        """Whether the part uses a multi-staff keyboard-style notation model."""
        return self.staff_count > 1


@dataclass(frozen=True)
class ScoreIR:
    # ``measures`` remains the first field for compatibility with the former
    # single-part IR and all existing consensus guards.  In a multi-part score it
    # is an alias of the first part's measures; new code must use ``parts``.
    measures: tuple[MeasureIR, ...]
    parts: tuple[PartIR, ...] = ()

    @property
    def effective_parts(self) -> tuple[PartIR, ...]:
        if self.parts:
            return self.parts
        if not self.measures:
            return ()
        staff_count = max((measure.staff_count for measure in self.measures), default=1)
        return (PartIR("P1", "", "", self.measures, staff_count),)


@dataclass(frozen=True)
class SemanticIssue:
    code: str
    message: str
    measure_index: int
    severity: str = "warning"
    part_id: str = ""
    staff: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "measure_index": self.measure_index,
            "severity": self.severity,
        }
        if self.part_id:
            result["part_id"] = self.part_id
        if self.staff is not None:
            result["staff"] = self.staff
        return result


@dataclass(frozen=True)
class AuditPolicy:
    """Structural boundary used by semantic validation.

    The legacy consensus pipeline intentionally keeps its single-staff audit.
    The production policy represents the user-visible scan boundary: a full
    score may contain independent parts, keyboard parts may contain multiple
    rhythmic voices, and no event is allowed to point outside its physical
    staff range.
    """

    max_physical_staves: int = MAXIMUM_PHYSICAL_STAVES_PER_SYSTEM
    max_keyboard_parts: int = MAXIMUM_KEYBOARD_PARTS
    max_keyboard_staves: int = MAXIMUM_KEYBOARD_STAVES
    max_keyboard_voices_per_staff: int = (
        MAXIMUM_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE
    )
    max_non_keyboard_voices_per_staff: int = (
        MAXIMUM_NON_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE
    )
    allow_keyboard_multiple_voices: bool = True
    allow_single_staff_multiple_voices: bool = False


PRODUCTION_SCAN_POLICY = AuditPolicy()


def _parse_fraction(value: str | None) -> Fraction:
    text = _text(value)
    if not text:
        return Fraction(0, 1)
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        try:
            return Fraction(float(text)).limit_denominator(64)
        except (ValueError, OverflowError):
            return Fraction(0, 1)


def _pitch(note: etree._Element) -> PitchIR | None:
    pitch = note.find("pitch")
    if pitch is None:
        return None
    step = _text(pitch.findtext("step")).upper()
    octave = _integer(pitch.findtext("octave"), 4)
    alter = _parse_fraction(pitch.findtext("alter"))
    if step not in _STEP_TO_SEMITONE:
        return None
    return PitchIR(step, alter, octave)


def _direction_ir(direction: etree._Element, cursor: Fraction, divisions: int) -> list[DirectionIR]:
    offset = Fraction(_integer(direction.findtext("offset"), 0), max(divisions, 1))
    onset = max(Fraction(0, 1), cursor + offset)
    placement = _text(direction.get("placement"))
    staff = max(1, _integer(direction.findtext("staff"), 1))
    voice = _text(direction.findtext("voice"))
    result: list[DirectionIR] = []
    for words in direction.findall("./direction-type/words"):
        text = _text(words.text)
        if text:
            result.append(DirectionIR(onset, placement, "words", text, staff=staff, voice=voice))
    for dynamics in direction.findall("./direction-type/dynamics"):
        for child in dynamics:
            value = _text(child.text) or child.tag
            result.append(DirectionIR(onset, placement, "dynamic", value, staff=staff, voice=voice))
    for wedge in direction.findall("./direction-type/wedge"):
        result.append(
            DirectionIR(
                onset,
                placement,
                "wedge",
                _text(wedge.get("type")),
                _text(wedge.get("number")) or "1",
                staff,
                voice,
            )
        )
    for metronome in direction.findall("./direction-type/metronome"):
        unit = _text(metronome.findtext("beat-unit"))
        dotted = "." if metronome.find("beat-unit-dot") is not None else ""
        per_minute = _text(metronome.findtext("per-minute"))
        result.append(
            DirectionIR(onset, placement, "metronome", f"{unit}{dotted}={per_minute}", staff=staff, voice=voice)
        )
    sound = direction.find("sound")
    if sound is not None and sound.get("tempo"):
        result.append(
            DirectionIR(onset, placement, "sound-tempo", _text(sound.get("tempo")), staff=staff, voice=voice)
        )
    return result


def _lyrics(note: etree._Element) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for lyric in note.findall("lyric"):
        text = normalize_words(" ".join((node.text or "") for node in lyric.findall("text")))
        if not text:
            continue
        syllabic = _text(lyric.findtext("syllabic")).casefold()
        extend_node = lyric.find("extend")
        extend = _text(extend_node.get("type") if extend_node is not None else "").casefold()
        rows.append((text, syllabic, extend))
    return tuple(rows)


def measure_from_xml(measure: etree._Element, inherited: dict[str, object] | None = None) -> tuple[MeasureIR, dict[str, object]]:
    state = dict(inherited or {})
    inherited_clefs = state.get("clefs")
    clefs: dict[int, tuple[str, int, int]] = (
        dict(inherited_clefs) if isinstance(inherited_clefs, dict) else {}
    )
    for attributes in measure.findall("attributes"):
        divisions = _integer(attributes.findtext("divisions"), int(state.get("divisions", 1) or 1))
        state["divisions"] = max(divisions, 1)
        staves = _integer(attributes.findtext("staves"), int(state.get("staves", 1) or 1))
        state["staves"] = max(staves, 1)
        time = attributes.find("time")
        if time is not None:
            state["time"] = (_integer(time.findtext("beats"), 4), _integer(time.findtext("beat-type"), 4))
        key = attributes.find("key")
        if key is not None:
            state["key"] = (_integer(key.findtext("fifths"), 0), _text(key.findtext("mode")))
        for clef in attributes.findall("clef"):
            staff_number = max(1, _integer(clef.get("number"), 1))
            clefs[staff_number] = (
                _text(clef.findtext("sign")),
                _integer(clef.findtext("line"), 0),
                _integer(clef.findtext("clef-octave-change"), 0),
            )
        transpose = attributes.find("transpose")
        if transpose is not None:
            state["transpose"] = (
                _integer(transpose.findtext("diatonic"), 0),
                _integer(transpose.findtext("chromatic"), 0),
                _integer(transpose.findtext("octave-change"), 0),
            )
    if clefs:
        state["clefs"] = clefs
        state["clef"] = clefs.get(1) or clefs[min(clefs)]
    divisions = max(1, int(state.get("divisions", 1) or 1))
    cursor = Fraction(0, 1)
    last_anchor_onset = Fraction(0, 1)
    notes: list[NoteIR] = []
    directions: list[DirectionIR] = []

    for child in measure:
        if child.tag == "backup":
            cursor -= Fraction(_integer(child.findtext("duration"), 0), divisions)
            cursor = max(Fraction(0, 1), cursor)
            continue
        if child.tag == "forward":
            cursor += Fraction(_integer(child.findtext("duration"), 0), divisions)
            continue
        if child.tag == "direction":
            directions.extend(_direction_ir(child, cursor, divisions))
            continue
        if child.tag != "note":
            continue
        chord = child.find("chord") is not None
        grace = child.find("grace") is not None
        duration = Fraction(_integer(child.findtext("duration"), 0), divisions)
        onset = last_anchor_onset if chord else cursor
        if not chord:
            last_anchor_onset = onset
        voice = _text(child.findtext("voice")) or "1"
        ties = tuple(sorted(_text(item.get("type")) for item in child.findall("tie") if _text(item.get("type"))))
        tied = tuple(sorted(_text(item.get("type")) for item in child.findall("./notations/tied") if _text(item.get("type"))))
        ties = tuple(sorted(set(ties + tied)))
        slurs = tuple(
            sorted(
                (_text(item.get("type")), _text(item.get("number")) or "1")
                for item in child.findall("./notations/slur")
            )
        )
        articulations = tuple(sorted(item.tag for item in child.findall("./notations/articulations/*")))
        ornaments = tuple(sorted(item.tag for item in child.findall("./notations/ornaments/*")))
        time_modification = child.find("time-modification")
        tuple_ratio = None
        if time_modification is not None:
            actual = _integer(time_modification.findtext("actual-notes"), 0)
            normal = _integer(time_modification.findtext("normal-notes"), 0)
            if actual > 0 and normal > 0:
                tuple_ratio = (actual, normal)
        notes.append(
            NoteIR(
                onset=onset,
                duration=duration,
                voice=voice,
                pitch=_pitch(child),
                rest=child.find("rest") is not None,
                chord=chord,
                grace=grace,
                note_type=_text(child.findtext("type")),
                dots=len(child.findall("dot")),
                accidental=_text(child.findtext("accidental")),
                ties=ties,
                slurs=slurs,
                articulations=articulations,
                ornaments=ornaments,
                tuple_ratio=tuple_ratio,
                lyrics=_lyrics(child),
                staff=max(1, _integer(child.findtext("staff"), 1)),
            )
        )
        if not chord and not grace:
            cursor += duration

    barlines: list[tuple[str, str, str, str]] = []
    for barline in measure.findall("barline"):
        repeat = barline.find("repeat")
        ending = barline.find("ending")
        barlines.append(
            (
                _text(barline.get("location")),
                _text(barline.findtext("bar-style")),
                _text(repeat.get("direction") if repeat is not None else None),
                f"{_text(ending.get('number') if ending is not None else None)}:{_text(ending.get('type') if ending is not None else None)}",
            )
        )
    detected_staff_count = max(
        [max(1, int(state.get("staves", 1) or 1))]
        + [note.staff for note in notes]
        + [direction.staff for direction in directions]
        + list(clefs)
    )
    result = MeasureIR(
        divisions=divisions,
        time_signature=state.get("time") if isinstance(state.get("time"), tuple) else None,
        key_signature=state.get("key") if isinstance(state.get("key"), tuple) else None,
        clef=state.get("clef") if isinstance(state.get("clef"), tuple) else None,
        notes=tuple(notes),
        directions=tuple(sorted(directions, key=lambda item: item.stable_tuple())),
        barlines=tuple(barlines),
        number=_text(measure.get("number")),
        staff_count=detected_staff_count,
        staff_clefs=tuple(
            (staff_number, clef[0], clef[1], clef[2])
            for staff_number, clef in sorted(clefs.items())
        ),
        transpose=state.get("transpose") if isinstance(state.get("transpose"), tuple) else None,
    )
    return result, state


def score_from_tree(tree: etree._ElementTree) -> ScoreIR:
    root = tree.getroot()
    xml_parts = root.findall("part")
    if not xml_parts:
        return ScoreIR(())
    metadata: dict[str, tuple[str, str]] = {}
    for score_part in root.findall("./part-list/score-part"):
        part_id = _text(score_part.get("id"))
        metadata[part_id] = (
            _text(score_part.findtext("part-name")),
            _text(score_part.findtext("part-abbreviation")),
        )
    parts: list[PartIR] = []
    for part_index, xml_part in enumerate(xml_parts, start=1):
        state: dict[str, object] = {}
        measures: list[MeasureIR] = []
        for measure in xml_part.findall("measure"):
            parsed, state = measure_from_xml(measure, state)
            measures.append(parsed)
        part_id = _text(xml_part.get("id")) or f"P{part_index}"
        name, abbreviation = metadata.get(part_id, ("", ""))
        staff_count = max((measure.staff_count for measure in measures), default=1)
        transposition = next((measure.transpose for measure in measures if measure.transpose is not None), None)
        parts.append(
            PartIR(
                id=part_id,
                name=name,
                abbreviation=abbreviation,
                measures=tuple(measures),
                staff_count=staff_count,
                transposition=transposition,
            )
        )
    return ScoreIR(parts[0].measures, tuple(parts))


def expected_note_duration(note: NoteIR) -> Fraction | None:
    base = _TYPE_QUARTERS.get(note.note_type)
    if base is None:
        return None
    multiplier = Fraction(1, 1)
    increment = Fraction(1, 2)
    for _ in range(note.dots):
        multiplier += increment
        increment /= 2
    value = base * multiplier
    if note.tuple_ratio:
        actual, normal = note.tuple_ratio
        value *= Fraction(normal, actual)
    return value


def audit_score(score: ScoreIR) -> tuple[SemanticIssue, ...]:
    issues: list[SemanticIssue] = []
    density_values = [len([note for note in measure.notes if not note.grace]) for measure in score.measures]
    nonzero_density = [value for value in density_values if value > 0]
    median_density = sorted(nonzero_density)[len(nonzero_density) // 2] if nonzero_density else 0

    for index, measure in enumerate(score.measures, start=1):
        if measure.voice_count > 1:
            issues.append(SemanticIssue("multiple_voices", "同一谱表检测到多个节奏声部", index))
        regular = [note for note in measure.notes if not note.grace]
        if not regular:
            issues.append(SemanticIssue("empty_measure", "小节中没有可播放的音符或休止符", index))
        for note in regular:
            if note.duration <= 0:
                issues.append(SemanticIssue("zero_duration", "非倚音事件时值为零", index, "error"))
            expected = expected_note_duration(note)
            if expected is not None and note.duration > 0:
                ratio = float(max(note.duration, expected) / min(note.duration, expected))
                if ratio > 1.08:
                    issues.append(SemanticIssue("type_duration_mismatch", "音符类型与 MusicXML 时值不一致", index))
            if note.pitch and not (1200 <= note.pitch.midi_cents <= 12000):
                issues.append(SemanticIssue("pitch_outlier", "音高超出常见印刷分谱范围", index))
        anchors: dict[Fraction, Fraction] = {}
        for note in regular:
            if not note.chord:
                anchors[note.onset] = note.duration
            elif note.onset in anchors and anchors[note.onset] != note.duration:
                issues.append(SemanticIssue("chord_duration_mismatch", "同一和弦中的音符时值不一致", index))
        if median_density and density_values[index - 1] > max(24, median_density * 5):
            issues.append(SemanticIssue("density_outlier", "小节事件密度明显偏离同页其他小节", index))
        direction_keys = [direction.stable_tuple() for direction in measure.directions]
        if len(direction_keys) != len(set(direction_keys)):
            issues.append(SemanticIssue("duplicate_direction", "同一位置存在重复的速度或力度标记", index))
    return tuple(issues)


def audit_production_score(
    score: ScoreIR,
    policy: AuditPolicy = PRODUCTION_SCAN_POLICY,
) -> tuple[SemanticIssue, ...]:
    """Audit a score without treating valid piano/polyphonic structure as doubt.

    This is deliberately separate from :func:`audit_score`: the latter is used
    by conservative single-staff patch guards and changing its meaning would let
    unsupported structures leak into legacy correction paths.
    """

    parts = score.effective_parts
    if not parts:
        return (SemanticIssue("missing_parts", "乐谱中没有可识别的乐器声部", 0, "error"),)

    issues: list[SemanticIssue] = []
    total_staves = sum(max(1, part.staff_count) for part in parts)
    if total_staves > policy.max_physical_staves:
        issues.append(
            SemanticIssue(
                "physical_staff_limit_exceeded",
                f"每个系统包含 {total_staves} 行谱表，超出高准确度边界 {policy.max_physical_staves} 行",
                0,
                "error",
            )
        )

    keyboard_part_count = sum(part.is_keyboard_part for part in parts)
    if keyboard_part_count > policy.max_keyboard_parts:
        issues.append(
            SemanticIssue(
                "keyboard_part_limit_exceeded",
                (
                    f"总谱包含 {keyboard_part_count} 个键盘部，"
                    f"超出当前边界 {policy.max_keyboard_parts} 个"
                ),
                0,
                "error",
            )
        )

    for part in parts:
        if part.is_keyboard_part and part.staff_count > policy.max_keyboard_staves:
            issues.append(
                SemanticIssue(
                    "keyboard_staff_limit_exceeded",
                    f"键盘声部包含 {part.staff_count} 行谱表，超出当前边界 {policy.max_keyboard_staves} 行",
                    0,
                    "error",
                    part.id,
                )
            )

        # Reuse the mature note/rhythm checks per part while changing only the
        # topology rule that is legitimately different for piano notation.
        legacy_issues = audit_score(ScoreIR(part.measures))
        for issue in legacy_issues:
            if issue.code == "multiple_voices":
                # The legacy check counts distinct labels over a whole measure.
                # Production scope instead counts timelines that truly overlap
                # on one staff, below, so sequential voice-label reuse is valid.
                continue
            issues.append(
                SemanticIssue(
                    issue.code,
                    issue.message,
                    issue.measure_index,
                    issue.severity,
                    part.id,
                    issue.staff,
                )
            )

        for measure_index, measure in enumerate(part.measures, start=1):
            declared_staff_count = max(1, measure.staff_count, part.staff_count)
            for staff_number, sign, _line, _octave_change in measure.staff_clefs:
                if sign.casefold() in {"percussion", "tab"}:
                    issues.append(
                        SemanticIssue(
                            "unsupported_clef",
                            f"当前高准确度边界不包含 {sign or '未知'} 谱号",
                            measure_index,
                            "error",
                            part.id,
                            staff_number,
                        )
                    )
            for note in measure.notes:
                if note.staff < 1 or note.staff > declared_staff_count:
                    issues.append(
                        SemanticIssue(
                            "invalid_note_staff",
                            "音符引用了不存在的谱表",
                            measure_index,
                            "error",
                            part.id,
                            note.staff,
                        )
                    )
            for direction in measure.directions:
                if direction.staff < 1 or direction.staff > declared_staff_count:
                    issues.append(
                        SemanticIssue(
                            "invalid_direction_staff",
                            "文字或记号引用了不存在的谱表",
                            measure_index,
                            "error",
                            part.id,
                            direction.staff,
                        )
                    )
            for staff_number in range(1, declared_staff_count + 1):
                simultaneous_voices = (
                    measure.maximum_simultaneous_voices_for_staff(
                        staff_number
                    )
                )
                if part.is_keyboard_part:
                    maximum_voices = (
                        policy.max_keyboard_voices_per_staff
                        if policy.allow_keyboard_multiple_voices
                        else 1
                    )
                    code = "keyboard_voice_limit_exceeded"
                    message = (
                        f"键盘谱表同时包含 {simultaneous_voices} 个独立声部，"
                        f"超出当前边界 {maximum_voices} 个"
                    )
                else:
                    maximum_voices = (
                        policy.max_non_keyboard_voices_per_staff
                        if not policy.allow_single_staff_multiple_voices
                        else max(
                            policy.max_non_keyboard_voices_per_staff,
                            simultaneous_voices,
                        )
                    )
                    code = "non_keyboard_voice_limit_exceeded"
                    message = (
                        f"单声部乐器谱表同时包含 {simultaneous_voices} 个"
                        f"独立声部，超出当前边界 {maximum_voices} 个"
                    )
                if simultaneous_voices > maximum_voices:
                    issues.append(
                        SemanticIssue(
                            code,
                            message,
                            measure_index,
                            "error",
                            part.id,
                            staff_number,
                        )
                    )
    return tuple(issues)


def _pitch_cost(left: PitchIR | None, right: PitchIR | None, left_rest: bool, right_rest: bool) -> float:
    if left_rest or right_rest:
        return 0.0 if left_rest == right_rest else 1.0
    if left is None or right is None:
        return 0.0 if left is right else 0.8
    distance = abs(left.midi_cents - right.midi_cents)
    if distance == 0:
        return 0.0
    if distance <= 100:
        return 0.25
    if distance <= 200:
        return 0.55
    if distance <= 500:
        return 0.8
    return 1.0


def note_substitution_cost(left: NoteIR, right: NoteIR) -> float:
    cost = 0.0
    cost += 0.48 * _pitch_cost(left.pitch, right.pitch, left.rest, right.rest)
    if left.duration == right.duration:
        duration_cost = 0.0
    elif left.duration <= 0 or right.duration <= 0:
        duration_cost = 1.0
    else:
        duration_cost = min(1.0, abs(math.log2(float(left.duration / right.duration))) / 2.0)
    cost += 0.30 * duration_cost
    onset_scale = max(left.duration, right.duration, Fraction(1, 4))
    onset_cost = min(1.0, float(abs(left.onset - right.onset) / onset_scale))
    cost += 0.10 * onset_cost
    cost += 0.04 if left.grace != right.grace else 0.0
    cost += 0.025 if left.chord != right.chord else 0.0
    cost += 0.02 if normalize_words(left.accidental) != normalize_words(right.accidental) else 0.0
    left_notation = set(left.ties) | {f"slur:{item}" for item in left.slurs} | set(left.articulations) | set(left.ornaments)
    right_notation = set(right.ties) | {f"slur:{item}" for item in right.slurs} | set(right.articulations) | set(right.ornaments)
    if left_notation or right_notation:
        cost += 0.025 * (len(left_notation ^ right_notation) / max(len(left_notation | right_notation), 1))
    # Semantic lyrics are outside the supported product boundary.  They remain
    # available on NoteIR to diagnostic evaluators, but must not steer product
    # alignment, candidate selection, or unresolved-disagreement reporting.
    return min(1.0, cost)


def _sequence_distance(left: tuple[NoteIR, ...], right: tuple[NoteIR, ...]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    previous = [float(index) for index in range(len(right) + 1)]
    for i, left_item in enumerate(left, start=1):
        current = [float(i)]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1.0,
                    previous[j] + 1.0,
                    previous[j - 1] + note_substitution_cost(left_item, right_item),
                )
            )
        previous = current
    return min(1.0, previous[-1] / max(len(left), len(right), 1))


def _string_distance(left: str, right: str) -> float:
    left = normalize_words(left)
    right = normalize_words(right)
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def _direction_distance(left: tuple[DirectionIR, ...], right: tuple[DirectionIR, ...]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    unmatched = list(right)
    costs: list[float] = []
    for item in left:
        if not unmatched:
            costs.append(1.0)
            continue
        ranked = []
        for index, candidate in enumerate(unmatched):
            kind_cost = 0.0 if item.kind == candidate.kind else 0.45
            text_cost = _string_distance(item.value, candidate.value)
            onset_cost = min(1.0, float(abs(item.onset - candidate.onset) / Fraction(1, 1)))
            ranked.append((0.65 * text_cost + 0.20 * kind_cost + 0.15 * onset_cost, index))
        cost, index = min(ranked)
        costs.append(cost)
        unmatched.pop(index)
    costs.extend([1.0] * len(unmatched))
    return sum(costs) / max(len(left), len(right), 1)


def measure_distance(left: MeasureIR, right: MeasureIR) -> float:
    """Return a bounded semantic distance in the range 0..1."""
    note_distance = _sequence_distance(left.notes, right.notes)
    direction_distance = _direction_distance(left.directions, right.directions)
    attribute_cost = 0.0
    attribute_cost += 0.40 if left.time_signature != right.time_signature else 0.0
    attribute_cost += 0.25 if left.key_signature != right.key_signature else 0.0
    attribute_cost += 0.20 if left.clef != right.clef else 0.0
    attribute_cost += 0.15 if left.barlines != right.barlines else 0.0
    return min(1.0, 0.80 * note_distance + 0.12 * direction_distance + 0.08 * min(1.0, attribute_cost))


def pairwise_measure_similarity(left: Iterable[MeasureIR], right: Iterable[MeasureIR]) -> float:
    left_items = tuple(left)
    right_items = tuple(right)
    if len(left_items) != len(right_items) or not left_items:
        return 0.0
    return sum(1.0 - measure_distance(a, b) for a, b in zip(left_items, right_items, strict=True)) / len(left_items)
