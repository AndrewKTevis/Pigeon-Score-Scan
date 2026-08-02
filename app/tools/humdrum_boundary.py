from __future__ import annotations

"""Conservative Humdrum topology analysis for the frozen product boundary.

This module does not decide whether a scan or transcription may be used for
training or release evaluation.  It only proves that the encoded score shape
fits the closed structural boundary.  Missing topology is rejected instead of
being guessed.
"""

from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Iterable

from scorescan.product_scope import (
    MAXIMUM_KEYBOARD_PARTS,
    MAXIMUM_KEYBOARD_STAVES,
    MAXIMUM_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE,
    MAXIMUM_NON_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE,
    MAXIMUM_PHYSICAL_STAVES_PER_SYSTEM,
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)


DISALLOWED_SEMANTIC_SPINE_TYPES = frozenset(
    {
        "**text",
        "**silbe",
        "**harm",
        "**fb",
        "**fba",
    }
)
STAFF_PATTERN = re.compile(r"^\*staff(\d+)$", re.IGNORECASE)
PART_PATTERN = re.compile(r"^\*part(\d+)$", re.IGNORECASE)
METER_PATTERN = re.compile(r"^\*M(?!M)([^ \t]+)$")

KEYBOARD_MARKERS = (
    "piano",
    "pianoforte",
    "fortepian",
    "keyboard",
    "organ",
    "organo",
    "harmonium",
    "harpsichord",
    "cembalo",
    "clavicembalo",
    "clavichord",
    "klav",
)
PERCUSSION_MARKERS = (
    "percussion",
    "timp",
    "drum",
    "cymbal",
    "idio",
    "membr",
)


@dataclass(frozen=True)
class SpineState:
    exclusive_type: str = ""
    staff: int | None = None
    part: int | None = None
    instrument_tokens: frozenset[str] = field(default_factory=frozenset)
    meter: str = ""

    @property
    def is_kern(self) -> bool:
        return self.exclusive_type == "**kern"


@dataclass(frozen=True)
class HumdrumBoundary:
    contract_version: str
    accepted: bool
    reasons: tuple[str, ...]
    score_shape: str
    part_staff_counts: tuple[int, ...]
    physical_staff_count: int
    part_count: int
    keyboard_part_count: int
    maximum_voices_per_staff: int
    maximum_voices_per_keyboard_staff: int
    maximum_voices_per_non_keyboard_staff: int
    exclusive_types: tuple[str, ...]
    true_polymeter: bool
    structurally_valid: bool
    topology_complete: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "score_shape": self.score_shape,
            "part_staff_counts": list(self.part_staff_counts),
            "counts": {
                "physical_staves": self.physical_staff_count,
                "parts": self.part_count,
                "keyboard_parts": self.keyboard_part_count,
                "maximum_voices_per_staff": self.maximum_voices_per_staff,
                "maximum_voices_per_keyboard_staff": (
                    self.maximum_voices_per_keyboard_staff
                ),
                "maximum_voices_per_non_keyboard_staff": (
                    self.maximum_voices_per_non_keyboard_staff
                ),
            },
            "exclusive_types": list(self.exclusive_types),
            "true_polymeter": self.true_polymeter,
            "structurally_valid": self.structurally_valid,
            "topology_complete": self.topology_complete,
        }


def reference_records(lines: Iterable[str]) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in lines:
        if not line.startswith("!!!") or ":" not in line:
            continue
        key, value = line[3:].split(":", 1)
        records.setdefault(key.strip(), value.strip())
    return records


def _is_instrument_token(token: str) -> bool:
    folded = token.casefold()
    return (
        folded.startswith("*i")
        and not folded.startswith(("*itrd", "*ig", "*iphrase"))
    )


def _update_interpretation(
    state: SpineState,
    token: str,
) -> SpineState:
    staff = STAFF_PATTERN.fullmatch(token)
    if staff is not None:
        return replace(state, staff=int(staff.group(1)))
    part = PART_PATTERN.fullmatch(token)
    if part is not None:
        return replace(state, part=int(part.group(1)))
    meter = METER_PATTERN.fullmatch(token)
    if meter is not None:
        return replace(state, meter=meter.group(1))
    if token.startswith("**"):
        return replace(state, exclusive_type=token)
    if _is_instrument_token(token):
        return replace(
            state,
            instrument_tokens=state.instrument_tokens | {token.casefold()},
        )
    return state


def _merge_states(states: list[SpineState]) -> tuple[SpineState, bool]:
    if not states:
        return SpineState(), False
    first = states[0]
    compatible = all(
        state.exclusive_type == first.exclusive_type
        and state.staff == first.staff
        and state.part == first.part
        for state in states[1:]
    )
    instruments = frozenset(
        token for state in states for token in state.instrument_tokens
    )
    meters = {state.meter for state in states if state.meter}
    compatible = compatible and len(meters) <= 1
    return (
        replace(
            first,
            instrument_tokens=instruments,
            meter=next(iter(meters), ""),
        ),
        compatible,
    )


def _apply_manipulators(
    active: list[SpineState],
    fields: list[str],
) -> tuple[list[SpineState], bool]:
    if len(active) != len(fields):
        return active, False

    exchanged = [index for index, token in enumerate(fields) if token == "*x"]
    if exchanged:
        if len(exchanged) != 2:
            return active, False
        left, right = exchanged
        active[left], active[right] = active[right], active[left]

    result: list[SpineState] = []
    index = 0
    valid = True
    while index < len(active):
        token = fields[index]
        state = active[index]
        if token == "*^":
            result.extend((state, state))
        elif token == "*+":
            result.extend((state, SpineState()))
        elif token == "*-":
            pass
        elif token == "*v":
            end = index + 1
            while end < len(active) and fields[end] == "*v":
                end += 1
            merged, compatible = _merge_states(active[index:end])
            if end - index < 2:
                compatible = False
            result.append(merged)
            valid = valid and compatible
            index = end - 1
        else:
            result.append(state)
        index += 1
    return result, valid


def _markers(states: Iterable[SpineState]) -> str:
    return " ".join(
        token for state in states for token in state.instrument_tokens
    )


def _contains_marker(value: str, markers: tuple[str, ...]) -> bool:
    folded = value.casefold()
    return any(marker in folded for marker in markers)


def _keyboard_only_instrumentation(value: str) -> bool:
    folded = value.casefold()
    if not _contains_marker(folded, KEYBOARD_MARKERS):
        return False
    keyboard_names = "|".join(
        re.escape(marker)
        for marker in sorted(KEYBOARD_MARKERS, key=len, reverse=True)
    )
    keyboard_count = sum(
        int(match.group(1))
        for match in re.finditer(
            rf"\b(\d+)\s+(?:{keyboard_names})s?\b",
            folded,
        )
    )
    if keyboard_count != 1:
        return False
    remainder = folded
    for marker in sorted(KEYBOARD_MARKERS, key=len, reverse=True):
        remainder = remainder.replace(marker, " ")
    remainder = re.sub(r"\b(?:empty|solo)\b", " ", remainder)
    remainder = re.sub(r"[\d\W_]+", " ", remainder, flags=re.UNICODE)
    return not remainder.strip()


def _informative_instrument_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    informative: list[str] = []
    for token in tokens:
        value = token.casefold()
        if value.startswith("*i"):
            value = value[2:]
        value = value.strip(" \t'\"[]()._-")
        if value and value not in {"empty", "solo"}:
            informative.append(value)
    return tuple(informative)


def analyze_humdrum_boundary(
    path: Path,
    *,
    instrumentation: str = "",
    source_lines: Iterable[str] | None = None,
) -> HumdrumBoundary:
    lines = (
        list(source_lines)
        if source_lines is not None
        else path.read_text(encoding="utf-8-sig").splitlines()
    )
    initial_index = next(
        (index for index, line in enumerate(lines) if line.startswith("**")),
        None,
    )
    if initial_index is None:
        reasons = ("empty_or_non_notated_score",)
        return HumdrumBoundary(
            contract_version=PRODUCTION_BOUNDARY_CONTRACT_VERSION,
            accepted=False,
            reasons=reasons,
            score_shape="unsupported",
            part_staff_counts=(),
            physical_staff_count=0,
            part_count=0,
            keyboard_part_count=0,
            maximum_voices_per_staff=0,
            maximum_voices_per_keyboard_staff=0,
            maximum_voices_per_non_keyboard_staff=0,
            exclusive_types=(),
            true_polymeter=False,
            structurally_valid=False,
            topology_complete=False,
        )

    initial_types = lines[initial_index].split("\t")
    active = [SpineState(exclusive_type=value) for value in initial_types]
    exclusive_types = {value for value in initial_types if value.startswith("**")}
    structurally_valid = bool(active)
    topology_complete = True
    true_polymeter = False
    observed_kern = False
    observed_data = False
    staff_ids_by_part: dict[int, set[int]] = defaultdict(set)
    instrument_tokens_by_part: dict[int, set[str]] = defaultdict(set)
    maximum_voices_by_part_staff: dict[tuple[int, int], int] = defaultdict(int)
    keyboard_only_instrumentation = _keyboard_only_instrumentation(
        instrumentation
    )
    has_explicit_part_tokens = any(
        PART_PATTERN.fullmatch(token)
        for line in lines[initial_index + 1 :]
        if line.startswith("*")
        for token in line.split("\t")
    )
    infer_single_keyboard_part = (
        keyboard_only_instrumentation and not has_explicit_part_tokens
    )

    def effective_part(state: SpineState) -> int | None:
        if state.part is not None:
            return state.part
        if infer_single_keyboard_part and state.staff is not None:
            return 1
        return None

    def observe(*, require_topology: bool) -> None:
        nonlocal observed_kern, topology_complete, true_polymeter
        kern_states = [state for state in active if state.is_kern]
        if kern_states:
            observed_kern = True
        if require_topology and any(
            state.staff is None or effective_part(state) is None
            for state in kern_states
        ):
            topology_complete = False
        meters = {state.meter for state in kern_states if state.meter}
        if len(meters) > 1:
            true_polymeter = True
        grouped: dict[tuple[int, int], int] = defaultdict(int)
        for state in kern_states:
            part = effective_part(state)
            if state.staff is None or part is None:
                continue
            key = (part, state.staff)
            grouped[key] += 1
            staff_ids_by_part[part].add(state.staff)
            instrument_tokens_by_part[part].update(
                state.instrument_tokens
            )
        for key, count in grouped.items():
            maximum_voices_by_part_staff[key] = max(
                maximum_voices_by_part_staff[key],
                count,
            )

    observe(require_topology=False)
    for line in lines[initial_index + 1 :]:
        if not line or line.startswith("!!"):
            continue
        fields = line.split("\t")
        if line.startswith("*"):
            if len(fields) != len(active):
                structurally_valid = False
                continue
            active = [
                _update_interpretation(state, token)
                for state, token in zip(active, fields)
            ]
            exclusive_types.update(
                state.exclusive_type
                for state in active
                if state.exclusive_type.startswith("**")
            )
            active, valid = _apply_manipulators(active, fields)
            structurally_valid = structurally_valid and valid
            observe(require_topology=observed_data)
            continue
        if line.startswith(("!", "=")):
            continue
        observed_data = True
        if len(fields) != len(active):
            structurally_valid = False
        observe(require_topology=True)

    all_instrument_markers = (
        _markers(active)
        + " "
        + " ".join(
            token
            for tokens in instrument_tokens_by_part.values()
            for token in tokens
        )
        + " "
        + instrumentation
    )
    encoded_part_ids = sorted(staff_ids_by_part)
    keyboard_parts: set[int] = set()
    explicit_non_keyboard_multistaff_parts: set[int] = set()
    mixed_identity_parts: set[int] = set()
    missing_instrument_identity_parts: set[int] = set()
    for part_id in encoded_part_ids:
        tokens = " ".join(sorted(instrument_tokens_by_part[part_id]))
        is_keyboard = _contains_marker(tokens, KEYBOARD_MARKERS)
        is_percussion = _contains_marker(tokens, PERCUSSION_MARKERS)
        staff_count = len(staff_ids_by_part[part_id])
        if not _informative_instrument_tokens(
            instrument_tokens_by_part[part_id]
        ):
            missing_instrument_identity_parts.add(part_id)
        if is_keyboard and is_percussion:
            mixed_identity_parts.add(part_id)
        if is_keyboard:
            keyboard_parts.add(part_id)
        elif staff_count > 1 and tokens:
            explicit_non_keyboard_multistaff_parts.add(part_id)
        elif staff_count > 1:
            # This mirrors the MusicXML boundary analyzer when no reliable
            # instrument identity is encoded.
            keyboard_parts.add(part_id)

    if keyboard_only_instrumentation and encoded_part_ids:
        # Some otherwise useful historical encodings represent the two hands,
        # organ pedal or an ossia as separate Humdrum parts.  AIN is the
        # authoritative instrumentation field here: when it proves exactly
        # one keyboard instrument and no second instrument, normalize those
        # encoded parts into the one keyboard part promised by the product.
        keyboard_parts = set(encoded_part_ids)
        explicit_non_keyboard_multistaff_parts.clear()
        missing_instrument_identity_parts.clear()

    maximum_keyboard_voices = max(
        (
            count
            for (part_id, _staff_id), count
            in maximum_voices_by_part_staff.items()
            if part_id in keyboard_parts
        ),
        default=0,
    )
    maximum_non_keyboard_voices = max(
        (
            count
            for (part_id, _staff_id), count
            in maximum_voices_by_part_staff.items()
            if part_id not in keyboard_parts
        ),
        default=0,
    )
    maximum_voices = max(maximum_voices_by_part_staff.values(), default=0)
    if keyboard_only_instrumentation and encoded_part_ids:
        physical_staff_ids = {
            staff_id
            for part_id in encoded_part_ids
            for staff_id in staff_ids_by_part[part_id]
        }
        part_staff_counts = (len(physical_staff_ids),)
        logical_part_count = 1
        logical_keyboard_part_count = 1
    else:
        part_staff_counts = tuple(
            len(staff_ids_by_part[part_id]) for part_id in encoded_part_ids
        )
        logical_part_count = len(encoded_part_ids)
        logical_keyboard_part_count = len(keyboard_parts)
    physical_staff_count = sum(part_staff_counts)

    reasons: list[str] = []
    if not observed_kern or not observed_data:
        reasons.append("empty_or_non_notated_score")
    if not structurally_valid:
        reasons.append("invalid_or_unsupported_spine_structure")
    if not topology_complete:
        reasons.append("missing_part_or_staff_topology")
    if exclusive_types & DISALLOWED_SEMANTIC_SPINE_TYPES:
        reasons.append("lyrics_harmony_or_figured_bass")
    if _contains_marker(all_instrument_markers, PERCUSSION_MARKERS):
        reasons.append("unpitched_or_percussion_notation")
    if true_polymeter:
        reasons.append("true_polymeter")
    if physical_staff_count > MAXIMUM_PHYSICAL_STAVES_PER_SYSTEM:
        reasons.append("more_than_16_physical_staves")
    if logical_keyboard_part_count > MAXIMUM_KEYBOARD_PARTS:
        reasons.append("more_than_one_keyboard_part")
    if any(value > MAXIMUM_KEYBOARD_STAVES for value in part_staff_counts):
        reasons.append("keyboard_part_with_more_than_four_staves")
    if (
        maximum_keyboard_voices
        > MAXIMUM_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE
    ):
        reasons.append(
            "more_than_eight_independent_voices_per_keyboard_staff"
        )
    if (
        maximum_non_keyboard_voices
        > MAXIMUM_NON_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE
    ):
        reasons.append(
            "more_than_one_independent_voice_per_non_keyboard_staff"
        )
    if explicit_non_keyboard_multistaff_parts:
        reasons.append("non_keyboard_multistaff_part")
    if mixed_identity_parts:
        reasons.append("mixed_keyboard_percussion_identity")
    if (
        logical_part_count > 1
        and missing_instrument_identity_parts
    ):
        reasons.append("missing_part_instrument_identity")

    if logical_keyboard_part_count and (
        logical_part_count > logical_keyboard_part_count
    ):
        score_shape = "keyboard_plus_single_staff_ensemble"
    elif logical_keyboard_part_count:
        score_shape = "keyboard"
    elif logical_part_count > 1:
        score_shape = "single_staff_ensemble"
    elif logical_part_count == 1:
        score_shape = "single_staff_solo"
    else:
        score_shape = "unsupported"

    return HumdrumBoundary(
        contract_version=PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        accepted=not reasons,
        reasons=tuple(sorted(set(reasons))),
        score_shape=score_shape,
        part_staff_counts=part_staff_counts,
        physical_staff_count=physical_staff_count,
        part_count=logical_part_count,
        keyboard_part_count=logical_keyboard_part_count,
        maximum_voices_per_staff=maximum_voices,
        maximum_voices_per_keyboard_staff=maximum_keyboard_voices,
        maximum_voices_per_non_keyboard_staff=maximum_non_keyboard_voices,
        exclusive_types=tuple(sorted(exclusive_types)),
        true_polymeter=true_polymeter,
        structurally_valid=structurally_valid,
        topology_complete=topology_complete,
    )
