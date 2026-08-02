from __future__ import annotations

"""Measure-localised OMR rescue for unresolved, already-supported measures.

The complete-page and system-localised candidates remain the primary recognition
sources.  This module creates one additional observation only when two independent
families already support a measure but production policy requires a third family before
an automatic decision.  The crop result is spliced into a complete-page template so it
can pass through the ordinary consensus, model-veto and MusicXML validation pipeline.

A local crop never edits attributes, never applies outside its declared measure, and
never succeeds partially.  It can add evidence, but it cannot lower the existing
three-family permission gates.
"""

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from fractions import Fraction

import cv2
from lxml import etree

from .musicxml import MUSICXML_DOCTYPE, analyze_musicxml, validate_musicxml
from .musicxml_signature import splice_content_signature
from .score_ir import measure_from_xml
from .policy import DEFAULT_POLICY
from .util import atomic_write_bytes, atomic_write_json, sha256_file
from .visual_evidence import VisualMeasureEvidence


VARIANT_PREFIX = "measure_localized:"


@dataclass(frozen=True)
class MeasureCrop:
    measure_index: int
    image_path: str
    source_bbox: tuple[int, int, int, int]
    padded_shape: tuple[int, int]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_bbox"] = list(self.source_bbox)
        payload["padded_shape"] = list(self.padded_shape)
        return payload


@dataclass(frozen=True)
class MeasureCropVariant:
    name: str
    image_path: str
    sha256: str
    pixel_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MeasureLocalizedVariantResult:
    name: str
    image_path: str
    xml_path: str | None
    return_code: int
    elapsed_seconds: float
    valid: bool
    observed_measure_count: int
    note_count: int
    local_rhythm_issue_count: int
    content_signature: str | None = None
    semantic_signature: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MeasureLocalizedResult:
    measure_index: int
    variant: str
    crop: MeasureCrop | None
    xml_path: str | None
    candidate_xml_path: str | None
    return_code: int
    elapsed_seconds: float
    valid: bool
    observed_measure_count: int
    note_count: int
    local_rhythm_issue_count: int = 0
    internal_variant_count: int = 0
    internal_valid_count: int = 0
    winning_exact_support: int = 0
    winning_signature: str | None = None
    internal_variants: tuple[MeasureLocalizedVariantResult, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["crop"] = self.crop.to_dict() if self.crop is not None else None
        payload["internal_variants"] = [item.to_dict() for item in self.internal_variants]
        return payload


def measure_localized_variant(measure_index: int) -> str:
    return f"{VARIANT_PREFIX}{max(1, int(measure_index))}"


def measure_localized_target(variant: str) -> int | None:
    text = str(variant or "").strip().lower()
    if not text.startswith(VARIANT_PREFIX):
        return None
    try:
        value = int(text[len(VARIANT_PREFIX) :])
    except ValueError:
        return None
    return value if value > 0 else None


def candidate_applies_to_measure(variant: str, measure_index: int) -> bool:
    """Return whether a candidate directly observed one complete measure.

    Full-page and system-localised candidates have no sparse target and therefore observe
    every measure.  A measure-localised candidate observes only its declared target; the
    remaining measures in its spliced MusicXML are copied from the template and must not
    be counted as independent evidence.
    """

    target = measure_localized_target(variant)
    return target is None or target == int(measure_index)


def candidate_applies_to_boundary(
    variant: str,
    left_measure_index: int,
    right_measure_index: int,
) -> bool:
    """Return whether a candidate directly observed both sides of a boundary.

    A one-measure rescue crop can never observe an adjacent two-measure boundary.  Its
    complete-page candidate contains copied template XML outside the target measure, so
    allowing it to vote on ties or other cross-measure topology would manufacture a
    false independent family.
    """

    return candidate_applies_to_measure(variant, left_measure_index) and candidate_applies_to_measure(
        variant, right_measure_index
    )


def eligible_measure_indices(consensus: object) -> tuple[int, ...]:
    """Return bounded unresolved measures which already have two-family support."""
    unresolved = set(int(value) for value in getattr(consensus, "unresolved_measure_indices", ()))
    votes = getattr(consensus, "votes", ())
    ranked: list[tuple[int, int, int]] = []
    for vote in votes:
        index = int(getattr(vote, "measure_index", 0))
        if index not in unresolved:
            continue
        exact = int(getattr(vote, "exact_family_support", 0))
        semantic = int(getattr(vote, "semantic_family_support", 0))
        support = max(exact, semantic)
        if support < DEFAULT_POLICY.measure_localized_minimum_existing_families:
            continue
        # Prefer exact two-family near-misses before fuzzy clusters.
        ranked.append((support, exact, -index))
    ranked.sort(reverse=True)
    selected = [-item[2] for item in ranked[: DEFAULT_POLICY.measure_localized_max_measures]]
    return tuple(sorted(selected))


def create_measure_crop(
    image_path: Path,
    evidence: VisualMeasureEvidence,
    output_dir: Path,
) -> MeasureCrop:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("measure-localised recognition cannot read page image")
    height, width = image.shape
    x1, y1, x2, y2 = (int(value) for value in evidence.bbox)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("measure-localised recognition evidence bbox is outside page image")
    spacing = max(3.0, float(evidence.spacing))
    horizontal = max(
        DEFAULT_POLICY.measure_localized_min_context_pixels,
        int(round(spacing * DEFAULT_POLICY.measure_localized_horizontal_context_ratio)),
    )
    left = max(0, x1 - horizontal)
    right = min(width, x2 + horizontal)
    source = image[y1:y2, left:right]
    if source.size == 0:
        raise ValueError("measure-localised recognition produced an empty crop")
    border = max(
        DEFAULT_POLICY.measure_localized_min_border_pixels,
        int(round(spacing * DEFAULT_POLICY.measure_localized_border_ratio)),
    )
    padded = cv2.copyMakeBorder(
        source,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    if padded.size > DEFAULT_POLICY.measure_localized_max_pixels:
        raise ValueError("measure-localised recognition crop budget exceeded")
    ok, encoded = cv2.imencode(".png", padded, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    if not ok:
        raise OSError("failed to encode measure-localised crop")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"measure_{evidence.measure_index:04d}_primary.png"
    atomic_write_bytes(path, encoded.tobytes())
    crop = MeasureCrop(
        measure_index=int(evidence.measure_index),
        image_path=str(path),
        source_bbox=(left, y1, right, y2),
        padded_shape=(int(padded.shape[1]), int(padded.shape[0])),
        sha256=sha256_file(path),
    )
    atomic_write_json(output_dir / f"measure_{evidence.measure_index:04d}_crop.json", crop.to_dict())
    return crop




def create_measure_crop_variants(crop: MeasureCrop, output_dir: Path) -> tuple[MeasureCropVariant, ...]:
    """Create bounded deterministic subvariants for one local measure crop.

    These images are deliberately correlated and therefore never count as separate
    recognition families.  They are used only to establish an internal exact majority
    before the single ``measure_localization:<index>`` family is allowed to vote.
    """

    primary_path = Path(crop.image_path)
    gray = cv2.imread(str(primary_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("measure-localised primary crop is unreadable")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[str, Path, object]] = [("primary", primary_path, gray)]

    kernel_size = max(15, (min(gray.shape) // 10) | 1)
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    flattened = cv2.divide(gray, background, scale=245)
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)
    generated.append(("flat", output_dir / f"measure_{crop.measure_index:04d}_flat.png", flattened))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    generated.append(("otsu", output_dir / f"measure_{crop.measure_index:04d}_otsu.png", otsu))

    total_pixels = sum(int(image.size) for _name, _path, image in generated)
    if total_pixels > DEFAULT_POLICY.measure_localized_max_total_variant_pixels:
        raise ValueError("measure-localised variant crop budget exceeded")

    results: list[MeasureCropVariant] = []
    for name, path, image in generated:
        if name != "primary":
            ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 2])
            if not ok:
                raise OSError(f"failed to encode measure-localised {name} crop")
            atomic_write_bytes(path, encoded.tobytes())
        results.append(
            MeasureCropVariant(
                name=name,
                image_path=str(path),
                sha256=sha256_file(path),
                pixel_count=int(image.size),
            )
        )
    atomic_write_json(
        output_dir / f"measure_{crop.measure_index:04d}_variants.json",
        {
            "format": 1,
            "measure_index": crop.measure_index,
            "total_pixels": total_pixels,
            "variants": [item.to_dict() for item in results],
        },
    )
    return tuple(results)


def measure_localized_semantic_signature(path: Path) -> str:
    """Return the legacy Score-IR signature used only for diagnostics.

    This intentionally omits engraving topology which :class:`MeasureIR` does not model
    (for example beams, stems, notehead shapes and uncommon notations).  It is useful for
    explaining *why* related treatments appear musically close, but is no longer strong
    enough to authorize the one-family internal exact gate.
    """

    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    tree = etree.parse(str(path), parser)
    part = _first_part(tree)
    measures = part.findall("measure")
    if len(measures) != 1:
        raise ValueError("measure-localised signature requires exactly one measure")
    semantic, _state = measure_from_xml(measures[0], {})
    payload = repr(
        (
            tuple(note.stable_tuple() for note in semantic.notes),
            tuple(direction.stable_tuple() for direction in semantic.directions),
            semantic.barlines,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def measure_localized_content_signature(path: Path) -> str:
    """Hash the exact normalised XML content that would actually be spliced.

    The shared canonicaliser covers beams, stems, notehead shapes, fermatas, glissandi,
    technical marks and unknown MusicXML children.  Crop-local attributes and print
    coordinates are excluded because the complete-page template remains authoritative.
    """

    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    tree = etree.parse(str(path), parser)
    part = _first_part(tree)
    measures = part.findall("measure")
    if len(measures) != 1:
        raise ValueError("measure-localised signature requires exactly one measure")
    source_divisions = _source_measure_divisions(measures[0])
    return splice_content_signature(measures[0], source_divisions)

def choose_measure_localized_variant(
    results: tuple[MeasureLocalizedVariantResult, ...],
) -> tuple[MeasureLocalizedVariantResult | None, int, str | None, str | None]:
    """Select one representative only after a strict internal exact majority.

    The subvariants are related image treatments, so their agreement improves the
    reliability of the one local family but never increases independent-family support.
    """

    valid = [item for item in results if item.valid and item.content_signature and item.xml_path]
    if len(valid) < DEFAULT_POLICY.measure_localized_internal_min_valid_variants:
        return None, 0, None, "insufficient_valid_internal_variants"
    groups: dict[str, list[MeasureLocalizedVariantResult]] = {}
    for item in valid:
        assert item.content_signature is not None
        groups.setdefault(item.content_signature, []).append(item)
    ranked = sorted(
        groups.items(),
        key=lambda item: (len(item[1]), -min(results.index(member) for member in item[1])),
        reverse=True,
    )
    winning_signature, members = ranked[0]
    runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
    support = len(members)
    if support < DEFAULT_POLICY.measure_localized_internal_min_exact_support:
        return None, support, winning_signature, "internal_exact_support_low"
    if support - runner_up < DEFAULT_POLICY.measure_localized_internal_min_margin:
        return None, support, winning_signature, "internal_exact_margin_low"
    representative = min(members, key=lambda item: results.index(item))
    return representative, support, winning_signature, None


def validate_measure_localized_xml(path: Path) -> tuple[bool, int, int, int, str | None]:
    """Validate crop output without trusting crop-local meter attributes.

    A one-measure crop does not reliably contain the preceding time/key/clef context.
    Rhythm closure is therefore diagnostic here and becomes a hard gate only after the
    local content is spliced into the already validated page consensus.
    """
    errors = validate_musicxml(path)
    if errors:
        return False, 0, 0, 0, "; ".join(errors[:3])
    try:
        parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
        tree = etree.parse(str(path), parser)
        part = _first_part(tree)
        measures = part.findall("measure")
        explicit_divisions = (
            measures[0].findtext("./attributes/divisions") if len(measures) == 1 else None
        )
        analysis = analyze_musicxml(path)
        measure_count = int(analysis.get("measure_count", 0) or 0)
        note_count = int(analysis.get("note_count", 0) or 0)
        rhythm_issue_count = len(analysis.get("rhythm_issues", ()) or ())
    except Exception as exc:
        return False, 0, 0, 0, f"MusicXML parse failed: {exc}"
    if measure_count != 1:
        return False, measure_count, note_count, rhythm_issue_count, "measure-localised output must contain exactly one measure"
    if note_count <= 0:
        return False, measure_count, note_count, rhythm_issue_count, "measure-localised output contains no notes or rests"
    if explicit_divisions is None or _positive_integer(explicit_divisions, 0) <= 0:
        return False, measure_count, note_count, rhythm_issue_count, "measure-localised output must declare explicit positive divisions"
    return True, measure_count, note_count, rhythm_issue_count, None


def _first_part(tree: etree._ElementTree) -> etree._Element:
    part = tree.getroot().find("part")
    if part is None:
        raise ValueError("MusicXML part missing")
    return part




def _normalized_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).casefold()


def _integer_text(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value or "").strip())
    except (TypeError, ValueError):
        return default


def _clef_context(node: etree._Element) -> tuple[object, ...]:
    return (
        _normalized_text(node.get("number") or "1"),
        _normalized_text(node.findtext("sign")),
        _integer_text(node.findtext("line"), 0),
        _integer_text(node.findtext("clef-octave-change"), 0),
        _normalized_text(node.get("additional")),
        _normalized_text(node.get("after-barline")),
    )


def _key_context(node: etree._Element) -> tuple[object, ...]:
    traditional = (
        _integer_text(node.findtext("cancel"), 0),
        _integer_text(node.findtext("fifths"), 0),
        _normalized_text(node.findtext("mode")),
    )
    nontraditional: list[tuple[object, ...]] = []
    steps = node.findall("key-step")
    alters = node.findall("key-alter")
    accidentals = node.findall("key-accidental")
    octaves = node.findall("key-octave")
    for index, step in enumerate(steps):
        alter = alters[index].text if index < len(alters) else ""
        accidental = accidentals[index].text if index < len(accidentals) else ""
        octave = octaves[index].text if index < len(octaves) else ""
        number = octaves[index].get("number", "") if index < len(octaves) else ""
        nontraditional.append(
            (
                _normalized_text(step.text),
                _normalized_text(alter),
                _normalized_text(accidental),
                _integer_text(octave, 0),
                _normalized_text(number),
            )
        )
    return (
        _normalized_text(node.get("number") or "1"),
        traditional,
        tuple(nontraditional),
    )


def _time_context(node: etree._Element) -> tuple[object, ...]:
    pairs = tuple(
        (
            _normalized_text(beats.text),
            _normalized_text(beat_type.text if beat_type is not None else ""),
        )
        for beats, beat_type in zip(
            node.findall("beats"),
            node.findall("beat-type"),
            strict=False,
        )
    )
    interchangeable = node.find("interchangeable")
    interchangeable_payload: tuple[object, ...] | None = None
    if interchangeable is not None:
        interchangeable_payload = (
            _normalized_text(interchangeable.get("symbol")),
            _normalized_text(interchangeable.get("separator")),
            tuple(_normalized_text(item.text) for item in interchangeable.findall("beats")),
            tuple(_normalized_text(item.text) for item in interchangeable.findall("beat-type")),
            _normalized_text(interchangeable.findtext("time-relation")),
        )
    return (
        _normalized_text(node.get("number") or "1"),
        _normalized_text(node.get("symbol")),
        _normalized_text(node.get("separator")),
        _normalized_text(node.findtext("senza-misura")),
        pairs,
        interchangeable_payload,
    )


def _transpose_context(node: etree._Element) -> tuple[object, ...]:
    return (
        _normalized_text(node.get("number") or "1"),
        _integer_text(node.findtext("diatonic"), 0),
        _integer_text(node.findtext("chromatic"), 0),
        _integer_text(node.findtext("octave-change"), 0),
        node.find("double") is not None,
    )


def _update_notation_context(
    state: dict[str, tuple[tuple[object, ...], ...]],
    attributes: etree._Element,
) -> None:
    for name, extractor in (
        ("clef", _clef_context),
        ("key", _key_context),
        ("time", _time_context),
        ("transpose", _transpose_context),
    ):
        nodes = attributes.findall(name)
        if nodes:
            state[name] = tuple(extractor(node) for node in nodes)


def _template_notation_context(
    measures: list[etree._Element],
    target_index: int,
) -> dict[str, tuple[tuple[object, ...], ...]]:
    state: dict[str, tuple[tuple[object, ...], ...]] = {}
    for previous in measures[:target_index]:
        for attributes in previous.findall("attributes"):
            _update_notation_context(state, attributes)

    performed_content_seen = False
    for child in measures[target_index]:
        local_tag = etree.QName(child).localname
        if local_tag == "print":
            continue
        if local_tag == "attributes":
            if performed_content_seen:
                raise ValueError(
                    "template measure has mid-measure attributes unsafe for local context"
                )
            _update_notation_context(state, child)
            continue
        performed_content_seen = True
    return state


def _localized_notation_context(
    measure: etree._Element,
) -> dict[str, tuple[tuple[object, ...], ...]]:
    state: dict[str, tuple[tuple[object, ...], ...]] = {}
    performed_content_seen = False
    for child in measure:
        local_tag = etree.QName(child).localname
        if local_tag == "print":
            continue
        if local_tag == "attributes":
            context_nodes = any(child.findall(name) for name in ("clef", "key", "time", "transpose"))
            if performed_content_seen and context_nodes:
                raise ValueError(
                    "measure-localised MusicXML changes notation context after performed content"
                )
            if not performed_content_seen:
                _update_notation_context(state, child)
            continue
        performed_content_seen = True
    return state


def _is_default_context(name: str, value: tuple[tuple[object, ...], ...]) -> bool:
    if name == "clef":
        return value == (("1", "g", 2, 0, "", ""),)
    if name == "key":
        return value == (("1", (0, 0, ""), ()),)
    if name == "time":
        return value == (("1", "", "", "", (("4", "4"),), None),)
    if name == "transpose":
        return value == (("1", 0, 0, 0, False),)
    return False


def validate_measure_localized_context(
    localized_path: Path,
    template_path: Path,
    measure_index: int,
) -> tuple[bool, str | None]:
    """Verify that crop-local pitch and meter context matches the page template.

    Crop attributes are not written to the result, but they influence how the OMR engine
    interpreted staff positions and durations.  Conflicting or missing non-default
    context therefore invalidates the local observation instead of being silently
    discarded during the splice.
    """

    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    try:
        localized_tree = etree.parse(str(localized_path), parser)
        template_tree = etree.parse(str(template_path), parser)
        localized_measures = _first_part(localized_tree).findall("measure")
        template_measures = _first_part(template_tree).findall("measure")
        index = int(measure_index) - 1
        if len(localized_measures) != 1 or index < 0 or index >= len(template_measures):
            return False, "notation context target is invalid"
        local = _localized_notation_context(localized_measures[0])
        expected = _template_notation_context(template_measures, index)
    except Exception as exc:
        return False, f"notation context parse failed: {exc}"

    for name in ("clef", "key", "time", "transpose"):
        actual = local.get(name)
        wanted = expected.get(name)
        if actual is not None:
            if wanted is None:
                if not _is_default_context(name, actual):
                    return False, f"local {name} context has no matching template context"
            elif actual != wanted:
                return False, f"local {name} context conflicts with template"
        elif wanted is not None and not _is_default_context(name, wanted):
            return False, f"local {name} context is missing for non-default template"
    return True, None


def _positive_integer(value: str | None, default: int = 1) -> int:
    try:
        parsed = int((value or "").strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _attribute_divisions(attributes: etree._Element, *, source: str) -> int | None:
    text = attributes.findtext("divisions")
    if text is None:
        return None
    parsed = _positive_integer(text, 0)
    if parsed <= 0:
        raise ValueError(f"{source} MusicXML has invalid divisions")
    return parsed


def _inherited_template_divisions(measures: list[etree._Element], target_index: int) -> int:
    """Return target divisions only when target attributes can be preserved in place.

    The local splice keeps the complete-page attributes and replaces performed content.
    Moving an attribute change which originally occurs after a note to the beginning of
    the measure would alter clef/key/meter/divisions semantics.  Such measures therefore
    abstain from local rescue rather than attempting an unsafe reordering.
    """

    divisions: int | None = None
    for previous in measures[:target_index]:
        for attributes in previous.findall("attributes"):
            parsed = _attribute_divisions(attributes, source="template")
            if parsed is not None:
                divisions = parsed

    target = measures[target_index]
    performed_content_seen = False
    for child in target:
        local_tag = etree.QName(child).localname
        if local_tag == "print":
            continue
        if local_tag == "attributes":
            if performed_content_seen:
                raise ValueError(
                    "template measure has mid-measure attributes unsafe for local splice"
                )
            parsed = _attribute_divisions(child, source="template")
            if parsed is not None:
                divisions = parsed
            continue
        performed_content_seen = True

    if divisions is None:
        raise ValueError("template MusicXML has no inherited divisions for local splice")
    return divisions


def _source_measure_divisions(measure: etree._Element) -> int:
    """Require one stable divisions unit before all crop-local performed content."""

    divisions: int | None = None
    performed_content_seen = False
    for child in measure:
        local_tag = etree.QName(child).localname
        if local_tag == "print":
            continue
        if local_tag == "attributes":
            parsed = _attribute_divisions(child, source="measure-localised")
            if parsed is None:
                continue
            if performed_content_seen:
                raise ValueError(
                    "measure-localised MusicXML changes divisions after performed content"
                )
            if divisions is not None and parsed != divisions:
                raise ValueError(
                    "measure-localised MusicXML contains conflicting divisions"
                )
            divisions = parsed
            continue
        performed_content_seen = True

    if divisions is None:
        raise ValueError("measure-localised MusicXML must declare explicit divisions")
    return divisions


def _scale_division_value(text: str | None, source_divisions: int, target_divisions: int) -> str:
    try:
        source_value = int((text or "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("measure-localised duration/offset is not an integer") from exc
    scaled = Fraction(source_value * target_divisions, source_divisions)
    if scaled.denominator != 1:
        raise ValueError(
            "measure-localised duration cannot be represented exactly in template divisions"
        )
    return str(scaled.numerator)


def _copy_scaled_local_content(
    source: etree._Element,
    *,
    source_divisions: int,
    target_divisions: int,
) -> list[etree._Element]:
    children: list[etree._Element] = []
    for child in source:
        if child.tag in {"print", "attributes"}:
            continue
        copied = copy.deepcopy(child)
        if source_divisions != target_divisions:
            nodes = list(copied.iter("duration")) + list(copied.iter("offset"))
            for node in nodes:
                node.text = _scale_division_value(
                    node.text,
                    source_divisions,
                    target_divisions,
                )
        children.append(copied)
    return children


def splice_measure_candidate(
    template_path: Path,
    localized_path: Path,
    measure_index: int,
    output_path: Path,
) -> None:
    """Create a complete candidate with one local musical-content replacement.

    Page/system layout directives and inherited attributes come from the complete-page
    template.  The local crop contributes only notes, directions, forwards/backups and
    barlines.  This prevents a crop-local clef or time-signature guess from leaking into
    the page score.
    """
    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    template_tree = etree.parse(str(template_path), parser)
    localized_tree = etree.parse(str(localized_path), parser)
    template_part = _first_part(template_tree)
    localized_part = _first_part(localized_tree)
    template_measures = template_part.findall("measure")
    localized_measures = localized_part.findall("measure")
    index = int(measure_index) - 1
    if index < 0 or index >= len(template_measures):
        raise ValueError("measure-localised target is outside template")
    if len(localized_measures) != 1:
        raise ValueError("measure-localised source must contain exactly one measure")

    target = template_measures[index]
    source = localized_measures[0]
    context_valid, context_error = validate_measure_localized_context(
        localized_path, template_path, measure_index
    )
    if not context_valid:
        raise ValueError(context_error or "measure-localised notation context mismatch")
    target_divisions = _inherited_template_divisions(template_measures, index)
    source_divisions = _source_measure_divisions(source)
    replacement = etree.Element("measure")
    for key, value in target.attrib.items():
        replacement.set(key, value)
    # Preserve page/system print information and inherited attributes from the complete
    # page.  All performed and printed measure content comes from the local crop.
    for child in target:
        if child.tag in {"print", "attributes"}:
            replacement.append(copy.deepcopy(child))
    for child in _copy_scaled_local_content(
        source,
        source_divisions=source_divisions,
        target_divisions=target_divisions,
    ):
        replacement.append(child)
    template_part.replace(target, replacement)

    payload = etree.tostring(
        template_tree,
        encoding="UTF-8",
        xml_declaration=True,
        pretty_print=True,
        doctype=MUSICXML_DOCTYPE,
    )
    atomic_write_bytes(output_path, payload)
    errors = validate_musicxml(output_path)
    if errors:
        output_path.unlink(missing_ok=True)
        raise ValueError("spliced measure-localised candidate is invalid: " + "; ".join(errors[:3]))
