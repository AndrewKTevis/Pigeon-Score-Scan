from __future__ import annotations

"""Generate a deterministic, copyright-independent full-score conformance corpus.

The generated pages are intentionally synthetic.  They exercise score topology and
notation-preservation regressions; they are not evidence for real-scan accuracy.
"""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.musicxml import MUSICXML_DOCTYPE  # noqa: E402
from scorescan.preview import _configure_toolkit  # noqa: E402

SEED = 20260726


def _score(part_definitions: list[tuple[str, str]]) -> tuple[etree._Element, dict[str, etree._Element]]:
    root = etree.Element("score-partwise", version="4.0")
    work = etree.SubElement(root, "work")
    etree.SubElement(work, "work-title").text = "ScoreScan generated conformance material"
    part_list = etree.SubElement(root, "part-list")
    parts: dict[str, etree._Element] = {}
    for part_id, name in part_definitions:
        score_part = etree.SubElement(part_list, "score-part", id=part_id)
        etree.SubElement(score_part, "part-name").text = name
        parts[part_id] = etree.SubElement(root, "part", id=part_id)
    return root, parts


def _attributes(
    measure: etree._Element,
    *,
    staves: int = 1,
    clefs: tuple[tuple[str, str], ...] = (("G", "2"),),
    fifths: int = 0,
    beats: int = 4,
    beat_type: int = 4,
) -> None:
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "4"
    key = etree.SubElement(attributes, "key")
    etree.SubElement(key, "fifths").text = str(fifths)
    time = etree.SubElement(attributes, "time")
    etree.SubElement(time, "beats").text = str(beats)
    etree.SubElement(time, "beat-type").text = str(beat_type)
    if staves > 1:
        etree.SubElement(attributes, "staves").text = str(staves)
    for number, (sign, line) in enumerate(clefs, start=1):
        clef = etree.SubElement(attributes, "clef")
        if staves > 1:
            clef.set("number", str(number))
        etree.SubElement(clef, "sign").text = sign
        etree.SubElement(clef, "line").text = line


def _note(
    measure: etree._Element,
    step: str,
    *,
    octave: int = 4,
    duration: int = 4,
    note_type: str = "quarter",
    voice: str = "1",
    staff: int = 1,
    chord: bool = False,
    grace: bool = False,
    tie: str | None = None,
    slur: tuple[str, str] | None = None,
    beam: str | None = None,
    articulations: tuple[str, ...] = (),
    ornaments: tuple[str, ...] = (),
    fingering: str | None = None,
    arpeggiate: bool = False,
) -> etree._Element:
    note = etree.SubElement(measure, "note")
    if chord:
        etree.SubElement(note, "chord")
    if grace:
        etree.SubElement(note, "grace")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = step
    etree.SubElement(pitch, "octave").text = str(octave)
    if not grace:
        etree.SubElement(note, "duration").text = str(duration)
    if tie:
        etree.SubElement(note, "tie", type=tie)
    etree.SubElement(note, "voice").text = voice
    etree.SubElement(note, "type").text = note_type
    etree.SubElement(note, "staff").text = str(staff)
    if beam:
        etree.SubElement(note, "beam", number="1").text = beam
    if tie or slur or articulations or ornaments or fingering or arpeggiate:
        notations = etree.SubElement(note, "notations")
        if tie:
            etree.SubElement(notations, "tied", type=tie)
        if slur:
            etree.SubElement(notations, "slur", type=slur[0], number=slur[1])
        if articulations:
            container = etree.SubElement(notations, "articulations")
            for mark in articulations:
                etree.SubElement(container, mark)
        if ornaments:
            container = etree.SubElement(notations, "ornaments")
            for mark in ornaments:
                etree.SubElement(container, mark)
        if fingering:
            technical = etree.SubElement(notations, "technical")
            etree.SubElement(technical, "fingering").text = fingering
        if arpeggiate:
            etree.SubElement(notations, "arpeggiate")
    return note


def _rest(
    measure: etree._Element,
    *,
    duration: int,
    note_type: str,
    voice: str = "1",
    staff: int = 1,
) -> None:
    note = etree.SubElement(measure, "note")
    etree.SubElement(note, "rest")
    etree.SubElement(note, "duration").text = str(duration)
    etree.SubElement(note, "voice").text = voice
    etree.SubElement(note, "type").text = note_type
    etree.SubElement(note, "staff").text = str(staff)


def _backup(measure: etree._Element, duration: int = 16) -> None:
    backup = etree.SubElement(measure, "backup")
    etree.SubElement(backup, "duration").text = str(duration)


def _direction(
    measure: etree._Element,
    *,
    staff: int = 1,
    words: str | None = None,
    dynamic: str | None = None,
    wedge: str | None = None,
    pedal: str | None = None,
    octave_shift: str | None = None,
) -> None:
    direction = etree.SubElement(measure, "direction", placement="below")
    direction_type = etree.SubElement(direction, "direction-type")
    if words:
        etree.SubElement(direction_type, "words").text = words
    if dynamic:
        dynamics = etree.SubElement(direction_type, "dynamics")
        etree.SubElement(dynamics, dynamic)
    if wedge:
        etree.SubElement(direction_type, "wedge", type=wedge, number="1")
    if pedal:
        etree.SubElement(direction_type, "pedal", type=pedal, line="yes")
    if octave_shift:
        etree.SubElement(direction_type, "octave-shift", type=octave_shift, size="8", number="1")
    etree.SubElement(direction, "staff").text = str(staff)


def _solo_case() -> etree._Element:
    root, parts = _score([("P1", "Solo instrument")])
    first = etree.SubElement(parts["P1"], "measure", number="1")
    _attributes(first, fifths=1)
    _direction(first, words="Allegro moderato")
    _direction(first, dynamic="p")
    _note(first, "C", grace=True, note_type="eighth", slur=("start", "1"))
    _note(first, "D", duration=4, slur=("stop", "1"), articulations=("staccato",))
    _note(first, "E", duration=4, ornaments=("trill-mark",))
    _direction(first, wedge="crescendo")
    _note(first, "F", duration=4, slur=("start", "2"))
    _note(first, "G", duration=4, tie="start", slur=("stop", "2"), articulations=("accent",))
    second = etree.SubElement(parts["P1"], "measure", number="2")
    _direction(second, wedge="stop")
    _direction(second, dynamic="f")
    _note(second, "G", duration=4, tie="stop")
    _note(second, "A", duration=4, beam="begin", note_type="eighth")
    _note(second, "B", duration=4, beam="end", note_type="eighth")
    _rest(second, duration=4, note_type="quarter")
    barline = etree.SubElement(second, "barline", location="right")
    etree.SubElement(barline, "repeat", direction="backward")
    return root


def _piano_part(parts: dict[str, etree._Element], part_id: str = "P1") -> None:
    first = etree.SubElement(parts[part_id], "measure", number="1")
    _attributes(first, staves=2, clefs=(("G", "2"), ("F", "4")), fifths=-2)
    _direction(first, dynamic="mp", staff=1)
    _direction(first, pedal="start", staff=2)
    _note(first, "C", octave=5, duration=4, voice="1", staff=1, fingering="1", arpeggiate=True)
    _note(first, "E", octave=5, duration=4, voice="1", staff=1, chord=True, arpeggiate=True)
    _note(first, "G", octave=5, duration=4, voice="1", staff=1, chord=True, arpeggiate=True)
    _note(first, "A", octave=4, duration=4, voice="1", staff=1, slur=("start", "1"))
    _note(first, "B", octave=4, duration=4, voice="1", staff=2, slur=("stop", "1"))
    _note(first, "C", octave=5, duration=4, voice="1", staff=1)
    _note(first, "D", octave=5, duration=4, voice="1", staff=1)
    _backup(first)
    _note(first, "E", octave=4, duration=8, note_type="half", voice="2", staff=1)
    _note(first, "D", octave=4, duration=8, note_type="half", voice="2", staff=1)
    _backup(first)
    _note(first, "C", octave=3, duration=8, note_type="half", voice="3", staff=2)
    _note(first, "G", octave=2, duration=8, note_type="half", voice="3", staff=2)

    second = etree.SubElement(parts[part_id], "measure", number="2")
    _direction(second, pedal="stop", staff=2)
    _direction(second, octave_shift="up", staff=1)
    for step, state in zip(("C", "D", "E", "F"), ("begin", "continue", "continue", "end"), strict=True):
        _note(
            second,
            step,
            octave=6,
            duration=4,
            note_type="eighth",
            voice="1",
            staff=1,
            beam=state,
        )
    _backup(second)
    _note(second, "C", octave=4, duration=16, note_type="whole", voice="2", staff=1)
    _backup(second)
    _note(second, "F", octave=2, duration=8, note_type="half", voice="3", staff=2)
    _note(second, "C", octave=3, duration=8, note_type="half", voice="3", staff=2)


def _piano_case() -> etree._Element:
    root, parts = _score([("P1", "Piano")])
    _piano_part(parts)
    return root


def _melodic_part(
    part: etree._Element,
    *,
    fifths: int,
    patterns: tuple[tuple[tuple[str, int], ...], ...],
) -> None:
    for measure_number, events in enumerate(patterns, start=1):
        measure = etree.SubElement(part, "measure", number=str(measure_number))
        if measure_number == 1:
            _attributes(measure, fifths=fifths)
        for index, (step, duration) in enumerate(events):
            note_type = {2: "eighth", 4: "quarter", 8: "half", 16: "whole"}[duration]
            slur = None
            if len(events) > 1:
                if index == 0:
                    slur = ("start", "1")
                elif index == len(events) - 1:
                    slur = ("stop", "1")
            _note(
                measure,
                step,
                duration=duration,
                note_type=note_type,
                slur=slur,
            )


def _ensemble_case() -> etree._Element:
    root, parts = _score([("P1", "Flute"), ("P2", "Clarinet in B-flat"), ("P3", "Cello")])
    _melodic_part(parts["P1"], fifths=2, patterns=((("C", 4), ("D", 4), ("E", 8)), (("F", 8), ("G", 4), ("A", 4))))
    _melodic_part(parts["P2"], fifths=4, patterns=((("G", 8), ("A", 8)), (("B", 4), ("A", 4), ("G", 4), ("F", 4))))
    _melodic_part(parts["P3"], fifths=2, patterns=((("C", 16),), (("D", 8), ("G", 8))))
    return root


def _mixed_case() -> etree._Element:
    root, parts = _score([("P1", "Piano"), ("P2", "Violin"), ("P3", "Flute")])
    _piano_part(parts)
    _melodic_part(parts["P2"], fifths=-2, patterns=((("D", 8), ("F", 8)), (("G", 4), ("A", 4), ("B", 8))))
    _melodic_part(parts["P3"], fifths=-2, patterns=((("A", 4), ("G", 4), ("F", 4), ("E", 4)), (("D", 16),)))
    return root


CASES = (
    ("solo_monophonic", "solo", 1, _solo_case),
    ("piano_polyphonic", "piano", 2, _piano_case),
    ("ensemble_independent_time", "ensemble", 3, _ensemble_case),
    ("piano_plus_ensemble", "mixed", 4, _mixed_case),
)


def _edge_executable() -> Path | None:
    discovered = shutil.which("msedge") or shutil.which("microsoft-edge")
    candidates = [
        Path(discovered) if discovered else None,
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


def _svg_to_png(svg: str, output: Path) -> None:
    """Rasterize Verovio SVG with the installed Windows browser.

    PyMuPDF 1.28 currently accepts Verovio's nested SVG without raising but emits
    an all-white page, so a post-render ink check is mandatory.
    """

    edge = _edge_executable()
    if edge is None:
        raise RuntimeError("Microsoft Edge is required to rasterize generated benchmark SVG")
    with tempfile.TemporaryDirectory(prefix="scorescan-svg-render-") as temporary:
        root = Path(temporary)
        svg_path = root / "page.svg"
        profile = root / "profile"
        svg_path.write_text(svg, encoding="utf-8")
        completed = subprocess.run(
            [
                str(edge),
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                "--run-all-compositor-stages-before-draw",
                "--force-device-scale-factor=3.5",
                "--window-size=706,998",
                f"--user-data-dir={profile}",
                f"--screenshot={output.resolve()}",
                svg_path.as_uri(),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
            check=False,
        )
        deadline = time.monotonic() + 8.0
        while not output.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout).strip()[:500]
            raise RuntimeError(f"Edge SVG rasterization failed ({completed.returncode}): {detail}")
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0 or int(image.min()) > 245:
        output.unlink(missing_ok=True)
        raise RuntimeError("SVG rasterization produced an empty page")


def _render(xml_path: Path, output: Path) -> None:
    import verovio  # type: ignore

    toolkit = _configure_toolkit(verovio)
    toolkit.setOptions(
        {
            "inputFrom": "musicxml",
            "adjustPageHeight": False,
            "breaks": "auto",
            "footer": "none",
            "header": "none",
            "pageWidth": 1680,
            "pageHeight": 2376,
            "scale": 42,
        }
    )
    if not toolkit.loadFile(str(xml_path)) or int(toolkit.getPageCount()) != 1:
        raise RuntimeError(f"Verovio did not render exactly one page: {xml_path}")
    svg = toolkit.renderToSVG(1)
    _svg_to_png(svg, output)


def _degrade(source: Path, destination: Path, *, seed: int) -> dict[str, float | int]:
    rng = random.Random(seed)
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read rendered page: {source}")
    height, width = image.shape
    angle = rng.uniform(-0.8, 0.8)
    blur_sigma = rng.uniform(0.25, 0.85)
    contrast = rng.uniform(0.82, 1.05)
    brightness = rng.uniform(-8.0, 4.0)
    noise_std = rng.uniform(1.5, 4.0)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    result = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=246)
    kernel = max(3, int(round(blur_sigma * 4)) * 2 + 1)
    result = cv2.GaussianBlur(result, (kernel, kernel), blur_sigma)
    result = result.astype(np.float32) * contrast + brightness
    result += np.random.default_rng(seed).normal(0, noise_std, result.shape)
    yy, xx = np.mgrid[0:height, 0:width]
    shade = (xx / max(width, 1) * 5.0) + (yy / max(height, 1) * 3.0)
    result = np.clip(result - shade, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise RuntimeError("JPEG round-trip encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if not cv2.imwrite(str(destination), decoded, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise RuntimeError(f"Cannot write {destination}")
    return {
        "seed": seed,
        "angle_degrees": angle,
        "blur_sigma": blur_sigma,
        "contrast": contrast,
        "brightness": brightness,
        "noise_std": noise_std,
        "jpeg_quality": 86,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for index, (case_id, category, staves, factory) in enumerate(CASES, start=1):
        case_root = output / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        xml_path = case_root / "gold.musicxml"
        clean_path = case_root / "clean.png"
        scan_path = case_root / "scan.png"
        etree.ElementTree(factory()).write(
            str(xml_path),
            encoding="UTF-8",
            xml_declaration=True,
            doctype=MUSICXML_DOCTYPE,
            pretty_print=True,
        )
        _render(xml_path, clean_path)
        degradation = _degrade(clean_path, scan_path, seed=SEED + index)
        cases.append(
            {
                "id": case_id,
                "category": category,
                "synthetic": True,
                "physical_staves": staves,
                "gold": str(xml_path.relative_to(output)).replace("\\", "/"),
                "clean_image": str(clean_path.relative_to(output)).replace("\\", "/"),
                "scan_image": str(scan_path.relative_to(output)).replace("\\", "/"),
                "degradation": degradation,
                "sha256": {
                    "gold": _sha256(xml_path),
                    "clean_image": _sha256(clean_path),
                    "scan_image": _sha256(scan_path),
                },
            }
        )
    manifest: dict[str, object] = {
        "format": "scorescan-generated-structural-benchmark@1",
        "seed": SEED,
        "license": "CC0-1.0",
        "synthetic": True,
        "accuracy_evidence": False,
        "purpose": "deterministic structural, notation-preservation, and pipeline regression",
        "cases": cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.txt").write_text(
        "ScoreScan generated structural benchmark\n\n"
        "These pages are deterministic synthetic conformance material created by this repository.\n"
        "They test topology and regression behavior. They are not a substitute for a licensed,\n"
        "independently annotated real scanned-score test set and must not be cited as real-scan accuracy.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.output)
    print(f"Generated {len(manifest['cases'])} structural cases in {args.output}")


if __name__ == "__main__":
    main()
