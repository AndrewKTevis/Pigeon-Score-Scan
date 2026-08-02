from __future__ import annotations

"""Non-regression evaluation of the context calibrator on reviewed MusicXML scores."""

import argparse
import json
import random
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.context_calibration import ContextCalibrator  # noqa: E402
from scorescan.score_ir import MeasureIR, PitchIR, measure_distance, score_from_tree  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402


def mutate(measure: MeasureIR, rng: random.Random) -> MeasureIR:
    notes = list(measure.notes)
    usable = [index for index, note in enumerate(notes) if not note.grace and not note.chord]
    if not usable:
        return replace(measure, key_signature=(5, "major"))
    index = rng.choice(usable)
    note = notes[index]
    kind = rng.choice(("octave", "pitch", "tie", "delete", "extra", "key", "rest"))
    if kind == "octave" and note.pitch:
        notes[index] = replace(note, pitch=replace(note.pitch, octave=max(1, min(8, note.pitch.octave + rng.choice((-1, 1))))))
    elif kind == "pitch" and note.pitch:
        notes[index] = replace(note, pitch=PitchIR(rng.choice("CDEFGAB"), note.pitch.alter, note.pitch.octave))
    elif kind == "tie":
        notes[index] = replace(note, ties=tuple(sorted(set(note.ties + (rng.choice(("start", "stop")),)))))
    elif kind == "delete" and len(usable) > 1:
        notes.pop(index)
    elif kind == "extra":
        notes.insert(index, replace(note, onset=note.onset + Fraction(1, 8), ties=()))
    elif kind == "key":
        fifths, mode = measure.key_signature or (0, "major")
        return replace(measure, key_signature=(max(-7, min(7, fifths + rng.choice((-4, 4)))), mode))
    elif kind == "rest":
        notes[index] = replace(note, rest=not note.rest, pitch=None if not note.rest else PitchIR("C", Fraction(0), 4), ties=())
    return replace(measure, notes=tuple(notes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("musicxml", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--mutations", type=int, default=5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    calibrator = ContextCalibrator()
    groups = correct = 0
    per_score: list[dict[str, object]] = []
    parser_xml = etree.XMLParser(resolve_entities=False, no_network=True)
    for path in args.musicxml:
        score = score_from_tree(etree.parse(str(path), parser_xml))
        score_groups = score_correct = 0
        for index in range(1, len(score.measures) - 1):
            previous, reference, following = score.measures[index - 1:index + 2]
            candidates = [reference] + [mutate(reference, rng) for _ in range(args.mutations)]
            probabilities = [calibrator.calibrate(previous, candidate, following).probability for candidate in candidates]
            choice = max(range(len(candidates)), key=lambda item: (probabilities[item], -item))
            score_groups += 1
            score_correct += int(measure_distance(reference, candidates[choice]) <= 0.002)
        groups += score_groups
        correct += score_correct
        per_score.append({"file": path.name, "groups": score_groups, "top1": score_correct / max(score_groups, 1)})
    report = {
        "model_version": calibrator.model_version,
        "seed": args.seed,
        "score_count": len(args.musicxml),
        "groups": groups,
        "mutations_per_group": args.mutations,
        "top1": correct / max(groups, 1),
        "scores": per_score,
        "note": "Reviewed MusicXML semantic perturbation non-regression; not a scan-image OMR benchmark.",
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
