from __future__ import annotations

"""Train the bounded page-candidate calibration prior on procedural MusicXML ensembles.

The generator creates valid single-staff, single-voice scores, semantically equivalent
serialisation variants, and controlled corruptions.  Grouped train/test splitting keeps
all variants of one source score in the same partition.  This is a calibration test,
not an end-to-end OMR benchmark.
"""

import argparse
import copy
import json
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from lxml import etree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.alignment import align_measure_sequences  # noqa: E402
from scorescan.candidate_calibration import FEATURE_NAMES, feature_vector  # noqa: E402
from scorescan.consensus import semantic_agreement  # noqa: E402
from scorescan.omr import EngineResult  # noqa: E402
from scorescan.recognition import RecognitionCandidate, assess_candidate  # noqa: E402
from scorescan.score_ir import measure_distance, score_from_tree  # noqa: E402

TYPE_FOR_DURATION = {32: "whole", 16: "half", 8: "quarter", 4: "eighth", 2: "16th", 1: "32nd"}
STEPS = ("C", "D", "E", "F", "G", "A", "B")


def _partition(total: int, rng: random.Random) -> list[int]:
    durations = (16, 16, 8, 8, 8, 4, 4)
    result: list[int] = []
    remaining = total
    while remaining:
        choices = [value for value in durations if value <= remaining]
        value = rng.choice(choices)
        result.append(value)
        remaining -= value
    return result


def build_score(rng: random.Random, measures: int) -> etree._ElementTree:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    divisions = 8
    fifths = rng.randint(-3, 3)
    for index in range(measures):
        measure = etree.SubElement(part, "measure", number=str(index + 1))
        if index == 0:
            attrs = etree.SubElement(measure, "attributes")
            etree.SubElement(attrs, "divisions").text = str(divisions)
            key = etree.SubElement(attrs, "key")
            etree.SubElement(key, "fifths").text = str(fifths)
            time = etree.SubElement(attrs, "time")
            etree.SubElement(time, "beats").text = "4"
            etree.SubElement(time, "beat-type").text = "4"
            clef = etree.SubElement(attrs, "clef")
            etree.SubElement(clef, "sign").text = "G"
            etree.SubElement(clef, "line").text = "2"
        if index == 0 or rng.random() < 0.10:
            direction = etree.SubElement(measure, "direction", placement="above")
            direction_type = etree.SubElement(direction, "direction-type")
            etree.SubElement(direction_type, "words").text = rng.choice(
                ["Allegro", "Andante cantabile", "dolce", "poco a poco", "a tempo"]
            )
        for duration in _partition(32, rng):
            note = etree.SubElement(measure, "note")
            if rng.random() < 0.12:
                etree.SubElement(note, "rest")
            else:
                pitch = etree.SubElement(note, "pitch")
                etree.SubElement(pitch, "step").text = rng.choice(STEPS)
                if rng.random() < 0.10:
                    etree.SubElement(pitch, "alter").text = rng.choice(["-1", "1"])
                etree.SubElement(pitch, "octave").text = str(rng.choice([4, 4, 5, 5, 5, 6]))
            etree.SubElement(note, "duration").text = str(duration)
            etree.SubElement(note, "voice").text = "1"
            etree.SubElement(note, "type").text = TYPE_FOR_DURATION[duration]
        if index == measures - 1:
            barline = etree.SubElement(measure, "barline", location="right")
            etree.SubElement(barline, "bar-style").text = "light-heavy"
    return etree.ElementTree(root)


def _all_notes(tree: etree._ElementTree) -> list[etree._Element]:
    return tree.getroot().findall("./part/measure/note")


def mutate(tree: etree._ElementTree, kind: str, rng: random.Random) -> etree._ElementTree:
    result = copy.deepcopy(tree)
    root = result.getroot()
    part = root.find("part")
    assert part is not None
    measures = part.findall("measure")
    notes = _all_notes(result)
    if kind == "equivalent_divisions":
        divisions = root.find("./part/measure/attributes/divisions")
        assert divisions is not None
        divisions.text = str(int(divisions.text or "8") * 2)
        for node in root.findall("./part/measure/note/duration"):
            node.text = str(int(node.text or "0") * 2)
    elif kind == "pitch_shift":
        pitched = [note.find("pitch") for note in notes if note.find("pitch") is not None]
        pitch = rng.choice(pitched)
        octave = pitch.find("octave")
        assert octave is not None
        octave.text = str(int(octave.text or "4") + rng.choice([-1, 1]))
    elif kind == "duration_error":
        note = rng.choice(notes)
        duration = note.find("duration")
        assert duration is not None
        duration.text = str(max(1, int(duration.text or "1") // 2))
    elif kind == "delete_note":
        note = rng.choice(notes)
        note.getparent().remove(note)
    elif kind == "extra_measure":
        clone = copy.deepcopy(rng.choice(measures))
        clone.set("number", str(len(measures) + 1))
        part.append(clone)
    elif kind == "missing_measure":
        if len(measures) > 2:
            part.remove(rng.choice(measures[1:-1]))
    elif kind == "duplicate_direction":
        target = rng.choice(measures)
        direction = target.find("direction")
        if direction is None:
            direction = etree.SubElement(target, "direction", placement="above")
            direction_type = etree.SubElement(direction, "direction-type")
            etree.SubElement(direction_type, "words").text = "Allegro"
        target.insert(target.index(direction) + 1, copy.deepcopy(direction))
    elif kind == "multiple_voice":
        note = rng.choice(notes)
        voice = note.find("voice")
        assert voice is not None
        voice.text = "2"
    elif kind == "wrong_accidental":
        note = rng.choice([item for item in notes if item.find("rest") is None])
        accidental = note.find("accidental")
        if accidental is None:
            accidental = etree.SubElement(note, "accidental")
        accidental.text = rng.choice(["sharp", "flat", "natural"])
    elif kind == "empty_measure":
        target = rng.choice(measures)
        for child in list(target):
            if child.tag not in {"attributes", "print"}:
                target.remove(child)
    else:
        raise ValueError(kind)
    return result


def write_tree(tree: etree._ElementTree, path: Path) -> None:
    path.write_bytes(etree.tostring(tree, encoding="UTF-8", xml_declaration=True, pretty_print=True))


def exact_label(reference: etree._ElementTree, candidate: etree._ElementTree) -> int:
    ref = score_from_tree(reference)
    cand = score_from_tree(candidate)
    alignment = align_measure_sequences(ref.measures, cand.measures)
    if any(value is None for value in alignment.reference_to_candidate) or alignment.unmatched_candidate_indices:
        return 0
    distances = [
        measure_distance(ref.measures[index], cand.measures[candidate_index])
        for index, candidate_index in enumerate(alignment.reference_to_candidate)
        if candidate_index is not None
    ]
    return int(bool(distances) and max(distances) <= 0.002)


def build_dataset(seed: int, ensembles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    rng = random.Random(seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    ensemble_rows: list[dict[str, object]] = []
    mutation_kinds = [
        "equivalent_divisions", "pitch_shift", "duration_error", "delete_note",
        "extra_measure", "missing_measure", "duplicate_direction", "multiple_voice",
        "wrong_accidental", "empty_measure",
    ]
    with tempfile.TemporaryDirectory(prefix="scorescan-calibration-") as temp_name:
        temp = Path(temp_name)
        for group in range(ensembles):
            reference = build_score(rng, rng.randint(3, 8))
            variants: list[tuple[str, etree._ElementTree]] = [
                ("clean", copy.deepcopy(reference)),
                ("equivalent", mutate(reference, "equivalent_divisions", rng)),
            ]
            selected_kinds = rng.sample(mutation_kinds[1:], k=5)
            variants.extend((kind, mutate(reference, kind, rng)) for kind in selected_kinds)
            candidates: list[RecognitionCandidate] = []
            labels_for_ensemble: list[int] = []
            for variant_index, (kind, tree) in enumerate(variants):
                path = temp / f"g{group:04d}_{variant_index}_{kind}.musicxml"
                write_tree(tree, path)
                assessed = assess_candidate(
                    f"g{group}_{variant_index}_{kind}",
                    path.with_suffix(".png"),
                    EngineResult(0, path, 0.01),
                    len(score_from_tree(reference).measures),
                )
                candidates.append(assessed)
                labels_for_ensemble.append(exact_label(reference, tree))
            agreements = semantic_agreement(candidates)
            enriched = [
                replace(candidate, agreement_ratio=float(agreements.get(candidate.variant, 0.0)))
                for candidate in candidates
            ]
            indices = []
            for candidate, label in zip(enriched, labels_for_ensemble, strict=True):
                rows.append(feature_vector(candidate))
                labels.append(label)
                groups.append(group)
                indices.append(len(rows) - 1)
            ensemble_rows.append({"group": group, "indices": indices, "labels": labels_for_ensemble})
    return np.asarray(rows), np.asarray(labels), np.asarray(groups), ensemble_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "src/scorescan/resources/candidate_calibrator.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training/candidate_calibrator_report_v1.json")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--ensembles", type=int, default=240)
    args = parser.parse_args()

    x, y, groups, ensembles = build_dataset(args.seed, args.ensembles)
    unique_groups = sorted(set(groups.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(unique_groups)
    split = int(len(unique_groups) * 0.80)
    train_groups = set(unique_groups[:split])
    train_mask = np.asarray([group in train_groups for group in groups])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask])
    x_test = scaler.transform(x[~train_mask])
    model = LogisticRegression(max_iter=600, class_weight="balanced", random_state=args.seed, solver="liblinear")
    model.fit(x_train, y[train_mask])
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    payload = {
        "format": 1,
        "model_version": "scorescan-candidate-calibrator-1",
        "seed": args.seed,
        "feature_names": list(FEATURE_NAMES),
        "means": [float(value) for value in scaler.mean_],
        "scales": [float(value) for value in scaler.scale_],
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(value) for value in model.coef_[0]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    test_group_set = set(unique_groups[split:])
    top1_total = 0
    baseline_correct = 0
    calibrated_correct = 0
    for ensemble in ensembles:
        if ensemble["group"] not in test_group_set:
            continue
        indices = ensemble["indices"]
        labels = ensemble["labels"]
        local_x = x[indices]
        local_probs = model.predict_proba(scaler.transform(local_x))[:, 1]
        baseline = max(range(len(indices)), key=lambda i: local_x[i][0] + 0.07 * local_x[i][2])
        calibrated = max(range(len(indices)), key=lambda i: (local_probs[i], local_x[i][2], local_x[i][0]))
        baseline_correct += labels[baseline]
        calibrated_correct += labels[calibrated]
        top1_total += 1

    report = {
        "evaluation": "procedural candidate calibration",
        "model_version": payload["model_version"],
        "seed": args.seed,
        "ensembles": args.ensembles,
        "samples": int(len(y)),
        "positive_samples": int(y.sum()),
        "train_samples": int(train_mask.sum()),
        "test_samples": int((~train_mask).sum()),
        "sample_accuracy": float(accuracy_score(y[~train_mask], predictions)),
        "sample_auc": float(roc_auc_score(y[~train_mask], probabilities)),
        "heldout_ensemble_count": top1_total,
        "baseline_top1_accuracy": baseline_correct / max(top1_total, 1),
        "calibrated_top1_accuracy": calibrated_correct / max(top1_total, 1),
        "notes": "Procedural MusicXML candidate-selection calibration; not an end-to-end scan OMR metric.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
