from __future__ import annotations

"""Train the conservative CPU staff-direction role classifier.

The model separates musical directions from titles, composer credits, page furniture,
lyrics, fingering numbers and OCR fragments.  It uses only auditable lexical/geometry
features and never edits score semantics.  Data are split by source phrase identity so
variants of one phrase do not leak between training and evaluation.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.direction_anchor import (  # noqa: E402
    FEATURE_NAMES,
    DirectionAnchorClassifier,
    extract_direction_anchor_features,
)
from scorescan.direction_model import DirectionCorrector  # noqa: E402
from scorescan.layout import PageLayout, StaffSystem  # noqa: E402
from scorescan.model_registry import build_manifest  # noqa: E402
from scorescan.util import atomic_write_json  # noqa: E402

NEGATIVE_PHRASES = (
    "Johann Sebastian Bach", "Wolfgang Amadeus Mozart", "Ludwig van Beethoven",
    "Edited by", "Arranged by", "Violin", "Flute", "Clarinet in B flat", "Op. 12",
    "Copyright 1928", "All rights reserved", "Printed in Germany", "Public domain",
    "Score", "Part", "Page", "Contents", "Exercises", "Lesson 4", "No. 3", "III",
    "Allegretto No. 3", "Sonata in G major", "Etude", "Practice slowly", "Teacher",
    "1", "2", "3", "4", "5", "12", "27", "40", "I", "II", "III", "IV",
    "la", "mi", "do", "sol", "Ah", "O", "the", "and", "with", "senza parole",
    "www.example.com", "ISBN 978-0-0000-0000-0", "scan", "archive", "download",
    "lI", "rn", "vv", "—", "|", "7.", "A", "B", "C", "D", "E", "F", "G",
)

DYNAMICS = {"p", "pp", "ppp", "pppp", "mp", "mf", "f", "ff", "fff", "ffff", "fp", "sf", "sfp", "sfz", "sffz", "rfz", "fz"}


def _layout() -> PageLayout:
    systems = []
    for index, top in enumerate((260, 620, 980, 1340)):
        spacing = 14.0
        lines = [top + spacing * item for item in range(5)]
        systems.append(
            StaffSystem(
                index=index + 1,
                line_y=lines,
                top=int(top - spacing * 4.2),
                bottom=int(lines[-1] + spacing * 4.2),
                left=120,
                right=2280,
                spacing=spacing,
                barlines=[480, 850, 1220, 1590, 1960],
                measure_count=6,
            )
        )
    return PageLayout(2400, 1800, systems, 0.98)


def _corrupt(text: str, rng: random.Random) -> str:
    value = text
    if len(value) > 3 and rng.random() < 0.34:
        index = rng.randrange(len(value))
        value = value[:index] + value[index + 1 :]
    if len(value) > 4 and rng.random() < 0.22:
        index = rng.randrange(len(value) - 1)
        value = value[:index] + value[index + 1] + value[index] + value[index + 2 :]
    substitutions = {"o": "0", "O": "0", "l": "1", "I": "l", "m": "rn", "rn": "m"}
    if rng.random() < 0.30:
        candidates = [item for item in substitutions if item in value]
        if candidates:
            old = rng.choice(candidates)
            value = value.replace(old, substitutions[old], 1)
    if rng.random() < 0.16:
        value = value.replace(" ", "", 1)
    if rng.random() < 0.14:
        value += rng.choice((".", ",", ":"))
    return value or text


def _kind(text: str, positive: bool, rng: random.Random) -> str:
    key = text.strip().lower().strip(".,:;()")
    if key in DYNAMICS:
        return "dynamic"
    if any(symbol in text for symbol in ("♩", "♪", "𝅘𝅥", "𝅘𝅥𝅮", "=")) and any(char.isdigit() for char in text):
        return "metronome"
    if positive:
        return "direction" if rng.random() < 0.88 else "text"
    return rng.choices(("metadata", "text", "other", "direction"), weights=(0.48, 0.31, 0.15, 0.06), k=1)[0]


def _sample(
    text: str,
    label: int,
    rng: random.Random,
    lexical: tuple[float, float, float],
    layout: PageLayout,
) -> tuple[float, ...]:
    positive = bool(label)
    observed = _corrupt(text, rng)
    base_probability, base_margin, base_edit_ratio = lexical
    # Corruption severity perturbs the cached phrase-level correction evidence.  This
    # keeps training deterministic and fast while preserving realistic overlap between
    # musical and non-musical strings.
    corruption = min(1.0, abs(len(observed) - len(text)) / max(len(text), 1) + (observed != text) * 0.18)
    correction_probability = max(0.0, min(1.0, base_probability - corruption * rng.uniform(0.08, 0.32)))
    correction_margin = max(-1.0, min(1.0, base_margin - corruption * rng.uniform(0.02, 0.18)))
    correction_edit_ratio = max(0.0, min(1.0, base_edit_ratio + corruption * rng.uniform(0.05, 0.28)))
    kind = _kind(text, positive, rng)
    system_index: int | None
    placement: str | None

    if positive:
        system_index = rng.randrange(len(layout.systems))
        system = layout.systems[system_index]
        placement = rng.choices(("above", "below", "within"), weights=(0.62, 0.32, 0.06), k=1)[0]
        if system_index == 0 and kind in {"direction", "metronome"} and rng.random() < 0.28:
            distance = rng.uniform(6.0, 10.8)
        else:
            distance = rng.uniform(0.3, 6.4)
        center_y = (
            system.line_y[0] - distance * system.spacing
            if placement == "above"
            else system.line_y[-1] + distance * system.spacing
            if placement == "below"
            else 0.5 * (system.line_y[0] + system.line_y[-1])
        )
        relative_x = rng.uniform(0.02, 0.98)
        center_x = system.left + relative_x * (system.right - system.left)
        height_spaces = rng.uniform(0.75, 2.1) if kind != "dynamic" else rng.uniform(0.9, 1.8)
        width_spaces = max(0.8, min(24.0, len(observed) * rng.uniform(0.38, 0.72)))
        score = rng.betavariate(7.0, 2.1) * 0.46 + 0.52
        backend_count = rng.choices((1, 2, 3, 4), weights=(0.08, 0.32, 0.42, 0.18), k=1)[0]
    else:
        scenario = rng.choices(("header", "footer", "near_staff", "page_number", "lyrics", "fragment"), weights=(0.25, 0.10, 0.25, 0.10, 0.16, 0.14), k=1)[0]
        if scenario in {"header", "footer", "page_number"}:
            system_index = None
            placement = "above" if scenario == "header" else "below"
            distance = rng.uniform(12.5, 35.0)
            center_x = rng.uniform(120, 2280)
            center_y = rng.uniform(30, 210) if scenario == "header" else rng.uniform(1640, 1780)
            height_spaces = rng.uniform(1.4, 5.6) if scenario == "header" else rng.uniform(0.8, 2.4)
        else:
            system_index = rng.randrange(len(layout.systems))
            system = layout.systems[system_index]
            placement = rng.choices(("above", "below", "within"), weights=(0.24, 0.52, 0.24), k=1)[0]
            distance = rng.uniform(0.0, 8.8)
            center_x = rng.uniform(system.left, system.right)
            center_y = (
                system.line_y[0] - distance * system.spacing
                if placement == "above"
                else system.line_y[-1] + distance * system.spacing
                if placement == "below"
                else rng.uniform(system.line_y[0], system.line_y[-1])
            )
            height_spaces = rng.uniform(0.55, 2.5)
        width_spaces = max(0.35, min(30.0, len(observed) * rng.uniform(0.25, 0.90)))
        score = rng.betavariate(4.3, 2.8) * 0.58 + 0.25
        backend_count = rng.choices((1, 2, 3), weights=(0.55, 0.32, 0.13), k=1)[0]

    spacing = 14.0
    width = max(4.0, width_spaces * spacing)
    height = max(4.0, height_spaces * spacing)
    box = [
        [center_x - width / 2, center_y - height / 2],
        [center_x + width / 2, center_y - height / 2],
        [center_x + width / 2, center_y + height / 2],
        [center_x - width / 2, center_y + height / 2],
    ]
    backend = "+".join(f"ocr{item}" for item in range(backend_count))
    features = extract_direction_anchor_features(
        text=observed,
        kind=kind,
        score=score,
        box=box,
        backend=backend,
        correction_probability=correction_probability,
        correction_margin=correction_margin,
        correction_edit_ratio=correction_edit_ratio,
        system_index=system_index,
        placement=placement,
        distance_staff_spaces=distance,
        layout=layout,
    )
    return features.vector()


def build_dataset(seed: int, variants: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = random.Random(seed)
    layout = _layout()
    corrector = DirectionCorrector()
    resource = ROOT / "src" / "scorescan" / "resources" / "direction_model.json"
    lexicon_payload = json.loads(resource.read_text(encoding="utf-8"))
    positives = [str(item["text"]) for item in lexicon_payload.get("lexicon", []) if isinstance(item, dict) and item.get("text")]
    # Keep the grouped dataset balanced while preserving broad direction coverage.
    positives = positives[:]
    rng.shuffle(positives)
    positives = positives[: min(520, len(positives))]
    phrases = [(text, 1) for text in positives] + [(text, 0) for text in NEGATIVE_PHRASES]
    lexical_cache: dict[str, tuple[float, float, float]] = {}
    for text, _label in phrases:
        suggestion = corrector.suggest(text)
        lexical_cache[text] = (suggestion.probability, suggestion.margin, suggestion.edit_ratio)
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    groups: list[int] = []
    names: list[str] = []
    for group, (text, label) in enumerate(phrases):
        local = random.Random(rng.randrange(1 << 30))
        sample_count = variants if label else variants * 5
        for _ in range(sample_count):
            rows.append(_sample(text, label, local, lexical_cache[text], layout))
            labels.append(label)
            groups.append(group)
            names.append(text)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
        names,
    )


def _choose_threshold(labels: np.ndarray, probabilities: np.ndarray, minimum_precision: float = 0.995) -> float:
    best = 0.90
    best_recall = -1.0
    for threshold in sorted(set(float(value) for value in probabilities)):
        predicted = probabilities >= threshold
        tp = int(np.sum((predicted == 1) & (labels == 1)))
        fp = int(np.sum((predicted == 1) & (labels == 0)))
        fn = int(np.sum((predicted == 0) & (labels == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision >= minimum_precision and recall > best_recall:
            best = threshold
            best_recall = recall
    return float(best)


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "coverage": float(np.mean(predicted)),
        "false_accepts": int(np.sum((predicted == 1) & (labels == 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "src" / "scorescan" / "resources" / "direction_anchor_classifier.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training" / "direction_anchor_classifier_report_v1.json")
    args = parser.parse_args()

    x, y, groups, _names = build_dataset(args.seed, args.variants)
    first = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=args.seed)
    train_cal, test = next(first.split(x, y, groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=args.seed + 1)
    train_relative, calibration_relative = next(second.split(x[train_cal], y[train_cal], groups[train_cal]))
    train = train_cal[train_relative]
    calibration = train_cal[calibration_relative]

    scaler = StandardScaler().fit(x[train])
    model = LogisticRegression(C=1.25, max_iter=2500, class_weight="balanced", random_state=args.seed)
    model.fit(scaler.transform(x[train]), y[train])
    calibration_probabilities = model.predict_proba(scaler.transform(x[calibration]))[:, 1]
    test_probabilities = model.predict_proba(scaler.transform(x[test]))[:, 1]
    threshold = _choose_threshold(y[calibration], calibration_probabilities)

    payload = {
        "format": 1,
        "model_version": "scorescan-direction-anchor-logistic-1",
        "seed": args.seed,
        "feature_names": list(FEATURE_NAMES),
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(value) for value in model.coef_[0]],
        "means": [float(value) for value in scaler.mean_],
        "scales": [float(value) for value in scaler.scale_],
        "recommended_writeback_threshold": threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    # Refresh the manifest before loading the trained model for a round-trip check.
    atomic_write_json(args.output.parent / "model_manifest.json", build_manifest(args.output.parent))
    classifier = DirectionAnchorClassifier(args.output)

    report = {
        "model_version": payload["model_version"],
        "seed": args.seed,
        "features": list(FEATURE_NAMES),
        "total_samples": int(len(y)),
        "positive_samples": int(np.sum(y == 1)),
        "negative_samples": int(np.sum(y == 0)),
        "phrase_groups": int(len(np.unique(groups))),
        "train_samples": int(len(train)),
        "calibration_samples": int(len(calibration)),
        "test_samples": int(len(test)),
        "calibration": metrics(y[calibration], calibration_probabilities, threshold),
        "test": metrics(y[test], test_probabilities, threshold),
        "runtime_roundtrip_enabled": classifier.enabled,
        "runtime_model_status": classifier.status,
        "scope": "CPU lexical and geometry classification of staff directions versus metadata/noise; not end-to-end OMR accuracy",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
