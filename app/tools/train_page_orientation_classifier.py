from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from scorescan.orientation import FEATURE_NAMES, ORIENTATIONS, extract_orientation_features, rotate_quadrant

SEED = 20260718

ROBUST_FEATURE_NAMES = (
    "horizontal_line_density",
    "vertical_line_density",
    "horizontal_vertical_log_ratio",
    "row_projection_variation",
    "column_projection_variation",
    "staff_left_ink",
    "staff_right_ink",
    "staff_left_right_asymmetry",
)
ROBUST_FEATURE_INDICES = tuple(FEATURE_NAMES.index(name) for name in ROBUST_FEATURE_NAMES)


def draw_upright_page(rng: random.Random) -> np.ndarray:
    width = rng.randint(520, 980)
    height = rng.randint(760, 1480)
    background = rng.randint(238, 255)
    image = np.full((height, width), background, dtype=np.uint8)
    # Slow illumination and paper grain.
    gradient_x = np.linspace(rng.randint(-10, 3), rng.randint(-3, 12), width, dtype=np.float32)
    gradient_y = np.linspace(rng.randint(-6, 5), rng.randint(-5, 8), height, dtype=np.float32)[:, None]
    image = np.clip(image.astype(np.float32) + gradient_x[None, :] + gradient_y, 0, 255).astype(np.uint8)

    top_margin = rng.randint(int(height * 0.055), int(height * 0.11))
    title = rng.choice(["Allegretto", "Andante cantabile", "No. 3", "Moderato", "Etude"])
    cv2.putText(
        image,
        title,
        (rng.randint(int(width * 0.30), int(width * 0.42)), top_margin),
        cv2.FONT_HERSHEY_COMPLEX,
        width / 1900.0,
        rng.randint(10, 55),
        max(1, width // 900),
        cv2.LINE_AA,
    )
    if rng.random() < 0.55:
        cv2.putText(
            image,
            rng.choice(["J. S. Bach", "Op. 12", "Violin", "Allegro con brio"]),
            (rng.randint(int(width * 0.55), int(width * 0.72)), top_margin + rng.randint(25, 55)),
            cv2.FONT_HERSHEY_SIMPLEX,
            width / 2600.0,
            rng.randint(20, 70),
            1,
            cv2.LINE_AA,
        )

    systems = rng.randint(4, 10)
    spacing = rng.randint(max(8, width // 150), max(12, width // 95))
    available = height - top_margin - int(height * 0.08)
    system_gap = max(spacing * 8, available // systems)
    y0 = top_margin + rng.randint(spacing * 3, spacing * 6)
    left = rng.randint(int(width * 0.055), int(width * 0.09))
    right = width - rng.randint(int(width * 0.045), int(width * 0.075))
    ink = rng.randint(5, 50)
    for system_index in range(systems):
        staff_top = y0 + system_index * system_gap
        if staff_top + 4 * spacing >= height - spacing * 3:
            break
        for line in range(5):
            y = staff_top + line * spacing
            cv2.line(image, (left, y), (right, y), ink, rng.choice([1, 1, 1, 2]))
        # Clef-like dense left object plus key/time marks.
        clef_x = left + int(spacing * 1.2)
        clef_y = staff_top + 2 * spacing
        cv2.ellipse(image, (clef_x, clef_y), (max(3, spacing // 2), max(8, spacing * 2)), -12, 0, 360, ink, 2)
        cv2.circle(image, (clef_x + spacing // 3, clef_y - spacing), max(2, spacing // 3), ink, -1)
        for key_index in range(rng.randint(0, 4)):
            x = left + int(spacing * (2.4 + key_index * 0.8))
            cv2.line(image, (x, staff_top + spacing), (x, staff_top + 3 * spacing), ink, 1)
            cv2.line(image, (x - 2, staff_top + 2 * spacing), (x + 3, staff_top + 2 * spacing - 2), ink, 1)
        # Notes spread across system.
        note_count = rng.randint(10, 28)
        note_left = left + int(spacing * 5.2)
        xs = sorted(rng.randint(note_left, right - spacing * 2) for _ in range(note_count))
        last_x = None
        last_stem = None
        for x in xs:
            step = rng.randint(-2, 12)
            y = int(round(staff_top + 4 * spacing - step * spacing / 2))
            axes = (max(3, int(spacing * 0.55)), max(2, int(spacing * 0.36)))
            cv2.ellipse(image, (x, y), axes, -18, 0, 360, ink, -1, cv2.LINE_AA)
            up = y >= staff_top + 2 * spacing
            stem_x = x + axes[0] if up else x - axes[0]
            stem_end = y - spacing * 3 if up else y + spacing * 3
            cv2.line(image, (stem_x, y), (stem_x, stem_end), ink, 1, cv2.LINE_AA)
            if last_x is not None and x - last_x < spacing * 4 and rng.random() < 0.45:
                cv2.line(image, last_stem, (stem_x, stem_end), ink, max(2, spacing // 5), cv2.LINE_AA)
            last_x = x
            last_stem = (stem_x, stem_end)
        # Right terminal barline and occasional internal bars.
        cv2.line(image, (right, staff_top), (right, staff_top + 4 * spacing), ink, rng.choice([1, 2, 2]))
        for fraction in (0.28, 0.52, 0.76):
            if rng.random() < 0.68:
                x = int(left + fraction * (right - left)) + rng.randint(-spacing, spacing)
                cv2.line(image, (x, staff_top), (x, staff_top + 4 * spacing), ink, 1)
        if rng.random() < 0.35:
            cv2.putText(image, rng.choice(["mf", "p", "rit.", "dolce"]), (note_left, staff_top - spacing), cv2.FONT_HERSHEY_SIMPLEX, spacing / 22.0, ink, 1, cv2.LINE_AA)

    # Page number at bottom centre: a useful upright/180 cue but not decisive alone.
    cv2.putText(image, str(rng.randint(2, 180)), (width // 2, height - rng.randint(18, 45)), cv2.FONT_HERSHEY_SIMPLEX, width / 3200.0, rng.randint(30, 90), 1, cv2.LINE_AA)
    # Scan degradation.
    if rng.random() < 0.75:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.15, 1.15))
    if rng.random() < 0.55:
        noise = np.random.default_rng(rng.randrange(1 << 31)).normal(0, rng.uniform(0.7, 5.5), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(42, 91)])
        if ok:
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--pages', type=int, default=80)
    args = parser.parse_args()

    rng = random.Random(SEED)
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    groups: list[int] = []
    rotations: list[int] = []
    for page_id in range(args.pages):
        upright = draw_upright_page(rng)
        for rotation in ORIENTATIONS:
            candidate = rotate_quadrant(upright, rotation)
            features.append(extract_orientation_features(candidate))
            labels.append(1 if rotation == 0 else 0)
            groups.append(page_id)
            rotations.append(rotation)

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    g = np.asarray(groups, dtype=np.int64)
    r = np.asarray(rotations, dtype=np.int64)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=SEED)
    train_idx, test_idx = next(splitter.split(x, y, groups=g))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    model = LogisticRegression(C=1.2, max_iter=800, solver='liblinear', class_weight='balanced', random_state=SEED)
    model.fit(x_train[:, ROBUST_FEATURE_INDICES], y[train_idx])
    probabilities = model.predict_proba(x_test[:, ROBUST_FEATURE_INDICES])[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)

    # Page-level selection: choose which correction candidate is most upright.
    all_probabilities = np.zeros(len(x), dtype=np.float64)
    all_probabilities[test_idx] = probabilities
    top1_correct = 0
    test_pages = sorted(set(int(value) for value in g[test_idx]))
    margins: list[float] = []
    for page_id in test_pages:
        indices = test_idx[g[test_idx] == page_id]
        ranked = sorted(indices, key=lambda idx: all_probabilities[idx], reverse=True)
        top1_correct += int(r[ranked[0]] == 0)
        margins.append(float(all_probabilities[ranked[0]] - all_probabilities[ranked[1]]))

    payload = {
        'model_version': 'scorescan-page-orientation-logistic-1',
        'model_type': 'binary_upright_logistic',
        'feature_names': list(FEATURE_NAMES),
        'means': [float(value) for value in scaler.mean_],
        'scales': [float(value) for value in scaler.scale_],
        'coefficients': [float(model.coef_[0][ROBUST_FEATURE_INDICES.index(index)]) if index in ROBUST_FEATURE_INDICES else 0.0 for index in range(len(FEATURE_NAMES))],
        'trained_feature_names': list(ROBUST_FEATURE_NAMES),
        'intercept': float(model.intercept_[0]),
        'training': {
            'seed': SEED,
            'source_pages': args.pages,
            'samples': len(y),
            'train_samples': len(train_idx),
            'test_samples': len(test_idx),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    from sklearn.metrics import roc_auc_score
    report = {
        'model_version': payload['model_version'],
        'seed': SEED,
        'source_pages': args.pages,
        'samples': len(y),
        'train_samples': len(train_idx),
        'test_samples': len(test_idx),
        'binary_test_accuracy': float(accuracy_score(y[test_idx], predictions)),
        'binary_test_auc': float(roc_auc_score(y[test_idx], probabilities)),
        'binary_test_log_loss': float(log_loss(y[test_idx], probabilities)),
        'page_orientation_top1': top1_correct / max(len(test_pages), 1),
        'mean_top1_margin': float(np.mean(margins)) if margins else 0.0,
        'test_pages': len(test_pages),
        'scope': 'synthetic printed-score uprightness/orientation; not end-to-end OMR accuracy',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
