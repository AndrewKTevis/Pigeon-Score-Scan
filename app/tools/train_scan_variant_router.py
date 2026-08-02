from __future__ import annotations

"""Train the bounded CPU scan-treatment router on deterministic synthetic scans.

Labels describe the degradation family whose corresponding treatment is expected to be
most useful.  This is a routing benchmark, not an end-to-end OMR accuracy benchmark.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scorescan.scan_routing import FEATURE_NAMES, VARIANT_NAMES, extract_scan_routing_features  # noqa: E402


def synthetic_page(rng: random.Random, width: int = 760, height: int = 1040) -> np.ndarray:
    page = np.full((height, width), rng.randint(242, 255), dtype=np.uint8)
    margin = 90
    spacing = rng.randint(10, 18)
    y = 170
    while y + 5 * spacing < height - 120:
        for line in range(5):
            cv2.line(page, (margin, y + line * spacing), (width - margin, y + line * spacing), rng.randint(15, 55), 1)
        x = margin + 30
        for _ in range(rng.randint(18, 38)):
            x += rng.randint(22, 46)
            if x > width - margin - 20:
                break
            note_y = y + rng.randint(-2, 10) * spacing // 2
            cv2.ellipse(page, (x, note_y), (max(3, spacing // 2), max(2, spacing // 3)), rng.uniform(-18, 18), 0, 360, rng.randint(10, 45), -1)
            if rng.random() < 0.88:
                direction = -1 if note_y > y + 2 * spacing else 1
                cv2.line(page, (x + max(2, spacing // 2), note_y), (x + max(2, spacing // 2), note_y - direction * rng.randint(25, 55)), rng.randint(10, 45), 1)
            if rng.random() < 0.18:
                cv2.circle(page, (x + spacing, note_y), max(1, spacing // 7), rng.randint(10, 45), -1)
        if rng.random() < 0.65:
            cv2.putText(page, rng.choice(["Allegro", "dolce", "rit.", "mf", "a tempo"]), (margin + rng.randint(0, 450), y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 35, 1, cv2.LINE_AA)
        y += rng.randint(160, 230)
    return page


def degrade(base: np.ndarray, label: str, rng: random.Random) -> np.ndarray:
    image = base.astype(np.float32)
    h, w = base.shape
    if label == "primary":
        sigma = rng.uniform(0.0, 0.45)
        if sigma:
            image = cv2.GaussianBlur(image, (0, 0), sigma)
    elif label == "flat":
        yy, xx = np.mgrid[0:h, 0:w]
        shade = 35 * (xx / max(w - 1, 1)) + 25 * np.sin(yy / rng.uniform(180, 360))
        image = image - shade + rng.uniform(-12, 8)
        image *= rng.uniform(0.82, 0.96)
    elif label == "otsu":
        image = 128 + (image - 128) * rng.uniform(0.32, 0.62)
        image += rng.normalvariate(0, 4)
    elif label == "adaptive":
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        radius = rng.uniform(0.35, 0.75) * max(w, h)
        shadow = np.clip(1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius, 0, 1)
        image -= shadow * rng.uniform(35, 85)
    elif label == "deblock":
        quality = rng.randint(18, 48)
        ok, encoded = cv2.imencode(".jpg", np.clip(image, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(2, 7), image.shape)
        image += noise
    elif label == "upscale":
        scale = rng.uniform(0.34, 0.62)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 0.9))
    elif label == "staffnorm":
        scale = rng.choice([rng.uniform(0.60, 0.76), rng.uniform(1.32, 1.58)])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        canvas = np.full_like(image, 255)
        rh, rw = resized.shape
        if rh <= h and rw <= w:
            y0, x0 = (h - rh) // 2, (w - rw) // 2
            canvas[y0:y0+rh, x0:x0+rw] = resized
        else:
            y0, x0 = max(0, (rh - h) // 2), max(0, (rw - w) // 2)
            canvas = resized[y0:y0+h, x0:x0+w]
        image = canvas
    return np.clip(image, 0, 255).astype(np.uint8)


def build_dataset(seed: int, samples_per_class: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in range(samples_per_class):
        base = synthetic_page(rng, rng.choice([640, 760, 900]), rng.choice([880, 1040, 1200]))
        for class_index, label in enumerate(VARIANT_NAMES):
            image = degrade(base, label, rng)
            features = extract_scan_routing_features(image)
            rows.append(features.vector())
            labels.append(class_index)
            groups.append(group)
    return np.asarray(rows, np.float64), np.asarray(labels, np.int64), np.asarray(groups, np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "src/scorescan/resources/scan_variant_router.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training/scan_variant_router_report_v1.json")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--samples-per-class", type=int, default=140)
    args = parser.parse_args()

    x, y, groups = build_dataset(args.seed, args.samples_per_class)
    group_ids = sorted(set(groups.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(group_ids)
    split = int(len(group_ids) * 0.80)
    train_groups = set(group_ids[:split])
    train_mask = np.asarray([group in train_groups for group in groups])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask])
    x_test = scaler.transform(x[~train_mask])
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed, solver="lbfgs")
    model.fit(x_train, y[train_mask])
    probabilities = model.predict_proba(x_test)
    predictions = probabilities.argmax(axis=1)

    payload = {
        "format": 1,
        "model_version": "scorescan-scan-router-1",
        "seed": args.seed,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(VARIANT_NAMES),
        "means": [float(value) for value in scaler.mean_],
        "scales": [float(value) for value in scaler.scale_],
        "intercepts": [float(value) for value in model.intercept_],
        "coefficients": [[float(value) for value in row] for row in model.coef_],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "seed": args.seed,
        "samples": int(len(y)),
        "training_samples": int(train_mask.sum()),
        "test_samples": int((~train_mask).sum()),
        "classes": list(VARIANT_NAMES),
        "accuracy": float(accuracy_score(y[~train_mask], predictions)),
        "log_loss": float(log_loss(y[~train_mask], probabilities, labels=list(range(len(VARIANT_NAMES))))),
        "confusion_matrix": confusion_matrix(y[~train_mask], predictions, labels=list(range(len(VARIANT_NAMES)))).tolist(),
        "note": "Synthetic scan-treatment routing benchmark; not end-to-end OMR accuracy.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
