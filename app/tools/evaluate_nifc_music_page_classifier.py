from __future__ import annotations

"""Evaluate scan-music-page detection with leave-one-edition-out splits.

The 102 page roles come from one primary visual reviewer. Results are useful
for diagnostics and workflow automation only, never production evidence.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.audit_nifc_scan_reference_alignment import (  # noqa: E402
    _load_gray,
    _normalized_page,
)
from app.tools.record_nifc_chopin_primary_visual_review import (  # noqa: E402
    REPORT_ROLE as PRIMARY_REVIEW_ROLE,
)
from scorescan.util import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)


REPORT_ROLE = (
    "nifc_music_page_classifier_leave_one_edition_out_diagnostic"
)
KERNEL_WIDTHS = (30, 45, 60, 75, 90, 120)
FEATURE_NAMES = (
    "ink_density",
    "horizontal_gradient_mean",
    "vertical_gradient_mean",
    "canny_edge_density",
    "hough_horizontal_line_count",
    "hough_horizontal_y_cluster_count",
    "hough_horizontal_length_ge_200_count",
    "hough_horizontal_length_ge_400_count",
    "hough_horizontal_max_length_ratio",
    "hough_horizontal_occupied_y_bin_fraction",
    *tuple(
        name
        for width in KERNEL_WIDTHS
        for name in (
            f"h{width}_pixel_fraction",
            f"h{width}_row_fraction_ge_{width}",
            f"h{width}_row_fraction_ge_{width * 2}",
            f"h{width}_row_strength_q95",
            f"h{width}_row_strength_max",
        )
    ),
)


def extract_page_features(path: Path) -> list[float]:
    normalized = _normalized_page(_load_gray(path))
    dark = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )[1]
    horizontal_gradient = np.abs(
        np.diff(normalized.astype(np.float32), axis=0)
    )
    vertical_gradient = np.abs(
        np.diff(normalized.astype(np.float32), axis=1)
    )
    edges = cv2.Canny(normalized, 30, 100)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=80,
        maxLineGap=20,
    )
    horizontal_lines = (
        []
        if raw_lines is None
        else [
            line[0]
            for line in raw_lines
            if abs(int(line[0][3]) - int(line[0][1])) <= 3
            and abs(int(line[0][2]) - int(line[0][0])) >= 80
        ]
    )
    line_lengths = [
        abs(int(line[2]) - int(line[0])) for line in horizontal_lines
    ]
    line_rows = sorted(
        int(round((int(line[1]) + int(line[3])) / 2))
        for line in horizontal_lines
    )
    row_clusters: list[list[int]] = []
    for row in line_rows:
        if not row_clusters or row > row_clusters[-1][-1] + 3:
            row_clusters.append([row])
        else:
            row_clusters[-1].append(row)
    features: list[float] = [
        float(np.count_nonzero(dark) / dark.size),
        float(horizontal_gradient.mean() / 255.0),
        float(vertical_gradient.mean() / 255.0),
        float(np.count_nonzero(edges) / edges.size),
        float(len(horizontal_lines) / 1000.0),
        float(len(row_clusters) / 100.0),
        float(sum(value >= 200 for value in line_lengths) / 1000.0),
        float(sum(value >= 400 for value in line_lengths) / 1000.0),
        float(
            max(line_lengths, default=0) / normalized.shape[1]
        ),
        float(
            len({row // 20 for row in line_rows})
            / (normalized.shape[0] / 20)
        ),
    ]
    for width in KERNEL_WIDTHS:
        horizontal = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1)),
        )
        strengths = np.count_nonzero(horizontal, axis=1).astype(
            np.float64
        )
        features.extend(
            (
                float(np.count_nonzero(horizontal) / horizontal.size),
                float(np.mean(strengths >= width)),
                float(np.mean(strengths >= width * 2)),
                float(np.quantile(strengths, 0.95) / normalized.shape[1]),
                float(strengths.max(initial=0) / normalized.shape[1]),
            )
        )
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError("music-page feature schema drifted")
    return features


def _model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=370,
        n_jobs=1,
    )


def evaluate(
    primary_review_report_path: Path,
    output_path: Path,
) -> dict[str, object]:
    primary = json.loads(
        primary_review_report_path.read_text(encoding="utf-8")
    )
    if primary.get("role") != PRIMARY_REVIEW_ROLE:
        raise ValueError("unexpected primary NIFC review report role")
    for field in (
        "training_authorized",
        "evaluation_authorized",
        "release_authorized",
    ):
        if primary.get(field) is not False:
            raise ValueError(f"primary review unexpectedly sets {field}")
    mappings = primary.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("primary review has no mappings")
    preparation_path = Path(
        str(primary["preparation_report_path"])
    ).resolve()
    if (
        not preparation_path.is_file()
        or sha256_file(preparation_path)
        != primary["preparation_report_sha256"]
    ):
        raise ValueError("primary review preparation report drifted")
    preparation = json.loads(
        preparation_path.read_text(encoding="utf-8")
    )
    prepared_by_pid = {
        str(item["parent_pid"]): item
        for item in preparation["candidates"]
    }

    rows: list[dict[str, object]] = []
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    for mapping in mappings:
        pid = str(mapping["parent_pid"])
        prepared = prepared_by_pid[pid]
        music = set(mapping["reviewed_music_page_indices"])
        non_music = set(mapping["reviewed_non_music_page_indices"])
        for page in prepared["pages"]:
            index = int(page["sequence_index"])
            if index not in music | non_music:
                raise ValueError("primary review page role is missing")
            path = Path(str(page["path"])).resolve()
            if not path.is_file() or sha256_file(path) != page["sha256"]:
                raise ValueError("page-role evaluation image drifted")
            feature_values = extract_page_features(path)
            label = int(index in music)
            rows.append(
                {
                    "parent_pid": pid,
                    "page_index": index,
                    "path": str(path),
                    "sha256": page["sha256"],
                    "primary_visual_label": (
                        "music" if label else "non_music"
                    ),
                    "features": dict(
                        zip(
                            FEATURE_NAMES,
                            feature_values,
                            strict=True,
                        )
                    ),
                }
            )
            features.append(feature_values)
            labels.append(label)
            groups.append(pid)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group_values = np.asarray(groups)
    probabilities = np.zeros(len(y), dtype=np.float64)
    fold_reports: list[dict[str, object]] = []
    splitter = LeaveOneGroupOut()
    for fold, (train, test) in enumerate(
        splitter.split(x, y, group_values),
        start=1,
    ):
        model = _model()
        model.fit(x[train], y[train])
        fold_probability = model.predict_proba(x[test])[:, 1]
        probabilities[test] = fold_probability
        prediction = (fold_probability >= 0.5).astype(np.int64)
        fold_reports.append(
            {
                "fold": fold,
                "held_out_parent_pid": str(group_values[test][0]),
                "page_count": len(test),
                "music_page_count": int(y[test].sum()),
                "correct_count": int(np.sum(prediction == y[test])),
            }
        )
    prediction = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    for row, probability, predicted in zip(
        rows,
        probabilities,
        prediction,
        strict=True,
    ):
        row["leave_one_edition_out_music_probability"] = float(
            probability
        )
        row["predicted_label"] = (
            "music" if int(predicted) else "non_music"
        )
        row["correct"] = bool(
            int(predicted)
            == int(row["primary_visual_label"] == "music")
        )
    trained = _model()
    trained.fit(x, y)
    importances = sorted(
        zip(
            FEATURE_NAMES,
            trained.feature_importances_.tolist(),
            strict=True,
        ),
        key=lambda item: (-item[1], item[0]),
    )
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "role": REPORT_ROLE,
        "primary_review_report_path": str(
            primary_review_report_path.resolve()
        ),
        "primary_review_report_sha256": sha256_file(
            primary_review_report_path
        ),
        "label_provenance": (
            "single_primary_visual_reviewer_not_independent_annotation"
        ),
        "split_policy": "leave_one_parent_edition_out",
        "page_count": len(y),
        "parent_edition_count": len(set(groups)),
        "music_page_count": int(y.sum()),
        "non_music_page_count": int(len(y) - y.sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision_music": float(
            precision_score(y, prediction, zero_division=0)
        ),
        "recall_music": float(
            recall_score(y, prediction, zero_division=0)
        ),
        "f1_music": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix_labels": ["non_music", "music"],
        "confusion_matrix": matrix.tolist(),
        "error_count": int(np.sum(prediction != y)),
        "feature_names": list(FEATURE_NAMES),
        "feature_importance": [
            {"feature": name, "importance": importance}
            for name, importance in importances
        ],
        "folds": fold_reports,
        "pages": rows,
        "production_threshold_selected": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "labels_have_only_one_primary_reviewer",
            "eight_related_chopin_edition_objects_are_not_domain_coverage",
            "threshold_not_calibrated_on_independent_holdout",
            "classifier_not_packaged_for_runtime",
        ],
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary_review_report_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.primary_review_report_path.resolve(),
        args.output_path.resolve(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "page_count",
                    "parent_edition_count",
                    "music_page_count",
                    "non_music_page_count",
                    "accuracy",
                    "precision_music",
                    "recall_music",
                    "f1_music",
                    "error_count",
                    "training_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
