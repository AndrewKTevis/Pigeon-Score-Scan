"""COCO-style detection AP without the legacy 100 detections/image cap."""

from __future__ import annotations

import math
from typing import Any

DENSE_MAP_VERSION = "dense-page-coco-101point-unbounded-detections@1"
COCO_IOU_THRESHOLDS = tuple(
    round(0.50 + 0.05 * index, 2) for index in range(10)
)


def _pairwise_box_iou(left: Any, right: Any) -> Any:
    import numpy as np

    left = np.asarray(left, dtype=np.float64).reshape(-1, 4)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 4)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    intersection_left_top = np.maximum(left[:, None, :2], right[None, :, :2])
    intersection_right_bottom = np.minimum(
        left[:, None, 2:],
        right[None, :, 2:],
    )
    intersection_size = np.maximum(
        0.0,
        intersection_right_bottom - intersection_left_top,
    )
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    left_area = np.maximum(
        1e-9,
        (left[:, 2] - left[:, 0]) * (left[:, 3] - left[:, 1]),
    )
    right_area = np.maximum(
        1e-9,
        (right[:, 2] - right[:, 0]) * (right[:, 3] - right[:, 1]),
    )
    return intersection / np.maximum(
        left_area[:, None] + right_area[None, :] - intersection,
        1e-9,
    )


def _interpolated_ap_101(
    true_positive: Any,
    false_positive: Any,
    *,
    total_targets: int,
) -> tuple[float, float]:
    import numpy as np

    if total_targets <= 0:
        raise ValueError("average precision requires positive target support")
    true_positive = np.asarray(true_positive, dtype=np.float64)
    false_positive = np.asarray(false_positive, dtype=np.float64)
    cumulative_true = np.cumsum(true_positive)
    cumulative_false = np.cumsum(false_positive)
    recall = cumulative_true / total_targets
    precision = cumulative_true / np.maximum(
        cumulative_true + cumulative_false,
        1e-12,
    )
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    sampled_precision = []
    for recall_threshold in np.linspace(0.0, 1.0, 101):
        eligible = precision_envelope[recall >= recall_threshold]
        sampled_precision.append(float(eligible.max()) if eligible.size else 0.0)
    maximum_recall = float(recall[-1]) if recall.size else 0.0
    return float(np.mean(sampled_precision)), maximum_recall


def compute_dense_detection_metrics(
    outputs: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """Compute greedy COCO 101-point AP while retaining every prediction.

    TorchMetrics 1.6.2 returns ``map=-1`` and ``map_per_class=-1`` when COCO's
    third maxDets value differs from 100. Dense music-score tiles and pages can
    legitimately exceed that limit. This implementation matches standard
    TorchMetrics/COCO results when the standard cap is not reached, but never
    turns correct lower-ranked notation detections into artificial misses.
    """

    import numpy as np

    if len(outputs) != len(targets):
        raise ValueError("detection outputs and targets have different lengths")
    if not iou_thresholds or any(
        not math.isfinite(value) or not 0 < value <= 1
        for value in iou_thresholds
    ):
        raise ValueError("IoU thresholds must be finite and in (0, 1]")

    normalized: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    supported_labels: set[int] = set()
    total_predictions = 0
    total_targets = 0
    for output, target in zip(outputs, targets, strict=True):
        prediction_boxes = np.asarray(
            output.get("boxes", ()), dtype=np.float64
        ).reshape(-1, 4)
        prediction_scores = np.asarray(
            output.get("scores", ()), dtype=np.float64
        ).reshape(-1)
        prediction_labels = np.asarray(
            output.get("labels", ()), dtype=np.int64
        ).reshape(-1)
        target_boxes = np.asarray(
            target.get("boxes", ()), dtype=np.float64
        ).reshape(-1, 4)
        target_labels = np.asarray(
            target.get("labels", ()), dtype=np.int64
        ).reshape(-1)
        if not (
            len(prediction_boxes)
            == len(prediction_scores)
            == len(prediction_labels)
        ):
            raise ValueError("detector output lengths disagree")
        if len(target_boxes) != len(target_labels):
            raise ValueError("detector target lengths disagree")
        if (
            not np.isfinite(prediction_boxes).all()
            or not np.isfinite(prediction_scores).all()
            or not np.isfinite(target_boxes).all()
        ):
            raise ValueError("detector metric input contains non-finite values")
        normalized.append(
            (
                prediction_boxes,
                prediction_scores,
                prediction_labels,
                target_boxes,
                target_labels,
            )
        )
        supported_labels.update(int(label) for label in target_labels)
        total_predictions += len(prediction_boxes)
        total_targets += len(target_boxes)

    labels = sorted(supported_labels)
    per_class_ap: list[float] = []
    per_class_recall: list[float] = []
    ap_by_iou: dict[float, list[float]] = {
        threshold: [] for threshold in iou_thresholds
    }
    for label in labels:
        total_class_targets = sum(
            int(np.count_nonzero(target_labels == label))
            for (
                _prediction_boxes,
                _prediction_scores,
                _prediction_labels,
                _target_boxes,
                target_labels,
            ) in normalized
        )
        records_by_iou: dict[
            float,
            list[tuple[float, int, int, bool]],
        ] = {threshold: [] for threshold in iou_thresholds}
        for image_index, (
            prediction_boxes,
            prediction_scores,
            prediction_labels,
            target_boxes,
            target_labels,
        ) in enumerate(normalized):
            selected_prediction_indices = np.flatnonzero(
                prediction_labels == label
            )
            selected_prediction_indices = sorted(
                (int(index) for index in selected_prediction_indices),
                key=lambda index: (
                    -float(prediction_scores[index]),
                    tuple(float(value) for value in prediction_boxes[index]),
                    index,
                ),
            )
            class_prediction_boxes = prediction_boxes[
                selected_prediction_indices
            ]
            class_target_boxes = target_boxes[target_labels == label]
            ious = _pairwise_box_iou(
                class_prediction_boxes,
                class_target_boxes,
            )
            for threshold in iou_thresholds:
                used_targets: set[int] = set()
                for rank, prediction_index in enumerate(
                    selected_prediction_indices
                ):
                    best_target = None
                    best_iou = -1.0
                    for target_index, iou in enumerate(ious[rank]):
                        if target_index in used_targets:
                            continue
                        if float(iou) > best_iou:
                            best_iou = float(iou)
                            best_target = target_index
                    matched = (
                        best_target is not None and best_iou >= threshold
                    )
                    if matched:
                        used_targets.add(int(best_target))
                    records_by_iou[threshold].append(
                        (
                            float(prediction_scores[prediction_index]),
                            image_index,
                            rank,
                            matched,
                        )
                    )

        class_aps = []
        class_recalls = []
        for threshold in iou_thresholds:
            records = sorted(
                records_by_iou[threshold],
                key=lambda item: (-item[0], item[1], item[2]),
            )
            true_positive = [1.0 if record[3] else 0.0 for record in records]
            false_positive = [0.0 if record[3] else 1.0 for record in records]
            ap, recall = _interpolated_ap_101(
                true_positive,
                false_positive,
                total_targets=total_class_targets,
            )
            class_aps.append(ap)
            class_recalls.append(recall)
            ap_by_iou[threshold].append(ap)
        per_class_ap.append(float(np.mean(class_aps)))
        per_class_recall.append(float(np.mean(class_recalls)))

    def mean_for_iou(value: float) -> float:
        matched = next(
            (
                threshold
                for threshold in iou_thresholds
                if math.isclose(threshold, value, abs_tol=1e-9)
            ),
            None,
        )
        if matched is None:
            return -1.0
        values = ap_by_iou[matched]
        return float(np.mean(values)) if values else -1.0

    return {
        "evaluation_protocol": DENSE_MAP_VERSION,
        "iou_thresholds": list(iou_thresholds),
        "unbounded_detections_per_image": True,
        "evaluated_predictions": total_predictions,
        "evaluated_targets": total_targets,
        "map": float(np.mean(per_class_ap)) if per_class_ap else -1.0,
        "map_50": mean_for_iou(0.50),
        "map_75": mean_for_iou(0.75),
        "map_per_class": per_class_ap,
        "mar_dense_per_class": per_class_recall,
        "mar_dense": (
            float(np.mean(per_class_recall)) if per_class_recall else -1.0
        ),
        "classes": labels,
    }
