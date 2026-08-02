from __future__ import annotations

from ppocr.metrics.det_metric import DetMetric


metric = DetMetric(
    iou_constraint=0.75,
    area_precision_constraint=0.6,
)
assert metric.evaluator.iou_constraint == 0.75
assert metric.evaluator.area_precision_constraint == 0.6
print("strict detection metric configuration passed")
