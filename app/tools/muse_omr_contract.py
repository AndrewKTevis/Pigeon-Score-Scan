"""Lightweight, dependency-free Muse OMR dataset role contracts."""

TRAINING_SELECTION_ROLE = "external_scan_degraded_training_only"
TRAINING_REGION_ROLE = (
    "training_only_disjoint_from_external_release_holdout"
)
BENCHMARK_SELECTION_ROLE = (
    "external_scan_degraded_development_benchmark_not_training"
)
SCAN_DEGRADED_IMAGE_ORIGIN = "synthetic_scan_degraded_render"
PHYSICAL_SCAN_RELEASE_BENCHMARK_ROLE = (
    "external_physical_scan_frozen_release_benchmark"
)

# This aliases the deployment contract so dataset preparation and runtime
# evaluation cannot silently use different definitions of a scan page.
from scorescan.semantic_detector_contract import (  # noqa: E402
    SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO as MAXIMUM_SCAN_PAGE_ASPECT_RATIO,
    SEMANTIC_DETECTOR_PAGE_SHAPE_CONTRACT as SCAN_PAGE_SHAPE_CONTRACT,
    page_aspect_ratio as scan_page_aspect_ratio,
    page_shape_is_supported as scan_page_shape_is_supported,
)
