from __future__ import annotations

import json
from pathlib import Path

from app.tools.catalog_nifc_chopin_reference_quality import (
    EXPECTED_REVISION,
    ROLE,
    conservative_instrumentation_hint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_infers_only_explicit_single_keyboard_titles() -> None:
    assert conservative_instrumentation_hint(
        {"OTL": "Nocturne pour le Pianoforté"}
    ) == ("1 piano", "title_explicit_single_keyboard_only")
    assert conservative_instrumentation_hint(
        {"OTL": "Sonate pour piano et violoncelle"}
    ) == ("", "no_safe_instrumentation_inference")
    assert conservative_instrumentation_hint(
        {"OTL": "Rondo pour deux pianos"}
    ) == ("", "no_safe_instrumentation_inference")
    assert conservative_instrumentation_hint(
        {"OTL": "Mazurka"}
    ) == ("", "no_safe_instrumentation_inference")


def test_checked_in_catalog_is_pinned_and_never_authorizes_use() -> None:
    path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "training_metadata"
        / "benchmarks"
        / "nifc_chopin_reference_quality_v1.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["role"] == ROLE
    assert report["repository_revision"] == EXPECTED_REVISION
    assert report["reference_count"] == 512
    assert report["training_authorized"] is False
    assert report["evaluation_authorized"] is False
    assert report["release_authorized"] is False
    assert all(
        case["training_authorized"] is False
        for case in report["high_priority_candidates"]
    )
