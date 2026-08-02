from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools import acquire_muse_omr_training as module


def test_training_selection_is_deterministic_and_disjoint() -> None:
    available = set(range(40))
    reserved = {2, 7, 11, 19}
    first = module.training_pair_ids(
        available,
        reserved=reserved,
        reserved_works=frozenset({"work-2", "work-7", "work-11", "work-19"}),
        work_by_pair={value: f"work-{value}" for value in available},
        limit=12,
        seed=29,
    )
    second = module.training_pair_ids(
        set(reversed(sorted(available))),
        reserved=reserved,
        reserved_works=frozenset({"work-2", "work-7", "work-11", "work-19"}),
        work_by_pair={value: f"work-{value}" for value in available},
        limit=12,
        seed=29,
    )
    assert first == second
    assert len(first) == 12
    assert not (set(first) & reserved)


def test_training_selection_rejects_invalid_holdout() -> None:
    with pytest.raises(ValueError, match="outside"):
        module.training_pair_ids(
            {1, 2, 3},
            reserved={4},
            reserved_works=frozenset({"work-4"}),
            work_by_pair={value: f"work-{value}" for value in {1, 2, 3}},
            limit=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="no pairs"):
        module.training_pair_ids(
            {1, 2},
            reserved={1, 2},
            reserved_works=frozenset({"work-1", "work-2"}),
            work_by_pair={1: "work-1", 2: "work-2"},
            limit=1,
            seed=1,
        )


def test_holdout_loader_requires_pinned_nontraining_role(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "repository": module.REPOSITORY,
                "revision": module.REVISION,
                    "role": "external_scan_degraded_development_benchmark_not_training",
                    "source_image_origin": (
                        module.SCAN_DEGRADED_IMAGE_ORIGIN
                    ),
                    "production_evidence_eligible": False,
                "selected_pair_ids": [9, 3, 9],
                "selected_work_count": 2,
                "selected_work_fingerprints": ["a" * 64, "b" * 64],
                "pair_work_fingerprints": [
                    {"pair_id": 9, "work_fingerprint": "a" * 64},
                    {"pair_id": 3, "work_fingerprint": "b" * 64},
                    {"pair_id": 9, "work_fingerprint": "a" * 64},
                ],
                "work_catalog_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        module.load_reserved_pair_ids(selection)

    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["selected_pair_ids"] = [9, 3]
    payload["pair_work_fingerprints"] = [
        {"pair_id": 9, "work_fingerprint": "a" * 64},
        {"pair_id": 3, "work_fingerprint": "b" * 64},
    ]
    selection.write_text(json.dumps(payload), encoding="utf-8")
    assert module.load_reserved_pair_ids(selection) == (3, 9)

    missing_origin = dict(payload)
    missing_origin.pop("source_image_origin")
    selection.write_text(json.dumps(missing_origin), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned benchmark"):
        module.load_reserved_pair_ids(selection)

    payload["role"] = "external_scan_degraded_training_only"
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned benchmark"):
        module.load_reserved_pair_ids(selection)
