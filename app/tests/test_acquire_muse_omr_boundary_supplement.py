from __future__ import annotations

import pytest

from app.tools.acquire_muse_omr_boundary_supplement import (
    select_supplement_pairs,
)
from app.tools.acquire_muse_omr_benchmark import REPOSITORY, REVISION
from scorescan.product_scope import PRODUCTION_BOUNDARY_CONTRACT_VERSION


def _case(
    pair_id: int,
    fingerprint: str,
    shape: str,
    *,
    accepted: bool = True,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "work_fingerprint": fingerprint,
        "boundary": {
            "contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
            "accepted": accepted,
            "score_shape": shape,
        },
    }


def test_select_supplement_pairs_filters_shapes_and_rejections() -> None:
    works = {1: "a" * 64, 2: "b" * 64, 3: "c" * 64}
    catalog = {
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "role": "candidate_catalog_not_training_or_evaluation",
        "repository": REPOSITORY,
        "revision": REVISION,
        "reserved_work_overlap": [],
        "cases": [
            _case(1, "a" * 64, "single_staff_solo"),
            _case(2, "b" * 64, "keyboard"),
            _case(
                3,
                "c" * 64,
                "keyboard_plus_single_staff_ensemble",
                accepted=False,
            ),
        ],
    }

    selected = select_supplement_pairs(
        catalog,
        score_shapes={"single_staff_solo"},
        work_by_pair=works,
        forbidden_works=set(),
    )

    assert selected == [(1, "a" * 64, "single_staff_solo")]


def test_select_supplement_pairs_rejects_training_or_holdout_leakage() -> None:
    catalog = {
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "role": "candidate_catalog_not_training_or_evaluation",
        "repository": REPOSITORY,
        "revision": REVISION,
        "reserved_work_overlap": [],
        "cases": [_case(1, "a" * 64, "single_staff_solo")],
    }

    with pytest.raises(ValueError, match="work-level isolation"):
        select_supplement_pairs(
            catalog,
            score_shapes={"single_staff_solo"},
            work_by_pair={1: "a" * 64},
            forbidden_works={"a" * 64},
        )
