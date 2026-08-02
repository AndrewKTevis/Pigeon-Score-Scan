from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.muse_omr_contract import (
    BENCHMARK_SELECTION_ROLE,
    TRAINING_SELECTION_ROLE,
)
from app.tools.promote_muse_training_cases_to_boundary import (
    BOUNDARY_FILTER_CONTRACT_VERSION,
    PARTITION_CONTRACT_VERSION,
    materialize_training_partition,
    select_promotions,
)


SHAPE_BY_CONFIGURATION = {
    "solo_monophonic": "single_staff_solo",
    "piano": "keyboard",
    "monophonic_ensemble": "single_staff_ensemble",
    "piano_plus_monophonic_ensemble": (
        "keyboard_plus_single_staff_ensemble"
    ),
}


def _case(
    index: int,
    configuration: str,
    pages: int,
) -> dict[str, object]:
    return {
        "id": f"case-{index}",
        "pair_id": index,
        "work_fingerprint": f"{index:064x}",
        "input_pdf_sha256": f"{index + 10000:064x}",
        "input_pdf_pages": pages,
        "boundary": {
            "accepted": True,
            "score_shape": SHAPE_BY_CONFIGURATION[configuration],
        },
    }


def test_promotions_close_only_configuration_deficits_without_model_output() -> None:
    base: list[dict[str, object]] = []
    index = 1
    for configuration, count, pages in (
        ("solo_monophonic", 10, 10),
        ("piano", 130, 8),
        ("monophonic_ensemble", 40, 10),
        ("piano_plus_monophonic_ensemble", 30, 10),
    ):
        for _ in range(count):
            base.append(_case(index, configuration, pages))
            index += 1
    training = [
        _case(index, "solo_monophonic", 100),
        _case(index + 1, "solo_monophonic", 100),
        _case(index + 2, "solo_monophonic", 100),
        _case(
            index + 3,
            "piano_plus_monophonic_ensemble",
            50,
        ),
        _case(
            index + 4,
            "piano_plus_monophonic_ensemble",
            50,
        ),
    ]

    promotions, report = select_promotions(base, training, seed=17)

    assert len(promotions) == 5
    assert report["model_outputs_used_for_selection"] is False
    assert report["coverage_gaps"] == {
        "solo_monophonic": 0,
        "piano": 0,
        "monophonic_ensemble": 0,
        "piano_plus_monophonic_ensemble": 0,
        "submitted_scan_page_count": 0,
        "source_group_count": 0,
    }
    assert report["promoted_pages_by_score_configuration"] == {
        "solo_monophonic": 300,
        "piano": 0,
        "monophonic_ensemble": 0,
        "piano_plus_monophonic_ensemble": 100,
    }


def test_promotions_fail_when_rare_configuration_is_unavailable() -> None:
    base = [
        _case(index, "piano", 10)
        for index in range(1, 201)
    ]
    with pytest.raises(ValueError, match="cannot close boundary coverage"):
        select_promotions(base, [], seed=1)


def test_materialized_training_removes_every_variant_of_promoted_work(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    work_a = "a" * 64
    work_b = "b" * 64
    work_c = "c" * 64
    rows = [
        {"pair_id": 1, "work_fingerprint": work_a},
        {"pair_id": 2, "work_fingerprint": work_a},
        {"pair_id": 3, "work_fingerprint": work_b},
        {"pair_id": 4, "work_fingerprint": work_c},
    ]
    dataset = {}
    for pair_id in (1, 2, 3, 4):
        score = source / "mscz" / f"score_file_{pair_id}.mscz"
        pdf = source / "pdf" / f"score_file_{pair_id}.pdf"
        score.parent.mkdir(exist_ok=True)
        pdf.parent.mkdir(exist_ok=True)
        score.write_bytes(f"score-{pair_id}".encode())
        pdf.write_bytes(f"pdf-{pair_id}".encode())
        dataset[str(pair_id)] = {
            "score": f"mscz/score_file_{pair_id}.mscz",
            "pdf_image": f"pdf/score_file_{pair_id}.pdf",
        }
    selection = {
        "format": 1,
        "license": "CC0-1.0",
        "repository": "fixture",
        "revision": "fixture-revision",
        "role": TRAINING_SELECTION_ROLE,
        "work_catalog_sha256": "c" * 64,
        "selection_seed": 1,
        "selected_pair_count": 4,
        "selected_pair_ids": [1, 2, 3, 4],
        "selected_work_count": 3,
        "selected_work_fingerprints": [work_a, work_b, work_c],
        "pair_work_fingerprints": rows,
        "reserved_holdout_pair_count": 0,
        "reserved_holdout_pair_ids": [],
        "reserved_holdout_work_count": 0,
        "reserved_holdout_work_fingerprints": [],
        "training_holdout_overlap": [],
        "training_holdout_work_overlap": [],
        "expected_download_bytes": 0,
    }
    (source / "selection.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )
    (source / "benchmark_dataset.json").write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "format": 1,
                "role": BENCHMARK_SELECTION_ROLE,
                "selected_pair_ids": [9],
                "selected_work_fingerprints": ["d" * 64],
                "pair_work_fingerprints": [
                    {
                        "pair_id": 9,
                        "work_fingerprint": "d" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text('{"format":1}', encoding="utf-8")

    result = materialize_training_partition(
        source_root=source,
        output_root=output,
        promoted_works={work_a},
        eligible_training_works={work_a, work_b},
        eligible_training_pair_ids={1, 2, 3},
        frozen_selections=[frozen],
        partition_plan=plan,
    )

    assert result["partition_contract_version"] == (
        PARTITION_CONTRACT_VERSION
    )
    assert result["selected_pair_ids"] == [3]
    assert result["selected_work_fingerprints"] == [work_b]
    assert result["promoted_evaluation_pair_ids"] == [1, 2]
    assert result["excluded_out_of_boundary_pair_ids"] == [4]
    assert result[
        "excluded_fully_out_of_boundary_work_fingerprints"
    ] == [work_c]
    assert result["reserved_holdout_pair_ids"] == [1, 2, 9]
    assert not (output / "mscz" / "score_file_1.mscz").exists()
    assert not (output / "mscz" / "score_file_4.mscz").exists()
    assert (output / "mscz" / "score_file_3.mscz").read_bytes() == b"score-3"
    provenance = json.loads(
        (output / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["downloaded_bytes"] == provenance[
        "expected_download_bytes"
    ]
    assert len(provenance["files"]) == 2

    filtered = materialize_training_partition(
        source_root=source,
        output_root=tmp_path / "filtered",
        promoted_works=set(),
        eligible_training_works={work_a, work_b},
        eligible_training_pair_ids={1, 2, 3},
        frozen_selections=[frozen],
        partition_plan=plan,
        partition_contract_version=BOUNDARY_FILTER_CONTRACT_VERSION,
    )
    assert filtered["partition_contract_version"] == (
        BOUNDARY_FILTER_CONTRACT_VERSION
    )
    assert filtered["selected_pair_ids"] == [1, 2, 3]
