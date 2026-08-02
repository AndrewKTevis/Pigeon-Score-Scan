from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from app.tools.audit_muse_semantic_tag_evidence import (
    audit,
    count_opening_tags,
    parse_required_tags,
)


def _dataset(
    root: Path,
    rows: list[tuple[int, str, str]],
) -> Path:
    dataset = root
    (dataset / "mscz").mkdir(parents=True)
    works = sorted({work for _pair, work, _payload in rows})
    selection = {
        "selected_pair_count": len(rows),
        "selected_pair_ids": [pair for pair, _work, _payload in rows],
        "selected_work_count": len(works),
        "selected_work_fingerprints": works,
        "pair_work_fingerprints": [
            {"pair_id": pair, "work_fingerprint": work}
            for pair, work, _payload in rows
        ],
    }
    (dataset / "selection.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )
    for pair, _work, payload in rows:
        with ZipFile(
            dataset / "mscz" / f"score_file_{pair}.mscz",
            "w",
            ZIP_DEFLATED,
        ) as archive:
            archive.writestr("score.mscx", payload)
    return dataset


def test_required_tags_and_opening_tag_count_are_strict() -> None:
    assert parse_required_tags(["Jump=25", "Marker=30"]) == {
        "Jump": 25,
        "Marker": 30,
    }
    assert count_opening_tags(
        "<museScore><Jump/><Jump id=\"1\"></Jump><Marker></Marker></museScore>",
        ("Jump", "Marker"),
    ) == {"Jump": 2, "Marker": 1}


def test_audit_counts_independent_works_and_forbidden_overlap(
    tmp_path: Path,
) -> None:
    holdout = _dataset(
        tmp_path / "holdout",
        [
            (1, "a" * 64, "<museScore><Jump/><Marker/></museScore>"),
            (2, "b" * 64, "<museScore><Jump/><Jump/></museScore>"),
        ],
    )
    training = _dataset(
        tmp_path / "training",
        [(3, "c" * 64, "<museScore/>")],
    )

    report = audit(
        dataset_dir=holdout,
        required_tags={"Jump": 3, "Marker": 1},
        forbidden_selection=training / "selection.json",
    )

    assert report["passed"]
    assert report["tag_counts"] == {"Jump": 3, "Marker": 1}
    assert report["independent_works_by_tag"] == {"Jump": 2, "Marker": 1}
    assert report["pair_overlap"] == []
    assert report["work_overlap"] == []

    overlapping = _dataset(
        tmp_path / "overlap",
        [(1, "a" * 64, "<museScore/>")],
    )
    failed = audit(
        dataset_dir=holdout,
        required_tags={"Jump": 4},
        forbidden_selection=overlapping / "selection.json",
    )
    assert not failed["passed"]
    assert failed["failures"] == [
        "pair_overlap=[1]",
        "work_overlap=['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa']",
        "Jump=3<4",
    ]
