from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from app.tools import audit_semantic_replay_holdout_isolation as module
from app.tools.build_muse_omr_work_catalog import (
    mscx_payload_fingerprint,
)


def _payload(pitch: int, eid: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<museScore version="4.60"><Score>'
        f"<eid>{eid}</eid><Measure><Chord><eid>{eid}note</eid>"
        f"<Note><pitch>{pitch}</pitch></Note></Chord></Measure>"
        "</Score></museScore>"
    ).encode()


def _write_mscz(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr("score.mscx", payload)


def test_replay_audit_proves_canonical_work_isolation(tmp_path: Path) -> None:
    lieder = tmp_path / "lieder"
    quartet = tmp_path / "quartet"
    _write_mscz(lieder / "one.mscz", _payload(60, "replay-a"))
    quartet.mkdir()
    (quartet / "two.mscx").write_bytes(_payload(62, "replay-b"))
    holdout_fingerprint = mscx_payload_fingerprint(
        _payload(65, "holdout")
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {"selected_work_fingerprints": [holdout_fingerprint]}
        ),
        encoding="utf-8",
    )
    replay_report = tmp_path / "prepare-report.json"
    replay_report.write_text(
        json.dumps(
            {
                "purpose": (
                    "combined synthetic semantic geometry; "
                    "not real-scan validation"
                ),
                "split_intersections": {
                    "calibration_test": [],
                    "train_calibration": [],
                    "train_test": [],
                },
            }
        ),
        encoding="utf-8",
    )

    report = module.audit(
        project_root=tmp_path,
        holdout_selection=selection,
        replay_prepare_report=replay_report,
        replay_roots=[lieder, quartet],
        workers=2,
    )

    assert report["holdout_selected_works"] == 1
    assert report["replay_works"] == 2
    assert report["work_overlap"] == []
    assert [row["score_files"] for row in report["replay_roots"]] == [1, 1]


def test_replay_audit_rejects_holdout_content_with_different_eid(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "replay"
    _write_mscz(replay / "same.mscz", _payload(60, "replay-eid"))
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_work_fingerprints": [
                    mscx_payload_fingerprint(_payload(60, "holdout-eid"))
                ]
            }
        ),
        encoding="utf-8",
    )
    replay_report = tmp_path / "prepare-report.json"
    replay_report.write_text(
        json.dumps(
            {
                "purpose": (
                    "combined synthetic semantic geometry; "
                    "not real-scan validation"
                ),
                "split_intersections": {
                    "calibration_test": [],
                    "train_calibration": [],
                    "train_test": [],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps"):
        module.audit(
            project_root=tmp_path,
            holdout_selection=selection,
            replay_prepare_report=replay_report,
            replay_roots=[replay],
            workers=1,
        )
