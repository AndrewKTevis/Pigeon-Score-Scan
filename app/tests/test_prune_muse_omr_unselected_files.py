from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools import prune_muse_omr_unselected_files as module


def _write_dataset(root: Path) -> None:
    (root / "mscz").mkdir(parents=True)
    (root / "pdf").mkdir()
    for pair_id in (1, 2):
        (root / "mscz" / f"score_file_{pair_id}.mscz").write_bytes(b"score")
        (root / "pdf" / f"score_file_{pair_id}.pdf").write_bytes(b"pdf")
    selection = {
        "role": "external_scan_degraded_training_only",
        "selected_pair_count": 1,
        "selected_pair_ids": [1],
    }
    (root / "selection.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )
    (root / "provenance.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )


def test_prunes_only_unselected_pair_files(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    assert [path.name for path in module.unselected_files(tmp_path)] == [
        "score_file_2.mscz",
        "score_file_2.pdf",
    ]
    assert module.main(["--dataset-dir", str(tmp_path)]) == 0
    assert (tmp_path / "pdf" / "score_file_2.pdf").is_file()
    assert module.main(
        ["--dataset-dir", str(tmp_path), "--execute"]
    ) == 0
    assert (tmp_path / "mscz" / "score_file_1.mscz").is_file()
    assert (tmp_path / "pdf" / "score_file_1.pdf").is_file()
    assert not (tmp_path / "mscz" / "score_file_2.mscz").exists()
    assert not (tmp_path / "pdf" / "score_file_2.pdf").exists()


def test_refuses_mismatched_provenance(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    provenance = json.loads(
        (tmp_path / "provenance.json").read_text(encoding="utf-8")
    )
    provenance["selected_pair_ids"] = [2]
    (tmp_path / "provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contract failed"):
        module.unselected_files(tmp_path)
