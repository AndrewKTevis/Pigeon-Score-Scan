from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import json

import pytest

from app.tools import build_muse_omr_work_catalog as module


def _score(path: Path, *, eid: str, pitch: int = 60) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "score.mscx",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<museScore version="4.60"><Score>'
                f"<eid>{eid}</eid><Measure><Chord><eid>{eid}note</eid>"
                f"<Note><pitch>{pitch}</pitch></Note></Chord></Measure>"
                "</Score></museScore>"
            ),
        )


def test_work_fingerprint_ignores_volatile_eids(tmp_path: Path) -> None:
    first = tmp_path / "first.mscz"
    second = tmp_path / "second.mscz"
    different = tmp_path / "different.mscz"
    _score(first, eid="one")
    _score(second, eid="two")
    _score(different, eid="three", pitch=61)
    assert module.work_fingerprint(first) == module.work_fingerprint(second)
    assert module.work_fingerprint(first) != module.work_fingerprint(different)
    with ZipFile(first) as archive:
        payload = archive.read("score.mscx")
    assert module.mscx_payload_fingerprint(payload) == (
        module.work_fingerprint(first)
    )


def test_work_fingerprint_rejects_ambiguous_archives(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.mscz"
    with ZipFile(path, "w") as archive:
        archive.writestr("one.mscx", "<museScore/>")
        archive.writestr("two.mscx", "<museScore/>")
    with pytest.raises(ValueError, match="exactly one"):
        module.work_fingerprint(path)


def test_work_fingerprint_ignores_part_excerpts(tmp_path: Path) -> None:
    path = tmp_path / "with-excerpt.mscz"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "score.mscx",
            "<museScore><Score><eid>main</eid></Score></museScore>",
        )
        archive.writestr(
            "Excerpts/part/part.mscx",
            "<museScore><Score><pitch>999</pitch></Score></museScore>",
        )
    assert len(module.work_fingerprint(path)) == 64


def test_catalog_loader_requires_complete_pair_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PAIR_COUNT", 2)
    path = tmp_path / "work-catalog.json"
    path.write_text(
        json.dumps(
            {
                "name": "scorescan-muse-omr-work-catalog-v1",
                "repository": module.REPOSITORY,
                "revision": module.REVISION,
                "fingerprint_version": module.FINGERPRINT_VERSION,
                "pair_count": 2,
                "work_count": 1,
                "pair_work_fingerprints": [
                    {"pair_id": 0, "work_fingerprint": "a" * 64},
                    {"pair_id": 1, "work_fingerprint": "a" * 64},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert module.load_work_catalog(path) == {0: "a" * 64, 1: "a" * 64}
