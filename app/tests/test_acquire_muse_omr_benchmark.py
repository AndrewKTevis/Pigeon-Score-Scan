from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "acquire_muse_omr_benchmark.py"
SPEC = importlib.util.spec_from_file_location("acquire_muse_omr_benchmark", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _manifest() -> bytes:
    rows = {
        str(index): {
            "score": f"mscz/score_file_{index}.mscz",
            "pdf_image": f"pdf/score_file_{index}.pdf",
        }
        for index in range(MODULE.PAIR_COUNT)
    }
    return json.dumps(rows, separators=(",", ":")).encode()


def test_safe_relative_path_rejects_traversal() -> None:
    for value in ("../escape", "/absolute", "C:/absolute", "safe\\escape"):
        with pytest.raises(ValueError):
            MODULE.safe_relative_path(value)
    assert MODULE.safe_relative_path("pdf/score_file_1.pdf") == Path(
        "pdf", "score_file_1.pdf"
    )


def test_pair_manifest_validates_pinned_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _manifest()
    monkeypatch.setattr(MODULE, "MANIFEST_BYTES", len(payload))
    monkeypatch.setattr(MODULE, "MANIFEST_SHA256", MODULE.sha256_bytes(payload))
    pairs = MODULE.parse_pair_manifest(payload)
    assert len(pairs) == MODULE.PAIR_COUNT
    assert pairs[17] == ("mscz/score_file_17.mscz", "pdf/score_file_17.pdf")

    altered = json.loads(payload)
    altered["17"]["pdf_image"] = "../escape.pdf"
    altered_payload = json.dumps(altered, separators=(",", ":")).encode()
    monkeypatch.setattr(MODULE, "MANIFEST_BYTES", len(altered_payload))
    monkeypatch.setattr(
        MODULE, "MANIFEST_SHA256", MODULE.sha256_bytes(altered_payload)
    )
    with pytest.raises(ValueError, match="unsafe"):
        MODULE.parse_pair_manifest(altered_payload)


def test_stable_pair_selection_is_deterministic_and_bounded() -> None:
    first = MODULE.stable_pair_ids(range(100), limit=12, seed=7)
    second = MODULE.stable_pair_ids(reversed(range(100)), limit=12, seed=7)
    assert first == second
    assert len(first) == 12
    assert first == sorted(first)
    assert MODULE.stable_pair_ids(range(4), limit=None, seed=7) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        MODULE.stable_pair_ids(range(4), limit=0, seed=7)


def test_annotation_stratified_pairs_extend_without_replacing_base() -> None:
    base = MODULE.stable_pair_ids(range(100), limit=12, seed=7)
    extension = next(value for value in range(100) if value not in base)
    selected, required = MODULE.augmented_pair_ids(
        range(100),
        limit=12,
        seed=7,
        required_pair_ids=[extension, extension],
    )

    assert set(base) <= set(selected)
    assert selected == sorted(base + [extension])
    assert required == [extension]
    with pytest.raises(ValueError, match="unavailable"):
        MODULE.augmented_pair_ids(
            range(4),
            limit=2,
            seed=7,
            required_pair_ids=[99],
        )


def test_annotation_stratified_pairs_must_add_independent_works() -> None:
    work_by_pair = {
        1: "a" * 64,
        2: "b" * 64,
        3: "b" * 64,
        4: "c" * 64,
    }
    assert MODULE.annotation_stratified_work_fingerprints(
        [1],
        [2, 4],
        work_by_pair,
    ) == ["b" * 64, "c" * 64]
    with pytest.raises(ValueError, match="base holdout"):
        MODULE.annotation_stratified_work_fingerprints(
            [1],
            [1],
            work_by_pair,
        )
    with pytest.raises(ValueError, match="duplicate works"):
        MODULE.annotation_stratified_work_fingerprints(
            [1],
            [2, 3],
            work_by_pair,
        )


def test_next_link_and_selected_file_validation() -> None:
    header = '<https://example.test/next?cursor=x>; rel="next"'
    assert MODULE.next_link(header) == "https://example.test/next?cursor=x"
    assert MODULE.next_link(None) is None

    pairs = {1: ("mscz/score_file_1.mscz", "pdf/score_file_1.pdf")}
    index = {
        path: MODULE.RemoteFile(path=path, size=100, sha256=None)
        for path in pairs[1]
    }
    result = MODULE.selected_remote_files(pairs, [1], index)
    assert [row.path for row in result] == [
        "mscz/score_file_1.mscz",
        "pdf/score_file_1.pdf",
    ]
    with pytest.raises(ValueError, match="missing"):
        MODULE.selected_remote_files(pairs, [1], {})


def test_reuse_or_download_hardlinks_verified_catalog_file(
    tmp_path: Path,
) -> None:
    reuse = tmp_path / "catalog"
    source = reuse / "mscz" / "score_file_1.mscz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"verified-score")
    remote = MODULE.RemoteFile(
        path="mscz/score_file_1.mscz",
        size=source.stat().st_size,
        sha256=MODULE.sha256_file(source),
    )
    output = tmp_path / "holdout"

    row = MODULE.reuse_or_download_file(
        remote,
        output,
        reuse_dirs=(reuse,),
        timeout=1,
        retries=1,
    )

    destination = output / remote.path
    assert destination.read_bytes() == b"verified-score"
    assert row["status"] in {"hardlinked", "copied"}
    assert row["local_path"] == str(destination.resolve())


def test_pdf_coverage_counts_only_one_variant_per_work(tmp_path: Path) -> None:
    import fitz

    pairs = {
        1: ("mscz/score_file_1.mscz", "pdf/score_file_1.pdf"),
        2: ("mscz/score_file_2.mscz", "pdf/score_file_2.pdf"),
        3: ("mscz/score_file_3.mscz", "pdf/score_file_3.pdf"),
    }
    for pair_id, pages in ((1, 2), (2, 2), (3, 3)):
        path = tmp_path / pairs[pair_id][1]
        path.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        for _ in range(pages):
            document.new_page()
        document.save(path)
        document.close()

    coverage = MODULE._pdf_coverage(
        tmp_path,
        pairs,
        [1, 2, 3],
        {1: "a" * 64, 2: "a" * 64, 3: "b" * 64},
    )

    assert coverage["selected_pdf_page_count"] == 7
    assert coverage["selected_independent_work_pdf_page_count"] == 5
    assert [
        row["pair_id"]
        for row in coverage["independent_work_pdf_representatives"]
    ] == [1, 3]
