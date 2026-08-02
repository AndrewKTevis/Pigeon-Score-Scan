import pytest

from app.tools.update_release_qa import (
    _read_test_log,
    invalidate_archive_reverification,
    parse_pytest_log,
)


def test_pytest_summary_parser_retains_skip_evidence() -> None:
    parsed = parse_pytest_log(
        """
SKIPPED [1] app/tests/test_train_deepscores_symbol_detector.py:384:
could not import 'torch'
1278 passed, 1 skipped, 2 warnings in 345.93s (0:05:45)
"""
    )

    assert parsed["passed"] == 1278
    assert parsed["skipped"] == 1
    assert parsed["warnings"] == 2
    assert parsed["failed"] == 0
    assert parsed["total"] == 1279
    assert len(parsed["skipped_details"]) == 1


def test_pytest_summary_parser_rejects_incomplete_log() -> None:
    with pytest.raises(ValueError, match="summary"):
        parse_pytest_log("test process started")


def test_release_qa_reads_windows_powershell_utf16_log(tmp_path) -> None:
    path = tmp_path / "pytest.out.log"
    path.write_text(
        "SKIPPED [1] app\\tests\\test_detector.py:1: no torch\n"
        "1280 passed, 1 skipped, 2 warnings in 316.66s\n",
        encoding="utf-16",
    )

    parsed = parse_pytest_log(_read_test_log(path))
    assert parsed["total"] == 1281
    assert parsed["skipped_details"]


def test_release_qa_refresh_invalidates_stale_archive_evidence() -> None:
    payload = {
        "release_package_reverification": {
            "completed": True,
            "pre_publication_windows_zip_sha256": "a" * 64,
        }
    }

    invalidate_archive_reverification(
        payload,
        refreshed_at="2026-08-01T00:00:00+00:00",
    )

    evidence = payload["release_package_reverification"]
    assert evidence["completed"] is False
    assert evidence["invalidated_at_utc"] == "2026-08-01T00:00:00+00:00"
    assert "pre_publication_windows_zip_sha256" not in evidence
