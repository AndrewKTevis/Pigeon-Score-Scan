from pathlib import Path

from app.tools.check_release_readiness import find_non_cache_temporary_files


def test_tmp_scan_skips_python_caches_but_covers_training_assets(
    tmp_path: Path,
) -> None:
    training_tmp = tmp_path / "training_data" / "active.tmp"
    source_tmp = tmp_path / "app" / "pending.tmp"
    ignored_tmp = tmp_path / "app" / "__pycache__" / "import.tmp"
    for path in (training_tmp, source_tmp, ignored_tmp):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tmp")

    assert find_non_cache_temporary_files(tmp_path) == [
        "app/pending.tmp",
        "training_data/active.tmp",
    ]
