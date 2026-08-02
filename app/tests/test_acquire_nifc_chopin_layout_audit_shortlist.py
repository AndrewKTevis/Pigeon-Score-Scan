from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools.acquire_nifc_chopin_layout_audit_shortlist import (
    MANIFEST_ROLE,
    validate_shortlist,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "training" / (
    "nifc_chopin_layout_audit_shortlist.v1.json"
)
REPOSITORY = ROOT / "training_data" / "external" / "catalogs" / (
    "humdrum-chopin-first-editions-ccby4"
)


def test_checked_in_shortlist_is_hash_bound_and_never_authorized() -> None:
    if not MANIFEST.is_file() or not REPOSITORY.is_dir():
        pytest.skip("optional NIFC reference repository is not installed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["role"] == MANIFEST_ROLE
    assert manifest["training_authorized"] is False
    assert manifest["evaluation_authorized"] is False
    assert manifest["release_authorized"] is False
    discovery_path, discovery, candidates = validate_shortlist(
        manifest,
        manifest_path=MANIFEST,
        repository=REPOSITORY,
    )
    assert discovery_path.is_file()
    assert discovery["training_authorized"] is False
    assert len(candidates) == 8
    assert len({candidate["parent_pid"] for candidate in candidates}) == 8
    assert len(
        {candidate["reference_sha256"] for candidate in candidates}
    ) == 6
