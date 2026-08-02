import json
from pathlib import Path

from scorescan.model_registry import build_manifest, load_verified_json
from scorescan.util import atomic_write_json


def test_model_manifest_detects_tampering(tmp_path: Path) -> None:
    model = tmp_path / "candidate_calibrator.json"
    model.write_text(json.dumps({"model_version": "test-1"}), encoding="utf-8")
    atomic_write_json(tmp_path / "model_manifest.json", build_manifest(tmp_path))
    loaded = load_verified_json(model, "page_candidate_calibration")
    assert loaded.verified
    assert loaded.payload["model_version"] == "test-1"

    model.write_text(json.dumps({"model_version": "test-2"}), encoding="utf-8")
    tampered = load_verified_json(model, "page_candidate_calibration")
    assert not tampered.verified
    assert tampered.payload == {}
    assert tampered.status == "hash_mismatch"


def test_model_manifest_checks_size_before_json_parsing(tmp_path: Path) -> None:
    from scorescan.model_registry import MAX_MODEL_BYTES

    model = tmp_path / "candidate_calibrator.json"
    model.write_text(json.dumps({"model_version": "test-1"}), encoding="utf-8")
    manifest = build_manifest(tmp_path)
    manifest["models"][0]["bytes"] += 1
    atomic_write_json(tmp_path / "model_manifest.json", manifest)
    mismatch = load_verified_json(model, "page_candidate_calibration")
    assert not mismatch.verified
    assert mismatch.payload == {}
    assert mismatch.status == "size_mismatch"

    (tmp_path / "model_manifest.json").unlink()
    with model.open("wb") as handle:
        handle.truncate(MAX_MODEL_BYTES + 1)
    oversized = load_verified_json(model, "page_candidate_calibration")
    assert not oversized.verified
    assert oversized.payload == {}
    assert oversized.status == "model_size_limit"


def test_complete_model_manifest_audit_fails_closed_on_missing_or_tampered_resources(tmp_path: Path, monkeypatch) -> None:
    from scorescan import model_registry

    monkeypatch.setattr(
        model_registry,
        "MODEL_ROLES",
        {
            "candidate_calibrator.json": "page_candidate_calibration",
            "measure_calibrator.json": "measure_candidate_calibration",
        },
    )
    for filename, version in (
        ("candidate_calibrator.json", "candidate-1"),
        ("measure_calibrator.json", "measure-1"),
    ):
        (tmp_path / filename).write_text(json.dumps({"model_version": version}), encoding="utf-8")
    atomic_write_json(tmp_path / "model_manifest.json", model_registry.build_manifest(tmp_path))
    verified = model_registry.audit_model_manifest(tmp_path)
    assert verified.verified
    assert verified.verified_count == 2
    assert not verified.errors

    (tmp_path / "measure_calibrator.json").write_text(json.dumps({"model_version": "tampered"}), encoding="utf-8")
    failed = model_registry.audit_model_manifest(tmp_path)
    assert not failed.verified
    assert failed.verified_count == 1
    assert any(error.startswith("measure_calibrator.json:") for error in failed.errors)


def test_model_versions_only_reports_verified_resources(tmp_path: Path, monkeypatch) -> None:
    from scorescan import model_registry

    monkeypatch.setattr(
        model_registry,
        "MODEL_ROLES",
        {"candidate_calibrator.json": "page_candidate_calibration"},
    )
    model = tmp_path / "candidate_calibrator.json"
    model.write_text(json.dumps({"model_version": "candidate-1"}), encoding="utf-8")
    atomic_write_json(tmp_path / "model_manifest.json", model_registry.build_manifest(tmp_path))
    assert model_registry.model_versions(tmp_path) == {"page_candidate_calibration": "candidate-1"}

    model.write_text(json.dumps({"model_version": "candidate-2"}), encoding="utf-8")
    assert model_registry.model_versions(tmp_path) == {}


def test_model_loader_rejects_snapshot_size_change(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "candidate_calibrator.json"
    model.write_text(json.dumps({"model_version": "test-1"}), encoding="utf-8")
    atomic_write_json(tmp_path / "model_manifest.json", build_manifest(tmp_path))

    original = Path.read_bytes

    def truncated_read(path: Path) -> bytes:
        data = original(path)
        return data[:-1] if path == model else data

    monkeypatch.setattr(Path, "read_bytes", truncated_read)
    loaded = load_verified_json(model, "page_candidate_calibration")
    assert not loaded.verified
    assert loaded.payload == {}
    assert loaded.status == "size_changed_during_read"


def test_model_loader_rejects_invalid_utf8_after_hash_verification(tmp_path: Path) -> None:
    import hashlib

    model = tmp_path / "candidate_calibrator.json"
    data = b"\xff\xfe\xfd"
    model.write_bytes(data)
    atomic_write_json(
        tmp_path / "model_manifest.json",
        {
            "format": 1,
            "models": [{
                "file": model.name,
                "role": "page_candidate_calibration",
                "model_version": None,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }],
        },
    )
    loaded = load_verified_json(model, "page_candidate_calibration")
    assert not loaded.verified
    assert loaded.payload == {}
    assert loaded.status == "missing_or_invalid"
