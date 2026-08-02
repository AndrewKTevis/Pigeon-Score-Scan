from __future__ import annotations

"""Integrity verification for small bundled calibration models."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .util import read_json, sha256_file

MANIFEST_NAME = "model_manifest.json"
MANIFEST_FORMAT = 1
MAX_MODEL_BYTES = 8 * 1024 * 1024

MODEL_ROLES = {
    "candidate_calibrator.json": "page_candidate_calibration",
    "measure_calibrator.json": "measure_candidate_calibration",
    "direction_model.json": "music_direction_correction",
    "direction_patch_calibrator.json": "direction_patch_calibration",
    "visual_measure_calibrator.json": "visual_measure_compatibility",
    "event_calibrator.json": "event_candidate_calibration",
    "scan_variant_router.json": "scan_variant_routing",
    "context_calibrator.json": "context_candidate_calibration",
    "ensemble_calibrator.json": "ensemble_candidate_calibration",
    "selection_risk.json": "selection_risk_calibration",
    "pitch_patch_calibrator.json": "pitch_patch_calibration",
    "pitch_visual_guard.json": "pitch_visual_guard",
    "accidental_presence_guard.json": "accidental_presence_guard",
    "accent_visual_guard.json": "accent_visual_guard",
    "rhythm_patch_calibrator.json": "rhythm_patch_calibration",
    "rhythm_symbol_guard.json": "rhythm_symbol_guard",
    "attribute_patch_calibrator.json": "attribute_patch_calibration",
    "event_kind_patch_calibrator.json": "event_kind_patch_calibration",
    "event_kind_visual_guard.json": "event_kind_visual_guard",
    "event_presence_patch_calibrator.json": "event_presence_patch_calibration",
    "event_presence_visual_guard.json": "event_presence_visual_guard",
    "patch_transaction_calibrator.json": "patch_transaction_calibration",
    "chord_patch_calibrator.json": "chord_patch_calibration",
    "tie_patch_calibrator.json": "tie_patch_calibration",
    "slur_patch_calibrator.json": "slur_patch_calibration",
    "tie_visual_guard.json": "tie_visual_guard",
    "articulation_patch_calibrator.json": "articulation_patch_calibration",
    "ornament_patch_calibrator.json": "ornament_patch_calibration",
    "grace_patch_calibrator.json": "grace_patch_calibration",
    "tuplet_patch_calibrator.json": "tuplet_patch_calibration",
    "cross_tie_patch_calibrator.json": "cross_tie_patch_calibration",
    "barline_patch_calibrator.json": "barline_patch_calibration",
    "barline_classifier.json": "barline_classification",
    "barline_sequence_classifier.json": "barline_sequence_classification",
    "direction_anchor_classifier.json": "direction_anchor_classification",
    "page_orientation_classifier.json": "page_orientation_classification",
    "measure_count_resolver.json": "measure_count_resolution",
}


@dataclass(frozen=True)
class ModelLoadResult:
    payload: dict[str, object]
    verified: bool
    status: str


@dataclass(frozen=True)
class ModelManifestAudit:
    verified: bool
    expected_count: int
    manifest_count: int
    verified_count: int
    errors: tuple[str, ...]
    statuses: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_bounded_model_bytes(path: Path, expected_size: int | None = None) -> tuple[bytes | None, str]:
    """Read one model snapshot under a hard allocation bound.

    Hash verification and JSON parsing must operate on the same immutable byte
    snapshot.  This avoids a validation/parse race and halves steady-state I/O.
    """
    try:
        actual_size = path.stat().st_size
    except (FileNotFoundError, OSError):
        return None, "missing_or_invalid"
    if actual_size <= 0 or actual_size > MAX_MODEL_BYTES:
        return None, "model_size_limit"
    if expected_size is not None and expected_size != actual_size:
        return None, "size_mismatch"
    try:
        data = path.read_bytes()
    except (FileNotFoundError, OSError):
        return None, "missing_or_invalid"
    if len(data) != actual_size:
        return None, "size_changed_during_read"
    return data, "ok"


def _parse_model_bytes(data: bytes) -> dict[str, object] | None:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def load_verified_json(path: Path, role: str) -> ModelLoadResult:
    manifest_path = path.parent / MANIFEST_NAME
    manifest = read_json(manifest_path)
    # Development trees and training runs may not have a manifest yet. Keep the
    # source checkout inspectable, but still enforce the hard allocation limit.
    if not isinstance(manifest, dict):
        data, status = _read_bounded_model_bytes(path)
        if data is None:
            return ModelLoadResult({}, False, status)
        payload = _parse_model_bytes(data)
        if payload is None:
            return ModelLoadResult({}, False, "missing_or_invalid")
        return ModelLoadResult(payload, False, "manifest_absent")
    if int(manifest.get("format", 0) or 0) != MANIFEST_FORMAT:
        return ModelLoadResult({}, False, "manifest_format")
    entries = manifest.get("models")
    if not isinstance(entries, list):
        return ModelLoadResult({}, False, "manifest_entries")
    entry = next(
        (item for item in entries if isinstance(item, dict) and item.get("file") == path.name),
        None,
    )
    if entry is None or entry.get("role") != role:
        return ModelLoadResult({}, False, "manifest_entry_missing")
    try:
        expected_size = int(entry.get("bytes", -1))
    except (TypeError, ValueError, OverflowError):
        return ModelLoadResult({}, False, "manifest_size")
    data, status = _read_bounded_model_bytes(path, expected_size)
    if data is None:
        return ModelLoadResult({}, False, status)
    actual_hash = hashlib.sha256(data).hexdigest()
    if entry.get("sha256") != actual_hash:
        return ModelLoadResult({}, False, "hash_mismatch")
    payload = _parse_model_bytes(data)
    if payload is None:
        return ModelLoadResult({}, False, "missing_or_invalid")
    expected_version = entry.get("model_version")
    if expected_version and payload.get("model_version") != expected_version:
        return ModelLoadResult({}, False, "version_mismatch")
    return ModelLoadResult(payload, True, "verified")


def audit_model_manifest(resources_dir: Path) -> ModelManifestAudit:
    manifest = read_json(resources_dir / MANIFEST_NAME, {})
    errors: list[str] = []
    statuses: dict[str, str] = {}
    if not isinstance(manifest, dict):
        return ModelManifestAudit(False, len(MODEL_ROLES), 0, 0, ("manifest_missing",), {})
    if int(manifest.get("format", 0) or 0) != MANIFEST_FORMAT:
        errors.append("manifest_format")
    entries = manifest.get("models")
    if not isinstance(entries, list):
        return ModelManifestAudit(False, len(MODEL_ROLES), 0, 0, tuple(errors + ["manifest_entries"]), {})
    rows = [entry for entry in entries if isinstance(entry, dict)]
    if len(rows) != len(entries):
        errors.append("manifest_invalid_entry")
    files = [str(entry.get("file", "")).strip() for entry in rows]
    roles = [str(entry.get("role", "")).strip() for entry in rows]
    if len(files) != len(set(files)):
        errors.append("manifest_duplicate_file")
    if len(roles) != len(set(roles)):
        errors.append("manifest_duplicate_role")
    expected_files = set(MODEL_ROLES)
    listed_files = {filename for filename in files if filename}
    for filename in sorted(expected_files - listed_files):
        errors.append(f"missing_entry:{filename}")
    for filename in sorted(listed_files - expected_files):
        errors.append(f"unknown_entry:{filename}")
    verified_count = 0
    for filename, role in sorted(MODEL_ROLES.items()):
        loaded = load_verified_json(resources_dir / filename, role)
        statuses[filename] = loaded.status
        if loaded.verified:
            verified_count += 1
        else:
            errors.append(f"{filename}:{loaded.status}")
    return ModelManifestAudit(
        verified=not errors and verified_count == len(MODEL_ROLES),
        expected_count=len(MODEL_ROLES),
        manifest_count=len(rows),
        verified_count=verified_count,
        errors=tuple(dict.fromkeys(errors)),
        statuses=statuses,
    )


def build_manifest(resources_dir: Path) -> dict[str, object]:
    models: list[dict[str, object]] = []
    for filename, role in sorted(MODEL_ROLES.items()):
        path = resources_dir / filename
        if not path.exists():
            continue
        payload = read_json(path, {})
        models.append(
            {
                "file": filename,
                "role": role,
                "model_version": payload.get("model_version") if isinstance(payload, dict) else None,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {"format": MANIFEST_FORMAT, "models": models}


def model_versions(resources_dir: Path) -> dict[str, str]:
    """Return verified manifest role -> model version for audit reports.

    The conversion report should not duplicate model version strings maintained by
    training scripts.  Missing or malformed entries are omitted rather than guessed.
    """
    manifest = read_json(resources_dir / MANIFEST_NAME, {})
    result: dict[str, str] = {}
    if not isinstance(manifest, dict) or int(manifest.get("format", 0) or 0) != MANIFEST_FORMAT:
        return result
    entries = manifest.get("models")
    if not isinstance(entries, list):
        return result
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get("file", "")).strip()
        role = str(entry.get("role", "")).strip()
        version = str(entry.get("model_version", "")).strip()
        if not filename or MODEL_ROLES.get(filename) != role or not version:
            continue
        loaded = load_verified_json(resources_dir / filename, role)
        if loaded.verified and str(loaded.payload.get("model_version", "")).strip() == version:
            result[role] = version
    return result
