from __future__ import annotations

"""Versioned integrity metadata for final page-recognition checkpoints.

External OMR results are already content-addressed per image variant.  The final
``selected.musicxml`` checkpoint additionally depends on ScoreScan's policy, bundled
calibration models, page layout, and workflow version.  Without this manifest, upgrading
the application could silently reuse a result selected under older decision rules.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import APP_VERSION, WORKFLOW_VERSION
from .engine_cache import homr_version
from .model_registry import MANIFEST_NAME
from .policy import DEFAULT_POLICY
from .util import atomic_write_json, read_json, sha256_file

CHECKPOINT_FORMAT = 1


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_workflow_fingerprint(resources_dir: Path | None = None) -> str:
    resources = resources_dir or Path(__file__).resolve().parent / "resources"
    manifest_path = resources / MANIFEST_NAME
    manifest_hash = sha256_file(manifest_path) if manifest_path.exists() else "manifest-absent"
    return _stable_digest(
        {
            "app_version": APP_VERSION,
            "workflow_version": WORKFLOW_VERSION,
            "policy": DEFAULT_POLICY.to_dict(),
            "model_manifest_sha256": manifest_hash,
            "checkpoint_format": CHECKPOINT_FORMAT,
        }
    )


@dataclass(frozen=True)
class RecognitionCheckpointKey:
    image_sha256: str
    layout_sha256: str
    engine_version: str
    workflow_fingerprint: str

    @classmethod
    def for_page(cls, image_path: Path, layout_path: Path | None = None) -> "RecognitionCheckpointKey":
        layout_hash = "layout-absent"
        if layout_path and layout_path.exists():
            layout_hash = sha256_file(layout_path)
        return cls(
            image_sha256=sha256_file(image_path),
            layout_sha256=layout_hash,
            engine_version=homr_version(),
            workflow_fingerprint=current_workflow_fingerprint(),
        )


class RecognitionCheckpoint:
    def __init__(self, xml_path: Path) -> None:
        self.xml_path = xml_path
        self.manifest_path = xml_path.with_suffix(".checkpoint.json")

    def is_valid(self, key: RecognitionCheckpointKey) -> bool:
        if not self.xml_path.exists() or self.xml_path.stat().st_size <= 300:
            return False
        payload = read_json(self.manifest_path)
        if not isinstance(payload, dict) or int(payload.get("format", 0) or 0) != CHECKPOINT_FORMAT:
            return False
        expected = {
            "image_sha256": key.image_sha256,
            "layout_sha256": key.layout_sha256,
            "engine_version": key.engine_version,
            "workflow_fingerprint": key.workflow_fingerprint,
        }
        if any(payload.get(name) != value for name, value in expected.items()):
            return False
        try:
            return payload.get("xml_sha256") == sha256_file(self.xml_path)
        except OSError:
            return False

    def invalidate(self, reason: str) -> None:
        self.manifest_path.unlink(missing_ok=True)
        if not self.xml_path.exists():
            return
        stale = self.xml_path.with_name(self.xml_path.stem + f".stale-{reason}.musicxml")
        stale.unlink(missing_ok=True)
        try:
            self.xml_path.replace(stale)
        except OSError:
            self.xml_path.unlink(missing_ok=True)

    def commit(
        self,
        key: RecognitionCheckpointKey,
        *,
        selected_variant: str,
        consensus_applied: bool,
    ) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "format": CHECKPOINT_FORMAT,
                "image_sha256": key.image_sha256,
                "layout_sha256": key.layout_sha256,
                "engine": "homr",
                "engine_version": key.engine_version,
                "workflow_fingerprint": key.workflow_fingerprint,
                "selected_variant": selected_variant,
                "consensus_applied": bool(consensus_applied),
                "xml_sha256": sha256_file(self.xml_path),
                "xml_size": self.xml_path.stat().st_size,
            },
        )
