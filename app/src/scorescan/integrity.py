from __future__ import annotations

"""Deterministic artifact-bundle verification.

A conversion is marked completed only after every user-facing artifact has been read
back, structurally checked, hashed, and recorded in one manifest.  Individual files
are still written atomically; this module adds bundle-level consistency so crashes or
partial review updates cannot leave a MusicXML/MXL/report set from different revisions.
"""

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lxml import etree

from .util import atomic_write_json, sha256_file, utc_now_iso


@dataclass(frozen=True)
class ArtifactRecord:
    role: str
    relative_path: str
    size: int
    sha256: str
    valid: bool
    details: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "relative_path": self.relative_path,
            "size": self.size,
            "sha256": self.sha256,
            "valid": self.valid,
            "details": self.details,
        }


@dataclass(frozen=True)
class BundleIntegrity:
    valid: bool
    bundle_id: str
    records: tuple[ArtifactRecord, ...]
    errors: tuple[str, ...]
    manifest_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "bundle_id": self.bundle_id,
            "records": [record.to_dict() for record in self.records],
            "errors": list(self.errors),
            "manifest_path": self.manifest_path,
        }


def _validate_xml(path: Path, expected_root: str | None = None) -> tuple[bool, str]:
    try:
        root = etree.parse(str(path), etree.XMLParser(resolve_entities=False, no_network=True)).getroot()
    except Exception as exc:
        return False, f"XML 无法解析：{exc}"
    if expected_root and root.tag != expected_root:
        return False, f"根元素为 {root.tag}，预期 {expected_root}"
    return True, root.tag


def _validate_json(path: Path) -> tuple[bool, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"JSON 无法解析：{exc}"
    return True, type(value).__name__


def _validate_mxl(path: Path) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "META-INF/container.xml" not in names:
                return False, "缺少 META-INF/container.xml"
            container = etree.fromstring(archive.read("META-INF/container.xml"))
            rootfile = container.find(".//rootfile")
            if rootfile is None or not rootfile.get("full-path"):
                return False, "MXL 容器没有 rootfile"
            score_name = str(rootfile.get("full-path"))
            if score_name not in names:
                return False, f"MXL 缺少 {score_name}"
            score_root = etree.fromstring(archive.read(score_name))
            if score_root.tag != "score-partwise":
                return False, f"MXL 乐谱根元素为 {score_root.tag}"
            bad = archive.testzip()
            if bad:
                return False, f"ZIP 校验失败：{bad}"
    except Exception as exc:
        return False, f"MXL 无法读取：{exc}"
    return True, "score-partwise"


def _validate_artifact(role: str, path: Path) -> tuple[bool, str]:
    suffix = path.suffix.casefold()
    if role == "musicxml" or suffix in {".musicxml", ".xml"}:
        return _validate_xml(path, "score-partwise")
    if role == "mxl" or suffix == ".mxl":
        return _validate_mxl(path)
    if suffix == ".json":
        return _validate_json(path)
    if suffix == ".svg":
        return _validate_xml(path, "{http://www.w3.org/2000/svg}svg")
    return True, "binary"


def _bundle_digest(records: Iterable[ArtifactRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: (item.role, item.relative_path)):
        digest.update(
            f"{record.role}\0{record.relative_path}\0{record.size}\0{record.sha256}\0{int(record.valid)}\n".encode()
        )
    return digest.hexdigest()


def build_bundle_integrity(
    output_dir: Path,
    artifacts: Iterable[tuple[str, Path | None]],
    manifest_name: str = "artifact_manifest.json",
) -> BundleIntegrity:
    records: list[ArtifactRecord] = []
    errors: list[str] = []
    for role, candidate in artifacts:
        if candidate is None:
            continue
        path = Path(candidate)
        if not path.exists() or not path.is_file():
            errors.append(f"{role} 文件不存在：{path.name}")
            records.append(ArtifactRecord(role, path.name, 0, "", False, "missing"))
            continue
        valid, details = _validate_artifact(role, path)
        try:
            relative = str(path.resolve().relative_to(output_dir.resolve()))
        except ValueError:
            relative = path.name
            valid = False
            details = "artifact outside output directory"
            errors.append(f"{role} 文件不在结果目录内：{path}")
        record = ArtifactRecord(
            role=role,
            relative_path=relative.replace("\\", "/"),
            size=path.stat().st_size,
            sha256=sha256_file(path),
            valid=valid,
            details=details,
        )
        records.append(record)
        if not valid:
            errors.append(f"{role} 校验失败：{details}")

    if not records:
        errors.append("结果包没有任何可验证产物")
    identities = [(record.role, record.relative_path) for record in records]
    if len(identities) != len(set(identities)):
        errors.append("完整性清单包含重复产物")
    bundle_id = _bundle_digest(records)
    manifest_path = output_dir / manifest_name
    payload = {
        "format": 1,
        "created_at": utc_now_iso(),
        "valid": not errors and all(record.valid for record in records),
        "bundle_id": bundle_id,
        "records": [record.to_dict() for record in records],
        "errors": errors,
    }
    atomic_write_json(manifest_path, payload)
    return BundleIntegrity(payload["valid"], bundle_id, tuple(records), tuple(errors), str(manifest_path))


def verify_bundle_manifest(output_dir: Path, manifest_path: Path) -> tuple[bool, list[str]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"完整性清单无法读取：{exc}"]
    errors: list[str] = []
    records: list[ArtifactRecord] = []
    seen: set[tuple[str, str]] = set()
    if int(payload.get("format", 0) or 0) != 1:
        errors.append("不支持的完整性清单格式")
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list) or not raw_records:
        errors.append("完整性清单没有产物记录")
        raw_records = []
    for item in raw_records:
        if not isinstance(item, dict):
            errors.append("完整性清单含无效记录")
            continue
        role = str(item.get("role", ""))
        relative_path = str(item.get("relative_path", ""))
        identity = (role, relative_path)
        if identity in seen:
            errors.append(f"清单重复记录：{role} {relative_path}")
            continue
        seen.add(identity)
        path = (output_dir / relative_path).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError:
            errors.append(f"清单路径越界：{relative_path}")
            continue
        if not path.exists() or not path.is_file():
            errors.append(f"缺少 {role}: {path.name}")
            continue
        expected_size = int(item.get("size", -1))
        expected_hash = str(item.get("sha256", ""))
        expected_valid = bool(item.get("valid", False))
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        valid, details = _validate_artifact(role, path)
        if actual_size != expected_size:
            errors.append(f"文件大小变化：{path.name}")
        if actual_hash != expected_hash:
            errors.append(f"哈希变化：{path.name}")
        if not valid:
            errors.append(f"{path.name} 结构校验失败：{details}")
        if valid != expected_valid:
            errors.append(f"结构状态变化：{path.name}")
        records.append(ArtifactRecord(role, relative_path, actual_size, actual_hash, valid, details))
    calculated_bundle_id = _bundle_digest(records)
    if calculated_bundle_id != str(payload.get("bundle_id", "")):
        errors.append("结果包标识与当前文件不一致")
    if not bool(payload.get("valid", False)):
        errors.append("完整性清单本身标记为无效")
    return not errors, errors
