from __future__ import annotations

"""Content-addressed cache metadata for external OMR engine results."""

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from .util import atomic_write_json, read_json, sha256_file

CACHE_FORMAT = 2
MAX_CACHED_MUSICXML_BYTES = 64 * 1024 * 1024
ENGINE_PROFILE = "homr-large-page-v1"


def valid_musicxml_structure(path: Path) -> bool:
    """Validate cached engine output without resolving entities or network resources."""
    try:
        if path.stat().st_size > MAX_CACHED_MUSICXML_BYTES:
            return False
        parser = etree.XMLParser(
            resolve_entities=False,
            no_network=True,
            recover=False,
            huge_tree=False,
            remove_blank_text=False,
        )
        tree = etree.parse(str(path), parser)
    except (OSError, etree.XMLSyntaxError, ValueError):
        return False
    internal_dtd = tree.docinfo.internalDTD
    if internal_dtd is not None:
        try:
            if any(True for _ in internal_dtd.iterentities()):
                return False
        except (etree.DTDParseError, ValueError):
            return False
    root = tree.getroot()
    if etree.QName(root).localname != "score-partwise":
        return False
    part_list = next((child for child in root if etree.QName(child).localname == "part-list"), None)
    parts = [child for child in root if etree.QName(child).localname == "part"]
    if part_list is None or not parts:
        return False
    listed_ids = {
        child.get("id")
        for child in part_list
        if etree.QName(child).localname == "score-part" and child.get("id")
    }
    if not listed_ids:
        return False
    return all(
        part.get("id") in listed_ids
        and any(etree.QName(child).localname == "measure" for child in part)
        for part in parts
    )


def homr_version() -> str:
    try:
        return importlib.metadata.version("homr")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True)
class EngineCacheKey:
    image_sha256: str
    engine_version: str
    profile: str = ENGINE_PROFILE

    @classmethod
    def for_image(
        cls,
        image_path: Path,
        *,
        profile: str = ENGINE_PROFILE,
    ) -> "EngineCacheKey":
        return cls(sha256_file(image_path), homr_version(), profile)


class EngineResultCache:
    def __init__(self, image_path: Path, xml_path: Path) -> None:
        self.image_path = image_path
        self.xml_path = xml_path
        self.manifest_path = xml_path.with_suffix(".omr-cache.json")

    def is_valid(self, key: EngineCacheKey) -> bool:
        try:
            xml_size = self.xml_path.stat().st_size
            if xml_size <= 300 or sha256_file(self.image_path) != key.image_sha256:
                return False
        except OSError:
            return False
        payload = read_json(self.manifest_path)
        if not isinstance(payload, dict):
            return False
        if int(payload.get("format", 0)) != CACHE_FORMAT:
            return False
        if payload.get("image_sha256") != key.image_sha256:
            return False
        if payload.get("engine_version") != key.engine_version:
            return False
        if payload.get("profile") != key.profile:
            return False
        if int(payload.get("xml_size", -1) or -1) != xml_size:
            return False
        expected_xml_hash = payload.get("xml_sha256")
        if not isinstance(expected_xml_hash, str):
            return False
        try:
            if expected_xml_hash != sha256_file(self.xml_path):
                return False
        except OSError:
            return False
        return valid_musicxml_structure(self.xml_path)

    def invalidate(self) -> None:
        self.manifest_path.unlink(missing_ok=True)
        if self.xml_path.exists():
            stale = self.xml_path.with_name(self.xml_path.stem + ".stale.musicxml")
            stale.unlink(missing_ok=True)
            try:
                self.xml_path.replace(stale)
            except OSError:
                self.xml_path.unlink(missing_ok=True)

    def commit(self, key: EngineCacheKey) -> None:
        if not valid_musicxml_structure(self.xml_path):
            raise ValueError("识别引擎输出不是有效的 score-partwise MusicXML")
        atomic_write_json(
            self.manifest_path,
            {
                "format": CACHE_FORMAT,
                "profile": key.profile,
                "engine": "homr",
                "engine_version": key.engine_version,
                "image_sha256": key.image_sha256,
                "xml_sha256": sha256_file(self.xml_path),
                "xml_size": self.xml_path.stat().st_size,
            },
        )
