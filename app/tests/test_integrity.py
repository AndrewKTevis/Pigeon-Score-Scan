from pathlib import Path

from lxml import etree

from scorescan.integrity import build_bundle_integrity, verify_bundle_manifest
from scorescan.musicxml import MUSICXML_DOCTYPE, package_mxl
from scorescan.util import atomic_write_json


def write_score(path: Path) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, doctype=MUSICXML_DOCTYPE)


def test_bundle_integrity_detects_later_change(tmp_path: Path) -> None:
    musicxml = tmp_path / "score.musicxml"
    mxl = tmp_path / "score.mxl"
    report = tmp_path / "conversion_report.json"
    write_score(musicxml)
    package_mxl(musicxml, mxl)
    atomic_write_json(report, {"ok": True})
    bundle = build_bundle_integrity(tmp_path, [("musicxml", musicxml), ("mxl", mxl), ("report", report)])
    assert bundle.valid
    valid, errors = verify_bundle_manifest(tmp_path, Path(bundle.manifest_path))
    assert valid and not errors
    report.write_text('{"ok": false}', encoding="utf-8")
    valid, errors = verify_bundle_manifest(tmp_path, Path(bundle.manifest_path))
    assert not valid
    assert any("变化" in item for item in errors)


def test_empty_bundle_is_invalid(tmp_path: Path) -> None:
    bundle = build_bundle_integrity(tmp_path, [])
    assert not bundle.valid
    valid, errors = verify_bundle_manifest(tmp_path, Path(bundle.manifest_path))
    assert not valid
    assert errors
