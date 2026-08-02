from pathlib import Path

import pytest
from lxml import etree

from scorescan.engine_cache import EngineCacheKey, EngineResultCache, valid_musicxml_structure
from scorescan.util import sha256_file


def _write_musicxml(path: Path) -> None:
    root = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(root, "part-list")
    score_part = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(score_part, "part-name").text = "Music"
    part = etree.SubElement(root, "part", id="P1")
    measure = etree.SubElement(part, "measure", number="1")
    attributes = etree.SubElement(measure, "attributes")
    etree.SubElement(attributes, "divisions").text = "1"
    note = etree.SubElement(measure, "note")
    pitch = etree.SubElement(note, "pitch")
    etree.SubElement(pitch, "step").text = "C"
    etree.SubElement(pitch, "octave").text = "4"
    etree.SubElement(note, "duration").text = "4"
    etree.SubElement(note, "voice").text = "1"
    etree.SubElement(note, "type").text = "whole"
    etree.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True, pretty_print=True)


def test_engine_cache_is_content_addressed_and_structurally_validated(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    xml = tmp_path / "page.musicxml"
    image.write_bytes(b"image-a")
    _write_musicxml(xml)
    cache = EngineResultCache(image, xml)
    key = EngineCacheKey(sha256_file(image), "0.7.0")
    cache.commit(key)
    assert cache.is_valid(key)
    assert valid_musicxml_structure(xml)

    image.write_bytes(b"image-b")
    assert not cache.is_valid(key)
    image.write_bytes(b"image-a")
    assert cache.is_valid(key)
    xml.write_bytes(xml.read_bytes() + b"tamper")
    assert not cache.is_valid(key)


def test_engine_cache_rejects_hashable_but_invalid_xml(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    xml = tmp_path / "page.musicxml"
    image.write_bytes(b"image")
    xml.write_bytes(b"<score-partwise>" + b"x" * 400 + b"</score-partwise>")
    cache = EngineResultCache(image, xml)
    key = EngineCacheKey(sha256_file(image), "0.7.0")
    with pytest.raises(ValueError, match="score-partwise"):
        cache.commit(key)
    assert not cache.is_valid(key)


def test_engine_cache_rejects_entity_or_wrong_root(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.musicxml"
    wrong.write_text("<score-timewise><part-list/></score-timewise>", encoding="utf-8")
    assert not valid_musicxml_structure(wrong)

    entity = tmp_path / "entity.musicxml"
    entity.write_text(
        '<!DOCTYPE score-partwise [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        '<score-partwise><part-list/><part id="P1"><measure number="1">&xxe;</measure></part></score-partwise>',
        encoding="utf-8",
    )
    # Entity resolution is disabled and internal entity declarations are rejected.
    assert not valid_musicxml_structure(entity)

    undeclared = tmp_path / "undeclared.musicxml"
    undeclared.write_text(
        '<score-partwise><part-list><score-part id="P1"/></part-list>'
        '<part id="P2"><measure number="1"/></part></score-partwise>',
        encoding="utf-8",
    )
    assert not valid_musicxml_structure(undeclared)
