from __future__ import annotations

from pathlib import Path

from scorescan.workflow_checkpoint import RecognitionCheckpoint, RecognitionCheckpointKey


def test_checkpoint_is_bound_to_image_layout_and_workflow(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    layout = tmp_path / "layout.json"
    xml = tmp_path / "selected.musicxml"
    image.write_bytes(b"image-v1")
    layout.write_text('{"systems": 4}', encoding="utf-8")
    xml.write_text("<score-partwise>" + (" " * 400) + "</score-partwise>", encoding="utf-8")

    store = RecognitionCheckpoint(xml)
    key = RecognitionCheckpointKey.for_page(image, layout)
    store.commit(key, selected_variant="primary", consensus_applied=True)
    assert store.is_valid(key)

    layout.write_text('{"systems": 5}', encoding="utf-8")
    changed = RecognitionCheckpointKey.for_page(image, layout)
    assert not store.is_valid(changed)


def test_checkpoint_invalidation_quarantines_stale_xml(tmp_path: Path) -> None:
    xml = tmp_path / "selected.musicxml"
    xml.write_text("<score-partwise>" + (" " * 400) + "</score-partwise>", encoding="utf-8")
    store = RecognitionCheckpoint(xml)
    store.invalidate("test")
    assert not xml.exists()
    assert (tmp_path / "selected.stale-test.musicxml").exists()
