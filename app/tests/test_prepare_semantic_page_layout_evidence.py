from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.tools.evaluate_semantic_detector_holdout import (
    load_page_layout_evidence,
)
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION,
)
from app.tools.prepare_semantic_page_layout_evidence import (
    main,
    sha256_file,
)


class _FakeLayout:
    def to_dict(self) -> dict[str, object]:
        return {
            "width": 100,
            "height": 80,
            "confidence": 1.0,
            "systems": [
                {
                    "index": 1,
                    "line_y": [20, 22, 24, 26, 28],
                    "spacing": 2.0,
                }
            ],
        }


def test_layout_evidence_is_hash_bound_and_covers_unique_pages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    images = tmp_path / "images"
    prepared.mkdir()
    images.mkdir()
    Image.new("L", (100, 80), 255).save(images / "page.png")
    manifest_path = prepared / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "target_assignment_version": (
                    COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
                )
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "source_key": "work-a",
            "image": "page.png",
            "image_id": "page-a",
            "crop_xyxy": [offset, 0, offset + 50, 50],
            "objects": [],
        }
        for offset in (0, 25)
    ]
    split_path = prepared / "test.jsonl"
    split_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.tools.prepare_semantic_page_layout_evidence.analyze_layout",
        lambda _path: _FakeLayout(),
    )
    output = tmp_path / "layout-evidence.json"

    assert main(
        [
            "--prepared-dir",
            str(prepared),
            "--images-dir",
            str(images),
            "--output",
            str(output),
            "--split",
            "test",
            "--workers",
            "1",
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["page_count"] == 1
    assert payload["staff_count"] == 1
    layouts = load_page_layout_evidence(
        rows,
        output,
        images_dir=images,
        prepared_manifest_sha256=sha256_file(manifest_path),
        split_jsonl_sha256={"test": sha256_file(split_path)},
    )
    assert set(layouts) == {("work-a", "page.png", "page-a")}

    split_path.write_text(
        split_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    try:
        load_page_layout_evidence(
            rows,
            output,
            images_dir=images,
            prepared_manifest_sha256=sha256_file(manifest_path),
            split_jsonl_sha256={"test": sha256_file(split_path)},
        )
    except ValueError as exc:
        assert "stale or invalid" in str(exc)
    else:
        raise AssertionError("stale layout evidence was accepted")
