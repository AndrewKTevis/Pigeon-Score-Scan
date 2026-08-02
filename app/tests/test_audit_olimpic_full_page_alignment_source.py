from __future__ import annotations

import base64
import json
from pathlib import Path

import yaml

from app.tools.audit_olimpic_full_page_alignment_source import (
    SOURCE_ARCHIVE_SHA256,
    audit,
)


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    scanned = root / "scanned"
    source = root / "source"
    samples = scanned / "samples"
    mappings = source / "corpus_to_imslp"
    pdfs = source / "imslp_pdfs"
    pages = source / "imslp_pngs" / "IMSLP1"
    systems = source / "imslp_systems"
    for path in (samples, mappings, pdfs, pages, systems):
        path.mkdir(parents=True, exist_ok=True)
    (scanned / "LICENSE").write_text(
        "Attribution-ShareAlike 4.0 International\n",
        encoding="utf-8",
    )
    (scanned / "README.md").write_text(
        "Fixture available under CC BY-SA.\n",
        encoding="utf-8",
    )
    (scanned / "samples.dev.txt").write_text(
        "samples/100/p1-s1\n",
        encoding="utf-8",
    )
    (scanned / "samples.test.txt").write_text(
        "samples/200/p1-s1\n",
        encoding="utf-8",
    )
    for score, page in ((100, 1), (200, 2)):
        sample = samples / str(score) / "p1-s1"
        sample.parent.mkdir()
        sample.with_suffix(".png").write_bytes(PNG)
        sample.with_suffix(".musicxml").write_text(
            "<score-partwise version=\"4.0\"/>",
            encoding="utf-8",
        )
        sample.with_suffix(".lmx").write_text("fixture\n", encoding="utf-8")
        (mappings / f"{score}.yaml").write_text(
            yaml.safe_dump(
                {
                    f"{score}/p1-s1": {
                        "imslpDocument": "#1",
                        "imslpPage": page,
                        "imslpSystem": 1,
                    }
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (pages / f"page-{page}.png").write_bytes(PNG)
    (pdfs / "IMSLP1-fixture.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (systems / "IMSLP1.yaml").write_text(
        yaml.safe_dump(
            {
                "pages": {
                    1: {
                        "width": 1,
                        "height": 1,
                        "image": "IMSLP1/page-1.png",
                        "systems": [{"boundingBox": {"left": 0}}],
                    },
                    2: {
                        "width": 1,
                        "height": 1,
                        "image": "IMSLP1/page-2.png",
                        "systems": [{"boundingBox": {"left": 0}}],
                    },
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    provenance = root / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "key": "olimpic_scanned_sources",
                        "archive_sha256": SOURCE_ARCHIVE_SHA256,
                        "license": "CC-BY-SA",
                        "provenance_url": "https://example.test/olimpic",
                        "downloaded_bytes": 123,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return scanned, source, provenance


def test_audit_proves_mapping_but_rejects_full_page_release(
    tmp_path: Path,
) -> None:
    scanned, source, provenance = _fixture(tmp_path)
    report = audit(
        scanned_root=scanned,
        source_root=source,
        provenance_path=provenance,
    )

    assert report["published_score_group_count"] == 2
    assert report["published_sample_count"] == 2
    assert report["extra_mapping_sample_count"] == 0
    assert report["source_pdf_count"] == 1
    assert report["full_page_png_count"] == 2
    assert report["mapped_full_page_count"] == 2
    assert report["mapped_system_count"] == 2
    assert report["published_score_group_overlap_count"] == 0
    assert report["published_dev_test_source_document_overlap_count"] == 1
    assert report["published_dev_test_source_page_overlap_count"] == 0
    assert report["source_document_disjoint_split_verified"] is False
    assert report["full_page_semantic_ground_truth_complete"] is False
    assert report["research_training_authorized"] is True
    assert (
        report["existing_published_split_unseen_source_evaluation_authorized"]
        is False
    )
    assert report["distributable_product_training_authorized"] is False
    assert report["final_release_evidence_authorized"] is False
