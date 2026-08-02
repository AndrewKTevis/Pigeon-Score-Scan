from __future__ import annotations

"""ScoreScan-owned homr entrypoint for isolated CPU inference."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

from .util import atomic_write_bytes, atomic_write_json, read_json


RAW_TUPLET_FLAG = "--scorescan-preserve-raw-tuplets"


def _install_raw_tuplet_preservation() -> None:
    """Disable only homr's page-median tuplet deletion heuristic.

    The normal lower-staff and duplicate-symbol cleanup remains active.  This is
    used by one correlated internal candidate; ordinary output remains the
    default and later candidate validation decides which sibling survives.
    """

    from homr.transformer import vocabulary

    cleanup = getattr(vocabulary, "_fix_over_eager_tuplets", None)
    if not callable(cleanup):
        raise RuntimeError("installed homr lacks the pinned tuplet cleanup hook")
    vocabulary._fix_over_eager_tuplets = lambda chords: chords


def _argument_value(name: str, default: str | None = None) -> str | None:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else default


def _run_ocr_enrichment(request_path: Path, requested: str) -> int:
    response_path = request_path.with_suffix(".result.json")
    try:
        payload = read_json(request_path)
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise RuntimeError("OCR worker request schema is invalid")
        image_path = Path(str(payload["image_path"])).resolve()
        xml_path = Path(str(payload["xml_path"])).resolve()
        layout_payload = payload.get("layout")
        semantic_text_regions = payload.get("semantic_text_regions")
        if not image_path.is_file() or not xml_path.is_file():
            raise RuntimeError("OCR worker input image or MusicXML is missing")
        if not isinstance(layout_payload, dict):
            raise RuntimeError("OCR worker layout is missing")
        if semantic_text_regions is not None and not isinstance(
            semantic_text_regions,
            list,
        ):
            raise RuntimeError("OCR worker semantic text regions are invalid")

        requested = requested.strip().casefold()
        if requested != "cpu":
            raise RuntimeError(f"unsupported OCR accelerator: {requested}")
        os.environ["SCORESCAN_OCR_ACCELERATOR"] = requested

        # Enrichment is transactional: a native OCR failure must not leave a
        # partially edited MusicXML that a CPU retry could duplicate.
        with tempfile.TemporaryDirectory(prefix="scorescan-ocr-worker-", dir=xml_path.parent) as temp_dir:
            working_xml = Path(temp_dir) / xml_path.name
            shutil.copy2(xml_path, working_xml)
            from .layout import PageLayout
            from .text_enrichment import (
                enrich_musicxml_with_ocr,
                marks_to_dicts,
                ocr_engine_runtime,
            )

            marks, warnings = enrich_musicxml_with_ocr(
                image_path,
                working_xml,
                PageLayout.from_dict(layout_payload),
                semantic_text_regions=(
                    tuple(
                        dict(item)
                        for item in semantic_text_regions
                        if isinstance(item, dict)
                    )
                    if isinstance(semantic_text_regions, list)
                    else None
                ),
            )
            runtime = ocr_engine_runtime()
            if runtime.get("selected") != "cpu":
                raise RuntimeError("RapidOCR did not bind to CPUExecutionProvider")
            atomic_write_bytes(xml_path, working_xml.read_bytes())

        atomic_write_json(
            response_path,
            {
                "schema_version": 1,
                "ok": True,
                "marks": marks_to_dicts(marks),
                "warnings": [str(value) for value in warnings],
                "runtime": runtime,
            },
        )
        return 0
    except Exception as exc:
        atomic_write_json(
            response_path,
            {
                "schema_version": 1,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "requested": requested,
            },
        )
        print(f"ScoreScan OCR worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 70


def _run_semantic_detection(request_path: Path, requested: str) -> int:
    response_path = request_path.with_suffix(".result.json")
    try:
        payload = read_json(request_path)
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise RuntimeError("semantic detector worker request schema is invalid")
        image_path = Path(str(payload["image_path"])).resolve()
        layout_payload = payload.get("layout")
        if not image_path.is_file() or not isinstance(layout_payload, dict):
            raise RuntimeError("semantic detector worker input image or layout is missing")
        requested = requested.strip().casefold()
        if requested != "cpu":
            raise RuntimeError(f"unsupported semantic detector accelerator: {requested}")

        from .layout import PageLayout
        from .semantic_detector import run_semantic_detector

        result = run_semantic_detector(
            image_path,
            PageLayout.from_dict(layout_payload),
            Path(__file__).resolve().parent / "resources",
            requested,
        )
        atomic_write_json(
            response_path,
            {
                "schema_version": 1,
                "ok": True,
                "result": result.to_dict(),
            },
        )
        return 0
    except Exception as exc:
        atomic_write_json(
            response_path,
            {
                "schema_version": 1,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "requested": requested,
            },
        )
        print(
            f"ScoreScan semantic detector worker failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 71


def main() -> None:
    preserve_raw_tuplets = RAW_TUPLET_FLAG in sys.argv
    if preserve_raw_tuplets:
        sys.argv = [value for value in sys.argv if value != RAW_TUPLET_FLAG]
    request_value = _argument_value("--scorescan-ocr-request")
    if request_value is not None:
        requested = _argument_value("--scorescan-ocr-accelerator", "cpu") or "cpu"
        raise SystemExit(_run_ocr_enrichment(Path(request_value).resolve(), requested))
    semantic_request = _argument_value("--scorescan-semantic-request")
    if semantic_request is not None:
        requested = (
            _argument_value("--scorescan-semantic-accelerator", "cpu") or "cpu"
        )
        raise SystemExit(
            _run_semantic_detection(Path(semantic_request).resolve(), requested)
        )
    if preserve_raw_tuplets:
        _install_raw_tuplet_preservation()
    from homr.main import main as homr_main

    homr_main()


if __name__ == "__main__":
    main()
