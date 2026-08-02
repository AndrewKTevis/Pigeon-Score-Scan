from __future__ import annotations

"""Acquire a pinned real-scan diagnostic set that is never a release gate.

The upstream derivative is explicitly evaluation-only and declares its license
as ``other``.  This tool therefore emits a diagnostic-only role that the release
manifest merger rejects.
"""

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorescan.util import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    analyze_reference_boundary,
)


DATASET_ID = "btrkeks/polish-scores"
DATASET_REVISION = "55428225335e21c0e4d77a58b32dbb2b58e25954"
DATASET_ROWS = 112
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DIAGNOSTIC_ROLE = (
    "external_real_scan_diagnostic_only_license_not_release_authorized"
)
USER_AGENT = "ScoreScan-dataset-audit/0.37"
MAXIMUM_RESPONSE_BYTES = 64 * 1024 * 1024
MAXIMUM_IMAGE_BYTES = 16 * 1024 * 1024


def _request_bytes(url: str, *, timeout: float, maximum_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        declared = int(response.headers.get("Content-Length", "0") or 0)
        if declared > maximum_bytes:
            raise ValueError(f"remote asset exceeds byte limit: {url}")
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"remote asset exceeds byte limit: {url}")
    return payload


def _normalized_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def musicxml_work_fingerprint(payload: bytes) -> str:
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        recover=False,
    )
    root = etree.fromstring(payload, parser=parser)
    fields = [
        root.findtext("./work/work-number"),
        root.findtext("./work/work-title"),
        root.findtext("./movement-number"),
        root.findtext("./movement-title"),
        *[
            f"{creator.get('type', '')}:{creator.text or ''}"
            for creator in root.findall("./identification/creator")
        ],
    ]
    normalized = [_normalized_text(value) for value in fields]
    identity = "\0".join(value for value in normalized if value)
    if not identity:
        # Missing metadata cannot prove that page rows are independent works.
        # Group them conservatively instead of inventing source diversity.
        identity = "unknown-work-metadata"
    return hashlib.sha256(
        f"{DATASET_ID}\0{identity}".encode("utf-8")
    ).hexdigest()


def _rows_url(*, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    return f"{ROWS_ENDPOINT}?{query}"


def _load_rows(*, timeout: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(0, DATASET_ROWS, 100):
        payload = json.loads(
            _request_bytes(
                _rows_url(
                    offset=offset,
                    length=min(100, DATASET_ROWS - offset),
                ),
                timeout=timeout,
                maximum_bytes=MAXIMUM_RESPONSE_BYTES,
            )
        )
        if (
            not isinstance(payload, dict)
            or int(payload.get("num_rows_total", -1)) != DATASET_ROWS
            or payload.get("partial") is not False
            or not isinstance(payload.get("rows"), list)
        ):
            raise ValueError("Polish scores rows API contract changed")
        rows.extend(payload["rows"])
    if (
        len(rows) != DATASET_ROWS
        or [int(row.get("row_idx", -1)) for row in rows]
        != list(range(DATASET_ROWS))
    ):
        raise ValueError("Polish scores row coverage is incomplete")
    return rows


def _download_image(
    row_index: int,
    image: dict[str, Any],
    destination: Path,
    *,
    timeout: float,
) -> dict[str, object]:
    url = str(image.get("src", ""))
    if f"/{DATASET_REVISION}/" not in url:
        raise ValueError("Polish scores image is not pinned to the revision")
    payload = _request_bytes(
        url,
        timeout=timeout,
        maximum_bytes=MAXIMUM_IMAGE_BYTES,
    )
    with Image.open(io.BytesIO(payload)) as source:
        source.verify()
    with Image.open(io.BytesIO(payload)) as source:
        width, height = source.size
        image_format = str(source.format or "")
    if (
        width != int(image.get("width", -1))
        or height != int(image.get("height", -1))
        or width < 500
        or height < 500
        or image_format not in {"JPEG", "PNG"}
    ):
        raise ValueError(f"invalid Polish scores image row {row_index}")
    atomic_write_bytes(destination, payload)
    return {
        "row_index": row_index,
        "image_sha256": sha256_file(destination),
        "width": width,
        "height": height,
        "bytes": len(payload),
    }


def acquire(
    output_dir: Path,
    *,
    workers: int = 6,
    timeout: float = 120.0,
) -> dict[str, object]:
    if workers <= 0 or timeout <= 0:
        raise ValueError("workers and timeout must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    references_dir = output_dir / "references"
    images_dir.mkdir(parents=True, exist_ok=True)
    references_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(timeout=timeout)

    cases: list[dict[str, object]] = []
    downloads: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for raw in rows:
            row_index = int(raw["row_idx"])
            row = raw.get("row")
            if not isinstance(row, dict) or not isinstance(row.get("image"), dict):
                raise ValueError(f"invalid Polish scores row {row_index}")
            xml = row.get("transcription_xml")
            if not isinstance(xml, str) or "<score-partwise" not in xml:
                raise ValueError(f"missing MusicXML row {row_index}")
            reference = references_dir / f"row-{row_index:03d}.musicxml"
            atomic_write_bytes(reference, xml.encode("utf-8"))
            image_path = images_dir / f"row-{row_index:03d}.jpg"
            futures[
                executor.submit(
                    _download_image,
                    row_index,
                    row["image"],
                    image_path,
                    timeout=timeout,
                )
            ] = (row_index, row, reference, image_path)
        for completed, future in enumerate(as_completed(futures), start=1):
            row_index, row, reference, image_path = futures[future]
            downloads[row_index] = future.result()
            boundary = analyze_reference_boundary(reference)
            cases.append(
                {
                    "id": f"polish-scores-{row_index:03d}",
                    "row_index": row_index,
                    "variant_key": f"polish-scores/{row_index}",
                    "work_fingerprint": musicxml_work_fingerprint(
                        reference.read_bytes()
                    ),
                    "role": "external_diagnostic_only",
                    "input_pdf": str(image_path.resolve()),
                    "input_pdf_pages": 1,
                    "input_pdf_sha256": downloads[row_index][
                        "image_sha256"
                    ],
                    "reference": str(reference.relative_to(output_dir)),
                    "reference_sha256": sha256_file(reference),
                    "boundary": boundary,
                }
            )
            print(
                f"[{completed}/{DATASET_ROWS}] row {row_index}: "
                f"{'accepted' if boundary['accepted'] else 'rejected'} "
                f"{boundary['score_shape']}",
                flush=True,
            )
    cases.sort(key=lambda case: int(case["row_index"]))
    accepted = [case for case in cases if case["boundary"]["accepted"]]
    report = {
        "format": 1,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "created_at": utc_now_iso(),
        "name": "Polish historical real-scan diagnostic benchmark",
        "role": DIAGNOSTIC_ROLE,
        "release_authorized": False,
        "license": "other_unspecified_by_upstream_evaluation_only",
        "repository": DATASET_ID,
        "revision": DATASET_REVISION,
        "rows_endpoint": ROWS_ENDPOINT,
        "case_count": len(cases),
        "work_count": len(
            {str(case["work_fingerprint"]) for case in cases}
        ),
        "accepted_case_count": len(accepted),
        "accepted_work_count": len(
            {str(case["work_fingerprint"]) for case in accepted}
        ),
        "rejected_case_count": len(cases) - len(accepted),
        "downloaded_bytes": sum(
            int(row["bytes"]) for row in downloads.values()
        ),
        "cases": cases,
    }
    atomic_write_json(output_dir / "boundary_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    report = acquire(
        args.output_dir.resolve(),
        workers=args.workers,
        timeout=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "work_count": report["work_count"],
                "accepted_case_count": report["accepted_case_count"],
                "accepted_work_count": report["accepted_work_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
