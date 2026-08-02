from __future__ import annotations

"""Acquire strictly pinned NIFC scan/reference match candidates.

This tool proves source identity, parent/child membership, scan integrity,
reference licensing and frozen-boundary compatibility.  It deliberately does
not prove page-level semantic alignment and therefore can never authorize the
result for training, evaluation or release evidence.

No source image is automatically rotated, deskewed, resampled or contrast
adjusted.  Historical two-page spreads are split at the exact stored-image
midpoint, with the crop recorded in the report.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import urllib.request

from lxml import etree, html
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.tools.humdrum_boundary import (  # noqa: E402
    analyze_humdrum_boundary,
    reference_records,
)
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)


MANIFEST_ROLE = (
    "external_scan_reference_match_candidates_not_training_or_evaluation"
)
REPORT_ROLE = (
    "acquired_external_scan_reference_match_candidates_not_training_or_evaluation"
)
NIFC_BASE = "https://repozytorium.nifc.pl/islandora/object"
CC_BY_4_PATTERN = re.compile(
    r"\bcc\s+by\s+4(?:[.,]0)?\b|creativecommons\.org/licenses/by/4\.0",
    re.IGNORECASE,
)
PID_PATTERN = re.compile(r"^nifc:\d+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
USER_AGENT = "ScoreScan-source-audit/0.38"
MAXIMUM_METADATA_BYTES = 4 * 1024 * 1024
MAXIMUM_IMAGE_BYTES = 64 * 1024 * 1024


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _pid_url(pid: str, suffix: str = "") -> str:
    if PID_PATTERN.fullmatch(pid) is None:
        raise ValueError(f"invalid NIFC PID: {pid}")
    return f"{NIFC_BASE}/{pid}{suffix}"


def _request_bytes(
    url: str,
    *,
    timeout: float,
    maximum_bytes: int,
) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        declared = int(response.headers.get("Content-Length", "0") or 0)
        if declared > maximum_bytes:
            raise ValueError(f"remote asset exceeds byte limit: {url}")
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"remote asset exceeds byte limit: {url}")
    return payload


def _read_or_download(
    path: Path,
    url: str,
    *,
    timeout: float,
    maximum_bytes: int,
) -> bytes:
    if path.is_file():
        payload = path.read_bytes()
        if not payload or len(payload) > maximum_bytes:
            raise ValueError(f"invalid cached source asset: {path}")
        return payload
    payload = _request_bytes(
        url,
        timeout=timeout,
        maximum_bytes=maximum_bytes,
    )
    atomic_write_bytes(path, payload)
    return payload


def parse_mods(payload: bytes) -> dict[str, object]:
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        recover=False,
    )
    root = etree.fromstring(payload, parser=parser)
    if etree.QName(root).localname != "mods":
        raise ValueError("NIFC metadata is not MODS")

    def values(local_name: str) -> list[str]:
        return [
            re.sub(r"\s+", " ", "".join(node.itertext())).strip()
            for node in root.xpath(f".//*[local-name()='{local_name}']")
            if "".join(node.itertext()).strip()
        ]

    titles = values("title")
    access_conditions = values("accessCondition")
    identifiers: dict[str, str] = {}
    for node in root.xpath(".//*[local-name()='recordIdentifier']"):
        source = str(node.get("source", "")).strip()
        value = re.sub(r"\s+", " ", "".join(node.itertext())).strip()
        if source and value:
            identifiers[source] = value
    return {
        "titles": titles,
        "names": values("namePart"),
        "notes": values("note"),
        "access_conditions": access_conditions,
        "record_identifiers": identifiers,
        "cc_by_4_explicit": any(
            CC_BY_4_PATTERN.search(value) is not None
            for value in access_conditions
        ),
        "normalized_document_text": _normalized(
            " ".join(root.itertext())
        ),
    }


def parse_parent_children(
    payload: bytes,
) -> list[dict[str, str]]:
    document = html.fromstring(payload)
    children: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in document.xpath(
        "//dl[contains(concat(' ', normalize-space(@class), ' '), "
        "' islandora-object ')]//a[@href and @title]"
    ):
        href = str(anchor.get("href", ""))
        match = re.search(r"/islandora/object/nifc(?:%3A|:)(\d+)", href)
        if match is None:
            continue
        pid = f"nifc:{match.group(1)}"
        title = str(anchor.get("title", "")).strip()
        if pid in seen:
            continue
        seen.add(pid)
        children.append({"pid": pid, "title": title})
    return children


def parse_child_membership(payload: bytes) -> tuple[str, str]:
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        recover=False,
    )
    root = etree.fromstring(payload, parser=parser)
    descriptions = root.xpath(
        ".//*[local-name()='Description']/@*[local-name()='about']"
    )
    memberships = root.xpath(
        ".//*[local-name()='isMemberOfCollection']"
        "/@*[local-name()='resource']"
    )
    if len(descriptions) != 1 or len(memberships) != 1:
        raise ValueError("NIFC child membership is missing or ambiguous")
    child = str(descriptions[0]).removeprefix("info:fedora/")
    parent = str(memberships[0]).removeprefix("info:fedora/")
    if (
        PID_PATTERN.fullmatch(child) is None
        or PID_PATTERN.fullmatch(parent) is None
    ):
        raise ValueError("NIFC child membership uses an unexpected PID")
    return child, parent


def reference_page_profile(lines: list[str]) -> dict[str, object]:
    page_break_indexes = [
        index
        for index, line in enumerate(lines)
        if line == "!!LO:PB:g=original"
    ]
    if not page_break_indexes:
        raise ValueError("reference has no original-layout page breaks")
    after_last = lines[page_break_indexes[-1] + 1 :]
    material_after_last = [
        line
        for line in after_last
        if line
        and not line.startswith("!")
        and not all(
            field in {"*-", "*"} for field in line.split("\t")
        )
    ]
    terminal_trailing_break = bool(material_after_last) and all(
        all(field.startswith("==") for field in line.split("\t"))
        for line in material_after_last
    )
    page_count = len(page_break_indexes) + (
        0 if terminal_trailing_break else 1
    )
    return {
        "original_page_break_count": len(page_break_indexes),
        "terminal_trailing_page_break": terminal_trailing_break,
        "encoded_music_page_count": page_count,
    }


def reference_problem_profile(lines: list[str]) -> dict[str, object]:
    """Count explicit encoder problem annotations without interpreting them.

    These comments often describe engraving or representation limitations, but
    they are still disqualifying for unattended full-semantic ground truth.
    """

    page_index = 1
    per_page: dict[int, list[str]] = {}
    record_count = 0
    marker_count = 0
    for line in lines:
        if line == "!!LO:PB:g=original":
            page_index += 1
            continue
        folded = line.casefold()
        if ":problem" not in folded:
            continue
        record_count += 1
        marker_count += folded.count(":problem")
        per_page.setdefault(page_index, []).append(line)
    return {
        "problem_record_count": record_count,
        "problem_marker_count": marker_count,
        "affected_page_count": len(per_page),
        "problem_records_per_page": {
            str(index): len(records)
            for index, records in sorted(per_page.items())
        },
        "pages_without_problem_records": [],
    }


def inspect_image(payload: bytes) -> dict[str, object]:
    with Image.open(io.BytesIO(payload)) as source:
        source.verify()
    with Image.open(io.BytesIO(payload)) as source:
        width, height = source.size
        image_format = str(source.format or "")
        orientation = int(source.getexif().get(274, 1) or 1)
        mode = source.mode
    if (
        image_format not in {"JPEG", "PNG"}
        or width < 600
        or height < 600
        or orientation != 1
    ):
        raise ValueError(
            "scan image failed format, size or no-auto-rotation contract"
        )
    return {
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
        "exif_orientation": orientation,
    }


def _split_spread_pages(
    payload: bytes,
) -> list[tuple[bytes, tuple[int, int, int, int]]]:
    with Image.open(io.BytesIO(payload)) as source:
        width, height = source.size
        midpoint = width // 2
        boxes = (
            (0, 0, midpoint, height),
            (midpoint, 0, width, height),
        )
        pages: list[tuple[bytes, tuple[int, int, int, int]]] = []
        for box in boxes:
            destination = io.BytesIO()
            source.crop(box).save(destination, format="PNG")
            pages.append((destination.getvalue(), box))
    return pages


def _git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def validate_manifest(
    manifest: dict[str, object],
    *,
    manifest_path: Path,
    repository: Path,
) -> list[dict[str, Any]]:
    if manifest.get("role") != MANIFEST_ROLE:
        raise ValueError("unexpected NIFC candidate manifest role")
    for field in (
        "training_authorized",
        "evaluation_authorized",
        "release_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"NIFC candidate manifest must set {field}=false")
    if (
        manifest.get("boundary_contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
    ):
        raise ValueError("NIFC candidate boundary contract drifted")
    source = manifest.get("source_repository")
    if not isinstance(source, dict):
        raise ValueError("missing Humdrum source repository contract")
    revision = str(source.get("revision", ""))
    if _git_revision(repository) != revision:
        raise ValueError("Humdrum source repository revision drifted")
    license_path = repository / str(source.get("license_path", ""))
    if (
        not license_path.is_file()
        or sha256_file(license_path) != source.get("license_sha256")
        or CC_BY_4_PATTERN.search(
            license_path.read_text(encoding="utf-8")
        )
        is None
    ):
        raise ValueError("Humdrum source repository license drifted")
    works = manifest.get("works")
    if not isinstance(works, list) or not works:
        raise ValueError("NIFC candidate manifest has no works")
    work_ids: set[str] = set()
    parent_pids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for value in works:
        if not isinstance(value, dict):
            raise ValueError("invalid NIFC candidate work")
        work: dict[str, Any] = value
        work_id = str(work.get("id", ""))
        parent_pid = str(work.get("parent_pid", ""))
        child_pids = work.get("child_pids")
        reference = repository / str(work.get("reference_path", ""))
        if (
            not work_id
            or work_id in work_ids
            or PID_PATTERN.fullmatch(parent_pid) is None
            or parent_pid in parent_pids
            or not isinstance(child_pids, list)
            or not child_pids
            or len(set(map(str, child_pids))) != len(child_pids)
            or any(PID_PATTERN.fullmatch(str(pid)) is None for pid in child_pids)
            or not reference.is_file()
            or SHA256_PATTERN.fullmatch(
                str(work.get("reference_sha256", ""))
            )
            is None
            or sha256_file(reference) != work.get("reference_sha256")
        ):
            raise ValueError(f"invalid or drifted NIFC candidate: {work_id}")
        lines = reference.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        records = reference_records(lines)
        required_records = work.get("required_reference_records")
        if not isinstance(required_records, dict) or any(
            records.get(str(key)) != str(expected)
            for key, expected in required_records.items()
        ):
            raise ValueError(f"reference metadata drifted: {work_id}")
        boundary = analyze_humdrum_boundary(
            reference,
            instrumentation=str(work.get("reference_instrumentation", "")),
            source_lines=lines,
        )
        profile = reference_page_profile(lines)
        problem_profile = reference_problem_profile(lines)
        problem_profile["pages_without_problem_records"] = [
            index
            for index in range(
                1,
                int(profile["encoded_music_page_count"]) + 1,
            )
            if str(index)
            not in problem_profile["problem_records_per_page"]
        ]
        if (
            not boundary.accepted
            or boundary.score_shape != "keyboard"
            or profile["encoded_music_page_count"]
            != work.get("expected_music_page_count")
        ):
            raise ValueError(f"reference boundary or page count failed: {work_id}")
        work_ids.add(work_id)
        parent_pids.add(parent_pid)
        work["_reference"] = reference
        work["_boundary"] = boundary.to_dict()
        work["_reference_profile"] = profile
        work["_reference_problem_profile"] = problem_profile
        validated.append(work)
    return validated


def _acquire_work(
    work: dict[str, Any],
    output_dir: Path,
    *,
    timeout: float,
    workers: int,
) -> dict[str, object]:
    work_id = str(work["id"])
    parent_pid = str(work["parent_pid"])
    source_dir = output_dir / "sources" / work_id
    page_dir = output_dir / "pages" / work_id
    source_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)

    parent_html_url = str(work["parent_page_url"])
    parent_mods_url = _pid_url(parent_pid, "/datastream/MODS/view")
    parent_html_path = source_dir / "parent.html"
    parent_mods_path = source_dir / "parent.mods.xml"
    parent_html = _read_or_download(
        parent_html_path,
        parent_html_url,
        timeout=timeout,
        maximum_bytes=MAXIMUM_METADATA_BYTES,
    )
    parent_mods = _read_or_download(
        parent_mods_path,
        parent_mods_url,
        timeout=timeout,
        maximum_bytes=MAXIMUM_METADATA_BYTES,
    )
    metadata = parse_mods(parent_mods)
    expected_title = _normalized(str(work["expected_parent_title"]))
    if (
        expected_title
        not in {_normalized(value) for value in metadata["titles"]}
        or metadata["cc_by_4_explicit"] is not True
        or any(
            _normalized(str(fragment))
            not in str(metadata["normalized_document_text"])
            for fragment in work["required_parent_mods_fragments"]
        )
    ):
        raise ValueError(f"parent MODS identity or rights failed: {work_id}")

    listed_children = parse_parent_children(parent_html)
    expected_pids = [str(pid) for pid in work["child_pids"]]
    expected_prefix = str(work["child_title_prefix"])
    if (
        [child["pid"] for child in listed_children] != expected_pids
        or any(
            child["title"] != f"{expected_prefix}{index:03d}"
            for index, child in enumerate(listed_children, start=1)
        )
    ):
        raise ValueError(f"parent child sequence drifted: {work_id}")

    def acquire_child(pid: str) -> dict[str, object]:
        stem = pid.replace(":", "-")
        urls = {
            "mods": _pid_url(pid, "/datastream/MODS/view"),
            "rels": _pid_url(pid, "/datastream/RELS-EXT/view"),
            "image": _pid_url(pid, "/datastream/OBJ/view"),
        }
        paths = {
            "mods": source_dir / f"{stem}.mods.xml",
            "rels": source_dir / f"{stem}.rels-ext.xml",
            "image": source_dir / f"{stem}.obj",
        }
        payloads = {
            key: _read_or_download(
                paths[key],
                url,
                timeout=timeout,
                maximum_bytes=(
                    MAXIMUM_IMAGE_BYTES
                    if key == "image"
                    else MAXIMUM_METADATA_BYTES
                ),
            )
            for key, url in urls.items()
        }
        child_pid, child_parent = parse_child_membership(payloads["rels"])
        child_mods = parse_mods(payloads["mods"])
        if (
            child_pid != pid
            or child_parent != parent_pid
            or expected_title
            not in {
                _normalized(value) for value in child_mods["titles"]
            }
        ):
            raise ValueError(f"child identity or membership failed: {pid}")
        image = inspect_image(payloads["image"])
        return {
            "pid": pid,
            "urls": urls,
            "paths": paths,
            "payloads": payloads,
            "image": image,
            "mods_sha256": hashlib.sha256(payloads["mods"]).hexdigest(),
            "rels_sha256": hashlib.sha256(payloads["rels"]).hexdigest(),
            "image_sha256": hashlib.sha256(payloads["image"]).hexdigest(),
        }

    acquired_by_pid: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(acquire_child, pid): pid for pid in expected_pids
        }
        for future in as_completed(futures):
            result = future.result()
            acquired_by_pid[str(result["pid"])] = result
    acquired = [acquired_by_pid[pid] for pid in expected_pids]

    music_children = acquired[int(work["non_music_leading_children"]) :]
    layout = str(work["scan_layout"])
    pages: list[dict[str, object]] = []
    for child in music_children:
        payload = child["payloads"]["image"]
        if layout == "left_to_right_two_page_spread_fixed_midpoint":
            derived = _split_spread_pages(payload)
        elif layout == "single_page_no_geometric_transform":
            width = int(child["image"]["width"])
            height = int(child["image"]["height"])
            derived = [(payload, (0, 0, width, height))]
        else:
            raise ValueError(f"unsupported pinned scan layout: {layout}")
        for page_payload, crop in derived:
            page_index = len(pages) + 1
            suffix = ".png" if layout.startswith("left_to_right") else ".jpg"
            page_path = page_dir / f"page-{page_index:03d}{suffix}"
            atomic_write_bytes(page_path, page_payload)
            page_image = inspect_image(page_payload)
            pages.append(
                {
                    "reference_page_index": page_index,
                    "printed_page_number": (
                        int(work["first_printed_music_page"])
                        + page_index
                        - 1
                    ),
                    "source_child_pid": child["pid"],
                    "source_image_sha256": child["image_sha256"],
                    "source_crop_xyxy": list(crop),
                    "geometric_transform": (
                        "fixed_midpoint_crop_only"
                        if layout.startswith("left_to_right")
                        else "none"
                    ),
                    "auto_rotation_applied": False,
                    "auto_deskew_applied": False,
                    "resampling_applied": False,
                    "path": str(page_path.resolve()),
                    "sha256": sha256_file(page_path),
                    "image": page_image,
                    "alignment_status": "not_verified",
                }
            )
    if len(pages) != int(work["expected_music_page_count"]):
        raise ValueError(f"derived music page count failed: {work_id}")

    children_report = [
        {
            "pid": child["pid"],
            "urls": child["urls"],
            "mods_path": str(child["paths"]["mods"].resolve()),
            "mods_sha256": child["mods_sha256"],
            "rels_ext_path": str(child["paths"]["rels"].resolve()),
            "rels_ext_sha256": child["rels_sha256"],
            "image_path": str(child["paths"]["image"].resolve()),
            "image_sha256": child["image_sha256"],
            "image": child["image"],
            "membership_verified": True,
        }
        for child in acquired
    ]
    source_group_fingerprint = hashlib.sha256(
        f"{parent_pid}\0{sha256_file(parent_mods_path)}".encode("utf-8")
    ).hexdigest()
    return {
        "id": work_id,
        "parent_pid": parent_pid,
        "parent_page_url": parent_html_url,
        "parent_html_path": str(parent_html_path.resolve()),
        "parent_html_sha256": sha256_file(parent_html_path),
        "parent_mods_url": parent_mods_url,
        "parent_mods_path": str(parent_mods_path.resolve()),
        "parent_mods_sha256": sha256_file(parent_mods_path),
        "parent_cc_by_4_explicit": True,
        "child_membership_verified": True,
        "source_group_fingerprint": source_group_fingerprint,
        "reference_path": str(work["_reference"].resolve()),
        "reference_sha256": sha256_file(work["_reference"]),
        "reference_license": "CC-BY-4.0",
        "boundary": work["_boundary"],
        "reference_page_profile": work["_reference_profile"],
        "reference_problem_profile": work["_reference_problem_profile"],
        "scan_layout": layout,
        "music_page_count": len(pages),
        "children": children_report,
        "pages": pages,
        "page_alignment_verified": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
    }


def acquire(
    manifest_path: Path,
    repository: Path,
    output_dir: Path,
    *,
    timeout: float = 120.0,
    workers: int = 6,
) -> dict[str, object]:
    if timeout <= 0 or workers <= 0:
        raise ValueError("timeout and workers must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    works = validate_manifest(
        manifest,
        manifest_path=manifest_path,
        repository=repository,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        _acquire_work(
            work,
            output_dir,
            timeout=timeout,
            workers=workers,
        )
        for work in works
    ]
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "role": REPORT_ROLE,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_repository_path": str(repository.resolve()),
        "source_repository_revision": _git_revision(repository),
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "all_page_scan_to_reference_alignment_not_verified",
            "references_contain_explicit_problem_annotations",
            "reference_semantic_completeness_not_audited",
            "independent_double_annotation_not_started",
            "work_disjoint_production_split_not_assigned",
        ],
        "work_count": len(reports),
        "physical_source_group_count": len(
            {str(work["source_group_fingerprint"]) for work in reports}
        ),
        "music_page_count": sum(
            int(work["music_page_count"]) for work in reports
        ),
        "parent_cc_by_4_verified_count": sum(
            work["parent_cc_by_4_explicit"] is True for work in reports
        ),
        "child_membership_verified_count": sum(
            len(work["children"])
            for work in reports
            if work["child_membership_verified"] is True
        ),
        "reference_problem_record_count": sum(
            int(
                work["reference_problem_profile"][
                    "problem_record_count"
                ]
            )
            for work in reports
        ),
        "reference_problem_free_page_count": sum(
            len(
                work["reference_problem_profile"][
                    "pages_without_problem_records"
                ]
            )
            for work in reports
        ),
        "automatic_geometric_policy": {
            "rotation": "forbidden",
            "deskew": "forbidden",
            "resampling": "forbidden",
            "two_page_spread_split": "fixed_stored_image_midpoint_only",
        },
        "works": reports,
    }
    atomic_write_json(output_dir / "candidate_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    report = acquire(
        args.manifest_path.resolve(),
        args.repository.resolve(),
        args.output_dir.resolve(),
        timeout=args.timeout_seconds,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "work_count",
                    "physical_source_group_count",
                    "music_page_count",
                    "parent_cc_by_4_verified_count",
                    "child_membership_verified_count",
                    "training_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
