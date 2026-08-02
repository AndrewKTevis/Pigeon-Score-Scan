from __future__ import annotations

"""Acquire pinned NIFC layout-audit candidates without authorizing reuse.

The output contains untouched repository images plus identity, rights and
parent/child evidence. It never turns metadata similarity into page alignment
or semantic ground truth.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.acquire_nifc_chopin_matched_scans import (  # noqa: E402
    MAXIMUM_IMAGE_BYTES,
    MAXIMUM_METADATA_BYTES,
    _pid_url,
    _read_or_download,
    inspect_image,
    parse_child_membership,
    parse_mods,
    parse_parent_children,
)
from app.tools.discover_nifc_chopin_scan_matches import (  # noqa: E402
    ROLE as DISCOVERY_ROLE,
)
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)


MANIFEST_ROLE = (
    "nifc_chopin_layout_audit_shortlist_not_training_or_evaluation"
)
REPORT_ROLE = (
    "acquired_nifc_chopin_layout_audit_shortlist_not_training_or_evaluation"
)
AUTHORIZATION_FIELDS = (
    "training_authorized",
    "evaluation_authorized",
    "release_authorized",
)


def _require_false_authorization(
    value: dict[str, object],
    *,
    label: str,
) -> None:
    for field in AUTHORIZATION_FIELDS:
        if value.get(field) is not False:
            raise ValueError(f"{label} unexpectedly sets {field}")


def validate_shortlist(
    manifest: dict[str, object],
    *,
    manifest_path: Path,
    repository: Path,
) -> tuple[Path, dict[str, object], list[dict[str, Any]]]:
    if manifest.get("schema_version") != 1:
        raise ValueError("unexpected layout-audit shortlist schema")
    if manifest.get("role") != MANIFEST_ROLE:
        raise ValueError("unexpected layout-audit shortlist role")
    if (
        manifest.get("boundary_contract_version")
        != PRODUCTION_BOUNDARY_CONTRACT_VERSION
    ):
        raise ValueError("layout-audit shortlist boundary drifted")
    _require_false_authorization(manifest, label="shortlist")

    raw_report_path = manifest.get("discovery_report_path")
    expected_report_hash = manifest.get("discovery_report_sha256")
    if not isinstance(raw_report_path, str) or not isinstance(
        expected_report_hash,
        str,
    ):
        raise ValueError("shortlist discovery report identity is missing")
    discovery_path = (PROJECT_ROOT / raw_report_path).resolve()
    if (
        not discovery_path.is_file()
        or sha256_file(discovery_path) != expected_report_hash
    ):
        raise ValueError("shortlist discovery report hash drifted")
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    if discovery.get("role") != DISCOVERY_ROLE:
        raise ValueError("unexpected NIFC discovery report role")
    _require_false_authorization(discovery, label="discovery report")

    raw_cases = discovery.get("cases")
    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_cases, list) or not isinstance(
        raw_candidates,
        list,
    ):
        raise ValueError("shortlist cases are missing")
    cases = {
        str(case["reference_path"]).replace("\\", "/"): case
        for case in raw_cases
        if isinstance(case, dict) and case.get("reference_path")
    }
    if not raw_candidates:
        raise ValueError("shortlist is empty")

    validated: list[dict[str, Any]] = []
    seen_pids: set[str] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            raise ValueError("shortlist candidate is invalid")
        pid = str(item.get("parent_pid", ""))
        reference_path = str(item.get("reference_path", "")).replace(
            "\\",
            "/",
        )
        if not pid or pid in seen_pids or reference_path not in cases:
            raise ValueError("shortlist candidate identity is invalid")
        case = cases[reference_path]
        ranked = case.get("layout_audit_candidates_ranked")
        if not isinstance(ranked, list):
            raise ValueError("discovery report has no ranked candidates")
        matches = [
            candidate
            for candidate in ranked
            if isinstance(candidate, dict)
            and candidate.get("pid") == pid
        ]
        if len(matches) != 1:
            raise ValueError("shortlist PID is not a layout-audit candidate")
        discovered = matches[0]
        evidence = discovered.get("match_evidence")
        if (
            not isinstance(evidence, dict)
            or evidence.get("layout_audit_candidate") is not True
            or item.get("selection_score") != evidence.get("selection_score")
            or item.get("parent_mods_sha256")
            != discovered.get("mods_sha256")
            or item.get("reference_sha256")
            != case.get("reference_sha256")
        ):
            raise ValueError("shortlist candidate evidence drifted")
        reference = (repository / reference_path).resolve()
        if (
            not reference.is_file()
            or sha256_file(reference) != item.get("reference_sha256")
        ):
            raise ValueError("shortlist reference hash drifted")
        seen_pids.add(pid)
        validated.append(
            {
                **item,
                "_reference": reference,
                "_discovered": discovered,
                "_case": case,
            }
        )
    return discovery_path, discovery, validated


def _acquire_candidate(
    item: dict[str, Any],
    output_dir: Path,
    *,
    timeout: float,
    workers: int,
) -> dict[str, object]:
    parent_pid = str(item["parent_pid"])
    stem = parent_pid.replace(":", "-")
    source_dir = output_dir / "sources" / stem
    source_dir.mkdir(parents=True, exist_ok=True)
    page_url = (
        "https://repozytorium.nifc.pl/islandora/object/" + parent_pid
    )
    mods_url = _pid_url(parent_pid, "/datastream/MODS/view")
    parent_html_path = source_dir / "parent.html"
    parent_mods_path = source_dir / "parent.mods.xml"
    parent_html = _read_or_download(
        parent_html_path,
        page_url,
        timeout=timeout,
        maximum_bytes=MAXIMUM_METADATA_BYTES,
    )
    parent_mods = _read_or_download(
        parent_mods_path,
        mods_url,
        timeout=timeout,
        maximum_bytes=MAXIMUM_METADATA_BYTES,
    )
    metadata = parse_mods(parent_mods)
    if (
        hashlib.sha256(parent_mods).hexdigest()
        != item["parent_mods_sha256"]
        or metadata.get("cc_by_4_explicit") is not True
    ):
        raise ValueError(f"parent identity or rights drifted: {parent_pid}")
    children = parse_parent_children(parent_html)
    child_pids = [str(child["pid"]) for child in children]
    if (
        not child_pids
        or len(child_pids) > 32
        or len(set(child_pids)) != len(child_pids)
    ):
        raise ValueError(f"unsafe child sequence: {parent_pid}")

    def acquire_child(
        index: int,
        child: dict[str, str],
    ) -> dict[str, object]:
        pid = str(child["pid"])
        child_stem = pid.replace(":", "-")
        urls = {
            "mods": _pid_url(pid, "/datastream/MODS/view"),
            "rels": _pid_url(pid, "/datastream/RELS-EXT/view"),
            "image": _pid_url(pid, "/datastream/OBJ/view"),
        }
        paths = {
            "mods": source_dir / f"{child_stem}.mods.xml",
            "rels": source_dir / f"{child_stem}.rels-ext.xml",
            "image": source_dir / f"{child_stem}.obj",
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
        actual_child, actual_parent = parse_child_membership(
            payloads["rels"]
        )
        if actual_child != pid or actual_parent != parent_pid:
            raise ValueError(f"child membership drifted: {pid}")
        child_metadata = parse_mods(payloads["mods"])
        image = inspect_image(payloads["image"])
        return {
            "sequence_index": index,
            "pid": pid,
            "listed_title": child["title"],
            "urls": urls,
            "mods_path": str(paths["mods"].resolve()),
            "mods_sha256": hashlib.sha256(payloads["mods"]).hexdigest(),
            "mods_titles": child_metadata["titles"],
            "rels_ext_path": str(paths["rels"].resolve()),
            "rels_ext_sha256": hashlib.sha256(
                payloads["rels"]
            ).hexdigest(),
            "image_path": str(paths["image"].resolve()),
            "image_sha256": hashlib.sha256(
                payloads["image"]
            ).hexdigest(),
            "image": image,
            "membership_verified": True,
            "auto_rotation_applied": False,
            "auto_deskew_applied": False,
            "resampling_applied": False,
            "page_role": "not_classified",
            "page_alignment_status": "not_verified",
        }

    acquired: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(acquire_child, index, child): index
            for index, child in enumerate(children, start=1)
        }
        for future in as_completed(futures):
            result = future.result()
            acquired[int(result["sequence_index"])] = result
    ordered = [acquired[index] for index in range(1, len(children) + 1)]
    return {
        "parent_pid": parent_pid,
        "parent_page_url": page_url,
        "parent_html_path": str(parent_html_path.resolve()),
        "parent_html_sha256": sha256_file(parent_html_path),
        "parent_mods_url": mods_url,
        "parent_mods_path": str(parent_mods_path.resolve()),
        "parent_mods_sha256": sha256_file(parent_mods_path),
        "parent_titles": metadata["titles"],
        "parent_access_conditions": metadata["access_conditions"],
        "parent_cc_by_4_explicit": True,
        "reference_path": str(item["_reference"]),
        "reference_sha256": sha256_file(item["_reference"]),
        "selection_score": item["selection_score"],
        "selection_reason": item["selection_reason"],
        "raw_child_count": len(ordered),
        "children": ordered,
        "child_membership_verified": True,
        "geometric_transform_applied": False,
        "page_roles_classified": False,
        "all_page_alignment_verified": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
    }


def acquire_shortlist(
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
    discovery_path, _discovery, selected = validate_shortlist(
        manifest,
        manifest_path=manifest_path,
        repository=repository,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        _acquire_candidate(
            item,
            output_dir,
            timeout=timeout,
            workers=workers,
        )
        for item in selected
    ]
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "role": REPORT_ROLE,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "discovery_report_path": str(discovery_path),
        "discovery_report_sha256": sha256_file(discovery_path),
        "source_repository_path": str(repository.resolve()),
        "output_dir": str(output_dir.resolve()),
        "candidate_count": len(candidates),
        "unique_reference_count": len(
            {candidate["reference_sha256"] for candidate in candidates}
        ),
        "raw_child_image_count": sum(
            int(candidate["raw_child_count"]) for candidate in candidates
        ),
        "parent_rights_verified_count": sum(
            candidate["parent_cc_by_4_explicit"] is True
            for candidate in candidates
        ),
        "child_membership_verified_count": sum(
            candidate["child_membership_verified"] is True
            for candidate in candidates
        ),
        "page_roles_classified": False,
        "all_page_alignment_verified": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "authorization_blockers": [
            "raw_repository_children_include_unknown_non_music_pages",
            "spread_or_single_page_layout_not_classified",
            "all_page_alignment_not_verified",
            "absence_of_reference_problem_comments_does_not_prove_completeness",
            "independent_double_annotation_not_started",
        ],
        "candidates": candidates,
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
    report = acquire_shortlist(
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
                    "candidate_count",
                    "unique_reference_count",
                    "raw_child_image_count",
                    "parent_rights_verified_count",
                    "child_membership_verified_count",
                    "all_page_alignment_verified",
                    "training_authorized",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
