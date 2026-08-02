from __future__ import annotations

"""Fail closed if the active semantic queue can see quarantined diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from app.tools.expand_overlapping_semantic_targets import (
    TRANSFORMATION_VERSION,
)
from app.tools.build_muse_omr_work_catalog import (
    mscx_payload_fingerprint,
)
from app.tools.prepare_openscore_svg_regions import (
    COMPLETE_PAGE_TARGET_PROVENANCE,
)
APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
TRAINING_DATA_ROOT = PROJECT_ROOT / "training_data"
sys.path.insert(0, str(APP_ROOT / "src"))

from scorescan.util import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    utc_now_iso,
)


REPORT_ROLE = "active_semantic_dataset_diagnostic_quarantine_audit"
HEX64_PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
IMAGE_PATTERN = re.compile(r'"image":"((?:[^"\\]|\\.)*)"')
SOURCE_KEY_PATTERN = re.compile(r'"source_key":"((?:[^"\\]|\\.)*)"')
SPLIT_PATTERN = re.compile(r'"split":"((?:[^"\\]|\\.)*)"')
FORBIDDEN_TEXT_FRAGMENTS = (
    "external/diagnostic",
    "external\\diagnostic",
    "nifc:",
    "nifc-",
    "collabscore_dataset_source_v1",
    "beethoven_piano_sonatas_source_v1",
)
DATASETS = (
    {
        "name": "active_muse_training",
        "prepared": (
            "training_data/prepared/"
            "muse_omr_scan_regions_stratified_complete_page_"
            "overlap_consistent_deduplicated_v7"
        ),
        "images": (
            "training_data/prepared/"
            "muse_omr_scan_regions_stratified_complete_page_v6"
        ),
        "expected_role": (
            "training_only_disjoint_from_external_release_holdout"
        ),
        "splits": ("train", "calibration", "test"),
        "source_key_prefix": "muse-omr-work/",
        "partition": "training",
    },
    {
        "name": "active_openscore_replay",
        "prepared": (
            "training_data/prepared/"
            "openscore_lieder_train_1091_svg_regions_complete_page_"
            "overlap_consistent_deduplicated_v4"
        ),
        "images": (
            "training_data/prepared/"
            "openscore_lieder_train_1091_svg_regions_complete_page_v2"
        ),
        "expected_role": "training_only_synthetic_semantic_geometry",
        "splits": ("train", "calibration", "test"),
        "source_key_prefix": "scores/",
        "partition": "replay",
    },
    {
        "name": "reserved_muse_holdout",
        "prepared": (
            "training_data/prepared/"
            "muse_omr_scan_holdout_regions_stratified_complete_page_"
            "overlap_consistent_deduplicated_v7"
        ),
        "images": (
            "training_data/prepared/"
            "muse_omr_scan_holdout_regions_stratified_complete_page_v6"
        ),
        "expected_role": (
            "external_scan_degraded_development_benchmark_not_training"
        ),
        "splits": ("test",),
        "source_key_prefix": "muse-omr-work/",
        "partition": "holdout",
    },
)
DIAGNOSTIC_REPORTS = (
    (
        "training_data/external/diagnostic/"
        "nifc_chopin_layout_audit_shortlist_v1/candidate_report.json"
    ),
    (
        "training_data/external/diagnostic/"
        "nifc_chopin_layout_audit_pages_v1/page_preparation_report.json"
    ),
    (
        "training_data/external/diagnostic/"
        "nifc_chopin_subwork_alignment_v1/subwork_alignment_report.json"
    ),
    (
        "training_data/external/diagnostic/"
        "nifc_chopin_primary_visual_review_v1/"
        "primary_visual_review_report.json"
    ),
    (
        "training_data/benchmarks/"
        "collabscore_pinned_metadata_boundary_audit_v1.json"
    ),
    (
        "training_data/benchmarks/"
        "beethoven_piano_sonatas_pinned_scan_reference_audit_v1.json"
    ),
)
REPLAY_ISOLATION_REPORT = (
    PROJECT_ROOT
    / "training_data"
    / "diagnostics"
    / "semantic-replay-holdout-isolation-v1.json"
)
LIEDER_CORPUS_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "external"
    / "corpora"
    / "openscore_lieder_6b2dc542"
    / "Lieder-6b2dc542ce2e8aa4b78c8ee62103b210efc07015"
)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _within_project_or_training_data(path: Path) -> bool:
    """Accept the project tree and its explicitly mounted training-data root."""

    return _within(path, PROJECT_ROOT) or _within(path, TRAINING_DATA_ROOT)


def _diagnostic_hashes() -> tuple[set[str], list[dict[str, object]]]:
    hashes: set[str] = set()
    reports: list[dict[str, object]] = []
    for relative in DIAGNOSTIC_REPORTS:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file():
            raise ValueError(f"required diagnostic report is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for field in (
            "training_authorized",
            "evaluation_authorized",
            "release_authorized",
        ):
            if payload.get(field) is not False:
                raise ValueError(
                    f"diagnostic report unexpectedly sets {field}: {path}"
                )

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if (
                        key.endswith("sha256")
                        and isinstance(child, str)
                        and HEX64_PATTERN.fullmatch(child)
                    ):
                        hashes.add(child)
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
        reports.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "role": payload.get("role", ""),
            }
        )
    return hashes, reports


def _audit_dataset(
    spec: dict[str, object],
    forbidden_hashes: set[str],
) -> tuple[dict[str, object], set[str]]:
    prepared = (PROJECT_ROOT / str(spec["prepared"])).resolve()
    images_root = (PROJECT_ROOT / str(spec["images"])).resolve()
    manifest_path = prepared / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"semantic dataset manifest is missing: {prepared}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("role") != spec["expected_role"]:
        raise ValueError(f"semantic dataset role drifted: {prepared}")
    if manifest.get("source_split_overlap") != 0:
        raise ValueError(f"semantic source split overlap: {prepared}")
    reserved_overlap = manifest.get("reserved_holdout_overlap")
    if (
        spec["partition"] != "replay"
        and reserved_overlap != 0
    ):
        raise ValueError(f"semantic reserved holdout overlap: {prepared}")
    if spec["partition"] == "replay" and reserved_overlap not in (None, 0):
        raise ValueError(f"semantic replay reserved overlap drifted: {prepared}")
    if manifest.get("target_assignment_version") != TRANSFORMATION_VERSION:
        raise ValueError(f"semantic target assignment drifted: {prepared}")
    if (
        manifest.get("target_geometry_provenance")
        != COMPLETE_PAGE_TARGET_PROVENANCE
    ):
        raise ValueError(f"semantic target geometry drifted: {prepared}")
    source_keys: set[str] = set()
    image_paths: set[str] = set()
    row_count = 0
    object_count = 0
    forbidden_text_occurrences = 0
    forbidden_hash_occurrences = 0
    split_reports: list[dict[str, object]] = []
    for split in spec["splits"]:
        jsonl_path = prepared / f"{split}.jsonl"
        if not jsonl_path.is_file():
            raise ValueError(f"semantic split is missing: {jsonl_path}")
        split_rows = 0
        split_objects = 0
        digest = hashlib.sha256()
        with jsonl_path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                line = raw_line.decode("utf-8")
                image_match = IMAGE_PATTERN.search(line)
                source_match = SOURCE_KEY_PATTERN.search(line)
                split_match = SPLIT_PATTERN.search(line)
                if (
                    image_match is None
                    or source_match is None
                    or split_match is None
                ):
                    raise ValueError(
                        f"semantic JSONL identity fields are missing "
                        f"{jsonl_path}:{line_number}"
                    )
                image = json.loads(f'"{image_match.group(1)}"')
                source_key = json.loads(f'"{source_match.group(1)}"')
                row_split = json.loads(f'"{split_match.group(1)}"')
                identity_text = f"{image}\n{source_key}".casefold()
                forbidden_text_occurrences += sum(
                    fragment in identity_text
                    for fragment in FORBIDDEN_TEXT_FRAGMENTS
                )
                forbidden_hash_occurrences += len(
                    set(HEX64_PATTERN.findall(identity_text))
                    & forbidden_hashes
                )
                if row_split != split:
                    raise ValueError(
                        f"semantic split label drifted: {jsonl_path}"
                    )
                if not source_key.startswith(
                    str(spec["source_key_prefix"])
                ):
                    raise ValueError(
                        f"semantic source-key domain drifted: {jsonl_path}"
                    )
                if (
                    not image
                    or Path(image).is_absolute()
                    or ".." in Path(image).parts
                ):
                    raise ValueError(
                        f"semantic image path escapes root: {jsonl_path}"
                    )
                row_object_count = line.count('"category_id":')
                if row_object_count < 0:
                    raise ValueError(
                        f"semantic row has no objects: {jsonl_path}"
                    )
                source_keys.add(source_key)
                image_paths.add(image)
                split_rows += 1
                split_objects += row_object_count
        row_count += split_rows
        object_count += split_objects
        split_reports.append(
            {
                "split": split,
                "jsonl_path": str(jsonl_path),
                "jsonl_sha256": digest.hexdigest(),
                "rows": split_rows,
                "objects": split_objects,
            }
        )
    missing_images = [
        image
        for image in sorted(image_paths)
        if not (images_root / image).is_file()
    ]
    if missing_images:
        raise ValueError(
            f"semantic dataset has {len(missing_images)} missing images"
        )
    if forbidden_text_occurrences or forbidden_hash_occurrences:
        raise ValueError(
            f"diagnostic data leaked into semantic dataset {spec['name']}"
        )
    return (
        {
            "name": spec["name"],
            "partition": spec["partition"],
            "prepared_dir": str(prepared),
            "images_root": str(images_root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "role": manifest["role"],
            "split_reports": split_reports,
            "row_count": row_count,
            "object_count": object_count,
            "unique_source_key_count": len(source_keys),
            "unique_image_count": len(image_paths),
            "missing_image_count": 0,
            "forbidden_text_occurrence_count": 0,
            "forbidden_diagnostic_hash_occurrence_count": 0,
        },
        source_keys,
    )


def _audit_replay_holdout_isolation(
    replay_source_keys: set[str],
) -> dict[str, object]:
    if not REPLAY_ISOLATION_REPORT.is_file():
        raise ValueError(
            "semantic replay/holdout isolation report is missing"
        )
    payload = json.loads(
        REPLAY_ISOLATION_REPORT.read_text(encoding="utf-8")
    )
    if (
        payload.get("role")
        != "training_holdout_work_isolation_evidence"
        or payload.get("score_source_priority") != "mscx-before-mscz"
        or payload.get("tool_source_sha256")
        != sha256_file(
            Path(__file__).with_name(
                "audit_semantic_replay_holdout_isolation.py"
            )
        )
        or int(payload.get("holdout_selected_works", 0)) != 395
        or int(payload.get("replay_works", 0)) != 1474
        or payload.get("work_overlap") != []
    ):
        raise ValueError("semantic replay/holdout isolation contract drifted")

    bound_files = (
        ("holdout_selection", "holdout_selection_sha256"),
        ("replay_prepare_report", "replay_prepare_report_sha256"),
    )
    bound_hashes: dict[str, str] = {}
    for path_field, hash_field in bound_files:
        candidate = Path(str(payload.get(path_field, ""))).resolve()
        if not _within_project_or_training_data(candidate) or not candidate.is_file():
            raise ValueError(
                f"semantic isolation bound file is invalid: {path_field}"
            )
        expected_hash = str(payload.get(hash_field, ""))
        actual_hash = sha256_file(candidate)
        if expected_hash != actual_hash:
            raise ValueError(
                f"semantic isolation bound hash drifted: {path_field}"
            )
        bound_hashes[hash_field] = actual_hash

    lieder_root = LIEDER_CORPUS_ROOT.resolve()
    matching_roots = [
        root
        for root in payload.get("replay_roots", [])
        if isinstance(root, dict)
        and Path(str(root.get("root", ""))).resolve()
        == lieder_root.parent
    ]
    if len(matching_roots) != 1:
        raise ValueError("semantic isolation has no unique Lieder root")
    audited_files: dict[str, dict[str, object]] = {}
    for item in matching_roots[0].get("files", []):
        if not isinstance(item, dict):
            raise ValueError("semantic isolation contains an invalid file")
        path = Path(str(item.get("path", ""))).resolve()
        if not _within(path, lieder_root):
            continue
        key = path.relative_to(lieder_root).with_suffix("").as_posix()
        audited_files[key] = item
    replay_stems = {
        Path(source_key).with_suffix("").as_posix()
        for source_key in replay_source_keys
    }
    missing = sorted(replay_stems - set(audited_files))
    if missing:
        raise ValueError(
            "active semantic replay has unaudited source keys: "
            f"{len(missing)}"
        )
    for source_key in sorted(replay_source_keys):
        source_path = (lieder_root / source_key).resolve()
        source_stem = Path(source_key).with_suffix("").as_posix()
        item = audited_files[source_stem]
        audited_path = Path(str(item.get("path", ""))).resolve()
        source_fingerprint = (
            mscx_payload_fingerprint(source_path.read_bytes())
            if source_path.suffix.casefold() == ".mscx"
            else ""
        )
        if (
            not source_path.is_file()
            or not audited_path.is_file()
            or sha256_file(audited_path) != item.get("sha256")
            or source_fingerprint != item.get("work_fingerprint")
            or not HEX64_PATTERN.fullmatch(
                str(item.get("work_fingerprint", ""))
            )
        ):
            raise ValueError(
                f"active semantic replay source drifted: {source_key}"
            )
    return {
        "path": str(REPLAY_ISOLATION_REPORT.resolve()),
        "sha256": sha256_file(REPLAY_ISOLATION_REPORT),
        "role": payload["role"],
        "holdout_selected_works": payload["holdout_selected_works"],
        "replay_works": payload["replay_works"],
        "work_overlap_count": 0,
        "active_replay_source_key_count": len(replay_source_keys),
        "active_replay_unaudited_source_key_count": 0,
        **bound_hashes,
    }


def audit(output_path: Path) -> dict[str, object]:
    forbidden_hashes, diagnostic_reports = _diagnostic_hashes()
    datasets: list[dict[str, object]] = []
    source_keys: dict[str, set[str]] = {}
    for spec in DATASETS:
        result, keys = _audit_dataset(spec, forbidden_hashes)
        datasets.append(result)
        source_keys[str(spec["partition"])] = keys
    replay_isolation = _audit_replay_holdout_isolation(
        source_keys["replay"]
    )
    train_holdout_overlap = sorted(
        source_keys["training"] & source_keys["holdout"]
    )
    replay_holdout_overlap = sorted(
        source_keys["replay"] & source_keys["holdout"]
    )
    if train_holdout_overlap or replay_holdout_overlap:
        raise ValueError("active semantic training overlaps reserved holdout")
    queue_script = PROJECT_ROOT / "training" / (
        "run_wsl_muse_scan_semantic_detector_full.sh"
    )
    queue_text = queue_script.read_text(encoding="utf-8")
    required_queue_fragments = [
        str(spec["prepared"]).split("/")[-1] for spec in DATASETS[:2]
    ]
    if any(fragment not in queue_text for fragment in required_queue_fragments):
        raise ValueError("active semantic queue dataset binding drifted")
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "role": REPORT_ROLE,
        "passed": True,
        "queue_script_path": str(queue_script.resolve()),
        "queue_script_sha256": sha256_file(queue_script),
        "diagnostic_report_count": len(diagnostic_reports),
        "forbidden_diagnostic_hash_count": len(forbidden_hashes),
        "diagnostic_reports": diagnostic_reports,
        "replay_holdout_isolation": replay_isolation,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "training_holdout_source_key_overlap_count": 0,
        "replay_holdout_source_key_overlap_count": 0,
        "diagnostic_text_occurrence_count": 0,
        "diagnostic_hash_occurrence_count": 0,
        "training_authorized_diagnostic_page_count": 0,
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output.resolve())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "diagnostic_report_count",
                    "forbidden_diagnostic_hash_count",
                    "dataset_count",
                    "training_holdout_source_key_overlap_count",
                    "replay_holdout_source_key_overlap_count",
                    "diagnostic_text_occurrence_count",
                    "diagnostic_hash_occurrence_count",
                    "training_authorized_diagnostic_page_count",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
