from __future__ import annotations

"""Extract and boundary-check PDMX MXL references linked to IMSLP scans."""

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from lxml import etree

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

from app.tools.acquire_pdmx_imslp_scans import (  # noqa: E402
    ROLE as SCAN_MANIFEST_ROLE,
)
from app.tools.acquire_pdmx_mxl_archive import (  # noqa: E402
    EXPECTED_BYTES,
    EXPECTED_MD5,
    RECORD_ID,
    ROLE as MXL_ACQUISITION_ROLE,
    VERSION,
)
from app.tools.filter_pdmx_imslp_license_candidates import (  # noqa: E402
    ROLE as FILTER_ROLE,
)
from app.tools.probe_pdmx_imslp_scan_sources import (  # noqa: E402
    ROLE as SOURCE_EVIDENCE_ROLE,
)
from app.tools.prepare_muse_omr_benchmark import (  # noqa: E402
    analyze_reference_boundary,
)
from scorescan.product_scope import (  # noqa: E402
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)
from scorescan.util import atomic_write_json, sha256_file, utc_now_iso  # noqa: E402


ROLE = "pdmx_imslp_semantic_candidates_not_aligned_or_training_authorized"
MAXIMUM_MXL_MEMBER_BYTES = 64 * 1024 * 1024
MAXIMUM_XML_BYTES = 256 * 1024 * 1024
CONTAINER_PATH = "META-INF/container.xml"
EXPECTED_SHAPES = {
    "keyboard_candidate": "keyboard",
    "keyboard_plus_single_staff_ensemble_candidate": (
        "keyboard_plus_single_staff_ensemble"
    ),
    "single_staff_ensemble_candidate": "single_staff_ensemble",
    "single_staff_solo_candidate": "single_staff_solo",
}
PMLP = re.compile(r"PMLP[0-9]+", re.IGNORECASE)


def _safe_zip_member(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        not path.is_absolute()
        and path.parts
        and all(part not in {"", ".", ".."} for part in path.parts)
        and ":" not in path.parts[0]
    )


def extract_musicxml(mxl_bytes: bytes) -> tuple[bytes, dict[str, object]]:
    if (
        len(mxl_bytes) < 4
        or len(mxl_bytes) > MAXIMUM_MXL_MEMBER_BYTES
        or not mxl_bytes.startswith(b"PK")
    ):
        raise ValueError("invalid or oversized MXL member")
    try:
        archive = zipfile.ZipFile(io.BytesIO(mxl_bytes))
    except zipfile.BadZipFile as error:
        raise ValueError("MXL ZIP parser rejected member") from error
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > 10_000:
            raise ValueError("MXL member count is unsafe")
        total_uncompressed = 0
        names: set[str] = set()
        for info in infos:
            if (
                not _safe_zip_member(info.filename)
                or info.flag_bits & 0x1
                or info.file_size < 0
                or info.file_size > MAXIMUM_XML_BYTES
                or info.filename in names
            ):
                raise ValueError("unsafe MXL ZIP member")
            names.add(info.filename)
            total_uncompressed += info.file_size
            if total_uncompressed > MAXIMUM_XML_BYTES:
                raise ValueError("MXL uncompressed byte count is unsafe")
        if CONTAINER_PATH not in names:
            raise ValueError("MXL container metadata is missing")
        parser = etree.XMLParser(
            resolve_entities=False,
            no_network=True,
            recover=False,
        )
        container = etree.fromstring(
            archive.read(CONTAINER_PATH),
            parser,
        )
        root_paths = container.xpath(
            "//*[local-name()='rootfile']/@full-path",
        )
        if len(root_paths) != 1:
            raise ValueError("MXL container has ambiguous rootfile")
        root_path = str(root_paths[0])
        if (
            root_path not in names
            or not _safe_zip_member(root_path)
            or not root_path.casefold().endswith((".xml", ".musicxml"))
        ):
            raise ValueError("MXL rootfile is unsafe or missing")
        musicxml = archive.read(root_path)
        if not musicxml or len(musicxml) > MAXIMUM_XML_BYTES:
            raise ValueError("MXL MusicXML root is empty or oversized")
        root = etree.fromstring(musicxml, parser)
        if etree.QName(root).localname != "score-partwise":
            raise ValueError("only score-partwise MusicXML is supported")
        return musicxml, {
            "mxl_member_count": len(infos),
            "mxl_uncompressed_bytes": total_uncompressed,
            "musicxml_rootfile": root_path,
            "musicxml_bytes": len(musicxml),
        }


def _work_group(candidate: dict[str, object]) -> str:
    source_evidence = candidate.get("imslp_source_evidence")
    if isinstance(source_evidence, list):
        values = sorted(
            {
                token.upper()
                for source in source_evidence
                if isinstance(source, dict)
                for token in PMLP.findall(
                    str(source.get("direct_pdf_url", ""))
                )
            }
        )
        if values:
            return "imslp-work-" + "-".join(values)
    source_ids = candidate.get("verified_imslp_source_ids")
    if isinstance(source_ids, list) and source_ids:
        return "imslp-source-" + "-".join(
            sorted((str(value) for value in source_ids), key=int)
        )
    raise ValueError("candidate has no stable IMSLP work group")


def _immutable_splits(groups: set[str]) -> dict[str, str]:
    ordered = sorted(
        groups,
        key=lambda value: (
            hashlib.sha256(value.encode()).hexdigest(),
            value,
        ),
    )
    if len(ordered) < 3:
        return {value: "train" for value in ordered}
    return {
        value: (
            "test"
            if index == len(ordered) - 1
            else "calibration"
            if index == len(ordered) - 2
            else "train"
        )
        for index, value in enumerate(ordered)
    }


def prepare(
    archive_path: Path,
    mxl_acquisition_path: Path,
    filtered_path: Path,
    scan_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, object]:
    acquisition = json.loads(
        mxl_acquisition_path.read_text(encoding="utf-8")
    )
    acquisition_asset = acquisition.get("asset")
    archive_sha256 = sha256_file(archive_path)
    if (
        acquisition.get("role") != MXL_ACQUISITION_ROLE
        or acquisition.get("record_id") != RECORD_ID
        or acquisition.get("version") != VERSION
        or acquisition.get("expected_bytes") != EXPECTED_BYTES
        or acquisition.get("expected_md5") != EXPECTED_MD5
        or acquisition.get("training_authorized") is not False
        or not isinstance(acquisition_asset, dict)
        or acquisition_asset.get("sha256") != archive_sha256
        or archive_path.stat().st_size != EXPECTED_BYTES
    ):
        raise ValueError("unexpected PDMX MXL acquisition contract")
    filtered = json.loads(filtered_path.read_text(encoding="utf-8"))
    filtered_candidates = filtered.get("candidates")
    if (
        filtered.get("role") != FILTER_ROLE
        or filtered.get("training_authorized") is not False
        or not isinstance(filtered_candidates, list)
    ):
        raise ValueError("unexpected PDMX filtered candidate contract")
    by_score_id: dict[int, dict[str, object]] = {}
    for candidate in filtered_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid PDMX filtered candidate")
        score_id = candidate.get("score_id")
        if (
            isinstance(score_id, bool)
            or not isinstance(score_id, int)
            or score_id in by_score_id
        ):
            raise ValueError("invalid or duplicate PDMX score id")
        by_score_id[score_id] = candidate
    scan_manifest = json.loads(
        scan_manifest_path.read_text(encoding="utf-8")
    )
    scan_assets = scan_manifest.get("assets")
    if (
        scan_manifest.get("role") != SCAN_MANIFEST_ROLE
        or scan_manifest.get("record_id") != RECORD_ID
        or scan_manifest.get("version") != VERSION
        or scan_manifest.get("training_authorized") is not False
        or scan_manifest.get("transport_authenticated") is not False
        or not isinstance(scan_assets, list)
    ):
        raise ValueError("unexpected PDMX IMSLP scan-byte contract")
    source_evidence_path = Path(
        str(scan_manifest.get("evidence_path", ""))
    )
    if (
        not source_evidence_path.is_file()
        or scan_manifest.get("evidence_sha256")
        != sha256_file(source_evidence_path)
    ):
        raise ValueError("scan manifest source evidence hash mismatch")
    source_evidence = json.loads(
        source_evidence_path.read_text(encoding="utf-8")
    )
    source_candidates = source_evidence.get("verified_candidates")
    if (
        source_evidence.get("role") != SOURCE_EVIDENCE_ROLE
        or source_evidence.get("training_authorized") is not False
        or not isinstance(source_candidates, list)
    ):
        raise ValueError("unexpected PDMX IMSLP source evidence contract")
    source_candidate_by_score_id: dict[int, dict[str, object]] = {}
    for candidate in source_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid verified PDMX source candidate")
        score_id = candidate.get("score_id")
        if (
            isinstance(score_id, bool)
            or not isinstance(score_id, int)
            or score_id in source_candidate_by_score_id
        ):
            raise ValueError("invalid verified PDMX source score id")
        source_candidate_by_score_id[score_id] = candidate
    scan_by_score_id: dict[int, dict[str, object]] = {}
    for asset in scan_assets:
        if not isinstance(asset, dict):
            raise ValueError("invalid PDMX IMSLP scan asset")
        scan_path = Path(str(asset.get("pdf_path", "")))
        if (
            not scan_path.is_file()
            or asset.get("pdf_sha256") != sha256_file(scan_path)
            or asset.get("pdf_bytes") != scan_path.stat().st_size
        ):
            raise ValueError("PDMX IMSLP scan asset hash mismatch")
        score_ids = asset.get("pdmx_score_ids")
        if not isinstance(score_ids, list) or not score_ids:
            raise ValueError("scan asset has no PDMX score identity")
        for score_id_value in score_ids:
            if (
                isinstance(score_id_value, bool)
                or not isinstance(score_id_value, int)
                or score_id_value in scan_by_score_id
                or score_id_value not in by_score_id
                or score_id_value not in source_candidate_by_score_id
            ):
                raise ValueError("invalid or duplicate scan-to-score mapping")
            scan_by_score_id[score_id_value] = asset
    target_members: dict[str, int] = {}
    for score_id in scan_by_score_id:
        member = str(
            by_score_id[score_id].get("pdmx_mxl_archive_member", "")
        )
        if (
            not re.fullmatch(
                r"mxl/[A-Za-z0-9_./=-]+\.mxl",
                member,
            )
            or ".." in PurePosixPath(member).parts
            or member in target_members
        ):
            raise ValueError("invalid or duplicate PDMX MXL member path")
        target_members[member] = score_id

    output_dir.mkdir(parents=True, exist_ok=True)
    found: dict[int, bytes] = {}
    member_count_scanned = 0
    duplicate_targets: set[str] = set()
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            member_count_scanned += 1
            score_id = target_members.get(member.name)
            if score_id is None:
                continue
            if (
                not member.isfile()
                or member.size <= 0
                or member.size > MAXIMUM_MXL_MEMBER_BYTES
                or score_id in found
            ):
                duplicate_targets.add(member.name)
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("target PDMX MXL member cannot be read")
            payload = stream.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError("target PDMX MXL member is truncated")
            found[score_id] = payload
    if duplicate_targets or set(found) != set(scan_by_score_id):
        raise ValueError("PDMX MXL targets are duplicate or missing")

    provisional: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    created_paths: list[Path] = []
    try:
        for score_id in sorted(found):
            candidate = by_score_id[score_id]
            expected_shape = EXPECTED_SHAPES.get(
                str(candidate.get("boundary_hint", ""))
            )
            if expected_shape is None:
                raise ValueError("candidate boundary hint is unsupported")
            try:
                musicxml, mxl_info = extract_musicxml(found[score_id])
                case_dir = output_dir / f"PDMX{score_id:07d}"
                case_dir.mkdir(parents=True, exist_ok=True)
                mxl_path = case_dir / "reference.mxl"
                xml_path = case_dir / "reference.musicxml"
                mxl_path.write_bytes(found[score_id])
                xml_path.write_bytes(musicxml)
                created_paths.extend((mxl_path, xml_path))
                boundary = analyze_reference_boundary(xml_path)
                boundary_consistent = bool(
                    boundary.get("accepted") is True
                    and boundary.get("score_shape") == expected_shape
                )
                if not boundary_consistent:
                    mxl_path.unlink(missing_ok=True)
                    xml_path.unlink(missing_ok=True)
                    rejected.append(
                        {
                            "score_id": score_id,
                            "reason": "parsed_reference_outside_declared_boundary",
                            "expected_score_shape": expected_shape,
                            "boundary": boundary,
                        }
                    )
                    continue
                scan_asset = scan_by_score_id[score_id]
                group = _work_group(
                    source_candidate_by_score_id[score_id]
                )
                provisional.append(
                    {
                        "case_id": f"pdmx-imslp-{score_id}",
                        "id": f"pdmx-imslp-{score_id}",
                        "score_id": score_id,
                        "work_group": group,
                        "work_fingerprint": hashlib.sha256(
                            group.encode()
                        ).hexdigest(),
                        "boundary_configuration": expected_shape,
                        "boundary": boundary,
                        "mxl_archive_member": candidate[
                            "pdmx_mxl_archive_member"
                        ],
                        "mxl_path": str(mxl_path),
                        "mxl_sha256": sha256_file(mxl_path),
                        "musicxml_path": str(xml_path),
                        "musicxml_sha256": sha256_file(xml_path),
                        **mxl_info,
                        "scan_path": scan_asset["pdf_path"],
                        "scan_sha256": scan_asset["pdf_sha256"],
                        "scan_page_count": scan_asset[
                            "actual_page_count"
                        ],
                        "imslp_source_id": scan_asset[
                            "imslp_source_id"
                        ],
                        "scan_transport_authenticated": False,
                        "exact_scan_to_semantic_alignment_verified": False,
                    }
                )
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                rejected.append(
                    {
                        "score_id": score_id,
                        "reason": (
                            f"{type(error).__name__}: {error}"
                        ),
                    }
                )
    except BaseException:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise
    splits = _immutable_splits(
        {str(row["work_group"]) for row in provisional}
    )
    cases = [
        {
            **row,
            "semantic_split": splits[str(row["work_group"])],
            "split_role": (
                "training_split_semantic_development_only"
                if splits[str(row["work_group"])] == "train"
                else "calibration_split_semantic_development_only"
                if splits[str(row["work_group"])] == "calibration"
                else "test_split_semantic_development_only"
            ),
            "input_pdf": row["scan_path"],
            "input_pdf_sha256": row["scan_sha256"],
            "input_pdf_pages": row["scan_page_count"],
            "reference": row["musicxml_path"],
            "reference_sha256": row["musicxml_sha256"],
            "reference_musicxml": row["musicxml_path"],
            "reference_musicxml_sha256": row["musicxml_sha256"],
            "whole_work_semantic_development_evaluation_authorized": (
                False
            ),
            "page_training_labels_authorized": False,
            "page_level_training_authorized": False,
            "page_level_release_evaluation_authorized": False,
            "independent_release_evaluation_authorized": False,
            "boundary_identity_consistent": True,
        }
        for row in provisional
    ]
    split_counts = Counter(str(row["semantic_split"]) for row in cases)
    boundary_counts = Counter(
        str(row["boundary_configuration"]) for row in cases
    )
    report = {
        "format": 1,
        "created_at": utc_now_iso(),
        "name": "PDMX/IMSLP semantic corpus candidates",
        "role": ROLE,
        "boundary_contract_version": PRODUCTION_BOUNDARY_CONTRACT_VERSION,
        "record_id": RECORD_ID,
        "version": VERSION,
        "mxl_archive_path": str(archive_path),
        "mxl_archive_sha256": archive_sha256,
        "mxl_acquisition_path": str(mxl_acquisition_path),
        "mxl_acquisition_sha256": sha256_file(mxl_acquisition_path),
        "filtered_path": str(filtered_path),
        "filtered_sha256": sha256_file(filtered_path),
        "scan_manifest_path": str(scan_manifest_path),
        "scan_manifest_sha256": sha256_file(scan_manifest_path),
        "source_evidence_path": str(source_evidence_path),
        "source_evidence_sha256": sha256_file(source_evidence_path),
        "training_authorized": False,
        "whole_work_semantic_development_evaluation_authorized": False,
        "page_training_labels_authorized": False,
        "page_level_training_authorized": False,
        "page_level_release_evaluation_authorized": False,
        "release_evaluation_authorized": False,
        "evaluation_authorized": False,
        "release_authorized": False,
        "independent_holdout": False,
        "independent_holdout_authorized": False,
        "independent_release_evaluation_authorized": False,
        "authorization_reason": (
            "symbolic and scan identities plus parsed product boundary are "
            "verified, but the scan transport lacks TLS/independent hash and "
            "exact page/measure alignment is not established; semantic "
            "comparison is therefore not authorized until a reviewed aligned "
            "movement/page-range derivative is prepared"
        ),
        "scan_transport_authenticated": False,
        "exact_scan_to_semantic_alignment_verified": False,
        "archive_member_count_scanned": member_count_scanned,
        "target_count": len(target_members),
        "accepted_case_count": len(cases),
        "rejected_case_count": len(rejected),
        "accepted_page_count": sum(
            int(row["scan_page_count"]) for row in cases
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "boundary_configuration_counts": dict(
            sorted(boundary_counts.items())
        ),
        "work_group_splits": dict(sorted(splits.items())),
        "cases": cases,
        "rejected_cases": rejected,
    }
    atomic_write_json(manifest_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_path", type=Path)
    parser.add_argument("mxl_acquisition_path", type=Path)
    parser.add_argument("filtered_path", type=Path)
    parser.add_argument("scan_manifest_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("manifest_path", type=Path)
    args = parser.parse_args()
    report = prepare(
        args.archive_path.resolve(),
        args.mxl_acquisition_path.resolve(),
        args.filtered_path.resolve(),
        args.scan_manifest_path.resolve(),
        args.output_dir.resolve(),
        args.manifest_path.resolve(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "archive_member_count_scanned",
                    "target_count",
                    "accepted_case_count",
                    "rejected_case_count",
                    "accepted_page_count",
                    "split_counts",
                    "boundary_configuration_counts",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
