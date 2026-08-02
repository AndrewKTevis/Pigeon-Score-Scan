from __future__ import annotations

"""Forward-only migrations for durable job state."""

from copy import deepcopy
from typing import Any

CURRENT_JOB_SCHEMA = 25


class UnsupportedJobSchema(ValueError):
    pass


def migrate_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(payload)
    version = int(result.get("schema_version", 1) or 1)
    if version > CURRENT_JOB_SCHEMA:
        raise UnsupportedJobSchema(
            f"任务状态版本 {version} 高于当前程序支持的 {CURRENT_JOB_SCHEMA}"
        )
    if version < 2:
        result.setdefault("artifact_manifest_path", None)
        result.setdefault("artifact_bundle_id", None)
        result.setdefault("quality_state", "processing")
        result.setdefault("quality_score", None)
        result.setdefault("review_issues", [])
        result.setdefault("review_resolved_count", 0)
        result.setdefault("resumable", True)
        result["schema_version"] = 2
        version = 2
    if version < 3:
        # PageInfo itself is tolerant of absent fields, but a schema step makes the
        # persistence contract explicit and auditable for this release.
        result["schema_version"] = 3
        version = 3
    if version < 4:
        # v4 persists the bounded ensemble meta-calibration evidence on each page.
        # PageInfo defaults keep older states readable without rewriting page payloads.
        result["schema_version"] = 4
        version = 4
    if version < 5:
        # v5 persists the selective replacement-risk evidence on each page.
        # PageInfo defaults preserve compatibility with older states.
        result["schema_version"] = 5
        version = 5
    if version < 6:
        # v6 persists OCR direction-role and barline-aware anchoring summaries.
        # PageInfo defaults keep historical tasks readable without rewriting pages.
        result["schema_version"] = 6
        version = 6
    if version < 7:
        # v7 persists page-orientation decisions and model audit evidence.
        # PageInfo defaults preserve older jobs without rewriting page payloads.
        result["schema_version"] = 7
        version = 7
    if version < 8:
        # v8 persists layout/OMR measure-count fusion evidence.  PageInfo defaults
        # preserve historical jobs without rewriting individual page payloads.
        result["schema_version"] = 8
        version = 8
    if version < 9:
        # v9 persists conservative one-event insertion/deletion consensus evidence.
        # PageInfo defaults keep existing jobs readable without rewriting page arrays.
        result["schema_version"] = 9
        version = 9
    if version < 10:
        # v10 persists conservative chord-topology consensus evidence.  PageInfo
        # defaults keep historical jobs readable without rewriting page arrays.
        result["schema_version"] = 10
        version = 10
    if version < 11:
        # v11 persists conservative tie-topology consensus and composed patch
        # transaction-guard evidence.  PageInfo defaults preserve older jobs.
        result["schema_version"] = 11
        version = 11
    if version < 12:
        # v12 persists conservative cross-measure tie-boundary consensus evidence.
        # PageInfo defaults keep existing jobs readable without rewriting page arrays.
        result["schema_version"] = 12
        version = 12
    if version < 13:
        # v13 persists conservative simple-triplet consensus evidence.  PageInfo
        # defaults keep historical jobs readable without rewriting page arrays.
        result["schema_version"] = 13
        version = 13
    if version < 14:
        # v14 persists within-measure slur consensus evidence and production
        # auto-release audit metadata. PageInfo and certificate defaults keep
        # historical jobs readable without rewriting page arrays.
        result["schema_version"] = 14
        version = 14
    if version < 15:
        # v15 persists conservative repeat/barline semantic consensus plus
        # production patch-burden and transaction-rejection audit evidence.
        # PageInfo and quality-certificate defaults preserve historical jobs.
        result["schema_version"] = 15
        version = 15
    if version < 16:
        # v16 persists conservative simple-articulation consensus evidence and the
        # complete runtime model-resource audit. Defaults preserve historical jobs.
        result["schema_version"] = 16
        version = 16
    if version < 17:
        # v17 persists conservative simple-ornament consensus evidence. PageInfo
        # defaults preserve historical jobs without rewriting page arrays.
        result["schema_version"] = 17
        version = 17
    if version < 18:
        # v18 persists conservative simple dynamic/metronome consensus evidence.
        # PageInfo defaults preserve historical jobs without rewriting page arrays.
        result["schema_version"] = 18
        version = 18
    if version < 19:
        # v19 persists conservative simple grace/regular-note topology consensus
        # evidence. PageInfo defaults preserve historical jobs without rewriting.
        result["schema_version"] = 19
        version = 19
    if version < 20:
        # v20 persists conservative single-verse lyric consensus evidence.  The
        # repository also writes an independent state snapshot, while PageInfo
        # defaults keep historical jobs readable without rewriting page arrays.
        result["schema_version"] = 20
        version = 20
    if version < 21:
        # v21 persists interaction-aware local patch transaction evidence.  Existing
        # page reports remain readable through dataclass defaults.
        result["schema_version"] = 21
        version = 21
    if version < 22:
        # v22 persists source-backed hairpin transaction and abstention evidence.
        # PageInfo defaults keep historical tasks readable without rewriting pages.
        result["schema_version"] = 22
        version = 22
    if version < 23:
        # v23 persists source-proven nested-slur relation transactions.
        result["schema_version"] = 23
        version = 23
    if version < 24:
        # v24 persists the explicit production-release decision separately from
        # advisory quality labels, so the UI cannot mistake a completed result for
        # an automatically released result.
        result.setdefault("production_ready", False)
        result.setdefault("release_blockers", [])
        result["schema_version"] = 24
        version = 24
    if version < 25:
        # v25 persists user-selected output naming and PDF render detail so a
        # resumed conversion uses exactly the settings chosen at submission.
        result.setdefault("output_name", None)
        result.setdefault("pdf_dpi", 400)
        result["schema_version"] = 25
        version = 25
    if version != CURRENT_JOB_SCHEMA:
        raise UnsupportedJobSchema(f"无法迁移任务状态版本 {version}")
    return result
