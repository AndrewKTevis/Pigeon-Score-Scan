from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .state_schema import CURRENT_JOB_SCHEMA, migrate_job_payload
from .util import utc_now_iso


@dataclass
class PageInfo:
    index: int
    source_name: str
    image_path: str
    source_file_index: int = 0
    source_page_number: int = 1
    width: int = 0
    height: int = 0
    render_dpi: float | None = None
    sha256: str | None = None
    normalized_path: str | None = None
    layout_path: str | None = None
    overlay_path: str | None = None
    xml_path: str | None = None
    blur_score: float | None = None
    contrast_score: float | None = None
    skew_degrees: float | None = None
    orientation_degrees: int = 0
    orientation_probability: float | None = None
    orientation_margin: float | None = None
    orientation_applied: bool = False
    orientation_model_version: str | None = None
    orientation_model_status: str | None = None
    orientation_probabilities: dict[str, float] = field(default_factory=dict)
    quality_score: float | None = None
    staff_system_count: int = 0
    physical_staff_count: int = 0
    score_system_count: int = 0
    estimated_measure_count: int = 0
    resolved_measure_count: int = 0
    measure_count_probability: float | None = None
    measure_count_margin: float | None = None
    measure_count_source: str | None = None
    measure_count_model_version: str | None = None
    measure_count_report_path: str | None = None
    quality_notes: list[str] = field(default_factory=list)
    omr_status: str = "pending"
    omr_error: str | None = None
    ocr_mark_count: int = 0
    ocr_injected_count: int = 0
    ocr_review_candidate_count: int = 0
    ocr_anchor_mean_probability: float | None = None
    ocr_anchor_model_version: str | None = None
    ocr_anchor_model_status: str | None = None
    ocr_barline_anchor_count: int = 0
    ocr_rescaled_anchor_count: int = 0
    ocr_accelerator_requested: str | None = None
    ocr_accelerator_selected: str | None = None
    ocr_accelerator_verified: bool = False
    ocr_accelerator_fallback_reason: str | None = None
    ocr_component_providers: dict[str, list[str]] = field(default_factory=dict)
    semantic_detector_report_path: str | None = None
    semantic_detector_enabled: bool = False
    semantic_detector_status: str | None = None
    semantic_detector_model_version: str | None = None
    semantic_detector_detection_count: int = 0
    semantic_detector_accelerator_requested: str | None = None
    semantic_detector_accelerator_selected: str | None = None
    semantic_detector_accelerator_verified: bool = False
    semantic_detector_fallback_reason: str | None = None
    semantic_detector_providers: list[str] = field(default_factory=list)
    semantic_source_audit_report_path: str | None = None
    semantic_source_audit_status: str = "pending"
    semantic_source_audit_omission_count: int = 0
    semantic_source_audit_extraneous_count: int = 0
    semantic_source_audit_positional_mismatch_count: int = 0
    notation_coverage_report_path: str | None = None
    notation_coverage_status: str = "pending"
    notation_candidate_count: int = 0
    notation_potential_omission_count: int = 0
    notation_unbalanced_structure_count: int = 0
    notation_source_curved_connector_count: int = 0
    notation_source_wedge_count: int = 0
    wedge_enrichment_report_path: str | None = None
    wedge_transaction_committed: bool = False
    wedge_auto_injected_count: int = 0
    wedge_abstention_count: int = 0
    slur_relation_report_path: str | None = None
    slur_relation_transaction_committed: bool = False
    slur_relation_repaired_count: int = 0
    slur_relation_abstention_count: int = 0
    recognition_variant: str | None = None
    recognition_score: float | None = None
    recognition_candidates: list[dict[str, Any]] = field(default_factory=list)
    variant_plan_path: str | None = None
    variant_router_model: str | None = None
    variant_order: list[str] = field(default_factory=list)
    consensus_agreement: float | None = None
    consensus_exact_agreement: float | None = None
    consensus_semantic_agreement: float | None = None
    consensus_confidence: float | None = None
    consensus_measure_probability: float | None = None
    consensus_visual_probability: float | None = None
    consensus_event_probability: float | None = None
    consensus_context_probability: float | None = None
    consensus_ensemble_probability: float | None = None
    consensus_selection_risk_probability: float | None = None
    consensus_chord_patch_probability: float | None = None
    consensus_tuplet_patch_probability: float | None = None
    consensus_pitch_patch_probability: float | None = None
    consensus_rhythm_patch_probability: float | None = None
    consensus_event_kind_patch_probability: float | None = None
    consensus_attribute_patch_probability: float | None = None
    consensus_event_presence_patch_probability: float | None = None
    consensus_event_presence_visual_guard_probability: float | None = None
    consensus_cross_tie_patch_probability: float | None = None
    consensus_slur_patch_probability: float | None = None
    consensus_articulation_patch_probability: float | None = None
    consensus_ornament_patch_probability: float | None = None
    consensus_grace_patch_probability: float | None = None
    consensus_lyric_patch_probability: float | None = None
    consensus_direction_patch_probability: float | None = None
    consensus_barline_patch_probability: float | None = None
    consensus_replacements: int = 0
    consensus_chord_patch_measures: int = 0
    consensus_chord_patch_events: int = 0
    consensus_tuplet_patch_measures: int = 0
    consensus_tuplet_patch_events: int = 0
    consensus_tuplet_patch_groups: int = 0
    consensus_pitch_patch_measures: int = 0
    consensus_pitch_patch_events: int = 0
    consensus_rhythm_patch_measures: int = 0
    consensus_rhythm_patch_events: int = 0
    consensus_event_kind_patch_measures: int = 0
    consensus_event_kind_patch_events: int = 0
    consensus_attribute_patch_measures: int = 0
    consensus_attribute_patch_attributes: int = 0
    consensus_event_presence_patch_measures: int = 0
    consensus_event_presence_inserted_events: int = 0
    consensus_event_presence_deleted_events: int = 0
    consensus_event_presence_visual_guard_transactions: int = 0
    consensus_event_presence_visual_guard_rejections: int = 0
    consensus_event_presence_visual_guard_model: str | None = None
    consensus_event_presence_visual_guard_note_threshold: float | None = None
    consensus_event_presence_visual_guard_rest_threshold: float | None = None
    consensus_cross_tie_patch_boundaries: int = 0
    consensus_cross_tie_patch_endpoints: int = 0
    consensus_cross_tie_transaction_rejections: int = 0
    consensus_slur_patch_measures: int = 0
    consensus_slur_patch_events: int = 0
    consensus_slur_patch_arcs: int = 0
    consensus_articulation_patch_measures: int = 0
    consensus_articulation_patch_events: int = 0
    consensus_articulation_patch_marks: int = 0
    consensus_ornament_patch_measures: int = 0
    consensus_ornament_patch_events: int = 0
    consensus_ornament_patch_marks: int = 0
    consensus_grace_patch_measures: int = 0
    consensus_grace_patch_events: int = 0
    consensus_grace_patch_added: int = 0
    consensus_grace_patch_removed: int = 0
    consensus_lyric_patch_measures: int = 0
    consensus_lyric_patch_events: int = 0
    consensus_lyric_patch_lyrics: int = 0
    consensus_direction_patch_measures: int = 0
    consensus_direction_patch_directions: int = 0
    consensus_barline_patch_measures: int = 0
    consensus_barline_patch_locations: int = 0
    consensus_barline_patch_repeats: int = 0
    consensus_patch_transaction_rejections: int = 0
    consensus_disagreements: list[int] = field(default_factory=list)
    consensus_unresolved: list[int] = field(default_factory=list)
    consensus_report_path: str | None = None
    review_issue_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in vars(self).items()
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PageInfo":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass
class ReviewIssue:
    id: str
    page_index: int
    category: str
    title: str
    message: str
    severity: str = "review"
    crop_path: str | None = None
    raw_value: str | None = None
    suggested_value: str | None = None
    options: list[str] = field(default_factory=list)
    global_measure_number: int | None = None
    local_measure_index: int | None = None
    kind: str | None = None
    placement: str | None = None
    offset_ratio: float = 0.0
    status: str = "pending"
    resolution: str | None = None
    writeback_supported: bool = True
    requires_value: bool = True
    risk_preserved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in vars(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReviewIssue":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in payload.items() if key in allowed})


PersistHook = Callable[["JobState"], None]


@dataclass
class JobState:
    id: str
    root: Path
    mode: str
    source_files: list[str]
    output_name: str | None = None
    pdf_dpi: int = 400
    schema_version: int = CURRENT_JOB_SCHEMA
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    status: str = "queued"
    stage: str = "Waiting to start"
    progress: float = 0.0
    current_page: int = 0
    total_pages: int = 0
    pages: list[PageInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    result_musicxml: str | None = None
    result_mxl: str | None = None
    preview_svg: str | None = None
    report_path: str | None = None
    layout_report_path: str | None = None
    artifact_manifest_path: str | None = None
    artifact_bundle_id: str | None = None
    review_path: str | None = None
    review_issues: list[ReviewIssue] = field(default_factory=list)
    review_resolved_count: int = 0
    error: str | None = None
    quality_state: str = "processing"
    quality_score: float | None = None
    production_ready: bool = False
    release_blockers: list[str] = field(default_factory=list)
    resumable: bool = True
    revision: int = field(default=0, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    persist_hook: PersistHook | None = field(default=None, repr=False)
    change_condition: threading.Condition = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.change_condition = threading.Condition(self.lock)

    def _persist(self) -> None:
        hook = self.persist_hook
        if hook is not None:
            hook(self)

    def add_log(self, message: str) -> None:
        line = str(message).strip()
        if not line:
            return
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > 600:
                self.logs = self.logs[-600:]
            self.updated_at = utc_now_iso()
        self._persist()

    def add_warning(self, message: str) -> None:
        with self.change_condition:
            if message not in self.warnings:
                self.warnings.append(message)
                self.revision += 1
                self.change_condition.notify_all()
            self.updated_at = utc_now_iso()
        self._persist()

    def update(self, **changes: Any) -> None:
        with self.change_condition:
            for key, value in changes.items():
                setattr(self, key, value)
            self.progress = min(1.0, max(0.0, float(self.progress)))
            self.updated_at = utc_now_iso()
            self.revision += 1
            self.change_condition.notify_all()
        self._persist()

    def wait_for_revision(self, after: int, timeout: float) -> None:
        """Wait for a user-visible state change without busy polling."""

        with self.change_condition:
            if self.revision <= after:
                self.change_condition.wait(timeout=max(0.0, min(float(timeout), 20.0)))

    def request_cancel(self) -> None:
        self.cancel_event.set()
        self.update(stage="Cancelling", status="cancelling")

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def to_dict(
        self,
        include_logs: bool = True,
        *,
        include_pages: bool = True,
        include_review_issues: bool = True,
    ) -> dict[str, Any]:
        with self.lock:
            payload = {
                "id": self.id,
                "schema_version": self.schema_version,
                "root": str(self.root),
                "mode": self.mode,
                "source_files": list(self.source_files),
                "output_name": self.output_name,
                "pdf_dpi": self.pdf_dpi,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "current_page": self.current_page,
                "total_pages": self.total_pages,
                "pages": [page.to_dict() for page in self.pages] if include_pages else [],
                "page_summary": [
                    {
                        "index": page.index,
                        "omr_status": page.omr_status,
                        "quality_score": page.quality_score,
                    }
                    for page in self.pages
                ],
                "warnings": list(self.warnings),
                "result_musicxml": self.result_musicxml,
                "result_mxl": self.result_mxl,
                "preview_svg": self.preview_svg,
                "report_path": self.report_path,
                "layout_report_path": self.layout_report_path,
                "artifact_manifest_path": self.artifact_manifest_path,
                "artifact_bundle_id": self.artifact_bundle_id,
                "review_path": self.review_path,
                "review_issues": (
                    [issue.to_dict() for issue in self.review_issues]
                    if include_review_issues
                    else []
                ),
                "review_issue_count": len(self.review_issues),
                "review_resolved_count": self.review_resolved_count,
                "error": self.error,
                "quality_state": self.quality_state,
                "quality_score": self.quality_score,
                "production_ready": self.production_ready,
                "release_blocker_count": len(self.release_blockers),
                "release_blockers": list(self.release_blockers),
                "resumable": self.resumable,
                "revision": self.revision,
            }
            payload["logs"] = list(self.logs[-120:]) if include_logs else []
            return payload

    @classmethod
    def from_dict(cls, root: Path, payload: dict[str, Any]) -> "JobState":
        payload = migrate_job_payload(payload)
        state = cls(
            id=str(payload["id"]),
            root=root,
            mode=str(payload.get("mode", "images")),
            source_files=[str(item) for item in payload.get("source_files", [])],
            output_name=(str(payload["output_name"]) if payload.get("output_name") else None),
            pdf_dpi=int(payload.get("pdf_dpi", 400)),
            schema_version=int(payload.get("schema_version", CURRENT_JOB_SCHEMA)),
            created_at=str(payload.get("created_at", utc_now_iso())),
        )
        for field_name in (
            "updated_at", "status", "stage", "progress", "current_page", "total_pages",
            "warnings", "logs", "result_musicxml", "result_mxl", "preview_svg",
            "report_path", "layout_report_path", "artifact_manifest_path", "artifact_bundle_id", "review_path", "review_resolved_count", "error", "quality_state", "quality_score", "production_ready", "release_blockers", "resumable",
        ):
            if field_name in payload:
                setattr(state, field_name, payload[field_name])
        state.pages = [PageInfo.from_dict(item) for item in payload.get("pages", [])]
        state.review_issues = [ReviewIssue.from_dict(item) for item in payload.get("review_issues", [])]
        return state
