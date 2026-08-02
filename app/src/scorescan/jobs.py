from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .beam_enrichment import (
    BeamEnrichmentReport,
    enrich_musicxml_with_source_beams,
)
from .compatibility import validate_with_musescore
from .config import APP_NAME, APP_VERSION, WORKFLOW_VERSION, Settings
from .importers import classify_mode, copy_uploads, list_copied_inputs, prepare_pages
from .implicit_triplet_transaction import (
    apply_evidence_confirmed_continuous_triplet_grid,
)
from .integrity import build_bundle_integrity
from .layout import PageLayout, write_layout_artifacts
from .model_registry import audit_model_manifest, model_versions
from .models import JobState, PageInfo
from .notation_coverage import (
    NotationCoverageReport,
    audit_notation_coverage,
    detect_notation_candidates,
)
from .musicxml import (
    PageDocument,
    analyze_musicxml,
    canonicalize_multivoice_timelines,
    extract_title,
    merge_pages,
    normalize_single_voice_musicxml,
    package_mxl,
    parse_or_placeholder,
    validate_musicxml,
)
from .omr import HomrRunner
from .ornament_enrichment import (
    OrnamentEnrichmentReport,
    enrich_musicxml_with_source_ornaments,
)
from .recognition import RecognitionEnsemble
from .review import (
    build_consensus_review_issues,
    build_notation_coverage_review_issues,
    build_text_review_issues,
)
from .semantic_detector import (
    SemanticDetection,
    TEXT_REGION_CLASSES,
    corroborate_notation_candidates,
    semantic_detector_status,
)
from .semantic_source_audit import (
    SemanticSourceAuditReport,
    audit_semantic_source_symbols,
)
from .persistence import JobRepository
from .preview import render_preview
from .policy import DEFAULT_POLICY
from .quality import inspect_page
from .quality_certificate import build_quality_certificate
from .storage import require_workspace_capacity
from .slur_relation_repair import (
    SlurRelationRepairReport,
    repair_source_proven_nested_slurs,
)
from .text_enrichment import OcrMark, enrich_musicxml_with_ocr, marks_to_dicts, update_direction_in_musicxml
from .util import atomic_write_json, path_is_within, read_json, safe_filename, sha256_file, utc_now_iso
from .wedge_enrichment import WedgeEnrichmentReport, enrich_musicxml_with_wedges


class CancelledError(RuntimeError):
    pass


def _preview_page_artifacts(preview_path: Path | None) -> list[tuple[str, Path]]:
    if preview_path is None or preview_path.suffix.casefold() == ".svg":
        return []
    return [
        (f"preview_page_{index:04d}", page)
        for index, page in enumerate(sorted(preview_path.parent.glob("page_*.svg")), start=1)
        if page.is_file()
    ]


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.workspace = settings.workspace
        self.repository = JobRepository(self.workspace)
        self.jobs: dict[str, JobState] = {}
        self._lock = threading.RLock()
        self._pending_jobs: deque[str] = deque()
        self._pending_job_ids: set[str] = set()
        self._active_job_ids: set[str] = set()
        self._dispatcher_count = 0
        self._review_operation_lock = threading.RLock()
        self._cleanup_stale_incoming()
        recoverable: list[JobState] = []
        for state in self.repository.load_all():
            if self._expired_terminal_job(state):
                shutil.rmtree(state.root, ignore_errors=True)
                continue
            self.jobs[state.id] = state
            if state.status in {"queued", "running", "cancelling", "interrupted"}:
                state.status = "interrupted"
                state.stage = "Resuming previous conversion"
                state.error = None
                state.cancel_event.clear()
                self.repository.persist(state)
                recoverable.append(state)
        for state in recoverable:
            self._start_worker(state)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _expired_terminal_job(self, state: JobState) -> bool:
        if self.settings.job_retention_days <= 0:
            return False
        if state.status not in {"completed", "failed", "cancelled"}:
            return False
        updated = self._parse_timestamp(state.updated_at)
        if updated is None:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.job_retention_days)
        return updated < cutoff

    def _cleanup_stale_incoming(self) -> None:
        incoming = self.workspace / "incoming"
        if not incoming.exists():
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        for child in incoming.iterdir():
            try:
                modified = datetime.fromtimestamp(child.stat().st_mtime, timezone.utc)
            except OSError:
                continue
            if modified >= cutoff:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    def _start_worker(self, state: JobState) -> None:
        """Enqueue a task in deterministic FIFO order using a bounded worker pool."""

        with self._lock:
            if state.id in self._pending_job_ids or state.id in self._active_job_ids:
                return
            self._pending_jobs.append(state.id)
            self._pending_job_ids.add(state.id)
            target_count = min(
                max(1, int(self.settings.max_parallel_jobs)),
                len(self._active_job_ids) + len(self._pending_jobs),
            )
            while self._dispatcher_count < target_count:
                self._dispatcher_count += 1
                worker = threading.Thread(
                    target=self._worker_loop,
                    daemon=True,
                    name=f"scorescan-worker-{self._dispatcher_count}",
                )
                worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                if not self._pending_jobs:
                    self._dispatcher_count = max(0, self._dispatcher_count - 1)
                    return
                job_id = self._pending_jobs.popleft()
                self._pending_job_ids.discard(job_id)
                state = self.jobs.get(job_id)
                if state is None:
                    continue
                self._active_job_ids.add(job_id)
            try:
                self._run_job_guarded(state)
            finally:
                with self._lock:
                    self._active_job_ids.discard(job_id)

    def create_job(
        self,
        uploaded_paths: list[Path],
        source_names: list[str] | None = None,
        *,
        consume_uploads: bool = False,
        output_name: str | None = None,
        pdf_dpi: int | None = None,
    ) -> JobState:
        mode = classify_mode(uploaded_paths)
        input_bytes = sum(path.stat().st_size for path in uploaded_paths if path.exists())
        already_in_workspace = bool(uploaded_paths) and all(
            path_is_within(path.resolve(), self.workspace.resolve()) for path in uploaded_paths
        )
        require_workspace_capacity(
            self.settings,
            additional_bytes=0 if consume_uploads and already_in_workspace else input_bytes,
            context="创建识别任务",
        )
        job_id = uuid.uuid4().hex[:12]
        root = self.workspace / job_id
        try:
            copied = copy_uploads(uploaded_paths, root / "input", move=consume_uploads)
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        state = JobState(
            job_id,
            root,
            mode,
            list(source_names or [path.name for path in uploaded_paths]),
            output_name=output_name,
            pdf_dpi=int(pdf_dpi or self.settings.pdf_dpi),
        )
        state.persist_hook = self.repository.persist
        state.add_log(f"创建任务 {job_id}，输入 {len(copied)} 个文件")
        with self._lock:
            self.jobs[job_id] = state
        self.repository.persist(state)
        self._start_worker(state)
        return state

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self.jobs.get(job_id)

    def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            states = sorted(self.jobs.values(), key=lambda item: item.updated_at, reverse=True)[:limit]
        return [self.describe(state, compact=True, include_logs=False) for state in states]

    def describe(
        self,
        state: JobState,
        *,
        compact: bool = False,
        include_logs: bool = True,
    ) -> dict[str, Any]:
        payload = state.to_dict(
            include_logs=include_logs,
            include_pages=not compact,
            include_review_issues=not compact,
        )
        with self._lock:
            try:
                queue_position = list(self._pending_jobs).index(state.id) + 1
            except ValueError:
                queue_position = None
            payload["queue_position"] = queue_position
            payload["active_job_count"] = len(self._active_job_ids)
            payload["parallel_job_limit"] = max(1, int(self.settings.max_parallel_jobs))
        return payload

    def cancel(self, job_id: str) -> bool:
        state = self.get(job_id)
        if state is None or state.status not in {"queued", "running", "interrupted", "cancelling"}:
            return False
        state.request_cancel()
        return True

    def _check_cancel(self, job: JobState) -> None:
        if job.cancelled:
            raise CancelledError("任务已取消")

    def _load_layout(self, page: PageInfo) -> PageLayout | None:
        if not page.layout_path:
            return None
        payload = read_json(Path(page.layout_path))
        return PageLayout.from_dict(payload) if isinstance(payload, dict) else None

    def _run_job_guarded(self, job: JobState) -> None:
        try:
            self._check_cancel(job)
            self._run_job(job)
        except CancelledError:
            job.update(status="cancelled", stage="Cancelled", error=None, progress=1.0, quality_state="cancelled")
        except Exception as exc:
            job.add_log(traceback.format_exc())
            job.update(status="failed", stage="Conversion failed", error=str(exc), progress=1.0, quality_state="failed")

    def _run_job(self, job: JobState) -> None:
        job_settings = replace(self.settings, pdf_dpi=job.pdf_dpi)
        model_resource_audit = audit_model_manifest(self.settings.resources)
        if not model_resource_audit.verified:
            job.add_warning(
                f"模型资源完整性未通过：{model_resource_audit.verified_count}/"
                f"{model_resource_audit.expected_count} 项已验证；本次结果禁止生产自动放行"
            )
        semantic_asset_status = semantic_detector_status(self.settings.resources)
        if (
            not semantic_asset_status.enabled
            and semantic_asset_status.status != "asset_absent"
        ):
            job.add_warning(
                "语义记号检测器未获准启用："
                f"{semantic_asset_status.status}；已安全回退到源图几何审计"
            )
        job.update(status="running", stage="Preparing input pages", progress=max(job.progress, 0.01), error=None)
        copied_inputs = list_copied_inputs(job.root)
        if not copied_inputs:
            raise RuntimeError("任务输入文件已经丢失")

        if not job.pages:
            pages = prepare_pages(job, copied_inputs, job_settings)
            job.update(pages=pages, total_pages=len(pages), progress=0.06)
        pages = job.pages
        job.update(total_pages=len(pages))
        self._check_cancel(job)

        job.update(stage="Checking scans and score layout", progress=max(job.progress, 0.07))

        def prepare_page(page: PageInfo) -> PageInfo:
            if not page.normalized_path or not Path(page.normalized_path).exists():
                inspect_page(page, job.root / "pages" / "normalized")
            if not page.layout_path or not Path(page.layout_path).exists():
                write_layout_artifacts(page, job.root / "layout")
            return page

        largest_page_pixels = max((page.width * page.height for page in pages), default=0)
        preparation_workers = (
            1
            if largest_page_pixels > 30_000_000
            else min(2, max(1, len(pages)))
        )
        with ThreadPoolExecutor(
            max_workers=preparation_workers,
            thread_name_prefix="scorescan-page-check",
        ) as executor:
            futures = {executor.submit(prepare_page, page): page for page in pages}
            for completed, future in enumerate(as_completed(futures), start=1):
                self._check_cancel(job)
                page = future.result()
                for note in page.quality_notes:
                    job.add_warning(f"第 {page.index} 页：{note}")
                job.update(
                    stage=f"Checking scans and score layout ({completed} / {len(pages)})",
                    current_page=completed,
                    progress=0.07 + 0.13 * completed / max(len(pages), 1),
                )

        self._check_cancel(job)
        runner = HomrRunner(
            job.add_log,
            page_timeout_seconds=job_settings.page_timeout_seconds,
            settings=job_settings,
        )
        pending = [page for page in pages if page.omr_status not in {"completed", "fallback"}]
        if pending and runner.available():
            job.update(stage="Loading recognition models", progress=max(job.progress, 0.20))
            init_result = runner.initialize_models(job.cancel_event)
            if init_result.cancelled:
                raise CancelledError()
            if init_result.return_code != 0:
                job.add_warning(init_result.error or "识别模型初始化异常，将继续尝试逐页识别")
        elif pending:
            job.add_warning("没有检测到 homr 识别引擎；将生成可打开的保底 MusicXML")

        ocr_pages: list[dict[str, object]] = []
        ocr_marks_by_page: dict[int, list[object]] = {}
        notation_reports_by_page: dict[int, NotationCoverageReport] = {}
        semantic_audit_reports_by_page: dict[int, SemanticSourceAuditReport] = {}
        ornament_reports_by_page: dict[int, OrnamentEnrichmentReport] = {}
        beam_reports_by_page: dict[int, BeamEnrichmentReport] = {}
        wedge_reports_by_page: dict[int, WedgeEnrichmentReport] = {}
        slur_relation_reports_by_page: dict[int, SlurRelationRepairReport] = {}
        page_documents: list[PageDocument] = []
        ensemble = RecognitionEnsemble(runner, job.add_log, job.root / "recognition")
        for index, page in enumerate(pages, start=1):
            self._check_cancel(job)
            job.update(
                stage=f"Recognizing notes, rhythm, and notation — page {index} / {len(pages)}",
                current_page=index,
                progress=0.22 + 0.55 * (index - 1) / max(len(pages), 1),
            )
            normalized = Path(page.normalized_path or page.image_path)
            xml_path = Path(page.xml_path) if page.xml_path else normalized.with_suffix(".musicxml")
            layout = self._load_layout(page)
            if page.omr_status != "completed" or not xml_path.exists():
                page_count = max(len(pages), 1)
                page_start = 0.22 + 0.55 * (index - 1) / page_count
                page_span = 0.55 / page_count

                def update_recognition_progress(fraction: float, detail: str) -> None:
                    bounded = min(1.0, max(0.0, float(fraction)))
                    job.update(
                        stage=f"Page {index} / {len(pages)} — {detail}",
                        progress=page_start + page_span * bounded,
                    )

                result = ensemble.run_page(
                    page,
                    layout,
                    job.cancel_event,
                    progress_callback=update_recognition_progress,
                )
                if result.cancelled:
                    raise CancelledError()
                resolution = result.measure_count_resolution
                # A checkpoint may be reused after a crash between committing the page
                # XML and persisting the page audit.  Preserve already-persisted full
                # candidate evidence; otherwise keep the singleton checkpoint record so
                # the selective-release gate fails closed rather than inventing support.
                if not (
                    resolution is not None
                    and resolution.source == "checkpoint"
                    and page.recognition_candidates
                ):
                    page.recognition_candidates = [candidate.to_dict() for candidate in result.candidates]
                if resolution is not None:
                    page.resolved_measure_count = resolution.selected_count
                    page.measure_count_probability = resolution.probability
                    page.measure_count_margin = resolution.margin
                    page.measure_count_source = resolution.source
                    page.measure_count_model_version = resolution.model_version
                    page.measure_count_report_path = str(
                        job.root / "recognition" / f"page_{page.index:04d}" / "measure_count_resolution.json"
                    )
                    if resolution.selected_count != page.estimated_measure_count:
                        job.add_log(
                            f"第 {page.index} 页：小节数证据融合将布局估计 "
                            f"{page.estimated_measure_count} 调整为 {resolution.selected_count}"
                        )
                if result.consensus is not None:
                    page.consensus_agreement = result.consensus.agreement_ratio
                    page.consensus_exact_agreement = result.consensus.exact_agreement_ratio
                    page.consensus_semantic_agreement = result.consensus.semantic_agreement_ratio
                    page.consensus_confidence = result.consensus.mean_measure_confidence
                    page.consensus_measure_probability = result.consensus.mean_selected_measure_probability
                    page.consensus_visual_probability = result.consensus.mean_visual_probability
                    page.consensus_event_probability = result.consensus.mean_event_probability
                    page.consensus_context_probability = result.consensus.mean_context_probability
                    page.consensus_ensemble_probability = result.consensus.mean_ensemble_probability
                    page.consensus_selection_risk_probability = result.consensus.mean_selection_risk_probability
                    page.consensus_chord_patch_probability = result.consensus.mean_chord_patch_probability
                    page.consensus_tuplet_patch_probability = result.consensus.mean_tuplet_patch_probability
                    page.consensus_pitch_patch_probability = result.consensus.mean_pitch_patch_probability
                    page.consensus_rhythm_patch_probability = result.consensus.mean_rhythm_patch_probability
                    page.consensus_event_kind_patch_probability = result.consensus.mean_event_kind_patch_probability
                    page.consensus_attribute_patch_probability = result.consensus.mean_attribute_patch_probability
                    page.consensus_event_presence_patch_probability = result.consensus.mean_event_presence_patch_probability
                    page.consensus_event_presence_visual_guard_probability = result.consensus.mean_event_presence_visual_guard_probability
                    page.consensus_cross_tie_patch_probability = result.consensus.mean_cross_tie_patch_probability
                    page.consensus_slur_patch_probability = result.consensus.mean_slur_patch_probability
                    page.consensus_articulation_patch_probability = result.consensus.mean_articulation_patch_probability
                    page.consensus_ornament_patch_probability = result.consensus.mean_ornament_patch_probability
                    page.consensus_grace_patch_probability = result.consensus.mean_grace_patch_probability
                    page.consensus_lyric_patch_probability = result.consensus.mean_lyric_patch_probability
                    page.consensus_direction_patch_probability = result.consensus.mean_direction_patch_probability
                    page.consensus_barline_patch_probability = result.consensus.mean_barline_patch_probability
                    page.consensus_replacements = result.consensus.replacements
                    page.consensus_chord_patch_measures = result.consensus.chord_patch_measure_count
                    page.consensus_chord_patch_events = result.consensus.chord_patch_event_count
                    page.consensus_tuplet_patch_measures = result.consensus.tuplet_patch_measure_count
                    page.consensus_tuplet_patch_events = result.consensus.tuplet_patch_event_count
                    page.consensus_tuplet_patch_groups = result.consensus.tuplet_patch_group_count
                    page.consensus_pitch_patch_measures = result.consensus.pitch_patch_measure_count
                    page.consensus_pitch_patch_events = result.consensus.pitch_patch_event_count
                    page.consensus_rhythm_patch_measures = result.consensus.rhythm_patch_measure_count
                    page.consensus_rhythm_patch_events = result.consensus.rhythm_patch_event_count
                    page.consensus_event_kind_patch_measures = result.consensus.event_kind_patch_measure_count
                    page.consensus_event_kind_patch_events = result.consensus.event_kind_patch_event_count
                    page.consensus_attribute_patch_measures = result.consensus.attribute_patch_measure_count
                    page.consensus_attribute_patch_attributes = result.consensus.attribute_patch_attribute_count
                    page.consensus_event_presence_patch_measures = result.consensus.event_presence_patch_measure_count
                    page.consensus_event_presence_inserted_events = result.consensus.event_presence_patch_inserted_event_count
                    page.consensus_event_presence_deleted_events = result.consensus.event_presence_patch_deleted_event_count
                    page.consensus_event_presence_visual_guard_transactions = result.consensus.event_presence_visual_guard_transaction_count
                    page.consensus_event_presence_visual_guard_rejections = result.consensus.event_presence_visual_guard_rejected_count
                    page.consensus_event_presence_visual_guard_model = result.consensus.event_presence_visual_guard_model
                    page.consensus_event_presence_visual_guard_note_threshold = result.consensus.event_presence_visual_guard_note_threshold
                    page.consensus_event_presence_visual_guard_rest_threshold = result.consensus.event_presence_visual_guard_rest_threshold
                    page.consensus_cross_tie_patch_boundaries = result.consensus.cross_tie_patch_boundary_count
                    page.consensus_cross_tie_patch_endpoints = result.consensus.cross_tie_patch_endpoint_count
                    page.consensus_cross_tie_transaction_rejections = result.consensus.cross_tie_patch_transaction_rejected_count
                    page.consensus_slur_patch_measures = result.consensus.slur_patch_measure_count
                    page.consensus_slur_patch_events = result.consensus.slur_patch_event_count
                    page.consensus_slur_patch_arcs = result.consensus.slur_patch_arc_count
                    page.consensus_articulation_patch_measures = result.consensus.articulation_patch_measure_count
                    page.consensus_articulation_patch_events = result.consensus.articulation_patch_event_count
                    page.consensus_articulation_patch_marks = result.consensus.articulation_patch_mark_count
                    page.consensus_ornament_patch_measures = result.consensus.ornament_patch_measure_count
                    page.consensus_ornament_patch_events = result.consensus.ornament_patch_event_count
                    page.consensus_ornament_patch_marks = result.consensus.ornament_patch_mark_count
                    page.consensus_grace_patch_measures = result.consensus.grace_patch_measure_count
                    page.consensus_grace_patch_events = result.consensus.grace_patch_event_count
                    page.consensus_grace_patch_added = result.consensus.grace_patch_added_count
                    page.consensus_grace_patch_removed = result.consensus.grace_patch_removed_count
                    page.consensus_lyric_patch_measures = result.consensus.lyric_patch_measure_count
                    page.consensus_lyric_patch_events = result.consensus.lyric_patch_event_count
                    page.consensus_lyric_patch_lyrics = result.consensus.lyric_patch_lyric_count
                    page.consensus_direction_patch_measures = result.consensus.direction_patch_measure_count
                    page.consensus_direction_patch_directions = result.consensus.direction_patch_direction_count
                    page.consensus_barline_patch_measures = result.consensus.barline_patch_measure_count
                    page.consensus_barline_patch_locations = result.consensus.barline_patch_location_count
                    page.consensus_barline_patch_repeats = result.consensus.barline_patch_repeat_count
                    page.consensus_patch_transaction_rejections = result.consensus.patch_transaction_rejected_count
                    page.consensus_disagreements = list(result.consensus.disagreement_measure_indices)
                    page.consensus_unresolved = list(result.consensus.unresolved_measure_indices)
                    page.consensus_report_path = str(job.root / "recognition" / f"page_{page.index:04d}" / "consensus.json")
                    semantic_stability = (
                        result.consensus.semantic_agreement_ratio >= 0.995
                        and result.consensus.mean_measure_confidence >= 0.98
                        and not result.consensus.unresolved_measure_indices
                    )
                    if result.consensus.disagreement_measure_indices and not semantic_stability:
                        job.add_warning(
                            f"第 {page.index} 页：{len(result.consensus.disagreement_measure_indices)} 个小节在不同图像候选间存在分歧；"
                            f"其中 {len(result.consensus.unresolved_measure_indices)} 个没有严格多数"
                        )
                    elif result.consensus.disagreement_measure_indices:
                        job.add_log(
                            f"第 {page.index} 页：审计保留 "
                            f"{len(result.consensus.disagreement_measure_indices)} 个候选编码差异，"
                            "演奏语义共识与小节置信度已通过"
                        )
                if result.selected and result.selected.xml_path and Path(result.selected.xml_path).exists():
                    xml_path = Path(result.selected.xml_path)
                    page.xml_path = str(xml_path)
                    page.omr_status = "completed"
                    page.omr_error = None
                    page.recognition_variant = result.selected.variant
                    page.recognition_score = result.selected.score
                    normalization = normalize_single_voice_musicxml(xml_path)
                    if any(normalization.values()):
                        job.add_log(f"第 {page.index} 页：规范化 MusicXML {normalization}")
                    timeline_repair = canonicalize_multivoice_timelines(xml_path)
                    if timeline_repair["repaired_count"]:
                        job.add_log(
                            f"第 {page.index} 页：安全重建 "
                            f"{timeline_repair['repaired_count']} 个多声部小节时间轴"
                        )
                    if timeline_repair["abstained_count"]:
                        job.add_warning(
                            f"第 {page.index} 页："
                            f"{timeline_repair['abstained_count']} 个超拍多声部小节无法无歧义重建"
                        )
                    ranked = sorted(result.candidates, key=lambda item: item.score, reverse=True)
                    if len(ranked) > 1 and abs(ranked[0].score - ranked[1].score) < 18:
                        semantic_stability = (
                            result.consensus is not None
                            and result.consensus.semantic_agreement_ratio >= 0.995
                            and result.consensus.mean_measure_confidence >= 0.98
                            and not result.consensus.unresolved_measure_indices
                        )
                        if semantic_stability:
                            job.add_log(
                                f"第 {page.index} 页：候选得分接近，"
                                "但演奏语义共识与小节置信度已通过"
                            )
                        else:
                            job.add_warning(
                                f"第 {page.index} 页：多个识别候选得分接近，建议检查转换报告"
                            )
                else:
                    page.omr_status = "fallback"
                    page.omr_error = result.error or "本页识别未完成"
                    job.add_warning(f"第 {page.index} 页：{page.omr_error}；已保留分页并继续")

            semantic_detections: tuple[SemanticDetection, ...] = ()
            if (
                semantic_asset_status.enabled
                and page.omr_status == "completed"
                and xml_path.exists()
                and layout is not None
            ):
                run_semantic = getattr(runner, "run_semantic_detection", None)
                if callable(run_semantic):
                    semantic_result = run_semantic(
                        normalized,
                        layout,
                        job.cancel_event,
                    )
                    if semantic_result.cancelled:
                        raise CancelledError()
                    page.semantic_detector_enabled = semantic_result.model_enabled
                    page.semantic_detector_status = semantic_result.model_status
                    page.semantic_detector_model_version = semantic_result.model_version
                    page.semantic_detector_accelerator_requested = (
                        semantic_result.requested_accelerator
                    )
                    page.semantic_detector_accelerator_selected = (
                        semantic_result.selected_accelerator
                    )
                    page.semantic_detector_accelerator_verified = (
                        semantic_result.runtime_verified
                    )
                    page.semantic_detector_fallback_reason = (
                        semantic_result.fallback_reason
                    )
                    page.semantic_detector_providers = list(semantic_result.providers)
                    if semantic_result.return_code == 0 and semantic_result.model_enabled:
                        try:
                            semantic_detections = tuple(
                                SemanticDetection.from_dict(dict(payload))
                                for payload in semantic_result.detections
                            )
                        except Exception as exc:
                            semantic_detections = ()
                            page.semantic_detector_enabled = False
                            page.semantic_detector_status = (
                                f"invalid_worker_payload:{type(exc).__name__}:{exc}"
                            )
                    page.semantic_detector_detection_count = len(semantic_detections)
                    semantic_path = (
                        job.root
                        / "recognition"
                        / f"page_{page.index:04d}"
                        / "semantic_detector.json"
                    )
                    atomic_write_json(
                        semantic_path,
                        {
                            "format": 1,
                            "return_code": semantic_result.return_code,
                            "enabled": page.semantic_detector_enabled,
                            "status": page.semantic_detector_status,
                            "model_version": page.semantic_detector_model_version,
                            "accelerator": {
                                "requested": (
                                    page.semantic_detector_accelerator_requested
                                ),
                                "selected": page.semantic_detector_accelerator_selected,
                                "verified": page.semantic_detector_accelerator_verified,
                                "providers": page.semantic_detector_providers,
                                "fallback_reason": (
                                    page.semantic_detector_fallback_reason
                                ),
                            },
                            "scale": semantic_result.scale,
                            "tile_count": semantic_result.tile_count,
                            "detections": [
                                item.to_dict() for item in semantic_detections
                            ],
                            "error": semantic_result.error,
                        },
                    )
                    page.semantic_detector_report_path = str(semantic_path)
                    if semantic_result.return_code != 0:
                        job.add_warning(
                            f"第 {page.index} 页：语义记号检测失败，"
                            "已安全回退到源图几何审计："
                            f"{semantic_result.error or '未知错误'}"
                        )

            if (
                page.omr_status == "completed"
                and xml_path.exists()
                and layout is not None
                and page.semantic_detector_enabled
                and page.semantic_detector_accelerator_verified
            ):
                try:
                    beam_report = enrich_musicxml_with_source_beams(
                        xml_path,
                        layout,
                        semantic_detections,
                    )
                    beam_reports_by_page[page.index] = beam_report
                    beam_path = (
                        job.root
                        / "recognition"
                        / f"page_{page.index:04d}"
                        / "source_beams.json"
                    )
                    atomic_write_json(beam_path, beam_report.to_dict())
                    if beam_report.injected_segment_count:
                        job.add_log(
                            f"第 {page.index} 页：依据高精度源图检测，"
                            f"事务式恢复 {beam_report.injected_segment_count} 个连梁段、"
                            f"{beam_report.injected_marker_count} 个连梁关系标记"
                        )
                except Exception as exc:
                    job.add_warning(
                        f"第 {page.index} 页：源图连梁关系恢复失败，"
                        f"已保持原 MusicXML 不变：{exc}"
                    )

            semantic_text_regions = tuple(
                item.to_dict()
                for item in semantic_detections
                if item.class_name in TEXT_REGION_CLASSES
            )
            job.update(stage=f"Recognizing tempo, dynamics, and text — page {index} / {len(pages)}")
            if page.omr_status == "completed" and xml_path.exists():
                run_isolated_ocr = getattr(runner, "run_ocr_enrichment", None)
                if callable(run_isolated_ocr) and layout is not None:
                    if semantic_text_regions:
                        ocr_result = run_isolated_ocr(
                            normalized,
                            xml_path,
                            layout,
                            job.cancel_event,
                            semantic_text_regions=semantic_text_regions,
                        )
                    else:
                        ocr_result = run_isolated_ocr(
                            normalized,
                            xml_path,
                            layout,
                            job.cancel_event,
                        )
                    if ocr_result.cancelled:
                        raise CancelledError()
                    page.ocr_accelerator_requested = ocr_result.requested_accelerator
                    page.ocr_accelerator_selected = ocr_result.selected_accelerator
                    page.ocr_accelerator_verified = ocr_result.runtime_verified
                    page.ocr_accelerator_fallback_reason = ocr_result.fallback_reason
                    page.ocr_component_providers = dict(
                        ocr_result.component_providers or {}
                    )
                    if ocr_result.return_code == 0:
                        marks = [
                            OcrMark.from_dict(payload)
                            for payload in ocr_result.marks
                        ]
                        ocr_warnings = list(ocr_result.warnings)
                    else:
                        marks = []
                        ocr_warnings = [
                            "隔离的文字识别进程失败，未向 MusicXML 写入未经验证的文字："
                            f"{ocr_result.error or '未知错误'}"
                        ]
                else:
                    # Compatibility path for embedders and unit-test runners that
                    # predate the isolated OCR worker.
                    if semantic_text_regions:
                        marks, ocr_warnings = enrich_musicxml_with_ocr(
                            normalized,
                            xml_path,
                            layout,
                            semantic_text_regions=semantic_text_regions,
                        )
                    else:
                        marks, ocr_warnings = enrich_musicxml_with_ocr(
                            normalized,
                            xml_path,
                            layout,
                        )
                    page.ocr_accelerator_requested = "cpu"
                    page.ocr_accelerator_selected = "cpu"
                    page.ocr_accelerator_verified = False
            else:
                # Text enrichment writes into MusicXML.  A failed OMR page has no
                # document to enrich and must never turn a clean recognition failure
                # into a misleading OCR/parser exception.
                marks, ocr_warnings = [], []
            page.ocr_mark_count = len(marks)
            page.ocr_injected_count = sum(bool(mark.injected) for mark in marks)
            page.ocr_review_candidate_count = sum(
                mark.kind in {"dynamic", "direction", "metronome", "text"} and not mark.injected
                for mark in marks
            )
            anchor_probabilities = [float(mark.musical_direction_probability) for mark in marks]
            page.ocr_anchor_mean_probability = (
                sum(anchor_probabilities) / len(anchor_probabilities) if anchor_probabilities else None
            )
            page.ocr_anchor_model_version = next(
                (mark.direction_anchor_model_version for mark in marks if mark.direction_anchor_model_version),
                None,
            )
            page.ocr_anchor_model_status = next(
                (mark.direction_anchor_model_status for mark in marks if mark.direction_anchor_model_status),
                None,
            )
            page.ocr_barline_anchor_count = sum(mark.measure_anchor_method == "barline_exact" for mark in marks)
            page.ocr_rescaled_anchor_count = sum(mark.measure_anchor_method == "barline_rescaled" for mark in marks)
            for warning in ocr_warnings:
                job.add_warning(f"第 {page.index} 页：{warning}")
            ocr_marks_by_page[page.index] = marks
            ocr_pages.append(
                {
                    "page": page.index,
                    "accelerator": {
                        "requested": page.ocr_accelerator_requested,
                        "selected": page.ocr_accelerator_selected,
                        "verified": page.ocr_accelerator_verified,
                        "fallback_reason": page.ocr_accelerator_fallback_reason,
                        "component_providers": page.ocr_component_providers,
                    },
                    "marks": marks_to_dicts(marks),
                }
            )

            if page.omr_status == "completed" and xml_path.exists() and layout is not None:
                try:
                    ornament_report = enrich_musicxml_with_source_ornaments(
                        normalized,
                        xml_path,
                        layout,
                    )
                    ornament_reports_by_page[page.index] = ornament_report
                    ornament_path = (
                        job.root
                        / "recognition"
                        / f"page_{page.index:04d}"
                        / "source_ornaments.json"
                    )
                    atomic_write_json(ornament_path, ornament_report.to_dict())
                    if (
                        ornament_report.inserted_mordent_count
                        or ornament_report.inserted_trill_count
                    ):
                        job.add_log(
                            f"第 {page.index} 页：依据源扫描形状定位 "
                            f"{ornament_report.inserted_mordent_count} 个波音和 "
                            f"{ornament_report.inserted_trill_count} 个颤音记号"
                        )
                except Exception as exc:
                    job.add_warning(f"第 {page.index} 页：源扫描装饰音审计失败：{exc}")

            notation_candidates = None
            if page.omr_status == "completed" and xml_path.exists() and layout is not None:
                try:
                    notation_candidates = detect_notation_candidates(normalized, layout)
                    if semantic_detections:
                        notation_candidates = corroborate_notation_candidates(
                            notation_candidates,
                            semantic_detections,
                            layout,
                        )
                except Exception as exc:
                    job.add_warning(f"第 {page.index} 页：源扫描记号检测失败：{exc}")

            if (
                page.omr_status == "completed"
                and xml_path.exists()
                and layout is not None
                and notation_candidates is not None
            ):
                try:
                    wedge_report = enrich_musicxml_with_wedges(
                        normalized,
                        xml_path,
                        layout,
                        candidates=notation_candidates,
                    )
                    wedge_path = (
                        job.root
                        / "recognition"
                        / f"page_{page.index:04d}"
                        / "wedge_enrichment.json"
                    )
                    atomic_write_json(wedge_path, wedge_report.to_dict())
                    wedge_reports_by_page[page.index] = wedge_report
                    page.wedge_enrichment_report_path = str(wedge_path)
                    page.wedge_transaction_committed = wedge_report.transaction_committed
                    page.wedge_auto_injected_count = wedge_report.injected_count
                    page.wedge_abstention_count = wedge_report.abstention_count
                    if wedge_report.injected_count:
                        job.add_log(
                            f"第 {page.index} 页：依据源扫描几何与完整谱表拓扑，"
                            f"事务式补写 {wedge_report.injected_count} 个渐强/渐弱发夹"
                        )
                except Exception as exc:
                    page.wedge_enrichment_report_path = None
                    page.wedge_transaction_committed = False
                    page.wedge_auto_injected_count = 0
                    job.add_warning(f"第 {page.index} 页：发夹关系审计失败：{exc}")

            if (
                page.omr_status == "completed"
                and xml_path.exists()
                and layout is not None
                and notation_candidates is not None
            ):
                try:
                    slur_report = repair_source_proven_nested_slurs(
                        normalized,
                        xml_path,
                        layout,
                        candidates=notation_candidates,
                    )
                    slur_path = (
                        job.root
                        / "recognition"
                        / f"page_{page.index:04d}"
                        / "slur_relation_repair.json"
                    )
                    atomic_write_json(slur_path, slur_report.to_dict())
                    slur_relation_reports_by_page[page.index] = slur_report
                    page.slur_relation_report_path = str(slur_path)
                    page.slur_relation_transaction_committed = (
                        slur_report.transaction_committed
                    )
                    page.slur_relation_repaired_count = slur_report.repaired_count
                    page.slur_relation_abstention_count = slur_report.abstention_count
                    if slur_report.repaired_count:
                        job.add_log(
                            f"第 {page.index} 页：源图双弧证据确认并修复 "
                            f"{slur_report.repaired_count} 个嵌套连奏线编号关系"
                        )
                except Exception as exc:
                    page.slur_relation_report_path = None
                    page.slur_relation_transaction_committed = False
                    page.slur_relation_repaired_count = 0
                    job.add_warning(f"第 {page.index} 页：连奏线关系审计失败：{exc}")

            if (
                page.omr_status == "completed"
                and xml_path.exists()
                and layout is not None
                and notation_candidates is not None
            ):
                try:
                    notation_report = audit_notation_coverage(
                        normalized,
                        xml_path,
                        layout,
                        candidates=notation_candidates,
                    )
                    notation_path = (
                        job.root
                        / "recognition"
                        / f"page_{page.index:04d}"
                        / "notation_coverage.json"
                    )
                    atomic_write_json(notation_path, notation_report.to_dict())
                    notation_reports_by_page[page.index] = notation_report
                    page.notation_coverage_report_path = str(notation_path)
                    page.notation_coverage_status = "completed"
                    page.notation_candidate_count = len(notation_report.candidates)
                    page.notation_potential_omission_count = notation_report.potential_omission_count
                    page.notation_unbalanced_structure_count = (
                        notation_report.emitted_unbalanced_slurs
                        + notation_report.emitted_unbalanced_ties
                        + notation_report.emitted_unbalanced_wedges
                    )
                    coverage_kinds = {item.kind: item for item in notation_report.kinds}
                    page.notation_source_curved_connector_count = coverage_kinds[
                        "curved_connector"
                    ].confident_source_count
                    page.notation_source_wedge_count = (
                        coverage_kinds["crescendo"].confident_source_count
                        + coverage_kinds["diminuendo"].confident_source_count
                    )
                    if notation_report.severe_structure_issue_count:
                        job.add_warning(
                            f"第 {page.index} 页：源扫描记号覆盖审计发现 "
                            f"{notation_report.potential_omission_count} 个潜在静默漏识别，"
                            f"{notation_report.emitted_unbalanced_slurs + notation_report.emitted_unbalanced_ties + notation_report.emitted_unbalanced_wedges} "
                            "个未闭合结构"
                        )
                except Exception as exc:
                    page.notation_coverage_status = "failed"
                    job.add_warning(f"第 {page.index} 页：源扫描记号覆盖审计失败：{exc}")
            elif page.omr_status == "completed":
                page.notation_coverage_status = "failed"
            else:
                page.notation_coverage_status = "not_applicable"

            if (
                page.omr_status == "completed"
                and xml_path.exists()
                and layout is not None
                and page.semantic_detector_enabled
                and page.semantic_detector_accelerator_verified
            ):
                try:
                    semantic_audit = audit_semantic_source_symbols(
                        xml_path,
                        layout,
                        semantic_detections,
                    )
                    semantic_audit_path = (
                        job.root
                        / "recognition"
                        / f"page_{page.index:04d}"
                        / "semantic_source_audit.json"
                    )
                    atomic_write_json(
                        semantic_audit_path,
                        semantic_audit.to_dict(),
                    )
                    semantic_audit_reports_by_page[page.index] = semantic_audit
                    page.semantic_source_audit_report_path = str(
                        semantic_audit_path
                    )
                    page.semantic_source_audit_status = semantic_audit.status
                    page.semantic_source_audit_omission_count = (
                        semantic_audit.omission_count
                    )
                    page.semantic_source_audit_extraneous_count = (
                        semantic_audit.extraneous_count
                    )
                    page.semantic_source_audit_positional_mismatch_count = (
                        semantic_audit.positional_mismatch_count
                    )
                    if semantic_audit.positional_mismatch_count:
                        job.add_warning(
                            f"第 {page.index} 页：源扫描语义符号审计发现 "
                            f"{semantic_audit.positional_mismatch_count} 个位置级不一致"
                        )
                except Exception as exc:
                    page.semantic_source_audit_status = "failed"
                    job.add_warning(
                        f"第 {page.index} 页：源扫描语义符号审计失败：{exc}"
                    )
            elif page.omr_status == "completed":
                page.semantic_source_audit_status = "not_available"
            else:
                page.semantic_source_audit_status = "not_applicable"

            if page.omr_status == "completed" and xml_path.exists():
                tree = parse_or_placeholder(xml_path, page.index, None)
            else:
                tree = parse_or_placeholder(None, page.index, page.omr_error)
            page_documents.append(PageDocument(tree, page.width, page.height, layout))
            job.update(progress=0.22 + 0.55 * index / max(len(pages), 1))

        self._check_cancel(job)
        if not any(page.omr_status == "completed" for page in pages):
            # A placeholder-only file is useful inside a partial multi-page result,
            # but it is not a conversion.  Fail closed when the engine recognized no
            # page at all so the UI cannot present an empty shell as downloadable work.
            raise RuntimeError("没有任何页面识别成功；请检查扫描质量、页面方向和适用范围")
        job.update(stage="Merging pages and preserving layout", progress=0.80)
        output_dir = job.root / "result"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "converted_score.musicxml"
        mxl_path = output_dir / "converted_score.mxl"
        summary = merge_pages(page_documents, output_path)
        triplet_output = output_dir / ".converted_score.triplet.musicxml"
        triplet_report_path = output_dir / "implicit_triplet_transaction.json"
        try:
            triplet_evidence, triplet_transaction = (
                apply_evidence_confirmed_continuous_triplet_grid(
                    output_path,
                    triplet_output,
                )
            )
            transaction_payload = {
                "format": 1,
                "evidence": triplet_evidence.to_dict(),
                "transaction": triplet_transaction.to_dict(),
            }
            if triplet_transaction.applied:
                transaction_errors = validate_musicxml(triplet_output)
                if transaction_errors:
                    transaction_payload["committed"] = False
                    transaction_payload["validation_errors"] = transaction_errors
                    job.add_warning(
                        "连续三连音事务未通过结构验证，已保留原识别结果"
                    )
                else:
                    triplet_output.replace(output_path)
                    transaction_payload["committed"] = True
                    summary["implicit_triplet_transaction"] = (
                        triplet_transaction.to_dict()
                    )
                    job.add_log(
                        "全曲节奏证据确认切分拍连续三连音："
                        f"已修复 {triplet_transaction.measures_repaired} 个小节、"
                        f"{triplet_transaction.notes_converted} 个误写时值，"
                        f"补回 {triplet_transaction.notes_cloned} 个重合声部事件"
                    )
            else:
                transaction_payload["committed"] = False
            atomic_write_json(triplet_report_path, transaction_payload)
        except Exception as exc:
            atomic_write_json(
                triplet_report_path,
                {
                    "format": 1,
                    "committed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            job.add_warning(
                "连续三连音安全事务执行失败，已保留原识别结果"
            )
        finally:
            triplet_output.unlink(missing_ok=True)
        package_mxl(output_path, mxl_path)

        page_measure_offsets: dict[int, int] = {}
        running_measure_offset = 0
        for item in summary.get("page_summaries", []):
            page_number = int(item.get("page", 0))
            page_measure_offsets[page_number] = running_measure_offset
            running_measure_offset += int(item.get("measures", 0))
        text_review_issues = build_text_review_issues(
            pages,
            ocr_marks_by_page,  # type: ignore[arg-type]
            page_measure_offsets,
            output_dir,
        )
        consensus_review_issues = build_consensus_review_issues(
            pages,
            page_measure_offsets,
            output_dir,
        )
        notation_review_issues = build_notation_coverage_review_issues(
            pages,
            notation_reports_by_page,
            output_dir,
        )
        review_issues = [
            *text_review_issues,
            *consensus_review_issues,
            *notation_review_issues,
        ]
        review_path = output_dir / "review_issues.json"
        atomic_write_json(review_path, {"issues": [issue.to_dict() for issue in review_issues]})
        notation_coverage_path = output_dir / "notation_coverage.json"
        atomic_write_json(
            notation_coverage_path,
            {
                "format": 1,
                "pages": [
                    {
                        "page": page_index,
                        **report.to_dict(),
                    }
                    for page_index, report in sorted(notation_reports_by_page.items())
                ],
            },
        )

        job.update(stage="Validating MusicXML and layout", progress=0.90)
        validation_errors = validate_musicxml(output_path)
        semantic_analysis = analyze_musicxml(output_path)
        preview_path, preview_warnings = render_preview(output_path, output_dir / "preview")
        compatibility = validate_with_musescore(output_path, output_dir / "compatibility")
        for warning in preview_warnings:
            job.add_warning(warning)
        if compatibility.checked and compatibility.success is False:
            job.add_warning(compatibility.message or "MuseScore 本机导入验证失败")
        if validation_errors:
            for error in validation_errors:
                job.add_warning(error)
        if semantic_analysis.get("page_count") != len(pages):
            job.add_warning("输出分页数量与输入页面数量不一致")
        rhythm_issues = semantic_analysis.get("rhythm_issues", [])
        tie_issues = semantic_analysis.get("tie_issues", [])
        slur_issues = semantic_analysis.get("slur_issues", [])
        if rhythm_issues:
            job.add_warning(f"发现 {len(rhythm_issues)} 个小节时值疑点；可能包含弱起或识别错误")
        if tie_issues:
            job.add_warning(f"发现 {len(tie_issues)} 个延音线结构疑点")
        if slur_issues:
            job.add_warning(f"发现 {len(slur_issues)} 个连音线结构疑点")

        fallback_pages = [page.index for page in pages if page.omr_status == "fallback"]
        quality_certificate = build_quality_certificate(
            pages,
            validation_errors,
            semantic_analysis,
            review_issues,
            fallback_pages,
            model_resource_audit.to_dict(),
        )
        quality_state = quality_certificate.state
        if quality_state == "verified":
            recovered_low_resolution_pages = {
                page.index
                for page in pages
                if page.consensus_semantic_agreement is not None
                and page.consensus_semantic_agreement >= 0.995
                and page.consensus_confidence is not None
                and page.consensus_confidence >= 0.98
                and page.notation_coverage_status == "completed"
                and page.notation_potential_omission_count == 0
                and not page.consensus_unresolved
            }
            retained_warnings = [
                warning
                for warning in job.warnings
                if not any(
                    warning.startswith(f"第 {page_index} 页：页面像素尺寸偏低")
                    for page_index in recovered_low_resolution_pages
                )
            ]
            if len(retained_warnings) != len(job.warnings):
                job.add_log(
                    "低分辨率提示已由多候选语义共识与源记号覆盖审计消解"
                )
                job.update(warnings=retained_warnings)

        title = extract_title(output_path)
        manifest = {
            "application": APP_NAME,
            "version": APP_VERSION,
            "workflow": WORKFLOW_VERSION,
            "created_at": job.created_at,
            "completed_at": utc_now_iso(),
            "job_id": job.id,
            "mode": job.mode,
            "source_files": job.source_files,
            "input_sha256": [sha256_file(path) for path in copied_inputs],
            "title": title,
            "output": output_path.name,
            "output_mxl": mxl_path.name,
            "output_sha256": sha256_file(output_path),
            "output_mxl_sha256": sha256_file(mxl_path),
            "quality_state": quality_state,
            "quality_certificate": quality_certificate.to_dict(),
            "summary": summary,
            "validation": {
                "xml_errors": validation_errors,
                "semantic": semantic_analysis,
                "musescore": {
                    "checked": compatibility.checked,
                    "success": compatibility.success,
                    "executable": compatibility.executable,
                    "message": compatibility.message,
                },
            },
            "fallback_pages": fallback_pages,
            "measure_count_resolution": {
                "pages": sum(page.resolved_measure_count > 0 for page in pages),
                "layout_adjustments": sum(
                    page.resolved_measure_count > 0
                    and page.resolved_measure_count != page.estimated_measure_count
                    for page in pages
                ),
                "mean_probability": (
                    sum(page.measure_count_probability or 0.0 for page in pages if page.measure_count_probability is not None)
                    / max(sum(page.measure_count_probability is not None for page in pages), 1)
                ),
                "mean_margin": (
                    sum(page.measure_count_margin or 0.0 for page in pages if page.measure_count_margin is not None)
                    / max(sum(page.measure_count_margin is not None for page in pages), 1)
                ),
            },
            "recognition_consensus": {
                "pages_with_consensus": sum(page.consensus_agreement is not None for page in pages),
                "mean_agreement_ratio": (
                    sum(page.consensus_agreement or 0.0 for page in pages if page.consensus_agreement is not None)
                    / max(sum(page.consensus_agreement is not None for page in pages), 1)
                ),
                "mean_exact_agreement": (
                    sum(page.consensus_exact_agreement or 0.0 for page in pages if page.consensus_exact_agreement is not None)
                    / max(sum(page.consensus_exact_agreement is not None for page in pages), 1)
                ),
                "mean_semantic_agreement": (
                    sum(page.consensus_semantic_agreement or 0.0 for page in pages if page.consensus_semantic_agreement is not None)
                    / max(sum(page.consensus_semantic_agreement is not None for page in pages), 1)
                ),
                "mean_measure_confidence": (
                    sum(page.consensus_confidence or 0.0 for page in pages if page.consensus_confidence is not None)
                    / max(sum(page.consensus_confidence is not None for page in pages), 1)
                ),
                "mean_measure_calibration_probability": (
                    sum(page.consensus_measure_probability or 0.0 for page in pages if page.consensus_measure_probability is not None)
                    / max(sum(page.consensus_measure_probability is not None for page in pages), 1)
                ),
                "mean_visual_compatibility_probability": (
                    sum(page.consensus_visual_probability or 0.0 for page in pages if page.consensus_visual_probability is not None)
                    / max(sum(page.consensus_visual_probability is not None for page in pages), 1)
                ),
                "mean_event_lattice_probability": (
                    sum(page.consensus_event_probability or 0.0 for page in pages if page.consensus_event_probability is not None)
                    / max(sum(page.consensus_event_probability is not None for page in pages), 1)
                ),
                "mean_cross_measure_context_probability": (
                    sum(page.consensus_context_probability or 0.0 for page in pages if page.consensus_context_probability is not None)
                    / max(sum(page.consensus_context_probability is not None for page in pages), 1)
                ),
                "mean_ensemble_calibration_probability": (
                    sum(page.consensus_ensemble_probability or 0.0 for page in pages if page.consensus_ensemble_probability is not None)
                    / max(sum(page.consensus_ensemble_probability is not None for page in pages), 1)
                ),
                "mean_selection_risk_probability": (
                    sum(page.consensus_selection_risk_probability or 0.0 for page in pages if page.consensus_selection_risk_probability is not None)
                    / max(sum(page.consensus_selection_risk_probability is not None for page in pages), 1)
                ),
                "mean_chord_patch_probability": (
                    sum(page.consensus_chord_patch_probability or 0.0 for page in pages if page.consensus_chord_patch_probability is not None)
                    / max(sum(page.consensus_chord_patch_probability is not None for page in pages), 1)
                ),
                "mean_tuplet_patch_probability": (
                    sum(page.consensus_tuplet_patch_probability or 0.0 for page in pages if page.consensus_tuplet_patch_probability is not None)
                    / max(sum(page.consensus_tuplet_patch_probability is not None for page in pages), 1)
                ),
                "mean_pitch_patch_probability": (
                    sum(page.consensus_pitch_patch_probability or 0.0 for page in pages if page.consensus_pitch_patch_probability is not None)
                    / max(sum(page.consensus_pitch_patch_probability is not None for page in pages), 1)
                ),
                "mean_rhythm_patch_probability": (
                    sum(page.consensus_rhythm_patch_probability or 0.0 for page in pages if page.consensus_rhythm_patch_probability is not None)
                    / max(sum(page.consensus_rhythm_patch_probability is not None for page in pages), 1)
                ),
                "mean_event_kind_patch_probability": (
                    sum(page.consensus_event_kind_patch_probability or 0.0 for page in pages if page.consensus_event_kind_patch_probability is not None)
                    / max(sum(page.consensus_event_kind_patch_probability is not None for page in pages), 1)
                ),
                "mean_attribute_patch_probability": (
                    sum(page.consensus_attribute_patch_probability or 0.0 for page in pages if page.consensus_attribute_patch_probability is not None)
                    / max(sum(page.consensus_attribute_patch_probability is not None for page in pages), 1)
                ),
                "mean_event_presence_patch_probability": (
                    sum(page.consensus_event_presence_patch_probability or 0.0 for page in pages if page.consensus_event_presence_patch_probability is not None)
                    / max(sum(page.consensus_event_presence_patch_probability is not None for page in pages), 1)
                ),
                "mean_event_presence_visual_guard_probability": (
                    sum(page.consensus_event_presence_visual_guard_probability or 0.0 for page in pages if page.consensus_event_presence_visual_guard_probability is not None)
                    / max(sum(page.consensus_event_presence_visual_guard_probability is not None for page in pages), 1)
                ),
                "mean_slur_patch_probability": (
                    sum(page.consensus_slur_patch_probability or 0.0 for page in pages if page.consensus_slur_patch_probability is not None)
                    / max(sum(page.consensus_slur_patch_probability is not None for page in pages), 1)
                ),
                "mean_articulation_patch_probability": (
                    sum(page.consensus_articulation_patch_probability or 0.0 for page in pages if page.consensus_articulation_patch_probability is not None)
                    / max(sum(page.consensus_articulation_patch_probability is not None for page in pages), 1)
                ),
                "mean_ornament_patch_probability": (
                    sum(page.consensus_ornament_patch_probability or 0.0 for page in pages if page.consensus_ornament_patch_probability is not None)
                    / max(sum(page.consensus_ornament_patch_probability is not None for page in pages), 1)
                ),
                "mean_grace_patch_probability": (
                    sum(page.consensus_grace_patch_probability or 0.0 for page in pages if page.consensus_grace_patch_probability is not None)
                    / max(sum(page.consensus_grace_patch_probability is not None for page in pages), 1)
                ),
                "mean_lyric_patch_probability": (
                    sum(page.consensus_lyric_patch_probability or 0.0 for page in pages if page.consensus_lyric_patch_probability is not None)
                    / max(sum(page.consensus_lyric_patch_probability is not None for page in pages), 1)
                ),
                "mean_direction_patch_probability": (
                    sum(page.consensus_direction_patch_probability or 0.0 for page in pages if page.consensus_direction_patch_probability is not None)
                    / max(sum(page.consensus_direction_patch_probability is not None for page in pages), 1)
                ),
                "mean_barline_patch_probability": (
                    sum(page.consensus_barline_patch_probability or 0.0 for page in pages if page.consensus_barline_patch_probability is not None)
                    / max(sum(page.consensus_barline_patch_probability is not None for page in pages), 1)
                ),
                "measure_replacements": sum(page.consensus_replacements for page in pages),
                "chord_patch_measures": sum(page.consensus_chord_patch_measures for page in pages),
                "chord_patch_events": sum(page.consensus_chord_patch_events for page in pages),
                "tuplet_patch_measures": sum(page.consensus_tuplet_patch_measures for page in pages),
                "tuplet_patch_events": sum(page.consensus_tuplet_patch_events for page in pages),
                "tuplet_patch_groups": sum(page.consensus_tuplet_patch_groups for page in pages),
                "pitch_patch_measures": sum(page.consensus_pitch_patch_measures for page in pages),
                "pitch_patch_events": sum(page.consensus_pitch_patch_events for page in pages),
                "rhythm_patch_measures": sum(page.consensus_rhythm_patch_measures for page in pages),
                "rhythm_patch_events": sum(page.consensus_rhythm_patch_events for page in pages),
                "event_kind_patch_measures": sum(page.consensus_event_kind_patch_measures for page in pages),
                "event_kind_patch_events": sum(page.consensus_event_kind_patch_events for page in pages),
                "attribute_patch_measures": sum(page.consensus_attribute_patch_measures for page in pages),
                "attribute_patch_attributes": sum(page.consensus_attribute_patch_attributes for page in pages),
                "event_presence_patch_measures": sum(page.consensus_event_presence_patch_measures for page in pages),
                "event_presence_inserted_events": sum(page.consensus_event_presence_inserted_events for page in pages),
                "event_presence_deleted_events": sum(page.consensus_event_presence_deleted_events for page in pages),
                "event_presence_visual_guard_transactions": sum(page.consensus_event_presence_visual_guard_transactions for page in pages),
                "event_presence_visual_guard_rejections": sum(page.consensus_event_presence_visual_guard_rejections for page in pages),
                "event_presence_visual_guard_models": sorted({
                    page.consensus_event_presence_visual_guard_model
                    for page in pages
                    if page.consensus_event_presence_visual_guard_model
                }),
                "event_presence_visual_guard_note_thresholds": sorted({
                    page.consensus_event_presence_visual_guard_note_threshold
                    for page in pages
                    if page.consensus_event_presence_visual_guard_note_threshold is not None
                }),
                "event_presence_visual_guard_rest_thresholds": sorted({
                    page.consensus_event_presence_visual_guard_rest_threshold
                    for page in pages
                    if page.consensus_event_presence_visual_guard_rest_threshold is not None
                }),
                "slur_patch_measures": sum(page.consensus_slur_patch_measures for page in pages),
                "slur_patch_events": sum(page.consensus_slur_patch_events for page in pages),
                "slur_patch_arcs": sum(page.consensus_slur_patch_arcs for page in pages),
                "articulation_patch_measures": sum(page.consensus_articulation_patch_measures for page in pages),
                "articulation_patch_events": sum(page.consensus_articulation_patch_events for page in pages),
                "articulation_patch_marks": sum(page.consensus_articulation_patch_marks for page in pages),
                "ornament_patch_measures": sum(page.consensus_ornament_patch_measures for page in pages),
                "ornament_patch_events": sum(page.consensus_ornament_patch_events for page in pages),
                "ornament_patch_marks": sum(page.consensus_ornament_patch_marks for page in pages),
                "grace_patch_measures": sum(page.consensus_grace_patch_measures for page in pages),
                "grace_patch_events": sum(page.consensus_grace_patch_events for page in pages),
                "grace_patch_added": sum(page.consensus_grace_patch_added for page in pages),
                "grace_patch_removed": sum(page.consensus_grace_patch_removed for page in pages),
                "lyric_patch_measures": sum(page.consensus_lyric_patch_measures for page in pages),
                "lyric_patch_events": sum(page.consensus_lyric_patch_events for page in pages),
                "lyric_patch_lyrics": sum(page.consensus_lyric_patch_lyrics for page in pages),
                "direction_patch_measures": sum(page.consensus_direction_patch_measures for page in pages),
                "direction_patch_directions": sum(page.consensus_direction_patch_directions for page in pages),
                "barline_patch_measures": sum(page.consensus_barline_patch_measures for page in pages),
                "barline_patch_locations": sum(page.consensus_barline_patch_locations for page in pages),
                "barline_patch_repeats": sum(page.consensus_barline_patch_repeats for page in pages),
                "patch_transaction_rejections": sum(page.consensus_patch_transaction_rejections for page in pages),
                "disagreement_measures": sum(len(page.consensus_disagreements) for page in pages),
                "unresolved_measures": sum(len(page.consensus_unresolved) for page in pages),
            },
            "warnings": job.warnings,
            "ocr": ocr_pages,
            "ocr_runtime": {
                "runtime": "cpu",
                "cpu_pages": sum(
                    page.ocr_accelerator_selected == "cpu"
                    for page in pages
                ),
                "verified_pages": sum(
                    page.ocr_accelerator_selected == "cpu"
                    and page.ocr_accelerator_verified
                    for page in pages
                ),
                "unverified_pages": sum(
                    page.omr_status == "completed"
                    and not page.ocr_accelerator_verified
                    for page in pages
                ),
            },
            "semantic_detector_runtime": {
                "runtime": "cpu",
                "authorized_at_job_start": semantic_asset_status.enabled,
                "asset_status": semantic_asset_status.status,
                "model_version": semantic_asset_status.model_version,
                "enabled_pages": sum(
                    page.semantic_detector_enabled for page in pages
                ),
                "detections": sum(
                    page.semantic_detector_detection_count for page in pages
                ),
                "cpu_pages": sum(
                    page.semantic_detector_accelerator_selected == "cpu"
                    and page.semantic_detector_accelerator_verified
                    for page in pages
                ),
                "unverified_enabled_pages": sum(
                    page.semantic_detector_enabled
                    and not page.semantic_detector_accelerator_verified
                    for page in pages
                ),
            },
            "semantic_source_audit": {
                "audited_pages": len(semantic_audit_reports_by_page),
                "failed_or_unavailable_pages": sum(
                    page.omr_status == "completed"
                    and page.semantic_source_audit_status != "completed"
                    for page in pages
                ),
                "potential_omissions": sum(
                    page.semantic_source_audit_omission_count for page in pages
                ),
                "extraneous_symbols": sum(
                    page.semantic_source_audit_extraneous_count for page in pages
                ),
                "positional_mismatches": sum(
                    page.semantic_source_audit_positional_mismatch_count
                    for page in pages
                ),
                "pages": [
                    {
                        "page": page_index,
                        **report.to_dict(),
                    }
                    for page_index, report in sorted(
                        semantic_audit_reports_by_page.items()
                    )
                ],
            },
            "ocr_direction_anchoring": {
                "marks": sum(page.ocr_mark_count for page in pages),
                "injected": sum(page.ocr_injected_count for page in pages),
                "review_candidates": sum(page.ocr_review_candidate_count for page in pages),
                "barline_exact": sum(page.ocr_barline_anchor_count for page in pages),
                "barline_rescaled": sum(page.ocr_rescaled_anchor_count for page in pages),
                "mean_role_probability": (
                    sum(page.ocr_anchor_mean_probability or 0.0 for page in pages if page.ocr_anchor_mean_probability is not None)
                    / max(sum(page.ocr_anchor_mean_probability is not None for page in pages), 1)
                ),
            },
            "notation_coverage": {
                "audited_pages": len(notation_reports_by_page),
                "failed_pages": sum(page.notation_coverage_status == "failed" for page in pages),
                "source_candidates": sum(page.notation_candidate_count for page in pages),
                "potential_omissions": sum(page.notation_potential_omission_count for page in pages),
                "unbalanced_structures": sum(page.notation_unbalanced_structure_count for page in pages),
                "artifact": notation_coverage_path.name,
                "pages": [
                    {
                        "page": page_index,
                        **report.to_dict(),
                    }
                    for page_index, report in sorted(notation_reports_by_page.items())
                ],
            },
            "source_ornament_enrichment": {
                "audited_pages": len(ornament_reports_by_page),
                "detected": sum(
                    report.detected_count
                    for report in ornament_reports_by_page.values()
                ),
                "inserted_mordents": sum(
                    report.inserted_mordent_count
                    for report in ornament_reports_by_page.values()
                ),
                "detected_trills": sum(
                    report.detected_trill_count
                    for report in ornament_reports_by_page.values()
                ),
                "inserted_trills": sum(
                    report.inserted_trill_count
                    for report in ornament_reports_by_page.values()
                ),
                "authoritative_source_commits": sum(
                    report.authoritative_source_commit
                    for report in ornament_reports_by_page.values()
                ),
                "reclassified_trills": sum(
                    report.reclassified_trill_count
                    for report in ornament_reports_by_page.values()
                ),
                "abstentions": sum(
                    report.abstention_count
                    for report in ornament_reports_by_page.values()
                ),
                "pages": [
                    {"page": page_index, **report.to_dict()}
                    for page_index, report in sorted(ornament_reports_by_page.items())
                ],
            },
            "source_beam_enrichment": {
                "audited_pages": len(beam_reports_by_page),
                "committed_pages": sum(
                    report.transaction_committed
                    for report in beam_reports_by_page.values()
                ),
                "detected_segments": sum(
                    report.detected_count
                    for report in beam_reports_by_page.values()
                ),
                "injected_segments": sum(
                    report.injected_segment_count
                    for report in beam_reports_by_page.values()
                ),
                "injected_markers": sum(
                    report.injected_marker_count
                    for report in beam_reports_by_page.values()
                ),
                "abstentions": sum(
                    report.abstention_count
                    for report in beam_reports_by_page.values()
                ),
                "pages": [
                    {"page": page_index, **report.to_dict()}
                    for page_index, report in sorted(
                        beam_reports_by_page.items()
                    )
                ],
            },
            "wedge_enrichment": {
                "audited_pages": len(wedge_reports_by_page),
                "committed_pages": sum(
                    report.transaction_committed
                    for report in wedge_reports_by_page.values()
                ),
                "auto_injected": sum(
                    report.injected_count
                    for report in wedge_reports_by_page.values()
                ),
                "abstentions": sum(
                    report.abstention_count
                    for report in wedge_reports_by_page.values()
                ),
                "pages": [
                    {
                        "page": page_index,
                        **report.to_dict(),
                    }
                    for page_index, report in sorted(wedge_reports_by_page.items())
                ],
            },
            "slur_relation_repair": {
                "audited_pages": len(slur_relation_reports_by_page),
                "committed_pages": sum(
                    report.transaction_committed
                    for report in slur_relation_reports_by_page.values()
                ),
                "repaired": sum(
                    report.repaired_count
                    for report in slur_relation_reports_by_page.values()
                ),
                "abstentions": sum(
                    report.abstention_count
                    for report in slur_relation_reports_by_page.values()
                ),
                "pages": [
                    {
                        "page": page_index,
                        **report.to_dict(),
                    }
                    for page_index, report in sorted(
                        slur_relation_reports_by_page.items()
                    )
                ],
            },
            "review": {
                "pending": len(review_issues),
                "preserved_risks": 0,
                "issues": [issue.to_dict() for issue in review_issues],
            },
            "models": {
                "omr": "homr-0.7.0-ensemble",
                **model_versions(self.settings.resources),
                "measure_consensus": "scorescan-semantic-consensus-13",
                "tempo_mark_parser": "scorescan-tempo-parser-1",
                "ocr_consensus": "scorescan-multipass-5",
                "decision_policy": DEFAULT_POLICY.version,
                "resource_manifest": read_json(self.settings.resources / "model_manifest.json", {}),
                "resource_audit": model_resource_audit.to_dict(),
            },
            "accelerator": runner.accelerator_status().to_dict(),
            "pages": [page.to_dict() for page in pages],
        }
        report_path = output_dir / "conversion_report.json"
        layout_report_path = output_dir / "source_layout.json"
        atomic_write_json(report_path, manifest)
        atomic_write_json(
            layout_report_path,
            {
                "pages": [read_json(Path(page.layout_path), {}) if page.layout_path else {} for page in pages]
            },
        )
        integrity = build_bundle_integrity(
            output_dir,
            [
                ("musicxml", output_path),
                ("mxl", mxl_path),
                ("preview", preview_path),
                *_preview_page_artifacts(preview_path),
                ("conversion_report", report_path),
                ("source_layout", layout_report_path),
                ("review_issues", review_path),
                ("notation_coverage", notation_coverage_path),
                ("implicit_triplet_transaction", triplet_report_path),
            ],
        )
        if not integrity.valid:
            for error in integrity.errors:
                job.add_warning(f"输出文件完整性检查：{error}")
            raise RuntimeError("输出产物未能原子提交完整性清单，结果不会发布。")
        job.update(
            status="completed",
            stage="Conversion complete",
            progress=1.0,
            current_page=len(pages),
            quality_state=quality_state,
            quality_score=quality_certificate.score,
            production_ready=quality_certificate.auto_release_eligible,
            release_blockers=list(quality_certificate.release_blockers),
            result_musicxml=str(output_path),
            result_mxl=str(mxl_path),
            preview_svg=str(preview_path) if preview_path else None,
            report_path=str(report_path),
            layout_report_path=str(layout_report_path),
            artifact_manifest_path=integrity.manifest_path,
            artifact_bundle_id=integrity.bundle_id,
            review_path=str(review_path),
            review_issues=review_issues,
            review_resolved_count=0,
        )

    def resolve_review_issue(self, job_id: str, issue_id: str, value: str | None, ignore: bool = False) -> tuple[bool, str]:
        with self._review_operation_lock:
            return self._resolve_review_issue_locked(job_id, issue_id, value, ignore)

    def _resolve_review_issue_locked(self, job_id: str, issue_id: str, value: str | None, ignore: bool = False) -> tuple[bool, str]:
        job = self.get(job_id)
        if job is None:
            return False, "任务不存在"
        if job.status != "completed" or not job.result_musicxml or not job.result_mxl:
            return False, "结果尚未完成"
        with job.lock:
            issue = next((item for item in job.review_issues if item.id == issue_id), None)
            if issue is None:
                return False, "疑点不存在"
            if issue.status == "resolved":
                return True, "该疑点已经处理"
            changed = False
            if issue.writeback_supported:
                if issue.global_measure_number is None or not issue.kind:
                    return False, "该疑点缺少写回定位信息"
                chosen = None if ignore else (value or issue.suggested_value or issue.raw_value)
                changed = update_direction_in_musicxml(
                    Path(job.result_musicxml),
                    issue.global_measure_number,
                    chosen,
                    issue.kind,
                    issue.offset_ratio,
                    issue.placement or "above",
                    [item for item in [issue.raw_value, issue.suggested_value, *issue.options] if item],
                )
                issue.resolution = "忽略" if ignore else (chosen or "")
            else:
                issue.resolution = "已确认风险" if not ignore else "跳过检查"
            issue.status = "resolved"
            job.review_resolved_count = sum(item.status == "resolved" for item in job.review_issues)

        # Regenerate all deterministic artifacts after a review decision.
        package_mxl(Path(job.result_musicxml), Path(job.result_mxl))
        preview_path, preview_warnings = render_preview(Path(job.result_musicxml), job.root / "result" / "preview")
        for warning in preview_warnings:
            job.add_warning(warning)
        validation_errors = validate_musicxml(Path(job.result_musicxml))
        semantic = analyze_musicxml(Path(job.result_musicxml))
        pending = [item for item in job.review_issues if item.status != "resolved"]
        fallback_pages = [page.index for page in job.pages if page.omr_status == "fallback"]
        certificate = build_quality_certificate(
            job.pages,
            validation_errors,
            semantic,
            job.review_issues,
            fallback_pages,
            audit_model_manifest(self.settings.resources).to_dict(),
        )
        job.quality_state = certificate.state
        preserved_risks = [item for item in job.review_issues if item.status == "resolved" and item.risk_preserved]
        if not pending and preserved_risks and not validation_errors:
            job.quality_state = "reviewed_with_warnings"
        elif not pending and certificate.state == "verified":
            job.quality_state = "verified_after_review"
        elif not pending and certificate.state == "review_recommended":
            job.quality_state = "reviewed_with_warnings"
        job.quality_score = certificate.score
        job.production_ready = certificate.auto_release_eligible
        job.release_blockers = list(certificate.release_blockers)
        if preview_path:
            job.preview_svg = str(preview_path)
        if job.review_path:
            atomic_write_json(Path(job.review_path), {"issues": [item.to_dict() for item in job.review_issues]})
        if job.report_path:
            report = read_json(Path(job.report_path), {})
            if isinstance(report, dict):
                report["quality_state"] = job.quality_state
                report["quality_certificate"] = certificate.to_dict()
                report["output_sha256"] = sha256_file(Path(job.result_musicxml))
                report["output_mxl_sha256"] = sha256_file(Path(job.result_mxl))
                report["review"] = {
                    "pending": len(pending),
                    "resolved": job.review_resolved_count,
                    "preserved_risks": sum(item.status == "resolved" and item.risk_preserved for item in job.review_issues),
                    "issues": [item.to_dict() for item in job.review_issues],
                }
                report.setdefault("validation", {})["xml_errors"] = validation_errors
                report["validation"]["semantic"] = semantic
                atomic_write_json(Path(job.report_path), report)
        integrity = build_bundle_integrity(
            job.root / "result",
            [
                ("musicxml", Path(job.result_musicxml)),
                ("mxl", Path(job.result_mxl)),
                ("preview", Path(job.preview_svg) if job.preview_svg else None),
                *_preview_page_artifacts(Path(job.preview_svg) if job.preview_svg else None),
                ("conversion_report", Path(job.report_path) if job.report_path else None),
                ("source_layout", Path(job.layout_report_path) if job.layout_report_path else None),
                ("review_issues", Path(job.review_path) if job.review_path else None),
                (
                    "implicit_triplet_transaction",
                    (
                        job.root / "result" / "implicit_triplet_transaction.json"
                        if (job.root / "result" / "implicit_triplet_transaction.json").is_file()
                        else None
                    ),
                ),
            ],
        )
        if not integrity.valid:
            job.quality_state = "best_effort"
            job.production_ready = False
            if "输出文件完整性检查未通过" not in job.release_blockers:
                job.release_blockers.append("输出文件完整性检查未通过")
            for error in integrity.errors:
                job.add_warning(f"输出文件完整性检查：{error}")
        job.update(
            quality_state=job.quality_state,
            quality_score=job.quality_score,
            production_ready=job.production_ready,
            release_blockers=job.release_blockers,
            review_resolved_count=job.review_resolved_count,
            preview_svg=job.preview_svg,
            artifact_manifest_path=integrity.manifest_path,
            artifact_bundle_id=integrity.bundle_id,
        )
        return True, "已更新 MusicXML" if changed else "已记录处理结果"

    def remove(self, job_id: str) -> bool:
        # A queued or interrupted task may already have a worker waiting on the global
        # semaphore.  Removing it from the index would not stop that worker and would
        # create an orphaned conversion.  Check and remove atomically under one lock.
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None or job.status in {"queued", "running", "cancelling", "interrupted"}:
                return False
            self.jobs.pop(job_id, None)
        shutil.rmtree(job.root, ignore_errors=True)
        return True

    def suggested_download_name(self, job: JobState) -> str:
        if job.output_name:
            return safe_filename(job.output_name)
        title = extract_title(Path(job.result_musicxml)) if job.result_musicxml else None
        base = title or (Path(job.source_files[0]).stem if job.source_files else "converted_score")
        return safe_filename(base)
