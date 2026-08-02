from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .direction_model import DirectionCorrector, normalize_direction
from .layout import PageLayout, StaffSystem, system_measure_bounds
from .models import PageInfo, ReviewIssue
from .notation_coverage import NotationCoverageReport, VisualNotationCandidate
from .policy import DEFAULT_POLICY
from .text_enrichment import OcrMark
from .util import read_json


_REVIEWABLE_KINDS = {"dynamic", "direction", "metronome", "text"}


def _box_bounds(box: Iterable[Iterable[float]]) -> tuple[int, int, int, int]:
    points = np.asarray(list(box), dtype=float)
    return (
        int(np.floor(points[:, 0].min())),
        int(np.floor(points[:, 1].min())),
        int(np.ceil(points[:, 0].max())),
        int(np.ceil(points[:, 1].max())),
    )


def _save_crop(image_path: Path, box: list[list[float]], destination: Path) -> str | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _box_bounds(box)
    margin_x = max(35, int((x2 - x1) * 1.4))
    margin_y = max(30, int((y2 - y1) * 2.2))
    x1, x2 = max(0, x1 - margin_x), min(width, x2 + margin_x)
    y1, y2 = max(0, y1 - margin_y), min(height, y2 + margin_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Enlarge small printed directions so the review page remains easy to read.
    if crop.shape[1] < 900:
        scale = min(3.0, 900 / max(crop.shape[1], 1))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), crop, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    return str(destination)


def _needs_review(mark: OcrMark, suggested: str, suggestion_probability: float) -> bool:
    if mark.kind not in _REVIEWABLE_KINDS or mark.measure_index is None or mark.system_index is None:
        return False
    if mark.distance_staff_spaces > 10.0 or mark.placement == "within":
        return False
    plausible_role = mark.musical_direction_probability >= DEFAULT_POLICY.direction_anchor_review_floor
    low_anchor_evidence = mark.measure_anchor_confidence < 0.52
    if mark.kind == "dynamic":
        # A one-letter OCR hypothesis that failed the source-geometry writeback gate
        # is extremely often a notehead, accidental, ornament or stem fragment.  It
        # did not change the delivered score and remains available in diagnostics;
        # presenting it as a required user correction creates false work.
        source_geometry_verified = (
            (mark.backend or "").startswith("source-dynamic-geometry")
            and mark.score >= 0.97
            and mark.musical_direction_probability >= 0.99
        )
        return (
            mark.injected
            and plausible_role
            and not source_geometry_verified
            and (mark.score < 0.78 or low_anchor_evidence)
        )
    if mark.kind == "metronome":
        return mark.injected and plausible_role and (mark.score < 0.86 or low_anchor_evidence)
    if not plausible_role:
        return False
    if suggestion_probability >= 0.72 and normalize_direction(suggested) != normalize_direction(mark.raw_text):
        return not mark.corrected or mark.score < 0.84
    if mark.kind == "text":
        # Generic alphabetic text is only review-worthy when reasonably legible, close
        # to the staff and positively classified as a musical direction.
        return mark.score >= 0.52 and mark.distance_staff_spaces <= 6.0 and not mark.injected
    return mark.score < 0.80 or low_anchor_evidence or not mark.injected


def build_text_review_issues(
    pages: list[PageInfo],
    marks_by_page: dict[int, list[OcrMark]],
    page_measure_offsets: dict[int, int],
    output_dir: Path,
) -> list[ReviewIssue]:
    corrector = DirectionCorrector()
    issues: list[ReviewIssue] = []
    crop_dir = output_dir / "review" / "crops"

    for page in pages:
        page_marks = marks_by_page.get(page.index, [])
        page_count = 0
        for mark_index, mark in enumerate(page_marks, start=1):
            suggestion = corrector.suggest(mark.raw_text)
            if not _needs_review(mark, suggestion.text, suggestion.probability):
                continue
            issue_id = f"p{page.index:04d}-text-{mark_index:04d}"
            crop_path = _save_crop(
                Path(page.normalized_path or page.image_path),
                mark.box,
                crop_dir / f"{issue_id}.png",
            )
            options: list[str] = []
            for value in (mark.text, suggestion.text, mark.raw_text):
                if value and normalize_direction(value) not in {normalize_direction(item) for item in options}:
                    options.append(value)
            global_measure = page_measure_offsets.get(page.index, 0) + int(mark.measure_index or 0) + 1
            confidence = round(mark.score * 100)
            role_confidence = round(mark.musical_direction_probability * 100)
            issue = ReviewIssue(
                id=issue_id,
                page_index=page.index,
                category="music_text",
                title="确认谱面文字或力度",
                message=f"第 {page.index} 页，第 {global_measure} 小节附近；OCR 置信度约 {confidence}%，谱面方向证据约 {role_confidence}%",
                crop_path=crop_path,
                raw_value=mark.raw_text,
                suggested_value=suggestion.text,
                options=options,
                global_measure_number=global_measure,
                local_measure_index=mark.measure_index,
                kind=mark.kind,
                placement=mark.placement or "above",
                offset_ratio=float(mark.offset_ratio),
            )
            issues.append(issue)
            page_count += 1
        page.review_issue_count = page_count
    return issues




def _save_measure_crop(image_path: Path, system: StaffSystem, x1: int, x2: int, destination: Path) -> str | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    margin_x = max(20, int(system.spacing * 2.4))
    margin_y = max(35, int(system.spacing * 5.2))
    left = max(0, x1 - margin_x)
    right = min(width, x2 + margin_x)
    top = max(0, system.top - margin_y)
    bottom = min(height, system.bottom + margin_y)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    if crop.shape[1] < 1100:
        scale = min(3.2, 1100 / max(crop.shape[1], 1))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), crop, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    return str(destination)


def build_consensus_review_issues(
    pages: list[PageInfo],
    page_measure_offsets: dict[int, int],
    output_dir: Path,
) -> list[ReviewIssue]:
    """Expose only low-confidence mutations that can affect the delivered score.

    Disagreement between preprocessing candidates is diagnostic evidence, not a
    user-actionable defect.  When the conservative template was retained, creating
    one pending issue per divergent measure only floods the review UI and can wrongly
    make a valid result look undeliverable.  The complete divergence evidence remains
    in ``consensus.json`` and the conversion report.  A review item is created only
    when a low-confidence consensus decision actually mutated the delivered score.
    """
    issues: list[ReviewIssue] = []
    crop_dir = output_dir / "review" / "measure_crops"
    for page in pages:
        unresolved = set(int(item) for item in page.consensus_unresolved if int(item) > 0)
        low_meta: set[int] = set()
        if page.consensus_report_path:
            consensus_payload = read_json(Path(page.consensus_report_path), {})
            votes = consensus_payload.get("votes", []) if isinstance(consensus_payload, dict) else []
            for vote in votes if isinstance(votes, list) else []:
                if not isinstance(vote, dict):
                    continue
                try:
                    local_measure = int(vote.get("measure_index", 0) or 0)
                    probability = float(vote.get("selected_ensemble_probability", 0.5) or 0.5)
                    risk_probability = float(vote.get("selected_selection_risk_probability", 0.5) or 0.5)
                    risk_accepted = bool(vote.get("selection_risk_accepted", False))
                    risk_applicable = bool(vote.get("selection_risk_applicable", False))
                    confidence = float(vote.get("semantic_confidence", 0.0) or 0.0)
                    semantic_support_ratio = float(vote.get("semantic_support_ratio", 0.0) or 0.0)
                    missing_candidates = int(vote.get("missing_candidates", 0) or 0)
                except (TypeError, ValueError):
                    continue
                decision = str(vote.get("decision", ""))
                semantic_equivalent_clean = (
                    decision == "retain_semantic_equivalent"
                    and bool(vote.get("strict_majority", False))
                    and confidence >= 0.995
                    and semantic_support_ratio >= 0.995
                    and missing_candidates == 0
                )
                mutates_output = decision.startswith(("replace_", "patch_"))
                if (
                    local_measure > 0
                    and mutates_output
                    and not semantic_equivalent_clean
                    and (
                        probability < DEFAULT_POLICY.ensemble_review_probability_floor
                        or (risk_applicable and risk_probability < DEFAULT_POLICY.selection_risk_review_probability_floor)
                        or (risk_applicable and decision == "retain_template_selection_risk_guard" and not risk_accepted)
                    )
                    and (
                        decision
                        not in {
                            "retain_agreement",
                            "retain_exact_majority",
                            "retain_redundant_state_attributes",
                        }
                        or confidence < 0.72
                    )
                ):
                    low_meta.add(local_measure)
        # ``unresolved`` means no candidate earned permission to replace the
        # conservative primary result.  Preserve it in diagnostics, but do not
        # manufacture a pending user task for a writeback that never happened.
        review_measures = sorted(low_meta)
        if not review_measures or not page.layout_path:
            continue
        payload = read_json(Path(page.layout_path), {})
        if not isinstance(payload, dict):
            continue
        layout = PageLayout.from_dict(payload)
        measure_map: dict[int, tuple[StaffSystem, int, int]] = {}
        local_number = 1
        for system in layout.systems:
            for x1, x2 in system_measure_bounds(system):
                measure_map[local_number] = (system, x1, x2)
                local_number += 1
        page_count = 0
        for local_measure in review_measures:
            region = measure_map.get(local_measure)
            if region is None:
                continue
            system, x1, x2 = region
            issue_id = f"p{page.index:04d}-measure-{local_measure:04d}"
            crop_path = _save_measure_crop(
                Path(page.normalized_path or page.image_path),
                system,
                x1,
                x2,
                crop_dir / f"{issue_id}.png",
            )
            global_measure = page_measure_offsets.get(page.index, 0) + local_measure
            low_meta_only = local_measure in low_meta and local_measure not in unresolved
            title = "核对小节内容"
            message = (
                f"第 {page.index} 页 · 第 {global_measure} 小节。当前结果保留基础结构；符号替换条件未满足。"
                if low_meta_only
                else f"第 {page.index} 页 · 第 {global_measure} 小节。扫描增强结果未达一致；当前结果采用结构得分最高的版本。"
            )
            issues.append(
                ReviewIssue(
                    id=issue_id,
                    page_index=page.index,
                    category="measure_consensus",
                    title=title,
                    message=message,
                    crop_path=crop_path,
                    options=["已核对，保留当前结果"],
                    global_measure_number=global_measure,
                    local_measure_index=local_measure - 1,
                    status="pending",
                    writeback_supported=False,
                    requires_value=False,
                    risk_preserved=True,
                )
            )
            page_count += 1
        page.review_issue_count += page_count
    return issues


def _save_notation_contact_sheet(
    image_path: Path,
    candidates: list[VisualNotationCandidate],
    destination: Path,
) -> str | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or not candidates:
        return None
    height, width = image.shape[:2]
    tiles: list[np.ndarray] = []
    for index, candidate in enumerate(candidates[:24], start=1):
        x1, y1, x2, y2 = candidate.bbox
        margin_x = max(18, int((x2 - x1) * 0.22))
        margin_y = max(18, int((y2 - y1) * 1.6))
        left = max(0, x1 - margin_x)
        right = min(width, x2 + margin_x)
        top = max(0, y1 - margin_y)
        bottom = min(height, y2 + margin_y)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        scale = min(3.0, 300 / max(crop.shape[1], 1), 92 / max(crop.shape[0], 1))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        tile = np.full((128, 330, 3), 255, np.uint8)
        crop = crop[:100, :300]
        paste_y = 24 + max(0, (100 - crop.shape[0]) // 2)
        paste_x = max(15, (330 - crop.shape[1]) // 2)
        tile[paste_y:paste_y + crop.shape[0], paste_x:paste_x + crop.shape[1]] = crop
        cv2.putText(
            tile,
            f"#{index}",
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return None
    columns = 3
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.full((rows * 128, columns * 330, 3), 238, np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * 128:(row + 1) * 128, column * 330:(column + 1) * 330] = tile
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    return str(destination)


def build_notation_coverage_review_issues(
    pages: list[PageInfo],
    reports_by_page: dict[int, NotationCoverageReport],
    output_dir: Path,
) -> list[ReviewIssue]:
    """Create at most two grouped reviews per page for silent-omission risk."""

    issues: list[ReviewIssue] = []
    crop_dir = output_dir / "review" / "notation_coverage"
    floors = {"curved_connector": 0.78, "crescendo": 0.82, "diminuendo": 0.82}
    for page in pages:
        report = reports_by_page.get(page.index)
        if report is None:
            continue
        by_kind = {item.kind: item for item in report.kinds}
        groups = (
            ("curved_connector", ("curved_connector",), "弧线（连奏线/延音线）"),
            ("wedge", ("crescendo", "diminuendo"), "渐强/渐弱发夹"),
        )
        page_count = 0
        for group_key, kinds, label in groups:
            potential = sum(
                by_kind[kind].potential_omission_count
                for kind in kinds
                if kind in by_kind
            )
            if potential <= 0:
                continue
            source_count = sum(
                by_kind[kind].confident_source_count
                for kind in kinds
                if kind in by_kind
            )
            emitted_count = sum(
                by_kind[kind].emitted_count
                for kind in kinds
                if kind in by_kind
            )
            candidates = [
                candidate
                for candidate in report.candidates
                if candidate.kind in kinds
                and candidate.confidence >= floors[candidate.kind]
            ]
            issue_id = f"p{page.index:04d}-coverage-{group_key}"
            crop_path = _save_notation_contact_sheet(
                Path(page.normalized_path or page.image_path),
                candidates,
                crop_dir / f"{issue_id}.png",
            )
            issues.append(
                ReviewIssue(
                    id=issue_id,
                    page_index=page.index,
                    category="notation_coverage",
                    title=f"核对{label}",
                    message=(
                        f"扫描件中检测到 {source_count} 个；MusicXML 中有 {emitted_count} 个。"
                        f"{potential} 个位置未匹配。"
                    ),
                    severity="warning",
                    crop_path=crop_path,
                    options=["已核对，保留当前结果"],
                    kind=group_key,
                    status="pending",
                    writeback_supported=False,
                    requires_value=False,
                    risk_preserved=True,
                )
            )
            page_count += 1
        page.review_issue_count += page_count
    return issues
