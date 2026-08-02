from __future__ import annotations

import csv
import io
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from lxml import etree

from .cpu_runtime import rapidocr_cpu_parameters
from .direction_anchor import DirectionAnchorClassifier, extract_direction_anchor_features
from .direction_model import DirectionCorrector, normalize_direction
from .layout import (
    PageLayout,
    ScoreSystemLayout,
    StaffSystem,
    analyze_layout,
    system_measure_bounds,
)
from .util import atomic_write_bytes, read_json, sha256_file
from .policy import DEFAULT_POLICY
from .process_control import popen_group_options, terminate_process_tree
from .semantic_detector import TEXT_REGION_CLASSES
from .semantic_detector_contract import NON_DIRECTION_TEXT_REGION_CLASSES
from .tempo_marks import parse_metronome_mark

try:  # RapidOCR is the bundled Windows OCR backend.
    from rapidocr import RapidOCR  # type: ignore
except Exception:  # pragma: no cover - exercised on minimal developer environments
    RapidOCR = None  # type: ignore

_OCR_ENGINE = None
_OCR_ENGINE_ACCELERATOR: str | None = None
_OCR_ENGINE_MODEL_VERSION = "rapidocr-bundled-default"
_OCR_ENGINE_MODEL_STATUS = "domain_model_absent"
_OCR_LOCK = threading.RLock()
_CORRECTOR: DirectionCorrector | None = None
_ANCHOR_CLASSIFIER: DirectionAnchorClassifier | None = None

_TESSERACT_TIMEOUT_SECONDS = 180.0
_TESSERACT_MAX_TSV_BYTES = 16 * 1024 * 1024
_TESSERACT_MAX_ROWS = 20_000
_TESSERACT_MAX_TEXT_CHARS = 512
_TESSERACT_POLL_SECONDS = 0.05
OCR_ACCELERATOR_ENVIRONMENT_VARIABLE = "SCORESCAN_OCR_ACCELERATOR"
_OCR_ACCELERATOR_VALUES = frozenset({"cpu"})
_DOMAIN_OCR_MANIFEST_FORMAT = 1
_DOMAIN_OCR_MAX_MODEL_BYTES = 512 * 1024 * 1024
_DOMAIN_OCR_MAX_KEYS_BYTES = 4 * 1024 * 1024
_DOMAIN_OCR_SCAN_ACCURACY_FLOOR = 0.998
_DOMAIN_OCR_SCAN_EDIT_FLOOR = 0.9995
_DOMAIN_OCR_CLEAN_ACCURACY_FLOOR = 0.998
_DOMAIN_OCR_CLEAN_EDIT_FLOOR = 0.9995
_DOMAIN_OCR_DETECTION_PRECISION_FLOOR = 0.995
_DOMAIN_OCR_DETECTION_RECALL_FLOOR = 0.995
_DOMAIN_OCR_DETECTION_HMEAN_FLOOR = 0.995
_DOMAIN_OCR_DETECTION_RUNTIME_PROFILE = (
    "ppocrv6-imagenet-db-scorescan-calibrated-v2"
)
_DOMAIN_OCR_DETECTION_PARAMETER_NAMES = {
    "Global.max_side_len",
    "Det.limit_side_len",
    "Det.limit_type",
    "Det.mean",
    "Det.std",
    "Det.thresh",
    "Det.box_thresh",
    "Det.max_candidates",
    "Det.unclip_ratio",
    "Det.use_dilation",
    "Det.score_mode",
}
_DOMAIN_OCR_DETECTION_FIXED_PARAMETERS = {
    "Global.max_side_len": 2000,
    "Det.limit_side_len": 736,
    "Det.limit_type": "min",
    "Det.mean": [0.485, 0.456, 0.406],
    "Det.std": [0.229, 0.224, 0.225],
    "Det.max_candidates": 1000,
    "Det.use_dilation": True,
    "Det.score_mode": "fast",
}
_DOMAIN_OCR_HOLDOUT_MINIMUM_SOURCES = 200
_DOMAIN_OCR_HOLDOUT_MINIMUM_WORDS = 1000
_DOMAIN_OCR_HOLDOUT_MINIMUM_PAGES = 100
_DOMAIN_OCR_HOLDOUT_MINIMUM_IOU = 0.75

_DYNAMIC_RE = re.compile(
    r"^(?:p{1,6}|f{1,6}|mp|mf|fp|pf|sf|sfp|sfpp|sfz|sffz|rf|rfz|fz)$", re.IGNORECASE
)
_METRONOME_RE = re.compile(
    r"(?:=|≈|~|ca\.?|circa).*?\b(\d{2,3})(?:\s*[-–]\s*(\d{2,3}))?\b|\b(\d{2,3})\s*(?:bpm|M\.?M\.?)",
    re.IGNORECASE,
)
_ALPHA_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
_DYNAMIC_TAGS = {
    "p", "pp", "ppp", "pppp", "ppppp", "pppppp", "mp", "mf",
    "f", "ff", "fff", "ffff", "fffff", "ffffff", "fp", "pf", "sf", "sfp",
    "sfpp", "sfz", "sffz", "rf", "rfz", "fz",
}
_INITIAL_TEMPO_TERMS = {
    "grave",
    "largo",
    "lento",
    "larghetto",
    "adagio",
    "andante",
    "andantino",
    "marcia",
    "moderato",
    "allegretto",
    "allegro",
    "vivace",
    "presto",
    "prestissimo",
}


@dataclass
class OcrMark:
    raw_text: str
    text: str
    score: float
    box: list[list[float]]
    kind: str
    system_index: int | None = None
    measure_index: int | None = None
    placement: str | None = None
    injected: bool = False
    corrected: bool = False
    correction_probability: float = 0.0
    correction_margin: float = 0.0
    correction_method: str | None = None
    correction_autocorrect_safe: bool = False
    correction_edit_ratio: float = 1.0
    offset_ratio: float = 0.0
    distance_staff_spaces: float = 0.0
    backend: str | None = None
    musical_direction_probability: float = 0.5
    direction_anchor_model_version: str | None = None
    direction_anchor_model_status: str | None = None
    measure_anchor_confidence: float = 0.0
    measure_anchor_method: str | None = None
    score_system_index: int | None = None
    target_part_id: str | None = None
    target_staff_number: int | None = None
    reanchored_existing: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OcrMark":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass(frozen=True)
class DomainOcrAssets:
    model_path: Path
    keys_path: Path
    model_version: str
    manifest_path: Path
    detection_model_path: Path
    detection_runtime_parameters: dict[str, object]


def _verified_domain_ocr_assets(
    resources_dir: Path | None = None,
) -> tuple[DomainOcrAssets | None, str]:
    """Load a domain OCR model only after its frozen release gate is verified."""

    root = (
        resources_dir
        if resources_dir is not None
        else Path(__file__).with_name("resources") / "ocr"
    )
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        return None, "domain_model_absent"
    if int(manifest.get("format", 0) or 0) != _DOMAIN_OCR_MANIFEST_FORMAT:
        return None, "domain_manifest_format"
    if manifest.get("integration_authorized") is not True:
        return None, "domain_release_gate_failed"
    model_version = str(manifest.get("model_version", "")).strip()
    files = manifest.get("files")
    evaluations = manifest.get("evaluations")
    if (
        not model_version
        or not isinstance(files, dict)
        or not isinstance(evaluations, dict)
        or manifest.get("detection_runtime_profile")
        != _DOMAIN_OCR_DETECTION_RUNTIME_PROFILE
    ):
        return None, "domain_manifest_contract"
    scan = evaluations.get("registered_scan_test")
    clean = evaluations.get("clean_render_test")
    holdout = evaluations.get("independent_registered_scan_holdout")
    holdout_coverage = manifest.get("independent_holdout_coverage")
    if (
        not isinstance(scan, dict)
        or not isinstance(clean, dict)
        or not isinstance(holdout, dict)
        or not isinstance(holdout_coverage, dict)
    ):
        return None, "domain_evaluation_missing"
    try:
        checks = (
            (float(scan["acc"]), _DOMAIN_OCR_SCAN_ACCURACY_FLOOR),
            (float(scan["norm_edit_dis"]), _DOMAIN_OCR_SCAN_EDIT_FLOOR),
            (float(clean["acc"]), _DOMAIN_OCR_CLEAN_ACCURACY_FLOOR),
            (float(clean["norm_edit_dis"]), _DOMAIN_OCR_CLEAN_EDIT_FLOOR),
            (
                float(scan["precision"]),
                _DOMAIN_OCR_DETECTION_PRECISION_FLOOR,
            ),
            (float(scan["recall"]), _DOMAIN_OCR_DETECTION_RECALL_FLOOR),
            (float(scan["hmean"]), _DOMAIN_OCR_DETECTION_HMEAN_FLOOR),
            (
                float(clean["precision"]),
                _DOMAIN_OCR_DETECTION_PRECISION_FLOOR,
            ),
            (float(clean["recall"]), _DOMAIN_OCR_DETECTION_RECALL_FLOOR),
            (float(clean["hmean"]), _DOMAIN_OCR_DETECTION_HMEAN_FLOOR),
            (float(holdout["acc"]), _DOMAIN_OCR_SCAN_ACCURACY_FLOOR),
            (float(holdout["norm_edit_dis"]), _DOMAIN_OCR_SCAN_EDIT_FLOOR),
            (
                float(holdout["precision"]),
                _DOMAIN_OCR_DETECTION_PRECISION_FLOOR,
            ),
            (
                float(holdout["recall"]),
                _DOMAIN_OCR_DETECTION_RECALL_FLOOR,
            ),
            (
                float(holdout["hmean"]),
                _DOMAIN_OCR_DETECTION_HMEAN_FLOOR,
            ),
        )
        gate_passed = all(
            math.isfinite(actual) and floor <= actual <= 1.0
            for actual, floor in checks
        )
        coverage_passed = (
            int(holdout_coverage["sources"])
            >= _DOMAIN_OCR_HOLDOUT_MINIMUM_SOURCES
            and int(holdout_coverage["words"])
            >= _DOMAIN_OCR_HOLDOUT_MINIMUM_WORDS
            and int(holdout_coverage["pages"])
            >= _DOMAIN_OCR_HOLDOUT_MINIMUM_PAGES
            and float(holdout_coverage["minimum_iou"])
            >= _DOMAIN_OCR_HOLDOUT_MINIMUM_IOU
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "domain_evaluation_invalid"
    if not gate_passed or not coverage_passed:
        return None, "domain_evaluation_below_floor"

    detection_runtime = manifest.get("detection_runtime_parameters")
    if (
        not isinstance(detection_runtime, dict)
        or set(detection_runtime) != _DOMAIN_OCR_DETECTION_PARAMETER_NAMES
        or any(
            detection_runtime.get(name) != expected
            for name, expected in (
                _DOMAIN_OCR_DETECTION_FIXED_PARAMETERS.items()
            )
        )
    ):
        return None, "domain_detection_runtime_parameters"
    try:
        threshold = float(detection_runtime["Det.thresh"])
        box_threshold = float(detection_runtime["Det.box_thresh"])
        unclip_ratio = float(detection_runtime["Det.unclip_ratio"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "domain_detection_runtime_parameters"
    if (
        not 0 < threshold < 1
        or not 0 < box_threshold < 1
        or not 0.5 <= unclip_ratio <= 3.0
    ):
        return None, "domain_detection_runtime_parameters"
    normalized_detection_runtime = dict(detection_runtime)
    normalized_detection_runtime["Det.thresh"] = threshold
    normalized_detection_runtime["Det.box_thresh"] = box_threshold
    normalized_detection_runtime["Det.unclip_ratio"] = unclip_ratio

    resolved: dict[str, Path] = {}
    for role, maximum_bytes in (
        ("recognition_model", _DOMAIN_OCR_MAX_MODEL_BYTES),
        ("recognition_keys", _DOMAIN_OCR_MAX_KEYS_BYTES),
        ("detection_model", _DOMAIN_OCR_MAX_MODEL_BYTES),
    ):
        record = files.get(role)
        if not isinstance(record, dict):
            return None, f"domain_{role}_missing"
        filename = str(record.get("file", "")).strip()
        if not filename or Path(filename).name != filename:
            return None, f"domain_{role}_path"
        path = root / filename
        try:
            expected_bytes = int(record["bytes"])
            actual_bytes = path.stat().st_size
        except (KeyError, TypeError, ValueError, OSError, OverflowError):
            return None, f"domain_{role}_missing"
        if (
            expected_bytes <= 0
            or expected_bytes > maximum_bytes
            or actual_bytes != expected_bytes
        ):
            return None, f"domain_{role}_size"
        try:
            actual_hash = sha256_file(path)
        except OSError:
            return None, f"domain_{role}_missing"
        if actual_hash != str(record.get("sha256", "")):
            return None, f"domain_{role}_hash"
        resolved[role] = path
    return (
        DomainOcrAssets(
            model_path=resolved["recognition_model"],
            keys_path=resolved["recognition_keys"],
            model_version=model_version,
            manifest_path=manifest_path,
            detection_model_path=resolved["detection_model"],
            detection_runtime_parameters=normalized_detection_runtime,
        ),
        "domain_model_verified",
    )


def _corrector() -> DirectionCorrector:
    global _CORRECTOR
    if _CORRECTOR is None:
        _CORRECTOR = DirectionCorrector()
    return _CORRECTOR


def _anchor_classifier() -> DirectionAnchorClassifier:
    global _ANCHOR_CLASSIFIER
    if _ANCHOR_CLASSIFIER is None:
        _ANCHOR_CLASSIFIER = DirectionAnchorClassifier()
    return _ANCHOR_CLASSIFIER


def _requested_ocr_accelerator() -> str:
    requested = os.environ.get(OCR_ACCELERATOR_ENVIRONMENT_VARIABLE, "cpu").strip().casefold()
    if requested not in _OCR_ACCELERATOR_VALUES:
        raise RuntimeError(f"不支持的 OCR 加速设备：{requested}")
    return requested


def _rapidocr_component_providers(engine: object) -> dict[str, list[str]]:
    providers: dict[str, list[str]] = {}
    for name, attribute in (
        ("detection", "text_det"),
        ("classification", "text_cls"),
        ("recognition", "text_rec"),
    ):
        component = getattr(engine, attribute, None)
        wrapped_session = getattr(component, "session", None)
        native_session = getattr(wrapped_session, "session", None)
        getter = getattr(native_session, "get_providers", None)
        if callable(getter):
            providers[name] = [str(value) for value in getter()]
    return providers


def ocr_engine_runtime() -> dict[str, object]:
    """Return the providers actually bound to all RapidOCR model sessions."""

    engine = _engine()
    requested = _requested_ocr_accelerator()
    providers = _rapidocr_component_providers(engine) if engine is not None else {}
    first_providers = {
        values[0]
        for values in providers.values()
        if values
    }
    selected = "cpu"
    verified = bool(providers) and first_providers == {"CPUExecutionProvider"}
    return {
        "requested": requested,
        "selected": selected,
        "verified": verified,
        "component_providers": providers,
        "recognition_model_version": _OCR_ENGINE_MODEL_VERSION,
        "recognition_model_status": _OCR_ENGINE_MODEL_STATUS,
        "detection_model_version": _OCR_ENGINE_MODEL_VERSION,
        "detection_model_status": _OCR_ENGINE_MODEL_STATUS,
    }


def _engine():
    global _OCR_ENGINE, _OCR_ENGINE_ACCELERATOR
    global _OCR_ENGINE_MODEL_STATUS, _OCR_ENGINE_MODEL_VERSION
    if RapidOCR is None:
        return None
    requested = _requested_ocr_accelerator()
    with _OCR_LOCK:
        if _OCR_ENGINE is None or _OCR_ENGINE_ACCELERATOR != requested:
            parameters: dict[str, object] = {
                "Global.log_level": "warning",
                **rapidocr_cpu_parameters(),
            }
            domain_assets, domain_status = _verified_domain_ocr_assets()
            if domain_assets is not None:
                parameters["Rec.model_path"] = str(domain_assets.model_path)
                parameters["Rec.rec_keys_path"] = str(domain_assets.keys_path)
                parameters["Det.model_path"] = str(
                    domain_assets.detection_model_path
                )
                parameters.update(
                    domain_assets.detection_runtime_parameters
                )
                _OCR_ENGINE_MODEL_VERSION = domain_assets.model_version
            else:
                _OCR_ENGINE_MODEL_VERSION = "rapidocr-bundled-default"
            _OCR_ENGINE_MODEL_STATUS = domain_status
            engine = RapidOCR(params=parameters)
            providers = _rapidocr_component_providers(engine)
            missing = [
                name
                for name, values in providers.items()
                if not values or values[0] != "CPUExecutionProvider"
            ]
            if len(providers) != 3 or missing:
                details = ", ".join(
                    f"{name}={values or ['unavailable']}"
                    for name, values in sorted(providers.items())
                )
                raise RuntimeError(
                    "RapidOCR CPU 会话未就绪"
                    f"（{details or '无法读取会话提供程序'}）"
                )
            _OCR_ENGINE = engine
            _OCR_ENGINE_ACCELERATOR = requested
        return _OCR_ENGINE


def _normalize_text(text: str) -> str:
    text = text.strip().replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def _term_key(text: str) -> str:
    return normalize_direction(text).strip(" .,:;()[]{}")


def classify_text(text: str) -> str:
    normalized = _term_key(text)
    if _DYNAMIC_RE.fullmatch(normalized):
        return "dynamic"
    if parse_metronome_mark(text) is not None:
        return "metronome"
    suggestion = _corrector().suggest(_normalize_text(text))
    if suggestion.probability >= 0.76 and suggestion.edit_ratio <= 0.48:
        candidate = _term_key(suggestion.text)
        if _DYNAMIC_RE.fullmatch(candidate):
            return "dynamic"
        if parse_metronome_mark(suggestion.text) is not None:
            return "metronome"
        return "direction"
    if _ALPHA_RE.search(normalized):
        return "text"
    return "other"


def _semantic_region_classes(backend: str | None) -> frozenset[str]:
    """Return source detector roles carried through OCR backend consensus."""

    marker = "rapid-semantic-region:"
    return frozenset(
        token[token.index(marker) + len(marker) :]
        for token in (backend or "").split("+")
        if marker in token and token[token.index(marker) + len(marker) :]
    )


def _allows_generic_direction_writeback(backend: str | None) -> bool:
    """Reject source roles that need a dedicated non-words MusicXML writer."""

    return not (
        _semantic_region_classes(backend)
        & NON_DIRECTION_TEXT_REGION_CLASSES
    )


def _box_center(box: Iterable[Iterable[float]]) -> tuple[float, float]:
    points = np.asarray(list(box), dtype=float)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _box_height(box: Iterable[Iterable[float]]) -> float:
    points = np.asarray(list(box), dtype=float)
    return float(points[:, 1].max() - points[:, 1].min())


def _rapidocr_rows(image_path: Path) -> list[tuple[str, float, list[list[float]]]]:
    engine = _engine()
    if engine is None:
        return []
    with _OCR_LOCK:
        output = engine(str(image_path), text_score=0.38)
    txts = tuple(getattr(output, "txts", ()) or ())
    scores = tuple(getattr(output, "scores", ()) or ())
    boxes = getattr(output, "boxes", None)
    if boxes is None:
        return []
    return [
        (
            str(text),
            float(score),
            [[float(x), float(y)] for x, y in box],
        )
        for text, score, box in zip(txts, scores, boxes, strict=False)
    ]


def _semantic_region_ocr_rows(
    image: np.ndarray,
    regions: Iterable[dict[str, object]],
    layout: PageLayout | None,
    output_dir: Path,
) -> list[tuple[str, float, list[list[float]], str]]:
    """Recognize release-gated text regions on bounded 1024px contact sheets.

    Crops are enlarged before OCR, then every returned point is mapped back to the
    original page.  The semantic class is only a region proposal/backend tag; OCR
    still determines the text and the ordinary musical anchor gates remain
    authoritative.
    """

    if image.ndim != 2:
        raise ValueError("semantic OCR contact sheet expects a grayscale page")
    spacing = 14.0
    if layout is not None:
        values = [float(staff.spacing) for staff in layout.systems if staff.spacing > 0]
        if values:
            spacing = float(np.median(values))
    height, width = image.shape
    proposals: list[tuple[float, str, tuple[int, int, int, int]]] = []
    for payload in regions:
        if not isinstance(payload, dict):
            continue
        class_name = str(payload.get("class_name") or "")
        bbox = payload.get("bbox")
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            class_name not in TEXT_REGION_CLASSES
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or not math.isfinite(confidence)
        ):
            continue
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        except (TypeError, ValueError, OverflowError):
            continue
        margin = max(4, int(round(spacing * 0.65)))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        proposals.append((confidence, class_name, (x1, y1, x2, y2)))
    # A detector bug or a title-heavy page must not create unbounded OCR calls.
    proposals = sorted(
        proposals,
        key=lambda item: (-item[0], item[2][1], item[2][0], item[1]),
    )[:128]
    if not proposals:
        return []

    sheet_size = 1024
    padding = 12
    maximum_sheets = 8
    sheets: list[np.ndarray] = []
    placements: list[
        list[tuple[tuple[int, int, int, int], tuple[int, int], float, str]]
    ] = []
    sheet = np.full((sheet_size, sheet_size), 255, dtype=np.uint8)
    sheet_placements: list[
        tuple[tuple[int, int, int, int], tuple[int, int], float, str]
    ] = []
    cursor_x = padding
    cursor_y = padding
    row_height = 0

    def commit_sheet() -> None:
        nonlocal sheet, sheet_placements, cursor_x, cursor_y, row_height
        if sheet_placements:
            sheets.append(sheet)
            placements.append(sheet_placements)
        sheet = np.full((sheet_size, sheet_size), 255, dtype=np.uint8)
        sheet_placements = []
        cursor_x = padding
        cursor_y = padding
        row_height = 0

    for _confidence, class_name, (x1, y1, x2, y2) in proposals:
        crop = image[y1:y2, x1:x2]
        crop_height, crop_width = crop.shape
        scale = max(1.0, min(3.0, 96.0 / max(float(crop_height), 1.0)))
        scale = min(
            scale,
            (sheet_size - 2 * padding) / max(float(crop_width), 1.0),
            (sheet_size - 2 * padding) / max(float(crop_height), 1.0),
        )
        scaled_width = max(1, int(round(crop_width * scale)))
        scaled_height = max(1, int(round(crop_height * scale)))
        resized = cv2.resize(
            crop,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )
        if cursor_x + scaled_width + padding > sheet_size:
            cursor_x = padding
            cursor_y += row_height + padding
            row_height = 0
        if cursor_y + scaled_height + padding > sheet_size:
            commit_sheet()
            if len(sheets) >= maximum_sheets:
                break
        right = cursor_x + scaled_width
        bottom = cursor_y + scaled_height
        sheet[cursor_y:bottom, cursor_x:right] = resized
        sheet_placements.append(
            (
                (cursor_x, cursor_y, right, bottom),
                (x1, y1),
                scale,
                class_name,
            )
        )
        cursor_x = right + padding
        row_height = max(row_height, scaled_height)
    if len(sheets) < maximum_sheets:
        commit_sheet()

    rows: list[tuple[str, float, list[list[float]], str]] = []
    for index, (contact_sheet, sheet_regions) in enumerate(
        zip(sheets, placements, strict=True)
    ):
        sheet_path = output_dir / f"semantic_regions_{index:02d}.png"
        cv2.imwrite(
            str(sheet_path),
            contact_sheet,
            [cv2.IMWRITE_PNG_COMPRESSION, 2],
        )
        for text, score, box in _rapidocr_rows(sheet_path):
            points = np.asarray(box, dtype=float)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or not np.all(np.isfinite(points))
                or not math.isfinite(float(score))
            ):
                continue
            box_left = float(points[:, 0].min())
            box_top = float(points[:, 1].min())
            box_right = float(points[:, 0].max())
            box_bottom = float(points[:, 1].max())
            box_area = max(0.0, box_right - box_left) * max(
                0.0,
                box_bottom - box_top,
            )
            if box_area <= 0:
                continue
            owned: list[
                tuple[
                    float,
                    tuple[
                        tuple[int, int, int, int],
                        tuple[int, int],
                        float,
                        str,
                    ],
                ]
            ] = []
            for item in sheet_regions:
                left, top, right, bottom = item[0]
                intersection = max(
                    0.0,
                    min(box_right, float(right)) - max(box_left, float(left)),
                ) * max(
                    0.0,
                    min(box_bottom, float(bottom)) - max(box_top, float(top)),
                )
                owned.append((intersection / box_area, item))
            overlap_fraction, owner = max(owned, key=lambda item: item[0])
            # RapidOCR occasionally joins words from adjacent contact-sheet
            # crops.  Such a box has no defensible page coordinate and must not
            # be injected as a misplaced musical direction.
            if overlap_fraction < 0.80:
                continue
            (left, top, _right, _bottom), (source_x, source_y), scale, class_name = owner
            mapped = [
                [
                    source_x
                    + (
                        min(max(float(x), float(left)), float(_right))
                        - left
                    )
                    / scale,
                    source_y
                    + (
                        min(max(float(y), float(top)), float(_bottom))
                        - top
                    )
                    / scale,
                ]
                for x, y in box
            ]
            rows.append(
                (
                    text,
                    score,
                    mapped,
                    f"rapid-semantic-region:{class_name}",
                )
            )
    return rows


@lru_cache(maxsize=1)
def _tesseract_language_spec() -> str:
    executable = shutil.which("tesseract")
    if not executable:
        return "eng"
    try:
        completed = subprocess.run(
            [executable, "--list-langs"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20,
        )
        available = {line.strip() for line in completed.stdout.splitlines()[1:] if line.strip()}
    except Exception:
        available = set()
    selected = [language for language in ("eng", "ita", "deu", "fra") if language in available]
    return "+".join(selected) if selected else "eng"


def _run_tesseract_tsv(command: list[str]) -> str | None:
    """Run Tesseract without retaining unbounded native output in memory.

    The TSV stream is spooled to disk, observed while the process runs, and
    rejected if it exceeds the fixed budget. Timeout/overflow termination is
    applied to the complete process tree so native OCR grandchildren cannot
    outlive the page job.
    """
    with tempfile.TemporaryDirectory(prefix="scorescan-tesseract-") as temp_dir:
        output_path = Path(temp_dir) / "output.tsv"
        process: subprocess.Popen[bytes] | None = None
        try:
            with output_path.open("wb") as output_handle:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=output_handle,
                    stderr=subprocess.DEVNULL,
                    **popen_group_options(),
                )
                deadline = time.monotonic() + _TESSERACT_TIMEOUT_SECONDS
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        terminate_process_tree(process, grace_seconds=0.5)
                        process.wait(timeout=5)
                        return None
                    try:
                        if output_path.stat().st_size > _TESSERACT_MAX_TSV_BYTES:
                            terminate_process_tree(process, grace_seconds=0.5)
                            process.wait(timeout=5)
                            return None
                    except FileNotFoundError:
                        pass
                    time.sleep(_TESSERACT_POLL_SECONDS)
                return_code = process.wait(timeout=5)
            if return_code != 0:
                return None
            try:
                size = output_path.stat().st_size
            except OSError:
                return None
            if size > _TESSERACT_MAX_TSV_BYTES:
                return None
            # Normalise CRLF/LF so downstream parsing and diagnostics are
            # deterministic across Windows and POSIX hosts.
            with output_path.open("r", encoding="utf-8", errors="replace", newline=None) as handle:
                text = handle.read(_TESSERACT_MAX_TSV_BYTES + 1)
            if len(text.encode("utf-8", errors="replace")) > _TESSERACT_MAX_TSV_BYTES:
                return None
            return text
        except (OSError, subprocess.SubprocessError):
            if process is not None and process.poll() is None:
                terminate_process_tree(process, grace_seconds=0.5)
                try:
                    process.wait(timeout=5)
                except subprocess.SubprocessError:
                    pass
            return None


def _tesseract_rows(image_path: Path) -> list[tuple[str, float, list[list[float]]]]:
    executable = shutil.which("tesseract")
    if not executable:
        return []
    command = [executable, str(image_path), "stdout", "-l", _tesseract_language_spec(), "--oem", "1", "--psm", "11", "-c", "user_defined_dpi=300", "tsv"]
    output = _run_tesseract_tsv(command)
    if output is None:
        return []
    results: list[tuple[str, float, list[list[float]]]] = []
    try:
        rows = csv.DictReader(io.StringIO(output), delimiter="\t")
        for index, row in enumerate(rows):
            if index >= _TESSERACT_MAX_ROWS:
                break
            text = (row.get("text") or "").strip()
            if not text or len(text) > _TESSERACT_MAX_TEXT_CHARS:
                continue
            try:
                confidence = max(0.0, float(row.get("conf") or 0) / 100.0)
                left, top = int(row["left"]), int(row["top"])
                width, height = int(row["width"]), int(row["height"])
            except (ValueError, TypeError, KeyError):
                continue
            if width <= 0 or height <= 0 or left < 0 or top < 0:
                continue
            results.append((text, confidence, [[left, top], [left + width, top], [left + width, top + height], [left, top + height]]))
    except csv.Error:
        return []
    return results


def _box_bounds(box: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    points = np.asarray(list(box), dtype=float)
    return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())


def _box_iou(a: Iterable[Iterable[float]], b: Iterable[Iterable[float]]) -> float:
    ax1, ay1, ax2, ay2 = _box_bounds(a)
    bx1, by1, bx2, by2 = _box_bounds(b)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return intersection / max(area_a + area_b - intersection, 1.0)


def _line_suppressed(gray: np.ndarray) -> np.ndarray:
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    width = gray.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 32), 1))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.subtract(binary, lines)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return cv2.bitwise_not(cleaned)


def _source_dynamic_rows(
    image: np.ndarray,
    layout: PageLayout | None,
) -> list[tuple[str, float, list[list[float]], str]]:
    """Recover common printed dynamics above or below every physical staff."""

    if layout is None or not layout.systems:
        return []
    binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    rows: list[tuple[str, float, list[list[float]], str]] = []
    emitted: set[tuple[str, int, int, int, int]] = set()
    for system in layout.systems:
        spacing = max(float(system.spacing), 1.0)
        bands = (
            (
                max(0, int(np.floor(float(system.line_y[0]) - spacing * 5.0))),
                max(0, int(np.ceil(float(system.line_y[0]) - spacing * 0.35))),
            ),
            (
                min(binary.shape[0], int(np.floor(float(system.line_y[-1]) + spacing * 0.35))),
                min(binary.shape[0], int(np.ceil(float(system.line_y[-1]) + spacing * 5.0))),
            ),
        )
        for band_top, band_bottom in bands:
            if band_bottom <= band_top:
                continue
            crop = binary[band_top:band_bottom, :]
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                (crop > 0).astype(np.uint8),
                8,
            )
            components: list[dict[str, float]] = []
            for label_id, (x, y, width, height, area) in enumerate(
                stats[1:count],
                start=1,
            ):
                if width <= 0 or height <= 0:
                    continue
                component_mask = labels[y:y + height, x:x + width] == label_id
                split_y = max(1, height // 2)
                split_x = max(1, width // 2)
                quadrants = (
                    int(np.count_nonzero(component_mask[:split_y, :split_x])),
                    int(np.count_nonzero(component_mask[:split_y, split_x:])),
                    int(np.count_nonzero(component_mask[split_y:, :split_x])),
                    int(np.count_nonzero(component_mask[split_y:, split_x:])),
                )
                components.append(
                    {
                        "x": float(x),
                        "y": float(y + band_top),
                        "width": float(width),
                        "height": float(height),
                        "area": float(area),
                        "fill": float(area) / float(width * height),
                        "center_y": float(y + band_top) + float(height) / 2.0,
                        "q_tl": quadrants[0] / max(float(area), 1.0),
                        "q_tr": quadrants[1] / max(float(area), 1.0),
                        "q_bl": quadrants[2] / max(float(area), 1.0),
                        "q_br": quadrants[3] / max(float(area), 1.0),
                    }
                )

            horizontal_minimum = float(system.left) + spacing * 2.5
            horizontal_maximum = float(system.right) + spacing

            def inside_system(component: dict[str, float]) -> bool:
                return (
                    component["x"] >= horizontal_minimum
                    and component["x"] + component["width"] <= horizontal_maximum
                )

            f_components = [
                component
                for component in components
                if inside_system(component)
                and 1.55 * spacing <= component["height"] <= 2.8 * spacing
                and 1.2 * spacing <= component["width"] <= 2.5 * spacing
                and 0.16 <= component["fill"] <= 0.34
                and 0.72 * spacing * spacing
                <= component["area"]
                <= 1.55 * spacing * spacing
                and component["q_tl"] <= 0.18
                and component["q_tr"] >= 0.32
                and component["q_bl"] >= 0.28
                and component["q_br"] <= 0.16
            ]
            p_components = [
                component
                for component in components
                if inside_system(component)
                and 1.55 * spacing <= component["height"] <= 2.15 * spacing
                and 1.35 * spacing <= component["width"] <= 2.2 * spacing
                and 0.32 <= component["fill"] <= 0.50
                and 0.75 * spacing * spacing
                <= component["area"]
                <= 1.60 * spacing * spacing
            ]

            consumed_small_components: set[int] = set()
            for f_component in f_components:
                prefix_matches = [
                    (index, component)
                    for index, component in enumerate(components)
                    if index not in consumed_small_components
                    and 0.45 * spacing <= component["width"] <= 1.35 * spacing
                    and 0.60 * spacing <= component["height"] <= 1.45 * spacing
                    and 0.26 <= component["fill"] <= 0.78
                    and f_component["x"] - 0.85 * spacing
                    <= component["x"]
                    <= f_component["x"] + 0.15 * spacing
                    and f_component["y"] + 0.05 * spacing
                    <= component["center_y"]
                    <= f_component["y"] + f_component["height"] + 0.2 * spacing
                ]
                if prefix_matches:
                    prefix_index, prefix = max(
                        prefix_matches,
                        key=lambda item: item[1]["area"],
                    )
                    consumed_small_components.add(prefix_index)
                    # A wider lower-case m preceding f is distinguishable from the
                    # compact s used by sf at this scale.
                    text = "mf" if prefix["width"] >= 0.78 * spacing else "sf"
                    left = min(f_component["x"], prefix["x"])
                    top = min(f_component["y"], prefix["y"])
                    right = max(
                        f_component["x"] + f_component["width"],
                        prefix["x"] + prefix["width"],
                    )
                    bottom = max(
                        f_component["y"] + f_component["height"],
                        prefix["y"] + prefix["height"],
                    )
                    confidence = 0.99
                else:
                    text = "f"
                    left = f_component["x"]
                    top = f_component["y"]
                    right = left + f_component["width"]
                    bottom = top + f_component["height"]
                    confidence = 0.985
                key = (text, round(left), round(top), round(right), round(bottom))
                if key not in emitted:
                    emitted.add(key)
                    rows.append(
                        (
                            text,
                            confidence,
                            [[left, top], [right, top], [right, bottom], [left, bottom]],
                            "source-dynamic-geometry-v2",
                        )
                    )

            for component in p_components:
                left = component["x"]
                top = component["y"]
                right = left + component["width"]
                bottom = top + component["height"]
                key = ("p", round(left), round(top), round(right), round(bottom))
                if key not in emitted:
                    emitted.add(key)
                    rows.append(
                        (
                            "p",
                            0.97,
                            [[left, top], [right, top], [right, bottom], [left, bottom]],
                            "source-dynamic-geometry",
                        )
                    )
    return rows


def _row_quality(text: str, score: float) -> float:
    suggestion = _corrector().suggest(text)
    musical_bonus = 0.20 * suggestion.probability
    alpha_bonus = 0.05 if _ALPHA_RE.search(text) else 0.0
    return float(score) + musical_bonus + alpha_bonus


def _merge_ocr_rows(
    rows: list[tuple[str, float, list[list[float]], str]],
) -> list[tuple[str, float, list[list[float]], str]]:
    groups: list[list[tuple[str, float, list[list[float]], str]]] = []
    for row in sorted(rows, key=lambda item: (min(p[1] for p in item[2]), min(p[0] for p in item[2]))):
        placed = False
        for group in groups:
            if any(_box_iou(row[2], existing[2]) >= 0.32 for existing in group):
                group.append(row)
                placed = True
                break
        if not placed:
            groups.append([row])

    merged: list[tuple[str, float, list[list[float]], str]] = []
    for group in groups:
        counts: dict[str, int] = {}
        for item in group:
            key = normalize_direction(item[0])
            counts[key] = counts.get(key, 0) + 1
        winning_key, winning_count = max(counts.items(), key=lambda item: (item[1], len(item[0])))
        pool = [item for item in group if normalize_direction(item[0]) == winning_key] if winning_count >= 2 else group
        best = max(pool, key=lambda item: _row_quality(item[0], item[1]))
        # Agreement between independent passes increases confidence, but never above 0.995.
        normalized = normalize_direction(best[0])
        agreements = sum(normalize_direction(item[0]) == normalized for item in group)
        confidence = min(0.995, best[1] + max(0, agreements - 1) * 0.055)
        # Record only backends agreeing with the selected text.  Merely overlapping
        # boxes with contradictory labels must not masquerade as independent support.
        backends = "+".join(
            sorted(
                {
                    item[3]
                    for item in group
                    if normalize_direction(item[0]) == normalized
                }
            )
        )
        merged.append((best[0], confidence, best[2], backends))
    return merged



def _vertical_overlap_ratio(a: Iterable[Iterable[float]], b: Iterable[Iterable[float]]) -> float:
    _, ay1, _, ay2 = _box_bounds(a)
    _, by1, _, by2 = _box_bounds(b)
    overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    return overlap / max(1.0, min(ay2 - ay1, by2 - by1))


def _merge_boxes(boxes: Iterable[Iterable[Iterable[float]]]) -> list[list[float]]:
    bounds = [_box_bounds(box) for box in boxes]
    x1 = min(item[0] for item in bounds)
    y1 = min(item[1] for item in bounds)
    x2 = max(item[2] for item in bounds)
    y2 = max(item[3] for item in bounds)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _remove_contained_rows(
    rows: list[tuple[str, float, list[list[float]], str]],
) -> list[tuple[str, float, list[list[float]], str]]:
    kept: list[tuple[str, float, list[list[float]], str]] = []
    for index, row in enumerate(rows):
        rx1, ry1, rx2, ry2 = _box_bounds(row[2])
        area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
        discard = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            ox1, oy1, ox2, oy2 = _box_bounds(other[2])
            intersection = max(0.0, min(rx2, ox2) - max(rx1, ox1)) * max(0.0, min(ry2, oy2) - max(ry1, oy1))
            containment = intersection / area
            if containment < 0.86 or (ox2 - ox1) <= (rx2 - rx1) * 1.15:
                continue
            larger_suggestion = _corrector().suggest(other[0])
            row_key = normalize_direction(row[0])
            larger_key = normalize_direction(other[0])
            if row_key and (row_key in larger_key or larger_suggestion.probability >= 0.78) and other[1] >= row[1] - 0.18:
                discard = True
                break
        if not discard:
            kept.append(row)
    return kept


def _assemble_phrase_rows(
    rows: list[tuple[str, float, list[list[float]], str]],
    layout: PageLayout | None = None,
) -> list[tuple[str, float, list[list[float]], str]]:
    """Join adjacent word boxes into musical direction phrases.

    OCR backends disagree on whether a direction is emitted as one phrase or several
    words. This pass chooses a conservative line partition using the trained music
    lexicon. It never joins across a large horizontal gap or a different baseline.
    """
    rows = _remove_contained_rows(rows)
    pending = sorted(rows, key=lambda item: ((_box_bounds(item[2])[1] + _box_bounds(item[2])[3]) / 2, _box_bounds(item[2])[0]))
    lines: list[list[tuple[str, float, list[list[float]], str]]] = []
    line_buckets: list[tuple[int | None, str | None]] = []
    passthrough: list[tuple[str, float, list[list[float]], str]] = []
    for row in pending:
        center_x, center_y = _box_center(row[2])
        bucket = (None, None)
        if layout is not None:
            system_index, placement, _distance = _nearest_system(center_x, center_y, layout)
            bucket = (system_index, placement)
            # OCR fragments inside the five staff lines must never be combined into a
            # plausible-looking direction phrase. Keep them isolated so later role
            # evidence can reject them conservatively.
            if placement == "within":
                passthrough.append(row)
                continue
        best_line = None
        best_line_index = -1
        best_distance = float("inf")
        for line_index, line in enumerate(lines):
            if layout is not None and line_buckets[line_index] != bucket:
                continue
            line_center = np.mean([(_box_bounds(item[2])[1] + _box_bounds(item[2])[3]) / 2 for item in line])
            overlap = max(_vertical_overlap_ratio(row[2], item[2]) for item in line)
            distance = abs(center_y - line_center)
            if overlap >= 0.45 and distance < best_distance:
                best_line = line
                best_line_index = line_index
                best_distance = distance
        if best_line is None:
            lines.append([row])
            line_buckets.append(bucket)
        else:
            lines[best_line_index].append(row)

    output: list[tuple[str, float, list[list[float]], str]] = list(passthrough)
    for line in lines:
        line.sort(key=lambda item: _box_bounds(item[2])[0])
        index = 0
        while index < len(line):
            best_end = index + 1
            best_value: tuple[float, str, float, list[list[float]], str] | None = None
            max_end = min(len(line), index + 6)
            previous_right = _box_bounds(line[index][2])[2]
            heights = [_box_height(line[index][2])]
            for end_index in range(index + 1, max_end):
                current = line[end_index]
                left = _box_bounds(current[2])[0]
                heights.append(_box_height(current[2]))
                gap = left - previous_right
                gap_limit = max(14.0, np.median(heights) * 2.6)
                if gap > gap_limit:
                    break
                previous_right = _box_bounds(current[2])[2]
                group = line[index:end_index + 1]
                text = " ".join(_normalize_text(item[0]) for item in group).strip()
                if not text:
                    continue
                suggestion = _corrector().suggest(text)
                kind = classify_text(text)
                metronome_swallowed_mark = (
                    kind == "metronome"
                    and any(
                        classify_text(item[0]) in {"dynamic", "direction"}
                        for item in group[1:]
                    )
                )
                token_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text))
                individual_best = max(_corrector().suggest(item[0]).probability for item in group)
                phrase_does_not_degrade = suggestion.probability >= individual_best - 0.025
                rows_have_text = all(
                    _ALPHA_RE.search(item[0]) or any(character.isdigit() for character in item[0])
                    for item in group
                )
                qualifies = (
                    (kind == "metronome" and not metronome_swallowed_mark)
                    or (
                        rows_have_text
                        and token_count >= 2
                        and suggestion.probability >= 0.79
                        and phrase_does_not_degrade
                    )
                    or (
                        rows_have_text
                        and token_count >= 2
                        and phrase_does_not_degrade
                        and any(normalize_direction(item) in normalize_direction(suggestion.text) for item in ("a tempo", "poco a poco", "con brio"))
                    )
                )
                if not qualifies:
                    continue
                confidence = min(0.995, sum(item[1] for item in group) / len(group) + 0.035)
                objective = suggestion.probability + min(0.09, token_count * 0.015) + confidence * 0.08
                backend = "phrase:" + "+".join(sorted({item[3] for item in group}))
                value = (objective, text, confidence, _merge_boxes(item[2] for item in group), backend)
                if best_value is None or value[0] > best_value[0] or (value[0] == best_value[0] and end_index + 1 > best_end):
                    best_value = value
                    best_end = end_index + 1
            if best_value is not None:
                _, text, confidence, box, backend = best_value
                output.append((text, confidence, box, backend))
                index = best_end
            else:
                output.append(line[index])
                index += 1
    return sorted(output, key=lambda item: (min(p[1] for p in item[2]), min(p[0] for p in item[2])))


def _staff_neighborhood_mask(gray: np.ndarray, layout: PageLayout | None) -> np.ndarray:
    """Keep only staff-adjacent bands where musical directions are expected.

    The image remains page-sized, so OCR coordinates do not require remapping.  A
    full-page pass is still retained; this targeted pass adds recall while preventing
    titles, page numbers, and copyright text from dominating sparse-text OCR.
    """
    if layout is None or not layout.systems:
        return gray.copy()
    masked = np.full_like(gray, 255)
    height = gray.shape[0]
    for system in layout.systems:
        top = max(0, int(system.line_y[0] - system.spacing * 11.0))
        bottom = min(height, int(system.line_y[-1] + system.spacing * 9.0))
        masked[top:bottom, :] = gray[top:bottom, :]
    return masked


def _ocr_contrast_variant(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(12, 12))
    enhanced = clahe.apply(gray)
    return cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def run_ocr(
    image_path: Path,
    layout: PageLayout | None = None,
    semantic_text_regions: Iterable[dict[str, object]] | None = None,
) -> tuple[list[tuple[str, float, list[list[float]], str]], str]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return [], "unavailable"
    rows: list[tuple[str, float, list[list[float]], str]] = []
    available_backends: set[str] = set()
    source_dynamic_rows = _source_dynamic_rows(image, layout)
    if source_dynamic_rows:
        rows.extend(source_dynamic_rows)
        available_backends.add("source-dynamic-geometry")
    with tempfile.TemporaryDirectory(prefix="scorescan-ocr-") as temp_dir:
        temp = Path(temp_dir)
        suppressed_path = temp / "line_suppressed.png"
        targeted_path = temp / "staff_targeted.png"
        targeted_suppressed_path = temp / "staff_targeted_no_lines.png"
        targeted_contrast_path = temp / "staff_targeted_contrast.png"
        suppressed = _line_suppressed(image)
        targeted = _staff_neighborhood_mask(image, layout)
        targeted_suppressed = _line_suppressed(targeted)
        cv2.imwrite(str(suppressed_path), suppressed, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        cv2.imwrite(str(targeted_path), targeted, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        cv2.imwrite(str(targeted_suppressed_path), targeted_suppressed, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        cv2.imwrite(str(targeted_contrast_path), _ocr_contrast_variant(targeted_suppressed), [cv2.IMWRITE_PNG_COMPRESSION, 2])
        try:
            rows.extend((text, score, box, "rapid-original") for text, score, box in _rapidocr_rows(image_path))
            rows.extend((text, score, box, "rapid-no-lines") for text, score, box in _rapidocr_rows(suppressed_path))
            rows.extend((text, score, box, "rapid-staff") for text, score, box in _rapidocr_rows(targeted_suppressed_path))
            rows.extend((text, score, box, "rapid-staff-contrast") for text, score, box in _rapidocr_rows(targeted_contrast_path))
            # A successful OCR call with no rows means that the page contains no
            # detectable text.  It does not mean that the bundled backend is
            # unavailable.
            available_backends.add("rapidocr")
            if semantic_text_regions is not None:
                semantic_rows = _semantic_region_ocr_rows(
                    image,
                    semantic_text_regions,
                    layout,
                    temp,
                )
                if semantic_rows:
                    rows.extend(semantic_rows)
                    available_backends.add("rapid-semantic-region")
        except Exception:
            pass
        if shutil.which("tesseract"):
            try:
                rows.extend((text, score, box, "tesseract-no-lines") for text, score, box in _tesseract_rows(suppressed_path))
                rows.extend((text, score, box, "tesseract-staff") for text, score, box in _tesseract_rows(targeted_suppressed_path))
                rows.extend((text, score, box, "tesseract-staff-contrast") for text, score, box in _tesseract_rows(targeted_contrast_path))
                available_backends.add("tesseract")
            except Exception:
                pass
    if not rows:
        return [], "+".join(sorted(available_backends)) or "unavailable"
    merged = _assemble_phrase_rows(_merge_ocr_rows(rows), layout)
    return merged, "+".join(sorted({row[3] for row in merged}))


def _nearest_system(x: float, y: float, layout: PageLayout) -> tuple[int | None, str | None, float]:
    if not layout.systems:
        return None, None, float("inf")
    ranked: list[tuple[float, int, str]] = []
    for index, system in enumerate(layout.systems):
        if system.left - 4 * system.spacing <= x <= system.right + 4 * system.spacing:
            if y < system.line_y[0]:
                distance, placement = system.line_y[0] - y, "above"
            elif y > system.line_y[-1]:
                distance, placement = y - system.line_y[-1], "below"
            else:
                distance, placement = 0.0, "within"
            ranked.append((distance / max(system.spacing, 1.0), index, placement))
    if not ranked:
        return None, None, float("inf")
    normalized_distance, index, placement = min(ranked)
    # Text farther than twelve staff spaces is more likely title/header/footer metadata.
    if normalized_distance > 12.0:
        return None, placement, normalized_distance
    return index, placement, normalized_distance


def _xml_system_measure_groups(part: etree._Element) -> list[list[etree._Element]]:
    groups: list[list[etree._Element]] = [[]]
    for measure in part.findall("measure"):
        print_node = measure.find("print")
        if groups[-1] and print_node is not None and print_node.get("new-system") == "yes":
            groups.append([])
        groups[-1].append(measure)
    return [group for group in groups if group]


def _measure_duration(measure: etree._Element) -> int:
    cursor = 0
    maximum = 0
    last_anchor = 0
    for child in measure:
        if child.tag == "note":
            chord = child.find("chord") is not None
            grace = child.find("grace") is not None
            onset = last_anchor if chord else cursor
            if not chord:
                last_anchor = onset
            try:
                duration = max(0, int(child.findtext("duration") or 0))
            except ValueError:
                duration = 0
            maximum = max(maximum, onset + duration)
            if not chord and not grace:
                cursor += duration
        elif child.tag == "forward":
            try:
                cursor += max(0, int(child.findtext("duration") or 0))
                maximum = max(maximum, cursor)
            except ValueError:
                pass
        elif child.tag == "backup":
            try:
                cursor = max(0, cursor - max(0, int(child.findtext("duration") or 0)))
            except ValueError:
                pass
    return max(maximum, 1)


def _part_staff_count(part: etree._Element) -> int:
    result = 1
    for value in part.xpath("./measure/attributes/staves/text()"):
        try:
            result = max(result, int(value))
        except (TypeError, ValueError):
            pass
    for value in part.xpath("./measure/note/staff/text()"):
        try:
            result = max(result, int(value))
        except (TypeError, ValueError):
            pass
    return result


def _source_staff_target(
    layout: PageLayout,
    source_staff_position: int,
    part_staff_counts: list[int],
) -> tuple[int, int, int, StaffSystem] | None:
    """Map one physical page staff to (score system, part, part-local staff)."""

    if not 0 <= source_staff_position < len(layout.systems):
        return None
    physical = layout.systems[source_staff_position]
    score_systems = layout.effective_score_systems
    score_system_index = next(
        (
            index
            for index, score_system in enumerate(score_systems)
            if physical.index in score_system.staff_indices
        ),
        None,
    )
    if score_system_index is None:
        return None
    ordered_indices = score_systems[score_system_index].staff_indices
    if len(ordered_indices) != sum(part_staff_counts):
        # Never guess a part/staff association when the candidate topology and
        # source page disagree.
        return None
    try:
        ordinal = ordered_indices.index(physical.index)
    except ValueError:
        return None
    base = 0
    for part_index, count in enumerate(part_staff_counts):
        if ordinal < base + count:
            return score_system_index, part_index, ordinal - base + 1, physical
        base += count
    return None


def _visual_notehead_columns(
    binary: np.ndarray,
    system: StaffSystem,
    *,
    music_left: float | None = None,
) -> tuple[float, ...]:
    spacing = max(float(system.spacing), 1.0)
    note_region_left = float(
        system.left if music_left is None else music_left
    ) + 7.0 * spacing
    distance = cv2.distanceTransform(
        (binary > 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    core = (distance >= max(1.0, spacing * 0.23)).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(core, 8)
    points: list[float] = []
    for x, y, width, height, area in stats[1:count]:
        center_x = float(x) + float(width) / 2.0
        center_y = float(y) + float(height) / 2.0
        if not (
            note_region_left <= center_x <= float(system.right)
            and float(system.line_y[0]) - 1.15 * spacing
            <= center_y
            <= float(system.line_y[-1]) + 1.15 * spacing
            and 0.68 * spacing <= float(width) <= 1.4 * spacing
            and 0.50 * spacing <= float(height) <= 1.2 * spacing
            and 0.32 * spacing * spacing
            <= float(area)
            <= 1.2 * spacing * spacing
            and 0.60 <= float(width) / max(float(height), 1.0) <= 1.8
        ):
            continue
        points.append(center_x)

    # Recover hollow half/whole noteheads from enclosed white regions.  A staff
    # line commonly splits the hollow centre into two small child contours; requiring
    # a compact multi-contour x cluster avoids treating accidentals and text holes as
    # noteheads.  The bounded left exclusion removes clef/key/time-signature holes.
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    hole_points: list[float] = []
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            if int(hierarchy[0][index][3]) < 0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            center_x = float(x) + float(width) / 2.0
            center_y = float(y) + float(height) / 2.0
            if not (
                note_region_left <= center_x <= float(system.right)
                and float(system.line_y[0]) - 1.15 * spacing
                <= center_y
                <= float(system.line_y[-1]) + 1.15 * spacing
                and 0.32 * spacing <= float(width) <= 1.15 * spacing
                and 0.25 * spacing <= float(height) <= 1.05 * spacing
            ):
                continue
            hole_points.append(center_x)
        hole_groups: list[list[float]] = []
        for point in sorted(hole_points):
            if not hole_groups or point - hole_groups[-1][-1] > 0.75 * spacing:
                hole_groups.append([point])
            else:
                hole_groups[-1].append(point)
        points.extend(
            float(np.mean(group))
            for group in hole_groups
            if len(group) >= 2 and max(group) - min(group) <= 1.15 * spacing
        )

    groups: list[list[float]] = []
    for point in sorted(points):
        if not groups or point - groups[-1][-1] > 0.45 * spacing:
            groups.append([point])
        else:
            groups[-1].append(point)
    return tuple(float(np.mean(group)) for group in groups)


def _pitched_measure_onsets(
    measure: etree._Element,
    staff_number: int | None = None,
) -> tuple[int, ...]:
    cursor = 0
    last_anchor = 0
    onsets: set[int] = set()
    for child in measure:
        if child.tag == "note":
            chord = child.find("chord") is not None
            grace = child.find("grace") is not None
            onset = last_anchor if chord else cursor
            if not chord:
                last_anchor = onset
            try:
                note_staff = int(child.findtext("staff") or 1)
            except ValueError:
                note_staff = 1
            if (
                child.find("pitch") is not None
                and (staff_number is None or note_staff == staff_number)
            ):
                onsets.add(max(0, onset))
            if not chord and not grace:
                try:
                    cursor += max(0, int(child.findtext("duration") or 0))
                except ValueError:
                    pass
        elif child.tag == "backup":
            try:
                cursor = max(0, cursor - max(0, int(child.findtext("duration") or 0)))
            except ValueError:
                pass
        elif child.tag == "forward":
            try:
                cursor += max(0, int(child.findtext("duration") or 0))
            except ValueError:
                pass
    return tuple(sorted(onsets))


def _snap_direction_to_notehead(
    *,
    x: float,
    system: StaffSystem | ScoreSystemLayout,
    local_index: int,
    group: list[etree._Element],
    notehead_columns: tuple[float, ...],
    staff_number: int | None = None,
) -> float | None:
    bounds = system_measure_bounds(system)
    if len(bounds) != len(group) or not 0 <= local_index < len(bounds):
        return None
    left, right = bounds[local_index]
    source_columns = tuple(
        column
        for column in notehead_columns
        if float(left) <= column < float(right)
    )
    target_onsets = _pitched_measure_onsets(
        group[local_index],
        staff_number,
    )
    if not target_onsets:
        return None
    relative_x = (float(x) - float(left)) / max(float(right - left), 1.0)
    if relative_x <= 0.22:
        return target_onsets[0] / max(
            float(_measure_duration(group[local_index])),
            1.0,
        )
    if not source_columns:
        return None
    # Hollow half/whole noteheads do not contain the thick distance-transform
    # core used by the high-precision detector.  When the direction sits clearly
    # before the first detected filled notehead, it belongs to the first pitched
    # onset rather than to a proportional point inside the measure.  This handles
    # the common ``f``/``mf`` aligned with an opening half note without relaxing
    # detection for later onsets.
    if float(x) <= source_columns[0] - max(
        float(system.spacing) * 1.5,
        float(right - left) * 0.08,
    ):
        return target_onsets[0] / max(
            float(_measure_duration(group[local_index])),
            1.0,
        )
    nearest_index = min(
        range(len(source_columns)),
        key=lambda index: abs(source_columns[index] - float(x)),
    )
    tolerance = max(
        float(system.spacing) * 2.4,
        float(right - left) * 0.18,
    )
    if abs(source_columns[nearest_index] - float(x)) > tolerance:
        return None
    if len(source_columns) == len(target_onsets):
        target_index = nearest_index
    elif len(source_columns) == 1 and abs(
        source_columns[nearest_index] - float(x)
    ) <= max(
        float(system.spacing) * 1.2,
        float(right - left) * 0.06,
    ):
        # The direction itself supplies an exact x anchor even when OMR added or
        # dropped an event. Preserve order and map the nearest visible notehead to
        # the closest target onset rank.
        target_index = 0
    else:
        return None
    return target_onsets[target_index] / max(
        float(_measure_duration(group[local_index])),
        1.0,
    )


def _direction_dedup_key(text: str, kind: str) -> str:
    if kind == "metronome":
        mark = parse_metronome_mark(text)
        if mark is not None:
            high = mark.per_minute_high if mark.per_minute_high is not None else mark.per_minute_low
            return (
                f"metronome:{mark.beat_unit}:{int(mark.dotted)}:"
                f"{mark.per_minute_low}:{high}"
            )
    return f"{kind}:{_term_key(text)}"


def _existing_direction_keys(measure: etree._Element) -> set[str]:
    keys: set[str] = set()
    for direction in measure.findall("direction"):
        for words in direction.findall("./direction-type/words"):
            if words.text:
                keys.add(_direction_dedup_key(words.text, "direction"))
        dynamics = direction.find("./direction-type/dynamics")
        if dynamics is not None and len(dynamics):
            child = dynamics[0]
            keys.add(_direction_dedup_key(child.text or child.tag, "dynamic"))
        metronome = direction.find("./direction-type/metronome")
        if metronome is not None:
            beat = metronome.findtext("beat-unit") or "quarter"
            dotted = metronome.find("beat-unit-dot") is not None
            per_minute = (metronome.findtext("per-minute") or "").strip()
            if per_minute:
                text = f"{'dotted ' if dotted else ''}{beat} = {per_minute}"
                keys.add(_direction_dedup_key(text, "metronome"))
    return keys


def _relocatable_direction_key(
    direction: etree._Element,
) -> str | None:
    """Return an exact-content key for source-proven position repair.

    Dynamics are reconciled by their separate complete-inventory transaction.
    Wedges and other graphical directions are deliberately excluded here.
    """

    if direction.find("./direction-type/dynamics") is not None:
        return None
    metronome = direction.find("./direction-type/metronome")
    if metronome is not None:
        beat = metronome.findtext("beat-unit") or "quarter"
        dotted = metronome.find("beat-unit-dot") is not None
        per_minute = (metronome.findtext("per-minute") or "").strip()
        if not per_minute:
            return None
        text = f"{'dotted ' if dotted else ''}{beat} = {per_minute}"
        return _direction_dedup_key(text, "metronome")
    words = direction.find("./direction-type/words")
    if words is None or not (words.text or "").strip():
        return None
    return _direction_dedup_key(words.text or "", "direction")


def _parse_metronome(text: str) -> tuple[str, int, int | None] | None:
    mark = parse_metronome_mark(text)
    if mark is None:
        return None
    unit = mark.beat_unit + ("-dot" if mark.dotted else "")
    return unit, mark.per_minute_low, mark.per_minute_high


def _insert_direction(
    measure: etree._Element,
    text: str,
    kind: str,
    offset_ratio: float,
    placement: str,
    *,
    staff_number: int | None = None,
) -> bool:
    normalized = _term_key(text)
    if _direction_dedup_key(text, kind) in _existing_direction_keys(measure):
        return False

    direction = etree.Element("direction", placement="below" if placement == "below" else "above")
    direction_type = etree.SubElement(direction, "direction-type")
    sound_tempo: int | None = None
    if kind == "dynamic" and _DYNAMIC_RE.fullmatch(normalized):
        dynamics = etree.SubElement(direction_type, "dynamics")
        if normalized in _DYNAMIC_TAGS:
            etree.SubElement(dynamics, normalized)
        else:
            etree.SubElement(dynamics, "other-dynamics").text = text
    elif kind == "metronome" and (parsed := _parse_metronome(text)) is not None:
        unit, low, high = parsed
        metronome = etree.SubElement(direction_type, "metronome", parentheses="no")
        beat_unit = unit.replace("-dot", "")
        etree.SubElement(metronome, "beat-unit").text = beat_unit
        if unit.endswith("-dot"):
            etree.SubElement(metronome, "beat-unit-dot")
        etree.SubElement(metronome, "per-minute").text = str(low) if high is None else f"{low}-{high}"
        sound_tempo = low
    else:
        words = etree.SubElement(direction_type, "words")
        words.text = text
        if kind == "direction":
            words.set("font-style", "italic")

    duration = _measure_duration(measure)
    offset = int(round(max(0.0, min(0.98, offset_ratio)) * duration))
    if offset > 0:
        etree.SubElement(direction, "offset").text = str(offset)
    if staff_number is not None:
        etree.SubElement(direction, "staff").text = str(max(1, int(staff_number)))
    if sound_tempo is not None:
        etree.SubElement(direction, "sound", tempo=str(sound_tempo))

    insert_at = 0
    for index, child in enumerate(measure):
        if child.tag in {"attributes", "print"}:
            insert_at = index + 1
        else:
            break
    for index in range(insert_at, len(measure)):
        child = measure[index]
        if child.tag != "direction":
            break
        try:
            child_offset = int(child.findtext("offset") or 0)
        except ValueError:
            child_offset = 0
        if child_offset <= offset:
            insert_at = index + 1
        else:
            break
    measure.insert(insert_at, direction)
    return True


def _correct_mark(
    text: str, kind: str, score: float
) -> tuple[str, bool, float, float, str | None, bool, float]:
    if kind not in {"direction", "dynamic", "metronome", "text"}:
        return text, False, 0.0, 0.0, None, False, 1.0
    suggestion = _corrector().suggest(text)
    safe = _corrector().should_autocorrect(suggestion)
    # Be conservative: only automatic corrections supported strongly by both OCR and
    # the trained model's unattended-writeback gate. Other suggestions remain visible
    # in the review UI with their method and edit distance preserved in the report.
    if score >= 0.50 and safe:
        return (
            suggestion.text, True, suggestion.probability, suggestion.margin,
            suggestion.method, suggestion.autocorrect_safe, suggestion.edit_ratio,
        )
    return (
        text, False, suggestion.probability, suggestion.margin,
        suggestion.method, suggestion.autocorrect_safe, suggestion.edit_ratio,
    )


def _set_root_text(root: etree._Element, parent_tag: str, child_tag: str, text: str) -> None:
    parent = root.find(parent_tag)
    if parent is None:
        parent = etree.Element(parent_tag)
        document_order = {
            "work": 0,
            "movement-number": 1,
            "movement-title": 2,
            "identification": 3,
            "defaults": 4,
            "credit": 5,
            "part-list": 6,
            "part": 7,
        }
        target_order = document_order.get(parent_tag, 6)
        insertion_index = len(root)
        for index, child in enumerate(root):
            if document_order.get(child.tag, 7) > target_order:
                insertion_index = index
                break
        root.insert(insertion_index, parent)
    child = parent.find(child_tag)
    if child is None:
        child = etree.SubElement(parent, child_tag)
    child.text = text


def _write_source_metadata(
    root: etree._Element,
    marks: list[OcrMark],
    layout: PageLayout,
    part_staff_counts: list[int],
) -> None:
    """Write only independently supported page furniture into MusicXML metadata.

    OCR already sees titles and instrument labels reliably on many clean scans, but
    treating them as staff directions either drops them or places them inside a
    measure.  This mapper uses the first score system as a geometric reference and
    requires agreement between at least two OCR passes before changing public score
    metadata.
    """

    if not layout.systems:
        return
    first_staff = layout.systems[0]
    first_line_y = min(first_staff.line_y) if first_staff.line_y else first_staff.top
    spacing = max(float(first_staff.spacing), 1.0)
    page_width = max(int(layout.width), 1)

    try:
        musicxml_page_height = float(root.findtext("./defaults/page-layout/page-height") or 300.0)
        musicxml_page_width = float(root.findtext("./defaults/page-layout/page-width") or 110.0)
    except ValueError:
        musicxml_page_height = 300.0
        musicxml_page_width = 110.0

    def upsert_credit(
        credit_type: str,
        text: str,
        *,
        x_ratio: float,
        y_ratio: float,
        font_size: float,
        justify: str,
    ) -> None:
        credit = next(
            (
                item
                for item in root.findall("credit")
                if item.findtext("credit-type") == credit_type
            ),
            None,
        )
        if credit is None:
            credit = etree.Element("credit", page="1")
            etree.SubElement(credit, "credit-type").text = credit_type
            part_list = root.find("part-list")
            root.insert(root.index(part_list) if part_list is not None else len(root), credit)
        words = credit.find("credit-words")
        if words is None:
            words = etree.SubElement(credit, "credit-words")
        words.text = text
        words.set("default-x", f"{musicxml_page_width * x_ratio:.3f}")
        words.set("default-y", f"{musicxml_page_height * y_ratio:.3f}")
        words.set("font-size", f"{font_size:g}")
        words.set("justify", justify)
        words.set("valign", "top")

    def supported(mark: OcrMark) -> bool:
        return bool(
            mark.score >= 0.90
            and "+" in (mark.backend or "")
            and len(re.findall(r"[A-Za-z]", mark.text)) >= 2
        )

    header_candidates: list[tuple[OcrMark, float, float, float, float]] = []
    for mark in marks:
        if not supported(mark):
            continue
        # Tempo words and metronome marks often occupy the same generous band
        # as a short title on a tightly cropped first page. Promoting them to
        # work-title metadata duplicates the visible text and prevents their
        # ordinary staff anchoring.
        if mark.kind in {"dynamic", "direction", "metronome"}:
            continue
        x1, y1, x2, y2 = _box_bounds(mark.box)
        center_x = (x1 + x2) / 2.0
        if (
            y2 < first_line_y - 9.5 * spacing
            and page_width * 0.22 <= center_x <= page_width * 0.78
            and x2 - x1 >= 8.0 * spacing
        ):
            header_candidates.append((mark, x1, y1, x2, y2))

    title_mark: OcrMark | None = None
    subtitle_mark: OcrMark | None = None
    if header_candidates:
        header_candidates.sort(
            key=lambda item: (
                item[4] - item[2],
                item[3] - item[1],
                item[0].score,
            ),
            reverse=True,
        )
        title_mark = header_candidates[0][0]
        title_mark.kind = "metadata"
        title_mark.injected = True
        title_text = title_mark.text.strip()
        _set_root_text(root, "work", "work-title", title_text)
        movement_title = root.find("movement-title")
        if movement_title is None:
            movement_title = etree.Element("movement-title")
            work = root.find("work")
            root.insert(root.index(work) + 1 if work is not None else 0, movement_title)
        movement_title.text = title_text
        upsert_credit(
            "title",
            title_text,
            x_ratio=0.5,
            y_ratio=0.973,
            font_size=18.0,
            justify="center",
        )
        remaining = sorted(
            header_candidates[1:],
            key=lambda item: (item[2], -(item[4] - item[2])),
        )
        if remaining:
            subtitle_mark = remaining[0][0]
            subtitle_mark.kind = "metadata"
            subtitle_mark.injected = True
            upsert_credit(
                "subtitle",
                subtitle_mark.text.strip(),
                x_ratio=0.5,
                y_ratio=0.92,
                font_size=11.0,
                justify="center",
            )

    excluded_ids = {
        id(mark)
        for mark in (title_mark, subtitle_mark)
        if mark is not None
    }
    byline_candidates: list[tuple[float, OcrMark]] = []
    for mark in marks:
        if (
            id(mark) in excluded_ids
            or not supported(mark)
            or mark.kind in {"dynamic", "direction", "metronome"}
        ):
            continue
        x1, y1, x2, y2 = _box_bounds(mark.box)
        center_x = (x1 + x2) / 2.0
        distance = (first_line_y - y2) / spacing
        if (
            4.5 <= distance <= 10.5
            and page_width * 0.42 <= center_x <= page_width * 0.88
            and x2 - x1 >= 12.0 * spacing
        ):
            byline_candidates.append((x2 - x1, mark))
    if byline_candidates:
        byline = max(byline_candidates, key=lambda item: item[0])[1]
        byline.kind = "metadata"
        byline.injected = True
        identification = root.find("identification")
        if identification is None:
            identification = etree.Element("identification")
            insertion_index = len(root)
            for index, child in enumerate(root):
                if child.tag in {"defaults", "credit", "part-list", "part"}:
                    insertion_index = index
                    break
            root.insert(insertion_index, identification)
        creators = identification.findall("creator")
        creator = next((item for item in creators if item.get("type") == "composer"), None)
        if creator is None:
            creator = etree.SubElement(identification, "creator", type="composer")
        creator.text = byline.text.strip()
        upsert_credit(
            "composer",
            byline.text.strip(),
            x_ratio=0.955,
            y_ratio=0.867,
            font_size=9.0,
            justify="right",
        )

    parts = root.findall("part")
    part_list = root.find("part-list")
    if not parts or part_list is None:
        return
    first_score_system = layout.score_systems[0] if layout.score_systems else None
    system_top = first_score_system.top if first_score_system is not None else first_staff.top
    system_bottom = (
        first_score_system.bottom
        if first_score_system is not None
        else max(system.bottom for system in layout.systems[: max(1, sum(part_staff_counts))])
    )
    leading_barlines = [
        int(value)
        for system in layout.systems[: max(1, sum(part_staff_counts))]
        for value in system.barlines
        if value > page_width * 0.08
    ]
    music_left = min(leading_barlines) if leading_barlines else max(first_staff.left, int(page_width * 0.12))
    labels_by_part: dict[int, tuple[int, OcrMark]] = {}
    staff_centers = [
        (
            index,
            float(sum(system.line_y) / len(system.line_y))
            if system.line_y
            else float((system.top + system.bottom) / 2.0),
        )
        for index, system in enumerate(layout.systems[: max(1, sum(part_staff_counts))])
    ]
    for mark in marks:
        if (
            id(mark) in excluded_ids
            or not supported(mark)
            or mark.kind in {"dynamic", "direction", "metronome"}
        ):
            continue
        x1, y1, x2, y2 = _box_bounds(mark.box)
        center_y = (y1 + y2) / 2.0
        if (
            x2 > music_left - 0.8 * spacing
            or not system_top <= center_y <= system_bottom
            or y2 - y1 > 5.0 * spacing
        ):
            continue
        physical_index = min(staff_centers, key=lambda item: abs(item[1] - center_y))[0]
        target = _source_staff_target(layout, physical_index, part_staff_counts)
        if target is None:
            continue
        _, part_index, staff_number, _ = target
        if not 0 <= part_index < len(parts):
            continue
        # Prefer the fuller first-system label over later abbreviations.
        rank = len(mark.text.strip())
        if part_index not in labels_by_part or rank > labels_by_part[part_index][0]:
            labels_by_part[part_index] = (rank, mark)
            mark.target_part_id = parts[part_index].get("id")
            mark.target_staff_number = staff_number

    score_parts = {
        item.get("id"): item
        for item in part_list.findall("score-part")
        if item.get("id")
    }
    for part_index, (_, mark) in labels_by_part.items():
        part_id = parts[part_index].get("id")
        score_part = score_parts.get(part_id)
        if score_part is None:
            continue
        part_name = score_part.find("part-name")
        if part_name is None:
            part_name = etree.SubElement(score_part, "part-name")
        part_name.text = mark.text.strip()
        mark.kind = "metadata"
        mark.injected = True


def enrich_musicxml_with_ocr(
    image_path: Path,
    xml_path: Path,
    layout: PageLayout | None = None,
    semantic_text_regions: Iterable[dict[str, object]] | None = None,
) -> tuple[list[OcrMark], list[str]]:
    warnings: list[str] = []
    layout = layout or analyze_layout(image_path)
    rows, backend = (
        run_ocr(image_path, layout, semantic_text_regions)
        if semantic_text_regions
        else run_ocr(image_path, layout)
    )
    if backend == "unavailable":
        return [], ["没有可用的本地文字识别后端"]
    marks: list[OcrMark] = []
    anchor_classifier = _anchor_classifier()

    for raw_text, score, box, mark_backend in rows:
        raw_text = _normalize_text(raw_text)
        if not raw_text:
            continue
        center_x, center_y = _box_center(box)
        system_index, placement, distance = _nearest_system(center_x, center_y, layout)
        kind = classify_text(raw_text)
        text, corrected, probability, margin, correction_method, correction_safe, correction_edit_ratio = _correct_mark(
            raw_text, kind, score
        )
        if corrected:
            kind = classify_text(text)
        # A known tempo term may legitimately appear in the generous title band above
        # the first system. Recover that case, while large or lexically unsupported page
        # furniture remains metadata.
        if system_index is None and layout.systems and center_y < layout.systems[0].top:
            first_system = layout.systems[0]
            first_spacing = first_system.spacing
            first_distance = (first_system.line_y[0] - center_y) / max(first_spacing, 1.0)
            tempo_title_band = (
                kind in {"direction", "metronome"}
                and probability >= 0.88
                and correction_edit_ratio <= 0.35
                and first_distance <= 20.0
                and _box_height(box) <= first_spacing * 4.2
            )
            if tempo_title_band:
                system_index = 0
                placement = "above"
                distance = first_distance
            elif first_distance > 12.0 or _box_height(box) > first_spacing * 4:
                kind = "metadata"
        anchor_features = extract_direction_anchor_features(
            text=text,
            kind=kind,
            score=score,
            box=box,
            backend=mark_backend,
            correction_probability=probability,
            correction_margin=margin,
            correction_edit_ratio=correction_edit_ratio,
            system_index=system_index,
            placement=placement,
            distance_staff_spaces=float(distance if np.isfinite(distance) else 999.0),
            layout=layout,
        )
        direction_probability = anchor_classifier.predict(anchor_features)
        # A high-confidence first-system direction can safely resolve a small OCR typo
        # even when the text-only model's generic unattended-writeback gate is slightly
        # too conservative (for example a trailing period plus one substituted letter).
        if (
            not corrected
            and kind == "direction"
            and direction_probability >= 0.94
            and probability >= 0.96
            and margin >= 0.45
            and correction_edit_ratio <= 0.20
        ):
            anchored_suggestion = _corrector().suggest(raw_text)
            if anchored_suggestion.changed:
                text = anchored_suggestion.text
                corrected = True
                correction_method = "anchor-assisted-phrase"
                correction_safe = True
                kind = classify_text(text)
        marks.append(
            OcrMark(
                raw_text=raw_text,
                text=text,
                score=float(score),
                box=box,
                kind=kind,
                system_index=system_index,
                placement=placement,
                corrected=corrected,
                correction_probability=probability,
                correction_margin=margin,
                correction_method=correction_method,
                correction_autocorrect_safe=correction_safe,
                correction_edit_ratio=correction_edit_ratio,
                distance_staff_spaces=float(distance if np.isfinite(distance) else 999.0),
                backend=mark_backend,
                musical_direction_probability=float(direction_probability),
                direction_anchor_model_version=anchor_classifier.model_version,
                direction_anchor_model_status=anchor_classifier.status,
            )
        )

    if not xml_path.exists():
        return marks, warnings

    try:
        parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
        tree = etree.parse(str(xml_path), parser)
        root = tree.getroot()
        original_dynamic_directions = [
            direction
            for direction in root.findall("./part/measure/direction")
            if direction.find("./direction-type/dynamics") is not None
        ]
        parts = root.findall("part")
        if not parts:
            return marks, ["文字已识别，但 MusicXML 中没有可关联的 part"]
        original_relocatable_directions: dict[
            tuple[str, str],
            list[etree._Element],
        ] = {}
        for part in parts:
            part_id = str(part.get("id") or "")
            if not part_id:
                continue
            for direction in part.findall("./measure/direction"):
                key = _relocatable_direction_key(direction)
                if key is not None:
                    original_relocatable_directions.setdefault(
                        (part_id, key),
                        [],
                    ).append(direction)
        part_staff_counts = [_part_staff_count(part) for part in parts]
        _write_source_metadata(root, marks, layout, part_staff_counts)
        part_systems = [_xml_system_measure_groups(part) for part in parts]
        part_measures = [part.findall("measure") for part in parts]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        width = image.shape[1] if image is not None else max(layout.width, 1)
        binary = (
            cv2.threshold(
                image,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )[1]
            if image is not None
            else None
        )
        notehead_columns_by_system: dict[int, tuple[float, ...]] = {}

        for mark in marks:
            # Preserve unknown alphabetic direction text near a staff instead of silently dropping it.
            if mark.kind not in {"dynamic", "direction", "metronome", "text"}:
                continue
            # A semantic detector role is stronger evidence than OCR lexical
            # resemblance.  For example, an instrument label named "Piano"
            # or a rehearsal box containing "A" must never be rewritten as a
            # measure-owned generic direction merely because OCR is confident.
            if not _allows_generic_direction_writeback(mark.backend):
                continue
            if mark.system_index is None:
                continue
            target_mapping = _source_staff_target(
                layout,
                mark.system_index,
                part_staff_counts,
            )
            if target_mapping is None:
                continue
            score_system_index, part_index, target_staff_number, source_staff = target_mapping
            xml_systems = part_systems[part_index]
            if not 0 <= score_system_index < len(xml_systems):
                continue
            source_measure_system = layout.effective_score_systems[score_system_index]
            group = xml_systems[score_system_index]
            measures = part_measures[part_index]
            target_part = parts[part_index]
            # Nearest-staff ownership is vertical evidence only. Text centred
            # outside the shared system line cannot own a measure: common
            # examples are instrument labels on the left and page furniture on
            # the right. Never clamp those boxes into the first/last measure.
            horizontal_tolerance = max(
                2.0,
                float(source_staff.spacing if source_staff is not None else 8.0)
                * 0.5,
            )
            center_x, _ = _box_center(mark.box)
            if not (
                float(source_measure_system.left) - horizontal_tolerance
                <= center_x
                <= float(source_measure_system.right) + horizontal_tolerance
            ):
                continue
            mark.score_system_index = score_system_index
            mark.target_part_id = target_part.get("id")
            mark.target_staff_number = target_staff_number
            if source_staff is not None:
                anchor = anchor_classifier.refine_measure_anchor(
                    source_measure_system,
                    center_x,
                    len(group),
                    kind=mark.kind,
                    placement=mark.placement,
                )
                local_index = anchor.local_index
                offset_ratio = anchor.offset_ratio
                mark.measure_anchor_confidence = anchor.confidence
                mark.measure_anchor_method = anchor.method
                if (
                    mark.kind == "metronome"
                    and score_system_index == 0
                    and local_index == 0
                ):
                    offset_ratio = 0.0
                    mark.measure_anchor_confidence = max(
                        mark.measure_anchor_confidence,
                        0.99,
                    )
                    mark.measure_anchor_method = "initial_metronome_start"
                elif (
                    mark.kind == "direction"
                    and score_system_index == 0
                    and local_index == 0
                    and _term_key(mark.text).split(" ", 1)[0]
                    in _INITIAL_TEMPO_TERMS
                ):
                    offset_ratio = 0.0
                    mark.measure_anchor_confidence = max(
                        mark.measure_anchor_confidence,
                        0.97,
                    )
                    mark.measure_anchor_method = "initial_tempo_start"
                elif mark.kind == "dynamic" and binary is not None:
                    columns = notehead_columns_by_system.get(mark.system_index)
                    if columns is None:
                        columns = _visual_notehead_columns(
                            binary,
                            source_staff,
                            music_left=source_measure_system.left,
                        )
                        notehead_columns_by_system[mark.system_index] = columns
                    snapped = _snap_direction_to_notehead(
                        x=center_x,
                        system=source_measure_system,
                        local_index=local_index,
                        group=group,
                        notehead_columns=columns,
                        staff_number=target_staff_number,
                    )
                    if snapped is not None:
                        offset_ratio = snapped
                        mark.measure_anchor_confidence = max(
                            mark.measure_anchor_confidence,
                            0.96,
                        )
                        mark.measure_anchor_method = "barline_notehead_snap"
            else:
                relative = max(0.0, min(0.999, center_x / max(width, 1)))
                measure_float = relative * len(group)
                local_index = min(len(group) - 1, int(measure_float))
                offset_ratio = measure_float - local_index
                mark.measure_anchor_confidence = 0.30
                mark.measure_anchor_method = "page_uniform_fallback"
            target = group[local_index]
            mark.measure_index = measures.index(target)
            mark.offset_ratio = float(offset_ratio)
            if mark.kind == "dynamic":
                threshold = 0.50
                role_floor = DEFAULT_POLICY.direction_anchor_dynamic_floor
                backend_consensus_required = False
            elif mark.kind == "metronome":
                threshold = 0.60
                role_floor = DEFAULT_POLICY.direction_anchor_metronome_floor
                backend_consensus_required = False
            elif mark.kind == "direction":
                threshold = 0.58
                role_floor = DEFAULT_POLICY.direction_anchor_direction_floor
                backend_consensus_required = False
            else:
                # Unknown alphabetic text is preserved for review, but only inserted
                # automatically when independent OCR passes agree and the conservative
                # staff-direction role model finds strong geometric support.
                threshold = 0.72
                role_floor = DEFAULT_POLICY.direction_anchor_text_floor
                backend_consensus_required = True
            source_dynamic_geometry = (
                mark.kind == "dynamic"
                and "source-dynamic-geometry" in (mark.backend or "")
                and mark.score >= 0.96
                and (
                    "+" in (mark.backend or "")
                    or mark.backend == "source-dynamic-geometry-v2"
                )
            )
            source_metronome_consensus = (
                mark.kind == "metronome"
                and parse_metronome_mark(mark.text) is not None
                and mark.score >= 0.90
                and "+" in (mark.backend or "")
                and mark.distance_staff_spaces <= 6.0
            )
            role_gate = source_dynamic_geometry or source_metronome_consensus or (
                mark.musical_direction_probability >= role_floor
                if anchor_classifier.enabled
                else mark.distance_staff_spaces <= 10.0
            )
            placement_gate = mark.placement in {"above", "below"}
            lexical_gate = True
            if mark.kind == "direction":
                lexical_gate = bool(
                    mark.corrected
                    or (
                        mark.correction_probability >= 0.90
                        and mark.correction_edit_ratio <= 0.20
                    )
                )
            elif mark.kind == "text":
                lexical_gate = bool(
                    mark.correction_probability >= 0.84
                    and mark.correction_edit_ratio <= 0.30
                )
            elif mark.kind == "dynamic":
                lexical_gate = bool(
                    _DYNAMIC_RE.fullmatch(mark.text.strip().casefold())
                )
                if mark.text.strip().casefold() in {"p", "f"}:
                    # Single music-font letters are readily confused with rests,
                    # flags and ornament fragments. Require independent OCR and
                    # source-glyph geometry to agree on the same bounding box.
                    lexical_gate = lexical_gate and source_dynamic_geometry
            if (
                mark.score < threshold
                or not role_gate
                or not placement_gate
                or not lexical_gate
                or (backend_consensus_required and "+" not in (mark.backend or ""))
            ):
                continue
            mark.injected = _insert_direction(
                target,
                mark.text,
                "direction" if mark.kind == "text" else mark.kind,
                offset_ratio,
                mark.placement or "above",
                staff_number=(
                    target_staff_number
                    if part_staff_counts[part_index] > 1
                    else None
                ),
            )

        # Correct existing text/tempo placement only when the source detector
        # localized the same exact content, every original occurrence has one
        # source occurrence, and every source anchor is strong. This fixes the
        # former duplicate/wrong-measure failure without allowing OCR to rewrite
        # content or guess among incompletely observed repeated phrases.
        source_reanchors: dict[tuple[str, str], list[OcrMark]] = {}
        for mark in marks:
            if (
                mark.kind not in {"direction", "metronome", "text"}
                or not _allows_generic_direction_writeback(mark.backend)
                or mark.target_part_id is None
                or mark.measure_index is None
                or mark.measure_anchor_confidence < 0.96
                or mark.musical_direction_probability < 0.94
                or mark.score < 0.90
                or "rapid-semantic-region:" not in (mark.backend or "")
                or "+" not in (mark.backend or "")
                or mark.placement not in {"above", "below"}
            ):
                continue
            write_kind = (
                "direction" if mark.kind == "text" else mark.kind
            )
            if (
                write_kind == "metronome"
                and parse_metronome_mark(mark.text) is None
            ):
                continue
            key = _direction_dedup_key(mark.text, write_kind)
            source_reanchors.setdefault(
                (mark.target_part_id, key),
                [],
            ).append(mark)

        part_by_id = {
            str(part.get("id") or ""): part
            for part in parts
            if part.get("id")
        }
        for identity, source_marks in source_reanchors.items():
            originals = original_relocatable_directions.get(identity, [])
            if not originals or len(originals) != len(source_marks):
                continue
            source_positions = {
                (
                    int(mark.measure_index or 0),
                    int(mark.target_staff_number or 1),
                    round(float(mark.offset_ratio), 6),
                )
                for mark in source_marks
            }
            if len(source_positions) != len(source_marks):
                continue
            for direction in originals:
                parent = direction.getparent()
                if parent is not None:
                    parent.remove(direction)
            target_part = part_by_id.get(identity[0])
            if target_part is None:
                continue
            target_measures = target_part.findall("measure")
            for mark in sorted(
                source_marks,
                key=lambda item: (
                    int(item.measure_index or 0),
                    float(item.offset_ratio),
                    int(item.target_staff_number or 1),
                ),
            ):
                measure_index = int(mark.measure_index or 0)
                if not 0 <= measure_index < len(target_measures):
                    continue
                write_kind = (
                    "direction" if mark.kind == "text" else mark.kind
                )
                mark.injected = _insert_direction(
                    target_measures[measure_index],
                    mark.text,
                    write_kind,
                    mark.offset_ratio,
                    mark.placement or "above",
                    staff_number=(
                        mark.target_staff_number
                        if _part_staff_count(target_part) > 1
                        else None
                    ),
                ) or mark.injected
                mark.reanchored_existing = True

        reliable_source_dynamics = [
            mark
            for mark in marks
            if (
                mark.kind == "dynamic"
                and mark.target_part_id is not None
                and mark.measure_index is not None
                and _DYNAMIC_RE.fullmatch(mark.text.strip().casefold())
                and (
                    (
                        mark.text.strip().casefold() in {"p", "f"}
                        and "source-dynamic-geometry-v2" in (mark.backend or "")
                    )
                    or (
                        mark.text.strip().casefold() not in {"p", "f"}
                        and mark.score >= 0.72
                        and "+" in (mark.backend or "")
                        and mark.musical_direction_probability
                        >= DEFAULT_POLICY.direction_anchor_dynamic_floor
                    )
                )
            )
        ]
        if (
            original_dynamic_directions
            and len(reliable_source_dynamics) >= 2
            and len(reliable_source_dynamics) >= len(original_dynamic_directions)
        ):
            # When independent source evidence covers at least the entire recognizer
            # dynamic inventory, treat it as an authoritative replacement set. This
            # removes internally plausible but visibly unsupported p/mf hallucinations.
            for direction in original_dynamic_directions:
                parent = direction.getparent()
                if parent is not None:
                    parent.remove(direction)
            part_by_id = {part.get("id"): part for part in parts}
            for mark in reliable_source_dynamics:
                target_part = part_by_id.get(mark.target_part_id)
                if target_part is None or mark.measure_index is None:
                    continue
                target_measures = target_part.findall("measure")
                if not 0 <= mark.measure_index < len(target_measures):
                    continue
                mark.injected = _insert_direction(
                    target_measures[mark.measure_index],
                    mark.text,
                    "dynamic",
                    mark.offset_ratio,
                    mark.placement or "above",
                    staff_number=(
                        mark.target_staff_number
                        if _part_staff_count(target_part) > 1
                        else None
                    ),
                ) or mark.injected

        atomic_write_bytes(
            xml_path,
            etree.tostring(
                tree,
                encoding="UTF-8",
                xml_declaration=True,
                pretty_print=True,
                doctype='<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
            ),
        )
    except Exception as exc:
        warnings.append(f"文字标记无法写入 MusicXML：{exc}")

    corrected_count = sum(mark.corrected for mark in marks)
    if corrected_count:
        warnings.append(f"音乐术语模型自动校正了 {corrected_count} 处高置信度 OCR 结果")
    if not anchor_classifier.enabled:
        warnings.append("谱面文字角色模型不可用，已使用保守几何回退")
    return marks, warnings


def marks_to_dicts(marks: list[OcrMark]) -> list[dict[str, object]]:
    return [asdict(mark) for mark in marks]


def _direction_value(direction: etree._Element) -> str | None:
    words = direction.find("./direction-type/words")
    if words is not None and words.text:
        return words.text
    dynamics = direction.find("./direction-type/dynamics")
    if dynamics is not None and len(dynamics):
        child = dynamics[0]
        return child.text or child.tag
    metronome = direction.find("./direction-type/metronome")
    if metronome is not None:
        beat = metronome.findtext("beat-unit") or "quarter"
        dotted = metronome.find("beat-unit-dot") is not None
        per_minute = metronome.findtext("per-minute") or ""
        return f"{'dotted ' if dotted else ''}{beat} = {per_minute}".strip()
    return None


def update_direction_in_musicxml(
    musicxml_path: Path,
    measure_number: int,
    value: str | None,
    kind: str,
    offset_ratio: float,
    placement: str,
    previous_values: Iterable[str] = (),
) -> bool:
    """Apply a user-reviewed direction to the merged MusicXML in place.

    Matching OCR-injected directions are removed first. Passing ``value=None`` only
    removes a matching direction, which supports an explicit "ignore" decision.
    """
    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False, no_network=True)
    tree = etree.parse(str(musicxml_path), parser)
    part = tree.getroot().find("part")
    if part is None:
        return False
    measure = next((item for item in part.findall("measure") if item.get("number") == str(measure_number)), None)
    if measure is None:
        return False
    keys = {_term_key(item) for item in previous_values if item}
    removed = False
    for direction in list(measure.findall("direction")):
        existing = _direction_value(direction)
        if existing and _term_key(existing) in keys:
            measure.remove(direction)
            removed = True
    inserted = False
    if value and value.strip():
        inserted = _insert_direction(
            measure,
            value.strip(),
            "direction" if kind == "text" else kind,
            offset_ratio,
            placement,
        )
    if removed or inserted:
        atomic_write_bytes(
            musicxml_path,
            etree.tostring(
                tree,
                encoding="UTF-8",
                xml_declaration=True,
                pretty_print=True,
                doctype='<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
            ),
        )
        return True
    return False
