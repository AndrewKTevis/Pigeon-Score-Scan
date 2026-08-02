from __future__ import annotations

"""Fail-closed ONNX verifier for positioned notation symbols.

The detector is deliberately not a second OMR engine.  It may corroborate source
geometry that was independently found by :mod:`notation_coverage`, but it cannot
create MusicXML objects by itself.  A release asset is enabled only when its
self-contained manifest proves independent-scan and ONNX-parity gates and every
artifact matches the committed hashes.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .layout import PageLayout
from .notation_coverage import VisualNotationCandidate
from .semantic_detector_contract import (
    GEOMETRY_CORROBORATION_CLASSES,
    MAX_CATEGORIES_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_ONNX_BYTES,
    HIGH_RECALL_MARK_CLASSES,
    MINIMUM_INDEPENDENT_WORKS,
    MINIMUM_HIGH_RECALL_MARK_RECALL,
    MINIMUM_OPERATING_POINT_PRECISION,
    MINIMUM_OPERATING_POINT_RECALL,
    MINIMUM_OPERATING_POINT_TRUE_POSITIVES,
    POSITIONAL_INVENTORY_CLASSES,
    SEMANTIC_DETECTOR_INPUT_SIZE,
    SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO,
    SEMANTIC_DETECTOR_MAXIMUM_SCALE,
    SEMANTIC_DETECTOR_MAXIMUM_TILES,
    SEMANTIC_DETECTOR_MINIMUM_SCALE,
    SEMANTIC_DETECTOR_FORMAT,
    SEMANTIC_DETECTOR_MANIFEST_NAME,
    SEMANTIC_DETECTOR_TARGET_STAFF_SPACING,
    SEMANTIC_DETECTOR_TILE_OVERLAP,
    page_shape_is_supported,
    SEMANTIC_PAGE_NMS_IOU,
    SUPPORTED_RUNTIME_CLASSES,
    SYMBOL_AUDIT_CLASSES,
    TEXT_REGION_CLASSES,
    TILE_FRAGMENT_FUSION_VERSION,
)
from .semantic_tile_fusion import (
    TileFragmentDetection,
    fuse_tile_fragments,
    scaled_page_dimension,
    semantic_page_scale,
    semantic_tile_origins,
    source_tile_bbox,
)
from .util import sha256_file

@dataclass(frozen=True)
class SemanticDetection:
    class_name: str
    label: int
    bbox: tuple[int, int, int, int]
    confidence: float
    staff_index: int
    placement: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        payload["confidence"] = round(self.confidence, 6)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticDetection":
        bbox = payload.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("semantic detection bbox is invalid")
        return cls(
            class_name=str(payload["class_name"]),
            label=int(payload["label"]),
            bbox=tuple(int(value) for value in bbox),
            confidence=float(payload["confidence"]),
            staff_index=int(payload["staff_index"]),
            placement=str(payload["placement"]),
        )


@dataclass(frozen=True)
class TiledSemanticDetection:
    detection: SemanticDetection
    tile_id: int
    tile_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class SemanticDetectorAssets:
    model_version: str
    model_path: Path
    categories_path: Path
    class_name_by_label: dict[int, str]
    thresholds: dict[str, float]
    input_size: int
    target_staff_spacing: float
    overlap: int
    page_nms_iou: float
    maximum_tiles: int
    minimum_scale: float
    maximum_scale: float
    input_name: str
    output_names: tuple[str, str, str]
    manifest_sha256: str


@dataclass(frozen=True)
class SemanticDetectorStatus:
    enabled: bool
    status: str
    model_version: str | None = None
    requested_accelerator: str | None = None
    selected_accelerator: str | None = None
    providers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["providers"] = list(self.providers)
        return payload


@dataclass(frozen=True)
class SemanticDetectorResult:
    detections: tuple[SemanticDetection, ...]
    status: SemanticDetectorStatus
    elapsed_seconds: float
    scale: float = 1.0
    tile_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": 1,
            "detections": [item.to_dict() for item in self.detections],
            "status": self.status.to_dict(),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "scale": round(self.scale, 6),
            "tile_count": self.tile_count,
        }


def _read_bounded_json(path: Path, maximum_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path.name} exceeds its bounded JSON size")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def _sha256_value(value: object, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field} is not a SHA-256 digest")
    return normalized


def _artifact_path(
    resources_dir: Path,
    payload: object,
    *,
    field: str,
    maximum_bytes: int,
    suffix: str,
) -> Path:
    if not isinstance(payload, dict):
        raise ValueError(f"{field} artifact is missing")
    filename = str(payload.get("file") or "").strip()
    if not filename or Path(filename).name != filename or not filename.casefold().endswith(suffix):
        raise ValueError(f"{field} artifact filename is invalid")
    path = resources_dir / filename
    expected_size = int(payload.get("bytes", -1))
    actual_size = path.stat().st_size
    if actual_size <= 0 or actual_size > maximum_bytes or actual_size != expected_size:
        raise ValueError(f"{field} artifact size is invalid")
    expected_hash = _sha256_value(payload.get("sha256"), f"{field}.sha256")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{field} artifact hash mismatch")
    return path


def load_semantic_detector_assets(resources_dir: Path) -> SemanticDetectorAssets:
    resources_dir = resources_dir.resolve()
    manifest_path = resources_dir / SEMANTIC_DETECTOR_MANIFEST_NAME
    manifest = _read_bounded_json(manifest_path, MAX_MANIFEST_BYTES)
    if int(manifest.get("format", 0)) != SEMANTIC_DETECTOR_FORMAT:
        raise ValueError("semantic detector manifest format is unsupported")
    if manifest.get("integration_authorized") is not True:
        raise ValueError("semantic detector integration is not authorized")
    model_version = str(manifest.get("model_version") or "").strip()
    if not model_version:
        raise ValueError("semantic detector model version is missing")

    release_gate = manifest.get("release_gate")
    if not isinstance(release_gate, dict):
        raise ValueError("semantic detector release gate is missing")
    independent = release_gate.get("independent_holdout")
    parity = release_gate.get("onnx_parity")
    gpu_parity = release_gate.get("onnx_gpu_parity")
    if (
        not isinstance(independent, dict)
        or independent.get("passed") is not True
        or int(independent.get("independent_works", 0)) < MINIMUM_INDEPENDENT_WORKS
        or not isinstance(parity, dict)
        or parity.get("passed") is not True
        or not isinstance(gpu_parity, dict)
        or gpu_parity.get("passed") is not True
        or gpu_parity.get("runtime") != "onnxruntime-gpu==1.26.0"
    ):
        raise ValueError("semantic detector release gates did not pass")
    independent_report_path = _artifact_path(
        resources_dir,
        independent,
        field="independent_holdout",
        maximum_bytes=MAX_MANIFEST_BYTES,
        suffix=".json",
    )
    parity_report_path = _artifact_path(
        resources_dir,
        parity,
        field="onnx_parity",
        maximum_bytes=MAX_MANIFEST_BYTES,
        suffix=".json",
    )
    gpu_parity_report_path = _artifact_path(
        resources_dir,
        gpu_parity,
        field="onnx_gpu_parity",
        maximum_bytes=MAX_MANIFEST_BYTES,
        suffix=".json",
    )
    independent_report = _read_bounded_json(
        independent_report_path,
        MAX_MANIFEST_BYTES,
    )
    parity_report = _read_bounded_json(parity_report_path, MAX_MANIFEST_BYTES)
    gpu_parity_report = _read_bounded_json(
        gpu_parity_report_path,
        MAX_MANIFEST_BYTES,
    )
    gpu_runtime = gpu_parity_report.get("runtime")
    gpu_metrics = gpu_parity_report.get("parity")
    if (
        independent_report.get("acceptance", {}).get("passed") is not True
        or int(independent_report.get("independent_works", 0))
        != int(independent.get("independent_works", 0))
        or independent_report.get("integration_authorized") is not False
        or parity_report.get("passed") is not True
        or gpu_parity_report.get("passed") is not True
        or not isinstance(gpu_runtime, dict)
        or gpu_runtime.get("onnxruntime") != "1.26.0"
        or not isinstance(gpu_runtime.get("cuda_session_providers"), list)
        or not gpu_runtime["cuda_session_providers"]
        or gpu_runtime["cuda_session_providers"][0] != "CUDAExecutionProvider"
        or not isinstance(gpu_metrics, dict)
        or gpu_metrics.get("passed") is not True
        or int(gpu_metrics.get("cpu_detections", 0)) <= 0
        or int(gpu_metrics.get("cuda_detections", 0)) <= 0
    ):
        raise ValueError("semantic detector bundled release reports are invalid")

    model_path = _artifact_path(
        resources_dir,
        manifest.get("model"),
        field="model",
        maximum_bytes=MAX_ONNX_BYTES,
        suffix=".onnx",
    )
    categories_path = _artifact_path(
        resources_dir,
        manifest.get("categories"),
        field="categories",
        maximum_bytes=MAX_CATEGORIES_BYTES,
        suffix=".json",
    )
    categories = _read_bounded_json(categories_path, MAX_CATEGORIES_BYTES)
    rows = categories.get("classes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("semantic detector categories are empty")
    class_name_by_label: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("semantic detector category row is invalid")
        label = int(row.get("label", 0))
        name = str(row.get("name") or "").strip()
        if label <= 0 or not name or label in class_name_by_label:
            raise ValueError("semantic detector categories are invalid or duplicated")
        class_name_by_label[label] = name
    if not SUPPORTED_RUNTIME_CLASSES <= set(class_name_by_label.values()):
        raise ValueError("semantic detector lacks required runtime classes")

    operating_points = manifest.get("operating_points")
    if not isinstance(operating_points, dict):
        raise ValueError("semantic detector operating points are missing")
    thresholds: dict[str, float] = {}
    for class_name in sorted(SUPPORTED_RUNTIME_CLASSES):
        point = operating_points.get(class_name)
        if not isinstance(point, dict):
            raise ValueError(f"semantic detector operating point is missing: {class_name}")
        threshold = float(point.get("threshold", -1.0))
        precision = float(point.get("precision", -1.0))
        recall = float(point.get("recall", -1.0))
        true_positives = int(point.get("true_positives", 0))
        if (
            not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
            or not math.isfinite(precision)
            or not MINIMUM_OPERATING_POINT_PRECISION <= precision <= 1.0
            or not math.isfinite(recall)
            or not MINIMUM_OPERATING_POINT_RECALL <= recall <= 1.0
            or true_positives < MINIMUM_OPERATING_POINT_TRUE_POSITIVES
        ):
            raise ValueError(f"semantic detector operating point is unsafe: {class_name}")
        if (
            class_name in HIGH_RECALL_MARK_CLASSES
            and recall < MINIMUM_HIGH_RECALL_MARK_RECALL
        ):
            raise ValueError(
                "semantic detector high-recall mark gate is unsafe: "
                f"{class_name}"
            )
        thresholds[class_name] = threshold
    try:
        gpu_comparison_floor = float(
            gpu_metrics.get("comparison_score_floor", -1.0)
        )
    except (TypeError, ValueError, OverflowError):
        gpu_comparison_floor = -1.0
    if (
        not 0.0 <= gpu_comparison_floor <= 1.0
        or gpu_comparison_floor > min(thresholds.values())
    ):
        raise ValueError(
            "semantic detector GPU parity did not cover deployment thresholds"
        )

    input_payload = manifest.get("input")
    output_payload = manifest.get("outputs")
    if not isinstance(input_payload, dict) or not isinstance(output_payload, dict):
        raise ValueError("semantic detector tensor contract is missing")
    input_size = int(input_payload.get("size", 0))
    target_staff_spacing = float(input_payload.get("target_staff_spacing", 0.0))
    overlap = int(input_payload.get("overlap", -1))
    page_nms_iou = float(input_payload.get("page_nms_iou", -1.0))
    tile_fragment_fusion_version = str(
        input_payload.get("tile_fragment_fusion_version") or ""
    )
    maximum_tiles = int(input_payload.get("maximum_tiles", 0))
    minimum_scale = float(input_payload.get("minimum_scale", 0.0))
    maximum_scale = float(input_payload.get("maximum_scale", 0.0))
    input_name = str(input_payload.get("name") or "").strip()
    output_names_raw = output_payload.get("names")
    if (
        input_size != SEMANTIC_DETECTOR_INPUT_SIZE
        or target_staff_spacing
        != SEMANTIC_DETECTOR_TARGET_STAFF_SPACING
        or overlap != SEMANTIC_DETECTOR_TILE_OVERLAP
        or not math.isclose(
            page_nms_iou,
            SEMANTIC_PAGE_NMS_IOU,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or tile_fragment_fusion_version != TILE_FRAGMENT_FUSION_VERSION
        or maximum_tiles != SEMANTIC_DETECTOR_MAXIMUM_TILES
        or minimum_scale != SEMANTIC_DETECTOR_MINIMUM_SCALE
        or maximum_scale != SEMANTIC_DETECTOR_MAXIMUM_SCALE
        or not input_name
        or not isinstance(output_names_raw, list)
        or len(output_names_raw) != 3
    ):
        raise ValueError("semantic detector tensor contract is invalid")
    output_names = tuple(str(value).strip() for value in output_names_raw)
    if any(not value for value in output_names) or len(set(output_names)) != 3:
        raise ValueError("semantic detector output names are invalid")

    return SemanticDetectorAssets(
        model_version=model_version,
        model_path=model_path,
        categories_path=categories_path,
        class_name_by_label=class_name_by_label,
        thresholds=thresholds,
        input_size=input_size,
        target_staff_spacing=target_staff_spacing,
        overlap=overlap,
        page_nms_iou=page_nms_iou,
        maximum_tiles=maximum_tiles,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
        input_name=input_name,
        output_names=(output_names[0], output_names[1], output_names[2]),
        manifest_sha256=sha256_file(manifest_path),
    )


def semantic_detector_status(resources_dir: Path) -> SemanticDetectorStatus:
    manifest_path = (
        resources_dir.resolve() / SEMANTIC_DETECTOR_MANIFEST_NAME
    )
    if not manifest_path.is_file():
        return SemanticDetectorStatus(False, "asset_absent")
    try:
        assets = load_semantic_detector_assets(resources_dir)
    except Exception as exc:
        return SemanticDetectorStatus(False, f"asset_rejected:{type(exc).__name__}:{exc}")
    return SemanticDetectorStatus(True, "authorized", assets.model_version)


def _origins(length: int, size: int, overlap: int) -> list[int]:
    return semantic_tile_origins(length, size, overlap)


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / max(1, left_area + right_area - intersection)


def fuse_tile_fragment_detections(
    detections: list[TiledSemanticDetection],
    layout: PageLayout,
) -> list[SemanticDetection]:
    """Fuse one-to-one boundary fragments without merging nested notation.

    Complete-page supervision intentionally labels the visible intersection in
    every retained overlapping tile. At inference this can produce two partial
    boxes for one long object. Candidate edges are therefore restricted to
    opposing tile boundaries, matched one-to-one for each tile pair/class/staff,
    and joined only when a component has at most one detection from every tile.
    """

    return [
        SemanticDetection(
            item.class_name,
            item.label,
            item.bbox,
            item.confidence,
            item.staff_index,
            item.placement,
        )
        for item in fuse_tile_fragments(
            [
                TileFragmentDetection(
                    item.detection.class_name,
                    item.detection.label,
                    item.detection.bbox,
                    item.detection.confidence,
                    item.detection.staff_index,
                    item.detection.placement,
                    item.tile_id,
                    item.tile_bbox,
                )
                for item in detections
            ],
            owner_resolver=lambda bbox: assign_bbox_to_staff(layout, bbox),
        )
    ]


def assign_bbox_to_staff(
    layout: PageLayout,
    bbox: tuple[int, int, int, int],
) -> tuple[int, str] | None:
    """Return the nearest bounded physical staff and relative placement."""

    if not layout.systems:
        return None
    center_y = 0.5 * (bbox[1] + bbox[3])
    scored: list[tuple[float, Any]] = []
    for staff in layout.systems:
        top_line = float(staff.line_y[0])
        bottom_line = float(staff.line_y[-1])
        if center_y < top_line:
            distance = top_line - center_y
            placement = "above"
        elif center_y > bottom_line:
            distance = center_y - bottom_line
            placement = "below"
        else:
            midpoint = 0.5 * (top_line + bottom_line)
            distance = 0.0
            placement = "above" if center_y <= midpoint else "below"
        scored.append((distance / max(float(staff.spacing), 1.0), (staff, placement)))
    distance_spaces, (staff, placement) = min(scored, key=lambda item: item[0])
    if distance_spaces > 8.0:
        return None
    return int(staff.index), str(placement)


def _squeezed(array: Any) -> np.ndarray:
    result = np.asarray(array)
    while result.ndim > 1 and result.shape[0] == 1:
        result = result[0]
    return result


def run_semantic_detector(
    image_path: Path,
    layout: PageLayout,
    resources_dir: Path,
    requested_accelerator: str,
    *,
    session_factory: Callable[[Path, list[str]], Any] | None = None,
) -> SemanticDetectorResult:
    started = time.monotonic()
    requested = str(requested_accelerator).strip().casefold()
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported semantic detector accelerator: {requested}")
    manifest_path = (
        resources_dir.resolve() / SEMANTIC_DETECTOR_MANIFEST_NAME
    )
    if not manifest_path.is_file():
        return SemanticDetectorResult(
            (),
            SemanticDetectorStatus(
                False,
                "asset_absent",
                requested_accelerator=requested,
            ),
            time.monotonic() - started,
        )
    try:
        assets = load_semantic_detector_assets(resources_dir)
    except Exception as exc:
        return SemanticDetectorResult(
            (),
            SemanticDetectorStatus(
                False,
                f"asset_rejected:{type(exc).__name__}:{exc}",
                requested_accelerator=requested,
            ),
            time.monotonic() - started,
        )

    if not layout.systems:
        return SemanticDetectorResult(
            (),
            SemanticDetectorStatus(
                False,
                "layout_has_no_staff",
                assets.model_version,
                requested,
            ),
            time.monotonic() - started,
        )
    from PIL import Image, ImageOps

    try:
        with Image.open(image_path) as source:
            source_image = ImageOps.grayscale(source)
            image = np.asarray(source_image, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(
            f"semantic detector could not read image: {image_path}"
        ) from exc
    if not page_shape_is_supported(image.shape[1], image.shape[0]):
        return SemanticDetectorResult(
            (),
            SemanticDetectorStatus(
                False,
                "page_exceeds_maximum_aspect_ratio:"
                f"{SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO:g}",
                assets.model_version,
                requested,
            ),
            time.monotonic() - started,
        )
    spacings = [float(staff.spacing) for staff in layout.systems if staff.spacing > 0]
    if not spacings:
        raise ValueError("semantic detector layout has no valid staff spacing")
    scale = semantic_page_scale(spacings)
    if abs(scale - 1.0) >= 0.01:
        resized = source_image.resize(
            (
                scaled_page_dimension(image.shape[1], scale),
                scaled_page_dimension(image.shape[0], scale),
            ),
            (
                Image.Resampling.BICUBIC
                if scale > 1.0
                else Image.Resampling.LANCZOS
            ),
        )
        scaled = np.asarray(resized, dtype=np.uint8)
    else:
        scale = 1.0
        scaled = image

    x_origins = _origins(scaled.shape[1], assets.input_size, assets.overlap)
    y_origins = _origins(scaled.shape[0], assets.input_size, assets.overlap)
    if len(x_origins) * len(y_origins) > assets.maximum_tiles:
        return SemanticDetectorResult(
            (),
            SemanticDetectorStatus(
                False,
                "page_exceeds_maximum_tiles",
                assets.model_version,
                requested,
            ),
            time.monotonic() - started,
            scale,
            len(x_origins) * len(y_origins),
        )

    if session_factory is None:
        import onnxruntime as ort  # type: ignore

        def session_factory(path: Path, providers: list[str]) -> Any:
            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = max(1, min(8, __import__("os").cpu_count() or 1))
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(str(path), sess_options=options, providers=providers)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if requested == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = session_factory(assets.model_path, providers)
    bound_providers = tuple(str(value) for value in session.get_providers())
    selected = "cuda" if bound_providers and bound_providers[0] == "CUDAExecutionProvider" else "cpu"
    if selected != requested:
        raise RuntimeError(
            "semantic detector provider verification mismatch: "
            f"requested={requested}, providers={list(bound_providers)}"
        )
    input_names = {str(item.name) for item in session.get_inputs()}
    output_names = {str(item.name) for item in session.get_outputs()}
    if assets.input_name not in input_names or not set(assets.output_names) <= output_names:
        raise RuntimeError("semantic detector ONNX tensor names do not match the manifest")

    detections: list[TiledSemanticDetection] = []
    tile_id = 0
    for y in y_origins:
        for x in x_origins:
            crop = scaled[
                y:min(y + assets.input_size, scaled.shape[0]),
                x:min(x + assets.input_size, scaled.shape[1]),
            ]
            valid_height, valid_width = crop.shape
            tile_bounds = source_tile_bbox(
                x=x,
                y=y,
                valid_width=valid_width,
                valid_height=valid_height,
                scale=scale,
                source_width=image.shape[1],
                source_height=image.shape[0],
            )
            tile = np.full((assets.input_size, assets.input_size), 255, dtype=np.uint8)
            tile[:valid_height, :valid_width] = crop
            rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
            tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
            raw_boxes, raw_scores, raw_labels = session.run(
                list(assets.output_names),
                {assets.input_name: tensor},
            )
            boxes = _squeezed(raw_boxes).reshape(-1, 4)
            scores = _squeezed(raw_scores).reshape(-1)
            labels = _squeezed(raw_labels).reshape(-1)
            if not (len(boxes) == len(scores) == len(labels)):
                raise RuntimeError("semantic detector ONNX outputs have inconsistent lengths")
            for box, raw_score, raw_label in zip(boxes, scores, labels, strict=True):
                score = float(raw_score)
                label = int(raw_label)
                class_name = assets.class_name_by_label.get(label)
                if class_name not in SUPPORTED_RUNTIME_CLASSES:
                    continue
                if not math.isfinite(score):
                    continue
                x1, y1, x2, y2 = (float(value) for value in box)
                center_x = 0.5 * (x1 + x2)
                center_y = 0.5 * (y1 + y2)
                if not (0 <= center_x < valid_width and 0 <= center_y < valid_height):
                    continue
                original = (
                    max(0, int(math.floor((x + max(0.0, x1)) / scale))),
                    max(0, int(math.floor((y + max(0.0, y1)) / scale))),
                    min(image.shape[1], int(math.ceil((x + min(float(valid_width), x2)) / scale))),
                    min(image.shape[0], int(math.ceil((y + min(float(valid_height), y2)) / scale))),
                )
                if original[2] <= original[0] or original[3] <= original[1]:
                    continue
                owner = assign_bbox_to_staff(layout, original)
                if owner is None:
                    continue
                detections.append(
                    TiledSemanticDetection(
                        SemanticDetection(
                            class_name,
                            label,
                            original,
                            min(1.0, max(0.0, score)),
                            owner[0],
                            owner[1],
                        ),
                        tile_id,
                        tile_bounds,
                    )
                )
            tile_id += 1

    retained: list[TiledSemanticDetection] = []
    for candidate in sorted(
        detections,
        key=lambda item: (
            -item.detection.confidence,
            item.detection.bbox,
            item.tile_id,
        ),
    ):
        if any(
            candidate.detection.class_name == other.detection.class_name
            and candidate.detection.staff_index == other.detection.staff_index
            and _bbox_iou(
                candidate.detection.bbox,
                other.detection.bbox,
            )
            >= SEMANTIC_PAGE_NMS_IOU
            for other in retained
        ):
            continue
        retained.append(candidate)
    fused = [
        item
        for item in fuse_tile_fragment_detections(retained, layout)
        if item.confidence >= assets.thresholds[item.class_name]
    ]
    fused.sort(key=lambda item: (item.staff_index, item.bbox[0], item.class_name))
    return SemanticDetectorResult(
        tuple(fused),
        SemanticDetectorStatus(
            True,
            "verified",
            assets.model_version,
            requested,
            selected,
            bound_providers,
        ),
        time.monotonic() - started,
        scale,
        len(x_origins) * len(y_origins),
    )


def _minimum_area_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / min(left_area, right_area)


def corroborate_notation_candidates(
    candidates: tuple[VisualNotationCandidate, ...],
    detections: tuple[SemanticDetection, ...],
    layout: PageLayout,
) -> tuple[VisualNotationCandidate, ...]:
    """Attach class-specific semantic support to independently fitted geometry.

    Detector-only boxes are intentionally discarded.  This makes a false-positive
    neural detection unable to create a wedge, slur, or tie transaction.
    """

    spacing_by_staff = {
        int(staff.index): max(1.0, float(staff.spacing))
        for staff in layout.systems
    }
    fused: list[VisualNotationCandidate] = []
    for candidate in candidates:
        compatible = (
            {"hairpin"}
            if candidate.kind in {"crescendo", "diminuendo"}
            else ({"slur", "tie"} if candidate.kind == "curved_connector" else set())
        )
        geometry = dict(candidate.geometry)
        spacing = spacing_by_staff.get(candidate.staff_index, 12.0)
        for detection in detections:
            if (
                detection.class_name not in compatible
                or detection.staff_index != candidate.staff_index
            ):
                continue
            overlap = _minimum_area_overlap(candidate.bbox, detection.bbox)
            horizontal_overlap = max(
                0,
                min(candidate.bbox[2], detection.bbox[2])
                - max(candidate.bbox[0], detection.bbox[0]),
            )
            candidate_width = max(1, candidate.bbox[2] - candidate.bbox[0])
            center_distance = abs(
                0.5 * (candidate.bbox[1] + candidate.bbox[3])
                - 0.5 * (detection.bbox[1] + detection.bbox[3])
            )
            if (
                overlap < 0.30
                or horizontal_overlap / candidate_width < 0.55
                or center_distance > spacing * 1.75
            ):
                continue
            key = f"semantic_{detection.class_name}_support"
            geometry[key] = max(float(geometry.get(key, 0.0)), detection.confidence)
        fused.append(
            VisualNotationCandidate(
                candidate.kind,
                candidate.staff_index,
                candidate.placement,
                candidate.bbox,
                candidate.confidence,
                tuple(sorted(geometry.items())),
            )
        )
    return tuple(fused)
