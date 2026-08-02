from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scorescan.layout import PageLayout, StaffSystem
from scorescan.notation_coverage import VisualNotationCandidate
from scorescan.semantic_detector import (
    SUPPORTED_RUNTIME_CLASSES,
    SemanticDetection,
    TiledSemanticDetection,
    corroborate_notation_candidates,
    fuse_tile_fragment_detections,
    load_semantic_detector_assets,
    run_semantic_detector,
    semantic_detector_status,
)
from scorescan.semantic_detector_contract import TILE_FRAGMENT_FUSION_VERSION
from scorescan.semantic_detector_contract import (
    SEMANTIC_DETECTOR_INPUT_SIZE,
    SEMANTIC_DETECTOR_MAXIMUM_SCALE,
    SEMANTIC_DETECTOR_MAXIMUM_TILES,
    SEMANTIC_DETECTOR_MINIMUM_SCALE,
    SEMANTIC_DETECTOR_TARGET_STAFF_SPACING,
    SEMANTIC_DETECTOR_TILE_OVERLAP,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_authorized_assets(resources: Path) -> None:
    model = resources / "semantic_detector.onnx"
    categories = resources / "semantic_detector_categories.json"
    holdout = resources / "semantic_detector_holdout.json"
    parity = resources / "semantic_detector_onnx_parity.json"
    gpu_parity = resources / "semantic_detector_onnx_gpu_parity.json"
    model.write_bytes(b"fake-onnx-for-injected-session")
    categories.write_text(
        json.dumps(
            {
                "format": 1,
                "classes": [
                    {"label": index, "name": name}
                    for index, name in enumerate(
                        ["hairpin", "slur", "tie"]
                        + sorted(
                            SUPPORTED_RUNTIME_CLASSES
                            - {"hairpin", "slur", "tie"}
                        ),
                        start=1,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    holdout.write_text(
        json.dumps(
            {
                "format": 1,
                "acceptance": {"passed": True},
                "independent_works": 200,
                "integration_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    parity.write_text(
        json.dumps({"format": 1, "passed": True}),
        encoding="utf-8",
    )
    gpu_parity.write_text(
        json.dumps(
            {
                "format": 1,
                "passed": True,
                "runtime": {
                    "onnxruntime": "1.26.0",
                    "cuda_session_providers": [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                },
                "parity": {
                    "passed": True,
                    "cpu_detections": 3,
                    "cuda_detections": 3,
                    "comparison_score_floor": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "format": 1,
        "model_version": "semantic-test-1",
        "integration_authorized": True,
        "model": {
            "file": model.name,
            "bytes": model.stat().st_size,
            "sha256": _sha256(model),
        },
        "categories": {
            "file": categories.name,
            "bytes": categories.stat().st_size,
            "sha256": _sha256(categories),
        },
        "release_gate": {
            "independent_holdout": {
                "passed": True,
                "independent_works": 200,
                "file": holdout.name,
                "bytes": holdout.stat().st_size,
                "sha256": _sha256(holdout),
            },
            "onnx_parity": {
                "passed": True,
                "file": parity.name,
                "bytes": parity.stat().st_size,
                "sha256": _sha256(parity),
            },
            "onnx_gpu_parity": {
                "passed": True,
                "runtime": "onnxruntime-gpu==1.26.0",
                "file": gpu_parity.name,
                "bytes": gpu_parity.stat().st_size,
                "sha256": _sha256(gpu_parity),
            },
        },
        "operating_points": {
            name: {
                "threshold": 0.995,
                "precision": 0.995,
                "recall": 0.99,
                "true_positives": 25,
            }
            for name in SUPPORTED_RUNTIME_CLASSES
        },
        "input": {
            "name": "image",
            "size": SEMANTIC_DETECTOR_INPUT_SIZE,
            "target_staff_spacing": (
                SEMANTIC_DETECTOR_TARGET_STAFF_SPACING
            ),
            "overlap": SEMANTIC_DETECTOR_TILE_OVERLAP,
            "page_nms_iou": 0.75,
            "tile_fragment_fusion_version": TILE_FRAGMENT_FUSION_VERSION,
            "maximum_tiles": SEMANTIC_DETECTOR_MAXIMUM_TILES,
            "minimum_scale": SEMANTIC_DETECTOR_MINIMUM_SCALE,
            "maximum_scale": SEMANTIC_DETECTOR_MAXIMUM_SCALE,
        },
        "outputs": {"names": ["boxes", "scores", "labels"]},
    }
    (resources / "semantic_detector.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _layout() -> PageLayout:
    staff = StaffSystem(
        1,
        [140, 154, 168, 182, 196],
        80,
        240,
        20,
        480,
        SEMANTIC_DETECTOR_TARGET_STAFF_SPACING,
        [20, 480],
        1,
    )
    return PageLayout(512, 512, [staff], 1.0)


def _wide_layout() -> PageLayout:
    staff = StaffSystem(
        1,
        [140, 154, 168, 182, 196],
        80,
        240,
        20,
        1514,
        SEMANTIC_DETECTOR_TARGET_STAFF_SPACING,
        [20, 1514],
        1,
    )
    return PageLayout(1536, 512, [staff], 1.0)


def test_optional_semantic_detector_is_a_clean_noop_when_absent(
    tmp_path: Path,
) -> None:
    status = semantic_detector_status(tmp_path)

    assert not status.enabled
    assert status.status == "asset_absent"


def test_present_but_incomplete_detector_is_rejected_not_reported_absent(
    tmp_path: Path,
) -> None:
    (tmp_path / "semantic_detector.json").write_text(
        json.dumps({"format": 1, "integration_authorized": True}),
        encoding="utf-8",
    )

    status = semantic_detector_status(tmp_path)

    assert not status.enabled
    assert status.status.startswith("asset_rejected:")


def test_semantic_detector_assets_fail_closed_on_gate_or_hash_tampering(
    tmp_path: Path,
) -> None:
    _write_authorized_assets(tmp_path)
    manifest_path = tmp_path / "semantic_detector.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["operating_points"]["slur"]["precision"] = 0.994
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="operating point is unsafe: slur"):
        load_semantic_detector_assets(tmp_path)

    _write_authorized_assets(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input"]["page_nms_iou"] = 0.5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="tensor contract is invalid"):
        load_semantic_detector_assets(tmp_path)

    _write_authorized_assets(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["operating_points"]["slur"]["recall"] = 0.949
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="operating point is unsafe: slur"):
        load_semantic_detector_assets(tmp_path)

    _write_authorized_assets(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["operating_points"]["slur"]["recall"] = 0.985
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="high-recall mark gate is unsafe: slur"):
        load_semantic_detector_assets(tmp_path)

    _write_authorized_assets(tmp_path)
    (tmp_path / "semantic_detector.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size is invalid|hash mismatch"):
        load_semantic_detector_assets(tmp_path)


def test_semantic_detector_runs_fixed_tile_contract_and_verifies_provider(
    tmp_path: Path,
) -> None:
    _write_authorized_assets(tmp_path)
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((512, 512), 255, dtype=np.uint8))
    observed: dict[str, object] = {}

    class FakeSession:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def get_outputs(self):
            return [
                SimpleNamespace(name="boxes"),
                SimpleNamespace(name="scores"),
                SimpleNamespace(name="labels"),
            ]

        def run(self, names, feed):
            observed["names"] = names
            observed["shape"] = feed["image"].shape
            return (
                np.asarray([[[100.0, 100.0, 300.0, 130.0]]], dtype=np.float32),
                np.asarray([[0.999]], dtype=np.float32),
                np.asarray([[2]], dtype=np.int64),
            )

    def factory(path: Path, providers: list[str]):
        observed["path"] = path
        observed["providers"] = providers
        return FakeSession()

    result = run_semantic_detector(
        image_path,
        _layout(),
        tmp_path,
        "cuda",
        session_factory=factory,
    )

    assert result.status.enabled
    assert result.status.selected_accelerator == "cuda"
    assert result.tile_count == 1
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.class_name == "slur"
    assert detection.label == 2
    assert detection.bbox == (100, 100, 300, 130)
    assert detection.confidence == pytest.approx(0.999)
    assert detection.staff_index == 1
    assert detection.placement == "above"
    assert observed["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert observed["shape"] == (1, 3, 1024, 1024)
    assert observed["names"] == ["boxes", "scores", "labels"]


def test_semantic_detector_rejects_stitched_page_before_session_creation(
    tmp_path: Path,
) -> None:
    _write_authorized_assets(tmp_path)
    image_path = tmp_path / "stitched-page.png"
    cv2.imwrite(
        str(image_path),
        np.full((100, 301), 255, dtype=np.uint8),
    )

    def fail_factory(_path: Path, _providers: list[str]):
        raise AssertionError("out-of-bound page reached model session")

    result = run_semantic_detector(
        image_path,
        _layout(),
        tmp_path,
        "cpu",
        session_factory=fail_factory,
    )

    assert not result.status.enabled
    assert result.status.status == "page_exceeds_maximum_aspect_ratio:3"
    assert result.tile_count == 0


def test_semantic_detector_fuses_one_to_one_cross_tile_text_fragments() -> None:
    left_tile = (0, 0, 1024, 512)
    right_tile = (896, 0, 1920, 512)
    fragments = [
        TiledSemanticDetection(
            SemanticDetection(
                "scoreText",
                12,
                (800, 100, 1022, 130),
                0.998,
                1,
                "above",
            ),
            0,
            left_tile,
        ),
        TiledSemanticDetection(
            SemanticDetection(
                "scoreText",
                12,
                (899, 101, 1200, 131),
                0.999,
                1,
                "above",
            ),
            1,
            right_tile,
        ),
    ]

    fused = fuse_tile_fragment_detections(fragments, _layout())

    assert len(fused) == 1
    assert fused[0].bbox == (800, 100, 1200, 131)
    assert fused[0].confidence == pytest.approx(0.998)


def test_semantic_detector_run_reconstructs_a_two_tile_text_region(
    tmp_path: Path,
) -> None:
    _write_authorized_assets(tmp_path)
    image_path = tmp_path / "wide-page.png"
    cv2.imwrite(str(image_path), np.full((512, 1536), 255, dtype=np.uint8))
    categories = json.loads(
        (tmp_path / "semantic_detector_categories.json").read_text(
            encoding="utf-8"
        )
    )
    score_text_label = next(
        int(row["label"])
        for row in categories["classes"]
        if row["name"] == "scoreText"
    )

    class FragmentSession:
        calls = 0

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def get_outputs(self):
            return [
                SimpleNamespace(name="boxes"),
                SimpleNamespace(name="scores"),
                SimpleNamespace(name="labels"),
            ]

        def run(self, _names, _feed):
            boxes = (
                [[500.0, 100.0, 1022.0, 130.0]]
                if self.calls == 0
                else [[3.0, 101.0, 688.0, 131.0]]
            )
            self.calls += 1
            return (
                np.asarray([boxes], dtype=np.float32),
                np.asarray([[0.999]], dtype=np.float32),
                np.asarray([[score_text_label]], dtype=np.int64),
            )

    session = FragmentSession()
    result = run_semantic_detector(
        image_path,
        _wide_layout(),
        tmp_path,
        "cpu",
        session_factory=lambda _path, _providers: session,
    )

    assert result.tile_count == 2
    assert len(result.detections) == 1
    assert result.detections[0].class_name == "scoreText"
    assert result.detections[0].bbox == (500, 100, 1200, 131)


def test_semantic_detector_rejects_fusion_when_one_fragment_is_below_gate(
    tmp_path: Path,
) -> None:
    _write_authorized_assets(tmp_path)
    image_path = tmp_path / "wide-page.png"
    cv2.imwrite(str(image_path), np.full((512, 1536), 255, dtype=np.uint8))
    categories = json.loads(
        (tmp_path / "semantic_detector_categories.json").read_text(
            encoding="utf-8"
        )
    )
    score_text_label = next(
        int(row["label"])
        for row in categories["classes"]
        if row["name"] == "scoreText"
    )

    class WeakFragmentSession:
        calls = 0

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def get_outputs(self):
            return [
                SimpleNamespace(name="boxes"),
                SimpleNamespace(name="scores"),
                SimpleNamespace(name="labels"),
            ]

        def run(self, _names, _feed):
            first = self.calls == 0
            self.calls += 1
            return (
                np.asarray(
                    [[
                        [500.0, 100.0, 1022.0, 130.0]
                        if first
                        else [3.0, 101.0, 688.0, 131.0]
                    ]],
                    dtype=np.float32,
                ),
                np.asarray(
                    [[0.999 if first else 0.990]],
                    dtype=np.float32,
                ),
                np.asarray([[score_text_label]], dtype=np.int64),
            )

    session = WeakFragmentSession()
    result = run_semantic_detector(
        image_path,
        _wide_layout(),
        tmp_path,
        "cpu",
        session_factory=lambda _path, _providers: session,
    )

    assert result.tile_count == 2
    assert result.detections == ()


def test_semantic_detector_keeps_nested_cross_tile_slurs_distinct() -> None:
    left_tile = (0, 0, 1024, 512)
    right_tile = (896, 0, 1920, 512)
    fragments = [
        TiledSemanticDetection(
            SemanticDetection("slur", 2, (800, 95, 1023, 125), 0.999, 1, "above"),
            0,
            left_tile,
        ),
        TiledSemanticDetection(
            SemanticDetection("slur", 2, (800, 140, 1023, 180), 0.998, 1, "above"),
            0,
            left_tile,
        ),
        TiledSemanticDetection(
            SemanticDetection("slur", 2, (898, 96, 1200, 126), 0.997, 1, "above"),
            1,
            right_tile,
        ),
        TiledSemanticDetection(
            SemanticDetection("slur", 2, (898, 141, 1250, 181), 0.996, 1, "above"),
            1,
            right_tile,
        ),
    ]

    fused = fuse_tile_fragment_detections(fragments, _layout())

    assert sorted(item.bbox for item in fused) == [
        (800, 95, 1200, 126),
        (800, 140, 1250, 181),
    ]


def test_semantic_detector_rejects_silent_cuda_fallback(tmp_path: Path) -> None:
    _write_authorized_assets(tmp_path)
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.full((512, 512), 255, dtype=np.uint8))

    class CpuSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    with pytest.raises(RuntimeError, match="provider verification mismatch"):
        run_semantic_detector(
            image_path,
            _layout(),
            tmp_path,
            "cuda",
            session_factory=lambda _path, _providers: CpuSession(),
        )


def test_semantic_detections_only_corroborate_overlapping_source_geometry() -> None:
    candidates = (
        VisualNotationCandidate(
            "curved_connector",
            1,
            "above",
            (100, 100, 300, 130),
            0.80,
            (("fit_p90_spaces", 0.05), ("length_spaces", 14.0)),
        ),
        VisualNotationCandidate(
            "crescendo",
            1,
            "below",
            (100, 220, 300, 250),
            0.82,
            (
                ("apex_separation_spaces", 0.1),
                ("length_spaces", 14.0),
                ("open_separation_spaces", 1.0),
            ),
        ),
    )
    detections = (
        SemanticDetection("slur", 2, (90, 95, 310, 135), 0.999, 1, "above"),
        # Same class but no spatial support: it must not affect the wedge.
        SemanticDetection("hairpin", 1, (350, 220, 480, 250), 0.999, 1, "below"),
    )

    fused = corroborate_notation_candidates(candidates, detections, _layout())

    assert dict(fused[0].geometry)["semantic_slur_support"] == 0.999
    assert "semantic_hairpin_support" not in dict(fused[1].geometry)
    assert fused[0].confidence == candidates[0].confidence
