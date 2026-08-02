from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.tools import prepare_muse_omr_scan_regions as module


def _reference_page(size: int = 640) -> np.ndarray:
    page = np.full((size, size), 255, dtype=np.uint8)
    for y in range(120, 521, 80):
        cv2.line(page, (60, y), (580, y), 0, 3)
        cv2.line(page, (60, y + 12), (580, y + 12), 0, 2)
    for x, y in ((130, 120), (250, 212), (390, 372), (520, 452)):
        cv2.ellipse(page, (x, y), (12, 8), -15, 0, 360, 0, -1)
        cv2.line(page, (x + 10, y), (x + 10, y - 55), 0, 3)
    cv2.putText(
        page,
        "Allegro mf",
        (90, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        0,
        2,
        cv2.LINE_AA,
    )
    return page


def _mean_absolute_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(first.astype(np.float32) - second.astype(np.float32))
        )
    )


def test_prune_unselected_pair_caches_is_exact_and_auditable(
    tmp_path: Path,
) -> None:
    for area in ("acceptances", "rejections", "pages", "reference_pages"):
        (tmp_path / area).mkdir()
    (tmp_path / "acceptances" / "pair-0007.json").write_bytes(b"keep")
    (tmp_path / "acceptances" / "pair-0008.json").write_bytes(b"old-a")
    (tmp_path / "rejections" / "pair-0008.json").write_bytes(b"old-r")
    (tmp_path / "rejections" / "notes.json").write_bytes(b"unrelated")
    (tmp_path / "pages" / "pair-0007").mkdir()
    stale_page = tmp_path / "pages" / "pair-0008"
    stale_page.mkdir()
    (stale_page / "page-1.jpg").write_bytes(b"old-page")
    stale_reference = tmp_path / "reference_pages" / "pair-0008"
    stale_reference.mkdir()
    (stale_reference / "page-1.svg").write_bytes(b"old-reference")
    (tmp_path / "pages" / "scratch").mkdir()

    report = module._prune_unselected_pair_caches(tmp_path, [7])

    assert report == {
        "contract": "scorescan-selected-pair-cache-prune@1",
        "selected_pairs": 1,
        "removed_files": 2,
        "removed_directories": 2,
        "removed_bytes": (
            len(b"old-a")
            + len(b"old-r")
            + len(b"old-page")
            + len(b"old-reference")
        ),
        "by_area": {
            "acceptances": {
                "files": 1,
                "directories": 0,
                "bytes": len(b"old-a"),
            },
            "rejections": {
                "files": 1,
                "directories": 0,
                "bytes": len(b"old-r"),
            },
            "pages": {
                "files": 0,
                "directories": 1,
                "bytes": len(b"old-page"),
            },
            "reference_pages": {
                "files": 0,
                "directories": 1,
                "bytes": len(b"old-reference"),
            },
        },
    }
    assert (tmp_path / "acceptances" / "pair-0007.json").is_file()
    assert (tmp_path / "pages" / "pair-0007").is_dir()
    assert (tmp_path / "rejections" / "notes.json").is_file()
    assert (tmp_path / "pages" / "scratch").is_dir()
    assert not (tmp_path / "acceptances" / "pair-0008.json").exists()
    assert not (tmp_path / "pages" / "pair-0008").exists()


def test_registration_recovers_small_scan_affine_offset() -> None:
    reference = _reference_page()
    scan_to_reference = np.array(
        [[1.002, 0.001, 3.0], [-0.001, 0.998, -2.0]],
        dtype=np.float32,
    )
    scan = cv2.warpAffine(
        reference,
        scan_to_reference,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )
    scan = cv2.GaussianBlur(scan, (3, 3), 0.5)
    gradient = np.linspace(0, 8, scan.shape[1], dtype=np.float32)[None, :]
    scan = np.clip(scan.astype(np.float32) + gradient, 0, 255).astype(np.uint8)

    before = _mean_absolute_error(scan, reference)
    registered, report = module.register_scan_page(
        scan,
        reference,
        minimum_ecc=0.75,
        maximum_linear_deviation=0.025,
        maximum_translation_fraction=0.025,
    )

    assert report["ecc"] > 0.9
    assert report["quality_policy_version"] == (
        module.REGISTRATION_QUALITY_POLICY_VERSION
    )
    assert report["effective_minimum_local_correlation_10p"] == (
        module.MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P
    )
    assert report["effective_minimum_median_local_correlation"] == (
        module.MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION
    )
    assert _mean_absolute_error(registered, reference) < before * 0.7
    assert registered.shape == reference.shape


def test_registration_rejects_unrelated_page() -> None:
    reference = _reference_page()
    unrelated = np.full_like(reference, 255)
    cv2.rectangle(unrelated, (80, 80), (560, 560), 0, 12)
    cv2.putText(
        unrelated,
        "NOT MUSIC",
        (150, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        0,
        5,
        cv2.LINE_AA,
    )

    with pytest.raises(ValueError, match="registration"):
        module.register_scan_page(
            unrelated,
            reference,
            minimum_ecc=0.86,
            maximum_linear_deviation=0.025,
            maximum_translation_fraction=0.025,
        )


def test_registration_ignores_broad_page_stain_behind_notation() -> None:
    reference = _reference_page()
    damaged_background = np.full_like(reference, 255)
    cv2.ellipse(
        damaged_background,
        (430, 280),
        (165, 225),
        12,
        0,
        360,
        105,
        -1,
    )
    scan = np.minimum(damaged_background, reference)
    scan = cv2.GaussianBlur(scan, (3, 3), 0.5)

    registered, report = module.register_scan_page(
        scan,
        reference,
        minimum_ecc=0.86,
        maximum_linear_deviation=0.1,
        maximum_translation_fraction=0.1,
        minimum_local_correlation_10p=0.62,
        minimum_median_local_correlation=0.72,
    )

    assert report["ecc"] > 0.9
    assert report["local_correlation_median"] > 0.8
    assert registered.shape == reference.shape


def test_registration_recovers_only_bounded_smooth_paper_deformation() -> None:
    reference = cv2.resize(
        _reference_page(),
        (1200, 1200),
        interpolation=cv2.INTER_CUBIC,
    )
    height, width = reference.shape
    grid_y, grid_x = np.mgrid[:height, :width].astype(np.float32)
    displacement_x = 5.0 * np.sin(grid_y / 160.0)
    displacement_y = 8.0 * np.sin(grid_x / 170.0)
    scan = cv2.remap(
        reference,
        grid_x + displacement_x.astype(np.float32),
        grid_y + displacement_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    scan = cv2.GaussianBlur(scan, (3, 3), 0.35)

    registered, report = module.register_scan_page(
        scan,
        reference,
        minimum_ecc=0.86,
        maximum_linear_deviation=0.12,
        maximum_translation_fraction=0.1,
    )

    assert report["method"].endswith("_bounded_elastic")
    assert report["ecc"] < 0.86
    assert report["elastic_flow_magnitude_fraction_999"] <= (
        module.ELASTIC_FLOW_MAXIMUM_FRACTION
    )
    assert report["elastic_jacobian_minimum"] >= (
        module.ELASTIC_MINIMUM_ABSOLUTE_JACOBIAN
    )
    assert report["elastic_jacobian_maximum"] <= (
        module.ELASTIC_MAXIMUM_ABSOLUTE_JACOBIAN
    )
    assert report["local_correlation_10p"] >= (
        module.MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P
    )
    assert report["local_correlation_median"] >= (
        module.MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION
    )
    assert registered.shape == reference.shape


def test_bounded_elastic_registration_cannot_rewrite_unrelated_symbols() -> None:
    reference = cv2.resize(
        _reference_page(),
        (1200, 1200),
        interpolation=cv2.INTER_CUBIC,
    )
    unrelated = np.full_like(reference, 255)
    for y in range(225, 978, 150):
        cv2.line(unrelated, (112, y), (1088, y), 0, 5)
        cv2.line(unrelated, (112, y + 22), (1088, y + 22), 0, 4)
    for x, y in ((300, 225), (520, 397), (760, 697), (900, 847)):
        cv2.ellipse(unrelated, (x, y), (22, 14), 15, 0, 360, 0, -1)
        cv2.line(unrelated, (x + 18, y), (x + 18, y - 100), 0, 5)
    cv2.putText(
        unrelated,
        "WRONG",
        (200, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        0,
        5,
        cv2.LINE_AA,
    )

    with pytest.raises(ValueError, match="elastic"):
        module._bounded_elastic_registration(
            unrelated,
            reference,
            minimum_local_correlation_10p=0.62,
            minimum_median_local_correlation=0.72,
        )


def test_local_quality_is_bounded_and_uses_strict_downsampled_floors() -> None:
    reference = cv2.resize(
        _reference_page(),
        (2200, 2200),
        interpolation=cv2.INTER_CUBIC,
    )
    report = module._local_alignment_quality(reference, reference)

    assert report["local_quality_downsampled"] is True
    assert max(
        report["local_quality_evaluation_width"],
        report["local_quality_evaluation_height"],
    ) == module.LOCAL_ALIGNMENT_MAXIMUM_DIMENSION
    assert report["local_correlation_10p"] > (
        module.MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P
    )
    assert report["local_correlation_median"] > (
        module.MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION
    )


def test_cached_registration_is_upgraded_to_current_quality_policy(
    tmp_path: Path,
) -> None:
    reference = _reference_page()
    svg_path = tmp_path / "reference.svg"
    reference_path = tmp_path / "reference.png"
    svg_path.write_text("<svg/>", encoding="utf-8")
    assert cv2.imwrite(str(reference_path), reference)
    destination = tmp_path / "pages" / "pair-0001"
    destination.mkdir(parents=True)
    registered_path = destination / "page-1.jpg"
    storage = module._write_registered_page(registered_path, reference)
    storage.pop("sha256")
    report = {
        "version": module.REGISTRATION_VERSION,
        "pages": [
            {
                "page": 1,
                "registered_image": registered_path.name,
                "storage": storage,
            }
        ],
    }

    upgraded = module._cached_registration_under_current_quality_policy(
        report=report,
        destination=destination,
        reference_pages=[(svg_path, reference_path)],
        minimum_local_correlation_10p=0.62,
        minimum_median_local_correlation=0.72,
    )

    assert upgraded is not None
    page = upgraded["pages"][0]
    assert page["quality_policy_version"] == (
        module.REGISTRATION_QUALITY_POLICY_VERSION
    )
    assert max(
        page["local_quality_evaluation_width"],
        page["local_quality_evaluation_height"],
    ) <= module.LOCAL_ALIGNMENT_MAXIMUM_DIMENSION
    assert page["local_correlation_10p"] >= (
        module.MINIMUM_DOWNSAMPLED_LOCAL_CORRELATION_10P
    )
    assert page["local_correlation_median"] >= (
        module.MINIMUM_DOWNSAMPLED_MEDIAN_LOCAL_CORRELATION
    )
    assert len(page["storage"]["sha256"]) == 64
    persisted = json.loads(
        (destination / "registration.json").read_text(encoding="utf-8")
    )
    assert persisted["quality_policy_version"] == (
        module.REGISTRATION_QUALITY_POLICY_VERSION
    )


def test_cached_registration_rejects_image_integrity_mismatch(
    tmp_path: Path,
) -> None:
    reference = _reference_page()
    svg_path = tmp_path / "reference.svg"
    reference_path = tmp_path / "reference.png"
    svg_path.write_text("<svg/>", encoding="utf-8")
    assert cv2.imwrite(str(reference_path), reference)
    destination = tmp_path / "pages" / "pair-0001"
    destination.mkdir(parents=True)
    registered_path = destination / "page-1.jpg"
    storage = module._write_registered_page(registered_path, reference)
    registered_path.write_bytes(b"corrupted")
    report = {
        "version": module.REGISTRATION_VERSION,
        "pages": [
            {
                "page": 1,
                "registered_image": registered_path.name,
                "storage": storage,
                "quality_policy_version": (
                    module.REGISTRATION_QUALITY_POLICY_VERSION
                ),
                "local_quality_downsampled": True,
                "local_quality_evaluation_width": 640,
                "local_quality_evaluation_height": 640,
                "effective_minimum_local_correlation_10p": 0.85,
                "effective_minimum_median_local_correlation": 0.92,
                "local_correlation_10p": 1.0,
                "local_correlation_median": 1.0,
            }
        ],
    }

    assert module._cached_registration_under_current_quality_policy(
        report=report,
        destination=destination,
        reference_pages=[(svg_path, reference_path)],
        minimum_local_correlation_10p=0.62,
        minimum_median_local_correlation=0.72,
    ) is None


def test_stable_subset_is_deterministic_and_validates_limit() -> None:
    first = module._stable_subset([9, 1, 7, 3, 7], 3, 29)
    second = module._stable_subset([3, 9, 1, 7], 3, 29)
    assert first == second
    assert len(first) == 3
    assert module._stable_subset([3, 1], 5, 29) == [1, 3]
    with pytest.raises(ValueError, match="positive"):
        module._stable_subset([1], 0, 29)


def test_training_selection_requires_disjoint_pinned_role(
    tmp_path: Path,
) -> None:
    valid = {
        "repository": module.REPOSITORY,
        "revision": module.REVISION,
        "license": module.LICENSE,
        "role": "external_scan_degraded_training_only",
        "source_image_origin": module.SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "training_holdout_overlap": [],
        "training_holdout_work_overlap": [],
        "selected_pair_ids": [1, 2],
        "selected_work_count": 2,
        "selected_work_fingerprints": ["a" * 64, "b" * 64],
        "pair_work_fingerprints": [
            {"pair_id": 1, "work_fingerprint": "a" * 64},
            {"pair_id": 2, "work_fingerprint": "b" * 64},
        ],
        "work_catalog_sha256": "c" * 64,
        "reserved_holdout_pair_ids": [7, 8],
        "reserved_holdout_work_fingerprints": ["d" * 64, "e" * 64],
    }
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(valid), encoding="utf-8")
    assert module._load_training_selection(tmp_path) == valid

    missing_origin = dict(valid)
    missing_origin.pop("source_image_origin")
    selection.write_text(json.dumps(missing_origin), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned Muse OMR role"):
        module._load_training_selection(tmp_path)

    overlapping = dict(valid)
    overlapping["selected_pair_ids"] = [1, 7]
    overlapping["pair_work_fingerprints"] = [
        {"pair_id": 1, "work_fingerprint": "a" * 64},
        {"pair_id": 7, "work_fingerprint": "b" * 64},
    ]
    selection.write_text(json.dumps(overlapping), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        module._load_training_selection(tmp_path)

    wrong_role = dict(valid)
    wrong_role["role"] = "external_scan_degraded_development_benchmark_not_training"
    selection.write_text(json.dumps(wrong_role), encoding="utf-8")
    with pytest.raises(ValueError, match="role"):
        module._load_training_selection(tmp_path)


def test_benchmark_selection_is_not_training_and_can_be_cross_checked(
    tmp_path: Path,
) -> None:
    benchmark = {
        "repository": module.REPOSITORY,
        "revision": module.REVISION,
        "license": module.LICENSE,
        "role": module.BENCHMARK_SELECTION_ROLE,
        "source_image_origin": module.SCAN_DEGRADED_IMAGE_ORIGIN,
        "production_evidence_eligible": False,
        "selected_pair_ids": [7, 8],
        "selected_work_count": 2,
        "selected_work_fingerprints": ["d" * 64, "e" * 64],
        "pair_work_fingerprints": [
            {"pair_id": 7, "work_fingerprint": "d" * 64},
            {"pair_id": 8, "work_fingerprint": "e" * 64},
        ],
        "work_catalog_sha256": "c" * 64,
    }
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(benchmark), encoding="utf-8")
    assert module._load_selection(
        tmp_path,
        expected_role=module.BENCHMARK_SELECTION_ROLE,
    ) == benchmark
    ids, works, digest = module._selection_ids(selection)
    assert ids == {7, 8}
    assert works == {"d" * 64, "e" * 64}
    assert len(digest) == 64
    missing_origin = dict(benchmark)
    missing_origin.pop("source_image_origin")
    selection.write_text(json.dumps(missing_origin), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned Muse OMR role"):
        module._load_selection(
            tmp_path,
            expected_role=module.BENCHMARK_SELECTION_ROLE,
        )
    selection.write_text(json.dumps(benchmark), encoding="utf-8")
    with pytest.raises(ValueError, match="role"):
        module._load_selection(
            tmp_path,
            expected_role=module.TRAINING_SELECTION_ROLE,
        )


def test_registered_pages_keeps_valid_pages_and_rejects_bad_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"scan")
    references = []
    for page_number in (1, 2):
        svg = tmp_path / f"page-{page_number}.svg"
        png = tmp_path / f"page-{page_number}.png"
        svg.write_text("<svg/>", encoding="utf-8")
        png.write_bytes(b"reference")
        references.append((svg, png))

    page = _reference_page(256)
    monkeypatch.setattr(
        module,
        "_render_pdf_pages",
        lambda _pdf, _references: ([page, page], 2),
    )
    monkeypatch.setattr(
        module.cv2,
        "imread",
        lambda _path, _mode: page,
    )
    calls = 0

    def fake_register(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("deliberately unregistrable")
        return page, {"ecc": 0.99}

    monkeypatch.setattr(module, "register_scan_page", fake_register)

    registered, report = module._registered_pages(
        pair_id=19,
        pdf_path=pdf,
        reference_pages=references,
        output_dir=tmp_path / "output",
        minimum_ecc=0.86,
        maximum_linear_deviation=0.12,
        maximum_translation_fraction=0.1,
        minimum_local_correlation_10p=0.62,
        minimum_median_local_correlation=0.72,
        minimum_accepted_page_fraction=0.5,
    )

    assert len(registered) == 1
    assert report["accepted_page_fraction"] == 0.5
    assert report["rejected_pages"] == [
        {"page": 2, "error": "deliberately unregistrable"}
    ]
    assert registered[0][1].name == "page-1.jpg"
    assert report["pages"][0]["registered_image"] == "page-1.jpg"
    assert report["pages"][0]["storage"]["quality"] == 95


def test_registered_pages_salvages_strictly_matching_prefix_after_reflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"scan")
    references = []
    for page_number in (1, 2):
        svg = tmp_path / f"page-{page_number}.svg"
        png = tmp_path / f"page-{page_number}.png"
        svg.write_text("<svg/>", encoding="utf-8")
        png.write_bytes(b"reference")
        references.append((svg, png))
    page = _reference_page(256)
    monkeypatch.setattr(
        module,
        "_render_pdf_pages",
        lambda _pdf, _references: ([page], 1),
    )
    monkeypatch.setattr(module.cv2, "imread", lambda _path, _mode: page)
    monkeypatch.setattr(
        module,
        "register_scan_page",
        lambda *_args, **_kwargs: (page, {"ecc": 0.99}),
    )

    registered, report = module._registered_pages(
        pair_id=23,
        pdf_path=pdf,
        reference_pages=references,
        output_dir=tmp_path / "output",
        minimum_ecc=0.86,
        maximum_linear_deviation=0.12,
        maximum_translation_fraction=0.1,
        minimum_local_correlation_10p=0.62,
        minimum_median_local_correlation=0.72,
        minimum_accepted_page_fraction=0.5,
    )

    assert len(registered) == 1
    assert report["pdf_pages"] == 1
    assert report["reference_pages"] == 2
    assert report["page_denominator"] == 2
    assert report["accepted_page_fraction"] == 0.5
    assert report["rejected_pages"][0]["page"] == 2
    assert "page-count mismatch" in report["rejected_pages"][0]["error"]


def test_rejected_pair_cache_cleanup_is_scoped(tmp_path: Path) -> None:
    rejected_reference = tmp_path / "reference_pages" / "pair-0007"
    rejected_scan = tmp_path / "pages" / "pair-0007"
    retained = tmp_path / "pages" / "pair-0008"
    for path in (rejected_reference, rejected_scan, retained):
        path.mkdir(parents=True)
        (path / "marker").write_text("x", encoding="utf-8")

    module._discard_rejected_pair_cache(tmp_path, 7)

    assert not rejected_reference.exists()
    assert not rejected_scan.exists()
    assert retained.is_dir()


def test_coverage_gate_requires_independent_works_not_just_variants() -> None:
    assert module._coverage_failures(
        selected_pairs=420,
        accepted_pairs=300,
        accepted_works=199,
        minimum_accepted_fraction=0.5,
        minimum_accepted_works=200,
    ) == ["accepted independent works 199 < 200"]
    assert module._coverage_failures(
        selected_pairs=420,
        accepted_pairs=210,
        accepted_works=205,
        minimum_accepted_fraction=0.5,
        minimum_accepted_works=200,
    ) == []


def test_rejection_cache_requires_exact_input_signature(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rejections" / "pair-0007.json"
    signature = {"registration_version": "v1", "pdf_sha256": "a" * 64}
    rejection = {"pair_id": 7, "error": "strict rejection"}
    module._atomic_json(
        path,
        {
            "format": 1,
            "signature": signature,
            "rejection": rejection,
        },
    )

    assert module._load_cached_rejection(path, signature) == rejection
    assert module._load_cached_rejection(
        path,
        {**signature, "pdf_sha256": "b" * 64},
    ) is None
    assert not path.exists()


def test_acceptance_cache_replays_only_complete_bounded_tiles(
    tmp_path: Path,
) -> None:
    image = tmp_path / "pages" / "pair-0007" / "page-1.jpg"
    image.parent.mkdir(parents=True)
    storage = module._write_registered_page(image, _reference_page())
    module._atomic_json(
        image.parent / "registration.json",
        {
            "version": module.REGISTRATION_VERSION,
            "quality_policy_version": (
                module.REGISTRATION_QUALITY_POLICY_VERSION
            ),
            "page_denominator": 1,
            "pages": [
                {
                    "page": 1,
                    "registered_image": image.name,
                    "storage": storage,
                    "quality_policy_version": (
                        module.REGISTRATION_QUALITY_POLICY_VERSION
                    ),
                    "local_correlation_10p": 0.95,
                    "local_correlation_median": 0.98,
                }
            ],
        },
    )
    path = tmp_path / "acceptances" / "pair-0007.json"
    signature = {
        "registration_version": module.REGISTRATION_VERSION,
        "registration_quality_policy_version": (
            module.REGISTRATION_QUALITY_POLICY_VERSION
        ),
        "pdf_sha256": "a" * 64,
        "tile_size": 1024,
        "minimum_local_correlation_10p": 0.62,
        "minimum_median_local_correlation": 0.72,
        "minimum_accepted_page_fraction": 0.75,
    }
    payload = {
        "format": 1,
        "signature": signature,
        "accepted": {"pair_id": 7, "split": "train"},
        "rows": [
            {
                "image": "pages/pair-0007/page-1.jpg",
                "objects": [{"category_id": "genericAccidental"}],
            }
        ],
        "counters": {
            "negative_tiles": 0,
            "dropped_counts": {},
            "excluded_page_counts": {},
        },
    }
    module._atomic_json(path, payload)

    assert (
        module._load_cached_acceptance(path, signature, tmp_path)
        == payload
    )
    image.unlink()
    assert module._load_cached_acceptance(path, signature, tmp_path) is None
    assert not path.exists()


def test_registered_page_shape_rejects_stitched_whole_work(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    cv2.imwrite(str(good), np.full((100, 300), 255, dtype=np.uint8))
    cv2.imwrite(str(bad), np.full((100, 301), 255, dtype=np.uint8))

    audit = module._validate_registered_page_shapes(
        1,
        [(tmp_path / "good.svg", good)],
    )
    assert audit[0]["aspect_ratio"] == 3.0

    with pytest.raises(ValueError, match=r"pair=2.*301x100"):
        module._validate_registered_page_shapes(
            2,
            [(tmp_path / "bad.svg", bad)],
        )
