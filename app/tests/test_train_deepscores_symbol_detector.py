from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

import app.tools.train_deepscores_symbol_detector as detector_training
from app.tools.train_deepscores_symbol_detector import (
    assert_compatible_category_manifests,
    assert_complete_page_target_dataset,
    assert_matching_run_config,
    assert_training_dataset_manifest,
    category_label_name_map,
    class_aware_sample_weights,
    detector_class_counts,
    detector_acceptance_failures,
    detector_acceptance_gates_configured,
    detector_checkpoint_position,
    detector_microbatch_plan,
    detector_runtime_final_epoch,
    detector_runtime_microbatch_size,
    detector_selection_evidence_failures,
    detector_occurrence_seed,
    evaluation_detection_limit,
    finalize_detector_recovery_checkpoint,
    insufficient_required_class_support,
    is_priority_mark_class,
    json_ready,
    legacy_detector_sampled_indices,
    load_grayscale_crop,
    main as train_detector_main,
    normalize_target_boxes,
    parse_required_class_maps,
    prepare_verified_detector_image_cache,
    priority_selection_score,
    reopen_runtime_truncated_detector_run,
    remove_empty_detector_resume_directory,
    remaining_detector_occurrences,
    reconcile_resume_run_config,
    replay_mixture_sample_weights,
    resolve_detector_device,
    reusable_zero_clip_target_box_audit,
    bind_zero_clip_target_box_audit,
    should_replace_detector_best,
    stable_subset,
    support_filtered_macro_map,
    synthetic_training_evidence,
)


def test_complete_page_training_contract_refuses_legacy_or_dropped_spans(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    manifest = {
        "target_assignment_version": (
            detector_training.COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
        ),
        "target_geometry_provenance": (
            detector_training.COMPLETE_PAGE_TARGET_PROVENANCE
        ),
        "minimum_object_fraction": 0.8,
        "long_span_minimum_object_fraction": 0.25,
        "tile_size": 1024,
        "overlap": 256,
        "oversized_fragment_visibility_version": (
            detector_training.OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
    }
    report = {
        "transformation_version": (
            detector_training.COMPLETE_PAGE_TARGET_ASSIGNMENT_VERSION
        ),
        "target_geometry_provenance": (
            detector_training.COMPLETE_PAGE_TARGET_PROVENANCE
        ),
        "minimum_object_fraction": 0.8,
        "long_span_minimum_object_fraction": 0.25,
        "dropped_object_counts": {},
        "tile_size": 1024,
        "overlap": 256,
        "oversized_fragment_visibility_version": (
            detector_training.OVERSIZED_FRAGMENT_VISIBILITY_VERSION
        ),
    }
    report_path = prepared / "prepare-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert_complete_page_target_dataset(prepared, manifest)

    legacy_manifest = {**manifest, "target_assignment_version": "legacy"}
    with pytest.raises(ValueError, match="complete-page target assignment"):
        assert_complete_page_target_dataset(prepared, legacy_manifest)

    report["dropped_object_counts"] = {"hairpin": 1, "notehead": 99}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="hairpin=1"):
        assert_complete_page_target_dataset(prepared, manifest)


def test_detector_recovery_checkpoint_is_kept_only_for_failed_gate(
    tmp_path: Path,
) -> None:
    accepted = tmp_path / "accepted.pt"
    accepted.write_bytes(b"accepted checkpoint")
    assert (
        finalize_detector_recovery_checkpoint(
            accepted,
            acceptance_failures=[],
        )
        is False
    )
    assert not accepted.exists()

    failed = tmp_path / "failed.pt"
    failed.write_bytes(b"failed checkpoint")
    assert (
        finalize_detector_recovery_checkpoint(
            failed,
            acceptance_failures=["priority map below floor"],
        )
        is True
    )
    assert failed.read_bytes() == b"failed checkpoint"

    external = tmp_path / "external.pt"
    external.write_bytes(b"pending independent holdout")
    assert (
        finalize_detector_recovery_checkpoint(
            external,
            acceptance_failures=[],
            external_acceptance_pending=True,
        )
        is True
    )
    assert external.read_bytes() == b"pending independent holdout"


def test_detector_runtime_epoch_cap_requires_audited_evaluation_boundary() -> None:
    assert detector_runtime_final_epoch(
        planned_epochs=4,
        eval_every=2,
        stop_after_epoch=2,
        stop_reason="superseded_by_stricter_downstream_stage",
    ) == 2
    assert detector_runtime_final_epoch(
        planned_epochs=4,
        eval_every=2,
        stop_after_epoch=None,
        stop_reason=None,
    ) == 4
    with pytest.raises(ValueError, match="evaluated epoch"):
        detector_runtime_final_epoch(
            planned_epochs=4,
            eval_every=2,
            stop_after_epoch=1,
            stop_reason="too_early",
        )
    with pytest.raises(ValueError, match="reason is required"):
        detector_runtime_final_epoch(
            planned_epochs=4,
            eval_every=2,
            stop_after_epoch=2,
            stop_reason=None,
        )


def test_detector_microbatch_plan_preserves_logical_batch_mean() -> None:
    assert detector_microbatch_plan(4, 1) == [
        (0, 1, 0.25),
        (1, 2, 0.25),
        (2, 3, 0.25),
        (3, 4, 0.25),
    ]
    assert detector_microbatch_plan(5, 2) == [
        (0, 2, 0.4),
        (2, 4, 0.4),
        (4, 5, 0.2),
    ]
    assert sum(weight for _, _, weight in detector_microbatch_plan(5, 2)) == 1
    with pytest.raises(ValueError, match="cannot exceed"):
        detector_microbatch_plan(1, 2)


def test_detector_microbatch_weighting_matches_logical_batch_gradient() -> None:
    inputs = [1.0, 2.0, 4.0, 8.0]
    targets = [0.5, -1.0, 3.0, 2.0]
    weight = 0.75
    direct_gradient = sum(
        2.0 * value * (weight * value - target)
        for value, target in zip(inputs, targets, strict=True)
    ) / len(inputs)

    micro_gradient = 0.0
    for start, end, fraction in detector_microbatch_plan(4, 1):
        local_gradient = sum(
            2.0 * value * (weight * value - target)
            for value, target in zip(
                inputs[start:end],
                targets[start:end],
                strict=True,
            )
        ) / (end - start)
        micro_gradient += fraction * local_gradient

    assert micro_gradient == pytest.approx(direct_gradient, abs=1e-12)


def test_detector_adaptive_microbatch_uses_only_sparse_fast_path() -> None:
    sparse = [{"boxes": [object()] * 35}, {"boxes": [object()] * 45}]
    dense = [{"boxes": [object()] * 40}, {"boxes": [object()] * 41}]

    assert detector_runtime_microbatch_size(
        sparse,
        configured_microbatch_size=1,
        adaptive_full_batch_object_limit=80,
        adaptive_full_batch_min_free_mib=4608,
        effective_free_mib=5000,
    ) == (2, 80)
    assert detector_runtime_microbatch_size(
        dense,
        configured_microbatch_size=1,
        adaptive_full_batch_object_limit=80,
        adaptive_full_batch_min_free_mib=4608,
        effective_free_mib=5000,
    ) == (1, 81)
    assert detector_runtime_microbatch_size(
        sparse,
        configured_microbatch_size=1,
        adaptive_full_batch_object_limit=80,
        adaptive_full_batch_min_free_mib=4608,
        effective_free_mib=4607,
    ) == (1, 80)
    assert detector_runtime_microbatch_size(
        sparse,
        configured_microbatch_size=1,
        adaptive_full_batch_object_limit=None,
    ) == (1, 80)


def test_detector_adaptive_microbatch_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="positive"):
        detector_runtime_microbatch_size(
            [{"boxes": []}],
            configured_microbatch_size=1,
            adaptive_full_batch_object_limit=0,
        )
    with pytest.raises(ValueError, match="missing boxes"):
        detector_runtime_microbatch_size(
            [{}],
            configured_microbatch_size=1,
            adaptive_full_batch_object_limit=80,
        )
    with pytest.raises(ValueError, match="current memory evidence"):
        detector_runtime_microbatch_size(
            [{"boxes": []}],
            configured_microbatch_size=1,
            adaptive_full_batch_object_limit=80,
            adaptive_full_batch_min_free_mib=4608,
        )


def test_verified_detector_image_cache_is_content_addressed_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    first = source_root / "pages" / "first.png"
    second = source_root / "pages" / "second.png"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"same-image-bytes")
    second.write_bytes(first.read_bytes())
    items = [
        ({"image": "pages/first.png"}, source_root),
        ({"image": "pages/second.png"}, source_root),
    ] + [({"image": "pages/first.png"}, source_root)] * 100
    original_resolve = Path.resolve
    resolve_calls = 0

    def counted_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        nonlocal resolve_calls
        resolve_calls += 1
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted_resolve)

    mappings, report = prepare_verified_detector_image_cache(
        items,
        tmp_path / "cache",
        populate=True,
    )

    assert report["contract"] == "scorescan-byte-verified-detector-image-cache@1"
    assert report["source_files"] == 2
    assert report["unique_blobs"] == 1
    assert report["bytes"] == len(b"same-image-bytes")
    assert resolve_calls == 3
    assert mappings[(str(source_root), "pages/first.png")] == mappings[
        (str(source_root), "pages/second.png")
    ]
    assert mappings[(str(source_root), "pages/first.png")].read_bytes() == (
        b"same-image-bytes"
    )

    copy_calls = 0
    original_copy = detector_training._copy_file_with_sha256

    def counted_copy(source: Path, destination: Path) -> tuple[str, int]:
        nonlocal copy_calls
        copy_calls += 1
        return original_copy(source, destination)

    monkeypatch.setattr(
        detector_training,
        "_copy_file_with_sha256",
        counted_copy,
    )
    repopulated, repopulated_report = prepare_verified_detector_image_cache(
        items,
        tmp_path / "cache",
        populate=True,
    )
    assert repopulated == mappings
    assert repopulated_report["manifest_sha256"] == report["manifest_sha256"]
    assert copy_calls == 0

    verified, second_report = prepare_verified_detector_image_cache(
        items,
        tmp_path / "cache",
        populate=False,
    )
    assert verified == mappings
    assert second_report["populated"] is False


def test_verified_detector_image_cache_fails_closed_on_tampering(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    image = source_root / "page.png"
    source_root.mkdir()
    image.write_bytes(b"trusted")
    mappings, _ = prepare_verified_detector_image_cache(
        [({"image": "page.png"}, source_root)],
        tmp_path / "cache",
        populate=True,
    )
    mappings[(str(source_root), "page.png")].write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="size mismatch|hash mismatch"):
        prepare_verified_detector_image_cache(
            [({"image": "page.png"}, source_root)],
            tmp_path / "cache",
            populate=False,
        )


def test_verified_detector_image_cache_rejects_root_escape(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        prepare_verified_detector_image_cache(
            [({"image": "../outside.png"}, tmp_path)],
            tmp_path / "cache",
            populate=True,
        )


def test_detector_report_json_normalization_is_recursive(tmp_path: Path) -> None:
    value = {
        "direct": tmp_path / "direct",
        "nested": [tmp_path / "nested", {"count": 3}],
    }
    assert json_ready(value) == {
        "direct": str(tmp_path / "direct"),
        "nested": [str(tmp_path / "nested"), {"count": 3}],
    }


def test_detector_device_selection_never_silently_occupies_cuda() -> None:
    assert resolve_detector_device("cpu", cuda_available=True) == "cpu"
    assert resolve_detector_device("auto", cuda_available=False) == "cpu"
    assert resolve_detector_device("auto", cuda_available=True) == "cuda"
    with pytest.raises(RuntimeError, match="not available"):
        resolve_detector_device("cuda", cuda_available=False)
    with pytest.raises(ValueError, match="unsupported"):
        resolve_detector_device("other", cuda_available=True)


def test_stable_subset_is_order_independent() -> None:
    rows = [
        {
            "split": "train",
            "image_id": str(index),
            "crop_xyxy": [0, 0, 10, 10],
        }
        for index in range(10)
    ]
    first = stable_subset(rows, 4, 42)
    second = stable_subset(list(reversed(rows)), 4, 42)
    assert first == second
    assert len(first) == 4


def test_stable_subset_returns_all_when_limit_is_large() -> None:
    rows = [{"split": "test", "image_id": "1", "crop_xyxy": [0, 0, 1, 1]}]
    assert stable_subset(rows, None, 1) is rows
    assert stable_subset(rows, 5, 1) is rows


def test_detector_occurrence_seed_is_stable_and_coordinate_specific() -> None:
    first = detector_occurrence_seed(
        base_seed=42,
        epoch_number=3,
        sample_position=1200,
        item_index=17,
    )
    assert first == detector_occurrence_seed(
        base_seed=42,
        epoch_number=3,
        sample_position=1200,
        item_index=17,
    )
    assert first != detector_occurrence_seed(
        base_seed=42,
        epoch_number=3,
        sample_position=1201,
        item_index=17,
    )
    assert 0 <= first < 2**63
    with pytest.raises(ValueError, match="invalid"):
        detector_occurrence_seed(
            base_seed=42,
            epoch_number=0,
            sample_position=0,
            item_index=0,
        )


def test_legacy_sample_index_recovery_matches_dataloader_generator_order() -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import WeightedRandomSampler

    weights = [0.5, 1.0, 2.0, 4.0]
    generator = torch.Generator().manual_seed(20260729)
    epoch_state = generator.get_state()
    sampler = WeightedRandomSampler(
        weights,
        num_samples=20,
        replacement=True,
        generator=generator,
    )
    sampler_iterator = iter(sampler)
    # _BaseDataLoaderIter consumes the worker base seed after constructing the
    # lazy sampler iterator and before requesting its first index.
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    expected = list(sampler_iterator)

    assert legacy_detector_sampled_indices(
        weights,
        num_samples=20,
        epoch_loader_generator_state=epoch_state,
    ) == expected


def test_resumable_occurrences_are_the_exact_full_epoch_suffix() -> None:
    sampled = [4, 1, 9, 1, 7]
    full = remaining_detector_occurrences(
        sampled,
        base_seed=42,
        epoch_number=2,
        completed_steps=0,
        batch_size=2,
    )
    resumed = remaining_detector_occurrences(
        sampled,
        base_seed=42,
        epoch_number=2,
        completed_steps=2,
        batch_size=2,
    )

    assert resumed == full[4:]
    assert resumed[0][0] == 7
    assert remaining_detector_occurrences(
        sampled,
        base_seed=42,
        epoch_number=2,
        completed_steps=3,
        batch_size=2,
    ) == []
    with pytest.raises(ValueError, match="exceed"):
        remaining_detector_occurrences(
            sampled,
            base_seed=42,
            epoch_number=2,
            completed_steps=4,
            batch_size=2,
        )


@pytest.mark.parametrize("mode", ("L", "RGB", "P"))
def test_detector_crop_first_is_pixel_equivalent_to_legacy_conversion(
    tmp_path,
    mode: str,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    source = Image.new(mode, (1800, 1400), 255)
    if mode == "P":
        palette = []
        for value in range(256):
            palette.extend((value, value, value))
        source.putpalette(palette)
    drawing = ImageDraw.Draw(source)
    drawing.rectangle((450, 300, 1450, 1100), fill=32)
    drawing.arc((550, 450, 1350, 950), 10, 170, fill=128, width=7)
    path = tmp_path / f"page-{mode}.png"
    source.save(path)
    crop = (400, 250, 1424, 1274)

    with Image.open(path) as opened:
        legacy = ImageOps.grayscale(
            opened.convert("RGB").crop(crop)
        )
    optimized = load_grayscale_crop(path, crop)

    assert optimized.mode == "L"
    assert optimized.size == (1024, 1024)
    assert optimized.tobytes() == legacy.tobytes()


def test_gpu_queue_cannot_skip_registered_scan_degraded_holdout_gates() -> None:
    source_root = Path(__file__).resolve().parents[2]
    queue = (
        source_root
        / "training"
        / "queue_gpu_benchmark_and_detector_pipeline.ps1"
    ).read_text(encoding="utf-8")
    ordered_markers = (
        "run_wsl_openscore_semantic_detector_full.sh",
        "Overlap-consistent semantic dataset preparation failed",
        "run_wsl_openscore_overlap_recovery.sh",
        "skip_dedicated_lieder_synthetic_finetune",
        "Muse OMR v2 datasets failed post-preparation validation",
        "run_wsl_muse_scan_semantic_detector_full.sh",
        "run_wsl_muse_holdout_evaluation.sh",
        "if (-not (Test-Path -LiteralPath $candidateReport",
    )
    offsets = [queue.index(marker) for marker in ordered_markers]

    assert offsets == sorted(offsets)
    assert queue.index("$holdout.acceptance.passed -ne $true") < offsets[-1]
    assert "queue_semantic_detector_release_candidate.ps1" in queue
    assert "isolated candidate export remains blocked" in queue
    assert "never authorize canonical resources" in queue
    assert "PIGEON_SCORE_SCAN_GPU_PYTHON" in queue
    assert "test_space\\Pigeon-Score-Scan-0.37.0" not in queue


def test_high_value_gpu_queue_is_serial_and_stops_on_stage_failure() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "queue_high_value_gpu_pipeline.ps1"
    ).read_text(encoding="utf-8")
    ordered_stages = (
        "queue_gpu_benchmark_and_detector_pipeline.ps1",
        "queue_ppocrv6_scorescan_training.ps1",
        "queue_ppocrv6_scorescan_detection.ps1",
        "queue_ppocrv6_exhaustive_detection_training.ps1",
        "queue_zeus_family_priority_training.ps1",
    )
    offsets = [script.index(stage) for stage in ordered_stages]

    assert offsets == sorted(offsets)
    assert "Start-Process" not in script
    assert "$stageExitCode = $LASTEXITCODE" in script
    assert "if ($stageExitCode -ne 0)" in script
    assert "high-value-gpu-pipeline-current.json" in script
    assert "Move-Item -LiteralPath $temporary -Destination $statusPath -Force" in script


def test_symbol_training_entry_points_export_project_pythonpath() -> None:
    source_root = Path(__file__).resolve().parents[2]
    expected = 'PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}'

    for name in (
        "run_wsl_symbol_detector_full.sh",
        "run_wsl_openscore_semantic_detector_full.sh",
        "run_wsl_openscore_overlap_recovery.sh",
        "run_wsl_lieder_semantic_detector_full.sh",
        "run_wsl_muse_scan_semantic_detector_full.sh",
    ):
        script = (source_root / "training" / name).read_text(encoding="utf-8")
        assert expected in script, f"{name} cannot import project modules"


def test_semantic_candidate_is_authorized_only_in_isolated_resources() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "queue_semantic_detector_release_candidate.ps1"
    ).read_text(encoding="utf-8")

    assert "canonical_resources_authorized = $false" in script
    assert "physical_scan_release_evidence = $false" in script
    assert 'source_image_origin = "rendered_scan_degraded"' in script
    assert (
        'boundary_contract_version = "printed-western-instrumental-scan-boundary@4"'
        in script
    )
    assert 'Join-Path $candidateDir "evaluation-resources"' in script
    assert "authorize_semantic_detector_release" in script
    assert "audit_model_manifest" in script
    assert "load_semantic_detector_assets" in script
    assert "isolated_product_evaluation_resources_verified = $true" in script
    assert "Copy-Item" in script
    assert "app\\src\\scorescan\\resources" in script
    assert "PIGEON_SCORE_SCAN_GPU_PYTHON" in script
    assert "test_space\\Pigeon-Score-Scan-0.37.0" not in script


def test_muse_v2_preparation_keeps_strict_registration_and_class_evidence() -> None:
    source_root = Path(__file__).resolve().parents[2]
    preparation = (
        source_root
        / "training"
        / "prepare_muse_scan_datasets_v2.ps1"
    ).read_text(encoding="utf-8")
    scan_training = (
        source_root
        / "training"
        / "run_wsl_muse_scan_semantic_detector_full.sh"
    ).read_text(encoding="utf-8")
    holdout = (
        source_root
        / "training"
        / "run_wsl_muse_holdout_evaluation.sh"
    ).read_text(encoding="utf-8")

    assert "muse-omr-bounded-elastic-page-filter-jpeg95@7" in preparation
    assert (
        "complete-page-overlap-consistent-deduplicated-semantic-targets@4"
        in preparation
    )
    assert "audit_muse_semantic_tag_evidence" in preparation
    assert "-SelectedPairs 435" in preparation
    assert "-SelectedWorks 395" in preparation
    assert '--minimum-ecc", "0.86"' in preparation
    assert '--minimum-accepted-page-fraction", "0.75"' in preparation
    assert "$trainingMinimumAcceptedWorks = 170" in preparation
    assert "$holdoutMinimumAcceptedWorks = 200" in preparation
    assert (
        "[int]$report.accepted_works -lt $MinimumAcceptedWorks"
        in preparation
    )
    assert (
        "-MinimumAcceptedWorks $holdoutMinimumAcceptedWorks"
        in preparation
    )
    assert "if ($count -lt 25)" in preparation
    assert "audit_semantic_replay_holdout_isolation" in preparation
    assert "[int]$existing.replay_works -eq 1584" in preparation
    assert (
        "muse_omr_scan_regions_stratified_complete_page_"
        "overlap_consistent_deduplicated_v7"
        in scan_training
    )
    assert (
        "muse_omr_scan_regions_stratified_complete_page_v6"
        in scan_training
    )
    assert (
        "openscore_lieder_train_1091_svg_regions_complete_page_"
        "overlap_consistent_deduplicated_v4"
    ) in scan_training
    assert "--replay-fraction 0.35" in scan_training
    assert "--replay-max-train-tiles 60000" in scan_training
    assert "--require-complete-page-targets" in scan_training
    assert "--eval-every 2" in scan_training
    assert (
        "muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729"
        in scan_training
    )
    assert (
        "muse_omr_scan_holdout_regions_stratified_complete_page_"
        "overlap_consistent_deduplicated_v7"
        in holdout
    )
    assert (
        "muse_omr_scan_holdout_regions_stratified_complete_page_v6"
        in holdout
    )
    assert "--operating-point-calibration-prepared-dir" in holdout
    assert "--minimum-operating-point-calibration-true-positives 10" in holdout
    assert "SCORESCAN_CALIBRATION_SPLIT" in holdout
    assert '"${CALIBRATION_SPLIT_ARGS[@]}"' in holdout
    assert 'PYTHONPATH="${PROJECT_DIR}/app/src:${PROJECT_DIR}' in holdout


def test_final_refit_holdout_uses_only_unseen_calibration_and_one_report() -> None:
    source_root = Path(__file__).resolve().parents[2]
    final_refit = (
        source_root
        / "training"
        / "run_wsl_muse_semantic_final_refit_v1.sh"
    ).read_text(encoding="utf-8")
    script = (
        source_root
        / "training"
        / "run_wsl_muse_final_refit_holdout_evaluation_v1.sh"
    ).read_text(encoding="utf-8")

    assert "--epochs 2" in final_refit
    assert "--eval-every 2" in final_refit
    assert "--eval-every 1" not in final_refit
    assert 'report.get("completed_epochs") != 2' in script
    assert 'report.get("best_model_sha256") != sha256(model_path)' in script
    assert "if train & calibration:" in script
    assert "if train & holdout:" in script
    assert "if calibration & holdout:" in script
    assert 'SCORESCAN_CALIBRATION_SPLIT="calibration"' in script
    assert (
        "muse-training-calibration-only-semantic-page-layout-evidence-v2-"
        "page-shape-refreeze.json"
        in script
    )
    assert (
        "evaluation.independent-muse-holdout.page-shape-refreeze-v2.json"
        in script
    )
    assert "selection_rule_model_independent" in script
    assert "model_predictions_observed_for_refreeze" in script


def test_muse_v2_registered_scans_feed_both_ocr_training_queues() -> None:
    source_root = Path(__file__).resolve().parents[2]
    scripts = {
        name: (
            source_root / "training" / name
        ).read_text(encoding="utf-8")
        for name in (
            "queue_muse_scan_text_preparation.ps1",
            "queue_muse_holdout_text_preparation.ps1",
            "queue_ppocrv6_scorescan_training.ps1",
            "queue_ppocrv6_scorescan_detection.ps1",
        )
    }

    assert "muse_omr_scan_regions_stratified_v3" in scripts[
        "queue_muse_scan_text_preparation.ps1"
    ]
    assert "muse_omr_scan_text_stratified_v3" in scripts[
        "queue_muse_scan_text_preparation.ps1"
    ]
    assert "--reuse-pdf-dir $reusablePdfDir" in scripts[
        "queue_muse_scan_text_preparation.ps1"
    ]
    assert "muse_omr_scan_text_v2\\pdf" in scripts[
        "queue_muse_scan_text_preparation.ps1"
    ]
    assert "--reuse-pdf-dir $reusablePdfDir" in scripts[
        "queue_muse_holdout_text_preparation.ps1"
    ]
    assert "muse_omr_scan_holdout_text_v2\\pdf" in scripts[
        "queue_muse_holdout_text_preparation.ps1"
    ]
    for name in (
        "queue_ppocrv6_scorescan_training.ps1",
        "queue_ppocrv6_scorescan_detection.ps1",
    ):
        assert "muse_omr_scan_text_stratified_v3" in scripts[name]
        assert "muse_omr_scan_text_v1" not in scripts[name]


def test_paddle_configs_cannot_reuse_pre_stratification_labels_or_models() -> None:
    training = Path(__file__).resolve().parents[2] / "training"
    contents = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            training / "ppocrv6_scorescan_rec.yml",
            training / "ppocrv6_scorescan_det.yml",
            training / "run_wsl_ppocrv6_scorescan_training.sh",
            training / "run_wsl_ppocrv6_scorescan_detection.sh",
            training / "run_wsl_ppocrv6_scorescan_detection_exhaustive.sh",
        )
    }
    combined = "\n".join(contents.values())
    assert "scorescan_ocr_training_stratified_v2" in combined
    assert "scorescan_ocr_detection_stratified_v2" in combined
    assert "scorescan_ocr_detection_stratified_v4" in combined
    assert "ppocrv6-scorescan-rec-stratified" in combined
    assert "ppocrv6-scorescan-det-stratified" in combined
    for retired in (
        "scorescan_ocr_training_v1",
        "scorescan_ocr_detection_v1",
        "scorescan_ocr_detection_v3",
        "ppocrv6-scorescan-rec-e18-b8-20260728",
        "ppocrv6-scorescan-det-e36-b2-20260728",
    ):
        assert retired not in combined


def test_registered_accidental_preparation_is_training_only_and_bounded() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "prepare_registered_accidental_presence.ps1"
    ).read_text(encoding="utf-8")
    assert "external\\training\\muse_omr_scan_train_stratified_v2" in script
    assert "muse_omr_scan_regions_stratified_v3" in script
    assert "muse_omr_scan_holdout_regions_stratified_v3" not in script
    assert "--maximum-samples-per-pair 240" in script
    assert "holdout_used_for_training" in script


def test_registered_accidental_holdout_is_explicitly_forbidden_to_training() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "prepare_registered_accidental_holdout.ps1"
    ).read_text(encoding="utf-8")
    assert "external\\benchmarks\\muse_omr_e27f6a8634_raremarks_v3" in script
    assert "muse_omr_scan_holdout_regions_stratified_v3" in script
    assert "--independent-holdout" in script
    assert 'role -ne "independent_holdout_evaluation_only"' in script
    assert "training_use_authorized -ne $false" in script
    assert "[int]$report.works_by_split.test -lt 200" in script


def test_registered_accidental_model_freezes_thresholds_before_holdout() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "queue_registered_accidental_model.ps1"
    ).read_text(encoding="utf-8")
    assert "prepare_accidental_presence_programmatic" in script
    assert "--registered-data-dir $registeredDir" in script
    assert "evaluate_registered_accidental_presence_holdout" in script
    assert "--minimum-works 200" in script
    assert "--minimum-roc-auc 0.94" in script
    assert "--minimum-class-recall 0.30" in script
    assert "app\\src\\scorescan\\resources" not in script


def test_ocr_detection_early_stop_uses_calibration_and_never_test_labels() -> None:
    source_root = Path(__file__).resolve().parents[2]
    script = (
        source_root
        / "training"
        / "run_wsl_ppocrv6_scorescan_detection.sh"
    ).read_text(encoding="utf-8")
    loop_start = script.index("while (( ")
    calibration_block = script[
        loop_start:
        script.index('\ndone\n\nif [[ ! -s "${BEST_MODEL}" ]]', loop_start)
    ]
    assert 'EPOCHS_PER_INVOCATION=2' in script
    assert "SCORESCAN_MAX_EPOCHS_THIS_RUN" in calibration_block
    assert "calibration.scan.paddle.det.txt" in calibration_block
    assert "calibration.clean.paddle.det.txt" in calibration_block
    assert "check_paddleocr_detection_calibration" in calibration_block
    assert "--minimum 0.997" in calibration_block
    assert "test.scan.paddle.det.txt" not in calibration_block
    assert "test.clean.paddle.det.txt" not in calibration_block
    assert "--minimum-scan-precision 0.995" in script
    assert "--minimum-clean-recall 0.995" in script


def test_paddle_epoch_chunk_patch_is_opt_in_and_bounded() -> None:
    program = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "paddleocr-patches"
        / "tools"
        / "program.py"
    ).read_text(encoding="utf-8")
    assert 'os.environ.get("SCORESCAN_MAX_EPOCHS_THIS_RUN", "0")' in program
    assert "end_epoch = epoch_num" in program
    assert "min(" in program
    assert "for epoch in range(start_epoch, end_epoch + 1):" in program
    loader = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "paddleocr-patches"
        / "ppocr"
        / "utils"
        / "save_load.py"
    ).read_text(encoding="utf-8")
    assert 'if "global_step" in states_dict:' in loader
    assert (
        'best_model_dict["global_step"] = int(states_dict["global_step"])'
        in loader
    )
    assert 'best_model_dict["acc"] = 0.0' not in loader


def test_ocr_recognition_early_stop_uses_calibration_and_never_test_labels() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "run_wsl_ppocrv6_scorescan_training.sh"
    ).read_text(encoding="utf-8")
    loop_start = script.index("while (( ")
    calibration_block = script[
        loop_start:
        script.index('\ndone\n\nif [[ ! -s "${BEST_MODEL}" ]]', loop_start)
    ]
    assert "EPOCHS_PER_INVOCATION=2" in script
    assert "SCORESCAN_MAX_EPOCHS_THIS_RUN" in calibration_block
    assert "calibration.scan.paddle.txt" in calibration_block
    assert "calibration.clean.paddle.txt" in calibration_block
    assert "check_paddleocr_recognition_calibration" in calibration_block
    assert "--minimum-accuracy 0.999" in calibration_block
    assert "--minimum-normalized-edit 0.9997" in calibration_block
    assert "test.scan.paddle.txt" not in calibration_block
    assert "test.clean.paddle.txt" not in calibration_block
    assert "--minimum-scan-accuracy 0.998" in script
    assert "--minimum-clean-normalized-edit 0.9995" in script


def test_lieder_stage_uses_bounded_quartet_replay_and_combined_evaluation() -> None:
    source_root = Path(__file__).resolve().parents[2]
    script = (
        source_root
        / "training"
        / "run_wsl_lieder_semantic_detector_full.sh"
    ).read_text(encoding="utf-8")

    assert "openscore_lieder_train_1091_svg_regions_overlap_consistent_deduplicated_v3" in script
    assert "openscore_quartet_lieder_semantic_overlap_consistent_deduplicated_v3" in script
    assert "openscore_string_quartets_svg_regions_overlap_consistent_deduplicated_v4" in script
    assert "--evaluation-prepared-dir" in script
    assert "--replay-fraction 0.20" in script
    assert "--epochs 8" in script
    assert "--evaluation-batch-size 8" in script
    assert "run_wsl_openscore_overlap_recovery.sh" in script
    assert "openscore-semantic-detector-overlap-recovery-v3-e4-b2-20260729" in script
    assert 'LOADER_WORKERS="${SCORESCAN_LOADER_WORKERS:-2}"' in script
    assert '--workers "${LOADER_WORKERS}"' in script


def test_quartet_recovery_trains_only_on_overlap_consistent_targets() -> None:
    source_root = Path(__file__).resolve().parents[2]
    script = (
        source_root
        / "training"
        / "run_wsl_openscore_overlap_recovery.sh"
    ).read_text(encoding="utf-8")

    assert "openscore_string_quartets_svg_regions_overlap_consistent_deduplicated_v4" in script
    assert "openscore_string_quartets_svg_regions_normalized_v2" not in script
    assert "--initial-model" in script
    assert "--epochs 4" in script
    assert "--eval-every 2" in script
    assert "--runtime-stop-after-epoch 2" in script
    assert (
        "superseded_by_bounded_lieder_replay_and_registered_scan_finetuning"
        in script
    )
    assert "--evaluation-batch-size 8" in script
    assert '--workers "${LOADER_WORKERS}"' in script
    assert "prepared_manifest_sha256" in script


def test_gpu_queue_never_promotes_legacy_quartet_weights_to_release_evidence() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "queue_gpu_benchmark_and_detector_pipeline.ps1"
    ).read_text(encoding="utf-8")
    assert "audit_legacy_detector_initialization" in script
    assert "initialization-only-audit.json" in script
    assert "$initializationAudit.deployment_eligible -ne $false" in script
    assert "$initializationAudit.release_accuracy_evidence -ne $false" in script
    assert "$initializationAudit.model_sha256 -ne $quartetModelHash" in script


def test_detector_uses_crowded_notation_nms_threshold() -> None:
    source_root = Path(__file__).resolve().parents[2]
    trainer = (
        source_root / "app" / "tools" / "train_deepscores_symbol_detector.py"
    ).read_text(encoding="utf-8")

    assert "DETECTOR_NMS_IOU = 0.75" in trainer
    assert "DETECTOR_FOREGROUND_IOU_THRESHOLD = 0.35" in trainer
    assert "DETECTOR_BACKGROUND_IOU_THRESHOLD = 0.25" in trainer
    assert "DETECTOR_STRICT_FOREGROUND_IOU_THRESHOLD = 0.50" in trainer
    assert "DETECTOR_STRICT_BACKGROUND_IOU_THRESHOLD = 0.40" in trainer
    assert "nms_thresh=DETECTOR_NMS_IOU" in trainer
    assert "class CategoryAwareMatcher(Matcher):" in trainer
    assert "model.proposal_matcher = CategoryAwareMatcher(strict_labels)" in trainer
    assert 'targets_per_image["labels"]' in trainer
    assert "allow_low_quality_matches=True" in trainer
    assert "DETECTOR_MODEL_CONTRACT_VERSION" in trainer
    assert '"model_contract": detector_model_contract(' in trainer
    assert "compute_dense_detection_metrics(" in trainer
    assert "evaluation_detection_limit(model, args.detections_per_tile)" in trainer
    assert "evaluation_detection_limit(model, 100)" not in trainer
    assert (
        '"evaluation_max_detections_per_tile": args.detections_per_tile'
        in trainer
    )
    assert '"evaluation_max_detections_per_tile": 100' not in trainer
    assert "persistent_workers=False" in trainer


def test_detector_matcher_contract_protects_regressed_classes() -> None:
    release_contract = detector_training.detector_model_contract()
    contract = detector_training.detector_model_contract(
        class_aware_matcher=True,
    )

    assert "strict_match_classes" not in release_contract
    assert release_contract["version"].endswith("matcher35-25-nms75@3")
    assert contract["foreground_iou_threshold"] == 0.35
    assert contract["background_iou_threshold"] == 0.25
    assert contract["strict_foreground_iou_threshold"] == 0.5
    assert contract["strict_background_iou_threshold"] == 0.4
    strict = set(contract["strict_match_classes"])
    assert {
        "augmentationDot",
        "beam",
        "genericBarline",
        "graceSlash",
        "scoreText",
    } <= strict
    assert {"hairpin", "slur", "tie", "ottava"}.isdisjoint(strict)


def test_matcher_ablation_is_bounded_and_uses_complete_page_data() -> None:
    source_root = Path(__file__).resolve().parents[2]
    script = (
        source_root
        / "training"
        / "run_wsl_muse_scan_semantic_matcher_ablation.sh"
    ).read_text(encoding="utf-8")

    assert "complete_page_overlap_consistent_deduplicated_v7" in script
    assert "muse-omr-scan-semantic-detector-v4-complete-page" in script
    assert "--initial-model" in script
    assert 'RUNTIME_STOP_AFTER_EPOCH="${SCORESCAN_RUNTIME_STOP_AFTER_EPOCH:-2}"' in script
    assert '--runtime-stop-after-epoch "${RUNTIME_STOP_AFTER_EPOCH}"' in script
    assert "RUNTIME_STOP_AFTER_EPOCH < 2" in script
    assert "RUNTIME_STOP_AFTER_EPOCH > 12" in script
    assert (
        "matcher35-25-structural-ablation-before-full-run"
        in script
    )
    assert "--epochs 12" in script
    assert "--eval-every 2" in script
    assert "--require-complete-page-targets" in script


def test_class_aware_matcher_ablation_starts_from_v5_and_is_bounded() -> None:
    source_root = Path(__file__).resolve().parents[2]
    script = (
        source_root
        / "training"
        / "run_wsl_muse_scan_semantic_class_aware_matcher_ablation.sh"
    ).read_text(encoding="utf-8")

    assert "complete_page_overlap_consistent_deduplicated_v7" in script
    assert "v6-class-aware-matcher-e12-runtime2" in script
    assert "v5-matcher35-25-ablation-e12-runtime2" in script
    assert 'RUNTIME_STOP_AFTER_EPOCH="${SCORESCAN_RUNTIME_STOP_AFTER_EPOCH:-2}"' in script
    assert "--runtime-stop-after-epoch" in script
    assert "class-aware-matcher-regression-recovery-window" in script
    assert "--class-aware-matcher-ablation" in script
    assert "--epochs 12" in script
    assert "--eval-every 2" in script
    assert "--require-complete-page-targets" in script


def test_relation_detector_v2_uses_unseen_backbone_and_support_audited_split() -> None:
    source_root = Path(__file__).resolve().parents[2]
    script = (
        source_root
        / "training"
        / "run_wsl_relation_detector_v2_baseline.sh"
    ).read_text(encoding="utf-8")

    assert "muse_omr_relation_detector_subset_v2" in script
    assert "v5-matcher35-25-ablation" in script
    assert "v7-final-refit" not in script
    assert 'STOP_AFTER_EPOCH="${SCORESCAN_RELATION_STOP_AFTER_EPOCH:-1}"' in script
    assert "--minimum-required-class-test-objects 50" in script
    assert "--required-class-map hairpin=0.85" in script
    assert "--require-complete-page-targets" in script


def test_class_aware_sampling_boosts_rare_marks_but_keeps_negatives() -> None:
    rows = [
        {"objects": [{"label": 1}]},
        {"objects": [{"label": 1}]},
        {"objects": [{"label": 2}]},
        {"objects": []},
    ]
    weights, counts = class_aware_sample_weights(
        rows, power=0.5, maximum_repeat=3.0
    )
    assert counts == {1: 2, 2: 1}
    assert weights[2] > weights[1]
    assert weights[3] > 0
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert detector_class_counts(rows) == {1: 2, 2: 1}


def test_replay_mixture_has_exact_bounded_probability_mass() -> None:
    primary = [
        {"objects": [{"label": 1}]},
        {"objects": [{"label": 1}]},
        {"objects": []},
    ]
    replay = [
        {"objects": [{"label": 2}]},
        {"objects": [{"label": 3}]},
    ]

    weights, primary_counts, replay_counts = replay_mixture_sample_weights(
        primary,
        replay,
        replay_fraction=0.2,
        power=0.5,
        maximum_repeat=3.0,
        replay_maximum_repeat=6.0,
    )

    assert primary_counts == {1: 2}
    assert replay_counts == {2: 1, 3: 1}
    assert sum(weights[: len(primary)]) / sum(weights) == pytest.approx(0.8)
    assert sum(weights[len(primary) :]) / sum(weights) == pytest.approx(0.2)
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="supplied together"):
        replay_mixture_sample_weights(
            primary,
            [],
            replay_fraction=0.2,
            power=0.5,
            maximum_repeat=3.0,
            replay_maximum_repeat=6.0,
        )


def test_synthetic_replay_evidence_accepts_role_or_pinned_preparation() -> None:
    assert synthetic_training_evidence(
        {"role": "training_only_synthetic_semantic_geometry"},
        None,
    )
    assert synthetic_training_evidence(
        {"name": "scorescan-openscore-regions"},
        {"purpose": "synthetic semantic geometry; not real-scan validation"},
    )
    assert not synthetic_training_evidence(
        {"role": "training_only_disjoint_from_external_release_holdout"},
        {"purpose": "registered real scan"},
    )


def test_detector_normalizes_only_sufficiently_visible_legacy_boxes() -> None:
    rows = [
        {
            "crop_xyxy": [0, 0, 100, 80],
            "objects": [
                {
                    "box_xyxy": [-5, 10, 100, 20],
                    "category_id": "hairpin",
                    "label": 1,
                }
            ],
        }
    ]
    normalized, audit = normalize_target_boxes(
        rows,
        minimum_visible_fraction=0.8,
    )
    assert normalized[0]["objects"][0]["box_xyxy"] == [
        0.0,
        10.0,
        100.0,
        20.0,
    ]
    assert audit == {
        "objects": 1,
        "clipped_objects": 1,
        "clipped_by_category": {"hairpin": 1},
        "minimum_visible_fraction": 0.8,
    }
    with pytest.raises(ValueError, match="visible fraction"):
        normalize_target_boxes(
            [
                {
                    "crop_xyxy": [0, 0, 100, 80],
                    "objects": [
                        {
                            "box_xyxy": [-30, 10, 100, 20],
                            "category_id": "hairpin",
                            "label": 1,
                        }
                    ],
                }
            ],
            minimum_visible_fraction=0.8,
        )


def test_detector_normalization_reuses_already_valid_rows() -> None:
    rows = [
        {
            "crop_xyxy": [0, 0, 100, 80],
            "objects": [
                {
                    "box_xyxy": [10, 10, 20, 20],
                    "category_id": "tie",
                    "label": 1,
                }
            ],
        }
    ]
    normalized, audit = normalize_target_boxes(
        rows,
        minimum_visible_fraction=0.8,
    )

    assert normalized[0] is rows[0]
    assert normalized[0]["objects"] is rows[0]["objects"]
    assert audit["objects"] == 1
    assert audit["clipped_objects"] == 0


def test_detector_complete_page_geometry_audit_is_hash_reusable() -> None:
    rows = [
        {
            "source_key": "work",
            "image": "page.png",
            "crop_xyxy": [0, 0, 10, 10],
            "objects": [
                {
                    "box_xyxy": [8, 2, 10, 4],
                    "page_box_xyxy": [8, 2, 12, 4],
                    "category_id": "slur",
                    "label": 31,
                    "source_object_id": "s" * 24,
                    "target_geometry_provenance": (
                        detector_training.COMPLETE_PAGE_TARGET_PROVENANCE
                    ),
                }
            ],
        }
    ]
    normalized, audit = normalize_target_boxes(
        rows,
        minimum_visible_fraction=0.8,
        require_complete_page_geometry=True,
        long_span_minimum_visible_fraction=0.25,
    )
    assert normalized[0] is rows[0]
    assert audit["complete_page_geometry_required"] is True
    assert audit["unique_source_objects"] == 1
    binding = bind_zero_clip_target_box_audit(
        audit,
        jsonl_sha256="f" * 64,
        row_count=1,
    )
    assert reusable_zero_clip_target_box_audit(
        binding,
        jsonl_sha256="f" * 64,
        row_count=1,
        minimum_visible_fraction=0.8,
        require_complete_page_geometry=True,
        long_span_minimum_visible_fraction=0.25,
    ) == audit

    rows[0]["objects"][0]["box_xyxy"] = [7, 2, 10, 4]
    with pytest.raises(ValueError, match="contradicts complete-page"):
        normalize_target_boxes(
            rows,
            minimum_visible_fraction=0.8,
            require_complete_page_geometry=True,
            long_span_minimum_visible_fraction=0.25,
        )


def test_detector_zero_clip_audit_reuse_is_exactly_hash_bound() -> None:
    audit = {
        "objects": 123,
        "clipped_objects": 0,
        "clipped_by_category": {},
        "minimum_visible_fraction": 0.8,
    }
    binding = bind_zero_clip_target_box_audit(
        audit,
        jsonl_sha256="a" * 64,
        row_count=12,
    )
    assert reusable_zero_clip_target_box_audit(
        binding,
        jsonl_sha256="a" * 64,
        row_count=12,
        minimum_visible_fraction=0.8,
    ) == audit
    assert reusable_zero_clip_target_box_audit(
        binding,
        jsonl_sha256="b" * 64,
        row_count=12,
        minimum_visible_fraction=0.8,
    ) is None
    assert reusable_zero_clip_target_box_audit(
        binding,
        jsonl_sha256="a" * 64,
        row_count=13,
        minimum_visible_fraction=0.8,
    ) is None
    clipped = {**audit, "clipped_objects": 1}
    assert bind_zero_clip_target_box_audit(
        clipped,
        jsonl_sha256="a" * 64,
        row_count=12,
    ) is None


def test_required_class_support_is_checked_before_training() -> None:
    assert insufficient_required_class_support(
        {"hairpin": 0.8, "slur": 0.8, "unknown": 0.5},
        class_name_by_label={1: "hairpin", 2: "slur"},
        test_class_counts={1: 25, 2: 3},
        minimum_objects=25,
    ) == {"slur": 3, "unknown": 0}


def test_detector_selection_prioritizes_expression_marks() -> None:
    score, priority = priority_selection_score(
        overall_map=0.8,
        per_class_map={
            "restQuarter": 0.95,
            "slur": 0.4,
            "tie": 0.6,
            "dynamicCrescendoHairpin": 0.5,
        },
    )
    assert priority == pytest.approx(0.5)
    assert score == pytest.approx(0.575)


def test_detector_selection_ignores_unsupported_metric_sentinel() -> None:
    score, priority = priority_selection_score(
        overall_map=-1.0,
        per_class_map={"slur": -1.0, "tie": -1.0},
    )
    assert priority == 0.0
    assert score == 0.0


def test_detector_selection_excludes_statistically_tiny_priority_classes() -> None:
    score, priority = priority_selection_score(
        overall_map=0.8,
        per_class_map={
            "hairpin": 0.7,
            "slur": 0.5,
            "scoreText": 0.0,
            "techniqueText": 1.0,
        },
        class_support={
            "hairpin": 120,
            "slur": 80,
            "scoreText": 4,
            "techniqueText": 1,
        },
        minimum_support=25,
    )

    assert priority == pytest.approx(0.6)
    assert score == pytest.approx(0.6)


def test_detector_selection_excludes_tiny_classes_from_overall_component() -> None:
    per_class_map = {
        "hairpin": 0.7,
        "slur": 0.5,
        "genericRest": 0.9,
        "genericOrnament": 1.0,
    }
    support = {
        "hairpin": 120,
        "slur": 80,
        "genericRest": 1000,
        "genericOrnament": 6,
    }

    score, priority = priority_selection_score(
        overall_map=0.775,
        per_class_map=per_class_map,
        class_support=support,
        minimum_support=25,
    )
    filtered_map, supported_classes = support_filtered_macro_map(
        per_class_map=per_class_map,
        class_support=support,
        minimum_support=25,
    )

    assert priority == pytest.approx(0.6)
    assert filtered_map == pytest.approx(0.7)
    assert score == pytest.approx(0.625)
    assert supported_classes == ["genericRest", "hairpin", "slur"]


def test_detector_selection_evidence_fails_closed_on_tampering() -> None:
    metrics = {
        "map": 0.7,
        "map_per_class_named": {"genericRest": 0.8, "slur": 0.6},
        "selection_score": 0.625,
        "selection_support_filtered_map": 0.7,
        "priority_mark_map": 0.6,
        "selection_minimum_class_support": 25,
        "priority_mark_minimum_class_support": 25,
        "selection_supported_classes": ["genericRest", "slur"],
        "priority_mark_supported_classes": ["slur"],
    }
    support = {"genericRest": 100, "slur": 80}

    assert not detector_selection_evidence_failures(
        metrics,
        class_support=support,
        minimum_support=25,
    )
    metrics["selection_score"] = 0.9
    assert detector_selection_evidence_failures(
        metrics,
        class_support=support,
        minimum_support=25,
    ) == ["selection_score"]


def test_gate_passing_epoch_outranks_higher_aggregate_failure() -> None:
    assert should_replace_detector_best(
        current_gate_passed=True,
        current_selection_score=0.80,
        best_gate_passed=False,
        best_selection_score=0.90,
    )
    assert not should_replace_detector_best(
        current_gate_passed=False,
        current_selection_score=0.95,
        best_gate_passed=True,
        best_selection_score=0.80,
    )


def test_detector_acceptance_checks_global_and_named_floors() -> None:
    metrics = {
        "map_50": 0.96,
        "map_75": 0.91,
        "priority_mark_map": 0.86,
        "map_per_class_named": {"hairpin": 0.84, "tie": 0.90},
    }
    failures = detector_acceptance_failures(
        metrics,
        minimum_map_50=0.95,
        minimum_map_75=0.90,
        minimum_priority_map=0.85,
        required_class_maps={"hairpin": 0.85, "tie": 0.85},
    )

    assert failures == ["class:hairpin=0.840000<0.850000"]


def test_detector_early_stop_requires_a_non_vacuous_accuracy_gate() -> None:
    assert not detector_acceptance_gates_configured(
        minimum_map_50=None,
        minimum_map_75=None,
        minimum_priority_map=None,
        required_class_maps={},
    )
    assert detector_acceptance_gates_configured(
        minimum_map_50=0.95,
        minimum_map_75=None,
        minimum_priority_map=None,
        required_class_maps={},
    )
    assert detector_acceptance_gates_configured(
        minimum_map_50=None,
        minimum_map_75=None,
        minimum_priority_map=None,
        required_class_maps={"tie": 0.85},
    )


def test_detector_main_rejects_vacuous_early_stop_before_training(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires at least one explicit accuracy gate",
    ):
        train_detector_main(
            [
                "--prepared-dir",
                str(tmp_path / "missing-prepared"),
                "--images-dir",
                str(tmp_path / "missing-images"),
                "--output-dir",
                str(tmp_path / "output"),
                "--stop-when-accepted",
            ]
        )


def test_priority_classes_include_semantic_geometry_and_text_regions() -> None:
    assert is_priority_mark_class("genericArticulation")
    assert is_priority_mark_class("genericDynamic")
    assert is_priority_mark_class("hairpin")
    assert is_priority_mark_class("tempoText")
    assert is_priority_mark_class("tie")
    assert is_priority_mark_class("tremoloSingle")
    assert is_priority_mark_class("rehearsalMarkText")
    assert is_priority_mark_class("pedal")
    assert is_priority_mark_class("glissando")
    assert not is_priority_mark_class("genericBarline")


def test_detector_resume_rejects_changed_data_or_configuration() -> None:
    config = {
        "format": 1,
        "prepared_manifest_sha256": "a" * 64,
        "train_tiles": 100,
    }
    assert_matching_run_config(config, dict(config))
    changed = dict(config)
    changed["train_tiles"] = 101
    with pytest.raises(ValueError, match="train_tiles"):
        assert_matching_run_config(config, changed)


def test_detector_resume_removes_only_a_completely_empty_legacy_directory(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty-output"
    empty.mkdir()

    assert remove_empty_detector_resume_directory(empty, resume=True)
    assert not empty.exists()

    occupied = tmp_path / "occupied-output"
    occupied.mkdir()
    marker = occupied / "unexpected.partial"
    marker.write_text("preserve and refuse", encoding="utf-8")
    assert not remove_empty_detector_resume_directory(occupied, resume=True)
    assert marker.read_text(encoding="utf-8") == "preserve and refuse"

    non_resume = tmp_path / "new-run-output"
    non_resume.mkdir()
    assert not remove_empty_detector_resume_directory(non_resume, resume=False)
    assert non_resume.is_dir()


def test_detector_resume_reopens_only_a_later_runtime_decision_window(
    tmp_path: Path,
) -> None:
    output = tmp_path / "detector"
    output.mkdir()
    report = {
        "format": 1,
        "planned_epochs": 12,
        "completed_epochs": 2,
        "runtime_truncated": True,
        "runtime_stop_after_epoch": 2,
    }
    report_path = output / "training_report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text('{"epochs": []}', encoding="utf-8")
    (output / "checkpoint.last.pt").write_bytes(b"checkpoint")
    (output / "run_config.json").write_text("{}", encoding="utf-8")

    reopened = reopen_runtime_truncated_detector_run(
        output,
        resume=True,
        planned_epochs=12,
        runtime_stop_after_epoch=4,
    )

    assert reopened is not None
    assert reopened["completed_epochs"] == 2
    assert reopened["requested_final_epoch"] == 4
    assert not report_path.exists()
    assert not (output / "metrics.json").exists()
    assert (output / "metrics.partial.json").is_file()
    snapshot = output / "training_report.runtime-stop-2.json"
    assert json.loads(snapshot.read_text(encoding="utf-8")) == report


def test_detector_resume_refuses_same_or_changed_completed_ablation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "detector"
    output.mkdir()
    (output / "training_report.json").write_text(
        json.dumps(
            {
                "format": 1,
                "planned_epochs": 12,
                "completed_epochs": 2,
                "runtime_truncated": True,
                "runtime_stop_after_epoch": 2,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="cannot be continued"):
        reopen_runtime_truncated_detector_run(
            output,
            resume=True,
            planned_epochs=12,
            runtime_stop_after_epoch=2,
        )
    with pytest.raises(RuntimeError, match="cannot be continued"):
        reopen_runtime_truncated_detector_run(
            output,
            resume=True,
            planned_epochs=10,
            runtime_stop_after_epoch=4,
        )
    assert (
        reopen_runtime_truncated_detector_run(
            output,
            resume=False,
            planned_epochs=12,
            runtime_stop_after_epoch=4,
        )
        is None
    )
    assert (output / "training_report.json").is_file()


def test_detector_resume_worker_change_is_explicit_and_hash_recorded() -> None:
    expected = {
        "format": 1,
        "arguments": {"workers": 4, "batch_size": 2},
        "prepared_manifest_sha256": "a" * 64,
    }
    actual = {
        "format": 1,
        "arguments": {"workers": 2, "batch_size": 2},
        "prepared_manifest_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="arguments"):
        reconcile_resume_run_config(
            expected,
            actual,
            allow_worker_change=False,
            checkpoint_sha256="b" * 64,
        )

    reconciled = reconcile_resume_run_config(
        expected,
        actual,
        allow_worker_change=True,
        checkpoint_sha256="b" * 64,
    )

    assert reconciled["arguments"]["workers"] == 2
    assert reconciled["runtime_worker_transitions"] == [
        {
            "field": "workers",
            "from": 4,
            "to": 2,
            "resume_checkpoint_sha256": "b" * 64,
            "augmentation_continuity": "legacy_worker_rng_stream_changed",
        }
    ]
    # Repeating a same-worker resume preserves, but does not duplicate, history.
    assert reconcile_resume_run_config(
        reconciled,
        actual,
        allow_worker_change=True,
        checkpoint_sha256="c" * 64,
    ) == reconciled


def test_detector_resume_worker_exception_cannot_hide_other_changes() -> None:
    expected = {
        "format": 1,
        "arguments": {"workers": 4, "batch_size": 2},
        "prepared_manifest_sha256": "a" * 64,
    }
    actual = {
        "format": 1,
        "arguments": {"workers": 2, "batch_size": 4},
        "prepared_manifest_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="arguments"):
        reconcile_resume_run_config(
            expected,
            actual,
            allow_worker_change=True,
            checkpoint_sha256="b" * 64,
        )


def test_detector_resume_preserves_runtime_execution_provenance() -> None:
    actual = {
        "format": 1,
        "arguments": {"workers": 2, "batch_size": 2},
        "prepared_manifest_sha256": "a" * 64,
    }
    stored = {
        **actual,
        "runtime_execution_records": [
            {
                "resume_checkpoint_sha256": "b" * 64,
                "logical_batch_size": 2,
                "microbatch_size": 1,
                "numerical_continuity": (
                    "mean_loss_gradient_equivalent_in_real_arithmetic;"
                    "cuda_fp16_not_bit_identical"
                ),
            }
        ],
    }

    assert reconcile_resume_run_config(
        stored,
        actual,
        allow_worker_change=False,
        checkpoint_sha256="c" * 64,
    ) == stored


def test_detector_checkpoint_position_accepts_epoch_and_optimizer_boundary() -> None:
    assert detector_checkpoint_position(
        {"epoch": 3},
        total_epochs=6,
        accumulate=2,
    ) == (3, 0)
    assert detector_checkpoint_position(
        {
            "epoch": 3,
            "in_progress_epoch": 4,
            "in_progress_step": 2000,
        },
        total_epochs=6,
        accumulate=2,
    ) == (3, 2000)
    with pytest.raises(ValueError, match="optimizer boundary"):
        detector_checkpoint_position(
            {
                "epoch": 3,
                "in_progress_epoch": 4,
                "in_progress_step": 2001,
            },
            total_epochs=6,
            accumulate=2,
        )
    with pytest.raises(ValueError, match="does not follow"):
        detector_checkpoint_position(
            {
                "epoch": 3,
                "in_progress_epoch": 5,
                "in_progress_step": 2000,
            },
            total_epochs=6,
            accumulate=2,
        )


def test_category_compatibility_requires_exact_label_semantics() -> None:
    target = {
        "classes": [
            {"label": 1, "name": "hairpin"},
            {"label": 2, "name": "slur"},
        ]
    }
    compatible = {
        "classes": [
            {"label": 2, "name": "slur", "source": "other corpus"},
            {"label": 1, "name": "hairpin", "source": "other corpus"},
        ]
    }
    assert category_label_name_map(target) == {1: "hairpin", 2: "slur"}
    assert_compatible_category_manifests(target, compatible)

    incompatible = {
        "classes": [
            {"label": 1, "name": "slur"},
            {"label": 2, "name": "hairpin"},
        ]
    }
    with pytest.raises(ValueError, match="do not match"):
        assert_compatible_category_manifests(target, incompatible)


def test_category_manifest_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        category_label_name_map(
            {
                "classes": [
                    {"label": 1, "name": "slur"},
                    {"label": 1, "name": "tie"},
                ]
            }
        )


def test_detector_dataset_manifest_rejects_holdout_and_leakage() -> None:
    valid = {
        "role": "training_only",
        "source_split_overlap": 0,
        "reserved_holdout_overlap": 0,
        "train": {"tiles": 100},
        "test": {"tiles": 20},
    }
    assert_training_dataset_manifest(valid)

    registered_scan = dict(valid)
    registered_scan["role"] = (
        "training_only_disjoint_from_external_release_holdout"
    )
    assert_training_dataset_manifest(registered_scan)

    legacy_deepscores = dict(valid)
    legacy_deepscores["role"] = ""
    legacy_deepscores["name"] = "scorescan-deepscores-v2-expression-tiles-v3"
    assert_training_dataset_manifest(legacy_deepscores)

    holdout = dict(valid)
    holdout["role"] = "external_development_benchmark_not_training"
    with pytest.raises(ValueError, match="non-training"):
        assert_training_dataset_manifest(holdout)

    deceptive_holdout = dict(valid)
    deceptive_holdout["role"] = "training_only_mirrors_release_holdout"
    with pytest.raises(ValueError, match="non-training"):
        assert_training_dataset_manifest(deceptive_holdout)

    unbound_roleless = dict(valid)
    unbound_roleless["role"] = ""
    unbound_roleless["name"] = "unbound-roleless-data"
    with pytest.raises(ValueError, match="non-training"):
        assert_training_dataset_manifest(unbound_roleless)

    leaked = dict(valid)
    leaked["source_split_overlap"] = 1
    with pytest.raises(ValueError, match="nonzero"):
        assert_training_dataset_manifest(leaked)


def test_relation_detector_manifest_requires_unique_test_support() -> None:
    quality = {
        name: {"sources": 8, "unique_objects": 100}
        for name in ("slur", "tie", "hairpin")
    }
    valid = {
        "role": "training_only_disjoint_from_external_release_holdout",
        "source_split_overlap": 0,
        "reserved_holdout_overlap": 0,
        "class_subset_contract": "relation-detector-class-subset@2",
        "class_subset": ["slur", "tie", "hairpin"],
        "minimum_test_sources_per_class": 5,
        "minimum_test_unique_objects_per_class": 50,
        "train": {"tiles": 100},
        "test": {"tiles": 20, "class_quality": quality},
    }
    assert_training_dataset_manifest(valid)

    unsupported = json.loads(json.dumps(valid))
    unsupported["test"]["class_quality"]["hairpin"] = {
        "sources": 1,
        "unique_objects": 32,
    }
    with pytest.raises(
        ValueError,
        match=r"hairpin:sources=1<5.*hairpin:unique_objects=32<50",
    ):
        assert_training_dataset_manifest(unsupported)


def test_evaluation_detection_limit_restores_deployment_cap() -> None:
    class Detector:
        detections_per_img = 300

    detector = Detector()
    with evaluation_detection_limit(detector, 100) as selected:
        assert selected == 100
        assert detector.detections_per_img == 100
    assert detector.detections_per_img == 300

    with pytest.raises(RuntimeError):
        with evaluation_detection_limit(detector, 50):
            raise RuntimeError("synthetic evaluation failure")
    assert detector.detections_per_img == 300


def test_required_class_map_parser_is_strict() -> None:
    assert parse_required_class_maps(["slur=0.75", "hairpin=0.8"]) == {
        "slur": 0.75,
        "hairpin": 0.8,
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_required_class_maps(["slur=0.7", "slur=0.8"])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_required_class_maps(["tie=1.1"])


def test_stratified_cpu_queue_uses_atomic_child_completion_records() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "queue_stratified_cpu_preparation.ps1"
    ).read_text(encoding="utf-8")
    runner = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "run_preparation_child.ps1"
    ).read_text(encoding="utf-8")

    assert "$item.Process.WaitForExit()" in script
    assert "$item.CompletionPath" in script
    assert "[int]$completion.exit_code -ne 0" in script
    assert "ValueFromRemainingArguments = $true" in runner
    assert "Move-Item" in runner


def test_preparation_child_binds_named_target_parameters(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    runner = source_root / "training" / "run_preparation_child.ps1"
    target = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "preparation_child_named_parameter_probe.ps1"
    )
    completion = tmp_path / "completion.json"

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner),
            "-TargetScript",
            str(target),
            "-CompletionPath",
            str(completion),
            "-ProbeValue",
            "42",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completion.read_text(encoding="utf-8-sig"))
    assert payload["exit_code"] == 0
    assert payload["target_arguments"] == ["-ProbeValue", "42"]
