from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
TOOLS = ROOT / "app" / "tools"


def test_failed_component_gate_does_not_cancel_independent_ocr_training() -> None:
    queue = (TRAINING / "queue_ppocrv6_scorescan_training.ps1").read_text(
        encoding="utf-8"
    )
    runner = (TRAINING / "run_wsl_ppocrv6_scorescan_training.sh").read_text(
        encoding="utf-8"
    )
    assert "independent_ocr_training_continues = $true" in queue
    assert "Semantic detector is not release-eligible" in queue
    assert "$semanticCandidateReport.physical_scan_release_evidence -eq $false" in queue
    assert '"rendered_scan_degraded"' in queue
    assert 'throw "Release-gated semantic detector candidate is missing"' not in queue
    assert (
        "exporting the non-promotable candidate for independent runtime evaluation"
        in runner
    )
    assert 'if "${PYTHON_BIN}" -m app.tools.gate_paddleocr_evaluation' in runner


def test_family_priority_queue_keeps_product_deployment_blocked() -> None:
    queue = (TRAINING / "queue_zeus_family_priority_training.ps1").read_text(
        encoding="utf-8"
    )
    assert "family-priority-work-balanced-v1" in queue
    assert "--precision mixed_float16" in queue
    assert "--minimum-tie-improvement 1.0" in queue
    assert "--minimum-slur-improvement 1.0" in queue
    assert "desktop_deployment_authorized = $false" in queue
    assert "final_product_release_evidence = $false" in queue


def test_future_semantic_runs_use_fast_deterministic_resume() -> None:
    for filename in (
        "run_wsl_lieder_semantic_detector_full.sh",
        "run_wsl_muse_scan_semantic_detector_full.sh",
    ):
        runner = (TRAINING / filename).read_text(encoding="utf-8")
        assert "--resumable-augmentation-v3" in runner
        assert "--image-cache-dir" in runner
        assert "--populate-image-cache" in runner
        assert "--microbatch-size 1" in runner
        assert "--adaptive-full-batch-object-limit" in runner
        assert "SCORESCAN_ADAPTIVE_FULL_BATCH_OBJECT_LIMIT:-80" in runner
        assert "--adaptive-full-batch-min-free-mib" in runner
        assert "SCORESCAN_ADAPTIVE_FULL_BATCH_MIN_FREE_MIB:-4608" in runner
        assert "PYTORCH_CUDA_ALLOC_CONF" in runner
        assert "expandable_segments:True" in runner
    lieder = (
        TRAINING / "run_wsl_lieder_semantic_detector_full.sh"
    ).read_text(encoding="utf-8")
    assert "--stop-when-accepted" in lieder
    assert "--minimum-best-map-50 0.95" in lieder
    assert "--minimum-best-map-75 0.90" in lieder
    assert "--minimum-best-priority-map 0.85" in lieder
    legacy_recovery = (
        TRAINING / "run_wsl_openscore_overlap_recovery.sh"
    ).read_text(encoding="utf-8")
    assert "--resumable-augmentation-v3" not in legacy_recovery
    assert "--allow-resume-worker-change" in legacy_recovery
    assert "--image-cache-dir" in legacy_recovery
    assert "--populate-image-cache" in legacy_recovery
    assert "${HOME}/.cache/scorescan/detector-images/" in legacy_recovery
    assert "--microbatch-size 1" in legacy_recovery
    assert "--adaptive-full-batch-object-limit" in legacy_recovery
    assert "SCORESCAN_ADAPTIVE_FULL_BATCH_OBJECT_LIMIT:-80" in legacy_recovery
    assert "--adaptive-full-batch-min-free-mib" in legacy_recovery
    assert "SCORESCAN_ADAPTIVE_FULL_BATCH_MIN_FREE_MIB:-4608" in legacy_recovery
    assert "PYTORCH_CUDA_ALLOC_CONF" in legacy_recovery
    assert "expandable_segments:True" in legacy_recovery


def test_complete_page_semantic_queue_reserves_project_volume_capacity() -> None:
    queue = (
        TRAINING / "queue_complete_page_semantic_release.ps1"
    ).read_text(encoding="utf-8")
    assert "assert_training_storage.ps1" in queue
    assert '-Stage "complete-page-semantic-detector-v4"' in queue
    assert "-MinimumReserveGiB 4.0" in queue
    assert "-RequiredNewArtifactGiB 1.5" in queue


def test_superseded_semantic_watcher_stops_only_after_epoch8_evidence() -> None:
    watcher = (
        TRAINING / "stop_superseded_semantic_after_epoch8.ps1"
    ).read_text(encoding="utf-8")
    assert "[int]$record.epoch -eq 8" in watcher
    assert "model.best.pt" in watcher
    assert "checkpoint.last.pt" in watcher
    assert "train_deepscores_symbol_detector.py" in watcher
    assert "muse-omr-scan-semantic-detector-v3-stratified-replay" in watcher
    assert "kill -TERM" in watcher
    assert "scorescan-superseded-semantic-epoch8-stop-v1" in watcher


def test_exhaustive_detection_skips_only_strictly_accepted_low_yield_epochs() -> None:
    runner = (
        TRAINING / "run_wsl_ppocrv6_scorescan_detection_exhaustive.sh"
    ).read_text(encoding="utf-8")
    assert "TOTAL_EPOCHS=24" in runner
    assert "EPOCHS_PER_INVOCATION=2" in runner
    assert "MINIMUM_EARLY_STOP_EPOCH=4" in runner
    assert "--minimum 0.997" in runner
    assert "after_epoch >= MINIMUM_EARLY_STOP_EPOCH" in runner
    assert "skipping low-yield remaining epochs" in runner
    assert "full 24-epoch schedule continues" not in runner


def test_release_readiness_requires_physical_frozen_boundary_evidence() -> None:
    readiness = (TOOLS / "check_release_readiness.py").read_text(
        encoding="utf-8"
    )
    evaluator = (TOOLS / "evaluate_release_dataset.py").read_text(
        encoding="utf-8"
    )
    assert 'real_production_evidence.get("passed") is True' in readiness
    assert '"printed-western-instrumental-scan-boundary@4"' in readiness
    assert '== "physical_scan"' in readiness
    assert 'real_scope.get("page_count", 0) or 0) >= 2_000' in readiness
    assert "lyric_topology_interval" not in readiness
    assert '"PRODUCTION_EVIDENCE_CONTRACT_zh-CN.md"' in readiness
    assert '"physical_scan_origin_evidence": 1' in evaluator
    assert '"required_evidence_file_role_count"' in evaluator
