param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,
    [string[]]$WaitForProcessName = @()
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $projectDir)
$gpuPython = Join-Path $workspaceRoot "test_space\Pigeon-Score-Scan-0.37.0\runtime\venv-gpu\Scripts\python.exe"
$mediumModel = Join-Path $projectDir "training_data\external\models\rapidocr-ppocrv6-medium\PP-OCRv6_rec_medium.onnx"
$textDataset = Join-Path $projectDir "training_data\prepared\openscore_pdf_text_smoke_v1"
$textImages = Join-Path $projectDir "training_data\prepared\openscore_svg_regions_smoke_v2"
$gpuBenchmark = Join-Path $projectDir "training_data\benchmarks\openscore-text-recognition-smoke-v1-gpu.json"
$quartetManifest = Join-Path $projectDir "training_data\prepared\openscore_string_quartets_svg_regions_normalized_v2\manifest.json"
$liederManifest = Join-Path $projectDir "training_data\prepared\openscore_quartet_lieder_semantic_v1\manifest.json"
$overlapPreparationScript = Join-Path $PSScriptRoot "prepare_overlap_consistent_semantic_datasets.ps1"
$deepScoresModelDir = Join-Path $projectDir "training_data\models\deepscores-symbol-detector-e8-b2-20260727"
$deepScoresModel = Join-Path $deepScoresModelDir "model.best.pt"
$deepScoresReport = Join-Path $deepScoresModelDir "training_report.json"
$quartetModelDir = Join-Path $projectDir "training_data\models\openscore-semantic-detector-e6-b2-20260728"
$quartetModel = Join-Path $quartetModelDir "model.best.pt"
$quartetReport = Join-Path $quartetModelDir "training_report.json"
$quartetInitializationAudit = Join-Path $quartetModelDir "initialization-only-audit.json"
$quartetRecoveryModelDir = Join-Path $projectDir "training_data\models\openscore-semantic-detector-overlap-recovery-v3-e4-b2-20260729"
$quartetRecoveryModel = Join-Path $quartetRecoveryModelDir "model.best.pt"
$quartetRecoveryReport = Join-Path $quartetRecoveryModelDir "training_report.json"
$liederReplayDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_svg_regions_complete_page_overlap_consistent_deduplicated_v4"
$combinedSemanticEvaluationDir = Join-Path $projectDir "training_data\prepared\openscore_quartet_lieder_semantic_overlap_consistent_deduplicated_v3"
$semanticEfficiencyDecision = Join-Path $projectDir "training_data\benchmarks\semantic-stage-efficiency-decision-v1.json"
$musePreparationScript = Join-Path $PSScriptRoot "prepare_muse_scan_datasets_v2.ps1"
$musePreparationOut = Join-Path $projectDir "training_data\logs\muse-scan-stratified-v3-preparation.out.log"
$musePreparationErr = Join-Path $projectDir "training_data\logs\muse-scan-stratified-v3-preparation.err.log"
$museTrainingPreparedReport = Join-Path $projectDir "training_data\prepared\muse_omr_scan_regions_stratified_complete_page_v6\prepare-report.json"
$museHoldoutPreparedReport = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_regions_stratified_complete_page_v6\prepare-report.json"
$museModelDir = Join-Path $projectDir "training_data\models\muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730"
$museModel = Join-Path $museModelDir "model.best.pt"
$museReport = Join-Path $museModelDir "training_report.json"
$holdoutReport = Join-Path $museModelDir "evaluation.independent-muse-holdout.json"
$candidateReport = Join-Path $projectDir "training_data\release_candidates\semantic-detector-muse-v4-complete-page-e12-20260730\release-candidate.json"
$releaseCandidateQueue = Join-Path $PSScriptRoot "queue_semantic_detector_release_candidate.ps1"
$auditPython = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$diagnosticQuarantineReport = Join-Path $projectDir (
    "training_data\benchmarks\" +
    "active_semantic_dataset_diagnostic_quarantine_v1.json"
)
$storageGuard = Join-Path $PSScriptRoot "assert_training_storage.ps1"
$cachePruner = Join-Path `
    $PSScriptRoot `
    "prune_regenerable_workspace_cache.ps1"

function Wait-CompetingGpuProcess {
    foreach ($processName in $WaitForProcessName) {
        if ([string]::IsNullOrWhiteSpace($processName)) {
            continue
        }
        $reported = $false
        while ($null -ne (Get-Process -Name $processName -ErrorAction SilentlyContinue)) {
            if (-not $reported) {
                Write-Output (
                    @{
                        state = "waiting_for_competing_gpu_process"
                        process_name = $processName
                    } | ConvertTo-Json -Compress
                )
                $reported = $true
            }
            Start-Sleep -Seconds 15
        }
        if ($reported) {
            Write-Output (
                @{
                    state = "competing_gpu_process_exited"
                    process_name = $processName
                } | ConvertTo-Json -Compress
            )
        }
    }
}

$waitProcess = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
if ($null -ne $waitProcess) {
    Wait-Process -Id $WaitPid
}

& $cachePruner `
    -Execute `
    -RetryGenerationsToKeep 2 `
    -MinimumRetryLogAgeMinutes 30
Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = (
    (Join-Path $projectDir "app\src"),
    $projectDir
) -join [IO.Path]::PathSeparator

& $auditPython -m app.tools.audit_active_semantic_dataset_quarantine `
    --output $diagnosticQuarantineReport
if ($LASTEXITCODE -ne 0) {
    throw "Active semantic dataset diagnostic quarantine audit failed"
}
$diagnosticQuarantine = Get-Content `
    -LiteralPath $diagnosticQuarantineReport `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
if (
    $diagnosticQuarantine.passed -ne $true -or
    [int]$diagnosticQuarantine.training_holdout_source_key_overlap_count -ne 0 -or
    [int]$diagnosticQuarantine.replay_holdout_source_key_overlap_count -ne 0 -or
    [int]$diagnosticQuarantine.diagnostic_text_occurrence_count -ne 0 -or
    [int]$diagnosticQuarantine.diagnostic_hash_occurrence_count -ne 0 -or
    [int]$diagnosticQuarantine.training_authorized_diagnostic_page_count -ne 0
) {
    throw "Diagnostic or holdout data leaked into active semantic training"
}

Wait-CompetingGpuProcess

$musePreparationProcess = $null
if (
    -not (Test-Path -LiteralPath $museTrainingPreparedReport -PathType Leaf) -or
    -not (Test-Path -LiteralPath $museHoldoutPreparedReport -PathType Leaf)
) {
    $musePreparationProcess = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "`"$musePreparationScript`""
        ) `
        -WorkingDirectory $projectDir `
        -RedirectStandardOutput $musePreparationOut `
        -RedirectStandardError $musePreparationErr `
        -WindowStyle Hidden `
        -PassThru
    Write-Output (
        @{
            state = "muse_scan_v2_preparation_started"
            pid = $musePreparationProcess.Id
        } | ConvertTo-Json -Compress
    )
} else {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        $musePreparationScript
    if ($LASTEXITCODE -ne 0) {
        throw "Existing Muse OMR v2 datasets failed validation"
    }
}

# This benchmark is diagnostic and must not block independent detector
# training if a local CUDA DLL/provider problem is encountered. A completed
# immutable report also makes the queue safe to restart.
$gpuBenchmarkCompleted = $false
if (Test-Path -LiteralPath $gpuBenchmark -PathType Leaf) {
    try {
        $benchmarkPayload = Get-Content -LiteralPath $gpuBenchmark -Raw -Encoding UTF8 | ConvertFrom-Json
        $gpuBenchmarkCompleted = (
            $benchmarkPayload.schema_version -eq 1 -and
            $benchmarkPayload.base_samples -ge 600 -and
            $benchmarkPayload.runtime.use_cuda -eq $true -and
            $null -ne $benchmarkPayload.models.ppocrv6_medium
        )
    } catch {
        throw "Existing GPU OCR benchmark is not valid JSON: $gpuBenchmark"
    }
    if (-not $gpuBenchmarkCompleted) {
        throw "Existing GPU OCR benchmark is incomplete: $gpuBenchmark"
    }
}
if (-not $gpuBenchmarkCompleted) {
    & $gpuPython -m app.tools.benchmark_openscore_text_recognition `
        --dataset-dir $textDataset `
        --images-dir $textImages `
        --medium-model $mediumModel `
        --output $gpuBenchmark `
        --split calibration `
        --max-samples 600 `
        --batch-size 24 `
        --use-cuda
    $benchmarkExit = $LASTEXITCODE
    if ($benchmarkExit -ne 0) {
        Write-Warning "GPU OCR benchmark failed with exit code $benchmarkExit; detector training will continue."
    }
} else {
    Write-Output (
        @{
            state = "gpu_ocr_benchmark_already_completed"
            report = $gpuBenchmark
        } | ConvertTo-Json -Compress
    )
}

if (
    -not (Test-Path -LiteralPath $deepScoresModel -PathType Leaf) -or
    -not (Test-Path -LiteralPath $deepScoresReport -PathType Leaf)
) {
    Wait-CompetingGpuProcess
    & $storageGuard `
        -Stage "deepscores-symbol-detector" `
        -RequiredNewArtifactGiB 1.5
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_symbol_detector_full.sh `
        /workspace/pigeon-score-scan/training_data/models/deepscores-symbol-detector-e8-b2-20260727
    if ($LASTEXITCODE -ne 0) {
        throw "DeepScores detector training failed with exit code $LASTEXITCODE"
    }
}

while (-not (Test-Path -LiteralPath $quartetManifest -PathType Leaf)) {
    Start-Sleep -Seconds 20
}
if (
    (Test-Path -LiteralPath $quartetModel -PathType Leaf) -and
    -not (Test-Path -LiteralPath $quartetReport -PathType Leaf) -and
    -not (Test-Path -LiteralPath $quartetInitializationAudit -PathType Leaf)
) {
    & (Join-Path $projectDir "app\.venv\Scripts\python.exe") `
        -m app.tools.audit_legacy_detector_initialization `
        --model $quartetModel `
        --metrics (Join-Path $quartetModelDir "metrics.partial.json") `
        --run-config (Join-Path $quartetModelDir "run_config.json") `
        --prepared-manifest $quartetManifest `
        --output $quartetInitializationAudit `
        --minimum-epochs 6 `
        --minimum-map-50 0.70 `
        --minimum-map-75 0.65 `
        --minimum-priority-map 0.50
    if ($LASTEXITCODE -ne 0) {
        throw "Legacy quartet detector is not safe even for initialization"
    }
}
if (
    -not (Test-Path -LiteralPath $quartetReport -PathType Leaf) -and
    (Test-Path -LiteralPath $quartetInitializationAudit -PathType Leaf)
) {
    $initializationAudit = Get-Content `
        -LiteralPath $quartetInitializationAudit `
        -Raw `
        -Encoding UTF8 |
        ConvertFrom-Json
    $quartetModelHash = (
        Get-FileHash -LiteralPath $quartetModel -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if (
        $initializationAudit.passed -ne $true -or
        $initializationAudit.deployment_eligible -ne $false -or
        $initializationAudit.release_accuracy_evidence -ne $false -or
        $initializationAudit.model_sha256 -ne $quartetModelHash
    ) {
        throw "Legacy quartet initialization audit is invalid or stale"
    }
}
if (
    -not (Test-Path -LiteralPath $quartetModel -PathType Leaf) -or
    (
        -not (Test-Path -LiteralPath $quartetReport -PathType Leaf) -and
        -not (
            Test-Path `
                -LiteralPath $quartetInitializationAudit `
                -PathType Leaf
        )
    )
) {
    Wait-CompetingGpuProcess
    & $storageGuard `
        -Stage "openscore-quartet-semantic-detector" `
        -RequiredNewArtifactGiB 1.5
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_openscore_semantic_detector_full.sh `
        /workspace/pigeon-score-scan/training_data/models/openscore-semantic-detector-e6-b2-20260728
    if ($LASTEXITCODE -ne 0) {
        throw "OpenScore quartet semantic detector training failed with exit code $LASTEXITCODE"
    }
}

while (-not (Test-Path -LiteralPath $liederManifest -PathType Leaf)) {
    Start-Sleep -Seconds 20
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    $overlapPreparationScript
if ($LASTEXITCODE -ne 0) {
    throw "Overlap-consistent semantic dataset preparation failed"
}
if (
    -not (Test-Path -LiteralPath $quartetRecoveryModel -PathType Leaf) -or
    -not (Test-Path -LiteralPath $quartetRecoveryReport -PathType Leaf)
) {
    Wait-CompetingGpuProcess
    & $storageGuard `
        -Stage "openscore-overlap-recovery" `
        -RequiredNewArtifactGiB 1.0
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_openscore_overlap_recovery.sh `
        /workspace/pigeon-score-scan/training_data/models/openscore-semantic-detector-overlap-recovery-v3-e4-b2-20260729 `
        /workspace/pigeon-score-scan/training_data/models/openscore-semantic-detector-e6-b2-20260728/model.best.pt
    if ($LASTEXITCODE -ne 0) {
        throw "Overlap-consistent quartet recovery training failed with exit code $LASTEXITCODE"
    }
}
$recoveryTraining = Get-Content `
    -LiteralPath $quartetRecoveryReport `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
$recoveryModelHash = (
    Get-FileHash -LiteralPath $quartetRecoveryModel -Algorithm SHA256
).Hash.ToLowerInvariant()
$quartetCategories = Join-Path $projectDir "training_data\prepared\openscore_string_quartets_svg_regions_overlap_consistent_deduplicated_v4\categories.json"
$combinedReplayCategories = Join-Path $liederReplayDir "categories.json"
$quartetCategoryHash = (
    Get-FileHash -LiteralPath $quartetCategories -Algorithm SHA256
).Hash.ToLowerInvariant()
$combinedCategoryHash = (
    Get-FileHash -LiteralPath $combinedReplayCategories -Algorithm SHA256
).Hash.ToLowerInvariant()
if (
    $recoveryTraining.acceptance.passed -ne $true -or
    [int]$recoveryTraining.completed_epochs -ne 2 -or
    [int]$recoveryTraining.planned_epochs -ne 4 -or
    $recoveryTraining.runtime_truncated -ne $true -or
    [double]$recoveryTraining.best_map_50 -lt 0.75 -or
    [double]$recoveryTraining.best_map_75 -lt 0.70 -or
    [double]$recoveryTraining.best_priority_mark_map -lt 0.55 -or
    $recoveryTraining.best_model_sha256 -ne $recoveryModelHash -or
    $quartetCategoryHash -ne $combinedCategoryHash
) {
    throw "Audited semantic recovery is not safe for the efficiency handoff"
}
$combinedTrainTiles = [Linq.Enumerable]::Count(
    [IO.File]::ReadLines((Join-Path $liederReplayDir "train.jsonl"))
)
$combinedTestTiles = [Linq.Enumerable]::Count(
    [IO.File]::ReadLines((Join-Path $combinedSemanticEvaluationDir "test.jsonl"))
)
$efficiencyDecisionPayload = [ordered]@{
    format = 1
    passed = $true
    decision = "skip_dedicated_lieder_synthetic_finetune"
    reason = "registered_scan_finetuning_with_bounded_combined_replay_has_higher_boundary_relevance"
    skipped_stage = [ordered]@{
        minimum_training_tiles_before_first_gate = 2 * $combinedTrainTiles
        evaluation_tiles_at_first_gate = $combinedTestTiles
    }
    retained_evidence = [ordered]@{
        overlap_recovery_model_sha256 = $recoveryModelHash
        overlap_recovery_completed_epochs = 2
        overlap_recovery_minimum_map_50 = 0.75
        overlap_recovery_minimum_map_75 = 0.70
        overlap_recovery_minimum_priority_mark_map = 0.55
        lieder_replay_train_tiles = $combinedTrainTiles
        lieder_replay_max_tiles = 60000
        lieder_replay_fraction = 0.35
        category_manifest_sha256 = $combinedCategoryHash
        registered_scan_minimum_epochs_before_first_gate = 4
        independent_holdout_minimum_works = 200
    }
    safeguards = @(
        "strict_registered_scan_global_and_per_class_gates",
        "disjoint_200_plus_work_holdout",
        "failed_gate_blocks_candidate_export",
        "dedicated_lieder_stage_remains_available_as_failure_remediation"
    )
}
$efficiencyDecisionTemporary = Join-Path `
    (Split-Path -Parent $semanticEfficiencyDecision) `
    ("." + (Split-Path -Leaf $semanticEfficiencyDecision) + "." + $PID + ".tmp")
$efficiencyDecisionPayload |
    ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $efficiencyDecisionTemporary -Encoding UTF8
Move-Item `
    -LiteralPath $efficiencyDecisionTemporary `
    -Destination $semanticEfficiencyDecision `
    -Force

if ($null -ne $musePreparationProcess) {
    Wait-Process -Id $musePreparationProcess.Id
    $musePreparationProcess.Refresh()
    if ($musePreparationProcess.ExitCode -ne 0) {
        throw (
            "Muse OMR v2 preparation failed with exit code " +
            $musePreparationProcess.ExitCode +
            "; inspect $musePreparationErr"
        )
    }
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    $musePreparationScript
if ($LASTEXITCODE -ne 0) {
    throw "Muse OMR v2 datasets failed post-preparation validation"
}

# Synthetic notation geometry is only initialization. The disjoint registered
# scan-degraded split and forbidden-to-train 200+ work holdout may authorize an
# isolated detector candidate for end-to-end development, but they are rendered
# pages and never authorize canonical resources or final product release claims.
# Keep these GPU stages serial so one 8 GB GPU is never oversubscribed.
if (
    -not (Test-Path -LiteralPath $museModel -PathType Leaf) -or
    -not (Test-Path -LiteralPath $museReport -PathType Leaf)
) {
    Wait-CompetingGpuProcess
    & $storageGuard `
        -Stage "muse-scan-semantic-detector" `
        -RequiredNewArtifactGiB 1.5
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_muse_scan_semantic_detector_full.sh `
        /workspace/pigeon-score-scan/training_data/models/muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730 `
        /workspace/pigeon-score-scan/training_data/models/muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729/model.best.pt
    if ($LASTEXITCODE -ne 0) {
        throw "Muse OMR scan-degraded detector training failed with exit code $LASTEXITCODE"
    }
}
$museTraining = Get-Content -LiteralPath $museReport -Raw -Encoding UTF8 | ConvertFrom-Json
if ($museTraining.acceptance.passed -ne $true) {
    throw "Muse OMR scan-degraded detector did not pass its in-domain accuracy gate"
}

if (-not (Test-Path -LiteralPath $holdoutReport -PathType Leaf)) {
    Wait-CompetingGpuProcess
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_muse_holdout_evaluation.sh
    if ($LASTEXITCODE -ne 0) {
        throw "Disjoint Muse OMR scan-degraded holdout failed with exit code $LASTEXITCODE"
    }
}
$holdout = Get-Content -LiteralPath $holdoutReport -Raw -Encoding UTF8 | ConvertFrom-Json
if ($holdout.acceptance.passed -ne $true) {
    throw "Muse OMR holdout did not pass; isolated candidate export remains blocked"
}
$museCheckpoint = Join-Path $museModelDir "checkpoint.last.pt"
if (Test-Path -LiteralPath $museCheckpoint -PathType Leaf) {
    $museModelHash = (
        Get-FileHash -LiteralPath $museModel -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if (
        $museTraining.best_model_sha256 -ne $museModelHash -or
        $museTraining.runtime.recovery_checkpoint_retained -ne $true -or
        $museTraining.runtime.recovery_checkpoint_policy -ne
            "retain_until_external_acceptance"
    ) {
        throw "Muse recovery checkpoint is not safe to prune after holdout"
    }
    Remove-Item -LiteralPath $museCheckpoint -Force
}

if (-not (Test-Path -LiteralPath $candidateReport -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $gpuPython -PathType Leaf)) {
        throw "Verified portable GPU runtime is unavailable: $gpuPython"
    }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        $releaseCandidateQueue `
        -WaitSemanticHoldoutPid 2147483647
    if ($LASTEXITCODE -ne 0) {
        throw "Semantic detector release-candidate pipeline failed with exit code $LASTEXITCODE"
    }
}
& $auditPython -m app.tools.assert_high_value_gpu_stage `
    --project-root $projectDir `
    --stage "semantic_detector_and_holdout" `
    --output-report (
        Join-Path $projectDir (
            "training_data\logs\high-value-stage-gates\" +
            "semantic-detector-and-holdout-current.json"
        )
    )
if ($LASTEXITCODE -ne 0) {
    throw "Semantic detector stage produced no accepted, hash-bound candidate"
}
