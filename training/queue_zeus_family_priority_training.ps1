param(
    [Parameter(Mandatory = $true)]
    [int]$WaitGpuPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$preparedDir = Join-Path $projectDir "training_data\prepared\olimpic_real_plus_replay_v4_source_document_safe"
$preparedManifest = Join-Path $preparedDir "manifest.json"
$baseModel = Join-Path $projectDir "training_data\models\zeus-olimpic-real-replay-e2-b8-lr1e5-20260727"
$smokeDir = Join-Path $projectDir "training_data\models\zeus-olimpic-family-priority-source-doc-safe-mp-smoke-20260729"
$modelDir = Join-Path $projectDir "training_data\models\zeus-olimpic-real-family-priority-source-doc-safe-mp-e6-b16-lr5e6-20260729"
$trainingReport = Join-Path $modelDir "training_report.json"
$calibrationGate = Join-Path $modelDir "family-priority-calibration-gate.json"
$evaluationDir = Join-Path $projectDir "training_data\models\zeus-olimpic-real-family-priority-source-doc-safe-upstream-test-20260729"
$evaluationReport = Join-Path $evaluationDir "training_report.json"
$upstreamGate = Join-Path $evaluationDir "upstream-candidate-test-gate.json"
$completionReport = Join-Path $projectDir "training_data\logs\zeus-family-priority-source-doc-safe-queue.completion.json"
$storageGuard = Join-Path $PSScriptRoot "assert_training_storage.ps1"

foreach ($required in @(
    $python,
    $preparedManifest,
    (Join-Path $preparedDir "train.pickle"),
    (Join-Path $preparedDir "calibration.pickle"),
    (Join-Path $preparedDir "candidate_test.pickle"),
    (Join-Path $baseModel "weights.h5"),
    (Join-Path $baseModel "tags.txt"),
    (Join-Path $baseModel "options.json")
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Zeus family-priority input is missing: $required"
    }
}

$manifest = Get-Content -LiteralPath $preparedManifest -Raw -Encoding UTF8 |
    ConvertFrom-Json
$selected = $manifest.splits.train.synthetic_selected_family_summary.tokens
$uniform = $manifest.splits.train.synthetic_uniform_family_summary.tokens
if (
    $manifest.name -ne "scorescan-olimpic-real-plus-synthetic-replay-v4-source-document-safe" -or
    $manifest.splits.train.synthetic_selection_profile -ne
        "family-priority-work-balanced-v1" -or
    [int]$manifest.source_group_overlap.train_calibration -ne 0 -or
    [int]$manifest.source_group_overlap.train_candidate_test -ne 0 -or
    [int]$manifest.source_group_overlap.calibration_candidate_test -ne 0 -or
    [int]$manifest.source_document_overlap.train_calibration -ne 0 -or
    [int]$manifest.source_document_overlap.train_candidate_test -ne 0 -or
    [int]$manifest.source_document_overlap.calibration_candidate_test -ne 0 -or
    [int]$manifest.splits.calibration.source_documents -lt 1 -or
    [int]$selected.tie -le [int]$uniform.tie -or
    [int]$selected.slur -le [int]$uniform.slur -or
    [int]$selected.articulation -le [int]$uniform.articulation -or
    [int]$selected.ornament -le [int]$uniform.ornament
) {
    throw "Family-priority dataset failed its immutable balance/leakage audit"
}

$gpuProcess = Get-Process -Id $WaitGpuPid -ErrorAction SilentlyContinue
if ($null -ne $gpuProcess) {
    Wait-Process -Id $WaitGpuPid
}
& $storageGuard `
    -Stage "zeus-family-priority" `
    -RequiredNewArtifactGiB 1.0
Set-Location -LiteralPath $projectDir

if (-not (Test-Path -LiteralPath (Join-Path $smokeDir "training_report.json") -PathType Leaf)) {
    if (Test-Path -LiteralPath $smokeDir) {
        throw "Incomplete mixed-precision smoke directory already exists: $smokeDir"
    }
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_zeus_training.sh `
        --base-model /workspace/pigeon-score-scan/training_data/models/zeus-olimpic-real-replay-e2-b8-lr1e5-20260727 `
        --prepared-dir /workspace/pigeon-score-scan/training_data/prepared/olimpic_real_plus_replay_v4_source_document_safe `
        --output-dir /workspace/pigeon-score-scan/training_data/models/zeus-olimpic-family-priority-source-doc-safe-mp-smoke-20260729 `
        --epochs 1 `
        --batch-size 8 `
        --learning-rate 0.000005 `
        --minimum-ser-improvement 0.0 `
        --maximum-family-f1-regression 100.0 `
        --max-train-samples 64 `
        --max-calibration-samples 32 `
        --precision mixed_float16
    if ($LASTEXITCODE -ne 0) {
        throw "Zeus mixed-precision smoke training failed"
    }
}
$smokeReport = Get-Content `
    -LiteralPath (Join-Path $smokeDir "training_report.json") `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
$smokeLoss = [double]$smokeReport.metrics_percent.epochs[0].fit.loss
if (
    $smokeReport.runtime.keras_policy -ne "mixed_float16" -or
    $smokeReport.runtime.gpu_devices.Count -lt 1 -or
    $smokeReport.metrics_percent.epochs.Count -ne 1 -or
    [double]::IsNaN($smokeLoss) -or
    [double]::IsInfinity($smokeLoss)
) {
    throw "Zeus mixed-precision smoke report is incomplete or non-finite"
}

if (-not (Test-Path -LiteralPath $trainingReport -PathType Leaf)) {
    if (Test-Path -LiteralPath $modelDir) {
        throw "Incomplete Zeus family-priority model directory exists: $modelDir"
    }
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_zeus_training.sh `
        --base-model /workspace/pigeon-score-scan/training_data/models/zeus-olimpic-real-replay-e2-b8-lr1e5-20260727 `
        --prepared-dir /workspace/pigeon-score-scan/training_data/prepared/olimpic_real_plus_replay_v4_source_document_safe `
        --output-dir /workspace/pigeon-score-scan/training_data/models/zeus-olimpic-real-family-priority-source-doc-safe-mp-e6-b16-lr5e6-20260729 `
        --epochs 6 `
        --batch-size 16 `
        --learning-rate 0.000005 `
        --minimum-ser-improvement 0.05 `
        --maximum-family-f1-regression 0.25 `
        --precision mixed_float16
    if ($LASTEXITCODE -ne 0) {
        throw "Zeus family-priority training failed"
    }
}

& $python -m app.tools.gate_zeus_family_priority_candidate `
    --training-report $trainingReport `
    --output-report $calibrationGate `
    --minimum-ser-improvement 0.05 `
    --minimum-tie-improvement 1.0 `
    --minimum-slur-improvement 1.0 `
    --maximum-other-family-regression 0.25
$calibrationPassed = $LASTEXITCODE -eq 0

if ($calibrationPassed) {
    if (-not (Test-Path -LiteralPath $evaluationReport -PathType Leaf)) {
        if (Test-Path -LiteralPath $evaluationDir) {
            throw "Incomplete Zeus upstream-test directory exists: $evaluationDir"
        }
        & wsl.exe -d Ubuntu-24.04 -- bash `
            /workspace/pigeon-score-scan/training/run_wsl_zeus_training.sh `
            --base-model /workspace/pigeon-score-scan/training_data/models/zeus-olimpic-real-family-priority-source-doc-safe-mp-e6-b16-lr5e6-20260729 `
            --prepared-dir /workspace/pigeon-score-scan/training_data/prepared/olimpic_real_plus_replay_v4_source_document_safe `
            --output-dir /workspace/pigeon-score-scan/training_data/models/zeus-olimpic-real-family-priority-source-doc-safe-upstream-test-20260729 `
            --baseline-only `
            --evaluate-candidate-test `
            --batch-size 16 `
            --precision mixed_float16
        if ($LASTEXITCODE -ne 0) {
            throw "Zeus one-time upstream candidate-test evaluation failed"
        }
    }
    & $python -m app.tools.gate_zeus_upstream_candidate_test `
        --evaluation-report $evaluationReport `
        --output-report $upstreamGate `
        --maximum-ser 5.0
    $upstreamPassed = $LASTEXITCODE -eq 0
} else {
    $upstreamPassed = $false
}

@{
    schema_version = 1
    state = "completed"
    calibration_passed = $calibrationPassed
    upstream_candidate_test_opened = $calibrationPassed
    upstream_candidate_passed = $upstreamPassed
    desktop_deployment_authorized = $false
    final_product_release_evidence = $false
    model = $modelDir
    calibration_gate = $calibrationGate
    upstream_gate = $(if ($calibrationPassed) { $upstreamGate } else { $null })
} | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath $completionReport -Encoding UTF8

Write-Output (
    Get-Content -LiteralPath $completionReport -Raw -Encoding UTF8
)
& $python -m app.tools.assert_high_value_gpu_stage `
    --project-root $projectDir `
    --stage "semantic_family_priority" `
    --output-report (
        Join-Path $projectDir (
            "training_data\logs\high-value-stage-gates\" +
            "semantic-family-priority-current.json"
        )
    )
if ($LASTEXITCODE -ne 0) {
    throw "Zeus family-priority stage did not pass both candidate gates"
}
