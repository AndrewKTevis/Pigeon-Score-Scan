param(
    [int]$WaitDataPid = 2147483647,
    [int]$WaitGpuPid = 2147483647
)

$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$programmaticDir = Join-Path $projectDir "training_data\prepared\accidental_presence_programmatic_v2"
$programmaticReport = Join-Path $programmaticDir "prepare-report.json"
$registeredDir = Join-Path $projectDir "training_data\prepared\muse_omr_registered_accidental_presence_v1"
$registeredReport = Join-Path $registeredDir "prepare-report.json"
$holdoutDir = Join-Path $projectDir "training_data\prepared\muse_omr_registered_accidental_presence_holdout_v1"
$holdoutReport = Join-Path $holdoutDir "prepare-report.json"
$modelDir = Join-Path $projectDir "training_data\models\accidental-presence-registered-v2-20260729"
$model = Join-Path $modelDir "model.json"
$trainingReport = Join-Path $modelDir "training-report.json"
$evaluation = Join-Path $modelDir "evaluation.independent-registered-scan.json"
$failureReport = "$modelDir.training.failed.txt"

foreach ($waitId in @($WaitDataPid, $WaitGpuPid)) {
    $waitProcess = Get-Process -Id $waitId -ErrorAction SilentlyContinue
    if ($null -ne $waitProcess) {
        Wait-Process -Id $waitId
    }
}

foreach ($required in @($python, $registeredReport, $holdoutReport)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Registered accidental model input is missing: $required"
    }
}

New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
Set-Location -LiteralPath $projectDir
try {
    if (-not (Test-Path -LiteralPath $programmaticReport -PathType Leaf)) {
        & $python -m app.tools.prepare_accidental_presence_programmatic `
            --output-dir $programmaticDir
        if ($LASTEXITCODE -ne 0) {
            throw "Programmatic accidental preparation exited with code $LASTEXITCODE"
        }
    }

    if (
        -not (Test-Path -LiteralPath $model -PathType Leaf) -or
        -not (Test-Path -LiteralPath $trainingReport -PathType Leaf)
    ) {
        & $python -m app.tools.train_accidental_presence_guard `
            --train-data (Join-Path $programmaticDir "train.npz") `
            --safety-data (Join-Path $programmaticDir "safety.npz") `
            --independent-data (Join-Path $programmaticDir "independent.npz") `
            --programmatic-report $programmaticReport `
            --registered-data-dir $registeredDir `
            --output $model `
            --report $trainingReport `
            --jobs 6
        if ($LASTEXITCODE -ne 0) {
            throw "Registered accidental model training exited with code $LASTEXITCODE"
        }
    }

    & $python -m app.tools.evaluate_registered_accidental_presence_holdout `
        --model $model `
        --training-report $trainingReport `
        --registered-training-dir $registeredDir `
        --holdout-dir $holdoutDir `
        --output $evaluation `
        --minimum-works 200 `
        --minimum-roc-auc 0.94 `
        --minimum-class-recall 0.30
    if ($LASTEXITCODE -ne 0) {
        throw "Registered accidental independent evaluation exited with code $LASTEXITCODE"
    }
} catch {
    $_.Exception.Message |
        Set-Content -LiteralPath "$failureReport.tmp" -Encoding UTF8
    Move-Item `
        -LiteralPath "$failureReport.tmp" `
        -Destination $failureReport `
        -Force
    throw
}
