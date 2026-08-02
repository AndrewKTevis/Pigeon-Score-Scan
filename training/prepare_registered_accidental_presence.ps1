$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$datasetDir = Join-Path $projectDir "training_data\external\training\muse_omr_scan_train_stratified_v2"
$regionDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_regions_stratified_v3"
$outputDir = Join-Path $projectDir "training_data\prepared\muse_omr_registered_accidental_presence_v1"
$completedReport = Join-Path $outputDir "prepare-report.json"
$failureReport = Join-Path $outputDir "preparation.failed.txt"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

foreach ($required in @(
    $python,
    (Join-Path $datasetDir "benchmark_dataset.json"),
    (Join-Path $regionDir "prepare-report.json"),
    $museScoreExe
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Registered accidental preparation input is missing: $required"
    }
}

if (Test-Path -LiteralPath $completedReport -PathType Leaf) {
    $report = Get-Content -LiteralPath $completedReport -Raw | ConvertFrom-Json
    if (
        $report.name -ne "scorescan-registered-scan-accidental-presence-v1" -or
        $report.holdout_used_for_training -ne $false -or
        @($report.work_intersections.train_calibration).Count -ne 0 -or
        @($report.work_intersections.train_test).Count -ne 0 -or
        @($report.work_intersections.calibration_test).Count -ne 0
    ) {
        throw "Existing registered accidental dataset failed its isolation contract"
    }
    exit 0
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
Set-Location -LiteralPath $projectDir
try {
    & $python -m app.tools.prepare_registered_accidental_presence `
        --dataset-dir $datasetDir `
        --region-dir $regionDir `
        --output-dir $outputDir `
        --musescore-exe $museScoreExe `
        --negative-ratio 2 `
        --maximum-samples-per-pair 240
    if ($LASTEXITCODE -ne 0) {
        throw "Registered accidental preparation exited with code $LASTEXITCODE"
    }
} catch {
    $_.Exception.Message | Set-Content -LiteralPath "$failureReport.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$failureReport.tmp" -Destination $failureReport -Force
    throw
}
