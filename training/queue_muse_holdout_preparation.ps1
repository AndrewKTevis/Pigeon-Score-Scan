param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPreparationPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$datasetDir = Join-Path $projectDir "training_data\external\benchmarks\muse_omr_e27f6a8634_raremarks_v3"
$trainingSelection = Join-Path $projectDir "training_data\external\training\muse_omr_scan_train_stratified_v2\selection.json"
$outputDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_regions_stratified_v3"
$completedReport = Join-Path $outputDir "prepare-report.json"
$failureReport = Join-Path $outputDir "preparation.failed.txt"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "$failureReport.tmp" -Force -ErrorAction SilentlyContinue
$preparation = Get-Process -Id $WaitPreparationPid -ErrorAction SilentlyContinue
if ($null -ne $preparation) {
    Wait-Process -Id $WaitPreparationPid
}

Set-Location -LiteralPath $projectDir
try {
    if (-not (Test-Path -LiteralPath $completedReport -PathType Leaf)) {
        & $python -m app.tools.prepare_muse_omr_scan_regions `
            --dataset-dir $datasetDir `
            --output-dir $outputDir `
            --musescore-exe $museScoreExe `
            --expected-selection-role "external_scan_degraded_development_benchmark_not_training" `
            --forbidden-selection $trainingSelection `
            --negative-ratio 0.08 `
            --minimum-ecc 0.86 `
            --maximum-linear-deviation 0.12 `
            --maximum-translation-fraction 0.10 `
            --minimum-local-correlation-10p 0.62 `
            --minimum-median-local-correlation 0.72 `
            --minimum-accepted-page-fraction 0.75 `
            --minimum-accepted-fraction 0.50 `
            --minimum-accepted-works 200 `
            --registration-workers 4 `
            --resume
        if ($LASTEXITCODE -ne 0) {
            throw "Muse OMR independent holdout preparation exited with code $LASTEXITCODE"
        }
    }
    $report = Get-Content -LiteralPath $completedReport -Raw | ConvertFrom-Json
    if (
        [int]$report.selected_pairs -ne 435 -or
        [int]$report.selected_works -ne 395 -or
        $report.registration_version -ne "muse-omr-bounded-elastic-page-filter-jpeg95@7" -or
        [int]$report.accepted_works -lt 200 -or
        [int]$report.source_count_by_split.test -ne [int]$report.accepted_works -or
        @($report.forbidden_selection_overlap).Count -ne 0 -or
        @($report.forbidden_work_overlap).Count -ne 0
    ) {
        throw "Muse OMR independent holdout failed work-level coverage/isolation"
    }
} catch {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $_.Exception.Message | Set-Content -LiteralPath "$failureReport.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$failureReport.tmp" -Destination $failureReport -Force
    throw
}
