param(
    [Parameter(Mandatory = $true)]
    [int]$WaitHoldoutRegionPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$datasetDir = Join-Path $projectDir "training_data\external\benchmarks\muse_omr_e27f6a8634_raremarks_v3"
$regionDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_regions_stratified_v3"
$regionReport = Join-Path $regionDir "prepare-report.json"
$regionFailure = Join-Path $regionDir "preparation.failed.txt"
$outputDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_text_stratified_v3"
$reusablePdfDir = Join-Path $projectDir (
    "training_data\prepared\muse_omr_scan_holdout_text_v2\pdf"
)
$completedReport = Join-Path $outputDir "prepare-report.json"
$failureReport = Join-Path $outputDir "preparation.failed.txt"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
$preparedRoot = Join-Path $projectDir "training_data\prepared"
. (Join-Path $PSScriptRoot "ocr_artifact_freshness.ps1")

Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "$failureReport.tmp" -Force -ErrorAction SilentlyContinue
$preparation = Get-Process -Id $WaitHoldoutRegionPid -ErrorAction SilentlyContinue
if ($null -ne $preparation) {
    Wait-Process -Id $WaitHoldoutRegionPid
}
if (Test-Path -LiteralPath $regionFailure -PathType Leaf) {
    throw "Independent Muse OMR holdout region preparation failed"
}
if (-not (Test-Path -LiteralPath $regionReport -PathType Leaf)) {
    throw "Independent Muse OMR holdout region report is missing"
}
if (
    (Test-Path -LiteralPath $completedReport -PathType Leaf) -and
    -not (Test-OcrArtifactFreshness `
        -Python $python `
        -ProjectDir $projectDir `
        -Kind "text" `
        -ArtifactReport $completedReport `
        -SourceReports @($regionReport))
) {
    Move-StaleOcrArtifact `
        -ArtifactDir $outputDir `
        -PreparedRoot $preparedRoot
}

Set-Location -LiteralPath $projectDir
try {
    if (-not (Test-Path -LiteralPath $completedReport -PathType Leaf)) {
        & $python -m app.tools.prepare_muse_omr_scan_text `
            --dataset-dir $datasetDir `
            --region-dir $regionDir `
            --output-dir $outputDir `
            --reuse-pdf-dir $reusablePdfDir `
            --musescore-exe $museScoreExe `
            --expected-region-role "external_scan_degraded_development_benchmark_not_training" `
            --padding-pixels 10 `
            --minimum-crop-stddev 4 `
            --minimum-dark-fraction 0.003 `
            --resume
        if ($LASTEXITCODE -ne 0) {
            throw "Independent Muse OMR holdout text preparation exited with code $LASTEXITCODE"
        }
    }
    & $python -m app.tools.prune_muse_omr_reference_cache `
        --region-dir $regionDir `
        --text-dir $outputDir `
        --execute
    if ($LASTEXITCODE -ne 0) {
        throw "Independent holdout reference-cache pruning failed with code $LASTEXITCODE"
    }
} catch {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $_.Exception.Message | Set-Content -LiteralPath "$failureReport.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$failureReport.tmp" -Destination $failureReport -Force
    throw
}
