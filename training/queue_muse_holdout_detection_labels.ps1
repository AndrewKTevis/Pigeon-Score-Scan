param(
    [Parameter(Mandatory = $true)]
    [int]$WaitHoldoutTextPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$textDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_text_stratified_v3"
$textReport = Join-Path $textDir "prepare-report.json"
$textFailure = Join-Path $textDir "preparation.failed.txt"
$outputDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_detection_labels_stratified_v2"
$completedReport = Join-Path $outputDir "prepare-report.json"
$failureReport = "$outputDir.preparation.failed.txt"
$preparedRoot = Join-Path $projectDir "training_data\prepared"
. (Join-Path $PSScriptRoot "ocr_artifact_freshness.ps1")

Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "$failureReport.tmp" -Force -ErrorAction SilentlyContinue
$preparation = Get-Process -Id $WaitHoldoutTextPid -ErrorAction SilentlyContinue
if ($null -ne $preparation) {
    Wait-Process -Id $WaitHoldoutTextPid
}
if (Test-Path -LiteralPath $textFailure -PathType Leaf) {
    throw "Independent Muse OMR holdout text preparation failed"
}
if (-not (Test-Path -LiteralPath $textReport -PathType Leaf)) {
    throw "Independent Muse OMR holdout text report is missing"
}
if (
    (Test-Path -LiteralPath $completedReport -PathType Leaf) -and
    -not (Test-OcrArtifactFreshness `
        -Python $python `
        -ProjectDir $projectDir `
        -Kind "holdout-labels" `
        -ArtifactReport $completedReport `
        -SourceReports @($textReport))
) {
    Move-StaleOcrArtifact `
        -ArtifactDir $outputDir `
        -PreparedRoot $preparedRoot
}

Set-Location -LiteralPath $projectDir
try {
    if (-not (Test-Path -LiteralPath $completedReport -PathType Leaf)) {
        & $python -m app.tools.prepare_ocr_holdout_detection_labels `
            --project-root $projectDir `
            --dataset-dir $textDir `
            --output-dir $outputDir
        if ($LASTEXITCODE -ne 0) {
            throw "Independent OCR detection label preparation exited with code $LASTEXITCODE"
        }
    }
} catch {
    $_.Exception.Message | Set-Content -LiteralPath "$failureReport.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$failureReport.tmp" -Destination $failureReport -Force
    throw
}
