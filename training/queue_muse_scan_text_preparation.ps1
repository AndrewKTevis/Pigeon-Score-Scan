$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$datasetDir = Join-Path $projectDir "training_data\external\training\muse_omr_scan_train_stratified_v2"
$regionDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_regions_stratified_v3"
$regionReport = Join-Path $regionDir "prepare-report.json"
$regionFailure = Join-Path $regionDir "registration-failure-report.json"
$textPython = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$textDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_text_stratified_v3"
$reusablePdfDir = Join-Path $projectDir (
    "training_data\prepared\muse_omr_scan_text_v2\pdf"
)
$textReport = Join-Path $textDir "prepare-report.json"
$textFailure = Join-Path $textDir "preparation.failed.txt"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
$preparedRoot = Join-Path $projectDir "training_data\prepared"
. (Join-Path $PSScriptRoot "ocr_artifact_freshness.ps1")

Remove-Item -LiteralPath $textFailure -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "$textFailure.tmp" -Force -ErrorAction SilentlyContinue
while (-not (Test-Path -LiteralPath $regionReport -PathType Leaf)) {
    if (Test-Path -LiteralPath $regionFailure -PathType Leaf) {
        throw "Registered Muse OMR dataset failed its acceptance gate"
    }
    Start-Sleep -Seconds 20
}
if (
    (Test-Path -LiteralPath $textReport -PathType Leaf) -and
    -not (Test-OcrArtifactFreshness `
        -Python $textPython `
        -ProjectDir $projectDir `
        -Kind "text" `
        -ArtifactReport $textReport `
        -SourceReports @($regionReport))
) {
    Move-StaleOcrArtifact `
        -ArtifactDir $textDir `
        -PreparedRoot $preparedRoot
}

Set-Location -LiteralPath $projectDir
try {
    if (-not (Test-Path -LiteralPath $textReport -PathType Leaf)) {
        & $textPython -m app.tools.prepare_muse_omr_scan_text `
            --dataset-dir $datasetDir `
            --region-dir $regionDir `
            --output-dir $textDir `
            --reuse-pdf-dir $reusablePdfDir `
            --musescore-exe $museScoreExe `
            --padding-pixels 10 `
    --minimum-crop-stddev 4 `
    --minimum-dark-fraction 0.003 `
    --minimum-visual-presence-ncc 0.15 `
    --resume
        if ($LASTEXITCODE -ne 0) {
            throw "Muse OMR registered scan text preparation exited with code $LASTEXITCODE"
        }
    }
    & $textPython -m app.tools.prune_muse_omr_reference_cache `
        --region-dir $regionDir `
        --text-dir $textDir `
        --execute
    if ($LASTEXITCODE -ne 0) {
        throw "Muse OMR reference-cache pruning failed with code $LASTEXITCODE"
    }
} catch {
    New-Item -ItemType Directory -Path $textDir -Force | Out-Null
    $_.Exception.Message | Set-Content -LiteralPath "$textFailure.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$textFailure.tmp" -Destination $textFailure -Force
    throw
}
