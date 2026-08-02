param(
    [ValidateRange(0, 63)]
    [int]$ShardIndex = 0,
    [ValidateRange(1, 64)]
    [int]$ShardCount = 4
)

$ErrorActionPreference = "Stop"

if ($ShardIndex -ge $ShardCount) {
    throw "ShardIndex must be smaller than ShardCount"
}

$projectDir = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$datasetDir = Join-Path $projectDir "training_data\external\training\muse_omr_scan_train_stratified_v2"
$regionDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_regions_stratified_v3"
$reusePdfDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_text_stratified_v3\pdf"
$regionReport = Join-Path $regionDir "prepare-report.json"
$regionFailure = Join-Path $regionDir "registration-failure-report.json"
$textReport = Join-Path (
    Split-Path -Parent $reusePdfDir
) "prepare-report.json"
$textFailure = Join-Path (
    Split-Path -Parent $reusePdfDir
) "preparation.failed.txt"
$shardName = "shard-{0:D2}-of-{1:D2}" -f $ShardIndex, $ShardCount
$outputDir = Join-Path $projectDir (
        "training_data\prepared\muse_omr_scan_detection_exhaustive_stratified_v2_$shardName"
)
$completedReport = Join-Path $outputDir "prepare-report.json"
$failureReport = Join-Path $outputDir "preparation.failed.txt"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

# Keep the GPU trainer fed while the clean exhaustive data is generated.
# Registered-scan NCC verification starts as soon as those four CPU shards
# finish, avoiding eight simultaneous PDF workers contending for the same CPU.
$cleanReports = 0..3 | ForEach-Object {
    $cleanShard = "shard-{0:D2}-of-04" -f $_
    Join-Path $projectDir (
        "training_data\prepared\openscore_lieder_train_1091_pdf_detection_exhaustive_v1_$cleanShard\prepare-report.json"
    )
}
while (@($cleanReports | Where-Object {
    -not (Test-Path -LiteralPath $_ -PathType Leaf)
}).Count -gt 0) {
    Start-Sleep -Seconds 10
}
while (
    -not (Test-Path -LiteralPath $regionReport -PathType Leaf) -or
    -not (Test-Path -LiteralPath $textReport -PathType Leaf)
) {
    if (
        (Test-Path -LiteralPath $regionFailure -PathType Leaf) -or
        (Test-Path -LiteralPath $textFailure -PathType Leaf)
    ) {
        throw "Registered stratified scan preparation failed"
    }
    Start-Sleep -Seconds 10
}

Set-Location -LiteralPath $projectDir
try {
    if (-not (Test-Path -LiteralPath $completedReport -PathType Leaf)) {
        Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
        & $pythonExe -m app.tools.prepare_muse_omr_scan_text `
            --dataset-dir $datasetDir `
            --region-dir $regionDir `
            --output-dir $outputDir `
            --musescore-exe $museScoreExe `
            --reuse-pdf-dir $reusePdfDir `
            --detection-all-visible-text `
            --shard-count $ShardCount `
            --shard-index $ShardIndex `
            --padding-pixels 10 `
            --minimum-crop-stddev 4 `
            --minimum-dark-fraction 0.003 `
            --minimum-visual-presence-ncc 0.15 `
            --resume
        if ($LASTEXITCODE -ne 0) {
            throw "Registered exhaustive detection preparation exited with code $LASTEXITCODE"
        }
    }
    Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
} catch {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $_.Exception.ToString() | Set-Content -LiteralPath $failureReport -Encoding UTF8
    throw
}
