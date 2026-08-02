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
$pythonExe = "python"
$corpusDir = Join-Path $projectDir "training_data\external\corpora\openscore_lieder_6b2dc542\Lieder-6b2dc542ce2e8aa4b78c8ee62103b210efc07015"
$sourceList = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_sources.txt"
$regionDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_svg_regions_v1"
$reusePdfDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_pdf_text_v2\pdf"
$shardName = "shard-{0:D2}-of-{1:D2}" -f $ShardIndex, $ShardCount
$outputDir = Join-Path $projectDir (
    "training_data\prepared\openscore_lieder_train_1091_pdf_detection_exhaustive_v1_$shardName"
)
$completedReport = Join-Path $outputDir "prepare-report.json"
$failureReport = Join-Path $outputDir "preparation.failed.txt"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Python runtime is missing: $pythonExe"
}
if (-not (Test-Path -LiteralPath (Join-Path $regionDir "prepare-report.json") -PathType Leaf)) {
    throw "The source-aligned rendered page dataset is incomplete: $regionDir"
}
if (-not (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $reusePdfDir) "prepare-report.json") -PathType Leaf)) {
    throw "The hash-verified reusable PDF dataset is incomplete: $reusePdfDir"
}

Set-Location -LiteralPath $projectDir
try {
    if (-not (Test-Path -LiteralPath $completedReport -PathType Leaf)) {
        Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
        & $pythonExe -m app.tools.prepare_openscore_pdf_text `
            --corpus-dir $corpusDir `
            --source-list $sourceList `
            --region-dataset-dir $regionDir `
            --output-dir $outputDir `
            --musescore-exe $museScoreExe `
            --reuse-pdf-dir $reusePdfDir `
            --detection-all-visible-text `
            --shard-count $ShardCount `
            --shard-index $ShardIndex `
            --crop-padding-pixels 6 `
            --resume
        if ($LASTEXITCODE -ne 0) {
            throw "Exhaustive visible-text preparation exited with code $LASTEXITCODE"
        }
    }
    Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
} catch {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $_.Exception.ToString() | Set-Content -LiteralPath $failureReport -Encoding UTF8
    throw
}
