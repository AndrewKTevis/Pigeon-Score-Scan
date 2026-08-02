$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$preparedDir = Join-Path $projectDir "training_data\prepared"
$cleanDirs = @(
    (Join-Path $preparedDir "openscore_lieder_train_1091_pdf_detection_exhaustive_v1_shard-00-of-04"),
    (Join-Path $preparedDir "openscore_lieder_train_1091_pdf_detection_exhaustive_v1_shard-01-of-04"),
    (Join-Path $preparedDir "openscore_lieder_train_1091_pdf_detection_exhaustive_v1_shard-02-of-04"),
    (Join-Path $preparedDir "openscore_lieder_train_1091_pdf_detection_exhaustive_v1_shard-03-of-04")
)
$scanDir = Join-Path $preparedDir "muse_omr_scan_text_stratified_v3"
$outputDir = Join-Path $preparedDir "scorescan_ocr_detection_v2"
$completedReport = Join-Path $outputDir "merge-report.json"
$failureReport = "$outputDir.preparation.failed.txt"

while ($true) {
    $failures = @(
        $cleanDirs |
            ForEach-Object { Join-Path $_ "preparation.failed.txt" } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    )
    if ($failures.Count -gt 0) {
        throw "Exhaustive detection preparation failed: $($failures -join ', ')"
    }
    $missingReports = @(
        $cleanDirs |
            ForEach-Object { Join-Path $_ "prepare-report.json" } |
            Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
    )
    if ($missingReports.Count -eq 0) {
        break
    }
    Start-Sleep -Seconds 10
}

if (-not (Test-Path -LiteralPath (Join-Path $scanDir "prepare-report.json") -PathType Leaf)) {
    throw "Registered-scan positive-label report is missing: $scanDir"
}
if (
    (Test-Path -LiteralPath $outputDir -PathType Container) -and
    (Get-ChildItem -LiteralPath $outputDir -Force | Select-Object -First 1)
) {
    if (Test-Path -LiteralPath $completedReport -PathType Leaf) {
        exit 0
    }
    throw "Detection merge output is nonempty but incomplete: $outputDir"
}

Set-Location -LiteralPath $projectDir
try {
    & $pythonExe -m app.tools.merge_ocr_detection_labels `
        --project-root $projectDir `
        --output-dir $outputDir `
        --clean-dataset "lieder_d0=$($cleanDirs[0])" `
        --clean-dataset "lieder_d1=$($cleanDirs[1])" `
        --clean-dataset "lieder_d2=$($cleanDirs[2])" `
        --clean-dataset "lieder_d3=$($cleanDirs[3])" `
        --scan-dataset "muse_scan=$scanDir" `
        --scan-target-fraction 0.35 `
        --seed 20260728
    if ($LASTEXITCODE -ne 0) {
        throw "Exhaustive OCR detection label merge exited with code $LASTEXITCODE"
    }
    Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
} catch {
    $_.Exception.ToString() | Set-Content -LiteralPath $failureReport -Encoding UTF8
    throw
}
