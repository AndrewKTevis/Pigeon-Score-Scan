$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$preparedDir = Join-Path $projectDir "training_data\prepared"
$cleanDirs = 0..3 | ForEach-Object {
    $shard = "shard-{0:D2}-of-04" -f $_
    Join-Path $preparedDir (
        "openscore_lieder_train_1091_pdf_detection_exhaustive_v1_$shard"
    )
}
$scanDirs = 0..3 | ForEach-Object {
    $shard = "shard-{0:D2}-of-04" -f $_
    Join-Path $preparedDir (
        "muse_omr_scan_detection_exhaustive_stratified_v2_$shard"
    )
}
$sourceDirs = @($cleanDirs) + @($scanDirs)
$outputDir = Join-Path $preparedDir "scorescan_ocr_detection_stratified_v4"
$completedReport = Join-Path $outputDir "merge-report.json"
$failureReport = "$outputDir.preparation.failed.txt"

while ($true) {
    $missingReports = @(
        $sourceDirs |
            ForEach-Object { Join-Path $_ "prepare-report.json" } |
            Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
    )
    if ($missingReports.Count -eq 0) {
        break
    }
    Start-Sleep -Seconds 10
}

if (
    (Test-Path -LiteralPath $outputDir -PathType Container) -and
    (Get-ChildItem -LiteralPath $outputDir -Force | Select-Object -First 1)
) {
    if (Test-Path -LiteralPath $completedReport -PathType Leaf) {
        exit 0
    }
    throw "Fully exhaustive detection merge is nonempty but incomplete"
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
        --scan-dataset "muse_scan_d0=$($scanDirs[0])" `
        --scan-dataset "muse_scan_d1=$($scanDirs[1])" `
        --scan-dataset "muse_scan_d2=$($scanDirs[2])" `
        --scan-dataset "muse_scan_d3=$($scanDirs[3])" `
        --scan-target-fraction 0.35 `
        --seed 20260728
    if ($LASTEXITCODE -ne 0) {
        throw "Fully exhaustive OCR detection merge exited with code $LASTEXITCODE"
    }
    Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
} catch {
    $_.Exception.ToString() | Set-Content -LiteralPath $failureReport -Encoding UTF8
    throw
}
