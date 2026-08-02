$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$corpusDir = Join-Path $projectDir "training_data\external\corpora\openscore_lieder_6b2dc542\Lieder-6b2dc542ce2e8aa4b78c8ee62103b210efc07015"
$sourceList = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_sources.txt"
$regionDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_svg_regions_v1"
$regionReport = Join-Path $regionDir "prepare-report.json"
$regionFailure = Join-Path $regionDir "preparation.failed.txt"
$textPython = "python"
$textDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_pdf_text_v2"
$reusePdfDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_pdf_text_v1\pdf"
$textCompletedReport = Join-Path $textDir "prepare-report.json"
$textFailureReport = Join-Path $textDir "preparation.failed.txt"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

while (-not (Test-Path -LiteralPath $regionReport -PathType Leaf)) {
    if (Test-Path -LiteralPath $regionFailure -PathType Leaf) {
        throw "Lieder semantic region preparation failed; text data cannot be aligned"
    }
    Start-Sleep -Seconds 20
}

Set-Location -LiteralPath $projectDir
try {
    if (-not (Test-Path -LiteralPath $textCompletedReport -PathType Leaf)) {
        & $textPython -m app.tools.prepare_openscore_pdf_text `
            --corpus-dir $corpusDir `
            --source-list $sourceList `
            --region-dataset-dir $regionDir `
            --output-dir $textDir `
            --musescore-exe $museScoreExe `
            --reuse-pdf-dir $reusePdfDir `
            --write-crops `
            --crop-padding-pixels 6 `
            --resume
        if ($LASTEXITCODE -ne 0) {
            throw "OpenScore Lieder 1091 text preparation exited with code $LASTEXITCODE"
        }
    }
} catch {
    New-Item -ItemType Directory -Path $textDir -Force | Out-Null
    $_.Exception.Message | Set-Content -LiteralPath $textFailureReport -Encoding UTF8
    throw
}
