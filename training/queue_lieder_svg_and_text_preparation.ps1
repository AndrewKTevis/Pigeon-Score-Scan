param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$corpusDir = Join-Path $projectDir "training_data\external\corpora\openscore_lieder_6b2dc542\Lieder-6b2dc542ce2e8aa4b78c8ee62103b210efc07015"
$sourceList = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_320_sources.txt"
$regionDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_320_svg_regions_v1"
$textDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_320_pdf_text_v1"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
$pdfPython = "python"

$waitProcess = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
if ($null -ne $waitProcess) {
    Wait-Process -Id $WaitPid
}

Set-Location -LiteralPath $projectDir
python -m app.tools.prepare_openscore_svg_regions `
    --corpus-dir $corpusDir `
    --source-list $sourceList `
    --output-dir $regionDir `
    --musescore-exe $museScoreExe `
    --resume
if ($LASTEXITCODE -ne 0) {
    throw "OpenScore Lieder SVG preparation failed with exit code $LASTEXITCODE"
}

& $pdfPython -m app.tools.prepare_openscore_pdf_text `
    --corpus-dir $corpusDir `
    --source-list $sourceList `
    --region-dataset-dir $regionDir `
    --output-dir $textDir `
    --musescore-exe $museScoreExe
if ($LASTEXITCODE -ne 0) {
    throw "OpenScore Lieder PDF text preparation failed with exit code $LASTEXITCODE"
}
