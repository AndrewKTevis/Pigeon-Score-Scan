$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
$corpusDir = Join-Path $projectDir "training_data\external\corpora\openscore_string_quartets_d13289cd\StringQuartets-d13289cd70797da94646e5cf64f7296a4c4fee40"
$outputDir = Join-Path $projectDir "training_data\prepared\openscore_string_quartets_svg_regions_v1"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

Set-Location -LiteralPath $projectDir
python -m app.tools.prepare_openscore_svg_regions `
    --corpus-dir $corpusDir `
    --output-dir $outputDir `
    --musescore-exe $museScoreExe `
    --resume
if ($LASTEXITCODE -ne 0) {
    throw "OpenScore SVG preparation failed with exit code $LASTEXITCODE"
}
