param()

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$museScore = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
$trainingRoot = Join-Path $projectDir (
    "training_data\external\training\" +
    "muse_omr_scan_train_e27f6a8634"
)
$outputDir = Join-Path $projectDir (
    "training_data\benchmarks\" +
    "muse_omr_training_boundary_classification_v1"
)
$manifestPath = Join-Path $outputDir "boundary_manifest.json"
$classificationRole = (
    "training_boundary_classification_only_not_evaluation"
)

foreach ($required in @(
    $python,
    $museScore,
    (Join-Path $trainingRoot "selection.json"),
    (Join-Path $trainingRoot "benchmark_dataset.json")
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Training boundary classification input is missing: $required"
    }
}

Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = (
    (Join-Path $projectDir "app\src"),
    $projectDir
) -join [IO.Path]::PathSeparator
& $python -m app.tools.prepare_muse_omr_benchmark `
    $trainingRoot `
    $outputDir `
    --musescore $museScore `
    --timeout-seconds 240 `
    --workers 2 `
    --allow-training-classification
if ($LASTEXITCODE -ne 0) {
    throw "Training boundary classification failed with code $LASTEXITCODE"
}

$manifest = (
    Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 |
        ConvertFrom-Json
)
if (
    $manifest.role -ne $classificationRole -or
    $manifest.source_image_origin -ne
        "synthetic_scan_degraded_render" -or
    $manifest.production_evidence_eligible -ne $false -or
    $manifest.production_scope_coverage_complete -ne $false -or
    [int]$manifest.case_count -ne 384 -or
    [int]$manifest.work_count -ne 381
) {
    throw "Training boundary classification produced an unsafe manifest role"
}

[ordered]@{
    state = "completed"
    manifest = $manifestPath
    accepted_work_count = [int]$manifest.accepted_work_count
    accepted_input_page_count = [int]$manifest.accepted_input_page_count
    pages_by_score_configuration = (
        $manifest.accepted_input_pages_by_score_configuration
    )
} | ConvertTo-Json -Depth 5
