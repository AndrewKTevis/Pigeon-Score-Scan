param()

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$museScore = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
$supplementRoot = Join-Path $projectDir (
    "training_data\external\benchmarks\" +
    "muse_omr_boundary_supplement_v2"
)
$provenancePath = Join-Path $supplementRoot "provenance.json"
$supplementOutput = Join-Path $projectDir (
    "training_data\benchmarks\" +
    "muse_omr_boundary_supplement_v2"
)
$supplementManifest = Join-Path $supplementOutput "boundary_manifest.json"
$baseManifest = Join-Path $projectDir (
    "training_data\benchmarks\muse_omr_boundary_v2\" +
    "boundary_manifest.json"
)
$combinedManifest = Join-Path $projectDir (
    "training_data\benchmarks\muse_omr_boundary_combined_v3\" +
    "boundary_manifest.json"
)
$contractVersion = "printed-western-instrumental-scan-boundary@4"

foreach ($required in @(
    $python,
    $museScore,
    $provenancePath,
    $baseManifest
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Boundary supplement preparation input is missing: $required"
    }
}
$provenance = (
    Get-Content -LiteralPath $provenancePath -Raw -Encoding UTF8 |
        ConvertFrom-Json
)
if (
    $provenance.boundary_contract_version -ne $contractVersion -or
    $provenance.role -ne
        "external_scan_degraded_development_benchmark_not_training" -or
    [int]$provenance.selected_pair_count -ne 135 -or
    [int]$provenance.selected_work_count -ne 135 -or
    [int]$provenance.selected_independent_work_pdf_page_count -ne 570 -or
    @($provenance.files).Count -ne 270
) {
    throw "Boundary supplement provenance failed its completeness contract"
}

Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = (
    (Join-Path $projectDir "app\src"),
    $projectDir
) -join [IO.Path]::PathSeparator

& $python -m app.tools.prepare_muse_omr_benchmark `
    $supplementRoot `
    $supplementOutput `
    --musescore $museScore `
    --timeout-seconds 240
if ($LASTEXITCODE -ne 0) {
    throw "Boundary supplement analysis failed with code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $supplementManifest -PathType Leaf)) {
    throw "Boundary supplement analysis produced no manifest"
}

& $python -m app.tools.merge_muse_omr_boundary_manifests `
    --manifest $baseManifest `
    --manifest $supplementManifest `
    --output $combinedManifest
if ($LASTEXITCODE -ne 0) {
    throw "Boundary manifest merge failed with code $LASTEXITCODE"
}

$combined = (
    Get-Content -LiteralPath $combinedManifest -Raw -Encoding UTF8 |
        ConvertFrom-Json
)
if (
    $combined.boundary_contract_version -ne $contractVersion -or
    [int]$combined.accepted_work_count -lt 200 -or
    [int]$combined.accepted_submitted_document_count -ne
        [int]$combined.accepted_work_count
) {
    throw "Combined boundary manifest failed its work-level contract"
}

[ordered]@{
    state = "completed"
    manifest = $combinedManifest
    accepted_work_count = [int]$combined.accepted_work_count
    accepted_input_page_count = [int]$combined.accepted_input_page_count
    pages_by_score_configuration = (
        $combined.accepted_input_pages_by_score_configuration
    )
    source_image_origin = $combined.source_image_origin
    production_evidence_eligible = [bool](
        $combined.production_evidence_eligible
    )
    development_shape_coverage_complete = [bool](
        $combined.development_shape_coverage_complete
    )
    development_coverage_against_production_shape_minimum = (
        $combined.development_coverage_against_production_shape_minimum
    )
} | ConvertTo-Json -Depth 6
