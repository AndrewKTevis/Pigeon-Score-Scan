param(
    [Parameter(Mandatory = $true)]
    [int]$WaitClassificationPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$baseManifest = Join-Path $projectDir (
    "training_data\benchmarks\muse_omr_boundary_combined_v3\" +
    "boundary_manifest.json"
)
$classificationManifest = Join-Path $projectDir (
    "training_data\benchmarks\muse_omr_training_boundary_classification_v1\" +
    "boundary_manifest.json"
)
$sourceTraining = Join-Path $projectDir (
    "training_data\external\training\muse_omr_scan_train_e27f6a8634"
)
$outputTraining = Join-Path $projectDir (
    "training_data\external\training\muse_omr_scan_train_stratified_v2"
)
$oldHoldoutSelection = Join-Path $projectDir (
    "training_data\external\benchmarks\" +
    "muse_omr_e27f6a8634_raremarks_v3\selection.json"
)
$supplementSelection = Join-Path $projectDir (
    "training_data\external\benchmarks\" +
    "muse_omr_boundary_supplement_v2\selection.json"
)
$outputDir = Join-Path $projectDir (
    "training_data\benchmarks\muse_omr_boundary_stratified_v4"
)
$partitionPlan = Join-Path $outputDir "partition_plan.json"
$evaluationManifest = Join-Path $outputDir "boundary_manifest.json"

$classification = Get-Process `
    -Id $WaitClassificationPid `
    -ErrorAction SilentlyContinue
if ($null -ne $classification) {
    Wait-Process -Id $WaitClassificationPid
}

foreach ($required in @(
    $python,
    $baseManifest,
    $classificationManifest,
    (Join-Path $sourceTraining "selection.json"),
    (Join-Path $sourceTraining "benchmark_dataset.json"),
    $oldHoldoutSelection,
    $supplementSelection
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Stratified partition input is missing: $required"
    }
}
foreach ($freshTarget in @($outputTraining, $outputDir)) {
    if (
        (Test-Path -LiteralPath $freshTarget -PathType Container) -and
        (Get-ChildItem -LiteralPath $freshTarget -Force | Select-Object -First 1)
    ) {
        throw "Refusing to reuse a nonempty partition target: $freshTarget"
    }
}

Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = (
    (Join-Path $projectDir "app\src"),
    $projectDir
) -join [IO.Path]::PathSeparator
& $python -m app.tools.promote_muse_training_cases_to_boundary `
    --base-manifest $baseManifest `
    --training-classification-manifest $classificationManifest `
    --source-training-root $sourceTraining `
    --output-training-root $outputTraining `
    --frozen-selection $oldHoldoutSelection `
    --frozen-selection $supplementSelection `
    --output-evaluation-manifest $evaluationManifest `
    --output-partition-plan $partitionPlan `
    --seed 20260729
if ($LASTEXITCODE -ne 0) {
    throw "Muse stratified partition construction failed with code $LASTEXITCODE"
}

$plan = Get-Content -LiteralPath $partitionPlan -Raw -Encoding UTF8 |
    ConvertFrom-Json
$evaluation = Get-Content `
    -LiteralPath $evaluationManifest `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
$selectionPath = Join-Path $outputTraining "selection.json"
$provenancePath = Join-Path $outputTraining "provenance.json"
$selection = Get-Content -LiteralPath $selectionPath -Raw -Encoding UTF8 |
    ConvertFrom-Json
$provenance = Get-Content -LiteralPath $provenancePath -Raw -Encoding UTF8 |
    ConvertFrom-Json
$configurationPages = $evaluation.accepted_input_pages_by_score_configuration
if (
    $plan.model_outputs_used_for_selection -ne $false -or
    @($plan.coverage_gaps.PSObject.Properties |
        Where-Object { [int]$_.Value -ne 0 }).Count -ne 0 -or
    $evaluation.selection_used_model_outputs -ne $false -or
    $evaluation.source_image_origin -ne
        "synthetic_scan_degraded_render" -or
    $evaluation.production_evidence_eligible -ne $false -or
    $evaluation.production_scope_coverage_complete -ne $false -or
    $evaluation.development_shape_coverage_complete -ne $true -or
    [int]$evaluation.accepted_input_page_count -lt 2000 -or
    [int]$evaluation.accepted_work_count -lt 200 -or
    [int]$configurationPages.solo_monophonic -lt 400 -or
    [int]$configurationPages.piano -lt 400 -or
    [int]$configurationPages.monophonic_ensemble -lt 400 -or
    [int]$configurationPages.piano_plus_monophonic_ensemble -lt 400 -or
    @($evaluation.training_evaluation_work_overlap).Count -ne 0 -or
    $selection.partition_contract_version -ne
        "muse-omr-boundary-promotion@1" -or
    [int]$selection.selected_work_count -lt 200 -or
    @($selection.training_holdout_overlap).Count -ne 0 -or
    @($selection.training_holdout_work_overlap).Count -ne 0 -or
    @($provenance.files).Count -ne
        (2 * [int]$selection.selected_pair_count)
) {
    throw "Muse stratified partition failed its coverage/isolation gate"
}

[ordered]@{
    state = "completed"
    promoted_work_count = [int]$plan.promoted_work_count
    training_pair_count = [int]$selection.selected_pair_count
    training_work_count = [int]$selection.selected_work_count
    evaluation_work_count = [int]$evaluation.accepted_work_count
    evaluation_page_count = [int]$evaluation.accepted_input_page_count
    pages_by_score_configuration = $configurationPages
    training_selection_sha256 = (
        Get-FileHash -LiteralPath $selectionPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    partition_plan_sha256 = (
        Get-FileHash -LiteralPath $partitionPlan -Algorithm SHA256
    ).Hash.ToLowerInvariant()
} | ConvertTo-Json -Depth 5
