param(
    [int]$WaitClassificationPid = 2147483647
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
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
$partitionPlan = Join-Path $projectDir (
    "training_data\partitions\muse_omr_training_boundary_filtered_v2\" +
    "partition_plan.json"
)

$classification = Get-Process `
    -Id $WaitClassificationPid `
    -ErrorAction SilentlyContinue
if ($null -ne $classification) {
    Wait-Process -Id $WaitClassificationPid
}
foreach ($required in @(
    $python,
    $classificationManifest,
    (Join-Path $sourceTraining "selection.json"),
    (Join-Path $sourceTraining "benchmark_dataset.json"),
    $oldHoldoutSelection,
    $supplementSelection
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Boundary-filtered training input is missing: $required"
    }
}
if (
    (Test-Path -LiteralPath $outputTraining -PathType Container) -and
    (Get-ChildItem -LiteralPath $outputTraining -Force |
        Select-Object -First 1)
) {
    throw "Refusing to reuse nonempty training target: $outputTraining"
}

Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = (
    (Join-Path $projectDir "app\src"),
    $projectDir
) -join [IO.Path]::PathSeparator
& $python -m app.tools.filter_muse_training_to_boundary `
    --training-classification-manifest $classificationManifest `
    --source-training-root $sourceTraining `
    --output-training-root $outputTraining `
    --frozen-selection $oldHoldoutSelection `
    --frozen-selection $supplementSelection `
    --output-partition-plan $partitionPlan
if ($LASTEXITCODE -ne 0) {
    throw "Muse boundary-filtered training construction failed"
}

$selectionPath = Join-Path $outputTraining "selection.json"
$provenancePath = Join-Path $outputTraining "provenance.json"
$selection = Get-Content -LiteralPath $selectionPath -Raw -Encoding UTF8 |
    ConvertFrom-Json
$provenance = Get-Content -LiteralPath $provenancePath -Raw -Encoding UTF8 |
    ConvertFrom-Json
$plan = Get-Content -LiteralPath $partitionPlan -Raw -Encoding UTF8 |
    ConvertFrom-Json
if (
    $selection.partition_contract_version -ne
        "muse-omr-boundary-filter@1" -or
    [int]$selection.selected_pair_count -lt 200 -or
    [int]$selection.selected_work_count -lt 200 -or
    [int]$selection.reserved_holdout_pair_count -ne 570 -or
    [int]$selection.reserved_holdout_work_count -ne 530 -or
    @($selection.promoted_evaluation_pair_ids).Count -ne 0 -or
    @($selection.training_holdout_overlap).Count -ne 0 -or
    @($selection.training_holdout_work_overlap).Count -ne 0 -or
    @($provenance.files).Count -ne
        (2 * [int]$selection.selected_pair_count) -or
    $plan.model_outputs_used_for_selection -ne $false -or
    $plan.release_evaluation_authorized -ne $false
) {
    throw "Boundary-filtered training failed its isolation gate"
}

[ordered]@{
    state = "completed"
    selected_pair_count = [int]$selection.selected_pair_count
    selected_work_count = [int]$selection.selected_work_count
    excluded_out_of_boundary_pair_count = @(
        $selection.excluded_out_of_boundary_pair_ids
    ).Count
    selection_sha256 = (
        Get-FileHash -LiteralPath $selectionPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    partition_plan_sha256 = (
        Get-FileHash -LiteralPath $partitionPlan -Algorithm SHA256
    ).Hash.ToLowerInvariant()
} | ConvertTo-Json -Depth 4
