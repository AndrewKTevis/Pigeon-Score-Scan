param(
    [Parameter(Mandatory = $true)]
    [int]$WaitAcquisitionPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$datasetDir = Join-Path $projectDir "training_data\external\training\muse_omr_scan_train_stratified_v2"
$provenance = Join-Path $datasetDir "provenance.json"
$holdoutSelectionPath = Join-Path $projectDir "training_data\external\benchmarks\muse_omr_e27f6a8634_raremarks_v3\selection.json"
$preparedDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_regions_stratified_v3"
$preparedReport = Join-Path $preparedDir "prepare-report.json"
$liederModelDir = Join-Path $projectDir "training_data\models\openscore-lieder-semantic-detector-e6-b2-20260728"
$liederReport = Join-Path $liederModelDir "training_report.json"
$liederModel = Join-Path $liederModelDir "model.best.pt"
$scanModelDir = Join-Path $projectDir "training_data\models\muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729"
$museScore = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
$minimumAcceptedTrainingWorks = 170

$acquisitionProcess = Get-Process -Id $WaitAcquisitionPid -ErrorAction SilentlyContinue
if ($null -ne $acquisitionProcess) {
    Wait-Process -Id $WaitAcquisitionPid
}
if (-not (Test-Path -LiteralPath $provenance -PathType Leaf)) {
    throw "Muse OMR acquisition did not produce verified provenance: $provenance"
}
if (-not (Test-Path -LiteralPath $holdoutSelectionPath -PathType Leaf)) {
    throw "Muse OMR augmented holdout selection is missing: $holdoutSelectionPath"
}
$acquisition = Get-Content -LiteralPath $provenance -Raw | ConvertFrom-Json
$expectedPairs = [int]$acquisition.selected_pair_count
$expectedWorks = [int]$acquisition.selected_work_count
$holdoutSelection = Get-Content -LiteralPath $holdoutSelectionPath -Raw |
    ConvertFrom-Json
$holdoutPairSet = [Collections.Generic.HashSet[int]]::new(
    [int[]]@($holdoutSelection.selected_pair_ids)
)
$holdoutWorkSet = [Collections.Generic.HashSet[string]]::new(
    [string[]]@($holdoutSelection.selected_work_fingerprints)
)
$actualPairOverlap = @(
    @($acquisition.selected_pair_ids) |
        Where-Object { $holdoutPairSet.Contains([int]$_) }
)
$actualWorkOverlap = @(
    @($acquisition.selected_work_fingerprints) |
        Where-Object { $holdoutWorkSet.Contains([string]$_) }
)
if ($acquisition.role -ne "external_scan_degraded_training_only" -or
    $acquisition.source_image_origin -ne
        "synthetic_scan_degraded_render" -or
    $acquisition.production_evidence_eligible -ne $false -or
    $holdoutSelection.source_image_origin -ne
        "synthetic_scan_degraded_render" -or
    $holdoutSelection.production_evidence_eligible -ne $false -or
    $acquisition.partition_contract_version -ne "muse-omr-boundary-filter@1" -or
    @($acquisition.training_holdout_overlap).Count -ne 0 -or
    @($acquisition.training_holdout_work_overlap).Count -ne 0 -or
    $actualPairOverlap.Count -ne 0 -or
    $actualWorkOverlap.Count -ne 0 -or
    $expectedPairs -lt 200 -or
    $expectedWorks -lt 200 -or
    [int]$holdoutSelection.selected_pair_count -ne 435 -or
    [int]$holdoutSelection.selected_work_count -ne 395 -or
    @($acquisition.files).Count -ne (2 * $expectedPairs)) {
    throw "Muse OMR acquisition provenance failed the isolation/completeness gate"
}

Set-Location -LiteralPath $projectDir
if (-not (Test-Path -LiteralPath $preparedReport -PathType Leaf)) {
    & $python -m app.tools.prepare_muse_omr_scan_regions `
        --dataset-dir $datasetDir `
        --output-dir $preparedDir `
        --musescore-exe $museScore `
        --negative-ratio 0.08 `
        --minimum-ecc 0.86 `
        --maximum-linear-deviation 0.12 `
        --maximum-translation-fraction 0.10 `
        --minimum-local-correlation-10p 0.62 `
        --minimum-median-local-correlation 0.72 `
        --minimum-accepted-page-fraction 0.75 `
        --minimum-accepted-fraction 0.50 `
        --minimum-accepted-works $minimumAcceptedTrainingWorks `
        --registration-workers 4 `
        --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Registered Muse OMR scan preparation failed with exit code $LASTEXITCODE"
    }
}
$preparation = Get-Content -LiteralPath $preparedReport -Raw | ConvertFrom-Json
if (
    $preparation.registration_version -ne "muse-omr-bounded-elastic-page-filter-jpeg95@7" -or
    [int]$preparation.selected_pairs -ne $expectedPairs -or
    [int]$preparation.selected_works -ne $expectedWorks -or
    [int]$preparation.accepted_works -lt $minimumAcceptedTrainingWorks -or
    @($preparation.forbidden_selection_overlap).Count -ne 0 -or
    @($preparation.forbidden_work_overlap).Count -ne 0
) {
    throw "Muse OMR v2 training preparation failed coverage/isolation"
}

while (-not (Test-Path -LiteralPath $liederReport -PathType Leaf)) {
    Start-Sleep -Seconds 20
}
if (-not (Test-Path -LiteralPath $liederModel -PathType Leaf)) {
    throw "Lieder semantic detector report exists but its best model is missing"
}

& wsl.exe -d Ubuntu-24.04 -- bash `
    /workspace/pigeon-score-scan/training/run_wsl_muse_scan_semantic_detector_full.sh `
    /workspace/pigeon-score-scan/training_data/models/muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729 `
    /workspace/pigeon-score-scan/training_data/models/openscore-lieder-semantic-detector-e6-b2-20260728/model.best.pt
if ($LASTEXITCODE -ne 0) {
    throw "Muse OMR scan semantic detector training failed with exit code $LASTEXITCODE"
}
