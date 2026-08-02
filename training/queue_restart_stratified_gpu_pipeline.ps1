param(
    [Parameter(Mandatory = $true)]
    [int]$WaitLegacyPipelinePid,
    [Parameter(Mandatory = $true)]
    [int]$WaitPartitionPid,
    [Parameter(Mandatory = $true)]
    [int]$WaitCpuPreparationPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$partitionSelection = Join-Path $projectDir (
    "training_data\external\training\muse_omr_scan_train_stratified_v2\" +
    "selection.json"
)
$pipeline = Join-Path $PSScriptRoot (
    "queue_gpu_benchmark_and_detector_pipeline.ps1"
)

foreach ($processId in @(
    $WaitLegacyPipelinePid,
    $WaitPartitionPid,
    $WaitCpuPreparationPid
)) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Wait-Process -Id $processId
    }
}
if (-not (Test-Path -LiteralPath $partitionSelection -PathType Leaf)) {
    throw "Stratified partition did not complete: $partitionSelection"
}
$selection = Get-Content `
    -LiteralPath $partitionSelection `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
if (
    $selection.partition_contract_version -ne
        "muse-omr-boundary-filter@1" -or
    [int]$selection.selected_work_count -lt 200 -or
    @($selection.training_holdout_work_overlap).Count -ne 0
) {
    throw "Stratified partition failed the GPU pipeline isolation gate"
}
foreach ($required in @(
    "training_data\prepared\muse_omr_scan_regions_stratified_v3\prepare-report.json",
    "training_data\prepared\muse_omr_scan_holdout_regions_stratified_v3\prepare-report.json",
    "training_data\prepared\muse_omr_scan_regions_stratified_overlap_consistent_deduplicated_v5\prepare-report.json",
    "training_data\prepared\muse_omr_scan_holdout_regions_stratified_overlap_consistent_deduplicated_v5\prepare-report.json"
)) {
    $path = Join-Path $projectDir $required
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Stratified semantic preparation is incomplete: $path"
    }
}

Set-Location -LiteralPath $projectDir
& powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File $pipeline `
    -WaitPid 2147483647
if ($LASTEXITCODE -ne 0) {
    throw "Stratified GPU semantic pipeline failed with code $LASTEXITCODE"
}
