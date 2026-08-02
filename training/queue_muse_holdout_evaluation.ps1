param(
    [Parameter(Mandatory = $true)]
    [int]$WaitHoldoutPid,
    [Parameter(Mandatory = $true)]
    [int]$WaitScanPipelinePid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$holdoutDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_regions_stratified_complete_page_overlap_consistent_deduplicated_v7"
$holdoutReport = Join-Path $holdoutDir "prepare-report.json"
$holdoutFailure = Join-Path $holdoutDir "preparation.failed.txt"
$modelDir = Join-Path $projectDir "training_data\models\muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730"
$trainingReport = Join-Path $modelDir "training_report.json"

foreach ($processId in @($WaitHoldoutPid, $WaitScanPipelinePid)) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Wait-Process -Id $processId
    }
}
if (Test-Path -LiteralPath $holdoutFailure -PathType Leaf) {
    throw "Independent Muse OMR holdout preparation failed"
}
if (-not (Test-Path -LiteralPath $holdoutReport -PathType Leaf)) {
    throw "Independent Muse OMR holdout report is missing"
}
if (-not (Test-Path -LiteralPath $trainingReport -PathType Leaf)) {
    throw "Scan-degraded semantic detector training report is missing"
}
$training = Get-Content -LiteralPath $trainingReport -Raw | ConvertFrom-Json
if ($training.acceptance.passed -ne $true) {
    throw "Scan-degraded semantic detector failed its in-domain training test gate"
}

Set-Location -LiteralPath $projectDir
& wsl.exe -d Ubuntu-24.04 -- bash `
    /workspace/pigeon-score-scan/training/run_wsl_muse_holdout_evaluation.sh
if ($LASTEXITCODE -ne 0) {
    throw "Independent Muse OMR holdout evaluation failed with code $LASTEXITCODE"
}
