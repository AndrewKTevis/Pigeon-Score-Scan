param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,
    [string]$CaseId = "pdmx-imslp-167583"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "app\.venv\Scripts\python.exe"
$manifest = Join-Path $projectRoot `
    "training_data\benchmarks\pdmx_imslp_semantic_v1\semantic_manifest.json"
$output = Join-Path $projectRoot `
    "training_data\benchmarks\pdmx_imslp_real_scan_semantic_run_v1"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "ScoreScan Python environment is missing"
}
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "PDMX semantic manifest is missing"
}

$process = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
if ($null -ne $process) {
    Write-Output "Waiting for real-scan benchmark PID $WaitPid"
    Wait-Process -Id $WaitPid
}

Set-Location -LiteralPath $projectRoot
& $python -u `
    "app\tools\run_openscore_real_scan_semantic_benchmark.py" `
    $manifest `
    $projectRoot `
    $output `
    --case-id $CaseId `
    --timeout-seconds 14400
if ($LASTEXITCODE -ne 0) {
    throw "PDMX real-scan semantic benchmark failed with $LASTEXITCODE"
}
