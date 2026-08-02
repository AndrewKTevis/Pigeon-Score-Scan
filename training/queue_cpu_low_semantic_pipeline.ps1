param(
    [Parameter(Mandatory = $true)]
    [int]$WaitOpenScorePid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$openScoreReport = Join-Path $projectDir "training_data\models\openscore-semantic-detector-e6-b2-20260728\training_report.json"
$liederReport = Join-Path $projectDir "training_data\models\openscore-lieder-semantic-detector-e6-b2-20260728\training_report.json"
$museReport = Join-Path $projectDir "training_data\models\muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729\training_report.json"
$holdoutReport = Join-Path $projectDir "training_data\models\muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729\evaluation.independent-muse-holdout.json"
$logDir = Join-Path $projectDir "training_data\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$openScoreProcess = Get-Process -Id $WaitOpenScorePid -ErrorAction SilentlyContinue
if ($null -ne $openScoreProcess) {
    Wait-Process -Id $WaitOpenScorePid
}
if (-not (Test-Path -LiteralPath $openScoreReport -PathType Leaf)) {
    throw "Low-CPU OpenScore training ended without a completed report"
}

$stages = @(
    [ordered]@{
        Name = "lieder"
        Report = $liederReport
        Script = "/workspace/pigeon-score-scan/training/run_wsl_lieder_semantic_detector_full.sh"
    },
    [ordered]@{
        Name = "muse-scan"
        Report = $museReport
        Script = "/workspace/pigeon-score-scan/training/run_wsl_muse_scan_semantic_detector_full.sh"
    },
    [ordered]@{
        Name = "independent-holdout"
        Report = $holdoutReport
        Script = "/workspace/pigeon-score-scan/training/run_wsl_muse_holdout_evaluation.sh"
    }
)

foreach ($stage in $stages) {
    if (Test-Path -LiteralPath $stage.Report -PathType Leaf) {
        continue
    }
    $stdout = Join-Path $logDir ("semantic-cpu-low-" + $stage.Name + ".out.log")
    $stderr = Join-Path $logDir ("semantic-cpu-low-" + $stage.Name + ".err.log")
    $arguments = @(
        "-d", "Ubuntu-24.04", "--", "env",
        "CUDA_VISIBLE_DEVICES=",
        "OMP_NUM_THREADS=2",
        "MKL_NUM_THREADS=2",
        "OPENBLAS_NUM_THREADS=2",
        "NUMEXPR_NUM_THREADS=2",
        "SCORESCAN_TRAIN_DEVICE=cpu",
        "SCORESCAN_CPU_THREADS=2",
        "taskset", "-c", "0-3",
        "nice", "-n", "12",
        "bash", $stage.Script
    )
    $process = Start-Process `
        -FilePath "wsl.exe" `
        -ArgumentList $arguments `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Low-CPU semantic stage $($stage.Name) failed with exit code $($process.ExitCode)"
    }
    if (-not (Test-Path -LiteralPath $stage.Report -PathType Leaf)) {
        throw "Low-CPU semantic stage $($stage.Name) did not produce $($stage.Report)"
    }
}
