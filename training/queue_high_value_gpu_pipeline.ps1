param(
    [int]$WaitPid = 2147483647
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$statusPath = Join-Path $projectDir (
    "training_data\logs\high-value-gpu-pipeline-current.json"
)
$startedAtUtc = [DateTimeOffset]::UtcNow.ToString("o")
$script:completedStages = @()
$script:currentStage = ""

function Write-PipelineStatus {
    param(
        [Parameter(Mandatory = $true)]
        [string]$State,
        [string]$CurrentStage = "",
        [int]$ExitCode = 0,
        [string]$Failure = ""
    )

    $payload = [ordered]@{
        format = 1
        name = "scorescan-high-value-gpu-pipeline-v1"
        state = $State
        started_at_utc = $startedAtUtc
        updated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        current_stage = $CurrentStage
        completed_stages = @($script:completedStages)
        exit_code = $ExitCode
        failure = $Failure
    }
    $temporary = Join-Path (
        Split-Path -Parent $statusPath
    ) ("." + (Split-Path -Leaf $statusPath) + "." + $PID + ".tmp")
    $payload |
        ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Invoke-QueueStage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$ScriptName,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $scriptPath = Join-Path $PSScriptRoot $ScriptName
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
        throw "GPU pipeline stage script is missing: $scriptPath"
    }
    $script:currentStage = $Name
    Write-PipelineStatus -State "running" -CurrentStage $Name
    & powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File $scriptPath `
        @Arguments
    $stageExitCode = $LASTEXITCODE
    if ($stageExitCode -ne 0) {
        throw "GPU pipeline stage $Name failed with exit code $stageExitCode"
    }
    $script:completedStages += $Name
    Write-PipelineStatus -State "stage_completed" -CurrentStage $Name
    $script:currentStage = ""
}

$waitProcess = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
if ($null -ne $waitProcess) {
    Wait-Process -Id $WaitPid
}

Set-Location -LiteralPath $projectDir
try {
    Invoke-QueueStage `
        -Name "semantic_detector_and_holdout" `
        -ScriptName "queue_gpu_benchmark_and_detector_pipeline.ps1" `
        -Arguments @("-WaitPid", "2147483647")
    Invoke-QueueStage `
        -Name "ocr_recognition" `
        -ScriptName "queue_ppocrv6_scorescan_training.ps1" `
        -Arguments @(
            "-WaitBootstrapPid", "2147483647",
            "-WaitGpuPipelinePid", "2147483647"
        )
    Invoke-QueueStage `
        -Name "ocr_detection" `
        -ScriptName "queue_ppocrv6_scorescan_detection.ps1" `
        -Arguments @(
            "-WaitRecognitionPid", "2147483647",
            "-WaitWeightPid", "2147483647",
            "-WaitHoldoutLabelsPid", "2147483647"
        )
    Invoke-QueueStage `
        -Name "ocr_detection_exhaustive" `
        -ScriptName "queue_ppocrv6_exhaustive_detection_training.ps1" `
        -Arguments @("-WaitGpuPid", "2147483647")
    Invoke-QueueStage `
        -Name "semantic_family_priority" `
        -ScriptName "queue_zeus_family_priority_training.ps1" `
        -Arguments @("-WaitGpuPid", "2147483647")
    Write-PipelineStatus -State "completed"
} catch {
    $message = $_.Exception.Message
    Write-PipelineStatus `
        -State "failed" `
        -CurrentStage $script:currentStage `
        -ExitCode 1 `
        -Failure $message
    throw
}
