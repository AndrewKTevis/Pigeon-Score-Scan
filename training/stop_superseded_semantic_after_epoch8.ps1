param(
    [int]$PollSeconds = 20
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 5 -or $PollSeconds -gt 60) {
    throw "PollSeconds must be between 5 and 60"
}

$projectDir = Split-Path -Parent $PSScriptRoot
$logPath = Join-Path $projectDir (
    "training_data\logs\high-value-gpu-pipeline.retry17.out.log"
)
$modelDir = Join-Path $projectDir (
    "training_data\models\" +
    "muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729"
)
$statusPath = Join-Path $projectDir (
    "training_data\logs\superseded-semantic-epoch8-stop.json"
)
$linuxModelDir = (
    "/workspace/pigeon-score-scan/training_data/models/" +
    "muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729"
)

function Write-Status {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.IDictionary]$Payload
    )
    $temporary = "$statusPath.$PID.tmp"
    try {
        $Payload | ConvertTo-Json -Depth 12 |
            Set-Content -LiteralPath $temporary -Encoding UTF8
        Move-Item -LiteralPath $temporary -Destination $statusPath -Force
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

function Read-EpochEightRecord {
    if (-not (Test-Path -LiteralPath $logPath -PathType Leaf)) {
        return $null
    }
    foreach ($line in @(Get-Content -LiteralPath $logPath -Tail 120)) {
        if (-not $line.StartsWith("{")) {
            continue
        }
        try {
            $record = $line | ConvertFrom-Json
        } catch {
            continue
        }
        if (
            [int]$record.epoch -eq 8 -and
            $null -ne $record.test -and
            $null -ne $record.test.map_50
        ) {
            return $record
        }
    }
    return $null
}

function Get-ExactLinuxTrainerProcesses {
    $lines = @(
        & wsl.exe -d Ubuntu-24.04 -- `
            ps -eo pid=,ppid=,args=
    )
    if ($LASTEXITCODE -ne 0) {
        throw "could not inspect the superseded WSL trainer"
    }
    $rows = @()
    foreach ($line in $lines) {
        if (
            $line -notmatch "^\s*(\d+)\s+(\d+)\s+(.*)$" -or
            -not $Matches[3].Contains(
                "train_deepscores_symbol_detector.py"
            ) -or
            -not $Matches[3].Contains($linuxModelDir)
        ) {
            continue
        }
        $rows += [pscustomobject]@{
            pid = [int]$Matches[1]
            parent_pid = [int]$Matches[2]
            command = $Matches[3]
        }
    }
    return @($rows)
}

Write-Status -Payload ([ordered]@{
    format = 1
    name = "scorescan-superseded-semantic-epoch8-stop-v1"
    state = "waiting"
    reason = (
        "legacy owner-tile labels are geometry-corrupted; retain only the " +
        "next evaluated checkpoint as diagnostic initialization evidence"
    )
    updated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
})

while ($true) {
    $record = Read-EpochEightRecord
    if ($null -eq $record) {
        if (@(Get-ExactLinuxTrainerProcesses).Count -eq 0) {
            throw "superseded trainer exited before evaluated epoch 8"
        }
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $bestModel = Join-Path $modelDir "model.best.pt"
    $checkpoint = Join-Path $modelDir "checkpoint.last.pt"
    if (
        -not (Test-Path -LiteralPath $bestModel -PathType Leaf) -or
        -not (Test-Path -LiteralPath $checkpoint -PathType Leaf)
    ) {
        throw "evaluated epoch 8 did not commit its model/checkpoint"
    }
    $processes = @(Get-ExactLinuxTrainerProcesses)
    if ($processes.Count -eq 0) {
        throw "evaluated epoch 8 exists but its exact trainer is absent"
    }
    $processIds = @($processes | Select-Object -ExpandProperty pid)
    $rootIds = @(
        $processes |
            Where-Object { $processIds -notcontains $_.parent_pid } |
            Select-Object -ExpandProperty pid
    )
    if ($rootIds.Count -ne 1) {
        throw "superseded trainer process tree is ambiguous"
    }

    Write-Status -Payload ([ordered]@{
        format = 1
        name = "scorescan-superseded-semantic-epoch8-stop-v1"
        state = "stopping"
        reason = (
            "legacy owner-tile labels are geometry-corrupted; epoch 8 is " +
            "retained only as an evaluated initialization contrast"
        )
        evaluated_epoch = 8
        metrics = $record.test
        model_best_sha256 = (
            Get-FileHash -LiteralPath $bestModel -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        checkpoint_last_sha256 = (
            Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        linux_process_ids = $processIds
        updated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    })

    & wsl.exe -d Ubuntu-24.04 -- kill -TERM $rootIds[0]
    if ($LASTEXITCODE -ne 0) {
        throw "could not terminate the exact superseded trainer root"
    }
    $deadline = (Get-Date).AddSeconds(15)
    do {
        Start-Sleep -Seconds 1
        $remaining = @(Get-ExactLinuxTrainerProcesses)
    } while ($remaining.Count -gt 0 -and (Get-Date) -lt $deadline)
    if ($remaining.Count -gt 0) {
        foreach ($processId in @(
            $remaining |
                Sort-Object parent_pid -Descending |
                Select-Object -ExpandProperty pid
        )) {
            & wsl.exe -d Ubuntu-24.04 -- kill -KILL $processId
            if ($LASTEXITCODE -ne 0) {
                throw "could not stop exact superseded trainer PID $processId"
            }
        }
    }
    Write-Status -Payload ([ordered]@{
        format = 1
        name = "scorescan-superseded-semantic-epoch8-stop-v1"
        state = "completed"
        reason = (
            "legacy owner-tile labels are geometry-corrupted; epoch 8 was " +
            "retained only as an evaluated initialization contrast"
        )
        evaluated_epoch = 8
        metrics = $record.test
        model_best_sha256 = (
            Get-FileHash -LiteralPath $bestModel -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        checkpoint_last_sha256 = (
            Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        stopped_linux_process_ids = $processIds
        updated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    })
    break
}
