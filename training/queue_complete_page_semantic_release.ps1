param(
    [int[]]$WaitPid = @(),
    [string]$WaitPidCsv = ""
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$statusPath = Join-Path $projectDir (
    "training_data\logs\complete-page-semantic-pipeline-current.json"
)
$trainingModelDir = Join-Path $projectDir (
    "training_data\models\" +
    "muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730"
)
$initialModel = Join-Path $projectDir (
    "training_data\models\" +
    "muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729\" +
    "model.best.pt"
)
$candidateReport = Join-Path $projectDir (
    "training_data\release_candidates\" +
    "semantic-detector-muse-v4-complete-page-e12-20260730\" +
    "release-candidate.json"
)

function Set-PipelineStatus {
    param(
        [Parameter(Mandatory = $true)]
        [string]$State,
        [Parameter(Mandatory = $true)]
        [string]$Stage,
        [string]$Failure = ""
    )
    $payload = [ordered]@{
        format = 1
        name = "scorescan-complete-page-semantic-release-pipeline-v1"
        state = $State
        stage = $Stage
        updated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        failure = $Failure
    }
    $temporary = "$statusPath.$PID.tmp"
    $payload | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

try {
    Set-PipelineStatus -State "waiting" -Stage "prior_work"
    $allWaitPids = @($WaitPid)
    if (-not [string]::IsNullOrWhiteSpace($WaitPidCsv)) {
        $allWaitPids += @(
            $WaitPidCsv.Split(",") |
                ForEach-Object { [int]$_.Trim() }
        )
    }
    foreach ($processId in @($allWaitPids | Sort-Object -Unique)) {
        if ($processId -le 0 -or $processId -eq 2147483647) {
            continue
        }
        if ($null -ne (Get-Process -Id $processId -ErrorAction SilentlyContinue)) {
            Wait-Process -Id $processId
        }
    }

    Set-Location -LiteralPath $projectDir
    $env:PYTHONPATH = (
        (Join-Path $projectDir "app\src"),
        $projectDir
    ) -join [IO.Path]::PathSeparator

    Set-PipelineStatus -State "running" -Stage "complete_page_data"
    & (Join-Path $PSScriptRoot "prepare_muse_scan_datasets_v2.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "complete-page Muse data preparation failed"
    }
    & (Join-Path $PSScriptRoot "queue_lieder_1091_semantic_preparation.ps1") `
        -WaitPid 2147483647
    if ($LASTEXITCODE -ne 0) {
        throw "complete-page Lieder replay preparation failed"
    }
    & (Join-Path $PSScriptRoot "prepare_overlap_consistent_semantic_datasets.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "complete-page Lieder replay expansion failed"
    }
    & $python -m app.tools.audit_active_semantic_dataset_quarantine `
        --output (
            Join-Path $projectDir (
                "training_data\benchmarks\" +
                "active_semantic_dataset_diagnostic_quarantine_v1.json"
            )
        )
    if ($LASTEXITCODE -ne 0) {
        throw "active complete-page semantic dataset quarantine failed"
    }
    if (-not (Test-Path -LiteralPath $initialModel -PathType Leaf)) {
        throw "evaluated legacy scan checkpoint is missing: $initialModel"
    }
    & (Join-Path $PSScriptRoot "assert_training_storage.ps1") `
        -Stage "complete-page-semantic-detector-v4" `
        -MinimumReserveGiB 4.0 `
        -RequiredNewArtifactGiB 1.5 `
        -Output (
            Join-Path $projectDir (
                "training_data\logs\" +
                "storage-capacity-complete-page-semantic-v4.json"
            )
        )
    if ($LASTEXITCODE -ne 0) {
        throw "complete-page semantic storage capacity gate failed"
    }

    Set-PipelineStatus -State "running" -Stage "gpu_training"
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_muse_scan_semantic_detector_full.sh `
        /workspace/pigeon-score-scan/training_data/models/muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730 `
        /workspace/pigeon-score-scan/training_data/models/muse-omr-scan-semantic-detector-v3-stratified-replay-e8-b2-20260729/model.best.pt
    if ($LASTEXITCODE -ne 0) {
        throw "complete-page semantic GPU training failed its local gates"
    }
    $trainingReportPath = Join-Path $trainingModelDir "training_report.json"
    $trainingReport = Get-Content `
        -LiteralPath $trainingReportPath `
        -Raw `
        -Encoding UTF8 |
        ConvertFrom-Json
    if (
        $trainingReport.acceptance.passed -ne $true -or
        $trainingReport.priority_selection_protocol -ne
            "support-filtered-priority-macro-map@1"
    ) {
        throw "complete-page semantic training report is not accepted"
    }

    Set-PipelineStatus -State "running" -Stage "independent_holdout"
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_muse_holdout_evaluation.sh
    if ($LASTEXITCODE -ne 0) {
        throw "complete-page independent semantic holdout failed"
    }
    $holdout = Get-Content `
        -LiteralPath (
            Join-Path $trainingModelDir "evaluation.independent-muse-holdout.json"
        ) `
        -Raw `
        -Encoding UTF8 |
        ConvertFrom-Json
    if (
        $holdout.acceptance.passed -ne $true -or
        $holdout.priority_selection_protocol -ne
            "support-filtered-priority-macro-map@1"
    ) {
        throw "complete-page semantic holdout report is not accepted"
    }

    Set-PipelineStatus -State "running" -Stage "isolated_candidate"
    & (Join-Path $PSScriptRoot "queue_semantic_detector_release_candidate.ps1") `
        -WaitSemanticHoldoutPid 2147483647 `
        -WaitGpuReleasePid 2147483647
    if ($LASTEXITCODE -ne 0) {
        throw "complete-page semantic candidate packaging failed"
    }
    if (-not (Test-Path -LiteralPath $candidateReport -PathType Leaf)) {
        throw "complete-page semantic candidate report is missing"
    }
    & $python -m app.tools.assert_high_value_gpu_stage `
        --stage semantic_detector_and_holdout `
        --project-root $projectDir `
        --output-report (
            Join-Path $projectDir (
                "training_data\benchmarks\" +
                "complete-page-semantic-stage-accepted-v1.json"
            )
        )
    if ($LASTEXITCODE -ne 0) {
        throw "complete-page semantic final stage assertion failed"
    }
    Set-PipelineStatus -State "completed" -Stage "completed"
} catch {
    Set-PipelineStatus `
        -State "failed" `
        -Stage "failed" `
        -Failure $_.Exception.Message
    throw
}
