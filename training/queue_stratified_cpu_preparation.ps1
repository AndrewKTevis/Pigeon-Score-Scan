param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPartitionPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$powershell = Join-Path $PSHOME "powershell.exe"
$childRunner = Join-Path $PSScriptRoot "run_preparation_child.ps1"
$logDir = Join-Path $projectDir "training_data\logs"
$selection = Join-Path $projectDir (
    "training_data\external\training\muse_omr_scan_train_stratified_v2\" +
    "selection.json"
)

$partition = Get-Process -Id $WaitPartitionPid -ErrorAction SilentlyContinue
if ($null -ne $partition) {
    Wait-Process -Id $WaitPartitionPid
}
if (-not (Test-Path -LiteralPath $selection -PathType Leaf)) {
    throw "Stratified training partition is unavailable: $selection"
}

function Start-Preparation {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Script,
        [string[]]$Arguments = @()
    )
    $scriptPath = Join-Path $PSScriptRoot $Script
    $completionPath = Join-Path $logDir "$Name.completion.json"
    Remove-Item `
        -LiteralPath $completionPath `
        -Force `
        -ErrorAction SilentlyContinue
    $tokens = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $childRunner,
        "-TargetScript",
        $scriptPath,
        "-CompletionPath",
        $completionPath
    ) + @($Arguments)
    $argumentText = (
        $tokens |
            ForEach-Object {
                '"' + ([string]$_).Replace('"', '\"') + '"'
            }
    ) -join " "
    $process = Start-Process `
        -FilePath $powershell `
        -ArgumentList $argumentText `
        -WorkingDirectory $projectDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir "$Name.out.log") `
        -RedirectStandardError (Join-Path $logDir "$Name.err.log") `
        -PassThru
    return [pscustomobject]@{
        Name = $Name
        Process = $process
        CompletionPath = $completionPath
    }
}

$regions = Start-Preparation `
    -Name "muse-stratified-regions-v3" `
    -Script "prepare_muse_scan_datasets_v2.ps1"
$scanText = Start-Preparation `
    -Name "muse-stratified-text-v3" `
    -Script "queue_muse_scan_text_preparation.ps1"
$holdoutText = Start-Preparation `
    -Name "muse-stratified-holdout-text-v3" `
    -Script "queue_muse_holdout_text_preparation.ps1" `
    -Arguments @(
        "-WaitHoldoutRegionPid",
        $regions.Process.Id.ToString()
    )
$holdoutLabels = Start-Preparation `
    -Name "muse-stratified-holdout-labels-v2" `
    -Script "queue_muse_holdout_detection_labels.ps1" `
    -Arguments @(
        "-WaitHoldoutTextPid",
        $holdoutText.Process.Id.ToString()
    )
$exhaustive = @(
    foreach ($shard in 0..3) {
        Start-Preparation `
            -Name ("muse-stratified-exhaustive-v2-shard-{0:D2}" -f $shard) `
            -Script "queue_muse_exhaustive_detection_preparation.ps1" `
            -Arguments @(
                "-ShardIndex",
                $shard.ToString(),
                "-ShardCount",
                "4"
            )
    }
)
$exhaustiveMerge = Start-Preparation `
    -Name "muse-stratified-exhaustive-merge-v4" `
    -Script "queue_merge_fully_exhaustive_detection_labels.ps1"

$processes = @(
    $regions,
    $scanText,
    $holdoutText,
    $holdoutLabels
) + @($exhaustive) + @($exhaustiveMerge)
foreach ($item in $processes) {
    $item.Process.WaitForExit()
    if (
        -not (
            Test-Path `
                -LiteralPath $item.CompletionPath `
                -PathType Leaf
        )
    ) {
        throw (
            "Stratified CPU preparation child did not write a completion " +
            "record: $($item.Name), PID $($item.Process.Id)"
        )
    }
    $completion = Get-Content `
        -LiteralPath $item.CompletionPath `
        -Raw `
        -Encoding UTF8 |
        ConvertFrom-Json
    if (
        [int]$completion.format -ne 1 -or
        [int]$completion.exit_code -ne 0
    ) {
        throw (
            "Stratified CPU preparation failed: $($item.Name), " +
            "PID $($item.Process.Id), error $($completion.error)"
        )
    }
}

foreach ($required in @(
    "training_data\prepared\muse_omr_scan_regions_stratified_v3\prepare-report.json",
    "training_data\prepared\muse_omr_scan_holdout_regions_stratified_v3\prepare-report.json",
    "training_data\prepared\muse_omr_scan_text_stratified_v3\prepare-report.json",
    "training_data\prepared\muse_omr_scan_holdout_text_stratified_v3\prepare-report.json",
    "training_data\prepared\muse_omr_scan_holdout_detection_labels_stratified_v2\prepare-report.json",
    "training_data\prepared\scorescan_ocr_detection_stratified_v4\merge-report.json"
)) {
    $path = Join-Path $projectDir $required
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Stratified CPU preparation output is missing: $path"
    }
}

[ordered]@{
    state = "completed"
    region_pid = $regions.Process.Id
    scan_text_pid = $scanText.Process.Id
    holdout_text_pid = $holdoutText.Process.Id
    holdout_labels_pid = $holdoutLabels.Process.Id
    exhaustive_pids = @(
        $exhaustive |
            ForEach-Object { $_.Process.Id }
    )
    exhaustive_merge_pid = $exhaustiveMerge.Process.Id
} | ConvertTo-Json -Depth 4
