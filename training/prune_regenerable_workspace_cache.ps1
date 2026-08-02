param(
    [switch]$Execute,
    [int]$RetryGenerationsToKeep = 2,
    [int]$MinimumRetryLogAgeMinutes = 30,
    [string]$ReportPath = ""
)

$ErrorActionPreference = "Stop"
$projectDir = (Split-Path -Parent $PSScriptRoot)
if ($RetryGenerationsToKeep -lt 1) {
    throw "RetryGenerationsToKeep must be at least 1"
}
if ($MinimumRetryLogAgeMinutes -lt 0) {
    throw "MinimumRetryLogAgeMinutes must not be negative"
}
if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $ReportPath = Join-Path `
        $projectDir `
        "training_data\logs\regenerable-cache-prune-current.json"
}
$targets = @(
    "app\.pytest_cache",
    "tmp\replaced-portable-20260727-1348",
    "tmp\testspace-previous-20260727-1539",
    "tmp\testspace-backup-before-gpu-ready-fix-20260727-1656",
    "tmp\testspace-backup-before-output-fix-20260727-1645",
    "tmp\release-rotation-disabled",
    "tmp\release-hot-switch-20260727",
    "tmp\release-output-final-20260727",
    "tmp\release-output-fix-20260727",
    "tmp\registration-worker-benchmark-20260728",
    "training_data\prepared\openscore_lieder_train_320_svg_regions_v1",
    "training_data\prepared\muse_omr_scan_holdout_detection_labels_v1.stale-20260729-090835-18800",
    "training_data\prepared\muse_omr_scan_text_v2.contaminated-crops-20260729-0600",
    "training_data\prepared\muse_omr_scan_text_v2.contaminated-visual-presence-20260729-0705",
    "training_data\prepared\muse_omr_scan_text_v2.stale-20260729-073526-45464",
    "training_data\prepared\muse_omr_scan_text_v2.stale-pre-atomic-page-20260729-0730",
    "training_data\prepared\muse_omr_scan_text_v2.stale-pre-occupancy-20260729-0720",
    "training_data\prepared\muse_omr_scan_text_v2.stale-pre-page-contract-20260729-0740",
    "training_data\prepared\openscore_quartet_lieder_semantic_v1.failed-box-audit-20260728-045627",
    "training_data\prepared\openscore_quartet_lieder_semantic_v1.interrupted-20260728-045706",
    "training_data\models\deepscores-symbol-detector-smoke-v3-6",
    "training_data\models\deepscores-symbol-detector-capacity-b2-v3",
    "training_data\models\zeus-olimpic-real-full-calibration-baseline-gpu-20260727\weights.h5",
    "training_data\models\zeus-olimpic-real-full-calibration-baseline-gpu-20260727\weights.best.h5",
    "training_data\models\zeus-olimpic-real-smoke-gpu-py310-r3-20260727\weights.h5",
    "training_data\models\zeus-olimpic-real-smoke-gpu-py310-r3-20260727\weights.best.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r1-20260727\weights.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r1-20260727\weights.best.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r2-20260727\weights.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r2-20260727\weights.best.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r5-fp32-b4-20260727\weights.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r5-fp32-b4-20260727\weights.best.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r6-fp32-b8-20260727\weights.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r6-fp32-b8-20260727\weights.best.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r7-fp32-b16-20260727\weights.h5",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r7-fp32-b16-20260727\weights.best.h5",
    "training_data\models\zeus-olimpic-real-replay-e2-lr1e5-20260727\weights.best.h5",
    "training_data\models\zeus-olimpic-real-smoke-gpu-20260727",
    "training_data\models\zeus-olimpic-real-smoke-gpu-py310-20260727",
    "training_data\models\zeus-olimpic-real-smoke-gpu-py310-r2-20260727",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r3-mp-20260727",
    "training_data\models\zeus-olimpic-real-train-smoke-gpu-r4-mp-20260727",
    "training_data\models\zeus-olimpic-real-baseline-gpu-r5-mp-20260727",
    "training_data\models\zeus-olimpic-real-baseline-gpu-r6-mp-20260727",
    "training_data\models\zeus-olimpic-real-baseline-gpu-r7-mp-20260727",
    "training_data\models\zeus-olimpic-real-baseline-gpu-r8-mp-20260727",
    "training_data\external\archives\olimpic-1.0-sources-for-scanned.2024-02-12.tar.gz.tgz",
    "training_data\external\archives\olimpic-1.0-synthetic.2024-02-12.tar.gz",
    "training_data\external\archives\olimpic-1.0-scanned.2024-02-12.tar.gz",
    "training_data\external\archives\ds2_dense.tar.gz",
    "training_data\external\archives\openscore-lieder-6b2dc542.zip",
    "training_data\external\archives\DoReMi_v1.zip",
    "training_data\external\archives\grandstaff-lmx.2024-02-12.tar.gz",
    "training_data\external\archives\openscore-string-quartets-d13289cd.zip",
    "training_data\external\archives\zeus-olimpic-1.0-2024-02-12.model.tar.gz"
)

$projectPrefix = $projectDir.TrimEnd("\") + "\"
$rows = @()
$totalBytes = [long]0
foreach ($relative in $targets) {
    $candidate = Join-Path $projectDir $relative
    if (-not (Test-Path -LiteralPath $candidate)) {
        continue
    }
    $item = Get-Item -LiteralPath $candidate -Force
    $resolved = $item.FullName
    if (
        -not $resolved.StartsWith(
            $projectPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        $resolved -eq $projectDir -or
        ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)
    ) {
        throw "Unsafe cache-prune target: $resolved"
    }
    $bytes = if ($item.PSIsContainer) {
        (
            Get-ChildItem -LiteralPath $resolved -Recurse -File -Force |
                Measure-Object -Property Length -Sum
        ).Sum
    } else {
        $item.Length
    }
    if ($null -eq $bytes) {
        $bytes = 0
    }
    $totalBytes += [long]$bytes
    $rows += [ordered]@{
        kind = "named_regenerable_cache"
        relative_path = $relative
        resolved_path = $resolved
        bytes = [long]$bytes
    }
}

$logsDir = Join-Path $projectDir "training_data\logs"
if (Test-Path -LiteralPath $logsDir -PathType Container) {
    $retryRows = @(
        Get-ChildItem -LiteralPath $logsDir -File |
            ForEach-Object {
                if (
                    $_.Name -match
                        "(?:\.retry|-r)(?<generation>[0-9]+)(?:-|\.|$)"
                ) {
                    [pscustomobject]@{
                        item = $_
                        generation = [int]$Matches.generation
                    }
                }
            }
    )
    $keptGenerations = @(
        $retryRows |
            Select-Object -ExpandProperty generation -Unique |
            Sort-Object -Descending |
            Select-Object -First $RetryGenerationsToKeep
    )
    $oldestAllowedWriteTime = (Get-Date).AddMinutes(
        -$MinimumRetryLogAgeMinutes
    )
    foreach ($retry in $retryRows) {
        if (
            $keptGenerations -contains $retry.generation -or
            $retry.item.LastWriteTime -gt $oldestAllowedWriteTime
        ) {
            continue
        }
        $resolved = $retry.item.FullName
        if (
            -not $resolved.StartsWith(
                $projectPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            ) -or
            $retry.item.PSIsContainer -or
            ($retry.item.Attributes -band
                [System.IO.FileAttributes]::ReparsePoint)
        ) {
            throw "Unsafe retry-log prune target: $resolved"
        }
        $bytes = [long]$retry.item.Length
        $totalBytes += $bytes
        $rows += [ordered]@{
            kind = "superseded_retry_log"
            relative_path = (
                $resolved.Substring($projectPrefix.Length)
            )
            resolved_path = $resolved
            bytes = $bytes
            retry_generation = $retry.generation
        }
    }
}

$removedBytes = [long]0
$skippedBytes = [long]0
if ($Execute) {
    foreach ($row in $rows) {
        try {
            Remove-Item -LiteralPath $row.resolved_path -Recurse -Force
            $row.deletion_state = "removed"
            $removedBytes += [long]$row.bytes
        } catch [System.IO.IOException] {
            $row.deletion_state = "skipped_in_use"
            $row.deletion_error = $_.Exception.Message
            $skippedBytes += [long]$row.bytes
        } catch [System.UnauthorizedAccessException] {
            $row.deletion_state = "skipped_in_use"
            $row.deletion_error = $_.Exception.Message
            $skippedBytes += [long]$row.bytes
        }
    }
} else {
    foreach ($row in $rows) {
        $row.deletion_state = "planned"
    }
}

$report = [ordered]@{
    format = 1
    name = "scorescan-regenerable-workspace-cache-prune-v3"
    generated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    executed = [bool]$Execute
    retry_generations_to_keep = $RetryGenerationsToKeep
    minimum_retry_log_age_minutes = $MinimumRetryLogAgeMinutes
    kept_retry_generations = $keptGenerations
    targets = $rows
    target_count = $rows.Count
    bytes = if ($Execute) { $removedBytes } else { $totalBytes }
    planned_bytes = $totalBytes
    removed_bytes = $removedBytes
    skipped_in_use_bytes = $skippedBytes
}
$reportJson = $report | ConvertTo-Json -Depth 5
$reportParent = Split-Path -Parent $ReportPath
New-Item -ItemType Directory -Path $reportParent -Force | Out-Null
$temporaryReport = Join-Path `
    $reportParent `
    ("." + (Split-Path -Leaf $ReportPath) + "." + $PID + ".tmp")
try {
    $reportJson |
        Set-Content -LiteralPath $temporaryReport -Encoding UTF8
    Move-Item -LiteralPath $temporaryReport -Destination $ReportPath -Force
} finally {
    Remove-Item `
        -LiteralPath $temporaryReport `
        -Force `
        -ErrorAction SilentlyContinue
}
$reportJson
