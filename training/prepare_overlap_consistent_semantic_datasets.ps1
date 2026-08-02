$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$transformationVersion = "complete-page-overlap-consistent-deduplicated-semantic-targets@4"
$visibilityVersion = "complete-page-oversized-axis-overlap-fragments@1"
$preparedRoot = [IO.Path]::GetFullPath(
    (Join-Path $projectDir "training_data\prepared")
)

function Invoke-OverlapExpansion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Output,
        [string[]]$Splits = @("train", "calibration", "test")
    )

    $sourceManifest = Join-Path $Source "manifest.json"
    $outputReport = Join-Path $Output "prepare-report.json"
    if (-not (Test-Path -LiteralPath $sourceManifest -PathType Leaf)) {
        throw "Semantic source manifest is missing: $sourceManifest"
    }
    $stale = $false
    if (Test-Path -LiteralPath $outputReport -PathType Leaf) {
        try {
            $existing = Get-Content -LiteralPath $outputReport -Raw -Encoding UTF8 |
                ConvertFrom-Json
            $sourceHash = (
                Get-FileHash -LiteralPath $sourceManifest -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            $stale = (
                $existing.transformation_version -ne $transformationVersion -or
                $existing.source_prepared_manifest_sha256 -ne $sourceHash -or
                $existing.oversized_fragment_visibility_version -ne
                    $visibilityVersion
            )
        } catch {
            $stale = $true
        }
    }
    if ($stale) {
        $resolvedOutput = [IO.Path]::GetFullPath($Output)
        $preparedPrefix = $preparedRoot.TrimEnd("\") + "\"
        if (
            -not $resolvedOutput.StartsWith(
                $preparedPrefix,
                [StringComparison]::OrdinalIgnoreCase
            ) -or
            $resolvedOutput -eq $preparedRoot
        ) {
            throw "Refusing unsafe stale semantic output removal: $resolvedOutput"
        }
        Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath $outputReport -PathType Leaf)) {
        $arguments = @(
            "-m", "app.tools.expand_overlapping_semantic_targets",
            "--prepared-dir", $Source,
            "--output-dir", $Output
        )
        foreach ($split in $Splits) {
            $arguments += @("--split", $split)
        }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Overlap-consistent semantic expansion failed: $Source"
        }
    }
    $report = Get-Content -LiteralPath $outputReport -Raw -Encoding UTF8 |
        ConvertFrom-Json
    $sourceHash = (
        Get-FileHash -LiteralPath $sourceManifest -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if (
        $report.transformation_version -ne $transformationVersion -or
        $report.source_prepared_manifest_sha256 -ne $sourceHash -or
        $report.oversized_fragment_visibility_version -ne $visibilityVersion
    ) {
        throw "Overlap-consistent semantic report is stale: $Output"
    }
    foreach ($split in $Splits) {
        $splitPath = Join-Path $Output "$split.jsonl"
        $splitReport = $report.splits.$split
        if (
            -not (Test-Path -LiteralPath $splitPath -PathType Leaf) -or
            [int]$splitReport.rows -le 0 -or
            [int]$splitReport.unique_objects -le 0 -or
            $null -eq $splitReport.duplicate_source_objects_removed -or
            [int]$splitReport.additional_target_instances -le 0
        ) {
            throw "Overlap-consistent semantic split is incomplete: $splitPath"
        }
    }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Preparation Python is missing: $python"
}
Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = (
    (Join-Path $projectDir "app\src"),
    $projectDir
) -join [IO.Path]::PathSeparator

Invoke-OverlapExpansion `
    -Source (Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_svg_regions_complete_page_v2") `
    -Output (Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_svg_regions_complete_page_overlap_consistent_deduplicated_v4")

$liederPrepared = Join-Path $projectDir (
    "training_data\prepared\" +
    "openscore_lieder_train_1091_svg_regions_complete_page_" +
    "overlap_consistent_deduplicated_v4"
)
$liederAudit = Join-Path $projectDir (
    "training_data\benchmarks\" +
    "openscore-lieder-complete-page-target-audit-v1.json"
)
& $python -m app.tools.audit_complete_page_semantic_targets `
    --prepared-dir $liederPrepared `
    --output $liederAudit `
    --split train `
    --split calibration `
    --split test
if ($LASTEXITCODE -ne 0) {
    throw "OpenScore Lieder complete-page target audit failed"
}
$liederAuditPayload = Get-Content -LiteralPath $liederAudit -Raw -Encoding UTF8 |
    ConvertFrom-Json
if (
    $liederAuditPayload.audit_version -ne
        "complete-page-semantic-target-audit@2" -or
    $liederAuditPayload.passed -ne $true -or
    $liederAuditPayload.oversized_fragment_visibility_version -ne
        $visibilityVersion
) {
    throw "OpenScore Lieder complete-page target audit evidence is invalid"
}

@{
    state = "overlap_consistent_semantic_datasets_ready"
    transformation_version = $transformationVersion
    active_dataset = $liederPrepared
    target_audit = $liederAudit
} | ConvertTo-Json -Compress
