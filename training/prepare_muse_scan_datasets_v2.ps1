$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$museScore = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
$trainingDataset = Join-Path $projectDir "training_data\external\training\muse_omr_scan_train_stratified_v2"
$holdoutDataset = Join-Path $projectDir "training_data\external\benchmarks\muse_omr_e27f6a8634_raremarks_v3"
$trainingSelection = Join-Path $trainingDataset "selection.json"
$trainingOutput = Join-Path $projectDir "training_data\prepared\muse_omr_scan_regions_stratified_complete_page_v6"
$holdoutOutput = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_regions_stratified_complete_page_v6"
$trainingConsistentOutput = Join-Path $projectDir "training_data\prepared\muse_omr_scan_regions_stratified_complete_page_overlap_consistent_deduplicated_v7"
$holdoutConsistentOutput = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_regions_stratified_complete_page_overlap_consistent_deduplicated_v7"
$trainingCacheSource = Join-Path $projectDir "training_data\prepared\muse_omr_scan_regions_stratified_v3"
$holdoutCacheSource = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_regions_stratified_v3"
$registrationVersion = "muse-omr-bounded-elastic-page-filter-jpeg95@7"
$targetAssignmentVersion = "complete-page-overlap-consistent-deduplicated-semantic-targets@4"
$visibilityVersion = "complete-page-oversized-axis-overlap-fragments@1"
$targetAuditVersion = "complete-page-semantic-target-audit@2"
$preparedRoot = [IO.Path]::GetFullPath(
    (Join-Path $projectDir "training_data\prepared")
)
$trainingMinimumAcceptedWorks = 170
$holdoutMinimumAcceptedWorks = 200
$replayPrepared = Join-Path $projectDir "training_data\prepared\openscore_quartet_lieder_semantic_v1"
$replayPrepareReport = Join-Path $replayPrepared "merge-report.json"
$liederReplayRoot = Join-Path $projectDir "training_data\external\corpora\openscore_lieder_6b2dc542"
$quartetReplayRoot = Join-Path $projectDir "training_data\external\corpora\openscore_string_quartets_d13289cd"
$replayIsolationReport = Join-Path $projectDir "training_data\diagnostics\semantic-replay-holdout-isolation-v1.json"
$trainingLayoutEvidence = Join-Path $projectDir (
    "training_data\benchmarks\" +
    "muse-training-semantic-page-layout-evidence-v1.json"
)
$holdoutLayoutEvidence = Join-Path $projectDir (
    "training_data\benchmarks\" +
    "muse-holdout-semantic-page-layout-evidence-v1.json"
)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Preparation Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $museScore -PathType Leaf)) {
    throw "MuseScore is missing: $museScore"
}
if (-not (Test-Path -LiteralPath $trainingSelection -PathType Leaf)) {
    throw "Stratified Muse training selection is missing: $trainingSelection"
}
$trainingSelectionPayload = Get-Content `
    -LiteralPath $trainingSelection `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
$trainingSelectedPairs = [int]$trainingSelectionPayload.selected_pair_count
$trainingSelectedWorks = [int]$trainingSelectionPayload.selected_work_count
if (
    $trainingSelectionPayload.partition_contract_version -ne
        "muse-omr-boundary-filter@1" -or
    $trainingSelectedPairs -lt 200 -or
    $trainingSelectedWorks -lt 200 -or
    @($trainingSelectionPayload.training_holdout_overlap).Count -ne 0 -or
    @($trainingSelectionPayload.training_holdout_work_overlap).Count -ne 0
) {
    throw "Stratified Muse training selection failed its isolation gate"
}

Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = (
    (Join-Path $projectDir "app\src"),
    $projectDir
) -join [IO.Path]::PathSeparator

function Assert-ReplayIsolation {
    $holdoutSelection = Join-Path $holdoutDataset "selection.json"
    $selectionHash = (
        Get-FileHash -LiteralPath $holdoutSelection -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $replayReportHash = (
        Get-FileHash -LiteralPath $replayPrepareReport -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $valid = $false
    if (Test-Path -LiteralPath $replayIsolationReport -PathType Leaf) {
        try {
            $existing = Get-Content `
                -LiteralPath $replayIsolationReport `
                -Raw `
                -Encoding UTF8 |
                ConvertFrom-Json
            $valid = (
                $existing.schema_version -eq 1 -and
                $existing.holdout_selection_sha256 -eq $selectionHash -and
                $existing.replay_prepare_report_sha256 -eq $replayReportHash -and
                [int]$existing.holdout_selected_works -eq 395 -and
                [int]$existing.replay_works -eq 1584 -and
                @($existing.work_overlap).Count -eq 0
            )
        } catch {
            $valid = $false
        }
    }
    if (-not $valid) {
        & $python -m app.tools.audit_semantic_replay_holdout_isolation `
            --project-root $projectDir `
            --holdout-selection $holdoutSelection `
            --replay-prepare-report $replayPrepareReport `
            --replay-root $liederReplayRoot `
            --replay-root $quartetReplayRoot `
            --workers 4 `
            --output $replayIsolationReport
        if ($LASTEXITCODE -ne 0) {
            throw "Synthetic replay/holdout isolation audit failed"
        }
    }
}

function Assert-HoldoutRareEvidence {
    $holdoutSelection = Join-Path $holdoutDataset "selection.json"
    $selectionHash = (
        Get-FileHash -LiteralPath $holdoutSelection -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $report = Join-Path $projectDir (
        "training_data\diagnostics\muse-holdout-rare-semantic-evidence-v2-" +
        $selectionHash.Substring(0, 16) +
        ".json"
    )
    if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
        & $python -m app.tools.audit_muse_semantic_tag_evidence `
            --dataset-dir $holdoutDataset `
            --forbidden-selection $trainingSelection `
            --required-tag "Parenthesis=25" `
            --required-tag "Jump=25" `
            --required-tag "Marker=25" `
            --output $report
        if ($LASTEXITCODE -ne 0) {
            throw "Muse OMR rare semantic source evidence failed"
        }
    }
    $evidence = Get-Content -LiteralPath $report -Raw -Encoding UTF8 |
        ConvertFrom-Json
    if (
        $evidence.passed -ne $true -or
        $evidence.audit_version -ne "muse-semantic-tag-evidence@2" -or
        $evidence.selection_sha256 -ne $selectionHash -or
        [int]$evidence.selected_pairs -ne 435 -or
        [int]$evidence.selected_works -ne 395 -or
        [int]$evidence.tag_counts.Parenthesis -lt 25 -or
        [int]$evidence.tag_counts.Jump -lt 25 -or
        [int]$evidence.tag_counts.Marker -lt 25 -or
        @($evidence.pair_overlap).Count -ne 0 -or
        @($evidence.work_overlap).Count -ne 0
    ) {
        throw "Muse OMR rare semantic source-evidence report is invalid"
    }
}

function Invoke-Preparation {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Dataset,
        [Parameter(Mandatory = $true)]
        [string]$Output,
        [Parameter(Mandatory = $true)]
        [int]$MinimumAcceptedWorks,
        [switch]$IndependentHoldout
    )

    $report = Join-Path $Output "prepare-report.json"
    if (Test-Path -LiteralPath $report -PathType Leaf) {
        try {
            $existing = Get-Content -LiteralPath $report -Raw -Encoding UTF8 |
                ConvertFrom-Json
            if (
                $existing.oversized_fragment_visibility_version -eq
                    $visibilityVersion -and
                $existing.source_image_origin -eq
                    "synthetic_scan_degraded_render" -and
                $existing.production_evidence_eligible -eq $false -and
                [int]$existing.tile_size -eq 1024 -and
                [int]$existing.overlap -eq 256
            ) {
                return
            }
        } catch {
        }
        Remove-Item -LiteralPath $report -Force
    }
    $arguments = @(
        "-m", "app.tools.prepare_muse_omr_scan_regions",
        "--dataset-dir", $Dataset,
        "--output-dir", $Output,
        "--musescore-exe", $museScore,
        "--negative-ratio", "0.08",
        "--long-span-minimum-object-fraction", "0.25",
        "--minimum-ecc", "0.86",
        "--maximum-linear-deviation", "0.12",
        "--maximum-translation-fraction", "0.10",
        "--minimum-local-correlation-10p", "0.62",
        "--minimum-median-local-correlation", "0.72",
        "--minimum-accepted-page-fraction", "0.75",
        "--minimum-accepted-fraction", "0.50",
        "--minimum-accepted-works", $MinimumAcceptedWorks.ToString(
            [System.Globalization.CultureInfo]::InvariantCulture
        ),
        "--registration-workers", "4",
        "--resume"
    )
    if ($IndependentHoldout) {
        $arguments += @(
            "--expected-selection-role",
            "external_scan_degraded_development_benchmark_not_training",
            "--forbidden-selection",
            $trainingSelection
        )
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Muse OMR v2 preparation failed with exit code $LASTEXITCODE"
    }
}

function Read-ValidatedReport {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Output,
        [Parameter(Mandatory = $true)]
        [int]$SelectedPairs,
        [Parameter(Mandatory = $true)]
        [int]$SelectedWorks,
        [Parameter(Mandatory = $true)]
        [int]$MinimumAcceptedWorks,
        [switch]$IndependentHoldout
    )

    $reportPath = Join-Path $Output "prepare-report.json"
    $categoriesPath = Join-Path $Output "categories.json"
    if (
        -not (Test-Path -LiteralPath $reportPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $categoriesPath -PathType Leaf)
    ) {
        throw "Muse OMR v2 preparation is incomplete: $Output"
    }
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 |
        ConvertFrom-Json
    if (
        $report.registration_version -ne $registrationVersion -or
        $report.target_geometry_provenance -ne
            "complete-page-svg-geometry-before-tile-clipping@1" -or
        $report.oversized_fragment_visibility_version -ne
            $visibilityVersion -or
        $report.source_image_origin -ne
            "synthetic_scan_degraded_render" -or
        $report.production_evidence_eligible -ne $false -or
        [int]$report.tile_size -ne 1024 -or
        [int]$report.overlap -ne 256 -or
        [double]$report.long_span_minimum_object_fraction -ne 0.25 -or
        [int]$report.selected_pairs -ne $SelectedPairs -or
        [int]$report.selected_works -ne $SelectedWorks -or
        [int]$report.accepted_works -lt $MinimumAcceptedWorks -or
        @($report.forbidden_selection_overlap).Count -ne 0 -or
        @($report.forbidden_work_overlap).Count -ne 0
    ) {
        throw "Muse OMR v2 preparation failed coverage/isolation: $Output"
    }
    if (
        $IndependentHoldout -and
        [int]$report.source_count_by_split.test -ne
            [int]$report.accepted_works
    ) {
        throw "Muse OMR v2 holdout is not entirely isolated in the test split"
    }
    return $report
}

function Invoke-OverlapConsistentDerivation {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Output,
        [Parameter(Mandatory = $true)]
        [string[]]$Splits
    )

    $outputReport = Join-Path $Output "prepare-report.json"
    $sourceManifest = Join-Path $Source "manifest.json"
    $sourceHash = (
        Get-FileHash -LiteralPath $sourceManifest -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $stale = $false
    if (Test-Path -LiteralPath $outputReport -PathType Leaf) {
        try {
            $existing = Get-Content -LiteralPath $outputReport -Raw -Encoding UTF8 |
                ConvertFrom-Json
            $stale = (
                $existing.transformation_version -ne
                    $targetAssignmentVersion -or
                $existing.target_geometry_provenance -ne
                    "complete-page-svg-geometry-before-tile-clipping@1" -or
                $existing.oversized_fragment_visibility_version -ne
                    $visibilityVersion -or
                $existing.source_prepared_manifest_sha256 -ne $sourceHash
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
            throw "Muse overlap-consistent target derivation failed: $Source"
        }
    }
    $report = Get-Content -LiteralPath $outputReport -Raw -Encoding UTF8 |
        ConvertFrom-Json
    if (
        $report.transformation_version -ne
            $targetAssignmentVersion -or
        $report.target_geometry_provenance -ne
            "complete-page-svg-geometry-before-tile-clipping@1" -or
        $report.oversized_fragment_visibility_version -ne
            $visibilityVersion -or
        [double]$report.long_span_minimum_object_fraction -ne 0.25 -or
        $report.source_prepared_manifest_sha256 -ne $sourceHash
    ) {
        throw "Muse overlap-consistent target report is stale: $Output"
    }
    foreach ($split in $Splits) {
        if (
            -not (Test-Path -LiteralPath (Join-Path $Output "$split.jsonl") -PathType Leaf) -or
            [int]$report.splits.$split.unique_objects -le 0 -or
            $null -eq $report.splits.$split.duplicate_source_objects_removed -or
            [int]$report.splits.$split.additional_target_instances -le 0
        ) {
            throw "Muse overlap-consistent target split is incomplete: $split"
        }
    }
    return $report
}

function Invoke-CompletePageTargetAudit {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prepared,
        [Parameter(Mandatory = $true)]
        [string]$Output,
        [Parameter(Mandatory = $true)]
        [string[]]$Splits
    )

    $arguments = @(
        "-m", "app.tools.audit_complete_page_semantic_targets",
        "--prepared-dir", $Prepared,
        "--output", $Output
    )
    foreach ($split in $Splits) {
        $arguments += @("--split", $split)
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Complete-page semantic target audit failed: $Prepared"
    }
    $audit = Get-Content -LiteralPath $Output -Raw -Encoding UTF8 |
        ConvertFrom-Json
    if (
        $audit.audit_version -ne $targetAuditVersion -or
        $audit.passed -ne $true -or
        $audit.oversized_fragment_visibility_version -ne
            $visibilityVersion -or
        @($audit.failures).Count -ne 0 -or
        @($audit.dropped_long_span_objects.PSObject.Properties).Count -ne 0
    ) {
        throw "Complete-page semantic target evidence is invalid: $Output"
    }
    return $audit
}

function Invoke-SemanticPageLayoutEvidence {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prepared,
        [Parameter(Mandatory = $true)]
        [string]$Images,
        [Parameter(Mandatory = $true)]
        [string]$Output,
        [Parameter(Mandatory = $true)]
        [string[]]$Splits
    )

    $manifestPath = Join-Path $Prepared "manifest.json"
    $manifestHash = (
        Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $layoutSourceHash = (
        Get-FileHash `
            -LiteralPath (
                Join-Path $projectDir "app\src\scorescan\layout.py"
            ) `
            -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $layoutBuilder = Join-Path $projectDir (
        "app\tools\prepare_semantic_page_layout_evidence.py"
    )
    $layoutBuilderHash = (
        Get-FileHash -LiteralPath $layoutBuilder -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $valid = $false
    if (Test-Path -LiteralPath $Output -PathType Leaf) {
        try {
            $existing = Get-Content -LiteralPath $Output -Raw -Encoding UTF8 |
                ConvertFrom-Json
            $valid = (
                $existing.version -eq
                    "scorescan-semantic-page-layout-evidence@1" -and
                $existing.passed -eq $true -and
                $existing.prepared_manifest_sha256 -eq $manifestHash -and
                $existing.product_layout_source_sha256 -eq
                    $layoutSourceHash -and
                $existing.builder_version -eq
                    "hash-bound-product-layout-process-pool@2" -and
                $existing.builder_source_sha256 -eq
                    $layoutBuilderHash -and
                $existing.target_assignment_version -eq
                    $targetAssignmentVersion -and
                @($existing.failures).Count -eq 0 -and
                [int]$existing.page_count -gt 0 -and
                [int]$existing.staff_count -gt 0
            )
            foreach ($split in $Splits) {
                $splitPath = Join-Path $Prepared "$split.jsonl"
                $splitHash = (
                    Get-FileHash -LiteralPath $splitPath -Algorithm SHA256
                ).Hash.ToLowerInvariant()
                if ($existing.split_jsonl_sha256.$split -ne $splitHash) {
                    $valid = $false
                }
            }
        } catch {
            $valid = $false
        }
    }
    if (-not $valid) {
        $arguments = @(
            "-m", "app.tools.prepare_semantic_page_layout_evidence",
            "--prepared-dir", $Prepared,
            "--images-dir", $Images,
            "--output", $Output,
            "--workers", "8"
        )
        foreach ($split in $Splits) {
            $arguments += @("--split", $split)
        }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Semantic page-layout evidence failed: $Prepared"
        }
    }
    $evidence = Get-Content -LiteralPath $Output -Raw -Encoding UTF8 |
        ConvertFrom-Json
    if (
        $evidence.passed -ne $true -or
        $evidence.version -ne
            "scorescan-semantic-page-layout-evidence@1" -or
        @($evidence.failures).Count -ne 0 -or
        [int]$evidence.page_count -le 0 -or
        [int]$evidence.staff_count -le 0
    ) {
        throw "Semantic page-layout evidence is invalid: $Output"
    }
    return $evidence
}

Assert-HoldoutRareEvidence

if (-not (Test-Path -LiteralPath (Join-Path $trainingOutput "prepare-report.json") -PathType Leaf)) {
    & $python -m app.tools.seed_muse_registration_cache `
        --source $trainingCacheSource `
        --destination $trainingOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Muse stratified training cache seed failed"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $holdoutOutput "prepare-report.json") -PathType Leaf)) {
    & $python -m app.tools.seed_muse_registration_cache `
        --source $holdoutCacheSource `
        --destination $holdoutOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Muse stratified holdout cache seed failed"
    }
}

Invoke-Preparation `
    -Dataset $trainingDataset `
    -Output $trainingOutput `
    -MinimumAcceptedWorks $trainingMinimumAcceptedWorks
$trainingReport = Read-ValidatedReport `
    -Output $trainingOutput `
    -SelectedPairs $trainingSelectedPairs `
    -SelectedWorks $trainingSelectedWorks `
    -MinimumAcceptedWorks $trainingMinimumAcceptedWorks

Assert-ReplayIsolation

Invoke-Preparation `
    -Dataset $holdoutDataset `
    -Output $holdoutOutput `
    -MinimumAcceptedWorks $holdoutMinimumAcceptedWorks `
    -IndependentHoldout
$holdoutReport = Read-ValidatedReport `
    -Output $holdoutOutput `
    -SelectedPairs 435 `
    -SelectedWorks 395 `
    -MinimumAcceptedWorks $holdoutMinimumAcceptedWorks `
    -IndependentHoldout

$categories = Get-Content `
    -LiteralPath (Join-Path $holdoutOutput "categories.json") `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
$underSupported = @(
    foreach ($row in @($categories.classes)) {
        $name = [string]$row.name
        $count = [int]$holdoutReport.object_counts.$name
        if ($count -lt 25) {
            "$name=$count"
        }
    }
)
if ($underSupported.Count -gt 0) {
    throw (
        "Independent Muse OMR v2 class evidence is below 25 objects: " +
        ($underSupported -join ", ")
    )
}

$trainingConsistentReport = Invoke-OverlapConsistentDerivation `
    -Source $trainingOutput `
    -Output $trainingConsistentOutput `
    -Splits @("train", "calibration", "test")
$holdoutConsistentReport = Invoke-OverlapConsistentDerivation `
    -Source $holdoutOutput `
    -Output $holdoutConsistentOutput `
    -Splits @("test")
$trainingTargetAudit = Invoke-CompletePageTargetAudit `
    -Prepared $trainingConsistentOutput `
    -Output (
        Join-Path $projectDir (
            "training_data\benchmarks\" +
            "muse-training-complete-page-target-audit-v1.json"
        )
    ) `
    -Splits @("train", "calibration", "test")
$holdoutTargetAudit = Invoke-CompletePageTargetAudit `
    -Prepared $holdoutConsistentOutput `
    -Output (
        Join-Path $projectDir (
            "training_data\benchmarks\" +
            "muse-holdout-complete-page-target-audit-v1.json"
        )
    ) `
    -Splits @("test")
$trainingLayout = Invoke-SemanticPageLayoutEvidence `
    -Prepared $trainingConsistentOutput `
    -Images $trainingOutput `
    -Output $trainingLayoutEvidence `
    -Splits @("train", "calibration", "test")
$holdoutLayout = Invoke-SemanticPageLayoutEvidence `
    -Prepared $holdoutConsistentOutput `
    -Images $holdoutOutput `
    -Output $holdoutLayoutEvidence `
    -Splits @("test")

@{
    state = "muse_scan_v2_preparation_completed"
    registration_version = $registrationVersion
    training_minimum_accepted_works = $trainingMinimumAcceptedWorks
    training_accepted_works = [int]$trainingReport.accepted_works
    holdout_minimum_accepted_works = $holdoutMinimumAcceptedWorks
    holdout_accepted_works = [int]$holdoutReport.accepted_works
    holdout_classes = @($categories.classes).Count
    target_assignment_version = $targetAssignmentVersion
    oversized_fragment_visibility_version = $visibilityVersion
    training_unique_objects = (
        [int]$trainingConsistentReport.splits.train.unique_objects +
        [int]$trainingConsistentReport.splits.calibration.unique_objects +
        [int]$trainingConsistentReport.splits.test.unique_objects
    )
    holdout_unique_objects = (
        [int]$holdoutConsistentReport.splits.test.unique_objects
    )
    training_target_audit_passed = [bool]$trainingTargetAudit.passed
    holdout_target_audit_passed = [bool]$holdoutTargetAudit.passed
    training_layout_pages = [int]$trainingLayout.page_count
    holdout_layout_pages = [int]$holdoutLayout.page_count
} | ConvertTo-Json -Compress
