param(
    [Parameter(Mandatory = $true)]
    [int]$WaitBenchmarkAcquisitionPid,
    [switch]$ReuseVerifiedPartitions
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$holdoutDir = Join-Path $projectDir "training_data\external\benchmarks\muse_omr_e27f6a8634_raremarks_v3"
$baseHoldoutDir = Join-Path $projectDir "training_data\external\benchmarks\muse_omr_e27f6a8634"
$holdoutSelection = Join-Path $holdoutDir "selection.json"
$holdoutProvenance = Join-Path $holdoutDir "provenance.json"
$trainingDir = Join-Path $projectDir "training_data\external\training\muse_omr_scan_train_stratified_v2"
$trainingProvenance = Join-Path $trainingDir "provenance.json"
$workCatalogDir = Join-Path $projectDir "training_data\external\catalogs\muse_omr_work_catalog_e27f6a8634"
$workCatalog = Join-Path $workCatalogDir "work-catalog.json"
$statePath = Join-Path $PSScriptRoot "muse_production_pipeline_state.json"
$failurePath = Join-Path $PSScriptRoot "muse_production_pipeline.failed.txt"
$powershell = Join-Path $PSHOME "powershell.exe"

function Write-PipelineState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Phase,
        [System.Collections.IDictionary]$Processes = @{}
    )
    $payload = [ordered]@{
        format = 1
        phase = $Phase
        updated_at = [DateTimeOffset]::Now.ToString("o")
        orchestrator_pid = $PID
        processes = $Processes
    }
    $temporary = "$statePath.tmp"
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
}

function Start-QueueScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptName,
        [string]$Arguments = ""
    )
    $scriptPath = Join-Path $PSScriptRoot $ScriptName
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
        throw "Queue script is missing: $scriptPath"
    }
    $argumentText = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" $Arguments"
    return Start-Process `
        -FilePath $powershell `
        -ArgumentList $argumentText `
        -WorkingDirectory $projectDir `
        -WindowStyle Hidden `
        -PassThru
}

try {
    Remove-Item -LiteralPath $failurePath -Force -ErrorAction SilentlyContinue
    if (-not $ReuseVerifiedPartitions) {
        throw (
            "The pre-stratification acquisition path is retired. " +
            "Build queue_muse_stratified_partition_v2.ps1 first and rerun " +
            "with -ReuseVerifiedPartitions."
        )
    }

    if (
        -not (Test-Path -LiteralPath $holdoutProvenance -PathType Leaf) -or
        -not (Test-Path -LiteralPath $workCatalog -PathType Leaf)
    ) {
        throw "Independent holdout/catalog provenance is missing"
    }
    $holdout = Get-Content -LiteralPath $holdoutProvenance -Raw | ConvertFrom-Json
    $workCatalogHash = (
        Get-FileHash -LiteralPath $workCatalog -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if (
        $holdout.role -ne "external_scan_degraded_development_benchmark_not_training" -or
        $holdout.source_image_origin -ne
            "synthetic_scan_degraded_render" -or
        $holdout.production_evidence_eligible -ne $false -or
        [int]$holdout.selected_pair_count -ne 435 -or
        [int]$holdout.selected_work_count -ne 395 -or
        [int]$holdout.selected_independent_work_pdf_page_count -lt 2000 -or
        $holdout.work_catalog_sha256 -ne $workCatalogHash -or
        @($holdout.files).Count -ne 870
    ) {
        throw "Independent holdout acquisition failed its completeness gate"
    }

    $training = Get-Content -LiteralPath $trainingProvenance -Raw | ConvertFrom-Json
    if (
        $training.role -ne "external_scan_degraded_training_only" -or
        $training.source_image_origin -ne
            "synthetic_scan_degraded_render" -or
        $training.production_evidence_eligible -ne $false -or
        $training.partition_contract_version -ne
            "muse-omr-boundary-filter@1" -or
        [int]$training.selected_pair_count -lt 200 -or
        [int]$training.selected_work_count -lt 200 -or
        [int]$training.reserved_holdout_pair_count -lt 570 -or
        [int]$training.reserved_holdout_work_count -lt 530 -or
        $training.work_catalog_sha256 -ne $workCatalogHash -or
        @($training.training_holdout_overlap).Count -ne 0 -or
        @($training.training_holdout_work_overlap).Count -ne 0 -or
        @($training.files).Count -ne
            (2 * [int]$training.selected_pair_count)
    ) {
        throw "Muse OMR training acquisition failed its isolation gate"
    }
    & $python -m app.tools.prune_muse_omr_unselected_files `
        --dataset-dir $trainingDir `
        --execute
    if ($LASTEXITCODE -ne 0) {
        throw "Muse OMR unselected-file pruning exited with code $LASTEXITCODE"
    }

    Write-PipelineState -Phase "starting_preparation_and_serial_gpu_queues"
    $scanPipeline = Start-QueueScript `
        -ScriptName "queue_muse_scan_finetune.ps1" `
        -Arguments "-WaitAcquisitionPid 2147483647"
    $scanText = Start-QueueScript `
        -ScriptName "queue_muse_scan_text_preparation.ps1" `
        -Arguments ""
    $holdoutRegions = Start-QueueScript `
        -ScriptName "queue_muse_holdout_preparation.ps1" `
        -Arguments "-WaitPreparationPid 2147483647"
    $holdoutText = Start-QueueScript `
        -ScriptName "queue_muse_holdout_text_preparation.ps1" `
        -Arguments "-WaitHoldoutRegionPid $($holdoutRegions.Id)"
    $holdoutLabels = Start-QueueScript `
        -ScriptName "queue_muse_holdout_detection_labels.ps1" `
        -Arguments "-WaitHoldoutTextPid $($holdoutText.Id)"
    $semanticHoldout = Start-QueueScript `
        -ScriptName "queue_muse_holdout_evaluation.ps1" `
        -Arguments "-WaitHoldoutPid $($holdoutRegions.Id) -WaitScanPipelinePid $($scanPipeline.Id)"
    $semanticCandidate = Start-QueueScript `
        -ScriptName "queue_semantic_detector_release_candidate.ps1" `
        -Arguments "-WaitSemanticHoldoutPid $($semanticHoldout.Id)"
    $recognition = Start-QueueScript `
        -ScriptName "queue_ppocrv6_scorescan_training.ps1" `
        -Arguments "-WaitBootstrapPid 2147483647 -WaitGpuPipelinePid $($semanticCandidate.Id)"
    $detection = Start-QueueScript `
        -ScriptName "queue_ppocrv6_scorescan_detection.ps1" `
        -Arguments "-WaitRecognitionPid $($recognition.Id) -WaitWeightPid 2147483647 -WaitHoldoutLabelsPid $($holdoutLabels.Id)"

    $processes = [ordered]@{
        scan_semantic_pipeline = $scanPipeline.Id
        scan_text_preparation = $scanText.Id
        holdout_region_preparation = $holdoutRegions.Id
        holdout_text_preparation = $holdoutText.Id
        holdout_detection_labels = $holdoutLabels.Id
        semantic_holdout_evaluation = $semanticHoldout.Id
        semantic_release_candidate = $semanticCandidate.Id
        ocr_recognition_pipeline = $recognition.Id
        ocr_detection_pipeline = $detection.Id
    }
    Write-PipelineState -Phase "queues_started" -Processes $processes
} catch {
    $_.Exception.ToString() | Set-Content -LiteralPath $failurePath -Encoding UTF8
    Write-PipelineState -Phase "failed"
    throw
}
