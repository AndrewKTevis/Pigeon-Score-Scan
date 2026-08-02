param(
    [Parameter(Mandatory = $true)]
    [int]$WaitSemanticHoldoutPid,
    [int]$WaitGpuReleasePid = 2147483647,
    [string]$GpuPython = $env:PIGEON_SCORE_SCAN_GPU_PYTHON
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
if (
    [string]::IsNullOrWhiteSpace($GpuPython) -or
    -not (Test-Path -LiteralPath $GpuPython -PathType Leaf)
) {
    throw "Set PIGEON_SCORE_SCAN_GPU_PYTHON to an isolated CUDA training Python executable"
}
$preparedDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_regions_stratified_complete_page_overlap_consistent_deduplicated_v7"
$modelDir = Join-Path $projectDir "training_data\models\muse-omr-scan-semantic-detector-v4-complete-page-e12-b2-20260730"
$candidateDir = Join-Path $projectDir "training_data\release_candidates\semantic-detector-muse-v4-complete-page-e12-20260730"
$sampleReport = Join-Path $candidateDir "parity-sample.json"
$onnxModel = Join-Path $candidateDir "semantic_detector.onnx"
$parityReport = Join-Path $candidateDir "semantic_detector_onnx_parity.json"
$gpuParityReport = Join-Path $candidateDir "semantic_detector_onnx_gpu_parity.json"
$candidateReport = Join-Path $candidateDir "release-candidate.json"
$canonicalResources = Join-Path $projectDir "app\src\scorescan\resources"
$evaluationResources = Join-Path $candidateDir "evaluation-resources"
$holdoutReport = Join-Path $modelDir "evaluation.independent-muse-holdout.json"
$trainingReport = Join-Path $modelDir "training_report.json"
$storageGuard = Join-Path $PSScriptRoot "assert_training_storage.ps1"

foreach ($processId in @($WaitSemanticHoldoutPid, $WaitGpuReleasePid)) {
    $waitProcess = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -ne $waitProcess) {
        Wait-Process -Id $processId
    }
}
foreach ($required in @(
    $trainingReport,
    $holdoutReport,
    (Join-Path $preparedDir "test.jsonl"),
    (Join-Path $preparedDir "categories.json"),
    (Join-Path $modelDir "model.best.pt")
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Semantic release-candidate input is missing: $required"
    }
}
& $storageGuard `
    -Stage "semantic-detector-candidate-export" `
    -RequiredNewArtifactGiB 1.0
$training = Get-Content -LiteralPath $trainingReport -Raw | ConvertFrom-Json
$holdout = Get-Content -LiteralPath $holdoutReport -Raw | ConvertFrom-Json
if ($training.acceptance.passed -ne $true) {
    throw "Semantic detector training gate did not pass"
}
if ($holdout.acceptance.passed -ne $true) {
    throw "Independent semantic holdout gate did not pass"
}
if (-not (Test-Path -LiteralPath $gpuPython -PathType Leaf)) {
    throw "Verified portable GPU runtime is unavailable: $gpuPython"
}

New-Item -ItemType Directory -Path $candidateDir -Force | Out-Null
Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = Join-Path $projectDir "app\src"
if (-not (Test-Path -LiteralPath $sampleReport -PathType Leaf)) {
    & $python -m app.tools.select_semantic_detector_parity_sample `
        --test-jsonl (Join-Path $preparedDir "test.jsonl") `
        --project-root $projectDir `
        --output-report $sampleReport
    if ($LASTEXITCODE -ne 0) {
        throw "Semantic detector parity-sample selection failed with code $LASTEXITCODE"
    }
}

& wsl.exe -d Ubuntu-24.04 -- bash `
    /workspace/pigeon-score-scan/training/run_wsl_export_semantic_detector_candidate.sh
if ($LASTEXITCODE -ne 0) {
    throw "Semantic detector ONNX export/parity failed with code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $gpuParityReport -PathType Leaf)) {
    $sample = Get-Content -LiteralPath $sampleReport -Raw | ConvertFrom-Json
    $crop = @($sample.crop_xyxy) -join ","
    $deploymentThresholds = @(
        $holdout.metrics.operating_points.PSObject.Properties |
            ForEach-Object { [double]$_.Value.threshold }
    )
    if ($deploymentThresholds.Count -eq 0) {
        throw "Independent holdout has no semantic deployment thresholds"
    }
    $comparisonFloor = (
        $deploymentThresholds |
            Measure-Object -Minimum
    ).Minimum.ToString(
        "R",
        [System.Globalization.CultureInfo]::InvariantCulture
    )
    & $gpuPython -m app.tools.verify_semantic_detector_onnx_runtime `
        --onnx $onnxModel `
        --parity-image $sample.image `
        --parity-crop $crop `
        --output-report $gpuParityReport `
        --minimum-detections 1 `
        --comparison-score-floor $comparisonFloor `
        --maximum-box-error 0.10 `
        --maximum-score-error 0.0001
    if ($LASTEXITCODE -ne 0) {
        throw "Semantic detector CUDA runtime parity failed with code $LASTEXITCODE"
    }
}

$cpuParity = Get-Content -LiteralPath $parityReport -Raw | ConvertFrom-Json
$gpuParity = Get-Content -LiteralPath $gpuParityReport -Raw | ConvertFrom-Json
if ($cpuParity.passed -ne $true -or $gpuParity.passed -ne $true) {
    throw "Semantic detector release-candidate parity reports did not pass"
}
$sourceModelHash = (
    Get-FileHash `
        -LiteralPath (Join-Path $modelDir "model.best.pt") `
        -Algorithm SHA256
).Hash.ToLowerInvariant()
$modelVersion = "semantic-muse-v4-complete-page-e12-$($sourceModelHash.Substring(0, 12))"

# Build a complete isolated resource overlay.  The detector can then be exercised
# through the real product pipeline without prematurely modifying canonical
# application resources.
if (-not (Test-Path -LiteralPath $evaluationResources -PathType Container)) {
    $stagingResources = (
        "$evaluationResources.staging-" +
        [Guid]::NewGuid().ToString("N")
    )
    New-Item -ItemType Directory -Path $stagingResources | Out-Null
    Get-ChildItem -LiteralPath $canonicalResources -File | ForEach-Object {
        Copy-Item `
            -LiteralPath $_.FullName `
            -Destination (Join-Path $stagingResources $_.Name)
    }
    & $python -m app.tools.authorize_semantic_detector_release `
        --source-model (Join-Path $modelDir "model.best.pt") `
        --onnx $onnxModel `
        --categories (Join-Path $preparedDir "categories.json") `
        --training-report $trainingReport `
        --holdout-report $holdoutReport `
        --parity-report $parityReport `
        --gpu-parity-report $gpuParityReport `
        --output-resources $stagingResources `
        --model-version $modelVersion
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Isolated semantic detector resource authorization failed with code " +
            $LASTEXITCODE
        )
    }
    Move-Item `
        -LiteralPath $stagingResources `
        -Destination $evaluationResources
}

& $python -c (
    "from pathlib import Path; " +
    "from scorescan.model_registry import audit_model_manifest; " +
    "from scorescan.semantic_detector import load_semantic_detector_assets; " +
    "p=Path(r'$evaluationResources'); " +
    "a=audit_model_manifest(p); " +
    "assert a.verified, a.errors; " +
    "s=load_semantic_detector_assets(p); " +
    "assert s.model_version == '$modelVersion'"
)
if ($LASTEXITCODE -ne 0) {
    throw "Isolated semantic detector resource overlay failed revalidation"
}
$semanticManifest = Join-Path $evaluationResources "semantic_detector.json"
$payload = [ordered]@{
    format = 2
    name = "scorescan-semantic-detector-release-candidate-v1"
    boundary_contract_version = "printed-western-instrumental-scan-boundary@4"
    canonical_resources_authorized = $false
    physical_scan_release_evidence = $false
    source_image_origin = "rendered_scan_degraded"
    reason = (
        "staged only; canonical resources require a separate full-page physical-" +
        "scan product benchmark and end-to-end regression gates"
    )
    source_model = (Join-Path $modelDir "model.best.pt")
    source_model_sha256 = $sourceModelHash
    onnx = $onnxModel
    onnx_sha256 = (Get-FileHash -LiteralPath $onnxModel -Algorithm SHA256).Hash.ToLowerInvariant()
    independent_holdout = $holdoutReport
    cpu_parity = $parityReport
    gpu_parity = $gpuParityReport
    isolated_product_evaluation_resources = $evaluationResources
    isolated_product_evaluation_resources_verified = $true
    semantic_manifest_sha256 = (
        Get-FileHash -LiteralPath $semanticManifest -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    model_version = $modelVersion
}
$temporary = "$candidateReport.tmp"
$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding UTF8
Move-Item -LiteralPath $temporary -Destination $candidateReport -Force
