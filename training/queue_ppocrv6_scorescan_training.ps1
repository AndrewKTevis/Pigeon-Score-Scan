param(
    [Parameter(Mandatory = $true)]
    [int]$WaitBootstrapPid,
    [Parameter(Mandatory = $true)]
    [int]$WaitGpuPipelinePid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$cleanDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_pdf_text_v2"
$scanDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_text_stratified_v3"
$cleanReport = Join-Path $cleanDir "prepare-report.json"
$scanReport = Join-Path $scanDir "prepare-report.json"
$cleanFailure = Join-Path $cleanDir "preparation.failed.txt"
$scanFailure = Join-Path $scanDir "preparation.failed.txt"
$mergedDir = Join-Path $projectDir "training_data\prepared\scorescan_ocr_training_stratified_v2"
$mergedReport = Join-Path $mergedDir "merge-report.json"
$semanticCandidate = Join-Path $projectDir "training_data\release_candidates\semantic-detector-muse-v4-complete-page-e12-20260730\release-candidate.json"
$paddlePython = $env:SCORESCAN_PADDLE_PYTHON
$preparedRoot = Join-Path $projectDir "training_data\prepared"
$storageGuard = Join-Path $PSScriptRoot "assert_training_storage.ps1"
. (Join-Path $PSScriptRoot "ocr_artifact_freshness.ps1")

$bootstrap = Get-Process -Id $WaitBootstrapPid -ErrorAction SilentlyContinue
if ($null -ne $bootstrap) {
    Wait-Process -Id $WaitBootstrapPid
}
if (
    [string]::IsNullOrWhiteSpace($paddlePython) -or
    -not (Test-Path -LiteralPath $paddlePython -PathType Leaf)
) {
    throw "Set SCORESCAN_PADDLE_PYTHON to the isolated PP-OCRv6 Python executable"
}

while (
    -not (Test-Path -LiteralPath $cleanReport -PathType Leaf) -or
    -not (Test-Path -LiteralPath $scanReport -PathType Leaf)
) {
    if (Test-Path -LiteralPath $cleanFailure -PathType Leaf) {
        throw "Clean exact-text preparation failed"
    }
    if (Test-Path -LiteralPath $scanFailure -PathType Leaf) {
        throw "Registered-scan exact-text preparation failed"
    }
    Start-Sleep -Seconds 20
}
if (
    (Test-Path -LiteralPath $mergedReport -PathType Leaf) -and
    -not (Test-OcrArtifactFreshness `
        -Python $python `
        -ProjectDir $projectDir `
        -Kind "merged-labels" `
        -ArtifactReport $mergedReport `
        -SourceReports @($cleanReport, $scanReport))
) {
    Move-StaleOcrArtifact `
        -ArtifactDir $mergedDir `
        -PreparedRoot $preparedRoot
}

Set-Location -LiteralPath $projectDir
if (-not (Test-Path -LiteralPath $mergedReport -PathType Leaf)) {
    & $python -m app.tools.merge_ocr_training_labels `
        --project-root $projectDir `
        --output-dir $mergedDir `
        --clean-dataset "lieder=$cleanDir" `
        --scan-dataset "muse_scan=$scanDir" `
        --scan-target-fraction 0.35 `
        --maximum-rows-per-normalized-text 256 `
        --seed 20260728
    if ($LASTEXITCODE -ne 0) {
        throw "OCR training label merge failed with code $LASTEXITCODE"
    }
}

$gpuPipeline = Get-Process -Id $WaitGpuPipelinePid -ErrorAction SilentlyContinue
if ($null -ne $gpuPipeline) {
    Wait-Process -Id $WaitGpuPipelinePid
}
& $storageGuard `
    -Stage "ppocrv6-scorescan-recognition" `
    -RequiredNewArtifactGiB 1.5
$semanticReleaseEligible = $false
if (Test-Path -LiteralPath $semanticCandidate -PathType Leaf) {
    $semanticCandidateReport = Get-Content `
        -LiteralPath $semanticCandidate `
        -Raw `
        -Encoding UTF8 |
        ConvertFrom-Json
    $semanticReleaseEligible = (
        [int]$semanticCandidateReport.format -eq 2 -and
        $semanticCandidateReport.boundary_contract_version -eq
            "printed-western-instrumental-scan-boundary@4" -and
        $semanticCandidateReport.canonical_resources_authorized -eq $false -and
        $semanticCandidateReport.physical_scan_release_evidence -eq $false -and
        $semanticCandidateReport.source_image_origin -eq
            "rendered_scan_degraded" -and
        (Test-Path -LiteralPath $semanticCandidateReport.onnx -PathType Leaf) -and
        (Test-Path -LiteralPath $semanticCandidateReport.cpu_parity -PathType Leaf) -and
        (Test-Path -LiteralPath $semanticCandidateReport.gpu_parity -PathType Leaf)
    )
}
if (-not $semanticReleaseEligible) {
    Write-Warning (
        "Semantic detector is not release-eligible. Independent OCR candidate " +
        "training will continue, but joint packaging and test_space deployment " +
        "remain blocked."
    )
}
Write-Output (
    @{
        state = "semantic_release_gate_observed"
        release_eligible = $semanticReleaseEligible
        independent_ocr_training_continues = $true
    } | ConvertTo-Json -Compress
)

& wsl.exe -d Ubuntu-24.04 -- bash `
    /workspace/pigeon-score-scan/training/run_wsl_ppocrv6_scorescan_training.sh
if ($LASTEXITCODE -ne 0) {
    throw "PP-OCRv6 ScoreScan domain training or its release gate failed"
}

$modelDir = Join-Path $projectDir "training_data\models\ppocrv6-scorescan-rec-stratified-e18-b8-20260729"
$onnxModel = Join-Path $modelDir "scorescan-ppocrv6-rec.onnx"
$keys = Join-Path $projectDir "training_data\external\corpora\paddleocr_2661c7c0\ppocr\utils\dict\ppocrv6_dict.txt"
$scanRuntimeLog = Join-Path $modelDir "evaluation.onnx-scan-test.log"
$scanRuntimeReport = Join-Path $modelDir "evaluation.onnx-scan-test.json"
$cleanRuntimeLog = Join-Path $modelDir "evaluation.onnx-clean-test.log"
$cleanRuntimeReport = Join-Path $modelDir "evaluation.onnx-clean-test.json"
$onnxGate = Join-Path $modelDir "onnx-release-gate.json"

& $python -m app.tools.evaluate_onnx_ocr_recognition `
    --model $onnxModel `
    --keys $keys `
    --labels (Join-Path $mergedDir "test.scan.paddle.txt") `
    --project-root $projectDir `
    --output-report $scanRuntimeReport `
    2>&1 | Tee-Object -FilePath $scanRuntimeLog
if ($LASTEXITCODE -ne 0) {
    throw "Exported ONNX scan evaluation failed with code $LASTEXITCODE"
}
& $python -m app.tools.evaluate_onnx_ocr_recognition `
    --model $onnxModel `
    --keys $keys `
    --labels (Join-Path $mergedDir "test.clean.paddle.txt") `
    --project-root $projectDir `
    --output-report $cleanRuntimeReport `
    2>&1 | Tee-Object -FilePath $cleanRuntimeLog
if ($LASTEXITCODE -ne 0) {
    throw "Exported ONNX clean evaluation failed with code $LASTEXITCODE"
}
& $python -m app.tools.gate_paddleocr_evaluation `
    --scan-log $scanRuntimeLog `
    --clean-log $cleanRuntimeLog `
    --output-report $onnxGate `
    --model $onnxModel `
    --dataset-report $mergedReport `
    --minimum-scan-accuracy 0.998 `
    --minimum-scan-normalized-edit 0.9995 `
    --minimum-clean-accuracy 0.998 `
    --minimum-clean-normalized-edit 0.9995
if ($LASTEXITCODE -ne 0) {
    throw "Exported ONNX recognition model failed its independent release gate"
}

# Only a passed independent ONNX gate makes the training-resume state
# regenerable. Keep the selected Paddle parameters, exported inference model,
# ONNX candidate and every metric/report; discard optimizer/latest state and
# any generic duplicate that have no deployment role.
$regenerableTrainingState = @(
    (Join-Path $modelDir "best_model"),
    (Join-Path $modelDir "best_accuracy.pdopt"),
    (Join-Path $modelDir "latest.pdparams"),
    (Join-Path $modelDir "latest.pdopt"),
    (Join-Path $modelDir "latest.states"),
    (Join-Path $modelDir "inference")
)
$modelDirPrefix = $modelDir.TrimEnd("\") + "\"
$regenerableTrainingState += @(
    Get-ChildItem -LiteralPath $modelDir -File -Force |
        Where-Object {
            $_.Name -match "^iter_epoch_[0-9]+\.(pdparams|pdopt|states)$"
        } |
        Select-Object -ExpandProperty FullName
)
foreach ($path in $regenerableTrainingState) {
    if (-not (Test-Path -LiteralPath $path)) {
        continue
    }
    $item = Get-Item -LiteralPath $path -Force
    if (
        -not $item.FullName.StartsWith(
            $modelDirPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)
    ) {
        throw "Unsafe recognition-state cleanup target: $($item.FullName)"
    }
    Remove-Item -LiteralPath $item.FullName -Recurse -Force
}
& $python -m app.tools.assert_high_value_gpu_stage `
    --project-root $projectDir `
    --stage "ocr_recognition" `
    --output-report (
        Join-Path $projectDir (
            "training_data\logs\high-value-stage-gates\" +
            "ocr-recognition-current.json"
        )
    )
if ($LASTEXITCODE -ne 0) {
    throw "OCR recognition stage produced no accepted, hash-bound candidate"
}
