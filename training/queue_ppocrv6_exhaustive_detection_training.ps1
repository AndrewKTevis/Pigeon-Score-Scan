param(
    [Parameter(Mandatory = $true)]
    [int]$WaitGpuPid
)

$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$labelDir = Join-Path $projectDir "training_data\prepared\scorescan_ocr_detection_stratified_v4"
$labelReport = Join-Path $labelDir "merge-report.json"
$modelDir = Join-Path $projectDir "training_data\models\ppocrv6-scorescan-det-exhaustive-stratified-e24-b2-20260729"
$onnxModel = Join-Path $modelDir "scorescan-ppocrv6-det.onnx"
$calibrationLabels = Join-Path $labelDir "calibration.scan.exhaustive.paddle.det.txt"
$testScanLabels = Join-Path $labelDir "test.scan.exhaustive.paddle.det.txt"
$testCleanLabels = Join-Path $labelDir "test.clean.exhaustive.paddle.det.txt"
$calibrationReport = Join-Path $modelDir "postprocess-calibration.scan.iou075.json"
$scanRuntimeLog = Join-Path $modelDir "evaluation.onnx-scan-test.iou075.log"
$scanRuntimeReport = Join-Path $modelDir "evaluation.onnx-scan-test.iou075.json"
$cleanRuntimeLog = Join-Path $modelDir "evaluation.onnx-clean-test.iou075.log"
$cleanRuntimeReport = Join-Path $modelDir "evaluation.onnx-clean-test.iou075.json"
$onnxGate = Join-Path $modelDir "onnx-release-gate.iou075.json"
$failureReport = Join-Path $modelDir "training.failed.txt"
$storageGuard = Join-Path $PSScriptRoot "assert_training_storage.ps1"

while (-not (Test-Path -LiteralPath $labelReport -PathType Leaf)) {
    Start-Sleep -Seconds 10
}
foreach ($required in @(
    $calibrationLabels,
    $testScanLabels,
    $testCleanLabels
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Exhaustive detection label file is missing: $required"
    }
}

$gpuProcess = Get-Process -Id $WaitGpuPid -ErrorAction SilentlyContinue
if ($null -ne $gpuProcess) {
    Wait-Process -Id $WaitGpuPid
}
& $storageGuard `
    -Stage "ppocrv6-scorescan-detection-exhaustive" `
    -RequiredNewArtifactGiB 2.0

Set-Location -LiteralPath $projectDir
try {
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    Remove-Item -LiteralPath $failureReport -Force -ErrorAction SilentlyContinue
    & wsl.exe -d Ubuntu-24.04 -- bash `
        /workspace/pigeon-score-scan/training/run_wsl_ppocrv6_scorescan_detection_exhaustive.sh
    if ($LASTEXITCODE -ne 0) {
        throw "Exhaustive PP-OCRv6 detection training/export failed with code $LASTEXITCODE"
    }

    & $pythonExe -m app.tools.calibrate_onnx_ocr_detection_thresholds `
        --model $onnxModel `
        --labels $calibrationLabels `
        --dataset-report $labelReport `
        --project-root $projectDir `
        --minimum-iou 0.75 `
        --output-report $calibrationReport
    if ($LASTEXITCODE -ne 0) {
        throw "Exhaustive ONNX detection threshold calibration failed"
    }
    $calibration = Get-Content -LiteralPath $calibrationReport -Raw |
        ConvertFrom-Json
    if (
        $calibration.test_split_used_for_selection -ne $false -or
        $calibration.minimum_iou -lt 0.75
    ) {
        throw "Exhaustive detection threshold calibration violated isolation"
    }
    $detThreshold = [double]$calibration.selected.threshold
    $detBoxThreshold = [double]$calibration.selected.box_threshold
    $detUnclipRatio = [double]$calibration.selected.unclip_ratio

    & $pythonExe -m app.tools.evaluate_onnx_ocr_detection `
        --model $onnxModel `
        --labels $testScanLabels `
        --project-root $projectDir `
        --output-report $scanRuntimeReport `
        --minimum-iou 0.75 `
        --threshold $detThreshold `
        --box-threshold $detBoxThreshold `
        --unclip-ratio $detUnclipRatio `
        2>&1 | Tee-Object -FilePath $scanRuntimeLog
    if ($LASTEXITCODE -ne 0) {
        throw "Exhaustive ONNX scan detection evaluation failed"
    }
    & $pythonExe -m app.tools.evaluate_onnx_ocr_detection `
        --model $onnxModel `
        --labels $testCleanLabels `
        --project-root $projectDir `
        --output-report $cleanRuntimeReport `
        --minimum-iou 0.75 `
        --threshold $detThreshold `
        --box-threshold $detBoxThreshold `
        --unclip-ratio $detUnclipRatio `
        2>&1 | Tee-Object -FilePath $cleanRuntimeLog
    if ($LASTEXITCODE -ne 0) {
        throw "Exhaustive ONNX clean detection evaluation failed"
    }
    & $pythonExe -m app.tools.gate_paddleocr_detection `
        --scan-log $scanRuntimeLog `
        --clean-log $cleanRuntimeLog `
        --output-report $onnxGate `
        --model $onnxModel `
        --dataset-report $labelReport `
        --scan-labels $testScanLabels `
        --clean-labels $testCleanLabels `
        --minimum-scan-precision 0.995 `
        --minimum-scan-recall 0.995 `
        --minimum-scan-hmean 0.995 `
        --minimum-clean-precision 0.995 `
        --minimum-clean-recall 0.995 `
        --minimum-clean-hmean 0.995
    if ($LASTEXITCODE -ne 0) {
        throw "Exhaustive ONNX detector failed the strict release gate"
    }
    & $pythonExe -m app.tools.assert_high_value_gpu_stage `
        --project-root $projectDir `
        --stage "ocr_detection_exhaustive" `
        --output-report (
            Join-Path $projectDir (
                "training_data\logs\high-value-stage-gates\" +
                "ocr-detection-exhaustive-current.json"
            )
        )
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Exhaustive OCR detection stage produced no accepted, " +
            "hash-bound candidate"
        )
    }
} catch {
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    $_.Exception.ToString() | Set-Content -LiteralPath $failureReport -Encoding UTF8
    throw
}
