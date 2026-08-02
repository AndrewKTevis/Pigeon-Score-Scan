param(
    [Parameter(Mandatory = $true)]
    [int]$WaitRecognitionPid,
    [Parameter(Mandatory = $true)]
    [int]$WaitWeightPid,
    [Parameter(Mandatory = $true)]
    [int]$WaitHoldoutLabelsPid
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
$mergedDir = Join-Path $projectDir "training_data\prepared\scorescan_ocr_detection_stratified_v2"
$mergedReport = Join-Path $mergedDir "merge-report.json"
$weight = Join-Path $projectDir "training_data\external\models\paddleocr-ppocrv6-medium-det-training\PP-OCRv6_medium_det_pretrained.pdparams"
$expectedWeightBytes = 67465190L
$expectedWeightMd5 = "7b3c850d9ced2ba2c3d11a5c206e3986"
$holdoutTextDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_text_stratified_v3"
$holdoutTextReport = Join-Path $holdoutTextDir "prepare-report.json"
$holdoutLabelsDir = Join-Path $projectDir "training_data\prepared\muse_omr_scan_holdout_detection_labels_stratified_v2"
$holdoutLabelsReport = Join-Path $holdoutLabelsDir "prepare-report.json"
$holdoutLabelsFailure = "$holdoutLabelsDir.preparation.failed.txt"
$preparedRoot = Join-Path $projectDir "training_data\prepared"
$storageGuard = Join-Path $PSScriptRoot "assert_training_storage.ps1"
. (Join-Path $PSScriptRoot "ocr_artifact_freshness.ps1")

$weightProcess = Get-Process -Id $WaitWeightPid -ErrorAction SilentlyContinue
if ($null -ne $weightProcess) {
    Wait-Process -Id $WaitWeightPid
}
if (
    -not (Test-Path -LiteralPath $weight -PathType Leaf) -or
    (Get-Item -LiteralPath $weight).Length -ne $expectedWeightBytes
) {
    throw "Verified PP-OCRv6 detection initialization weight is missing"
}
$actualWeightMd5 = (
    Get-FileHash -LiteralPath $weight -Algorithm MD5
).Hash.ToLowerInvariant()
if ($actualWeightMd5 -ne $expectedWeightMd5) {
    throw "PP-OCRv6 detection initialization weight MD5 mismatch"
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
    & $python -m app.tools.merge_ocr_detection_labels `
        --project-root $projectDir `
        --output-dir $mergedDir `
        --clean-dataset "lieder=$cleanDir" `
        --scan-dataset "muse_scan=$scanDir" `
        --scan-target-fraction 0.35 `
        --seed 20260728
    if ($LASTEXITCODE -ne 0) {
        throw "OCR detection label merge failed with code $LASTEXITCODE"
    }
}

$recognition = Get-Process -Id $WaitRecognitionPid -ErrorAction SilentlyContinue
if ($null -ne $recognition) {
    Wait-Process -Id $WaitRecognitionPid
}
$holdoutLabels = Get-Process -Id $WaitHoldoutLabelsPid -ErrorAction SilentlyContinue
if ($null -ne $holdoutLabels) {
    Wait-Process -Id $WaitHoldoutLabelsPid
}
if (Test-Path -LiteralPath $holdoutLabelsFailure -PathType Leaf) {
    throw "Independent OCR holdout detection label preparation failed"
}
if (
    -not (Test-Path -LiteralPath $holdoutTextReport -PathType Leaf) -or
    -not (Test-Path -LiteralPath $holdoutLabelsReport -PathType Leaf)
) {
    throw "Independent OCR holdout reports are missing"
}
& $storageGuard `
    -Stage "ppocrv6-scorescan-detection" `
    -RequiredNewArtifactGiB 1.5

& wsl.exe -d Ubuntu-24.04 -- bash `
    /workspace/pigeon-score-scan/training/run_wsl_ppocrv6_scorescan_detection.sh
if ($LASTEXITCODE -ne 0) {
    throw "PP-OCRv6 ScoreScan detection training or export failed"
}

$modelDir = Join-Path $projectDir "training_data\models\ppocrv6-scorescan-det-stratified-e36-b2-20260729"
$onnxModel = Join-Path $modelDir "scorescan-ppocrv6-det.onnx"
$scanRuntimeLog = Join-Path $modelDir "evaluation.onnx-scan-test.log"
$scanRuntimeReport = Join-Path $modelDir "evaluation.onnx-scan-test.json"
$cleanRuntimeLog = Join-Path $modelDir "evaluation.onnx-clean-test.log"
$cleanRuntimeReport = Join-Path $modelDir "evaluation.onnx-clean-test.json"
$onnxGate = Join-Path $modelDir "onnx-release-gate.json"
$calibrationReport = Join-Path $modelDir "postprocess-calibration.scan.iou075.json"

& $python -m app.tools.calibrate_onnx_ocr_detection_thresholds `
    --model $onnxModel `
    --labels (Join-Path $mergedDir "calibration.scan.paddle.det.txt") `
    --dataset-report $mergedReport `
    --project-root $projectDir `
    --minimum-iou 0.75 `
    --output-report $calibrationReport
if ($LASTEXITCODE -ne 0) {
    throw "Exported ONNX detection threshold calibration failed"
}
$calibration = Get-Content -LiteralPath $calibrationReport -Raw |
    ConvertFrom-Json
if (
    $calibration.test_split_used_for_selection -ne $false -or
    $calibration.minimum_iou -lt 0.75
) {
    throw "Detection threshold calibration violated release isolation"
}
$detThreshold = [double]$calibration.selected.threshold
$detBoxThreshold = [double]$calibration.selected.box_threshold
$detUnclipRatio = [double]$calibration.selected.unclip_ratio

& $python -m app.tools.evaluate_onnx_ocr_detection `
    --model $onnxModel `
    --labels (Join-Path $mergedDir "test.scan.paddle.det.txt") `
    --project-root $projectDir `
    --output-report $scanRuntimeReport `
    --minimum-iou 0.75 `
    --threshold $detThreshold `
    --box-threshold $detBoxThreshold `
    --unclip-ratio $detUnclipRatio `
    2>&1 | Tee-Object -FilePath $scanRuntimeLog
if ($LASTEXITCODE -ne 0) {
    throw "Exported ONNX scan detection evaluation failed with code $LASTEXITCODE"
}
& $python -m app.tools.evaluate_onnx_ocr_detection `
    --model $onnxModel `
    --labels (Join-Path $mergedDir "test.clean.paddle.det.txt") `
    --project-root $projectDir `
    --output-report $cleanRuntimeReport `
    --minimum-iou 0.75 `
    --threshold $detThreshold `
    --box-threshold $detBoxThreshold `
    --unclip-ratio $detUnclipRatio `
    2>&1 | Tee-Object -FilePath $cleanRuntimeLog
if ($LASTEXITCODE -ne 0) {
    throw "Exported ONNX clean detection evaluation failed with code $LASTEXITCODE"
}
& $python -m app.tools.gate_paddleocr_detection `
    --scan-log $scanRuntimeLog `
    --clean-log $cleanRuntimeLog `
    --output-report $onnxGate `
    --model $onnxModel `
    --dataset-report $mergedReport `
    --minimum-scan-precision 0.995 `
    --minimum-scan-recall 0.995 `
    --minimum-scan-hmean 0.995 `
    --minimum-clean-precision 0.995 `
    --minimum-clean-recall 0.995 `
    --minimum-clean-hmean 0.995
if ($LASTEXITCODE -ne 0) {
    throw "Exported ONNX detection model failed its independent release gate"
}

$recognitionModelDir = Join-Path $projectDir "training_data\models\ppocrv6-scorescan-rec-stratified-e18-b8-20260729"
$recognitionModel = Join-Path $recognitionModelDir "scorescan-ppocrv6-rec.onnx"
$recognitionGate = Join-Path $recognitionModelDir "onnx-release-gate.json"
$recognitionKeys = Join-Path $projectDir "training_data\external\corpora\paddleocr_2661c7c0\ppocr\utils\dict\ppocrv6_dict.txt"
$holdoutRecognitionLabels = Join-Path $holdoutLabelsDir "test.paddle.txt"
$holdoutDetectionLabels = Join-Path $holdoutLabelsDir "test.paddle.det.txt"
$holdoutRecognitionReport = Join-Path $recognitionModelDir "evaluation.onnx-independent-holdout.json"
$holdoutDetectionReport = Join-Path $modelDir "evaluation.onnx-independent-holdout.json"
$holdoutGate = Join-Path $modelDir "onnx-independent-holdout-gate.json"

& $python -m app.tools.evaluate_onnx_ocr_recognition `
    --model $recognitionModel `
    --keys $recognitionKeys `
    --labels $holdoutRecognitionLabels `
    --project-root $projectDir `
    --output-report $holdoutRecognitionReport
if ($LASTEXITCODE -ne 0) {
    throw "Recognition ONNX model failed independent holdout evaluation"
}
& $python -m app.tools.evaluate_onnx_ocr_detection `
    --model $onnxModel `
    --labels $holdoutDetectionLabels `
    --project-root $projectDir `
    --output-report $holdoutDetectionReport `
    --minimum-iou 0.75 `
    --threshold $detThreshold `
    --box-threshold $detBoxThreshold `
    --unclip-ratio $detUnclipRatio
if ($LASTEXITCODE -ne 0) {
    throw "Detection ONNX model failed independent holdout evaluation"
}
& $python -m app.tools.gate_ocr_independent_holdout `
    --recognition-report $holdoutRecognitionReport `
    --detection-report $holdoutDetectionReport `
    --detection-calibration-report $calibrationReport `
    --holdout-dataset-report $holdoutTextReport `
    --detection-label-report $holdoutLabelsReport `
    --recognition-model $recognitionModel `
    --recognition-keys $recognitionKeys `
    --detection-model $onnxModel `
    --recognition-labels $holdoutRecognitionLabels `
    --detection-labels $holdoutDetectionLabels `
    --output-report $holdoutGate `
    --minimum-sources 200 `
    --minimum-words 1000 `
    --minimum-pages 100 `
    --minimum-iou 0.75 `
    --minimum-accuracy 0.998 `
    --minimum-normalized-edit 0.9995 `
    --minimum-precision 0.995 `
    --minimum-recall 0.995 `
    --minimum-hmean 0.995
if ($LASTEXITCODE -ne 0) {
    throw "Exported OCR models failed the disjoint registered-scan holdout gate"
}

$ocrResources = Join-Path $projectDir "app\src\scorescan\resources\ocr"
& $python -m app.tools.package_verified_domain_ocr `
    --recognition-model $recognitionModel `
    --recognition-keys $recognitionKeys `
    --recognition-gate $recognitionGate `
    --detection-model $onnxModel `
    --detection-gate $onnxGate `
    --independent-holdout-gate $holdoutGate `
    --output-dir $ocrResources `
    --model-version "scorescan-ppocrv6-domain-stratified-20260729"
if ($LASTEXITCODE -ne 0) {
    throw "Jointly verified OCR model packaging failed"
}
& $python -m app.tools.assert_high_value_gpu_stage `
    --project-root $projectDir `
    --stage "ocr_detection" `
    --output-report (
        Join-Path $projectDir (
            "training_data\logs\high-value-stage-gates\" +
            "ocr-detection-current.json"
        )
    )
if ($LASTEXITCODE -ne 0) {
    throw "OCR detection stage produced no accepted, hash-bound candidate"
}
