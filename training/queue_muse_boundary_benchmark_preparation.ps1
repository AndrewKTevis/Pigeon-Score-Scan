param(
    [Parameter(Mandatory = $true)]
    [int]$WaitTrainingRegistrationPid,
    [Parameter(Mandatory = $true)]
    [int]$WaitHoldoutRegistrationPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$benchmarkRoot = Join-Path $projectDir "training_data\external\benchmarks\muse_omr_e27f6a8634_raremarks_v3"
$outputDir = Join-Path $projectDir "training_data\benchmarks\muse_omr_boundary_v2"
$manifestPath = Join-Path $outputDir "boundary_manifest.json"
$failurePath = Join-Path $PSScriptRoot "muse_boundary_preparation.failed.txt"
$museScore = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

function Assert-RegistrationCompleted {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PreparedName
    )
    $preparedDir = Join-Path $projectDir "training_data\prepared\$PreparedName"
    $manifest = Join-Path $preparedDir "manifest.json"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw "Registration did not produce its final manifest: $PreparedName"
    }
}

try {
    Remove-Item -LiteralPath $failurePath -Force -ErrorAction SilentlyContinue
    foreach ($registrationPid in @(
        $WaitTrainingRegistrationPid,
        $WaitHoldoutRegistrationPid
    )) {
        $process = Get-Process -Id $registrationPid -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            Wait-Process -Id $registrationPid
        }
    }
    Assert-RegistrationCompleted "muse_omr_scan_regions_stratified_v3"
    Assert-RegistrationCompleted "muse_omr_scan_holdout_regions_stratified_v3"
    if (-not (Test-Path -LiteralPath $museScore -PathType Leaf)) {
        throw "MuseScore executable is missing: $museScore"
    }

    Set-Location -LiteralPath $projectDir
    & $python -m app.tools.prepare_muse_omr_benchmark `
        $benchmarkRoot `
        $outputDir `
        --musescore $museScore `
        --timeout-seconds 240
    if ($LASTEXITCODE -ne 0) {
        throw "Muse OMR boundary preparation exited with code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Muse OMR boundary preparation produced no manifest"
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $coverageGapDetails = @(
        foreach (
            $property in
                $manifest.development_coverage_against_production_shape_minimum.PSObject.Properties
        ) {
            "$($property.Name)=$($property.Value)"
        }
    ) -join ","
    if (
        [int]$manifest.case_count -ne 435 -or
        [int]$manifest.work_count -ne 395 -or
        [int]$manifest.accepted_work_count -lt 200 -or
        [int]$manifest.accepted_submitted_document_count -ne
            [int]$manifest.accepted_work_count -or
        [int]$manifest.accepted_input_page_count -le 0 -or
        $manifest.source_image_origin -ne
            "synthetic_scan_degraded_render" -or
        $manifest.production_evidence_eligible -ne $false -or
        $manifest.production_scope_coverage_complete -ne $false
    ) {
        throw (
            "Muse OMR boundary manifest failed coverage gates: " +
            "cases=$($manifest.case_count), works=$($manifest.work_count), " +
            "accepted_works=$($manifest.accepted_work_count), " +
            "accepted_documents=$($manifest.accepted_submitted_document_count), " +
            "accepted_pages=$($manifest.accepted_input_page_count), " +
            "coverage_gaps=($coverageGapDetails)"
        )
    }
    [ordered]@{
        state = "completed"
        manifest = $manifestPath
        case_count = [int]$manifest.case_count
        accepted_work_count = [int]$manifest.accepted_work_count
        accepted_submitted_document_count = [int]$manifest.accepted_submitted_document_count
        accepted_input_page_count = [int]$manifest.accepted_input_page_count
        pages_by_score_configuration = $manifest.accepted_input_pages_by_score_configuration
        source_image_origin = $manifest.source_image_origin
        production_evidence_eligible = [bool](
            $manifest.production_evidence_eligible
        )
        development_shape_coverage_complete = [bool](
            $manifest.development_shape_coverage_complete
        )
        development_coverage_against_production_shape_minimum = (
            $manifest.development_coverage_against_production_shape_minimum
        )
    } | ConvertTo-Json -Depth 5
}
catch {
    $message = $_.Exception.ToString()
    Set-Content -LiteralPath $failurePath -Value $message -Encoding UTF8
    Write-Error $message
    exit 1
}
