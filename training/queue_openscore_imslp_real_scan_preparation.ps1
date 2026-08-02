param(
    [Parameter(Mandatory = $true)]
    [int]$WaitAcquisitionPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$museScore = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
$byteManifest = Join-Path $projectDir (
    "training_data\benchmarks\" +
    "openscore_imslp_quartet_scan_bytes_v1.json"
)
$scoreRoot = Join-Path $projectDir (
    "training_data\external\corpora\" +
    "openscore_string_quartets_d13289cd\" +
    "StringQuartets-d13289cd70797da94646e5cf64f7296a4c4fee40"
)
$outputDir = Join-Path $projectDir (
    "training_data\benchmarks\" +
    "openscore_imslp_real_scan_semantic_v1"
)
$semanticManifest = Join-Path $outputDir "semantic_manifest.json"
$boundaryContract = "printed-western-instrumental-scan-boundary@4"
$textPreparationReports = @(
    (
        "training_data\prepared\muse_omr_scan_text_stratified_v3\" +
        "prepare-report.json"
    ),
    (
        "training_data\prepared\" +
        "muse_omr_scan_holdout_text_stratified_v3\prepare-report.json"
    )
)

$acquisition = Get-Process `
    -Id $WaitAcquisitionPid `
    -ErrorAction SilentlyContinue
if ($null -ne $acquisition) {
    Wait-Process -Id $WaitAcquisitionPid
}
if (-not (Test-Path -LiteralPath $byteManifest -PathType Leaf)) {
    throw "Full exact-mirror scan acquisition is incomplete: $byteManifest"
}
$bytes = Get-Content `
    -LiteralPath $byteManifest `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
if (
    $bytes.role -ne
        "downloaded_imslp_scan_bytes_not_aligned_or_training_authorized" -or
    [int]$bytes.source_count -ne 6 -or
    [int]$bytes.work_count -ne 6 -or
    [int]$bytes.page_count -ne 209 -or
    $bytes.archive_mirror_required -ne $true -or
    $bytes.training_authorized -ne $false -or
    $bytes.evaluation_authorized -ne $false -or
    $bytes.release_authorized -ne $false
) {
    throw "Full exact-mirror scan byte manifest failed its identity/use gate"
}

# MuseScore command-line export is not safely re-entrant at high concurrency on
# Windows.  Let the two large registered-text preparations finish before these
# six reference exports instead of causing nondeterministic native crashes.
foreach ($relativeReport in $textPreparationReports) {
    $report = Join-Path $projectDir $relativeReport
    while (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
        Start-Sleep -Seconds 10
    }
}

Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = (
    (Join-Path $projectDir "app\src"),
    $projectDir
) -join [IO.Path]::PathSeparator
& $python -m app.tools.prepare_openscore_real_scan_semantic_corpus `
    $byteManifest `
    $scoreRoot `
    $outputDir `
    --musescore $museScore `
    --timeout-seconds 300
if ($LASTEXITCODE -ne 0) {
    throw "OpenScore exact-mirror semantic preparation failed"
}
if (-not (Test-Path -LiteralPath $semanticManifest -PathType Leaf)) {
    throw "OpenScore exact-mirror semantic manifest is missing"
}
$semantic = Get-Content `
    -LiteralPath $semanticManifest `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
if (
    $semantic.role -ne
        (
            "external_real_scan_split_inherited_semantic_development_" +
            "not_release_holdout"
        ) -or
    $semantic.boundary_contract_version -ne $boundaryContract -or
    $semantic.boundary_audit_complete -ne $true -or
    [int]$semantic.source_count -ne 6 -or
    [int]$semantic.source_work_count -ne 6 -or
    [int]$semantic.source_page_count -ne 209 -or
    (
        [int]$semantic.case_count +
        [int]$semantic.excluded_source_count
    ) -ne 6 -or
    (
        [int]$semantic.page_count +
        [int]$semantic.excluded_page_count
    ) -ne 209 -or
    $semantic.training_authorized -ne $false -or
    $semantic.page_training_labels_authorized -ne $false -or
    $semantic.release_evaluation_authorized -ne $false -or
    $semantic.release_authorized -ne $false -or
    $semantic.independent_holdout -ne $false
) {
    throw "OpenScore exact-mirror semantic corpus failed its boundary/use gate"
}

[ordered]@{
    state = "completed"
    audited_work_count = [int]$semantic.source_work_count
    audited_page_count = [int]$semantic.source_page_count
    accepted_work_count = [int]$semantic.work_count
    accepted_page_count = [int]$semantic.page_count
    excluded_work_count = [int]$semantic.excluded_source_count
    excluded_page_count = [int]$semantic.excluded_page_count
    whole_work_development_only = $true
    page_training_authorized = $false
    release_evaluation_authorized = $false
} | ConvertTo-Json -Compress
