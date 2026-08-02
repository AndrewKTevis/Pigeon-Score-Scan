param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
$corpusDir = Join-Path $projectDir "training_data\external\corpora\openscore_lieder_6b2dc542\Lieder-6b2dc542ce2e8aa4b78c8ee62103b210efc07015"
$sourceList = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_sources.txt"
$legacyRegionDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_svg_regions_v1"
$regionDir = Join-Path $projectDir "training_data\prepared\openscore_lieder_train_1091_svg_regions_complete_page_v2"
$completedReport = Join-Path $regionDir "prepare-report.json"
$failureReport = Join-Path $regionDir "preparation.failed.txt"
$museScoreExe = "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
$visibilityVersion = "complete-page-oversized-axis-overlap-fragments@1"
$targetAssignmentVersion = (
    "complete-page-overlap-consistent-deduplicated-semantic-targets@4"
)
$builderPath = Join-Path $projectDir (
    "app\tools\prepare_openscore_svg_regions.py"
)
$builderHash = (
    Get-FileHash -LiteralPath $builderPath -Algorithm SHA256
).Hash.ToLowerInvariant()

$waitProcess = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
if ($null -ne $waitProcess) {
    Wait-Process -Id $WaitPid
}

Set-Location -LiteralPath $projectDir
try {
    if (Test-Path -LiteralPath $completedReport -PathType Leaf) {
        $existing = Get-Content -LiteralPath $completedReport -Raw -Encoding UTF8 |
            ConvertFrom-Json
        if (
            $existing.oversized_fragment_visibility_version -ne
                $visibilityVersion -or
            $existing.target_assignment_version -ne
                $targetAssignmentVersion -or
            $existing.builder_source_sha256 -ne $builderHash -or
            [int]$existing.tile_size -ne 1024 -or
            [int]$existing.overlap -ne 256 -or
            $existing.role -ne
                "training_only_synthetic_semantic_geometry"
        ) {
            Remove-Item -LiteralPath $completedReport -Force
        }
    }
    if (-not (Test-Path -LiteralPath $completedReport -PathType Leaf)) {
        & $python -m app.tools.seed_openscore_render_cache `
            --source $legacyRegionDir `
            --destination $regionDir
        if ($LASTEXITCODE -ne 0) {
            throw "OpenScore Lieder render-cache seed failed"
        }
        & $python -m app.tools.prepare_openscore_svg_regions `
            --corpus-dir $corpusDir `
            --source-list $sourceList `
            --output-dir $regionDir `
            --musescore-exe $museScoreExe `
            --negative-ratio 0.08 `
            --long-span-minimum-object-fraction 0.25 `
            --resume
        if ($LASTEXITCODE -ne 0) {
            throw "OpenScore Lieder 1091 preparation exited with code $LASTEXITCODE"
        }
    }
    $report = Get-Content -LiteralPath $completedReport -Raw -Encoding UTF8 |
        ConvertFrom-Json
    if (
        $report.target_geometry_provenance -ne
            "complete-page-svg-geometry-before-tile-clipping@1" -or
        [double]$report.long_span_minimum_object_fraction -ne 0.25 -or
        $report.oversized_fragment_visibility_version -ne
            $visibilityVersion -or
        $report.target_assignment_version -ne
            $targetAssignmentVersion -or
        $report.builder_source_sha256 -ne $builderHash -or
        [int]$report.tile_size -ne 1024 -or
        [int]$report.overlap -ne 256 -or
        $report.role -ne
            "training_only_synthetic_semantic_geometry"
    ) {
        throw "OpenScore Lieder complete-page target evidence is invalid"
    }
} catch {
    New-Item -ItemType Directory -Path $regionDir -Force | Out-Null
    $_.Exception.Message | Set-Content -LiteralPath $failureReport -Encoding UTF8
    throw
}
