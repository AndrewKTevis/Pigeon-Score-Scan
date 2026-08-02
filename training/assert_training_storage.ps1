param(
    [string]$Stage = "unspecified",
    [double]$MinimumReserveGiB = 4.0,
    [double]$RequiredNewArtifactGiB = 2.0,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path `
        $projectDir `
        "training_data\logs\storage-capacity-current.json"
}
if ($MinimumReserveGiB -lt 0.0) {
    throw "MinimumReserveGiB must not be negative"
}
if ($RequiredNewArtifactGiB -lt 0.0) {
    throw "RequiredNewArtifactGiB must not be negative"
}

$projectItem = Get-Item -LiteralPath $projectDir -ErrorAction Stop
$drive = $projectItem.PSDrive
if ($null -eq $drive -or [string]::IsNullOrWhiteSpace($drive.Root)) {
    throw "Unable to resolve the project volume: $projectDir"
}
$driveInfo = [System.IO.DriveInfo]::new($drive.Root)
$freeBytes = [long]$driveInfo.AvailableFreeSpace
$requiredBytes = [long](
    ($MinimumReserveGiB + $RequiredNewArtifactGiB) * 1GB
)
$passed = $freeBytes -ge $requiredBytes

$payload = [ordered]@{
    format = 1
    name = "scorescan-training-storage-capacity-v1"
    checked_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    stage = $Stage
    project_dir = $projectDir
    volume_root = $driveInfo.RootDirectory.FullName
    free_bytes = $freeBytes
    free_gib = $freeBytes / 1GB
    minimum_reserve_bytes = [long]($MinimumReserveGiB * 1GB)
    minimum_reserve_gib = $MinimumReserveGiB
    required_new_artifact_bytes = [long]($RequiredNewArtifactGiB * 1GB)
    required_new_artifact_gib = $RequiredNewArtifactGiB
    required_free_bytes = $requiredBytes
    required_free_gib = $requiredBytes / 1GB
    passed = $passed
}

$outputParent = Split-Path -Parent $Output
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
$temporaryOutput = Join-Path `
    $outputParent `
    ("." + (Split-Path -Leaf $Output) + "." + $PID + ".tmp")
try {
    $payload |
        ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $temporaryOutput -Encoding UTF8
    Move-Item -LiteralPath $temporaryOutput -Destination $Output -Force
} finally {
    Remove-Item `
        -LiteralPath $temporaryOutput `
        -Force `
        -ErrorAction SilentlyContinue
}

$payload | ConvertTo-Json -Compress
if (-not $passed) {
    throw (
        "Insufficient project-volume capacity for stage '{0}': " +
        "{1:N2} GiB free, {2:N2} GiB required " +
        "({3:N2} GiB artifact allowance + {4:N2} GiB reserve)."
    ) -f @(
        $Stage,
        ($freeBytes / 1GB),
        ($requiredBytes / 1GB),
        $RequiredNewArtifactGiB,
        $MinimumReserveGiB
    )
}
