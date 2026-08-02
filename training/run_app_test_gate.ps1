param(
    [string]$Output = "",
    [string]$Report = "",
    [string[]]$PytestTarget = @("app/tests"),
    [string]$BaseTempRoot = ""
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir "app\.venv\Scripts\python.exe"
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $projectDir "training_data\logs\app-tests-current.out.log"
}
if ([string]::IsNullOrWhiteSpace($Report)) {
    $Report = Join-Path $projectDir "training_data\logs\app-tests-current.json"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Application test Python is missing: $python"
}

if ([string]::IsNullOrWhiteSpace($BaseTempRoot)) {
    if (Test-Path -LiteralPath "E:\" -PathType Container) {
        $BaseTempRoot = "E:\ScoreScan-Ephemeral\pytest"
    } else {
        $BaseTempRoot = Join-Path $projectDir "training_data\tmp\pytest"
    }
}
$baseTempRootItem = New-Item `
    -ItemType Directory `
    -Path $BaseTempRoot `
    -Force
$baseTempRootResolved = $baseTempRootItem.FullName.TrimEnd("\")
$baseTempDrive = $baseTempRootItem.PSDrive
$baseTempDriveInfo = [System.IO.DriveInfo]::new($baseTempDrive.Root)
if ($baseTempDriveInfo.AvailableFreeSpace -lt 2GB) {
    throw (
        "Application test temp volume has less than 2 GiB free: " +
        $baseTempDriveInfo.RootDirectory.FullName
    )
}
$pytestBaseTemp = Join-Path `
    $baseTempRootResolved `
    ("app-tests-" + $PID + "-" + [Guid]::NewGuid().ToString("N"))

$outputParent = Split-Path -Parent $Output
$reportParent = Split-Path -Parent $Report
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
New-Item -ItemType Directory -Path $reportParent -Force | Out-Null
$temporaryOutput = Join-Path `
    $outputParent `
    ("." + (Split-Path -Leaf $Output) + "." + $PID + ".tmp")
$temporaryReport = Join-Path `
    $reportParent `
    ("." + (Split-Path -Leaf $Report) + "." + $PID + ".tmp")

$started = [DateTimeOffset]::UtcNow
Set-Location -LiteralPath $projectDir
try {
    # Windows PowerShell 5 adapts a native program's stderr into non-terminating
    # ErrorRecord objects. Deprecation warnings must remain in the captured log
    # and must not bypass the real pytest exit code.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python -m pytest -q -ra -p no:cacheprovider `
        --basetemp $pytestBaseTemp `
        @PytestTarget *> $temporaryOutput
    $testExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    Move-Item `
        -LiteralPath $temporaryOutput `
        -Destination $Output `
        -Force
    $completed = [DateTimeOffset]::UtcNow
    $summary = @(
        Get-Content -LiteralPath $Output -Tail 8 -Encoding UTF8 |
            ForEach-Object { [string]$_ }
    )
    $payload = [ordered]@{
        format = 1
        command = (
            "python -m pytest -q -ra -p no:cacheprovider " +
            "--basetemp <ephemeral> " +
            ($PytestTarget -join " ")
        )
        passed = ($testExitCode -eq 0)
        exit_code = $testExitCode
        started_at = $started.ToString("o")
        completed_at = $completed.ToString("o")
        elapsed_seconds = ($completed - $started).TotalSeconds
        output = $Output
        output_sha256 = (
            Get-FileHash -LiteralPath $Output -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        ephemeral_base_temp_root = $baseTempRootResolved
        summary = $summary
    }
    $payload |
        ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $temporaryReport -Encoding UTF8
    Move-Item `
        -LiteralPath $temporaryReport `
        -Destination $Report `
        -Force
    exit $testExitCode
} catch {
    $failure = $_ | Out-String
    if (Test-Path -LiteralPath $temporaryOutput -PathType Leaf) {
        Move-Item `
            -LiteralPath $temporaryOutput `
            -Destination $Output `
            -Force
    }
    $failurePayload = [ordered]@{
        format = 1
        command = (
            "python -m pytest -q -ra -p no:cacheprovider " +
            "--basetemp <ephemeral> " +
            ($PytestTarget -join " ")
        )
        passed = $false
        exit_code = 255
        started_at = $started.ToString("o")
        completed_at = [DateTimeOffset]::UtcNow.ToString("o")
        output = $Output
        ephemeral_base_temp_root = $baseTempRootResolved
        infrastructure_error = $failure
    }
    $failurePayload |
        ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $temporaryReport -Encoding UTF8
    Move-Item `
        -LiteralPath $temporaryReport `
        -Destination $Report `
        -Force
    throw
} finally {
    Remove-Item -LiteralPath $temporaryOutput -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $temporaryReport -Force -ErrorAction SilentlyContinue
    $baseTempPrefix = $baseTempRootResolved + "\"
    if (
        $pytestBaseTemp.StartsWith(
            $baseTempPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -and
        $pytestBaseTemp -ne $baseTempRootResolved
    ) {
        Remove-Item `
            -LiteralPath $pytestBaseTemp `
            -Recurse `
            -Force `
            -ErrorAction SilentlyContinue
    }
}
