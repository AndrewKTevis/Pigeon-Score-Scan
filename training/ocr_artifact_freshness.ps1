function Test-OcrArtifactFreshness {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Python,
        [Parameter(Mandatory = $true)]
        [string]$ProjectDir,
        [Parameter(Mandatory = $true)]
        [ValidateSet("text", "holdout-labels", "merged-labels")]
        [string]$Kind,
        [Parameter(Mandatory = $true)]
        [string]$ArtifactReport,
        [Parameter(Mandatory = $true)]
        [string[]]$SourceReports
    )

    $arguments = @(
        "-m",
        "app.tools.validate_ocr_artifact_freshness",
        "--kind",
        $Kind,
        "--artifact-report",
        $ArtifactReport
    )
    foreach ($sourceReport in $SourceReports) {
        $arguments += @("--source-report", $sourceReport)
    }
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Python
    $startInfo.WorkingDirectory = $ProjectDir
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.Arguments = (
        $arguments |
            ForEach-Object {
                # Validator arguments are local paths/options and never end in a
                # backslash. Quote every token so workspace spaces are preserved.
                '"' + ([string]$_).Replace('"', '\"') + '"'
            }
    ) -join " "
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    try {
        # A stale artifact is an expected boolean outcome. Windows PowerShell
        # promotes a native validator's diagnostic stderr to a terminating
        # NativeCommandError when the caller uses Stop globally. Run it through
        # ProcessStartInfo so only its exit code controls this boolean helper.
        if (-not $process.Start()) {
            throw "OCR artifact freshness validator did not start"
        }
        $standardOutput = $process.StandardOutput.ReadToEnd()
        $standardError = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        $validatorExitCode = $process.ExitCode
    } finally {
        $process.Dispose()
    }
    return $validatorExitCode -eq 0
}

function Move-StaleOcrArtifact {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ArtifactDir,
        [Parameter(Mandatory = $true)]
        [string]$PreparedRoot
    )

    $artifact = [IO.DirectoryInfo]::new([IO.Path]::GetFullPath($ArtifactDir))
    $prepared = [IO.DirectoryInfo]::new([IO.Path]::GetFullPath($PreparedRoot))
    if (
        -not $artifact.Exists -or
        $null -eq $artifact.Parent -or
        $artifact.Parent.FullName -ne $prepared.FullName
    ) {
        throw "Refusing to quarantine an OCR artifact outside the prepared-data root"
    }
    $suffix = [DateTimeOffset]::Now.ToString("yyyyMMdd-HHmmss")
    $quarantine = Join-Path `
        $prepared.FullName `
        "$($artifact.Name).stale-$suffix-$PID"
    if (Test-Path -LiteralPath $quarantine) {
        throw "OCR artifact quarantine destination already exists: $quarantine"
    }
    Move-Item -LiteralPath $artifact.FullName -Destination $quarantine
    Write-Output "Quarantined stale OCR artifact: $quarantine"
}
