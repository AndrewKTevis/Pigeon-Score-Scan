$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RuntimeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$UvExe = Join-Path $RuntimeRoot "uv.exe"
$UvHashFile = Join-Path $RuntimeRoot "uv.sha256"
$Archive = Join-Path $RuntimeRoot "uv-download.zip"
$ExtractRoot = Join-Path $RuntimeRoot "uv-extract.tmp"
$UvTemp = Join-Path $RuntimeRoot "uv.exe.tmp"
$HashTemp = Join-Path $RuntimeRoot "uv.sha256.tmp"
$UvUrl = "https://github.com/astral-sh/uv/releases/download/0.9.26/uv-x86_64-pc-windows-msvc.zip"
$ExpectedArchiveSha256 = "eb02fd95d8e0eed462b4a67ecdd320d865b38c560bffcda9a0b87ec944bdf036"

function Remove-Safely([string[]]$Paths) {
    foreach ($Path in $Paths) {
        if (Test-Path -LiteralPath $Path) {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Test-PeX64([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $Stream = $null
    $Reader = $null
    try {
        $Stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        if ($Stream.Length -lt 512) { return $false }
        $Reader = New-Object IO.BinaryReader($Stream)
        if ($Reader.ReadUInt16() -ne 0x5A4D) { return $false }
        $Stream.Position = 0x3C
        $PeOffset = $Reader.ReadInt32()
        if ($PeOffset -lt 0 -or ($PeOffset + 26) -gt $Stream.Length) { return $false }
        $Stream.Position = $PeOffset
        if ($Reader.ReadUInt32() -ne 0x00004550) { return $false }
        if ($Reader.ReadUInt16() -ne 0x8664) { return $false }
        $Stream.Position = $PeOffset + 24
        if ($Reader.ReadUInt16() -ne 0x020B) { return $false }
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $Reader) { $Reader.Dispose() }
        elseif ($null -ne $Stream) { $Stream.Dispose() }
    }
}

function Test-UvExecutable([string]$Path) {
    if (-not (Test-PeX64 $Path)) { return $false }
    Unblock-File -LiteralPath $Path -ErrorAction SilentlyContinue
    $Process = $null
    try {
        $Info = New-Object Diagnostics.ProcessStartInfo
        $Info.FileName = $Path
        $Info.Arguments = "--version"
        $Info.UseShellExecute = $false
        $Info.CreateNoWindow = $true
        $Info.RedirectStandardOutput = $true
        $Info.RedirectStandardError = $true
        $Process = [Diagnostics.Process]::Start($Info)
        if ($null -eq $Process) { return $false }
        if (-not $Process.WaitForExit(30000)) {
            try { $Process.Kill() } catch {}
            return $false
        }
        $Output = $Process.StandardOutput.ReadToEnd().Trim()
        return ($Process.ExitCode -eq 0 -and $Output -match '^uv\s+[0-9]')
    } catch {
        return $false
    } finally {
        if ($null -ne $Process) { $Process.Dispose() }
    }
}

function Test-InstalledUv {
    if (-not (Test-Path -LiteralPath $UvExe -PathType Leaf)) { return $false }
    if (-not (Test-Path -LiteralPath $UvHashFile -PathType Leaf)) { return $false }
    try {
        $Expected = ([IO.File]::ReadAllText($UvHashFile)).Trim().ToLowerInvariant()
        if ($Expected -notmatch '^[0-9a-f]{64}$') { return $false }
        $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $UvExe).Hash.ToLowerInvariant()
        if ($Actual -ne $Expected) { return $false }
        return (Test-UvExecutable $UvExe)
    } catch {
        return $false
    }
}

if (Test-InstalledUv) { exit 0 }
Remove-Safely @($UvExe, $UvHashFile, $Archive, $ExtractRoot, $UvTemp, $HashTemp)

try {
    Invoke-WebRequest -UseBasicParsing -Uri $UvUrl -OutFile $Archive
    Unblock-File -LiteralPath $Archive -ErrorAction SilentlyContinue
    $ArchiveSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
    if ($ArchiveSha256 -ne $ExpectedArchiveSha256) {
        throw "uv archive SHA-256 mismatch"
    }

    Expand-Archive -LiteralPath $Archive -DestinationPath $ExtractRoot -Force
    $Candidate = Get-ChildItem -LiteralPath $ExtractRoot -Filter "uv.exe" -File -Recurse | Select-Object -First 1
    if ($null -eq $Candidate) { throw "uv.exe not found in verified archive" }

    Copy-Item -LiteralPath $Candidate.FullName -Destination $UvTemp -Force
    Unblock-File -LiteralPath $UvTemp -ErrorAction SilentlyContinue
    if (-not (Test-PeX64 $UvTemp)) { throw "downloaded uv.exe is not a PE32+ x64 executable" }
    if (-not (Test-UvExecutable $UvTemp)) { throw "downloaded uv.exe failed its execution smoke test" }

    $UvSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $UvTemp).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText($HashTemp, $UvSha256 + "`r`n", (New-Object Text.UTF8Encoding($false)))

    Move-Item -LiteralPath $UvTemp -Destination $UvExe -Force
    Move-Item -LiteralPath $HashTemp -Destination $UvHashFile -Force
    Unblock-File -LiteralPath $UvExe -ErrorAction SilentlyContinue
    if (-not (Test-InstalledUv)) { throw "installed uv.exe failed final validation" }
} finally {
    Remove-Safely @($Archive, $ExtractRoot, $UvTemp, $HashTemp)
}
