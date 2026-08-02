param(
    [Parameter(Mandatory = $true)]
    [string]$TargetScript,
    [Parameter(Mandatory = $true)]
    [string]$CompletionPath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TargetArguments = @()
)

$ErrorActionPreference = "Stop"
$exitCode = 1
$errorMessage = ""
try {
    if (-not (Test-Path -LiteralPath $TargetScript -PathType Leaf)) {
        throw "Preparation target script is missing: $TargetScript"
    }
    $powershell = Join-Path $PSHOME "powershell.exe"
    & $powershell `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File $TargetScript `
        @TargetArguments
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Preparation target exited with code $LASTEXITCODE`: " +
            $TargetScript
        )
    }
    $exitCode = 0
} catch {
    $errorMessage = $_.Exception.Message
    Write-Error $_
} finally {
    $completionDirectory = Split-Path -Parent $CompletionPath
    New-Item `
        -ItemType Directory `
        -Path $completionDirectory `
        -Force |
        Out-Null
    $temporary = (
        "$CompletionPath.tmp-" +
        [Guid]::NewGuid().ToString("N")
    )
    [ordered]@{
        format = 1
        target_script = $TargetScript
        target_arguments = @($TargetArguments)
        exit_code = $exitCode
        error = $errorMessage
        completed_at = [DateTimeOffset]::UtcNow.ToString("o")
    } |
        ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item `
        -LiteralPath $temporary `
        -Destination $CompletionPath `
        -Force
}
exit $exitCode
