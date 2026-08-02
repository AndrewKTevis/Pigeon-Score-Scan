param(
    [int]$ParallelTransfers = 16,
    [int64]$ChunkBytes = 4194304,
    [int]$MaximumAttempts = 8
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams"
$expectedBytes = 67465190L
$expectedMd5 = "7b3c850d9ced2ba2c3d11a5c206e3986"
$modelDir = Join-Path $projectDir "training_data\external\models\paddleocr-ppocrv6-medium-det-training"
$canonical = Join-Path $modelDir "PP-OCRv6_medium_det_pretrained.pdparams"
$complete = "$canonical.complete"
$chunkDir = Join-Path $projectDir "tmp\ppocrv6-det-weight-chunks"
$provenance = Join-Path $modelDir "provenance.json"

New-Item -ItemType Directory -Force -Path $modelDir, $chunkDir | Out-Null

function Test-VerifiedWeight([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    if ((Get-Item -LiteralPath $Path).Length -ne $expectedBytes) {
        return $false
    }
    $md5 = (Get-FileHash -LiteralPath $Path -Algorithm MD5).Hash.ToLowerInvariant()
    return $md5 -eq $expectedMd5
}

if (Test-VerifiedWeight $canonical) {
    Write-Output "Verified PP-OCRv6 detection weight already exists."
    exit 0
}

for ($attempt = 1; $attempt -le $MaximumAttempts; $attempt++) {
    $curlArguments = @(
        "--fail",
        "--silent",
        "--show-error",
        "--parallel",
        "--parallel-max",
        "$ParallelTransfers"
    )
    $transferCount = 0
    for ($start = 0L; $start -lt $expectedBytes; $start += $ChunkBytes) {
        $end = [Math]::Min($expectedBytes - 1, $start + $ChunkBytes - 1)
        $index = [int]($start / $ChunkBytes)
        $part = Join-Path $chunkDir ("part-{0:D3}.bin" -f $index)
        $partBytes = $end - $start + 1
        if (
            (Test-Path -LiteralPath $part -PathType Leaf) -and
            (Get-Item -LiteralPath $part).Length -eq $partBytes
        ) {
            continue
        }
        if ($transferCount -gt 0) {
            $curlArguments += "--next"
        }
        $curlArguments += @(
            "--range",
            "$start-$end",
            "--output",
            $part,
            $url
        )
        $transferCount++
    }
    if ($transferCount -eq 0) {
        break
    }
    Write-Output (
        "Attempt {0}/{1}: downloading {2} verified range chunks with {3} transfers" -f
        $attempt,
        $MaximumAttempts,
        $transferCount,
        $ParallelTransfers
    )
    & curl.exe @curlArguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "curl range attempt failed with code $LASTEXITCODE"
    }
}

$parts = Get-ChildItem -LiteralPath $chunkDir -Filter "part-*.bin" |
    Sort-Object Name
$expectedPartCount = [int][Math]::Ceiling($expectedBytes / $ChunkBytes)
if ($parts.Count -ne $expectedPartCount) {
    throw "Wrong verified range part count: $($parts.Count) != $expectedPartCount"
}
$sum = ($parts | Measure-Object Length -Sum).Sum
if ($sum -ne $expectedBytes) {
    throw "Wrong verified range byte count: $sum != $expectedBytes"
}

$output = [IO.File]::Open(
    $complete,
    [IO.FileMode]::Create,
    [IO.FileAccess]::Write,
    [IO.FileShare]::None
)
try {
    foreach ($part in $parts) {
        $inputStream = [IO.File]::OpenRead($part.FullName)
        try {
            $inputStream.CopyTo($output)
        }
        finally {
            $inputStream.Dispose()
        }
    }
}
finally {
    $output.Dispose()
}

if (-not (Test-VerifiedWeight $complete)) {
    throw "Assembled detection weight failed official size/MD5 verification"
}
$sha256 = (
    Get-FileHash -LiteralPath $complete -Algorithm SHA256
).Hash.ToLowerInvariant()

if (Test-Path -LiteralPath $canonical -PathType Leaf) {
    $invalidBytes = (Get-Item -LiteralPath $canonical).Length
    $backup = "$canonical.unverified-$invalidBytes"
    if (Test-Path -LiteralPath $backup) {
        $backup = "$backup-$(Get-Date -Format 'yyyyMMddHHmmss')"
    }
    Move-Item -LiteralPath $canonical -Destination $backup
}
Move-Item -LiteralPath $complete -Destination $canonical

@{
    schema_version = 1
    source = $url
    source_type = "official Paddle model ecology object storage"
    expected_bytes = $expectedBytes
    etag_md5 = $expectedMd5
    sha256 = $sha256
    retrieved_utc = (Get-Date).ToUniversalTime().ToString("o")
    integration_authorized = $false
    warning = "Pretrained initialization only; never a release model."
} |
    ConvertTo-Json |
    Set-Content -LiteralPath $provenance -Encoding UTF8

Write-Output (
    "Verified detection initialization weight: bytes={0} md5={1} sha256={2}" -f
    $expectedBytes,
    $expectedMd5,
    $sha256
)
