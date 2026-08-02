param(
    [Parameter(Mandatory = $true)]
    [int]$ProbeValue
)

$ErrorActionPreference = "Stop"
if ($ProbeValue -ne 42) {
    throw "Named parameter was not bound correctly"
}
