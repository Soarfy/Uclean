param([string]$Version = "v2.0.0")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DistRoot = Join-Path $ProjectRoot "portable_dist"
$PackageDir = Get-ChildItem -LiteralPath $DistRoot -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
$ReadmeSource = Get-ChildItem -LiteralPath $PSScriptRoot -Filter "*.txt" |
    Select-Object -First 1 -ExpandProperty FullName
$SmokeLog = Join-Path $PackageDir "portable_smoke_test.log"
$ZipPath = Join-Path $DistRoot "UClean-Windows-Portable-$Version.zip"

Copy-Item -LiteralPath $ReadmeSource -Destination $PackageDir -Force
if (Test-Path -LiteralPath $SmokeLog) {
    Remove-Item -LiteralPath $SmokeLog -Force
}
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -LiteralPath $PackageDir -DestinationPath $ZipPath -CompressionLevel Optimal
Write-Host $ZipPath
