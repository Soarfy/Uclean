param(
    [string]$EnvironmentPython = "D:\SoftWare\Anaconda\envs\Uclean\python.exe",
    [string]$Version = "v2.0.0"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SpecFile = Join-Path $PSScriptRoot "SelectableCleanUI.spec"
$BuildRoot = Join-Path $ProjectRoot "portable_build"
$DistRoot = Join-Path $ProjectRoot "portable_dist"

if (-not (Test-Path -LiteralPath $EnvironmentPython -PathType Leaf)) {
    throw "Uclean environment Python was not found: $EnvironmentPython"
}

& $EnvironmentPython -m PyInstaller `
    --noconfirm `
    --clean `
    --workpath $BuildRoot `
    --distpath $DistRoot `
    $SpecFile

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

& (Join-Path $PSScriptRoot "Finalize-Portable.ps1") -Version $Version
