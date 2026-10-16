param(
    [Parameter(Mandatory=$true)][string]$ModelRepo,
    [Parameter(Mandatory=$true)][string]$InputVideo,
    [Parameter(Mandatory=$true)][string]$Mask,
    [string]$OutputDirectory = "results",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath $ModelRepo).Path
$video = (Resolve-Path -LiteralPath $InputVideo).Path
$maskPath = (Resolve-Path -LiteralPath $Mask).Path
$outPath = [System.IO.Path]::GetFullPath($OutputDirectory)
$inference = Join-Path $repo "inference_matanyone2.py"
if (!(Test-Path -LiteralPath $inference)) { throw "Missing inference_matanyone2.py in ModelRepo." }
Push-Location $repo
try {
    & $Python $inference -i $video -m $maskPath -o $outPath --save_image
    if ($LASTEXITCODE -ne 0) { throw "MatAnyone 2 inference failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}
