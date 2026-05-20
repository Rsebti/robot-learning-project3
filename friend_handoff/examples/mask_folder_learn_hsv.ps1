param(
    [Parameter(Mandatory = $true)]
    [string]$ImageDir,
    [Parameter(Mandatory = $true)]
    [string]$Color
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
conda activate trim
python -m toolset.perception.cube_mask_batch `
    --image_dir $ImageDir `
    --color $Color `
    --learn_hsv
Write-Host "`nOpen: $(Join-Path $ImageDir 'cube_masks\gallery.html')"
