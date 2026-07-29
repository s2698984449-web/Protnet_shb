# List domain_3/3_damage folders; mark specimen-22 QC excludes
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path $PSScriptRoot -Parent
$root = (Resolve-Path (Join-Path $ProjectDir "..\data\aviation\domain_3\3_damage")).Path
$listFile = Join-Path $root "exclude_qc_specimen22.txt"
$exclude = @{}
Get-Content -LiteralPath $listFile -Encoding UTF8 | Where-Object { $_ -and -not $_.StartsWith('#') } | ForEach-Object { $exclude[$_.Trim()] = $true }

Write-Host "=== domain_3/3_damage (total $(@(Get-ChildItem -LiteralPath $root -Directory).Count) folders) ==="
Write-Host ""
Write-Host "[SKIP - specimen 22 damage QC]"
Get-ChildItem -LiteralPath $root -Directory | Sort-Object Name | ForEach-Object {
    if ($exclude.ContainsKey($_.Name)) { Write-Host "  SKIP  $($_.Name)" }
}
Write-Host ""
Write-Host "[KEEP - other specimens]"
Get-ChildItem -LiteralPath $root -Directory | Sort-Object Name | ForEach-Object {
    if (-not $exclude.ContainsKey($_.Name)) { Write-Host "  KEEP  $($_.Name)" }
}
Write-Host ""
Write-Host "Note: 0_scratch/22_1, 22_2 are NOT in 3_damage — they stay in eval."
