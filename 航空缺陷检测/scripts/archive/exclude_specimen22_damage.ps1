# Read exclude list for specimen 22 damage (relative path, no hardcoded Chinese path)
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path $PSScriptRoot -Parent
$listFile = Join-Path $ProjectDir "..\data\aviation\domain_3\3_damage\exclude_qc_specimen22.txt"
$listFile = (Resolve-Path -LiteralPath $listFile).Path
$ids = Get-Content -LiteralPath $listFile -Encoding UTF8 | Where-Object { $_ -and -not $_.StartsWith('#') }
($ids -join ',')
