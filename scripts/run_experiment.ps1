# Unified experiment runner (wraps scripts/run_experiment.py)
param(
    [string]$Exp = "",
    [string]$Grid = "",
    [string]$Config = "",
    [ValidateSet("", "train", "eval", "diagnose", "pretrain")]
    [string]$Stage = "",
    [switch]$List,
    [switch]$DryRun,
    [string[]]$Set = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$ProjectDir = Split-Path $ScriptDir -Parent

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$argsList = @("$ScriptDir\run_experiment.py")
if ($Config) {
    $cfgPath = if ($Config -match '[\\/]') { $Config } else { Join-Path $ProjectDir "experiments\$Config" }
    $argsList += @("--config", $cfgPath)
}
if ($List) { $argsList += "--list" }
if ($Exp) { $argsList += @("--exp", $Exp) }
if ($Grid) { $argsList += @("--grid", $Grid) }
if ($Stage) { $argsList += @("--stage", $Stage) }
if ($DryRun) { $argsList += "--dry-run" }
foreach ($s in $Set) { $argsList += @("--set", $s) }

if (-not $List -and -not $Exp -and -not $Grid) {
    Write-Host @"
Usage:
  .\scripts\run_experiment.ps1 -List
  .\scripts\run_experiment.ps1 -Exp exp08_protonet -Stage train
  .\scripts\run_experiment.ps1 -Exp exp08_eval
  .\scripts\run_experiment.ps1 -Grid exp04_fastlr
  .\scripts\run_experiment.ps1 -Config cmpl.yaml -List
  .\scripts\run_experiment.ps1 -Config cmpl.yaml -Exp cmpl_p0_v3 -Stage pretrain

Registry: experiments/aviation.yaml | experiments/cmpl.yaml
"@
    exit 0
}

Set-Location $ProjectDir
& $python @argsList
exit $LASTEXITCODE
