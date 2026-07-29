# P0: hard-task diagnosis for exp08 ProtoNet
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Write-Host "=== P0 hard-task diagnosis (exp08 ProtoNet, domain_3) ==="

& $python scripts/diagnose_hard_tasks.py `
    --checkpoint ./checkpoints/exp08_protonet_unfreeze/best.pt `
    --seeds 42,43,44 `
    --eval_tasks 100 `
    --hard_threshold 0.7 `
    --out_dir ./reports/diagnose `
    --out_name hard_tasks_exp08.json

Write-Host "=== Done. See reports/diagnose/hard_tasks_exp08.json ==="
