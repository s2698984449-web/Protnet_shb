# 3-seed final eval for ProtoNet checkpoint
param(
    [string]$ExpName = "exp08_protonet_unfreeze",
    [string]$Distance = "euclidean"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$ckpt = "./checkpoints/$ExpName/best.pt"
$out = "./reports/final_eval_${ExpName}.json"

Write-Host "=== Final eval ProtoNet: $ExpName | seeds 42,43,44 ==="

& $python final_eval.py `
    --checkpoint $ckpt `
    --meta_algorithm protonet `
    --protonet_distance $Distance `
    --data_dir_path ../data `
    --test_domain 3 `
    --train_domains 0,1,2 `
    --ways 2 --shots 5 `
    --freeze_backbone False `
    --rgb_only True `
    --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
    --eval_tasks 100 `
    --seeds 42,43,44 `
    --out_json $out

Write-Host "=== Done. Compare with exp04 MAML (0.679) ==="
