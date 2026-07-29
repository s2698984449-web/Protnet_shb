# ProtoNet inference on exp04 MAML checkpoint (no retrain — diagnostic only)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Write-Host "=== ProtoNet eval on exp04 MAML weights (inference-only) ==="

& $python final_eval.py `
    --checkpoint ./checkpoints/exp04_full_unfreeze_lr002/best.pt `
    --meta_algorithm protonet `
    --protonet_distance euclidean `
    --data_dir_path ../data `
    --test_domain 3 `
    --train_domains 0,1,2 `
    --ways 2 --shots 5 `
    --freeze_backbone False `
    --rgb_only True `
    --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
    --eval_tasks 100 `
    --seeds 42,43,44 `
    --out_json ./reports/final_eval_exp04_as_protonet.json

Write-Host "=== Done ==="
