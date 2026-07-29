# exp08 final eval after QC delete (approach B — 22 damage folders removed from disk)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Write-Host "=== exp08 final eval (domain_3 after QC delete specimen 22 damage) ==="

& $python final_eval.py `
    --checkpoint ./checkpoints/exp08_protonet_unfreeze/best.pt `
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
    --out_json ./reports/final_eval_exp08_qc_deleted.json

Write-Host "=== Done ==="
