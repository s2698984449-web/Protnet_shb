# 3-seed final eval for exp06 best checkpoint (compare to exp04 pooled 0.679)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$ckpt = "./checkpoints/exp06_epaug_unfreeze_lr002/best.pt"
$out = "./reports/final_eval_exp06_full.json"

Write-Host "=== Final eval: exp06 best.pt | seeds 42,43,44 ==="

& $python final_eval.py `
    --checkpoint $ckpt `
    --data_dir_path ../data `
    --test_domain 3 `
    --train_domains 0,1,2 `
    --ways 2 --shots 5 `
    --adapt_steps 5 `
    --fast_lr 0.02 `
    --freeze_backbone False `
    --rgb_only True `
    --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
    --eval_tasks 100 `
    --seeds 42,43,44 `
    --out_json $out

Write-Host "=== Done. Compare pooled mean with exp04 (0.679, CI low 0.656) ==="
