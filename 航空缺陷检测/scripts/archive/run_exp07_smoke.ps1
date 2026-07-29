# exp07 smoke test — cross-domain + early stop (20 iters)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Write-Host "=== exp07 smoke: cross_domain + eval_patience=2 | 20 iters ==="

& $python train.py `
    --exp_name exp07_smoke_crossdom `
    --ways 2 --shots 5 `
    --rgb_only True `
    --data_dir_path ../data `
    --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
    --freeze_backbone False `
    --backbone_meta_lr 1e-5 `
    --head_meta_lr 1e-3 `
    --fast_lr 0.02 `
    --adapt_steps 5 `
    --meta_batch_size 2 `
    --first_order True `
    --train_domains 0,1,2 `
    --test_domain 3 `
    --cross_domain_episodes True `
    --eval_patience 2 `
    --early_stop_min_delta 0.0 `
    --w_rl_max 0 `
    --stage1_end 1000 --stage2_end 1000 `
    --eval_tasks 20 `
    --eval_every 10 `
    --iters 20 `
    --seed 42

Write-Host "=== smoke done ==="
