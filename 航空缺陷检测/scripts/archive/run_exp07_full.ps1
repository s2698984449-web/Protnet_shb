# exp07 full — cross-domain episodes + early stopping (relative exp04)
param(
    [string]$ExpName = "exp07_crossdom_es_lr002",
    [string]$FastLr = "0.02",
    [int]$AdaptSteps = 5,
    [int]$EvalPatience = 3,
    [int]$Iters = 300
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Write-Host "=== exp07: cross_domain + early_stop | fast_lr=$FastLr adapt_steps=$AdaptSteps patience=$EvalPatience ==="

& $python train.py `
    --exp_name $ExpName `
    --ways 2 --shots 5 `
    --rgb_only True `
    --data_dir_path ../data `
    --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
    --freeze_backbone False `
    --backbone_meta_lr 1e-5 `
    --head_meta_lr 1e-3 `
    --fast_lr $FastLr `
    --adapt_steps $AdaptSteps `
    --meta_batch_size 2 `
    --first_order True `
    --train_domains 0,1,2 `
    --test_domain 3 `
    --cross_domain_episodes True `
    --eval_patience $EvalPatience `
    --early_stop_min_delta 0.001 `
    --w_rl_max 0 `
    --stage1_end 1000 --stage2_end 1000 `
    --eval_tasks 100 `
    --eval_every 50 `
    --iters $Iters `
    --seed 42

Write-Host "=== exp07 training finished ==="
