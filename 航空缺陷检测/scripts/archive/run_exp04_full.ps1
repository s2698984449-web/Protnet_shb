# P1-1 exp04 full run — set FAST_LR to best from exp04a/b/c
param(
    [string]$ExpName = "exp04_full_unfreeze_lr002",
    [string]$FastLr = "0.02"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

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
    --meta_batch_size 2 `
    --first_order True `
    --train_domains 0,1,2 `
    --test_domain 3 `
    --w_rl_max 0 `
    --stage1_end 1000 --stage2_end 1000 `
    --eval_tasks 100 `
    --eval_every 50 `
    --iters 300 `
    --seed 42
