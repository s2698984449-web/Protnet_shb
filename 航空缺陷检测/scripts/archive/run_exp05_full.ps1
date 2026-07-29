# exp05: unfreeze MAML + AeBAD domains 4-7 (expand meta-task diversity)
# Same hyperparams as exp04_full; only train_domains changes.
param(
    [string]$ExpName = "exp05_unfreeze_full7_lr002",
    [string]$FastLr = "0.02",
    [int]$Iters = 300
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Write-Host "=== exp05: train_domains 0,1,2,4,5,6,7 | fast_lr=$FastLr | iters=$Iters ==="

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
    --train_domains 0,1,2,4,5,6,7 `
    --test_domain 3 `
    --w_rl_max 0 `
    --stage1_end 1000 --stage2_end 1000 `
    --eval_tasks 100 `
    --eval_every 50 `
    --iters $Iters `
    --seed 42

Write-Host "=== exp05 finished ==="
