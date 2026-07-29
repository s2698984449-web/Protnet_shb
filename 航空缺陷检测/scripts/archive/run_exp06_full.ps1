# exp06: exp04 config + episode augmentation (meta-train only)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Write-Host "=== exp06: episode_aug + unfreeze + real domains 0-2 | 300 iter ==="

& $python train.py `
    --exp_name exp06_epaug_unfreeze_lr002 `
    --ways 2 --shots 5 `
    --rgb_only True `
    --data_dir_path ../data `
    --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
    --freeze_backbone False `
    --backbone_meta_lr 1e-5 `
    --head_meta_lr 1e-3 `
    --fast_lr 0.02 `
    --meta_batch_size 2 `
    --first_order True `
    --train_domains 0,1,2 `
    --test_domain 3 `
    --w_rl_max 0 `
    --stage1_end 1000 --stage2_end 1000 `
    --eval_tasks 100 `
    --eval_every 50 `
    --iters 300 `
    --episode_aug True `
    --seed 42

Write-Host "=== exp06 training finished ==="
