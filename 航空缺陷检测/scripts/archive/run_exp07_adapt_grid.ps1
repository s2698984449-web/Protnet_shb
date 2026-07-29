# exp07 quick grid — adapt_steps on cross-domain + early stop (50 iters each)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$steps = @(3, 5, 7)
foreach ($s in $steps) {
    $name = "exp07a_crossdom_adapt{0}" -f $s
    Write-Host "`n=== $name | adapt_steps=$s | 50 iters ==="
    & $python train.py `
        --exp_name $name `
        --ways 2 --shots 5 `
        --rgb_only True `
        --data_dir_path ../data `
        --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
        --freeze_backbone False `
        --backbone_meta_lr 1e-5 `
        --head_meta_lr 1e-3 `
        --fast_lr 0.02 `
        --adapt_steps $s `
        --meta_batch_size 2 `
        --first_order True `
        --train_domains 0,1,2 `
        --test_domain 3 `
        --cross_domain_episodes True `
        --eval_patience 99 `
        --w_rl_max 0 `
        --stage1_end 1000 --stage2_end 1000 `
        --eval_tasks 100 `
        --eval_every 50 `
        --iters 50 `
        --seed 42
}

Write-Host "`n=== adapt grid done. Pick best adapt_steps, then run run_exp07_full.ps1 -AdaptSteps <N> ==="
