# exp08 ProtoNet smoke test (20 iters)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Write-Host "=== exp08 smoke: ProtoNet | 20 iters ==="

& $python train.py `
    --exp_name exp08_protonet_smoke `
    --meta_algorithm protonet `
    --protonet_distance euclidean `
    --ways 2 --shots 5 `
    --rgb_only True `
    --data_dir_path ../data `
    --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
    --freeze_backbone False `
    --backbone_meta_lr 1e-5 `
    --head_meta_lr 1e-3 `
    --meta_batch_size 2 `
    --train_domains 0,1,2 `
    --test_domain 3 `
    --w_rl_max 0 `
    --stage1_end 1000 --stage2_end 1000 `
    --eval_tasks 20 `
    --eval_every 10 `
    --iters 20 `
    --seed 42

Write-Host "=== smoke done ==="
