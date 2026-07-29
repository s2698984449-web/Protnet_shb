# P1-1 exp04: fast_lr grid (50 iters each) — unfreeze + split meta lr
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$common = @(
    "train.py",
    "--ways", "2",
    "--shots", "5",
    "--rgb_only", "True",
    "--data_dir_path", "../data",
    "--yolo_weights", "./weights/aviation_yolo26_pretrained.pt",
    "--freeze_backbone", "False",
    "--backbone_meta_lr", "1e-5",
    "--head_meta_lr", "1e-3",
    "--meta_lr", "0.001",
    "--meta_batch_size", "2",
    "--first_order", "True",
    "--train_domains", "0,1,2",
    "--test_domain", "3",
    "--w_rl_max", "0",
    "--stage1_end", "1000",
    "--stage2_end", "1000",
    "--eval_tasks", "100",
    "--eval_every", "25",
    "--iters", "50",
    "--seed", "42"
)

foreach ($pair in @(
    @("exp04a_unfreeze_lr001", "0.01"),
    @("exp04b_unfreeze_lr002", "0.02"),
    @("exp04c_unfreeze_lr003", "0.03")
)) {
    $exp = $pair[0]
    $flr = $pair[1]
    Write-Host "=== $exp fast_lr=$flr ==="
    & $python @common "--exp_name", $exp, "--fast_lr", $flr
}

Write-Host "=== exp04 fast_lr grid finished ==="
