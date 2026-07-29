# P0-1 / P0-2 ablation runs (same backbone as baseline)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$commonArgs = @(
    "train.py",
    "--iters", "500",
    "--rgb_only", "True",
    "--data_dir_path", "../data",
    "--yolo_weights", "./weights/aviation_yolo26_pretrained.pt",
    "--freeze_backbone", "True",
    "--test_domain", "3",
    "--ways", "2",
    "--shots", "5",
    "--meta_batch_size", "2",
    "--first_order", "True",
    "--eval_every", "50",
    "--eval_tasks", "50",
    "--w_rl_max", "0",
    "--seed", "42"
)

Write-Host "=== P0-1: no RL, full train domains ==="
& $python @commonArgs `
    "--exp_name", "exp01_p0-1_no_rl_full7" `
    "--train_domains", "0,1,2,4,5,6,7"

Write-Host "=== P0-2: no RL, real domains 0-2 only ==="
& $python @commonArgs `
    "--exp_name", "exp02_p0-2_no_rl_real3" `
    "--train_domains", "0,1,2"

Write-Host "=== P0 ablations finished ==="
