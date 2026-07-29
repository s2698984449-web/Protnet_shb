# 3-seed final eval for exp07 best checkpoint
param(
    [string]$ExpName = "exp07_crossdom_es_lr002",
    [int]$AdaptSteps = 5,
    [string]$FastLr = "0.02"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$ckpt = "./checkpoints/$ExpName/best.pt"
$out = "./reports/final_eval_${ExpName}.json"

Write-Host "=== Final eval: $ExpName | seeds 42,43,44 ==="

& $python final_eval.py `
    --checkpoint $ckpt `
    --data_dir_path ../data `
    --test_domain 3 `
    --train_domains 0,1,2 `
    --ways 2 --shots 5 `
    --adapt_steps $AdaptSteps `
    --fast_lr $FastLr `
    --freeze_backbone False `
    --rgb_only True `
    --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
    --eval_tasks 100 `
    --seeds 42,43,44 `
    --out_json $out

Write-Host "=== Done. Compare pooled mean with exp04 (0.679, CI low 0.656) ==="
Write-Host "Report: $out"
