# exp08 final eval excluding QC-failed specimen 22 damage ROIs (approach A)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "D:\miniconda3\envs\MAML\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

# fallback if list file unreadable
$exclude = "22,22_flip,22_flip0,22_flip1,22_rot1,22_rot2,22_rot3"
try {
    $fromFile = & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "exclude_specimen22_damage.ps1") 2>$null
    if ($fromFile -and $fromFile -match '^\d') { $exclude = $fromFile.Trim() }
} catch {}

Write-Host "=== Exclude sample_ids: $exclude ==="

& $python final_eval.py `
    --checkpoint ./checkpoints/exp08_protonet_unfreeze/best.pt `
    --meta_algorithm protonet `
    --protonet_distance euclidean `
    --data_dir_path ../data `
    --test_domain 3 `
    --train_domains 0,1,2 `
    --ways 2 --shots 5 `
    --freeze_backbone False `
    --rgb_only True `
    --yolo_weights ./weights/aviation_yolo26_pretrained.pt `
    --eval_tasks 100 `
    --seeds 42,43,44 `
    --exclude_sample_ids "$exclude" `
    --out_json ./reports/final_eval_exp08_exclude_specimen22_damage.json

Write-Host "=== Done. Compare with full-set exp08 (0.852) ==="
