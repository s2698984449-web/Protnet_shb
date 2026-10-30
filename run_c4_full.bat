# run_c4_full.bat
# 在管理员 powershell 里调用此 bat 启动训练 (后台 + GPU)
# 用法: 右键管理员 → "D:\元强化学习\代码\run_c4_full.bat"

@echo off
chcp 65001 >nul
cd /d "D:\元强化学习\代码"

call conda activate MAML

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM 启动训练 (后台, 输出到 log)
start "C4_full_train" /min ^
  python -u scripts/run_cais_pipeline.py ^
  --only C4_full ^
  --clear-existing ^
  --cooldown-seconds 30 ^
  --iters 600 ^
  --eval-every 100 ^
  --eval-tasks-final 100 ^
  --device cuda ^
  --force ^
  > "D:\元强化学习\代码\航空缺陷检测\reports\pipeline_console_v4_full_unfrozen.log" 2>&1

echo [C4] 已启动后台训练, PID 见 powershell Get-Process python
echo [C4] 实时查看: Get-Content "D:\元强化学习\代码\航空缺陷检测\reports\pipeline_console_v4_full_unfrozen.log" -Wait
exit /b 0
