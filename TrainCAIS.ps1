# TrainCAIS.ps1
# 一键以管理员身份启动 powershell, 激活 MAML 环境, 进入项目目录.
# 把这个文件放在 D:\元强化学习\代码\ 下, 之后:
#   1. 在 cursor 的 powershell 里运行:  & "D:\元强化学习\代码\TrainCAIS.ps1"
#   2. 它会自动请求管理员 (UAC 弹窗), 然后开新窗口准备就绪
#   3. 在新窗口里直接跑训练命令 (已经激活 MAML + 启动了 NVIDIA 服务)

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'D:\元强化学习\代码'
$EnvName = 'MAML'

# 检查是否已经是管理员
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    # 启动一个新的管理员 powershell 窗口, 让用户在新窗口里继续操作
    Write-Host "[TrainCAIS] 当前不是管理员, 启动新窗口并请求提权..." -ForegroundColor Yellow
    $script = @"
`$ErrorActionPreference = 'Stop'
Write-Host '[TrainCAIS] 已获得管理员权限' -ForegroundColor Green
# 启动 NVIDIA 驱动内核模块
Start-Service nvlddmkm -ErrorAction SilentlyContinue
Write-Host '[TrainCAIS] NVIDIA 服务状态:' (Get-Service nvlddmkm -ErrorAction SilentlyContinue).Status
# 验证 CUDA 可用
`$env:PYTHONIOENCODING='utf-8'
`$env:PYTHONUTF8='1'
`$verify = & conda run -n $EnvName python -c "import torch; print('torch.cuda.is_available() =', torch.cuda.is_available())" 2>`$null
Write-Host `[TrainCAIS`] `$verify
# 激活 conda 环境 + cd
conda activate $EnvName
Set-Location '$ProjectRoot'
Write-Host '[TrainCAIS] 环境就绪. 可执行训练命令:' -ForegroundColor Cyan
Write-Host '   python -u scripts/run_cais_pipeline.py --only C4_full --clear-existing --iters 600' -ForegroundColor Cyan
"@
    $tempScript = [System.IO.Path]::GetTempFileName() + '.ps1'
    [System.IO.File]::WriteAllText($tempScript, $script, [System.Text.Encoding]::UTF8)

    # 以管理员启动
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $tempScript `
        -Verb RunAs
    Write-Host "[TrainCAIS] 新窗口已启动, 在那里跑训练命令." -ForegroundColor Green
    exit 0
}

# 已在管理员模式 (用户主动选了管理员启动 cursor powershell)
Start-Service nvlddmkm -ErrorAction SilentlyContinue
$nvStatus = (Get-Service nvlddmkm -ErrorAction SilentlyContinue).Status
Write-Host "[TrainCAIS] 已在管理员模式. NVIDIA 服务: $nvStatus" -ForegroundColor Green

conda activate $EnvName
Set-Location $ProjectRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$verify = conda run -n $EnvName python -c "import torch; print('torch.cuda.is_available() =', torch.cuda.is_available())" 2>$null
Write-Host "[TrainCAIS] $verify" -ForegroundColor Cyan
Write-Host '[TrainCAIS] 环境就绪. 可执行训练命令:' -ForegroundColor Cyan
Write-Host '   python -u scripts/run_cais_pipeline.py --only C4_full --clear-existing --iters 600' -ForegroundColor Cyan
