"""添加 Windows 计划任务: 以管理员权限 + 启动 NVIDIA 驱动服务, 一键启动训练.

作用:
  1. 创建一个计划任务 "TrainCAIS_Admin", 每次手动触发时:
     - 以 SYSTEM 权限运行 (等同管理员, 但不需要用户密码/UAC)
     - 启动 nvlddmkm 服务 (NVIDIA 驱动)
     - 激活 MAML conda 环境
     - 调用 run_cais_pipeline.py

为什么用 SYSTEM 而非管理员:
  SYSTEM 权限比 BUILTIN\Administrator 更高, 但不需要交互式登录.
  计划任务的 "Run with highest privileges" + "SYSTEM" 账号是 Windows 批处理自动化常用模式.

Usage:
  python scripts/setup_admin_training.py install   # 安装任务
  python scripts/setup_admin_training.py remove    # 卸载
  python scripts/setup_admin_training.py run       # 立即触发任务
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

TASK_NAME = 'TrainCAIS_Admin'
TASK_DESC = 'Run CA-IS-MetaProto training with admin privileges (NVIDIA GPU access)'

REPO_ROOT = r'D:\元强化学习\代码'
CONDA_ENV = 'MAML'
PIPELINE_SCRIPT = r'D:\元强化学习\代码\scripts\run_cais_pipeline.py'


def build_task_xml(args: str) -> str:
    """构造计划任务 XML. args 是要传给 pipeline 的命令行参数."""
    return f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{TASK_DESC}</Description>
    <URI>\\{TASK_NAME}</URI>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Enabled>false</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>S-1-5-18</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>PT8H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-NoProfile -ExecutionPolicy Bypass -Command "Start-Service nvlddmkm -ErrorAction SilentlyContinue; & {{ conda activate {CONDA_ENV} ; cd '{REPO_ROOT}' ; $env:PYTHONIOENCODING='utf-8' ; $env:PYTHONUTF8='1' ; python -u {PIPELINE_SCRIPT} {args} }}"</Arguments>
      <WorkingDirectory>{REPO_ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
'''


def install(args_str: str = ''):
    xml_path = Path(os.environ.get('TEMP', '/tmp')) / 'train_cais_task.xml'
    with open(xml_path, 'w', encoding='utf-16') as f:
        f.write(build_task_xml(args_str))
    # schtasks /create 需要管理员
    print(f'[setup] 创建计划任务 "{TASK_NAME}" (需要管理员权限)...')
    rc = subprocess.call(['schtasks', '/Create', '/TN', TASK_NAME, '/XML', str(xml_path)])
    if rc == 0:
        print(f'[setup] ✓ 计划任务已创建. 触发方式: schtasks /Run /TN {TASK_NAME}')
        print(f'[setup]   或运行: python scripts/setup_admin_training.py run')
    else:
        print(f'[setup] ✗ schtasks /Create 失败 rc={rc}')
        print(f'[setup]   请以管理员身份运行此脚本 (右键 -> 以管理员身份运行)')
    return rc


def remove():
    rc = subprocess.call(['schtasks', '/Delete', '/TN', TASK_NAME, '/F'])
    if rc == 0:
        print(f'[setup] ✓ 已删除计划任务 {TASK_NAME}')
    else:
        print(f'[setup] ✗ 删除失败 rc={rc}')
    return rc


def run(args_str: str = ''):
    """安装任务并立即触发 (无参 = 用上次保存的参数)."""
    if args_str:
        install(args_str)
    print(f'[run] 触发计划任务 {TASK_NAME}...')
    rc = subprocess.call(['schtasks', '/Run', '/TN', TASK_NAME])
    if rc == 0:
        print('[run] ✓ 任务已触发. 查看 logs:')
        print('       Get-Content D:\\元强化学习\\代码\\航空缺陷检测\\reports\\pipeline_console.log -Tail 30 -Wait')
    return rc


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['install', 'remove', 'run'])
    p.add_argument('--args', default='', help='传给 pipeline 的参数, e.g. "--only C4_full --iters 600"')
    a = p.parse_args()
    sys.exit({'install': lambda: install(a.args),
              'remove': lambda: remove(),
              'run': lambda: run(a.args)}[a.action]())
