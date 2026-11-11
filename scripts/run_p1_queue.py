"""按顺序跑当前批次的留出。上一个训练进程退出后才启动下一个。

基准配置由 --config 指定，默认是 v3。本脚本不改基准文件。
输出在 output/cspd/<output_group>/<exp_name>/。已有 last.pt 就跳过。

    python scripts/run_p1_queue.py
    python scripts/run_p1_queue.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config' / 'cspd' / 'curriculum_v3.yaml'

# 三个留出来源各一个种子。种子差在 v1、v2 里都很小。
JOBS = [
    ('NEU', 0, 'NEU'),
    ('CR7', 0, 'CR7'),
    ('GC10', 0, 'GC10'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    group = str(base.get('output_group') or '').strip()
    if not group:
        raise SystemExit('配置需要 output_group，例如 v3')
    queue_dir = ROOT / 'output' / 'cspd' / group / '_queue'
    queue_dir.mkdir(parents=True, exist_ok=True)
    pending = 0
    for holdout, seed, name in JOBS:
        done = ROOT / 'output' / 'cspd' / group / name / 'last.pt'
        if done.is_file():
            print('skip {}  last.pt 已在'.format(name))
            continue
        pending += 1
        cfg = dict(base)
        cfg['holdout_source'] = holdout
        cfg['seed'] = int(seed)
        cfg['exp_name'] = name
        cfg_path = queue_dir / '{}.yaml'.format(name)
        cfg_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
            encoding='utf-8',
        )
        if args.dry_run:
            print('would run {}  holdout={} seed={} -> output/cspd/{}/{}'.format(
                name, holdout, seed, group, name))
            continue
        print('start {}  holdout={} seed={}'.format(name, holdout, seed))
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / 'train_cspd.py'), '--config', str(cfg_path)],
            cwd=str(ROOT),
        )
        print('exit {} code {}'.format(name, completed.returncode))
        if completed.returncode != 0:
            sys.exit(completed.returncode)
    if pending == 0:
        print('queue empty')
    elif args.dry_run:
        print('dry-run pending {}'.format(pending))
    else:
        print('queue done')


if __name__ == '__main__':
    main()
