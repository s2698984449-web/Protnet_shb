"""CA-IS-MetaProto 消融实验: C0→C4 全模块对比.

5 个实验, 每个共享同一份 MultiDomainPretrainModel + 同一份 train_meta.py,
只通过开关决定哪些模块参与. 这样得到的 5 个数字才能真正代表 CA-IS-MetaProto
在新管线下的消融链, 而不是和旧 AviationDefectModel 实验混在一起.

C0: euclidean + mean + (no curriculum) + (no domain reliability)  ← 新管线基线
C1: euclidean + mean + curriculum                              ← +Curriculum
C2: mahalanobis + mean + curriculum                            ← +Mahalanobis
C3: mahalanobis + trim_mean + curriculum                       ← +Trim-Mean
C4: mahalanobis + trim_mean + curriculum + domain_reliability  ← Full (Current)

Usage:
    cd "D:/元强化学习/代码"
    python scripts/run_ablation_cais.py --tag baseline --out-root "D:/元强化学习/代码/航空缺陷检测/checkpoints"

或者使用 PowerShell wrapper:
    powershell -ExecutionPolicy Bypass -File scripts/run_ablation_cais.ps1
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ABLATIONS = [
    {
        'name': 'C0_baseline',
        'desc': '新管线基线 (euclidean + mean, 无 curriculum, 无 domain reliability)',
        'distance': 'euclidean',
        'prototype': 'mean',
        'curriculum_probs': '0.0,0.0,0.0',  # 关闭跨域采样
        'domain_reliability': False,
    },
    {
        'name': 'C1_curriculum',
        'desc': '+ 课程学习跨域采样',
        'distance': 'euclidean',
        'prototype': 'mean',
        'curriculum_probs': '0.2,0.4,0.8',
        'domain_reliability': False,
    },
    {
        'name': 'C2_mahal',
        'desc': '+ Mahalanobis 距离',
        'distance': 'mahalanobis',
        'prototype': 'mean',
        'curriculum_probs': '0.2,0.4,0.8',
        'domain_reliability': False,
    },
    {
        'name': 'C3_trim_mean',
        'desc': '+ Trim-Mean 原型',
        'distance': 'mahalanobis',
        'prototype': 'trim_mean',
        'curriculum_probs': '0.2,0.4,0.8',
        'domain_reliability': False,
    },
    {
        'name': 'C4_full',
        'desc': '+ Domain Reliability 加权 (Full = CA-IS-MetaProto)',
        'distance': 'mahalanobis',
        'prototype': 'trim_mean',
        'curriculum_probs': '0.2,0.4,0.8',
        'domain_reliability': True,
    },
]


def run_one(cfg, args, common_args):
    """跑单个消融实验."""
    out_dir = Path(args.out_root) / f"phase1_4w5s_{cfg['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, 'scripts/train_meta.py',
        '--n-way', str(common_args['n_way']),
        '--k-shot', str(common_args['k_shot']),
        '--q-query', str(common_args['q_query']),
        '--iters', str(common_args['iters']),
        '--eval-every', str(common_args['eval_every']),
        '--eval-episodes', str(common_args['eval_episodes']),
        '--lr', str(common_args['lr']),
        '--distance', cfg['distance'],
        '--prototype', cfg['prototype'],
        '--curriculum-stages', '0.6,0.8,1.0',
        '--curriculum-probs', cfg['curriculum_probs'],
    ]
    if cfg['domain_reliability']:
        cmd.append('--use-domain-reliability')
    else:
        cmd.append('--no-domain-reliability')

    # 数据/路径
    cmd += [
        '--mvtec-root', common_args['mvtec_root'],
        '--gc10-root', common_args['gc10_root'],
        '--neu-root', common_args['neu_root'],
        '--pretrain-ckpt', common_args['pretrain_ckpt'],
        '--out-dir', str(out_dir),
        '--seed', str(common_args['seed']),
    ]

    log_file = out_dir / 'train.log'
    print(f'\n{"=" * 70}')
    print(f'[run] {cfg["name"]}: {cfg["desc"]}')
    print(f'  cmd: {" ".join(cmd[:8])} ...')
    print(f'  log: {log_file}')
    print(f'{"=" * 70}')

    # 写到 run_config.json 给后续引用
    summary = {
        'name': cfg['name'],
        'desc': cfg['desc'],
        'config': cfg,
        'common_args': common_args,
        'status': 'running',
    }
    with open(out_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(log_file, 'w', encoding='utf-8') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=args.cwd)
    summary['status'] = 'done' if proc.returncode == 0 else 'failed'
    summary['exit_code'] = proc.returncode
    with open(out_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return proc.returncode == 0, out_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', default=r'D:\元强化学习\代码\航空缺陷检测\checkpoints',
                   help='消融实验 checkpoint 输出根目录')
    p.add_argument('--cwd', default=r'D:\元强化学习\代码',
                   help='训练子进程的工作目录')
    p.add_argument('--n-way', type=int, default=4)
    p.add_argument('--k-shot', type=int, default=5)
    p.add_argument('--q-query', type=int, default=5)
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mvtec-root', default=r'D:\元强化学习\代码\data\MVTEC')
    p.add_argument('--gc10-root', default=r'D:\元强化学习\代码\data\GC10-DET')
    p.add_argument('--neu-root', default=r'D:\元强化学习\代码\data\NEU-CLS\NEU-CLS')
    p.add_argument('--pretrain-ckpt', default=r'D:\元强化学习\代码\outputs\multi_domain_smoke.pt')
    p.add_argument('--only', default=None,
                   help='只跑指定名称的实验, e.g. "C4_full" (跳过其他)')
    return p.parse_args()


def main():
    args = parse_args()
    common_args = {
        'n_way': args.n_way,
        'k_shot': args.k_shot,
        'q_query': args.q_query,
        'iters': args.iters,
        'eval_every': args.eval_every,
        'eval_episodes': args.eval_episodes,
        'lr': args.lr,
        'seed': args.seed,
        'mvtec_root': args.mvtec_root,
        'gc10_root': args.gc10_root,
        'neu_root': args.neu_root,
        'pretrain_ckpt': args.pretrain_ckpt,
    }

    results = []
    for cfg in ABLATIONS:
        if args.only and cfg['name'] != args.only:
            print(f'[skip] {cfg["name"]} (not in --only)')
            continue
        ok, out_dir = run_one(cfg, args, common_args)
        results.append({'name': cfg['name'], 'ok': ok, 'out_dir': str(out_dir)})
        if not ok:
            print(f'[fail] {cfg["name"]} failed, aborting.')
            break

    print('\n' + '=' * 70)
    print('Ablation summary:')
    for r in results:
        print(f'  {r["name"]:20s} {"OK" if r["ok"] else "FAIL":5s}  {r["out_dir"]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
