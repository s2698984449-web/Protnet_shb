"""CA-IS-MetaProto 流水线编排: 5 个消融阶段 + 每阶段完成后自动多 seed 评测.

流水线结构:
  阶段 1..5: C0_baseline / C1_curriculum / C2_mahal / C3_trim_mean / C4_full
  每个阶段:
    1. 检查 best.pt 是否已存在 (跳过 — 必须重跑才能拿新数据)
       注: 健壮设计是"如果 best.pt 已存在且 --resume, 跳过训练直接评测".
       当前版本强制重跑, 因为 C4 目录仅有 logs 无 ckpt (上个进程被中断).
    2. train_meta.py 训练直到 DONE.
    3. eval_meta.py 用 seeds=42,43,44 跑多 seed 评测.
    4. 汇总 best_val_acc + multi-seed pooled acc + CI95 到 pipeline_report.json

设计原则:
  - 串行, 不并行 (CPU 设备下并行无意义, 且容易 IO 冲突)
  - 每个阶段独立进程, 一个失败不会污染其他阶段
  - 阶段间 sleep 30s (冷却, 防止 IO/资源未释放)
  - 所有 checkpoint / report / log 落到磁盘, 重启可恢复 (目前每阶段强制重跑; 如需 resume, 见 RESUME_BEST 变量)

Usage:
    cd "D:/元强化学习/代码"
    python scripts/run_cais_pipeline.py
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 强制 UTF-8 stdout (Windows PowerShell 默认是 GBK 936, 否则 emoji/全角符号会炸)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# 同时把 PowerShell/cmd 的 console output code page 切到 65001 (UTF-8)
# 解决 subprocess.PIPE 解码后 sys.stdout.write 时 GBK 二次编码问题
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
except Exception:
    pass
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')


# 5 个消融阶段配置 (v2: 500 iter 验证方向, P0/P1/P2/P3 修复)
# 变更:
#   C1 curriculum_probs: 0.2,0.4,0.8 -> 0.0,0.1,0.3 (更保守起步)
#   C2 distance: mahalanobis -> cosine (避开小样本协方差病态)
#   C4 去掉 mahal+trim_mean, 改用 euclidean+mean+domain_reliability (干净验证 DR)
STAGES = [
    {
        'name': 'C0_baseline',
        'desc': '新管线基线 (euclidean + mean, 无 curriculum, 无 domain reliability, FiLM+Prod 解冻)',
        'distance': 'euclidean',
        'prototype': 'mean',
        'curriculum_probs': '0.0,0.0,0.0',
        'use_domain_reliability': False,
        'unfreeze_film': True,
        'unfreeze_prod': True,
        'use_anil_head': False,
        'num_workers': 4, 'prefetch_batches': 4,
        'async_cross_domain_prob': 0.0,  # baseline 不需要跨域
    },
    {
        'name': 'C1_curriculum',
        'desc': '+ 课程学习跨域采样 (保守曲线 0.0->0.3, FiLM+Prod 解冻)',
        'distance': 'euclidean',
        'prototype': 'mean',
        'curriculum_probs': '0.0,0.1,0.3',
        'use_domain_reliability': False,
        'unfreeze_film': True,
        'unfreeze_prod': True,
        'use_anil_head': False,
        'num_workers': 4, 'prefetch_batches': 4,
        'async_cross_domain_prob': 0.3,
    },
    {
        'name': 'C2_cosine',
        'desc': '+ Cosine 距离 (替代 Mahal, 避开小样本协方差病态, FiLM+Prod 解冻)',
        'distance': 'cosine',
        'prototype': 'mean',
        'curriculum_probs': '0.0,0.1,0.3',
        'use_domain_reliability': False,
        'unfreeze_film': True,
        'unfreeze_prod': True,
        'use_anil_head': False,
        'num_workers': 4, 'prefetch_batches': 4,
        'async_cross_domain_prob': 0.3,
    },
    {
        'name': 'C3_trim_mean',
        'desc': '+ Trim-Mean 原型 (基于 cosine + curriculum, FiLM+Prod 解冻)',
        'distance': 'cosine',
        'prototype': 'trim_mean',
        'curriculum_probs': '0.0,0.1,0.3',
        'use_domain_reliability': False,
        'unfreeze_film': True,
        'unfreeze_prod': True,
        'use_anil_head': False,
        'num_workers': 4, 'prefetch_batches': 4,
        'async_cross_domain_prob': 0.3,
    },
    {
        'name': 'C4_full',
        'desc': '+ Domain Reliability 加权 (基于 euclidean+mean, Full v2, FiLM+Prod 解冻)',
        'distance': 'euclidean',
        'prototype': 'mean',
        'curriculum_probs': '0.0,0.1,0.3',
        'use_domain_reliability': True,
        'unfreeze_film': True,
        'unfreeze_prod': True,
        'use_anil_head': False,
        'num_workers': 4, 'prefetch_batches': 4,
        'async_cross_domain_prob': 0.4,  # DR 需要足够跨域样本
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', default=r'D:\元强化学习\代码\航空缺陷检测\checkpoints',
                   help='checkpoint 根目录')
    p.add_argument('--report-root', default=r'D:\元强化学习\代码\航空缺陷检测\reports',
                   help='评测报告根目录')
    p.add_argument('--cwd', default=r'D:\元强化学习\代码')
    p.add_argument('--n-way', type=int, default=4)
    p.add_argument('--k-shot', type=int, default=5)
    p.add_argument('--q-query', type=int, default=5)
    p.add_argument('--iters', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--eval-episodes', type=int, default=50)
    p.add_argument('--eval-tasks-final', type=int, default=100,
                   help='多 seed 评测时, 每个 seed 的 episode 数')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds-eval', default='42,43,44')
    p.add_argument('--mvtec-root', default=r'D:\元强化学习\代码\data\MVTEC')
    p.add_argument('--gc10-root', default=r'D:\元强化学习\代码\data\GC10-DET')
    p.add_argument('--neu-root', default=r'D:\元强化学习\代码\data\NEU-CLS\NEU-CLS')
    p.add_argument('--pretrain-ckpt', default=r'D:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_4w5s_aug_unfreeze\best.pt',
                   help='v2 用 phase1_4w5s_aug_unfreeze/best.pt (替代已删除的 multi_domain_smoke.pt)')
    p.add_argument('--only', default=None,
                   help='只跑指定 name, e.g. "C4_full"')
    p.add_argument('--skip-train', action='store_true',
                   help='跳过训练, 直接评测 (假设已有 best.pt)')
    p.add_argument('--skip-eval', action='store_true',
                   help='跳过评测, 只跑训练')
    p.add_argument('--force', action='store_true',
                   help='如果 best.pt 已存在仍然重跑训练 (默认检测到存在就跳过该阶段)')
    p.add_argument('--cooldown-seconds', type=int, default=30,
                   help='阶段间冷却时间, 防 IO/资源未释放')
    p.add_argument('--clear-existing', action='store_true',
                   help='清掉上次中断的目录 (C4_full 当前有 logs 但无 ckpt)')
    p.add_argument('--device', default=None,
                   help='训练设备 (默认自动检测: cuda 优先, fallback cpu). '
                        '可显式指定 "cuda" / "cpu".')
    return p.parse_args()


def run_subprocess(cmd, log_file, cwd, stream_to_parent=True):
    """跑子进程, 全部输出到 log_file. 同时可选流式回传到当前终端 (实时可见).
    返回 (returncode, elapsed_seconds)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # 子进程环境变量强制 UTF-8, 防 PowerShell GBK 把 stdout 抛异常
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('>>> STDOUT/STDERR captured below (UTF-8).\n')
        f.flush()
    with open(log_file, 'a', encoding='utf-8') as outf, \
         subprocess.Popen(
             cmd, cwd=cwd,
             stdout=subprocess.PIPE,
             stderr=subprocess.STDOUT,
             bufsize=1,  # 行缓冲
             env=env) as proc:
        # 流式逐行读 -> 同时写 log + 转发到当前终端
        for raw in proc.stdout:
            try:
                line = raw.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                line = raw.decode('gbk', errors='replace')
            outf.write(line)
            outf.flush()
            if stream_to_parent:
                # 转发到父进程 stdout (这样无论从哪个终端启动都能实时看)
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except Exception:
                    pass
        proc.wait()
    elapsed = time.time() - t0
    return proc.returncode, elapsed


def stage_train(cfg, args, common):
    """阶段训练. 返回 dict {best_val_acc, final_acc, n_iters_ran, duration_sec}."""
    ckpt_dir = Path(args.out_root) / f"phase1_4w5s_{cfg['name']}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / 'pipeline.log'

    # 检查已有 best.pt
    best_ckpt = ckpt_dir / 'best.pt'
    if best_ckpt.exists():
        try:
            import torch
            sd = torch.load(str(best_ckpt), map_location='cpu', weights_only=False)
            best_acc = sd.get('best_val_acc')
            print(f'  [skip-train] {cfg["name"]} best.pt 已存在, best_val_acc={best_acc:.4f}')
            return {
                'best_val_acc': best_acc,
                'final_acc': None,
                'n_iters_ran': 0,
                'duration_sec': 0,
                'skipped': True,
            }
        except Exception as e:
            print(f'  [skip-train] {cfg["name"]} best.pt 存在但解析失败 ({e}), 重跑')

    # 跑 train_meta.py
    cmd = [
        sys.executable, 'scripts/train_meta.py',
        '--n-way', str(common['n_way']),
        '--k-shot', str(common['k_shot']),
        '--q-query', str(common['q_query']),
        '--iters', str(common['iters']),
        '--eval-every', str(common['eval_every']),
        '--eval-episodes', str(common['eval_episodes']),
        '--lr', str(common['lr']),
        '--distance', cfg['distance'],
        '--prototype', cfg['prototype'],
        '--curriculum-stages', '0.6,0.8,1.0',
        '--curriculum-probs', cfg['curriculum_probs'],
    ]
    if cfg['use_domain_reliability']:
        cmd.append('--use-domain-reliability')
    else:
        cmd.append('--no-domain-reliability')
    # ── 解冻 FiLM / ProductEncoder (与 cond_enc 协同, 必要设计) ──
    if cfg.get('unfreeze_film', False):
        cmd.append('--unfreeze-film')
    else:
        cmd.append('--no-unfreeze-film')
    if cfg.get('unfreeze_prod', False):
        cmd.append('--unfreeze-prod')
    else:
        cmd.append('--no-unfreeze-prod')
    # ── ANIL head (默认关闭, baseline 阶段不要开) ──
    if cfg.get('use_anil_head', False):
        cmd.append('--use-anil-head')
    else:
        cmd.append('--no-anil-head')
    # ── 异步采样参数 (显式传, 不依赖 train_meta 默认值) ──
    cmd += [
        '--num-workers', str(cfg.get('num_workers', 2)),
        '--prefetch-batches', str(cfg.get('prefetch_batches', 4)),
        '--async-cross-domain-prob', str(cfg.get('async_cross_domain_prob', 0.3)),
    ]
    if cfg.get('no_async_sampler', False):
        cmd.append('--no-async-sampler')
    cmd += [
        '--mvtec-root', common['mvtec_root'],
        '--gc10-root', common['gc10_root'],
        '--neu-root', common['neu_root'],
        '--pretrain-ckpt', common['pretrain_ckpt'],
        '--out-dir', str(ckpt_dir),
        '--seed', str(common['seed']),
        '--device', args.device,
    ]

    print(f'  [train] {" ".join(cmd[:6])} ... -> {ckpt_dir}')
    rc, elapsed = run_subprocess(cmd, log_path, args.cwd)

    # 解析 best_val_acc
    best_acc = None
    final_acc = None
    n_iters_ran = 0
    if best_ckpt.exists():
        try:
            import torch
            sd = torch.load(str(best_ckpt), map_location='cpu', weights_only=False)
            best_acc = sd.get('best_val_acc')
            n_iters_ran = sd.get('iter', 0)
        except Exception as e:
            print(f'  ⚠️  解析 best.pt 失败: {e}')

    final_ckpt = ckpt_dir / 'final.pt'
    if final_ckpt.exists():
        try:
            import torch
            sd = torch.load(str(final_ckpt), map_location='cpu', weights_only=False)
            final_acc = sd.get('best_val_acc')  # final 写的也是 best_val_acc
        except Exception:
            pass

    return {
        'best_val_acc': best_acc,
        'final_acc': final_acc,
        'n_iters_ran': n_iters_ran,
        'duration_sec': elapsed,
        'exit_code': rc,
        'skipped': False,
    }


def stage_eval(cfg, train_info, args, common):
    """阶段评测 (multi-seed 95% CI)."""
    ckpt_dir = Path(args.out_root) / f"phase1_4w5s_{cfg['name']}"
    best_ckpt = ckpt_dir / 'best.pt'
    if not best_ckpt.exists():
        print(f'  ⚠️  best.pt 不存在, 跳过评测')
        return {'status': 'skipped_no_ckpt'}

    eval_dir = Path(args.report_root) / f'final_eval_{cfg["name"]}'
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, 'scripts/eval_meta.py',
        '--checkpoint', str(best_ckpt),
        '--seeds', args.seeds_eval,
        '--n-way', str(common['n_way']),
        '--k-shot', str(common['k_shot']),
        '--q-query', str(common['q_query']),
        '--eval-tasks', str(args.eval_tasks_final),
        '--out-dir', str(eval_dir),
        '--mvtec-root', common['mvtec_root'],
        '--gc10-root', common['gc10_root'],
        '--neu-root', common['neu_root'],
    ]
    print(f'  [eval]  {" ".join(cmd[:6])} ... -> {eval_dir}')
    log_path = eval_dir / 'pipeline.log'
    rc, elapsed = run_subprocess(cmd, log_path, args.cwd)

    # 读报告
    report_path = eval_dir / f'final_eval_phase1_4w5s_{cfg["name"]}.json'
    pooled = {}
    if report_path.exists():
        with open(report_path, encoding='utf-8') as f:
            report = json.load(f)
        pooled = report.get('pooled', {})

    return {
        'status': 'ok' if rc == 0 else 'failed',
        'exit_code': rc,
        'duration_sec': elapsed,
        'pooled_mean': pooled.get('mean'),
        'pooled_std': pooled.get('std'),
        'pooled_ci95_low': pooled.get('ci95_low'),
        'pooled_ci95_high': pooled.get('ci95_high'),
        'pooled_n_tasks': pooled.get('n_tasks'),
        'verdict': pooled.get('verdict'),
        'report_path': str(report_path),
    }


def main():
    args = parse_args()
    # 解析设备 (默认自动检测 cuda/cpu)
    if args.device is None:
        try:
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            args.device = 'cpu'
    print(f'  device:        {args.device}')
    common = {
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

    print('=' * 78)
    print(f'CA-IS-MetaProto 流水线编排 ({len(STAGES)} 阶段, 串行)')
    print(f'  out_root:      {args.out_root}')
    print(f'  report_root:   {args.report_root}')
    print(f'  protocol:      {common["n_way"]}-way {common["k_shot"]}-shot q={common["q_query"]}')
    print(f'  train iters:   {common["iters"]} (eval-every={common["eval_every"]})')
    print(f'  eval seeds:    {args.seeds_eval} (×{args.eval_tasks_final} eps/seed)')
    print(f'  cooldown:      {args.cooldown_seconds}s between stages')
    print(f'  skip_train={args.skip_train}, skip_eval={args.skip_eval}, force={args.force}')
    if args.only:
        print(f'  ONLY:           {args.only}')
    print('=' * 78)

    pipeline_summary = {
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'common_args': common,
        'eval_args': {
            'seeds_eval': args.seeds_eval,
            'eval_tasks_final': args.eval_tasks_final,
        },
        'stages': {},
    }

    started_overall = time.time()
    failed_stage = None

    for idx, cfg in enumerate(STAGES, start=1):
        if args.only and cfg['name'] != args.only:
            print(f'\n[stage {idx}/{len(STAGES)}] SKIP {cfg["name"]} (not in --only)')
            continue

        # 如果要求 clear-existing, 清掉上次中断的目录
        if args.clear_existing:
            ckpt_dir = Path(args.out_root) / f"phase1_4w5s_{cfg['name']}"
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                print(f'\n[stage {idx}/{len(STAGES)}] {cfg["name"]}: cleared {ckpt_dir}')

        print(f'\n[stage {idx}/{len(STAGES)}] {cfg["name"]}: {cfg["desc"]}')
        print(f'  config: distance={cfg["distance"]}, prototype={cfg["prototype"]}, '
              f'curriculum={cfg["curriculum_probs"]}, domain_reliability={cfg["use_domain_reliability"]}')
        t0 = time.time()

        stage_info = {'description': cfg['desc']}

        # 1) 训练
        if not args.skip_train:
            train_info = stage_train(cfg, args, common)
            stage_info['train'] = train_info
            if not train_info['skipped'] and train_info.get('exit_code', 0) != 0:
                print(f'  [FATAL] 训练失败 (exit_code={train_info.get("exit_code")}), 终止流水线')
                failed_stage = cfg['name']
                stage_info['fatal'] = True
                pipeline_summary['stages'][cfg['name']] = stage_info
                break
        else:
            stage_info['train'] = {'skipped': True, 'note': 'skipped by --skip-train'}

        # 2) 评测
        if not args.skip_eval:
            eval_info = stage_eval(cfg, stage_info.get('train', {}), args, common)
            stage_info['eval'] = eval_info

        elapsed_stage = time.time() - t0
        stage_info['duration_sec'] = elapsed_stage
        pipeline_summary['stages'][cfg['name']] = stage_info

        # 输出阶段结果
        bva = stage_info.get('train', {}).get('best_val_acc')
        if bva is not None:
            print(f'  [OK] best_val_acc (单 seed): {bva:.4f}')
        p_mean = (stage_info.get('eval') or {}).get('pooled_mean')
        p_ci = (stage_info.get('eval') or {}).get('pooled_ci95_low')
        p_ci_hi = (stage_info.get('eval') or {}).get('pooled_ci95_high')
        if p_mean is not None:
            print(f'  [OK] multi-seed pooled: {p_mean:.4f}  CI95 [{p_ci:.4f}, {p_ci_hi:.4f}]  '
                  f'(n={stage_info["eval"].get("pooled_n_tasks")} tasks)')
        print(f'  [TIME] stage 耗时: {elapsed_stage/60:.1f} min')

        # 阶段间冷却
        if idx < len(STAGES):
            print(f'  [SLEEP] 冷却 {args.cooldown_seconds}s ...')
            time.sleep(args.cooldown_seconds)

    pipeline_summary['ended_at'] = datetime.now().isoformat(timespec='seconds')
    pipeline_summary['total_duration_sec'] = time.time() - started_overall
    pipeline_summary['failed_stage'] = failed_stage

    # 写总报告
    report_path = Path(args.report_root) / 'pipeline_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 78)
    print('流水线完成')
    print(f'总耗时: {pipeline_summary["total_duration_sec"]/60:.1f} min')
    print(f'总报告: {report_path}')
    if failed_stage:
        print(f'失败阶段: {failed_stage}')
    print('=' * 78)

    # 打印简表
    print('\n+----+-----------------+------------+----------------------+')
    print('| Stage               | best_val   | multi-seed pooled    |')
    print('+---------------------+------------+----------------------+')
    for name, info in pipeline_summary['stages'].items():
        bva = info.get('train', {}).get('best_val_acc')
        bva_s = f'{bva:.4f}' if bva is not None else 'N/A'
        pm = info.get('eval', {}).get('pooled_mean')
        ci_lo = info.get('eval', {}).get('pooled_ci95_low')
        ci_hi = info.get('eval', {}).get('pooled_ci95_high')
        if pm is not None:
            pooled_s = f'{pm:.4f} [{ci_lo:.3f}, {ci_hi:.3f}]'
        else:
            pooled_s = 'N/A'
        print(f'| {name:19s} | {bva_s:>10s} | {pooled_s:>20s} |')
    print('+---------------------+------------+----------------------+')


if __name__ == '__main__':
    main()
