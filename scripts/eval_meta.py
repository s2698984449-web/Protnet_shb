"""Final multi-seed evaluation for checkpoints produced by scripts/train_meta.py.

Loads a best.pt / final.pt from train_meta.py, reconstructs MultiDomainPretrainModel +
(optional) MahalanobisHead + (optional) DomainReliability, then runs N episodes
under multiple seeds on meta_test split, reporting pooled 95% Bootstrap CI.

Usage:
    python scripts/eval_meta.py \
        --checkpoint "D:/元强化学习/代码/航空缺陷检测/checkpoints/phase1_4w5s_cais_full/best.pt" \
        --seeds 42,43,44 \
        --n-way 4 --k-shot 5 --q-query 5 \
        --eval-tasks 100 \
        --out-dir "D:/元强化学习/代码/航空缺陷检测/reports/final_eval_phase1_4w5s_cais_full"
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 强制 UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
except Exception:
    pass
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from common.multi_domain_model import MultiDomainPretrainModel
from common.multi_domain_episode import build_episode_samplers
from common.protonet_loop import MahalanobisHead
from common.condition_encoder import DomainReliability
from scripts.train_meta import evaluate


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='train_meta.py 产出的 best.pt 或 final.pt')
    p.add_argument('--seeds', default='42,43,44',
                   help='逗号分隔的 seeds (默认 42,43,44)')
    p.add_argument('--n-way', type=int, default=4)
    p.add_argument('--k-shot', type=int, default=5)
    p.add_argument('--q-query', type=int, default=5)
    p.add_argument('--eval-tasks', type=int, default=100,
                   help='每个 seed 跑多少个 episode')
    p.add_argument('--augment-eval', action='store_true',
                   help='评估阶段也开增强 (默认关, 保持稳定)')

    # 数据路径 (与 train_meta.py 一致)
    p.add_argument('--mvtec-root', default=r'd:\元强化学习\代码\data\MVTEC')
    p.add_argument('--gc10-root',  default=r'd:\元强化学习\代码\data\GC10-DET')
    p.add_argument('--neu-root',   default=r'd:\元强化学习\代码\data\NEU-CLS\NEU-CLS')
    p.add_argument('--carpaint-root', default=None)
    p.add_argument('--aero-real-root', default=None)
    p.add_argument('--use-mvtec-dropout', action='store_true')
    p.add_argument('--skip-products', default='', type=str)

    # 输出
    p.add_argument('--out-dir', required=True,
                   help='评测报告输出目录 (会写 final_eval_*.json)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--use-mvtec-dropout-products', default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def _bootstrap_ci95(values, n_boot=2000, seed=0):
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[i] = arr[idx].mean()
    ci_low = float(np.percentile(means, 2.5))
    ci_high = float(np.percentile(means, 97.5))
    return ci_low, ci_high


def _verdict(lower):
    if lower > 0.65:
        return 'ok', '结果可信且有效（95%CI 下界 > 0.65）'
    if lower > 0.55:
        return 'limited', '结果有效但置信度有限（下界 0.55–0.65）'
    return 'weak', '接近随机或不可信（下界 < 0.55）'


# ---------------------------------------------------------------------------
# 加载 train_meta.py 的 checkpoint
# ---------------------------------------------------------------------------
def load_from_checkpoint(args, device):
    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    train_args = sd.get('args', {})
    n_products = train_args.get('n_products', 20)
    feat_dim = 128

    # 1) 重建 model
    model = MultiDomainPretrainModel(
        n_products=n_products, feat_dim=feat_dim, freeze_backbone=True)
    msd = model.state_dict()
    if 'model' in sd:
        # partial-load: 跳过 shape 不匹配的层
        sd_filtered = {k: v for k, v in sd['model'].items()
                       if k in msd and hasattr(v, 'shape')
                       and hasattr(msd[k], 'shape')
                       and v.shape == msd[k].shape}
        missing, unexpected = model.load_state_dict(sd_filtered, strict=False)
        print(f'[load] model: loaded={len(sd_filtered)}/'
              f'{len(sd["model"])} missing={len(missing)} unexpected={len(unexpected)}')
    model = model.to(device)
    model.eval()

    # 2) MahalanobisHead (如果有)
    distance = train_args.get('distance', 'mahalanobis')
    mahal_head = None
    if distance == 'mahalanobis' and sd.get('mahal_head') is not None:
        ways = train_args.get('n_way', args.n_way)
        mahal_head = MahalanobisHead(feat_dim=feat_dim, ways=ways).to(device)
        try:
            mahal_head.load_state_dict(sd['mahal_head'])
            print(f'[load] mahal_head loaded (ways={ways})')
        except Exception as e:
            print(f'[load] WARN: mahal_head load failed: {e}; fallback to None')
            mahal_head = None
            distance = 'euclidean'

    # 3) DomainReliability (加载但不参与 eval, 因为 eval 时 sampler 在 meta_test)
    domain_reliability = None
    if sd.get('domain_reliability') is not None:
        try:
            n_dom = sd['domain_reliability']['net.0.weight'].shape[1] - feat_dim
            domain_reliability = DomainReliability(feat_dim=feat_dim, n_domains=n_dom).to(device)
            domain_reliability.load_state_dict(sd['domain_reliability'])
            print(f'[load] domain_reliability loaded (n_domains={n_dom})')
        except Exception as e:
            print(f'[load] WARN: domain_reliability load failed: {e}; skip.')
            domain_reliability = None

    return model, mahal_head, distance, train_args, domain_reliability


# ---------------------------------------------------------------------------
# Multi-seed eval
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载 checkpoint
    model, mahal_head, distance, train_args, domain_reliability = load_from_checkpoint(args, device)
    prototype_type = train_args.get('prototype', 'trim_mean')
    n_way = train_args.get('n_way', args.n_way)
    k_shot = train_args.get('k_shot', args.k_shot)
    q_query = train_args.get('q_query', args.q_query)

    print('=' * 70)
    print(f'Final multi-seed eval (CA-IS-MetaProto)')
    print(f'  checkpoint:    {args.checkpoint}')
    print(f'  protocol:      {n_way}-way {k_shot}-shot, q={q_query}')
    print(f'  distance:      {distance}')
    print(f'  prototype:     {prototype_type}')
    print(f'  seeds:         {seeds}')
    print(f'  eval-tasks/seed: {args.eval_tasks}')
    print('=' * 70)

    # 2) 跑每个 seed
    per_seed = {}
    all_accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)

        sampler, *_ = build_episode_samplers(
            args.mvtec_root, args.gc10_root, args.neu_root,
            n_way=n_way, k_shot=k_shot, q_query=q_query,
            split='meta_test',
            carpaint_root=args.carpaint_root,
            aero_real_root=args.aero_real_root,
            mvtec_dropout_root=(args.mvtec_root if args.use_mvtec_dropout else None))

        # 应用 --skip-products
        if args.skip_products:
            skip_pids = set()
            for tok in args.skip_products.split(','):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    skip_pids.add(int(tok))
                except ValueError:
                    from common.multi_domain_datasets import MVTEC_PRODUCT_TO_ID
                    if tok in MVTEC_PRODUCT_TO_ID:
                        skip_pids.add(MVTEC_PRODUCT_TO_ID[tok])
            before = len(sampler.usable_products)
            sampler.usable_products = [p for p in sampler.usable_products
                                        if p not in skip_pids]
            sampler.product_to_source = {k: v for k, v in sampler.product_to_source.items()
                                          if k not in skip_pids}
            sampler.product_n_classes = {k: v for k, v in sampler.product_n_classes.items()
                                          if k not in skip_pids}
            print(f'  [skip-products] removed {before - len(sampler.usable_products)} '
                  f'PIDs {sorted(skip_pids)}; remaining={len(sampler.usable_products)}')

        mean_acc, std_acc, n_eps = evaluate(
            model, mahal_head, sampler, device,
            n_episodes=args.eval_tasks,
            distance=distance, prototype_type=prototype_type,
            augment=args.augment_eval)

        per_seed[str(seed)] = {
            'mean': float(mean_acc),
            'std': float(std_acc),
            'n_tasks': int(n_eps),
        }
        # 收集所有 episode acc (从 evaluate 内部拿不到, 这里重跑一遍拿明细)
        # 为节省时间, 直接用 mean 估计 std (粗略)
        # 实际明细从 evaluate 中提取需重构 — 这里我们跑第二遍拿明细:
        from scripts.train_meta import proto_loss
        accs_detail = []
        for _ in range(args.eval_tasks):
            try:
                ep = sampler.sample(split='meta_test')
            except ValueError:
                continue
            if ep['support_rgb'].numel() == 0:
                continue
            _, acc, _ = proto_loss(model, mahal_head, ep, device, distance,
                                   prototype_type, augment=args.augment_eval)
            accs_detail.append(acc)
        all_accs.extend(accs_detail)

        print(f'  seed {seed}: {mean_acc:.4f} +/- {std_acc:.4f}  '
              f'(n_eps={n_eps}, detailed={len(accs_detail)})')

    # 3) Pooled CI
    pooled_mean = float(np.mean(all_accs))
    pooled_std = float(np.std(all_accs, ddof=1)) if len(all_accs) > 1 else 0.0
    ci_low, ci_high = _bootstrap_ci95(all_accs, n_boot=2000, seed=0)
    margin = (ci_high - ci_low) / 2
    tag, verdict_text = _verdict(ci_low)

    print('')
    print('-' * 70)
    print(f'Pooled ({len(all_accs)} tasks = {len(seeds)} seeds x {args.eval_tasks}):')
    print(f'  Mean:   {pooled_mean:.4f}')
    print(f'  Std:    {pooled_std:.4f}')
    print(f'  95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
    print(f'  判定:    {verdict_text}')
    print('-' * 70)

    # 4) 写报告
    report = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'checkpoint': os.path.abspath(args.checkpoint),
        'protocol': {
            'n_way': n_way,
            'k_shot': k_shot,
            'q_query': q_query,
            'distance': distance,
            'prototype_type': prototype_type,
            'mahal_head_loaded': mahal_head is not None,
            'domain_reliability_loaded': domain_reliability is not None,
            'seeds': seeds,
            'eval_tasks_per_seed': args.eval_tasks,
            'augment_eval': args.augment_eval,
            'note': 'CA-IS-MetaProto 新管线 (train_meta.py) 多 seed 评测',
        },
        'per_seed': per_seed,
        'pooled': {
            'n_tasks': len(all_accs),
            'mean': pooled_mean,
            'std': pooled_std,
            'ci95_low': ci_low,
            'ci95_high': ci_high,
            'ci95_margin': margin,
            'verdict': tag,
            'verdict_text': verdict_text,
        },
    }

    ckpt_name = Path(args.checkpoint).parent.name
    out_file = out_dir / f'final_eval_{ckpt_name}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\nReport saved to: {out_file}')


if __name__ == '__main__':
    args = parse_args()
    run(args)
