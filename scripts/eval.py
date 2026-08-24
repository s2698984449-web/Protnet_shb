"""Evaluate checkpoint: single-seed or multi-seed with 95% CI on held-out domain.

Supports --seeds to run multiple seeds (e.g. "42,43,44") for final evaluation.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.eval_loop import evaluate_tasks
from common import maml_origin
from common.multimodal_model import AviationDefectModel
from common.task_builder import create_tasksets, _parse_exclude_sample_ids
from common.project_paths import apply_meta_train_defaults


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    return value in ('1', 'true', 'yes', 'y', 't')


def _parse_seeds(text):
    return [int(s.strip()) for s in text.split(',') if s.strip()]


def _ci95(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    margin = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
    return mean, std, mean - margin, mean + margin, margin


def _verdict(lower):
    if lower > 0.65:
        return 'ok', '结果可信且有效（95%CI 下界 > 0.65）'
    if lower > 0.55:
        return 'limited', '结果有效但置信度有限（下界 0.55–0.65）'
    return 'weak', '接近随机或不可信（下界 < 0.55）'


def _load_model(args, device, checkpoint_path):
    model = AviationDefectModel(
        num_classes=args.ways,
        embed_dim=args.embed_dim,
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
        rgb_cbam=getattr(args, 'rgb_cbam', False),
        use_multiscale_cff=getattr(args, 'use_multiscale_cff', False),
    ).to(device)
    if args.freeze_backbone:
        model.set_backbone_frozen(True)
    else:
        model.set_backbone_frozen(False)
    state = torch.load(checkpoint_path, map_location=device)
    # Strict=False: 新 CFF/MahalHead 结构和旧 checkpoint 的 mahal_head 可能维度不同
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'  load_state_dict strict=False: missing={missing}, unexpected={unexpected}')
    if args.meta_algorithm == 'protonet':
        return model, model
    else:
        maml = maml_origin.MAML(
            model, lr=args.fast_lr, first_order=args.first_order, allow_nograd=True)
        return model, maml


def _parse_tta(args):
    tta_str = getattr(args, 'tta_scales', '') or ''
    if not tta_str.strip():
        return None
    try:
        return tuple(sorted(int(s.strip()) for s in tta_str.split(',') if s.strip()))
    except Exception:
        print('WARNING: failed to parse tta_scales="{}", disabling TTA.'.format(tta_str))
        return None


def _run_single_seed(args, device, per_seed):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    _, test_tasks, domain_meta_map = create_tasksets(args)
    test_meta = domain_meta_map[args.test_domain]

    model, eval_target = _load_model(args, device, args.checkpoint)

    tta_scales = _parse_tta(args)
    if tta_scales:
        print('TTA enabled: scales={}'.format(tta_scales))

    mean, std, accs = evaluate_tasks(
        eval_target, test_tasks, test_meta, args.shots, args.ways,
        args.adapt_steps, args.eval_tasks, device,
        fusion_only=args.freeze_backbone,
        meta_algorithm=args.meta_algorithm,
        protonet_distance=args.protonet_distance,
        protonet_prototype_type=getattr(args, 'protonet_prototype_type', 'mean'),
        mahal_head=model.mahal_head if args.meta_algorithm == 'protonet' else None,
        adaptive_proto=model.adaptive_proto
            if args.meta_algorithm == 'protonet' and getattr(args, 'use_adaptive_proto', True)
            else None,
        tta_scales=tta_scales,
        tta_base_imgsz=args.yolo_imgsz)

    result = {
        'mean': float(mean),
        'std': float(std),
        'min': float(min(accs)),
        'max': float(max(accs)),
        'n_tasks': len(accs),
    }
    print('Seed {}: {:.4f} +/- {:.4f}  (min {:.4f}, max {:.4f})'.format(
        args.seed, mean, std, min(accs), max(accs)))
    return accs, result


def run_final_eval(args):
    """Multi-seed evaluation with pooled 95% CI."""
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError('Checkpoint not found: {}'.format(ckpt))

    if not getattr(args, '_rgb_cbam_explicit', False):
        cfg_path = os.path.join(os.path.dirname(ckpt), 'run_config.json')
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding='utf-8') as f:
                cfg = json.load(f)
            if 'rgb_cbam' in cfg:
                args.rgb_cbam = bool(cfg['rgb_cbam'])

    seeds = _parse_seeds(args.seeds)
    per_seed = {}
    all_accs = []

    print('=' * 60)
    print('Final eval (multi-seed)')
    print('  checkpoint:', ckpt)
    print('  domain_{}, {}-way {}-shot'.format(args.test_domain, args.ways, args.shots))
    if args.meta_algorithm == 'maml':
        print('  adapt_steps={}, fast_lr={}'.format(args.adapt_steps, args.fast_lr))
    else:
        print('  protonet_distance={}, protonet_prototype_type={}'.format(
            args.protonet_distance, args.protonet_prototype_type))
    print('  meta_algorithm={}, freeze_backbone={}'.format(
        args.meta_algorithm, args.freeze_backbone))
    print('  tasks/seed={}, seeds={}'.format(args.eval_tasks, seeds))
    print('=' * 60)

    for seed in seeds:
        args.seed = seed
        accs, result = _run_single_seed(args, device, per_seed)
        per_seed[str(seed)] = result
        all_accs.extend(accs)

    pooled_mean, pooled_std, ci_low, ci_high, ci_margin = _ci95(all_accs)
    tag, verdict_text = _verdict(ci_low)

    print('')
    print('Pooled ({} tasks = {} seeds x {} tasks/seed):'.format(
        len(all_accs), len(seeds), args.eval_tasks))
    print('  Mean:  {:.4f}'.format(pooled_mean))
    print('  Std:   {:.4f}'.format(pooled_std))
    print('  95%CI: [{:.4f}, {:.4f}]'.format(ci_low, ci_high))
    print('  下界:  {:.4f}'.format(ci_low))
    print('  判定:  {}'.format(verdict_text))

    report = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'checkpoint': ckpt,
        'protocol': {
            'test_domain': args.test_domain,
            'ways': args.ways,
            'shots': args.shots,
            'meta_algorithm': args.meta_algorithm,
            'adapt_steps': args.adapt_steps if args.meta_algorithm == 'maml' else None,
            'fast_lr': args.fast_lr if args.meta_algorithm == 'maml' else None,
            'protonet_distance': args.protonet_distance if args.meta_algorithm == 'protonet' else None,
            'protonet_prototype_type': args.protonet_prototype_type if args.meta_algorithm == 'protonet' else None,
            'use_adaptive_proto': getattr(args, 'use_adaptive_proto', True) if args.meta_algorithm == 'protonet' else None,
            'freeze_backbone': args.freeze_backbone,
            'rgb_only': args.rgb_only,
            'eval_tasks_per_seed': args.eval_tasks,
            'seeds': seeds,
            'query_per_task': args.shots * args.ways,
            'exclude_sample_ids': _parse_exclude_sample_ids(getattr(args, 'exclude_sample_ids', '')),
        },
        'per_seed': per_seed,
        'pooled': {
            'n_tasks': len(all_accs),
            'mean': pooled_mean,
            'std': pooled_std,
            'ci95_low': ci_low,
            'ci95_high': ci_high,
            'ci95_margin': ci_margin,
            'verdict': tag,
            'verdict_text': verdict_text,
        },
        'all_task_accs': [float(x) for x in all_accs],
    }

    if args.out_json:
        out_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print('')
        print('Report saved:', out_path)

    return report


def main():
    p = argparse.ArgumentParser(description='Evaluate checkpoint (single or multi-seed)')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--track', type=str, default='aerospace',
                   help='aerospace|automotive. Sets data paths and train_domains.')
    p.add_argument('--data_dir_path', type=str, default=None)
    p.add_argument('--dataset_name', type=str, default=None, choices=('aviation', 'cmpl'))
    p.add_argument('--test_domain', type=int, default=3)
    p.add_argument('--train_domains', type=str, default=None)
    p.add_argument('--anil', type=_str2bool, default=False)
    p.add_argument('--train_task_num', type=int, default=40)
    p.add_argument('--test_task_num', type=int, default=100)
    p.add_argument('--ways', type=int, default=2)
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--adapt_steps', type=int, default=5)
    p.add_argument('--fast_lr', type=float, default=0.02)
    p.add_argument('--first_order', type=_str2bool, default=True)
    p.add_argument('--eval_tasks', type=int, default=100)
    p.add_argument('--seeds', type=str, default='',
                   help='Comma-separated seeds for multi-seed eval. '
                        'Empty=single-seed mode (default). e.g. "42,43,44"')
    p.add_argument('--freeze_backbone', type=_str2bool, default=False)
    p.add_argument('--rgb_only', type=_str2bool, default=True)
    p.add_argument('--yolo_weights', type=str,
                   default='./weights/aviation_yolo26_pretrained.pt')
    p.add_argument('--yolo_imgsz', type=int, default=640)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--meta_algorithm', type=str, default='maml',
                   choices=('maml', 'protonet'))
    p.add_argument('--protonet_distance', type=str, default='euclidean',
                   choices=('euclidean', 'cosine', 'mahalanobis'))
    p.add_argument('--protonet_prototype_type', type=str, default='mean',
                   choices=('mean', 'trim_mean', 'mean_shift'))
    p.add_argument('--rgb_cbam', type=_str2bool, default=False)
    p.add_argument('--use_multiscale_cff', type=_str2bool, default=False)
    p.add_argument('--use_adaptive_proto', type=_str2bool, default=True,
                   help='Adaptive Prototype: cross-attention between query and prototypes (default ON)')
    p.add_argument('--cuda', type=_str2bool, default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude_sample_ids', type=str, default='')
    p.add_argument('--out_json', type=str, default=None)
    p.add_argument('--tta_scales', type=str, default='',
                   help='Comma-separated TTA scales, e.g. "320,480,640"')

    args = p.parse_args()
    args._rgb_only_explicit = '--rgb_only' in sys.argv
    args._anil_explicit = '--anil' in sys.argv
    args._meta_algorithm_explicit = '--meta_algorithm' in sys.argv
    args._rgb_cbam_explicit = '--rgb_cbam' in sys.argv
    args._use_multiscale_cff_explicit = '--use_multiscale_cff' in sys.argv

    info = apply_meta_train_defaults(args)
    if args.anil:
        args.freeze_backbone = True
    if not args.out_json:
        if args.track == 'automotive' or args.dataset_name == 'cmpl':
            args.out_json = os.path.join(info['artifacts'], 'reports', 'final_eval_cmpl.json')
        else:
            args.out_json = './reports/final_eval.json'

    run_final_eval(args)


if __name__ == '__main__':
    main()
