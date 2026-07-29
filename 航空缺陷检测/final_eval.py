"""Multi-seed final evaluation on held-out domain (independent of training loop)."""

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

from common import maml_origin
from common.eval_loop import evaluate_tasks
from common.multimodal_model import AviationDefectModel
from common.project_paths import apply_meta_train_defaults
from common.task_builder import create_tasksets, _parse_exclude_sample_ids


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


def run_final_eval(args):
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
    print('  domain_{}, {}-way {}-shot'.format(
        args.test_domain, args.ways, args.shots))
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
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)

        _, test_tasks, domain_meta_map = create_tasksets(args)
        test_meta = domain_meta_map[args.test_domain]

        model = AviationDefectModel(
            num_classes=args.ways,
            embed_dim=args.embed_dim,
            yolo_weights=args.yolo_weights,
            yolo_imgsz=args.yolo_imgsz,
            rgb_cbam=getattr(args, 'rgb_cbam', False),
        ).to(device)
        if args.freeze_backbone:
            model.set_backbone_frozen(True)
        else:
            model.set_backbone_frozen(False)

        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)

        if args.meta_algorithm == 'protonet':
            eval_target = model
        else:
            eval_target = maml_origin.MAML(
                model, lr=args.fast_lr, first_order=args.first_order, allow_nograd=True)

        mean, std, accs = evaluate_tasks(
            eval_target, test_tasks, test_meta, args.shots, args.ways,
            args.adapt_steps, args.eval_tasks, device,
            fusion_only=args.freeze_backbone,
            meta_algorithm=args.meta_algorithm,
            protonet_distance=args.protonet_distance,
            protonet_prototype_type=getattr(args, 'protonet_prototype_type', 'mean'),
            mahal_head=model.mahal_head if args.meta_algorithm == 'protonet' else None)

        per_seed[str(seed)] = {
            'mean': float(mean),
            'std': float(std),
            'min': float(min(accs)),
            'max': float(max(accs)),
            'n_tasks': len(accs),
        }
        all_accs.extend(accs)
        print('Seed {}: {:.4f} +/- {:.4f}  (min {:.4f}, max {:.4f})'.format(
            seed, mean, std, min(accs), max(accs)))

    pooled_mean, pooled_std, ci_low, ci_high, ci_margin = _ci95(all_accs)
    tag, verdict = _verdict(ci_low)

    print('')
    print('Pooled ({} tasks = {} seeds x {} tasks/seed):'.format(
        len(all_accs), len(seeds), args.eval_tasks))
    print('  Mean:  {:.4f}'.format(pooled_mean))
    print('  Std:   {:.4f}'.format(pooled_std))
    print('  95%CI: [{:.4f}, {:.4f}]'.format(ci_low, ci_high))
    print('  下界:  {:.4f}'.format(ci_low))
    print('  判定:  {}'.format(verdict))

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
            'freeze_backbone': args.freeze_backbone,
            'rgb_only': args.rgb_only,
            'eval_tasks_per_seed': args.eval_tasks,
            'seeds': seeds,
            'query_per_task': args.shots * args.ways,
            'exclude_sample_ids': _parse_exclude_sample_ids(
                getattr(args, 'exclude_sample_ids', '')),
            'note': 'ProtoNet: prototype classify; MAML: inner-loop adapt then argmax',
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
            'verdict_text': verdict,
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
    p = argparse.ArgumentParser(description='Multi-seed final MAML eval on domain_3')
    p.add_argument('--track', type=str, default='aerospace',
                   help='aerospace|automotive (or 航天|车). Sets data paths and train_domains.')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_dir_path', type=str, default=None)
    p.add_argument('--dataset_name', type=str, default=None, choices=('aviation', 'cmpl'))
    p.add_argument('--test_domain', type=int, default=3)
    p.add_argument('--train_domains', type=str, default=None)
    p.add_argument('--anil', type=_str2bool, default=False,
                   help='ANIL: alias for freeze_backbone=True (match cmpl training)')
    p.add_argument('--train_task_num', type=int, default=40)
    p.add_argument('--test_task_num', type=int, default=100)
    p.add_argument('--ways', type=int, default=2)
    p.add_argument('--shots', type=int, default=5,
                   help='Must match training (exp04: 5)')
    p.add_argument('--adapt_steps', type=int, default=5)
    p.add_argument('--fast_lr', type=float, default=0.02,
                   help='Inner-loop lr (exp04: 0.02)')
    p.add_argument('--first_order', type=_str2bool, default=True)
    p.add_argument('--eval_tasks', type=int, default=100)
    p.add_argument('--seeds', type=str, default='42,43,44')
    p.add_argument('--freeze_backbone', type=_str2bool, default=False,
                   help='exp04_full: False')
    p.add_argument('--rgb_only', type=_str2bool, default=True)
    p.add_argument('--yolo_weights', type=str, default='./weights/aviation_yolo26_pretrained.pt',
                   help='Architecture init only; weights loaded from --checkpoint')
    p.add_argument('--yolo_imgsz', type=int, default=640)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--meta_algorithm', type=str, default='maml',
                   choices=('maml', 'protonet'))
    p.add_argument('--protonet_distance', type=str, default='euclidean',
                   choices=('euclidean', 'cosine', 'mahalanobis'))
    p.add_argument('--protonet_prototype_type', type=str, default='mean',
                   choices=('mean', 'trim_mean', 'mean_shift'),
                   help='Prototype aggregation for ProtoNet')
    p.add_argument('--rgb_cbam', type=_str2bool, default=False,
                   help='Must match training (exp10: True)')
    p.add_argument('--cuda', type=_str2bool, default=True)
    p.add_argument('--exclude_sample_ids', type=str, default='',
                   help='Comma-separated sample folder names to drop from test domain only')
    p.add_argument('--out_json', type=str, default=None)
    args = p.parse_args()
    args._rgb_only_explicit = '--rgb_only' in sys.argv
    args._anil_explicit = '--anil' in sys.argv
    args._meta_algorithm_explicit = '--meta_algorithm' in sys.argv
    args._rgb_cbam_explicit = '--rgb_cbam' in sys.argv
    info = apply_meta_train_defaults(args)
    if args.anil:
        args.freeze_backbone = True
    if not args.out_json:
        if args.track == 'automotive' or args.dataset_name == 'cmpl':
            args.out_json = os.path.join(info['artifacts'], 'reports', 'final_eval_cmpl.json')
        else:
            args.out_json = './reports/final_eval_exp04.json'
    run_final_eval(args)


if __name__ == '__main__':
    main()
