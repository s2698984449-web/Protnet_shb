"""
P0: Hard-task diagnosis for ProtoNet on domain_3.

Traces each eval episode to ROI paths / specimen IDs; finds tasks with acc < threshold
and query samples that fail most often.

Run from 航空缺陷检测/:
    python scripts/diagnose_hard_tasks.py
    python scripts/diagnose_hard_tasks.py --checkpoint ./checkpoints/exp08_protonet_unfreeze/best.pt
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.multimodal_model import AviationDefectModel
from common.task_builder import create_tasksets
from common.task_eval_detail import evaluate_tasks_detailed_protonet


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    return value in ('1', 'true', 'yes', 'y', 't')


def _parse_seeds(text):
    return [int(s.strip()) for s in text.split(',') if s.strip()]


def _summarize(all_tasks, all_query_stats, hard_threshold):
    accs = [t['acc'] for t in all_tasks]
    hard_tasks = [t for t in all_tasks if t['hard']]

    specimen_errors = defaultdict(lambda: {'query': 0, 'wrong': 0})
    class_confusion = Counter()
    for t in all_tasks:
        for q in t['query']:
            sp = q['specimen_id']
            specimen_errors[sp]['query'] += 1
            specimen_errors[sp]['wrong'] += 0 if q['correct'] else 1
            if not q['correct']:
                true_name = 'scratch' if q['true_label'] == 0 else 'damage'
                pred_name = 'scratch' if q['pred_label'] == 0 else 'damage'
                class_confusion['{}->{}'.format(true_name, pred_name)] += 1

    sample_rows = []
    for path, st in all_query_stats.items():
        if st['query_count'] == 0:
            continue
        err_rate = st['wrong_count'] / st['query_count']
        sample_rows.append({
            'sample_id': st['sample_id'],
            'specimen_id': st['specimen_id'],
            'class_name': st['class_name'],
            'query_count': st['query_count'],
            'wrong_count': st['wrong_count'],
            'error_rate': round(err_rate, 4),
            'sample_dir': st['sample_dir'],
        })
    sample_rows.sort(key=lambda r: (-r['error_rate'], -r['wrong_count']))

    specimen_rows = []
    for sp, st in specimen_errors.items():
        if st['query'] == 0:
            continue
        specimen_rows.append({
            'specimen_id': sp,
            'query_count': st['query'],
            'wrong_count': st['wrong'],
            'error_rate': round(st['wrong'] / st['query'], 4),
        })
    specimen_rows.sort(key=lambda r: (-r['error_rate'], -r['wrong_count']))

    return {
        'n_tasks': len(all_tasks),
        'mean_acc': float(np.mean(accs)),
        'std_acc': float(np.std(accs)),
        'min_acc': float(min(accs)),
        'max_acc': float(max(accs)),
        'hard_threshold': hard_threshold,
        'n_hard_tasks': len(hard_tasks),
        'hard_task_rate': round(len(hard_tasks) / max(len(all_tasks), 1), 4),
        'class_confusion_query': dict(class_confusion),
        'hardest_samples': sample_rows[:20],
        'hardest_specimens': specimen_rows,
        'hard_tasks_preview': [
            {
                'seed': t['seed'],
                'task_id': t['task_id'],
                'acc': t['acc'],
                'query_specimens': sorted(set(q['specimen_id'] for q in t['query'])),
                'wrong_queries': [
                    {
                        'sample_id': q['sample_id'],
                        'class_name': q['class_name'],
                        'true_label': q['true_label'],
                        'pred_label': q['pred_label'],
                    }
                    for q in t['query'] if not q['correct']
                ],
            }
            for t in sorted(hard_tasks, key=lambda x: x['acc'])[:15]
        ],
    }


def _print_summary(summary):
    print('\n' + '=' * 60)
    print('Hard-task diagnosis summary')
    print('=' * 60)
    print('Tasks: {}  mean acc: {:.4f}  hard (acc<{:.2f}): {} ({:.1%})'.format(
        summary['n_tasks'], summary['mean_acc'], summary['hard_threshold'],
        summary['n_hard_tasks'], summary['hard_task_rate']))
    print('Acc range: [{:.2f}, {:.2f}]'.format(summary['min_acc'], summary['max_acc']))

    print('\nQuery confusion (wrong only):')
    if summary['class_confusion_query']:
        for k, v in sorted(summary['class_confusion_query'].items(), key=lambda x: -x[1]):
            print('  {} : {}'.format(k, v))
    else:
        print('  (none)')

    print('\nHardest specimens (by query error rate):')
    for row in summary['hardest_specimens'][:10]:
        print('  specimen {}  err {}/{} ({:.1%})'.format(
            row['specimen_id'], row['wrong_count'], row['query_count'], row['error_rate']))

    print('\nHardest ROI samples (top 10):')
    for row in summary['hardest_samples'][:10]:
        print('  {} [{}]  err {}/{} ({:.1%})'.format(
            row['sample_id'], row['class_name'],
            row['wrong_count'], row['query_count'], row['error_rate']))

    print('\nHardest tasks (preview):')
    for t in summary['hard_tasks_preview'][:5]:
        print('  seed={} task={} acc={:.2f} specimens={} wrong={}'.format(
            t['seed'], t['task_id'], t['acc'], t['query_specimens'],
            len(t['wrong_queries'])))


def run(args):
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError('Checkpoint not found: {}'.format(ckpt))

    seeds = _parse_seeds(args.seeds)
    all_tasks = []
    merged_query_stats = defaultdict(lambda: {
        'query_count': 0, 'wrong_count': 0,
        'class_name': '', 'specimen_id': '', 'sample_id': '', 'sample_dir': '',
    })

    print('Hard-task diagnosis | checkpoint:', ckpt)
    print('seeds={}, tasks/seed={}, hard_threshold={}'.format(
        seeds, args.eval_tasks, args.hard_threshold))

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
        model.load_state_dict(torch.load(ckpt, map_location=device))

        tasks, q_stats = evaluate_tasks_detailed_protonet(
            model, test_tasks, test_meta, args.shots, args.ways,
            args.eval_tasks, device, distance=args.protonet_distance,
            hard_threshold=args.hard_threshold)

        for t in tasks:
            t['seed'] = seed
            all_tasks.append(t)
        for path, st in q_stats.items():
            m = merged_query_stats[path]
            m['query_count'] += st['query_count']
            m['wrong_count'] += st['wrong_count']
            m.update({
                'class_name': st['class_name'],
                'specimen_id': st['specimen_id'],
                'sample_id': st['sample_id'],
                'sample_dir': st['sample_dir'],
            })

        seed_accs = [t['acc'] for t in tasks]
        print('Seed {}: mean {:.4f}, hard tasks {}'.format(
            seed, np.mean(seed_accs), sum(t['hard'] for t in tasks)))

    summary = _summarize(all_tasks, merged_query_stats, args.hard_threshold)
    _print_summary(summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'checkpoint': ckpt,
        'seeds': seeds,
        'eval_tasks_per_seed': args.eval_tasks,
        'hard_threshold': args.hard_threshold,
        'summary': summary,
    }
    if args.save_tasks:
        report['tasks'] = all_tasks

    out_json = out_dir / args.out_name
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('\nReport saved:', out_json)
    return report


def main():
    p = argparse.ArgumentParser(description='Hard-task diagnosis (ProtoNet, domain_3)')
    p.add_argument('--checkpoint', type=str,
                   default='./checkpoints/exp08_protonet_unfreeze/best.pt')
    p.add_argument('--data_dir_path', type=str, default='../data/航天/processed')
    p.add_argument('--dataset_name', type=str, default='aviation')
    p.add_argument('--test_domain', type=int, default=3)
    p.add_argument('--train_domains', type=str, default='0,1,2')
    p.add_argument('--train_task_num', type=int, default=40)
    p.add_argument('--test_task_num', type=int, default=100)
    p.add_argument('--ways', type=int, default=2)
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--protonet_distance', type=str, default='euclidean')
    p.add_argument('--eval_tasks', type=int, default=100)
    p.add_argument('--seeds', type=str, default='42,43,44')
    p.add_argument('--hard_threshold', type=float, default=0.7)
    p.add_argument('--freeze_backbone', type=_str2bool, default=False)
    p.add_argument('--rgb_only', type=_str2bool, default=True)
    p.add_argument('--yolo_weights', type=str,
                   default='./weights/aviation_yolo26_pretrained.pt')
    p.add_argument('--yolo_imgsz', type=int, default=640)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--rgb_cbam', type=_str2bool, default=False)
    p.add_argument('--cuda', type=_str2bool, default=True)
    p.add_argument('--out_dir', type=str, default='./reports/diagnose')
    p.add_argument('--out_name', type=str, default='hard_tasks_exp08.json')
    p.add_argument('--save_tasks', type=_str2bool, default=False,
                   help='Include full per-task lists in JSON (large file)')
    run(p.parse_args())


if __name__ == '__main__':
    main()
