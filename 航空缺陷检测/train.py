"""Entry point: aviation multimodal defect detection with MAML."""

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import common  # noqa: F401 — ensure project root on path via train.py sys.path
from common.project_paths import apply_meta_train_defaults
from common.utils import setup_logger
import maml_train


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('1', 'true', 'yes', 'y', 't'):
        return True
    if value in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value, got {!r}.'.format(value))


def parse_args():
    parser = argparse.ArgumentParser(
        description='MAML/ANIL meta-learning (aerospace multimodal or automotive RGB CMPL)')

    parser.add_argument('--track', type=str, default='aerospace',
                        help='aerospace|automotive (or 航天|车). Sets data + artifact dirs.')
    parser.add_argument('--ways', type=int, default=2,
                        help='N-way; current Aero-engine domains mostly have 2 effective classes')
    parser.add_argument('--shots', type=int, default=3)
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Outer lr when freeze_backbone=True (proj+fusion)')
    parser.add_argument('--backbone_meta_lr', type=float, default=1e-5,
                        help='Outer lr for YOLO26 det_model when freeze_backbone=False')
    parser.add_argument('--head_meta_lr', type=float, default=1e-3,
                        help='Outer lr for proj/fusion/encoders when freeze_backbone=False')
    parser.add_argument('--fast_lr', type=float, default=0.01)
    parser.add_argument('--adapt_steps', type=int, default=5)
    parser.add_argument('--meta_batch_size', type=int, default=2)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--first_order', type=_str2bool, default=True,
                        help='True=FOMAML（4GB 默认）；False=二阶 MAML（需大显存）')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--yolo_weights', type=str, default='./weights/yolo26n.pt',
                        help='Local YOLO26 weights (avoid auto-download if SSL fails)')
    parser.add_argument('--yolo_imgsz', type=int, default=640,
                        help='YOLO26 input size; RGB is resized in encoder')
    parser.add_argument('--rgb_only', type=_str2bool, default=False,
                        help='Phase-A ablation: RGB encoder only')
    parser.add_argument('--freeze_backbone', type=_str2bool, default=True,
                        help='Phase-1: freeze YOLO26, inner loop adapts fusion only')
    parser.add_argument('--eval_tasks', type=int, default=50,
                        help='Multi-task eval on test domain (0=disable)')
    parser.add_argument('--eval_every', type=int, default=50,
                        help='Run multi-task eval every N iters')

    parser.add_argument('--stage1_end', type=int, default=100)
    parser.add_argument('--stage2_end', type=int, default=200)
    parser.add_argument('--w_rl_max', type=float, default=0.5)

    parser.add_argument('--cuda', type=_str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_dir_path', type=str, default=None,
                        help='Processed data root (default: data/{track}/processed)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        choices=('aviation', 'cmpl'),
                        help='Meta subfolder (default from --track)')
    parser.add_argument('--anil', type=_str2bool, default=False,
                        help='ANIL mode: alias for freeze_backbone=True (inner loop adapts fusion only)')
    parser.add_argument('--train_domains', type=str, default='0,1,2')
    parser.add_argument('--test_domain', type=int, default=3)
    parser.add_argument('--train_task_num', type=int, default=40)
    parser.add_argument('--test_task_num', type=int, default=50)

    parser.add_argument('--plot', type=_str2bool, default=True)
    parser.add_argument('--plot_path', type=str, default='./images')
    parser.add_argument('--plot_step', type=int, default=50)
    parser.add_argument('--log', type=_str2bool, default=True)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--checkpoint', type=_str2bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_step', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='',
                        help='Experiment subfolder under checkpoints/logs/images')
    parser.add_argument('--episode_aug', type=_str2bool, default=False,
                        help='Episode-level RGB aug on meta-train support/query only')
    parser.add_argument('--aug_illumination', type=_str2bool, default=False,
                        help='Enable illumination augmentation (gamma+brightness+specular) in episode_aug. '
                             'supersedes --episode_aug when set.')
    parser.add_argument('--input_scales', type=str, default='',
                        help='Comma-separated train input scales for multi-scale training, '
                             'e.g. "320,480,640". Empty=fixed 640. Applied in meta-train only.')
    parser.add_argument('--cross_domain_episodes', type=_str2bool, default=False,
                        help='Pool scratch/damage across train domains for meta-train episodes')
    parser.add_argument('--eval_patience', type=int, default=0,
                        help='Stop after N evals without improvement (0=disabled)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='Minimum eval gain to reset early-stop patience')
    parser.add_argument('--meta_algorithm', type=str, default='maml',
                        choices=('maml', 'protonet'),
                        help='Meta-learning algorithm: maml (inner-loop adapt) or protonet')
    parser.add_argument('--protonet_distance', type=str, default='euclidean',
                        choices=('euclidean', 'cosine', 'mahalanobis'),
                        help='Prototype distance for ProtoNet (ignored for MAML)')
    parser.add_argument('--protonet_prototype_type', type=str, default='mean',
                        choices=('mean', 'trim_mean', 'mean_shift'),
                        help='Prototype aggregation for ProtoNet: mean / trim_mean / mean_shift')
    parser.add_argument('--rgb_cbam', type=_str2bool, default=False,
                        help='CBAM attention on YOLO neck features before global pool (exp10)')

    args = parser.parse_args()
    args._rgb_only_explicit = '--rgb_only' in sys.argv
    args._anil_explicit = '--anil' in sys.argv
    args._freeze_backbone_explicit = '--freeze_backbone' in sys.argv
    args._meta_algorithm_explicit = '--meta_algorithm' in sys.argv
    args._w_rl_max_explicit = '--w_rl_max' in sys.argv
    args._stage_end_explicit = '--stage1_end' in sys.argv or '--stage2_end' in sys.argv
    args._rgb_cbam_explicit = '--rgb_cbam' in sys.argv
    apply_meta_train_defaults(args)
    return args


def _apply_exp_dirs(args):
    """Isolate each run under exp_name subfolders."""
    if not args.exp_name:
        return
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.exp_name)
    args.log_path = os.path.join(args.log_path, args.exp_name)
    args.plot_path = os.path.join(args.plot_path, args.exp_name)


def _save_run_config(args, experiment_title):
    os.makedirs(args.checkpoint_path, exist_ok=True)
    config_path = os.path.join(args.checkpoint_path, 'run_config.json')
    payload = {
        'experiment_title': experiment_title,
        'exp_name': args.exp_name,
        'ways': args.ways,
        'shots': args.shots,
        'iters': args.iters,
        'meta_lr': args.meta_lr,
        'backbone_meta_lr': args.backbone_meta_lr,
        'head_meta_lr': args.head_meta_lr,
        'fast_lr': args.fast_lr,
        'meta_batch_size': args.meta_batch_size,
        'train_domains': args.train_domains,
        'test_domain': args.test_domain,
        'yolo_weights': os.path.abspath(args.yolo_weights),
        'freeze_backbone': args.freeze_backbone,
        'rgb_only': args.rgb_only,
        'w_rl_max': args.w_rl_max,
        'stage1_end': args.stage1_end,
        'stage2_end': args.stage2_end,
        'seed': args.seed,
        'eval_every': args.eval_every,
        'eval_tasks': args.eval_tasks,
        'episode_aug': args.episode_aug,
        'cross_domain_episodes': args.cross_domain_episodes,
        'eval_patience': args.eval_patience,
        'early_stop_min_delta': args.early_stop_min_delta,
        'adapt_steps': args.adapt_steps,
        'meta_algorithm': args.meta_algorithm,
        'protonet_distance': args.protonet_distance,
        'protonet_prototype_type': args.protonet_prototype_type,
        'rgb_cbam': args.rgb_cbam,
        'dataset_name': args.dataset_name,
        'anil': getattr(args, 'anil', False),
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import json

    args = parse_args()
    if args.anil and not args._freeze_backbone_explicit:
        args.freeze_backbone = True
    _apply_exp_dirs(args)

    if args.track == 'automotive':
        prefix = 'CMPL_ProtoNet' if args.meta_algorithm == 'protonet' else 'CMPL_MAML'
    else:
        prefix = 'AviationMAML'
    experiment_title = '{}_{}w_{}s'.format(prefix, args.ways, args.shots)
    if args.rgb_only:
        experiment_title += '_rgb'
    if args.freeze_backbone:
        experiment_title += '_frozen'
    else:
        experiment_title += '_unfreeze'
    if args.episode_aug:
        experiment_title += '_epaug'
    if getattr(args, 'aug_illumination', False):
        experiment_title += '_illum'
    scales_str = getattr(args, 'input_scales', '') or ''
    if scales_str.strip():
        # abbreviate "320,480,640" -> "ms320_480_640"
        scales_abbrev = '_'.join(s.strip() for s in scales_str.split(',') if s.strip())
        experiment_title += '_ms' + scales_abbrev
    if args.cross_domain_episodes:
        experiment_title += '_crossdom'
    if args.meta_algorithm == 'protonet':
        experiment_title += '_protonet'
    if args.rgb_cbam:
        experiment_title += '_cbam'
    experiment_title += '_yolo26'

    for path in (args.plot_path, args.checkpoint_path, args.log_path):
        if path:
            os.makedirs(path, exist_ok=True)

    _save_run_config(args, experiment_title)

    if args.log:
        setup_logger(args.log_path, experiment_title)

    maml_train.train(args, experiment_title)
