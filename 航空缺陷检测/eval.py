"""Evaluate checkpoint with multi-task protocol on held-out domain."""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from common.eval_loop import evaluate_tasks
from common import maml_origin
from common.multimodal_model import AviationDefectModel
from common.task_builder import create_tasksets


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    return value in ('1', 'true', 'yes', 'y', 't')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_dir_path', type=str, default='../data')
    p.add_argument('--test_domain', type=int, default=3)
    p.add_argument('--train_domains', type=str, default='0,1,2')
    p.add_argument('--train_task_num', type=int, default=40)
    p.add_argument('--ways', type=int, default=2)
    p.add_argument('--shots', type=int, default=3)
    p.add_argument('--adapt_steps', type=int, default=5)
    p.add_argument('--fast_lr', type=float, default=0.01)
    p.add_argument('--first_order', type=_str2bool, default=True)
    p.add_argument('--eval_tasks', type=int, default=100)
    p.add_argument('--freeze_backbone', type=_str2bool, default=True)
    p.add_argument('--rgb_only', type=_str2bool, default=True)
    p.add_argument('--yolo_weights', type=str, default='./weights/yolo26n.pt')
    p.add_argument('--yolo_imgsz', type=int, default=640)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--cuda', type=_str2bool, default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test_task_num', type=int, default=100)
    p.add_argument('--meta_algorithm', type=str, default='maml',
                   choices=('maml', 'protonet'))
    p.add_argument('--protonet_distance', type=str, default='euclidean',
                   choices=('euclidean', 'cosine', 'mahalanobis'))
    p.add_argument('--protonet_prototype_type', type=str, default='mean',
                   choices=('mean', 'trim_mean', 'mean_shift'),
                   help='Prototype aggregation for ProtoNet')
    p.add_argument('--rgb_cbam', type=_str2bool, default=False,
                   help='CBAM on YOLO neck (must match checkpoint, e.g. exp10)')
    p.add_argument('--tta_scales', type=str, default='',
                   help='Comma-separated TTA scales, e.g. "320,480,640". '
                        'Empty=disabled (zero-cost eval on existing exp14 checkpoint).')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    _, test_tasks, domain_meta_map = create_tasksets(args)
    test_meta = domain_meta_map[args.test_domain]

    model = AviationDefectModel(
        num_classes=args.ways,
        embed_dim=args.embed_dim,
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
        rgb_cbam=args.rgb_cbam,
    ).to(device)
    if args.freeze_backbone:
        model.set_backbone_frozen(True)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.meta_algorithm == 'protonet':
        eval_target = model
    else:
        eval_target = maml_origin.MAML(
            model, lr=args.fast_lr, first_order=args.first_order, allow_nograd=True)

    # parse TTA scales
    tta_scales = None
    tta_str = getattr(args, 'tta_scales', '') or ''
    if tta_str.strip():
        try:
            tta_scales = tuple(sorted(int(s.strip()) for s in tta_str.split(',') if s.strip()))
            print('TTA enabled: scales={}'.format(tta_scales))
        except Exception:
            print('WARNING: failed to parse tta_scales="{}", disabling TTA.'.format(tta_str))
            tta_scales = None

    mean, std, accs = evaluate_tasks(
        eval_target, test_tasks, test_meta, args.shots, args.ways,
        args.adapt_steps, args.eval_tasks, device,
        fusion_only=args.freeze_backbone,
        meta_algorithm=args.meta_algorithm,
        protonet_distance=args.protonet_distance,
        protonet_prototype_type=getattr(args, 'protonet_prototype_type', 'mean'),
        mahal_head=model.mahal_head if args.meta_algorithm == 'protonet' else None,
        tta_scales=tta_scales,
        tta_base_imgsz=args.yolo_imgsz)

    print('Algorithm {} | Domain {} | {} tasks | mean acc {:.4f} +/- {:.4f}'.format(
        args.meta_algorithm, args.test_domain, args.eval_tasks, mean, std))
    print('Min {:.4f}  Max {:.4f}'.format(min(accs), max(accs)))


if __name__ == '__main__':
    main()
