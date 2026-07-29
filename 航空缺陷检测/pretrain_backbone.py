"""Phase 0: multi-task YOLO26 backbone pretraining (aerospace or automotive track)."""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import common  # noqa: F401
from common.project_paths import apply_pretrain_defaults, apply_pretrain_variant
from common.utils import setup_logger
import pretrain_train


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
        description='Phase 0: YOLO26 backbone pretrain (aerospace: bladeSynth+AeBAD; '
                    'automotive: CarDD+NEU+ASDE)')

    parser.add_argument('--track', type=str, default='aerospace',
                        help='aerospace|automotive (or 航天|车). Sets data + output dirs.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Processed data root (default: data/{track}/processed)')
    parser.add_argument('--yolo_weights', type=str, default=None,
                        help='Initial YOLO26 weights (default: {track}/weights/yolo26n.pt)')
    parser.add_argument('--yolo_imgsz', type=int, default=640)
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Shared embedding dim before task heads (classsplit locks 128)')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--backbone_train_mode', type=str, default=None,
                        help='frozen|partial|full|indices; overrides unfreeze_epoch schedule when set')
    parser.add_argument('--freeze_indices', type=str, default='0,1,2,3,4,5',
                        help='YOLO block indices to freeze when backbone_train_mode=indices')

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--mix_strategy', choices=('alternate', 'combined'), default='alternate',
                        help='alternate: one task per step; combined: weighted sum each step')
    parser.add_argument('--unfreeze_epoch', type=int, default=16,
                        help='Epoch to unfreeze last YOLO blocks')
    parser.add_argument('--unfreeze_last_n', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_steps_per_epoch', type=int, default=0,
                        help='Cap steps per epoch for debug (0=full epoch)')

    parser.add_argument('--multi_gpu', choices=('none', 'dp', 'ddp', 'auto'), default='auto',
                        help='none=single GPU/CPU; dp=DataParallel; ddp=DistributedDataParallel; '
                             'auto=ddp if RANK set else dp if 2+ GPUs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', type=_str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--out_weights', type=str, default=None,
                        help='Output pretrained backbone (default: {track}/weights/*.pt)')
    parser.add_argument('--out_checkpoint', type=str, default=None,
                        help='Best checkpoint (default: {track}/checkpoints/pretrain_best.pt)')
    parser.add_argument('--log', type=_str2bool, default=True)
    parser.add_argument('--log_path', type=str, default=None,
                        help='Log dir (default: {track}/logs)')

    parser.add_argument('--loss_mode', choices=('ce', 'supcon_ce'), default='ce',
                        help='ce=classification only; supcon_ce=0.5*SupCon+0.5*CE per task')
    parser.add_argument('--supcon_weight', type=float, default=0.5)
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--proj_dim', type=int, default=64,
                        help='Projection dim for SupCon head')
    parser.add_argument('--supcon_temperature', type=float, default=0.07)
    parser.add_argument('--label_namespace', choices=('per_task', 'global'), default=None,
                        help='per_task: each task computes SupCon independently with its own labels; '
                             'global: tasks contribute features into one SupCon batch with offset labels (v5). '
                             'When unset, the v9 default is per_task (avoids negative transfer from '
                             'cross-task clusters; cross-task scratch fusion is delegated to --aux_scratch_head).')
    parser.add_argument('--temperature_adapt', type=_str2bool, default=False,
                        help='Adapt SupCon temperature dynamically by per-batch feature norm (experimental)')
    parser.add_argument('--aux_scratch_head', type=_str2bool, default=False,
                        help='Enable auxiliary scratch binary head (层次1 cross-task fusion)')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Run validation every N epochs (smoke can set 99 to skip)')
    parser.add_argument('--pretrain_variant', type=str, default=None,
                        help='v4=CarDD+ASDE SupCon; classsplit_neu=NEU-only; '
                             'classsplit3 / classsplit3_v2=NEU+MVTec mix; v5=6-task combined+global SupCon+aux')
    # v7 additions: all default off — aerospace / legacy configs unaffected
    parser.add_argument('--supcon_min_class_fallback', type=_str2bool, default=False,
                        help='When a batch has zero positive pairs, fall back to SimCLR-style '
                             'self-contrast so the anchor still receives gradient.')
    parser.add_argument('--pretrain_sampler', type=str, default=None,
                        help='class_balanced|None; per-task WeightedRandomSampler.')
    parser.add_argument('--pretrain_sampler_apply', type=str, default='',
                        help='Comma-separated task names to apply sampler to, e.g. carparts,syndcar')
    parser.add_argument('--per_task_class_weight', type=str, default=None,
                        help='none|inverse_sqrt|inverse; per-class weight in CE loss.')
    parser.add_argument('--per_task_class_weight_apply', type=str, default='',
                        help='Comma-separated task names to apply class_weight to, e.g. carparts,syndcar')
    # v8 additions: focal loss + uncertainty weighting
    parser.add_argument('--use_focal_for', type=str, default='',
                        help='Comma-separated task names to apply Focal Loss instead of CE, e.g. carparts')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter (default 2.0)')
    parser.add_argument('--uncertainty_weighting', type=_str2bool, default=False,
                        help='Enable learnable uncertainty weighting per task (Kendall et al., 2018)')
    parser.add_argument('--uncertainty_warmup_epochs', type=int, default=0,
                        help='Disable uncertainty_weighting for the first N epochs (gradient warmup).')
    parser.add_argument('--use_ldam_for', type=str, default='',
                        help='Comma-separated task names to use LDAM Loss (Cao et al., 2019).')
    parser.add_argument('--ldam_max_margin', type=float, default=0.3,
                        help='LDAM scaling factor s (max margin).')
    parser.add_argument('--eval_metric', type=str, default='mean_acc',
                        help='mean_acc | global_retrieval_top1. '
                             'global_retrieval_top1 builds a shared embedding space and computes '
                             'cross-task nearest-neighbor top-1 accuracy using the supcon projection.')
    parser.add_argument('--val_split_strategy', type=str, default='random',
                        help='random | stratified_kfold. Applies to datasets without official split '
                             '(carparts, syndcar, minor_dent).')
    parser.add_argument('--val_split_fold', type=int, default=0,
                        help='Fold index when val_split_strategy=stratified_kfold (0..K-1).')
    parser.add_argument('--val_split_k', type=int, default=5,
                        help='Number of folds when val_split_strategy=stratified_kfold.')
    parser.add_argument('--active_tasks', type=str, default='',
                        help='Comma-separated subset of task names to use. Empty=use all from task_specs. '
                             'Useful for v9: drop minor_dent / syndcar without changing code.')

    args = parser.parse_args()
    # Mark whether the user (or YAML) explicitly set label_namespace; used by
    # the v9 default resolver to decide whether to override.
    import yaml as _yaml
    if '--label_namespace' in sys.argv:
        args._label_namespace_explicit = True
    else:
        args._label_namespace_explicit = False
    apply_pretrain_defaults(args)
    apply_pretrain_variant(args)

    if args.multi_gpu == 'auto':
        if 'RANK' in os.environ:
            args.multi_gpu = 'ddp'
        elif args.cuda and __import__('torch').cuda.device_count() > 1:
            args.multi_gpu = 'dp'
        else:
            args.multi_gpu = 'none'
    return args


if __name__ == '__main__':
    args = parse_args()
    loss_tag = getattr(args, 'loss_mode', 'ce')
    experiment_title = 'PretrainBackbone_{}_{}_{}_{}'.format(
        args.track, loss_tag, args.mix_strategy, args.multi_gpu)

    for path in (os.path.dirname(args.out_weights), os.path.dirname(args.out_checkpoint), args.log_path):
        if path and not os.path.exists(path):
            os.makedirs(path)

    if args.log and (not 'RANK' in os.environ or int(os.environ.get('RANK', 0)) == 0):
        setup_logger(args.log_path, experiment_title)

    pretrain_train.train(args)
