"""Project-root paths for aerospace vs automotive (car) tracks."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(ROOT, 'data')

AEROSPACE = os.path.join(DATA_ROOT, '航天')
AUTOMOTIVE = os.path.join(DATA_ROOT, '车')

AEROSPACE_RAW = os.path.join(AEROSPACE, 'raw')
AEROSPACE_PROCESSED = os.path.join(AEROSPACE, 'processed')

AUTOMOTIVE_RAW = os.path.join(AUTOMOTIVE, 'raw')
AUTOMOTIVE_PROCESSED = os.path.join(AUTOMOTIVE, 'processed')

# Artifact roots (weights / checkpoints / logs / plots) — under outputs/
AEROSPACE_ARTIFACTS = os.path.join(ROOT, 'outputs', '航天')
AUTOMOTIVE_ARTIFACTS = os.path.join(ROOT, 'outputs', '车')

# Meta-learning layout: {processed}/{dataset_name}/domain_{id}/...
AEROSPACE_DATASET_NAME = 'aviation'
AUTOMOTIVE_DATASET_NAME = 'cmpl'

AEROSPACE_META_ROOT = os.path.join(AEROSPACE_PROCESSED, AEROSPACE_DATASET_NAME)
AUTOMOTIVE_META_ROOT = os.path.join(AUTOMOTIVE_PROCESSED, AUTOMOTIVE_DATASET_NAME)

AEROSPACE_PRETRAIN_ROOT = os.path.join(AEROSPACE_PROCESSED, 'pretrain')
AUTOMOTIVE_PRETRAIN_ROOT = os.path.join(AUTOMOTIVE_PROCESSED, 'pretrain')

# Code entry point directory (train / pretrain / eval scripts live here)
CODE_DIR = os.path.join(ROOT, '航空缺陷检测')
LEGACY_WEIGHTS = os.path.join(CODE_DIR, 'weights')  # deprecated, kept for compat only


# Phase0 v6 (CMPL, 2026-07): car-only. 删除 neu/asde/kolektor/mvtec_*
#   - 4 个 task: cardd + carparts + minor_dent + syndcar (全部与车漆相关)
#   - combined mix + label_namespace=global + aux_scratch_head (层次1融合)
#   - 样本分布: cardd=6882, carparts=9084, minor_dent=3158, syndcar=1295
#   - 权重策略:
#       * cardd (真实多变工况) -> 最高 0.40
#       * carparts (细粒度 8 类, 车间集中光单一工况) -> 0.30
#       * minor_dent (单类大样本 dent) -> 0.20
#       * syndcar (合成 3D 渲染) -> 0.10 (避免污染真实光照理解)
# 任务类总样本: 9084 + 6882 + 3158 + 1295 = 20419 ROI
# 与 test_pool (CarDD test ~543 ROI) 比例 ~37:1, 远超 few-shot 收敛所需
PRETRAIN_TASKS_AUTOMOTIVE_V6 = (
    ('cardd',      4, 0.40),
    ('carparts',   8, 0.30),
    ('minor_dent', 1, 0.20),
    ('syndcar',    4, 0.10),
)

PRETRAIN_TASKS_AUTOMOTIVE_NEU = (
    ('neu_cls',    6, 1.0),
)

PRETRAIN_TASKS = {
    'aerospace': (
        ('blade', 5, 0.7),
        ('aebad', 2, 0.3),
    ),
    # CMPL v6 default (car-only): 不再含 neu/asde/kolektor/mvtec
    'automotive': PRETRAIN_TASKS_AUTOMOTIVE_V6,
}

# ---- 兼容旧实验的 stub ----
# 旧 variant (v4 / classsplit* / v5) 已被删除, 因为依赖 neu/asde/kolektor/mvtec_*,
# 这些数据集在 v6 中移除了. 报清晰错误优于静默断流.
_DEPRECATED_VARIANTS = {
    'v4', 'supcon', 'p0_v4',  # 依赖已删除的 asde
    'classsplit_neu', 'p0_cs_neu', 'neu_only', 'class_split_neu',  # 依赖 neu
    'classsplit3', 'p0_cs3', 'classsplit', 'class_split3',
    'classsplit3_v2', 'p0_cs3_v2', 'class_split3_v2',  # 依赖 neu+mvtec
    # v5 已被 v6 取代
    'v5', 'p0_v5', 'cmpl_v5', 'car_v5',
}

# 跨 task 的同名划痕类别（用于 auxiliary scratch binary head, CMPL v6 精简）
#   cardd.scratch + carparts.Scratch + syndcar.scratch  → is_scratch=1
#   其余所有类（dent/crack/glass/paint_chip/...）→ is_scratch=0
SCRATCH_CLASS_NAMES = frozenset({
    'scratch',   # cardd
    'Scratch',   # carparts
})


def _artifact_subdir(artifacts_root, name):
    return os.path.join(artifacts_root, name)


def resolve_track(track):
    track = str(track).strip().lower()
    if track in ('aerospace', 'aviation', '航天', 'hangtian'):
        artifacts = AEROSPACE_ARTIFACTS
        return {
            'track': 'aerospace',
            'raw': AEROSPACE_RAW,
            'processed': AEROSPACE_PROCESSED,
            'dataset_name': AEROSPACE_DATASET_NAME,
            'pretrain_root': AEROSPACE_PRETRAIN_ROOT,
            'artifacts': artifacts,
            'weights_dir': _artifact_subdir(artifacts, 'weights'),
            'checkpoints_dir': _artifact_subdir(artifacts, 'checkpoints'),
            'logs_dir': _artifact_subdir(artifacts, 'logs'),
            'images_dir': _artifact_subdir(artifacts, 'images'),
            'pretrain_weights_name': 'aviation_yolo26_pretrained.pt',
            'pretrain_ckpt_name': 'pretrain_best.pt',
            'yolo_init_name': 'yolo26n.pt',
            'pretrain_tasks': PRETRAIN_TASKS['aerospace'],
        }
    if track in ('automotive', 'car', 'cmpl', '车', 'che'):
        artifacts = AUTOMOTIVE_ARTIFACTS
        return {
            'track': 'automotive',
            'raw': AUTOMOTIVE_RAW,
            'processed': AUTOMOTIVE_PROCESSED,
            'dataset_name': AUTOMOTIVE_DATASET_NAME,
            'pretrain_root': AUTOMOTIVE_PRETRAIN_ROOT,
            'artifacts': artifacts,
            'weights_dir': _artifact_subdir(artifacts, 'weights'),
            'checkpoints_dir': _artifact_subdir(artifacts, 'checkpoints'),
            'logs_dir': _artifact_subdir(artifacts, 'logs'),
            'images_dir': _artifact_subdir(artifacts, 'images'),
            'pretrain_weights_name': 'cmpl_yolo26_pretrained.pt',
            'pretrain_ckpt_name': 'pretrain_best.pt',
            'yolo_init_name': 'yolo26n.pt',
            'pretrain_tasks': PRETRAIN_TASKS['automotive'],
        }
    raise ValueError('Unknown track: {!r}. Use aerospace or automotive.'.format(track))


def _first_existing(*paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return paths[0] if paths else ''


def _apply_automotive_supcon_defaults(args):
    """Shared defaults for v6 / v7 / v8 / v9 automotive supcon variants.

    v9 (2026-07): switch label_namespace default to per_task.  v6/v7/v8 used
    'global' which merged cardd.scratch and carparts.Scratch into one SupCon
    cluster — empirically this causes negative transfer because the two
    visual domains (full-car vs. part close-up) are far apart.  Cross-task
    scratch-family fusion is now handled by the auxiliary scratch binary
    head (--aux_scratch_head), which is decoupled from the SupCon loss.
    Aerospace is not affected: this function only runs when track=automotive.
    """
    if getattr(args, 'backbone_train_mode', None) in (None, ''):
        args.backbone_train_mode = 'indices'
    if getattr(args, 'freeze_indices', None) in (None, ''):
        args.freeze_indices = '0,1,2,3,4,5'
    if getattr(args, 'loss_mode', None) in (None, 'ce'):
        args.loss_mode = 'supcon_ce'
    if getattr(args, 'mix_strategy', None) in (None, 'alternate'):
        args.mix_strategy = 'combined'
    # v9: per_task by default (v6/v7/v8 explicitly set 'global' via YAML).
    if getattr(args, 'label_namespace', None) in (None, 'per_task'):
        # Only override when YAML didn't specify. If the experiment YAML says
        # 'global' we still respect it.
        if getattr(args, '_label_namespace_explicit', False):
            pass  # respect YAML
        else:
            args.label_namespace = 'per_task'
    if getattr(args, 'proj_dim', None) in (None, 256):
        args.proj_dim = 64
    if getattr(args, 'feat_dim', None) in (None, 256):
        args.feat_dim = 256
    raw = getattr(args, 'aux_scratch_head', None)
    if raw is None or raw is False:
        args.aux_scratch_head = True


def apply_pretrain_variant(args):
    """Override task mix for automotive Phase0 variants.

    CMPL v6: 仅支持 v6 (car-only). 老 variant (v4 / classsplit* / v5) 已被删除,
    因为依赖 neu/asde/kolektor/mvtec_* 工业数据集, 这些数据在 v6 中已彻底删除.
    """
    variant = getattr(args, 'pretrain_variant', None)
    if not variant:
        return
    variant = str(variant).strip().lower()
    track = getattr(args, 'track', 'aerospace')
    if track != 'automotive':
        return  # aerospace 不变

    # v6 主路径 (car-only 4 task combined+global supcon+aux_scratch_head)
    if variant in ('v6', 'p0_v6', 'cmpl_v6', 'car_v6'):
        tasks = PRETRAIN_TASKS_AUTOMOTIVE_V6
        _apply_automotive_supcon_defaults(args)

    # v7: 与 v6 相同数据组成, 仅 hyper-param 不同 (tau=0.05 + adapt + class-balanced)
    elif variant in ('v7', 'p0_v7', 'cmpl_v7', 'car_v7'):
        tasks = PRETRAIN_TASKS_AUTOMOTIVE_V6
        _apply_automotive_supcon_defaults(args)

    # v8: 与 v6/v7 相同数据组成, loss strategy 不同 (focal + uncertainty weight)
    elif variant in ('v8', 'p0_v8', 'cmpl_v8', 'car_v8'):
        tasks = PRETRAIN_TASKS_AUTOMOTIVE_V6
        _apply_automotive_supcon_defaults(args)

    # v9: per_task SupCon (avoid cross-task negative transfer) + LDAM for carparts
    #     + global retrieval top-1 evaluation + K-fold val split + uncertainty warmup.
    #     task composition controlled by --active_tasks (v9 default drops minor_dent & syndcar).
    elif variant in ('v9', 'p0_v9', 'cmpl_v9', 'car_v9'):
        tasks = PRETRAIN_TASKS_AUTOMOTIVE_V6
        _apply_automotive_supcon_defaults(args)

    # NEU-only Phase 0: 单任务 NEU-CLS 6 类预训练, 输出 neu_yolo26_supcon.pt
    elif variant in ('neu', 'neu_cls', 'neu_p0', 'p0_neu'):
        tasks = PRETRAIN_TASKS_AUTOMOTIVE_NEU
        _apply_automotive_supcon_defaults(args)
    elif variant in _DEPRECATED_VARIANTS:
        raise ValueError(
            "pretrain_variant='{var}' 已弃用 (依赖 neu/asde/kolektor/mvtec_* 工业数据集, "
            "这些数据集在 v6 中已彻底删除). 请使用 pretrain_variant='v6' (car-only). "
            "如需类似功能, v6 已用 carparts (8 类细粒度) + minor_dent (单类) + "
            "syndcar (合成 4 类) 替代.".format(var=variant))
    else:
        raise ValueError('Unknown pretrain_variant {!r} for track {!r}.'.format(variant, track))
    args.task_specs = [(name, n_cls) for name, n_cls, _w in tasks]
    args.task_weights = {name: weight for name, _n, weight in tasks}


def apply_pretrain_defaults(args):
    """Fill Phase-0 paths from --track when not explicitly overridden."""
    info = resolve_track(getattr(args, 'track', 'aerospace'))
    args.track = info['track']
    args.task_specs = [(name, n_cls) for name, n_cls, _w in info['pretrain_tasks']]
    args.task_weights = {name: weight for name, _n, weight in info['pretrain_tasks']}

    if not getattr(args, 'data_dir', None):
        args.data_dir = info['processed']
    if not getattr(args, 'out_weights', None):
        args.out_weights = os.path.join(info['weights_dir'], info['pretrain_weights_name'])
    if not getattr(args, 'out_checkpoint', None):
        args.out_checkpoint = os.path.join(info['checkpoints_dir'], info['pretrain_ckpt_name'])
    if not getattr(args, 'log_path', None):
        args.log_path = info['logs_dir']
    if not getattr(args, 'yolo_weights', None):
        args.yolo_weights = _first_existing(
            os.path.join(info['weights_dir'], info['yolo_init_name']),
            os.path.join(LEGACY_WEIGHTS, info['yolo_init_name']),
        )
    return info


def apply_meta_train_defaults(args):
    """Fill MAML/ANIL paths and hyper-params from --track when using defaults."""
    info = resolve_track(getattr(args, 'track', 'aerospace'))
    args.track = info['track']

    if getattr(args, 'data_dir_path', None) in (None, '', 'data', '../data', '../data/航天/processed', '../data/车/processed'):
        args.data_dir_path = info['processed']
    if getattr(args, 'dataset_name', None) is None:
        args.dataset_name = info['dataset_name']

    if not getattr(args, 'checkpoint_path', None) or args.checkpoint_path in ('./checkpoints',):
        args.checkpoint_path = info['checkpoints_dir']
    if not getattr(args, 'log_path', None) or args.log_path in ('./logs',):
        args.log_path = info['logs_dir']
    if not getattr(args, 'plot_path', None) or args.plot_path in ('./images',):
        args.plot_path = info['images_dir']
    if not getattr(args, 'yolo_weights', None) or args.yolo_weights in (
        './weights/yolo26n.pt', './weights/aviation_yolo26_pretrained.pt',
        './weights/cmpl_yolo26_pretrained.pt',
    ):
        args.yolo_weights = _first_existing(
            os.path.join(info['weights_dir'], info['pretrain_weights_name']),
            os.path.join(LEGACY_WEIGHTS, info['pretrain_weights_name']),
            os.path.join(info['weights_dir'], info['yolo_init_name']),
            os.path.join(LEGACY_WEIGHTS, info['yolo_init_name']),
        )

    if info['track'] == 'automotive':
        if getattr(args, 'train_domains', None) in (None, '0,1,2', '0,1,2,'):
            args.train_domains = '0'
        if getattr(args, 'test_domain', None) == 3 and info['track'] == 'automotive':
            pass  # default 3 is correct
        if hasattr(args, 'rgb_only') and not getattr(args, '_rgb_only_explicit', False):
            args.rgb_only = True
        if hasattr(args, 'meta_algorithm') and not getattr(args, '_meta_algorithm_explicit', False):
            args.meta_algorithm = 'protonet'
        if hasattr(args, 'anil') and not getattr(args, '_anil_explicit', False):
            if not getattr(args, '_freeze_backbone_explicit', False):
                args.anil = True
                args.freeze_backbone = True
        if hasattr(args, 'w_rl_max') and not getattr(args, '_w_rl_max_explicit', False):
            args.w_rl_max = 0.0
        if hasattr(args, 'stage1_end') and not getattr(args, '_stage_end_explicit', False):
            args.stage1_end = 1000
            args.stage2_end = 1000
        if getattr(args, 'shots', None) == 3:
            args.shots = 5
    elif getattr(args, 'train_domains', None) is None:
        args.train_domains = '0,1,2'
    return info
