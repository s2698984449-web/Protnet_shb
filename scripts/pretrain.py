"""Phase 0: YOLO26 backbone pretraining (aerospace or automotive).

Merged from pretrain_backbone.py (entry) + pretrain_train.py (loop).
"""

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import common  # noqa: F401
from common.cmpl_pretrain_datasets import (
    CarDDPretrainDataset,
    CarPartsPretrainDataset, MinorDentPretrainDataset, SyndcarPretrainDataset,
    NEUPretrainDataset,
)
from common.pretrain_datasets import AeBADBinaryPretrainDataset, BladeSynthPretrainDataset
from common.pretrain_model import PretrainMultitaskModel, export_pretrained_yolo
from common.project_paths import apply_pretrain_defaults, apply_pretrain_variant
from common.utils import setup_logger


# ─────────────────────────── argparse ───────────────────────────

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
                        help='aerospace|automotive')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Processed data root (default: data/{track}/processed)')
    parser.add_argument('--yolo_weights', type=str, default=None,
                        help='Initial YOLO26 weights (default: {track}/weights/yolo26n.pt)')
    parser.add_argument('--yolo_imgsz', type=int, default=640)
    parser.add_argument('--feat_dim', type=int, default=256)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--backbone_train_mode', type=str, default=None,
                        help='frozen|partial|full|indices')
    parser.add_argument('--freeze_indices', type=str, default='0,1,2,3,4,5')

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--mix_strategy', choices=('alternate', 'combined'), default='alternate')
    parser.add_argument('--unfreeze_epoch', type=int, default=16)
    parser.add_argument('--unfreeze_last_n', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_steps_per_epoch', type=int, default=0)

    parser.add_argument('--multi_gpu', choices=('none', 'dp', 'ddp', 'auto'), default='auto')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', type=_str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--out_weights', type=str, default=None)
    parser.add_argument('--out_checkpoint', type=str, default=None)
    parser.add_argument('--log', type=_str2bool, default=True)
    parser.add_argument('--log_path', type=str, default=None)

    parser.add_argument('--loss_mode', choices=('ce', 'supcon_ce'), default='ce')
    parser.add_argument('--supcon_weight', type=float, default=0.5)
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--supcon_temperature', type=float, default=0.07)
    parser.add_argument('--label_namespace', choices=('per_task', 'global'), default=None)
    parser.add_argument('--temperature_adapt', type=_str2bool, default=False)
    parser.add_argument('--aux_scratch_head', type=_str2bool, default=False)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--pretrain_variant', type=str, default=None)
    parser.add_argument('--supcon_min_class_fallback', type=_str2bool, default=False)
    parser.add_argument('--pretrain_sampler', type=str, default=None)
    parser.add_argument('--pretrain_sampler_apply', type=str, default='')
    parser.add_argument('--per_task_class_weight', type=str, default=None)
    parser.add_argument('--per_task_class_weight_apply', type=str, default='')
    parser.add_argument('--use_focal_for', type=str, default='')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--uncertainty_weighting', type=_str2bool, default=False)
    parser.add_argument('--uncertainty_warmup_epochs', type=int, default=0)
    parser.add_argument('--use_ldam_for', type=str, default='')
    parser.add_argument('--ldam_max_margin', type=float, default=0.3)
    parser.add_argument('--eval_metric', type=str, default='mean_acc')
    parser.add_argument('--val_split_strategy', type=str, default='random')
    parser.add_argument('--val_split_fold', type=int, default=0)
    parser.add_argument('--val_split_k', type=int, default=5)
    parser.add_argument('--active_tasks', type=str, default='')

    args = parser.parse_args()
    if '--label_namespace' in sys.argv:
        args._label_namespace_explicit = True
    else:
        args._label_namespace_explicit = False
    apply_pretrain_defaults(args)
    apply_pretrain_variant(args)

    if args.multi_gpu == 'auto':
        if 'RANK' in os.environ:
            args.multi_gpu = 'ddp'
        elif args.cuda and torch.cuda.device_count() > 1:
            args.multi_gpu = 'dp'
        else:
            args.multi_gpu = 'none'
    return args


# ─────────────────────────── helpers ───────────────────────────

def _parse_task_list(raw):
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple)):
        return set(str(x).strip() for x in raw if str(x).strip())
    return set(s.strip() for s in str(raw).split(',') if s.strip())


def _should_apply(task_name, apply_raw):
    appl = _parse_task_list(apply_raw)
    if not appl:
        return False
    return task_name in appl


def _make_class_balanced_sampler(dataset):
    if not hasattr(dataset, 'samples'):
        raise AttributeError('Dataset {} has no .samples attribute'.format(type(dataset).__name__))
    labels = [int(s[1]) for s in dataset.samples]
    class_count = {}
    for lab in labels:
        class_count[lab] = class_count.get(lab, 0) + 1
    num_classes = len(class_count)
    total = len(labels)
    weights = [1.0 / (num_classes * class_count[lab]) for lab in labels]
    return WeightedRandomSampler(weights=weights, num_samples=total, replacement=True)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _model_core(model):
    return model.module if isinstance(model, (torch.nn.DataParallel, DDP)) else model


def _parse_freeze_indices(args):
    raw = getattr(args, 'freeze_indices', None)
    if raw is None or raw == '':
        return [0, 1, 2, 3, 4, 5]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(x.strip()) for x in str(raw).split(',') if x.strip()]


def _resolve_class_weight(task_name, args, train_loaders):
    mode = getattr(args, 'per_task_class_weight', None)
    if not mode:
        return None
    if not _should_apply(task_name, getattr(args, 'per_task_class_weight_apply', '')):
        return None
    loader = train_loaders.get(task_name)
    if loader is None:
        return None
    ds = loader.dataset
    if not hasattr(ds, 'samples'):
        return None
    labels = [int(s[1]) for s in ds.samples]
    if not labels:
        return None
    class_to_idx = {}
    for lab in labels:
        class_to_idx[lab] = class_to_idx.get(lab, 0) + 1
    num_classes = max(class_to_idx.keys()) + 1
    counts = [class_to_idx.get(i, 0) for i in range(num_classes)]
    if mode == 'inverse':
        weights = [1.0 / max(c, 1) for c in counts]
    elif mode == 'inverse_sqrt':
        weights = [1.0 / (max(c, 1) ** 0.5) for c in counts]
    else:
        return None
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


def _resolve_backbone_mode(args, epoch):
    explicit = getattr(args, 'backbone_train_mode', None)
    if explicit:
        return str(explicit).strip().lower()
    if epoch > args.unfreeze_epoch:
        return 'partial'
    return 'frozen'


def _cls_counts_from_dataset(train_loaders, task_name):
    loader = train_loaders.get(task_name)
    if loader is None:
        return None
    ds = loader.dataset
    if not hasattr(ds, 'samples'):
        return None
    counts = {}
    for s in ds.samples:
        lab = int(s[1])
        counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return None
    n_cls = max(counts.keys()) + 1
    return [counts.get(i, 0) for i in range(n_cls)]


def _global_retrieval_top1(model, device, val_loaders, task_names, max_per_task=64):
    """Cross-task nearest-neighbor top-1 in SupCon projection space."""
    import torch.nn.functional as _F
    from common.cmpl_taxonomy import SCRATCH_CLASS_NAMES as _SCRATCH_NAMES
    core = _model_core(model)
    core.eval()
    if core.projection is None:
        return float('nan'), 0
    gallery_feats = []
    gallery_keys = []
    with torch.no_grad():
        for name in task_names:
            loader = val_loaders[name]
            count = 0
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    images, labels = batch[0], batch[1]
                    raw_class = None
                else:
                    images = batch['img']
                    raw_class = batch.get('raw_class', None)
                images = images.to(device, non_blocking=True)
                feats = core.encode(images)
                proj = core.project(feats)
                gallery_feats.append(proj.cpu())
                if raw_class is not None:
                    if isinstance(raw_class, torch.Tensor):
                        keys = raw_class.cpu().tolist()
                    else:
                        keys = list(raw_class)
                    gallery_keys.extend([(name, k) for k in keys])
                else:
                    gallery_keys.extend([(name, None)] * images.size(0))
                count += images.size(0)
                if count >= max_per_task:
                    break
    if not gallery_feats:
        return float('nan'), 0
    gallery = torch.cat(gallery_feats, dim=0)
    gallery = _F.normalize(gallery, dim=1)
    n = gallery.size(0)
    sim = torch.matmul(gallery, gallery.T)
    sim.fill_diagonal_(-float('inf'))
    nn_idx = sim.argmax(dim=1).tolist()
    correct = 0
    total = 0
    for i, j in enumerate(nn_idx):
        src_name, src_raw = gallery_keys[i]
        tgt_name, tgt_raw = gallery_keys[j]
        if src_name == tgt_name:
            continue
        if src_raw is None or tgt_raw is None:
            continue
        if src_raw == tgt_raw:
            correct += 1
        elif (src_raw in _SCRATCH_NAMES and tgt_raw in _SCRATCH_NAMES):
            correct += 1
        total += 1
    if total == 0:
        return float('nan'), 0
    return correct / total, total


# ─────────────────────────── distributed ───────────────────────

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model, args, local_rank, is_distributed):
    device = torch.device('cuda', local_rank) if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None)
        logging.info('Using DistributedDataParallel (world_size={}).'.format(dist.get_world_size()))
    elif args.multi_gpu == 'dp' and args.cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logging.info('Using DataParallel on {} GPUs.'.format(torch.cuda.device_count()))
    else:
        logging.info('Using single-device training.')
    return model, device


# ─────────────────────────── datasets ──────────────────────────

def _build_aerospace_datasets(args):
    data_dir = args.data_dir
    kw = {'imgsz': args.yolo_imgsz}
    return {
        'blade': (
            BladeSynthPretrainDataset(data_dir, split='train', augment=True, **kw),
            BladeSynthPretrainDataset(data_dir, split='val', augment=False, **kw),
        ),
        'aebad': (
            AeBADBinaryPretrainDataset(data_dir, split='train', augment=True, **kw),
            AeBADBinaryPretrainDataset(data_dir, split='val', augment=False, **kw),
        ),
    }


def _build_automotive_datasets(args):
    pretrain_root = os.path.join(args.data_dir, 'pretrain')
    split_kw = {
        'val_split_strategy': getattr(args, 'val_split_strategy', 'random'),
        'val_split_k': getattr(args, 'val_split_k', 5),
        'val_split_fold': getattr(args, 'val_split_fold', 0),
    }
    kw = {'pretrain_root': pretrain_root, 'imgsz': args.yolo_imgsz}
    kw.update(split_kw)
    builders = {
        'cardd': lambda: (CarDDPretrainDataset(split='train', augment=True, **kw),
                          CarDDPretrainDataset(split='val', augment=False, **kw)),
        'carparts': lambda: (CarPartsPretrainDataset(split='train', augment=True, **kw),
                             CarPartsPretrainDataset(split='val', augment=False, **kw)),
        'minor_dent': lambda: (MinorDentPretrainDataset(split='train', augment=True, **kw),
                               MinorDentPretrainDataset(split='val', augment=False, **kw)),
        'syndcar': lambda: (SyndcarPretrainDataset(split='train', augment=True, **kw),
                            SyndcarPretrainDataset(split='val', augment=False, **kw)),
        'neu_cls': lambda: (NEUPretrainDataset(split='train', augment=True, **kw),
                            NEUPretrainDataset(split='val', augment=False, **kw)),
    }
    task_names = [name for name, _n in args.task_specs]
    missing = [n for n in task_names if n not in builders]
    if missing:
        raise KeyError('Unknown automotive pretrain task(s): {}. v6 only supports: {}'.format(
            missing, sorted(builders.keys())))
    return {name: builders[name]() for name in task_names}


def build_loaders(args, is_distributed, rank, world_size):
    track = getattr(args, 'track', 'aerospace')
    datasets = _build_automotive_datasets(args) if track == 'automotive' else _build_aerospace_datasets(args)

    task_names = [name for name, _n in args.task_specs]
    train_loaders = {}
    val_loaders = {}
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': args.cuda and torch.cuda.is_available(),
    }

    for name in task_names:
        train_ds, val_ds = datasets[name]
        sampler = None
        if is_distributed:
            sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        apply_sampler = _should_apply(name, getattr(args, 'pretrain_sampler_apply', ''))
        if getattr(args, 'pretrain_sampler', None) == 'class_balanced' and apply_sampler:
            try:
                sampler = _make_class_balanced_sampler(train_ds)
            except Exception as exc:
                logging.warning('class_balanced sampler failed for task={} ({}); falling back'.format(name, exc))
                sampler = None
        train_loaders[name] = DataLoader(
            train_ds, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
        val_loaders[name] = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loaders, val_loaders


# ─────────────────────────── optim ─────────────────────────────

def build_optimizer(model, args, backbone_mode):
    core = _model_core(model)
    head_params = []
    for head in core.heads.values():
        head_params.extend(list(head.parameters()))
    if core.projection is not None:
        head_params.extend(list(core.projection.parameters()))
    param_groups = []
    if backbone_mode == 'frozen':
        param_groups.append({'params': head_params, 'lr': args.lr_head})
        param_groups.append({'params': list(core.encoder.proj.parameters()), 'lr': args.lr_head})
    else:
        backbone_params = [p for p in core.encoder.det_model.parameters() if p.requires_grad]
        proj_params = list(core.encoder.proj.parameters())
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone})
        param_groups.append({'params': head_params + proj_params, 'lr': args.lr_head})
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def validate(model, device, val_loaders, task_names, task_weights):
    model.eval()
    accs = {}
    with torch.no_grad():
        for name in task_names:
            correct = total = 0
            for batch in val_loaders[name]:
                if isinstance(batch, (tuple, list)):
                    images, labels = batch[0], batch[1]
                else:
                    images = batch['img']
                    labels = batch['label']
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = _model_core(model).forward_task(name, images)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.numel()
            accs[name] = correct / max(total, 1)
    score = sum(task_weights.get(name, 1.0 / len(task_names)) * accs[name] for name in task_names)
    return accs, score


# ─────────────────────────── main train ────────────────────────

def train(args):
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main = rank == 0
    task_names = [name for name, _n in args.task_specs]
    task_weights = getattr(args, 'task_weights', {name: 1.0 / len(task_names) for name in task_names})

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    # v9 active_tasks filter
    active_raw = getattr(args, 'active_tasks', '') or ''
    if active_raw.strip():
        wanted = _parse_task_list(active_raw)
        spec_map = {n: (n, c) for n, c in args.task_specs}
        unknown = [n for n in wanted if n not in spec_map]
        if unknown:
            raise KeyError('Unknown active_tasks: {} (known: {})'.format(unknown, list(spec_map.keys())))
        new_specs = [spec_map[n] for n in wanted]
        args.task_specs = new_specs
        w_map = {n: w for n, w in task_weights.items()}
        active_w = [w_map.get(n, 1.0 / len(new_specs)) for n in wanted]
        s = sum(active_w) or 1.0
        task_weights = {n: w / s for n, w in zip(wanted, active_w)}
        task_names = list(wanted)
        if is_main:
            logging.info('v9 active_tasks filter: {} (renormalized weights={})'.format(
                list(wanted), task_weights))

    if is_main:
        logging.info('Phase 0 pretrain track={} tasks={} loss_mode={} feat_dim={}'.format(
            args.track, task_names, getattr(args, 'loss_mode', 'ce'),
            getattr(args, 'feat_dim', 256)))
        logging.info('Data dir: {}'.format(os.path.abspath(args.data_dir)))
        if getattr(args, 'pretrain_sampler', None):
            logging.info('v7 sampler: strategy={}, apply={}'.format(
                args.pretrain_sampler, getattr(args, 'pretrain_sampler_apply', '')))
        if getattr(args, 'per_task_class_weight', None):
            logging.info('v7 class_weight: mode={}, apply={}'.format(
                args.per_task_class_weight, getattr(args, 'per_task_class_weight_apply', '')))

    model = PretrainMultitaskModel(
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
        feat_dim=args.feat_dim,
        task_specs=args.task_specs,
        loss_mode=getattr(args, 'loss_mode', 'ce'),
        proj_dim=getattr(args, 'proj_dim', 64),
        supcon_temperature=getattr(args, 'supcon_temperature', 0.07),
        label_namespace=getattr(args, 'label_namespace', 'per_task'),
        temperature_adapt=getattr(args, 'temperature_adapt', False),
        aux_scratch_head=getattr(args, 'aux_scratch_head', False),
        supcon_min_class_fallback=getattr(args, 'supcon_min_class_fallback', False),
    )
    model, device = wrap_model(model, args, local_rank, is_distributed)

    train_loaders, val_loaders = build_loaders(args, is_distributed, rank, world_size)
    train_iters = {name: _cycle(train_loaders[name]) for name in task_names}
    _per_task_class_weight_raw = {name: _resolve_class_weight(name, args, train_loaders)
                                  for name in task_names}
    _per_task_class_weight = {
        name: w.to(device) if w is not None else None
        for name, w in _per_task_class_weight_raw.items()
    }
    _ldam_cls_counts = {name: _cls_counts_from_dataset(train_loaders, name)
                        for name in task_names}

    steps_per_epoch = max(len(loader) for loader in train_loaders.values())
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    best_score = -1.0
    bad_epochs = 0
    optimizer = None
    scheduler = None
    active_mode = None
    freeze_indices = _parse_freeze_indices(args)
    label_smoothing = float(getattr(args, 'label_smoothing', 0.0) or 0.0)

    use_uncertainty = getattr(args, 'uncertainty_weighting', False)
    warmup_epochs = int(getattr(args, 'uncertainty_warmup_epochs', 0) or 0)
    focal_tasks = _parse_task_list(getattr(args, 'use_focal_for', ''))
    focal_gamma = float(getattr(args, 'focal_gamma', 2.0))
    ldam_tasks = _parse_task_list(getattr(args, 'use_ldam_for', ''))
    ldam_max_margin = float(getattr(args, 'ldam_max_margin', 0.3))
    if use_uncertainty:
        log_sigmas = {name: nn.Parameter(torch.full((1,), -0.693, device=device))
                      for name in task_names}
        if is_main:
            logging.info('v8 uncertainty_weighting=True, warmup_epochs={}, tasks={}'.format(
                warmup_epochs, list(task_names)))

    def _task_loss_parts(core, name, images, labels, supcon_w, ce_w):
        use_focal = name in focal_tasks
        use_ldam = name in ldam_tasks
        return core.compute_task_loss(
            name, images, labels,
            supcon_weight=supcon_w, ce_weight=ce_w,
            label_smoothing=label_smoothing,
            class_weight=_per_task_class_weight.get(name, None),
            use_focal=use_focal, focal_gamma=focal_gamma,
            use_ldam=use_ldam, ldam_cls_counts=_ldam_cls_counts.get(name, None),
            ldam_max_margin=ldam_max_margin)

    for epoch in range(1, args.epochs + 1):
        backbone_mode = _resolve_backbone_mode(args, epoch)

        if is_distributed:
            for loader in train_loaders.values():
                loader.sampler.set_epoch(epoch)

        _model_core(model).configure_backbone(
            backbone_mode,
            args.unfreeze_last_n if backbone_mode == 'partial' else 0,
            freeze_indices=freeze_indices if backbone_mode == 'indices' else None,
        )
        if backbone_mode != active_mode:
            optimizer = build_optimizer(model, args, backbone_mode)
            if use_uncertainty:
                sigma_opt = torch.optim.AdamW(
                    [log_sigmas[name] for name in task_names],
                    lr=1e-3, weight_decay=0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(steps_per_epoch * args.epochs, 1))
            active_mode = backbone_mode

        model.train()
        if backbone_mode == 'frozen':
            _model_core(model).encoder.det_model.eval()

        sum_loss = 0.0
        sum_task = {name: 0.0 for name in task_names}
        t0 = time.time()

        for step in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            core = _model_core(model)
            supcon_w = getattr(args, 'supcon_weight', 0.5)
            ce_w = getattr(args, 'ce_weight', 0.5)
            mix_strategy = getattr(args, 'mix_strategy', 'alternate')
            label_ns = getattr(args, 'label_namespace', 'per_task')

            if mix_strategy == 'combined':
                agg_feats = []
                agg_labels = []
                agg_is_scratch = []
                agg_ce_list = []
                batches = []
                for name in task_names:
                    batch = next(train_iters[name])
                    batches.append(batch)
                offset = core._resolve_label_offsets()
                for name, batch in zip(task_names, batches):
                    if isinstance(batch, (tuple, list)):
                        images, labels = batch[0], batch[1]
                        is_scratch = None
                    else:
                        images = batch['img']
                        labels = batch['label']
                        is_scratch = batch.get('is_scratch', None)
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    feats = core.encode(images)
                    logits = core.heads[name](feats)
                    if name in focal_tasks:
                        from common.pretrain_model import FocalLoss
                        class_w = _per_task_class_weight.get(name, None)
                        ce_loss = FocalLoss(gamma=focal_gamma, alpha=class_w)(logits, labels)
                    elif name in ldam_tasks:
                        from common.pretrain_model import LDAMLoss
                        class_w = _per_task_class_weight.get(name, None)
                        ce_loss = LDAMLoss(cls_counts=_ldam_cls_counts.get(name, None),
                                           max_margin=ldam_max_margin, weight=class_w)(logits, labels)
                    else:
                        ce_loss = F.cross_entropy(
                            logits, labels, label_smoothing=label_smoothing,
                            weight=_per_task_class_weight.get(name, None))
                    agg_ce_list.append(ce_loss)
                    if label_ns == 'global':
                        agg_feats.append(feats)
                        agg_labels.append(labels + offset[name])
                        if is_scratch is not None:
                            agg_is_scratch.append(is_scratch.to(device, non_blocking=True))
                for name, ce_item in zip(task_names, agg_ce_list):
                    sum_task[name] = sum_task.get(name, 0.0) + ce_item.item()
                if use_uncertainty and epoch > warmup_epochs:
                    agg_task_loss = torch.zeros_like(agg_ce_list[0])
                    for i, name in enumerate(task_names):
                        sigma = log_sigmas[name].exp()
                        agg_task_loss = agg_task_loss + agg_ce_list[i] / (2 * sigma ** 2) + sigma.log()
                    agg_ce_total = agg_task_loss
                else:
                    agg_ce_total = torch.stack(agg_ce_list).sum()
                if label_ns == 'global' and len(agg_feats) > 0 and core.projection is not None:
                    feats_cat = torch.cat(agg_feats, dim=0)
                    labels_cat = torch.cat(agg_labels, dim=0)
                    proj_cat = core.project(feats_cat)
                    supcon = core.supcon(proj_cat, labels_cat)
                    loss_value = ce_w * agg_ce_total / max(len(task_names), 1) + supcon_w * supcon
                    if core.aux_scratch_head and len(agg_is_scratch) > 0:
                        is_scratch_cat = torch.cat(agg_is_scratch)
                        scratch_loss = core.scratch_binary_loss(proj_cat, is_scratch_cat)
                        loss_value = loss_value + 0.1 * scratch_loss
                        sum_task['__scratch_aux__'] = sum_task.get('__scratch_aux__', 0.0) + scratch_loss.item()
                else:
                    supcon_total = agg_ce_total * 0.0
                    for name in task_names:
                        batch = next(train_iters[name])
                        if isinstance(batch, (tuple, list)):
                            images, labels = batch[0], batch[1]
                        else:
                            images = batch['img']
                            labels = batch['label']
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        if core.projection is not None and images.size(0) >= 2:
                            proj = core.project(core.encode(images))
                            supcon_total = supcon_total + core.supcon(proj, labels)
                    loss_value = ce_w * agg_ce_total / max(len(task_names), 1) + supcon_w * supcon_total / max(len(task_names), 1)
                loss = loss_value
                loss.backward()
                sum_task['__combined__'] = sum_task.get('__combined__', 0.0) + loss.item()
                loss_value = loss.item()
            else:
                name = task_names[step % len(task_names)]
                batch = next(train_iters[name])
                if isinstance(batch, (tuple, list)):
                    images, labels = batch[0], batch[1]
                    is_scratch = None
                else:
                    images = batch['img']
                    labels = batch['label']
                    is_scratch = batch.get('is_scratch', None)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                task_total, task_ce = _task_loss_parts(
                    core, name, images, labels, supcon_w, ce_w)
                weight = task_weights.get(name, 1.0)
                if core.aux_scratch_head and is_scratch is not None and core.projection is not None:
                    feats = core.encode(images)
                    proj = core.project(feats)
                    scratch_loss = core.scratch_binary_loss(proj, is_scratch.to(device, non_blocking=True))
                    task_total = task_total + 0.1 * scratch_loss
                    sum_task['__scratch_aux__'] = sum_task.get('__scratch_aux__', 0.0) + scratch_loss.item()
                if use_uncertainty and epoch > warmup_epochs:
                    sigma = log_sigmas[name].exp()
                    task_loss = task_total / (2 * sigma ** 2) + sigma.log()
                else:
                    task_loss = weight * task_total
                loss = task_loss
                loss.backward()
                for other in task_names:
                    if other != name:
                        sum_task[other] += 0.0
                sum_task[name] += task_total.item()
                loss_value = loss.item()

            torch.nn.utils.clip_grad_norm_(_model_core(model).parameters(), args.grad_clip)
            optimizer.step()
            if use_uncertainty and epoch > warmup_epochs:
                sigma_opt.zero_grad(set_to_none=True)
                sigma_opt.step()
            scheduler.step()
            sum_loss += loss_value

        eval_every = getattr(args, 'eval_every', 1)
        eval_metric = getattr(args, 'eval_metric', 'mean_acc')
        if eval_every <= 0 or epoch % eval_every == 0 or epoch == args.epochs:
            accs, score = validate(model, device, val_loaders, task_names, task_weights)
            retrieval_diag = None
            if eval_metric == 'global_retrieval_top1':
                r_score, r_n = _global_retrieval_top1(
                    model, device, val_loaders, task_names)
                if not (r_score != r_score):
                    retrieval_diag = (r_score, r_n)
        else:
            accs = {name: float('nan') for name in task_names}
            score = float('nan')
            retrieval_diag = None
        elapsed = time.time() - t0

        if is_main:
            sigma_str = ''
            if use_uncertainty:
                sigma_str = ' sigmas=' + ','.join(
                    '{}={:.3f}'.format(n, log_sigmas[n].exp().item()) for n in task_names)
            diag_str = ''
            if retrieval_diag is not None:
                r_score, r_n = retrieval_diag
                diag_str = ' retrieval={:.3f}(n={})'.format(r_score, r_n)
            acc_str = ' '.join('{}={:.3f}'.format(name, accs[name]) for name in task_names)
            train_str = ' '.join(
                '{}={:.4f}'.format(name, sum_task[name] / max(steps_per_epoch, 1))
                for name in task_names)
            logging.info(
                'Epoch {}/{} mode={} loss={:.4f} train[{}] val[{}] score={:.3f}{}{}'.format(
                    epoch, args.epochs, backbone_mode,
                    sum_loss / max(steps_per_epoch, 1),
                    train_str, acc_str, score, diag_str, sigma_str))

            if score > best_score:
                best_score = score
                bad_epochs = 0
                export_pretrained_yolo(
                    _model_core(model), args.yolo_weights, args.out_weights)
                ckpt = {
                    'epoch': epoch,
                    'score': score,
                    'track': args.track,
                    'loss_mode': getattr(args, 'loss_mode', 'ce'),
                    'feat_dim': args.feat_dim,
                    'pretrain_variant': getattr(args, 'pretrain_variant', None),
                    'label_namespace': label_ns,
                    'temperature_adapt': getattr(args, 'temperature_adapt', False),
                    'backbone_train_mode': backbone_mode,
                    'freeze_indices': freeze_indices if backbone_mode == 'indices' else None,
                    'model_state': _model_core(model).state_dict(),
                }
                for name in task_names:
                    ckpt['val_{}'.format(name)] = accs[name]
                if use_uncertainty:
                    for name in task_names:
                        ckpt['sigma_{}'.format(name)] = float(log_sigmas[name].exp().item())
                torch.save(ckpt, args.out_checkpoint)
                logging.info('New best score {:.3f}; saved {}.'.format(
                    score, args.out_weights))
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    logging.info('Early stop at epoch {} (patience={}).'.format(
                        epoch, args.patience))
                    break

    cleanup_distributed(is_distributed)
    if is_main:
        logging.info('Phase 0 done. Best score={:.3f}. Weights: {}'.format(
            best_score, args.out_weights))


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

    train(args)
