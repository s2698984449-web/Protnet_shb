#!/usr/bin/env python
"""
Unified experiment runner for aerospace and automotive tracks.

Usage (run from project root D:\元强化学习\代码\):
  python scripts/run_experiment.py --list
  python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp08_protonet
  python scripts/run_experiment.py --config config/automotive/defaults.yaml --exp cmpl_p0_v9 --stage pretrain
  python scripts/run_experiment.py --config config/aerospace/defaults.yaml --grid exp04_fastlr
  python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp08_protonet --set iters=20
"""

from __future__ import print_function

import argparse
import copy
import os
import shlex
import subprocess
import sys


def _load_yaml(path):
    try:
        import yaml
    except ImportError:
        print('PyYAML required: pip install pyyaml', file=sys.stderr)
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key == 'extends':
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _resolve_extends(name, registry, cache=None):
    cache = cache if cache is not None else {}
    if name in cache:
        return copy.deepcopy(cache[name])
    if name not in registry:
        raise KeyError('Unknown extends/base: {}'.format(name))
    node = copy.deepcopy(registry[name])
    parent = node.pop('extends', None)
    if parent:
        resolved = _resolve_extends(parent, registry, cache)
        node = _deep_merge(resolved, node)
    cache[name] = node
    return copy.deepcopy(node)


def _resolve_experiment(cfg, exp_name, _stack=None):
    exps = cfg.get('experiments', {})
    if exp_name not in exps:
        raise KeyError('Unknown experiment: {}'.format(exp_name))
    stack = list(_stack or [])
    if exp_name in stack:
        raise KeyError('Circular extends: {} -> {}'.format(' -> '.join(stack + [exp_name]), exp_name))
    bases = cfg.get('bases', {})
    node = copy.deepcopy(exps[exp_name])
    extends = node.pop('extends', None)
    merged = copy.deepcopy(cfg.get('defaults', {}))
    if extends:
        if extends in exps and extends != exp_name:
            parent = _resolve_experiment(cfg, extends, stack + [exp_name])
            parent.pop('name', None)
            merged = _deep_merge(merged, parent)
        elif extends in bases:
            merged = _deep_merge(merged, _resolve_extends(extends, bases))
        else:
            raise KeyError('Unknown extends/base: {}'.format(extends))
    merged = _deep_merge(merged, node)
    merged['name'] = exp_name
    if 'description' in exps[exp_name]:
        merged['description'] = exps[exp_name]['description']
    return merged


def _resolve_grid_item(cfg, grid_name, item):
    grid = cfg['grids'][grid_name]
    merged = {}
    if 'defaults' in cfg:
        merged.update(copy.deepcopy(cfg['defaults']))
    extends = grid.get('extends')
    if extends:
        merged = _deep_merge(merged, _resolve_extends(extends, cfg.get('bases', {})))
    grid_copy = {k: v for k, v in grid.items()
                 if k not in ('description', 'items', 'extends')}
    merged = _deep_merge(merged, grid_copy)
    if 'exp' in item:
        return _resolve_experiment(cfg, item['exp'])
    merged = _deep_merge(merged, item)
    merged.setdefault('stage', 'train')
    merged.setdefault('name', item.get('exp_name', grid_name))
    return merged


def _read_exclude_file(path, project_root):
    full = path if os.path.isabs(path) else os.path.join(project_root, path)
    if not os.path.isfile(full):
        raise FileNotFoundError('exclude list not found: {}'.format(full))
    ids = []
    with open(full, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(line)
    return ','.join(ids)


TRAIN_ARGS = {
    'track', 'ways', 'shots', 'meta_lr', 'backbone_meta_lr', 'head_meta_lr', 'fast_lr',
    'adapt_steps', 'meta_batch_size', 'iters', 'first_order', 'grad_clip', 'embed_dim',
    'yolo_weights', 'yolo_imgsz', 'rgb_only', 'freeze_backbone', 'eval_tasks', 'eval_every',
    'stage1_end', 'stage2_end', 'w_rl_max', 'cuda', 'seed', 'data_dir_path', 'dataset_name',
    'anil', 'train_domains', 'test_domain', 'train_task_num', 'test_task_num', 'plot',
    'plot_path', 'plot_step', 'log', 'log_path', 'checkpoint', 'checkpoint_path',
    'checkpoint_step', 'exp_name', 'episode_aug', 'aug_illumination', 'input_scales',
    'cross_domain_episodes', 'eval_patience', 'early_stop_min_delta', 'meta_algorithm',
    'protonet_distance', 'protonet_prototype_type', 'rgb_cbam',
}

EVAL_ARGS = {
    'track', 'checkpoint', 'data_dir_path', 'dataset_name', 'test_domain', 'train_domains',
    'anil', 'train_task_num', 'test_task_num', 'ways', 'shots', 'adapt_steps', 'fast_lr',
    'first_order', 'eval_tasks', 'seeds', 'freeze_backbone', 'rgb_only', 'yolo_weights',
    'yolo_imgsz', 'embed_dim', 'meta_algorithm', 'protonet_distance', 'protonet_prototype_type',
    'cuda',
    'exclude_sample_ids', 'out_json', 'rgb_cbam', 'tta_scales',
}

DIAGNOSE_ARGS = {
    'checkpoint', 'data_dir_path', 'dataset_name', 'test_domain', 'train_domains', 'train_task_num',
    'test_task_num', 'ways', 'shots', 'protonet_distance', 'protonet_prototype_type', 'eval_tasks', 'seeds',
    'hard_threshold', 'freeze_backbone', 'rgb_only', 'yolo_weights', 'yolo_imgsz',
    'embed_dim', 'cuda', 'out_dir', 'out_name', 'save_tasks', 'rgb_cbam',
}

PRETRAIN_ARGS = {
    'track', 'data_dir', 'yolo_weights', 'yolo_imgsz', 'feat_dim', 'epochs', 'batch_size',
    'lr_head', 'lr_backbone', 'weight_decay', 'grad_clip', 'mix_strategy', 'unfreeze_epoch',
    'unfreeze_last_n', 'patience', 'max_steps_per_epoch', 'multi_gpu', 'num_workers', 'cuda',
    'seed', 'out_weights', 'out_checkpoint', 'log', 'log_path',
    'loss_mode', 'supcon_weight', 'ce_weight', 'proj_dim', 'supcon_temperature',
    'pretrain_variant', 'label_smoothing', 'backbone_train_mode', 'freeze_indices',
    'task_specs', 'label_namespace', 'temperature_adapt', 'eval_every',
    # v7 additions (all default off, aerospace unaffected)
    'supcon_min_class_fallback', 'pretrain_sampler', 'pretrain_sampler_apply',
    'per_task_class_weight', 'per_task_class_weight_apply',
    # v8 additions
    'use_focal_for', 'focal_gamma', 'uncertainty_weighting', 'label_smoothing',
    # v9 additions (all default off / no-op for legacy configs)
    'uncertainty_warmup_epochs', 'use_ldam_for', 'ldam_max_margin',
    'eval_metric', 'val_split_strategy', 'val_split_k', 'val_split_fold',
    'active_tasks',
}


def _allowed_keys(stage):
    if stage == 'train':
        return TRAIN_ARGS
    if stage == 'eval':
        return EVAL_ARGS
    if stage == 'diagnose':
        return DIAGNOSE_ARGS
    if stage == 'pretrain':
        return PRETRAIN_ARGS
    raise ValueError('Unknown stage: {}'.format(stage))


def _format_arg(val):
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (list, tuple)):
        return ','.join(str(x) for x in val)
    return str(val)


def _build_argv(exp, project_root, overrides=None):
    cfg = copy.deepcopy(exp)
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    stage = cfg.get('stage', 'train')
    exclude_from = cfg.pop('exclude_sample_ids_from', None)
    if exclude_from:
        cfg['exclude_sample_ids'] = _read_exclude_file(exclude_from, project_root)

    reserved = {
        'name', 'description', 'stage', 'extends', 'script', 'python', 'cwd',
        'exclude_sample_ids_from',
    }

    # Apply track-aware defaults (fills checkpoint_path, log_path, plot_path, etc.)
    # Only for train/eval stages; pretrain has its own defaults in apply_pretrain_defaults.
    if stage in ('train', 'eval'):
        try:
            from common.project_paths import apply_meta_train_defaults
            class _Args:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            args_obj = _Args(cfg)
            apply_meta_train_defaults(args_obj)
            for k in ('checkpoint_path', 'log_path', 'plot_path', 'yolo_weights',
                      'data_dir_path', 'dataset_name'):
                v = getattr(args_obj, k, None)
                if v:
                    cfg[k] = v
        except Exception:
            pass  # fallback: let argv args handle None (will use script defaults)
    if stage == 'train':
        script = os.path.join('scripts', cfg.get('script', 'train.py'))
    elif stage == 'eval':
        script = os.path.join('scripts', cfg.get('script', 'eval.py'))
    elif stage == 'diagnose':
        script = os.path.join('scripts', cfg.get('script', 'diagnose_hard_tasks.py'))
    elif stage == 'pretrain':
        script = os.path.join('scripts', cfg.get('script', 'pretrain.py'))
    else:
        raise ValueError('Unknown stage: {}'.format(stage))

    argv = [script]
    allowed = _allowed_keys(stage)
    for key, val in sorted(cfg.items()):
        if key in reserved or key not in allowed or val is None:
            continue
        argv.extend(['--' + key, _format_arg(val)])
    return argv


def _find_python(explicit):
    if explicit and explicit != 'auto':
        return explicit
    candidates = [
        r'D:\miniconda3\envs\MAML\python.exe',
        os.environ.get('CONDA_PREFIX', '') + os.sep + 'python.exe'
        if os.environ.get('CONDA_PREFIX') else '',
        sys.executable,
        'python',
    ]
    for c in candidates:
        if c and (c == 'python' or os.path.isfile(c)):
            return c
    return 'python'


def _run(argv, python, cwd, dry_run=False):
    cmd = [python] + argv
    printable = ' '.join(shlex.quote(c) for c in cmd)
    print('=== cwd: {} ==='.format(cwd))
    print('=== cmd: {} ==='.format(printable))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def _parse_overrides(pairs):
    out = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError('--set expects key=value, got: {}'.format(pair))
        key, val = pair.split('=', 1)
        if val.lower() in ('true', 'false'):
            out[key] = val.lower() == 'true'
        else:
            try:
                if '.' in val:
                    out[key] = float(val)
                else:
                    out[key] = int(val)
            except ValueError:
                out[key] = val
    return out


def cmd_list(cfg):
    print('Experiments:')
    for name, spec in sorted(cfg.get('experiments', {}).items()):
        desc = spec.get('description', '')
        stage = spec.get('stage', 'train')
        print('  {:28} [{:9}] {}'.format(name, stage, desc))
    print('\nGrids:')
    for name, spec in sorted(cfg.get('grids', {}).items()):
        desc = spec.get('description', '')
        n = len(spec.get('items', []))
        print('  {:28} ({:d} runs) {}'.format(name, n, desc))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    default_config = os.path.join(project_root, 'config', 'aerospace', 'defaults.yaml')

    parser = argparse.ArgumentParser(description='Run experiments from YAML registry (aerospace or automotive)')
    parser.add_argument('--config', default=default_config,
                        help='Recipe YAML (default: config/aerospace/defaults.yaml; '
                             'automotive: config/automotive/defaults.yaml)')
    parser.add_argument('--list', action='store_true', help='List experiments and grids')
    parser.add_argument('--exp', default='', help='Experiment name from registry')
    parser.add_argument('--grid', default='', help='Run a grid of experiments sequentially')
    parser.add_argument('--stage', default='',
                        help='Override stage: train|eval|diagnose (default: from YAML)')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        help='Override arg, e.g. --set iters=20 --set fast_lr=0.01')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    if args.list:
        cmd_list(cfg)
        return 0

    overrides = _parse_overrides(args.overrides)
    python = _find_python(cfg.get('defaults', {}).get('python', 'auto'))
    cwd = project_root

    if args.grid:
        if args.grid not in cfg.get('grids', {}):
            print('Unknown grid: {}'.format(args.grid), file=sys.stderr)
            return 1
        items = cfg['grids'][args.grid]['items']
        print('=== Grid: {} ({} runs) ==='.format(args.grid, len(items)))
        for i, item in enumerate(items, 1):
            exp = _resolve_grid_item(cfg, args.grid, item)
            if args.stage:
                exp['stage'] = args.stage
            print('\n--- [{}/{}] {} ---'.format(i, len(items), exp.get('exp_name', exp.get('name'))))
            argv = _build_argv(exp, project_root, overrides)
            rc = _run(argv, python, cwd, dry_run=args.dry_run)
            if rc != 0:
                return rc
        print('\n=== Grid {} finished ==='.format(args.grid))
        return 0

    if not args.exp:
        parser.print_help()
        return 1

    exp = _resolve_experiment(cfg, args.exp)
    if args.stage:
        exp['stage'] = args.stage
    elif args.exp.endswith('_eval') or args.exp.endswith('_eval_exclude22') or args.exp.endswith('_eval_qc_deleted'):
        pass  # keep YAML stage
    elif args.exp.endswith('_eval_as_protonet'):
        pass

    desc = exp.get('description', '')
    print('=== Experiment: {} | stage={} ==='.format(args.exp, exp.get('stage')))
    if desc:
        print(desc)

    argv = _build_argv(exp, project_root, overrides)
    rc = _run(argv, python, cwd, dry_run=args.dry_run)
    if rc == 0:
        print('=== Done: {} ==='.format(args.exp))
    return rc


if __name__ == '__main__':
    sys.exit(main())
