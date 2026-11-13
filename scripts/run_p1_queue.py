"""Sequential P1 queue. Dry-run is read-only; never overwrite existing runs.

Legacy: --config config/cspd/curriculum_v3.yaml
Default suite: B, seed 0, folds NEU/CR7/GC10 = 3 jobs.
  python scripts/run_p1_queue.py --suite --run-id hp_abc_20261111
A/C or seeds 1/2 are opt-in only: --experiments A B C and/or --seeds 0 1 2
--skip-completed skips ONLY matching run_complete.json, never partial weights.
"""
import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/cspd/curriculum_v3.yaml'
JOBS = [('NEU', 0, 'NEU'), ('CR7', 0, 'CR7'), ('GC10', 0, 'GC10')]


def local_path(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('all configuration/output paths must stay under project root')
    return path


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value or ''):
        raise ValueError('run/group/experiment names require ASCII letters, digits, _ or -')
    return value


def build_jobs(args):
    if args.suite:
        if not args.run_id:
            raise ValueError('--suite requires a fresh --run-id')
        spec = yaml.safe_load(local_path(args.config or 'config/cspd/hardpair_abc.yaml').read_text(encoding='utf-8'))
        base = yaml.safe_load(local_path(spec['base_config']).read_text(encoding='utf-8'))
        base.update(spec['defaults'])
        group = safe_name(args.run_id)
        result = []
        default_experiments = list(spec.get('default_experiments') or ['B'])
        default_seeds = list(spec.get('default_seeds') or [0])
        for variant in args.experiments or default_experiments:
            settings = dict(spec['variants'][variant])
            variant_seeds = list(settings.pop('seeds', default_seeds))
            if args.seeds is not None:
                seeds = args.seeds
            elif args.experiments:
                seeds = variant_seeds
            else:
                seeds = default_seeds
            for seed in seeds:
                for fold in args.folds or spec['folds']:
                    cfg = copy.deepcopy(base)
                    cfg.update(settings)
                    name = '{}_{}_s{}'.format(variant, fold, seed)
                    cfg.update(output_group=group, exp_name=name, seed=int(seed),
                               holdout_source=fold, experiment_variant=variant)
                    result.append((name, cfg))
        return result
    if args.experiments or args.seeds or args.folds:
        raise ValueError('variant/seed/fold selection requires --suite')
    base = yaml.safe_load(local_path(args.config or DEFAULT_CONFIG).read_text(encoding='utf-8'))
    group = safe_name(args.run_id or str(base.get('output_group') or ''))
    result = []
    for fold, seed, name in JOBS:
        cfg = copy.deepcopy(base)
        cfg.update(output_group=group, exp_name=name, seed=seed, holdout_source=fold)
        result.append((name, cfg))
    return result


def plan_jobs(jobs, skip_completed=False):
    plan = []
    seen = set()
    for name, cfg in jobs:
        group = safe_name(cfg['output_group'])
        safe_name(name)
        out = local_path(ROOT / 'output/cspd' / group / name)
        if out in seen:
            raise ValueError('duplicate queue job: ' + name)
        seen.add(out)
        config_path = local_path(ROOT / 'config/cspd/queues' / group / (name + '.yaml'))
        status = 'run'
        if out.exists():
            marker = out / 'run_complete.json'
            if not skip_completed or not marker.is_file():
                raise FileExistsError('existing/partial run protected: {}; choose a new run-id'.format(out))
            done = json.loads(marker.read_text(encoding='utf-8'))
            if done.get('status') != 'complete' or done.get('config') != cfg:
                raise ValueError('completed config mismatch: ' + name)
            status = 'skip-completed'
        if config_path.exists():
            saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            if saved != cfg:
                raise ValueError('queue config collision: ' + str(config_path))
        plan.append((name, cfg, config_path, status))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--suite', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--experiments', nargs='+', choices=['A', 'B', 'C'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--folds', nargs='+', choices=['NEU', 'CR7', 'GC10'])
    parser.add_argument('--skip-completed', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan = plan_jobs(build_jobs(args), args.skip_completed)  # preflight ALL before launching any
    pending = sum(status == 'run' for _, _, _, status in plan)
    print('jobs={} pending={} completed={}'.format(len(plan), pending, len(plan)-pending))
    for name, cfg, config_path, status in plan:
        print('{} {} fold={} seed={} layer={} hard={} -> output/cspd/{}/{}'.format(
            status, name, cfg['holdout_source'], cfg['seed'], cfg.get('feature_layer', 3),
            cfg.get('same_hard_pairs', False), cfg['output_group'], name), flush=True)
        if args.dry_run or status != 'run':
            continue
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            with config_path.open('x', encoding='utf-8') as handle:
                yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
        completed = subprocess.run(
            [sys.executable, '-B', str(ROOT / 'scripts/train_cspd.py'), '--config', str(config_path)],
            cwd=str(ROOT))
        if completed.returncode:
            raise SystemExit('FAILED {} code {}; partial output protected, use a fresh run-id'.format(
                name, completed.returncode))
    print('dry-run: no files written, no training started' if args.dry_run else 'queue done')


if __name__ == '__main__':
    main()
