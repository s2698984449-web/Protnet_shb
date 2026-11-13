"""Read expected completed suite metrics. Default: B seed 0, folds NEU/CR7/GC10. Print JSON; no files are written.

Per-pair recall = mean of the two per-class hit/query recalls (not accuracy
conditioned on only those two predictions). Directed confusion rate uses query
opportunities where BOTH classes were in the same episode. Zero support => null.
MDE is read from pre-run config; never inferred from observed outcomes.
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    'NEU': [('inclusion', 'pit_texture')],
    'CR7': [('oil', 'inclusion'), ('linear', 'pit_local'), ('pit_local', 'inclusion')],
    'GC10': [('oil', 'inclusion'), ('oil', 'water'), ('punch', 'weld_line')],
}


def recall(metrics, name):
    hit, count = metrics.get('per_class', {}).get(name, [0, 0])
    return hit / count if count else None


def pair_report(bucket, pair):
    a, b = pair
    ra, rb = recall(bucket, a), recall(bucket, b)
    rates = []
    for left, right in ((a, b), (b, a)):
        key = left + ' -> ' + right
        n = bucket.get('opportunities', {}).get(key, 0)
        err = bucket.get('confusion', {}).get(key, 0)
        rates.append({'direction': key, 'errors': err, 'opportunities': n,
                      'rate': err / n if n else None})
    return {'pair': pair, 'macro_recall': (ra + rb) / 2 if ra is not None and rb is not None else None,
            'class_support': {c: bucket.get('per_class', {}).get(c, [0, 0])[1] for c in pair},
            'directed': rates}


DEFAULT_PLAN = [('B', [0])]


def summarize(group, plan=None):
    if not group or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-' for c in group):
        raise ValueError('invalid run-id')
    plan = list(plan or DEFAULT_PLAN)
    runs, missing = {}, []
    for variant, seeds in plan:
        for seed in seeds:
            for fold in PAIRS:
                name = '{}_{}_s{}'.format(variant, fold, seed)
                directory = ROOT / 'output/cspd' / group / name
                metrics = directory / 'metrics.json'
                if not (directory / 'run_complete.json').is_file() or not metrics.is_file():
                    missing.append(name)
                    continue
                result = json.loads(metrics.read_text(encoding='utf-8'))
                marker = json.loads((directory / 'run_complete.json').read_text(encoding='utf-8'))
                cfg = result['config']
                if marker.get('config') != cfg or cfg['exp_name'] != name:
                    raise ValueError('result identity mismatch: ' + name)
                hold = result.get('holdout')
                if not hold or not hold.get('query_count'):
                    missing.append(name + ':no_holdout_support')
                    continue
                bucket = hold.get('by_source', {}).get(fold, {})
                runs[name] = {'acc': hold['acc'], 'query_count': hold['query_count'],
                              'per_class': hold['per_class'],
                              'pairs': [pair_report(bucket, p) for p in PAIRS[fold]],
                              'train_sources_same_acc': result['test_same']['acc'],
                              'hard_sampling': result['hard_sampling'],
                              'neu_mde_preregistered': cfg.get('acceptance_neu_mde'),
                              'skipped': hold['skipped'],
                              'group_failures': hold.get('group_failures', {})}
    comparisons = {}
    for fold in PAIRS:
        rows = []
        for seed in [0, 1, 2]:
            a, b = runs.get('A_{}_s{}'.format(fold, seed)), runs.get('B_{}_s{}'.format(fold, seed))
            if not a or not b:
                continue
            deltas = []
            for ap, bp in zip(a['pairs'], b['pairs']):
                av, bv = ap['macro_recall'], bp['macro_recall']
                deltas.append({'pair': ap['pair'], 'delta_macro_recall': bv-av if av is not None and bv is not None else None})
            rows.append({'seed': seed, 'delta_acc_B_minus_A': b['acc']-a['acc'],
                         'pair_deltas': deltas})
        values = [r['delta_acc_B_minus_A'] for r in rows]
        comparisons[fold] = {'matched': rows, 'mean_delta': statistics.mean(values) if values else None,
                             'sample_sd_delta': statistics.stdev(values) if len(values)>1 else None}
    historical = {'NEU': 0.951, 'CR7': 0.706, 'GC10': 0.665}
    versus_historical = {}
    for fold, threshold in historical.items():
        row = runs.get('B_{}_s0'.format(fold))
        versus_historical[fold] = {
            'historical_layer3_p1_seed0': threshold,
            'acc': None if not row else row['acc'],
            'delta_B_minus_historical': None if not row else row['acc'] - threshold,
        }
    return {'run_id': group, 'plan': plan, 'runs': runs, 'missing': missing, 'comparisons': comparisons,
            'historical_seed0_thresholds': historical,
            'versus_historical_seed0': versus_historical,
            'acceptance': 'MANUAL / NOT ESTABLISHED: default plan is B seed0 x 3 folds vs historical layer3_p1 seed0; '
                          'inspect per-class/pair recalls and >=2 sources; '
                          'NEU MDE must have been set before running; missing results are NA, not zero; '
                          'seed0 thresholds are screening only, no statistical significance claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--experiments', nargs='+', choices=['A', 'B', 'C'])
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    variants = args.experiments or ['B']
    seeds = args.seeds if args.seeds is not None else ([0] if not args.experiments else None)
    if seeds is None:
        plan = [(v, {'A': [0, 1, 2], 'B': [0], 'C': [0]}[v]) for v in variants]
    else:
        plan = [(v, seeds) for v in variants]
    print(json.dumps(summarize(args.run_id, plan), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
