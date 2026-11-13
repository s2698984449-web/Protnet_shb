"""Paired holdout evaluation on a fixed episode manifest.

Creates the manifest once (if missing), then evaluates two checkpoints
sequentially. Does not overwrite an existing manifest. Does not train.

Example:
  D:\\miniconda3\\envs\\MAML\\python.exe scripts\\eval_paired_holdout.py \\
    --config config\\cspd\\gc10_hp_control_a.yaml \\
    --baseline output\\cspd\\hp_b_samplerfix_20261111_v2\\B_GC10_s0\\last.pt \\
    --candidate output\\cspd\\gc10_hp_control_20261111\\A_GC10_s0\\last.pt \\
    --manifest output\\cspd\\gc10_hp_control_20261111\\paired_eval\\manifest.json \\
    --output output\\cspd\\gc10_hp_control_20261111\\paired_eval
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cspd.data import make_curriculum
from cspd.runtime import build_model, forward_episode, move_episode, pick_device


ALLOWED_DIFF_KEYS = {
    'exp_name', 'same_hard_pairs', 'output_group', 'experiment_variant', 'config_path',
}
ARCH_KEYS = (
    'train_mode', 'feat_dim', 'cond_dim', 'cond_slots', 'context_dilate_px',
    'feature_layer', 'pretrained_backbone', 'freeze_backbone', 'use_policy',
    'freeze_policy', 'fixed_beta', 'beta_center', 'beta_scale', 'image_size',
    'crop_to_mask', 'crop_margin', 'crop_min_side', 'crop_square',
    'require_mask', 'n_way', 'k_choices', 'k_probs', 'q_query', 'strict_groups',
    'holdout_source', 'holdout_eval', 'holdout_pool', 'data_root', 'seed',
)


def project_path(path, must_exist=False, kind='path'):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError('{} must stay under the project root: {}'.format(kind, path)) from exc
    if must_exist and not path.exists():
        raise FileNotFoundError('{} not found: {}'.format(kind, path))
    return path


def load_yaml_config(path):
    path = project_path(path, must_exist=True, kind='config')
    with path.open('r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError('config is not a mapping: ' + str(path))
    return path, cfg


def holdout_cfg(cfg):
    hold_name = str(cfg.get('holdout_source') or '').strip()
    if not hold_name:
        raise ValueError('holdout_source is required')
    out = dict(cfg)
    out['query_only'] = hold_name
    return out, hold_name


def holdout_cross(cfg):
    return str(cfg.get('holdout_eval') or 'cross') != 'same'


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def compare_configs(left, right, left_name, right_name):
    keys = sorted(set(left) | set(right))
    rows = []
    for key in keys:
        in_l = key in left
        in_r = key in right
        same = in_l and in_r and left[key] == right[key]
        rows.append({
            'key': key,
            'equal': same,
            'allowed_unique': key in ALLOWED_DIFF_KEYS,
            left_name: left.get(key, '<missing>'),
            right_name: right.get(key, '<missing>'),
        })
    unexpected = [row for row in rows if not row['equal'] and not row['allowed_unique']]
    allowed = [row for row in rows if not row['equal'] and row['allowed_unique']]
    return {
        'left': left_name,
        'right': right_name,
        'allowed_differences': jsonable(allowed),
        'unexpected_differences': jsonable(unexpected),
        'ok': not unexpected,
    }


def baseline_saved_config(baseline_pt):
    cfg_json = baseline_pt.parent / 'config.json'
    if not cfg_json.is_file():
        raise FileNotFoundError('B config.json not found next to baseline: ' + str(cfg_json))
    return json.loads(cfg_json.read_text(encoding='utf-8'))


def make_empty_metrics():
    return {
        'acc': 0.0,
        'loss': 0.0,
        'query_count': 0,
        'episode_count': 0,
        'per_class': {},
        'confusion': {},
        'predictions': [],
    }


def accumulate_episode(metrics, episode, pred, label):
    logits = pred['logits']
    y = episode['query_y']
    names = list(episode['class_names'])
    cls = F.cross_entropy(logits, y)
    pred_y = logits.argmax(dim=-1)
    hits = pred_y == y
    y_list = y.detach().cpu().tolist()
    p_list = pred_y.detach().cpu().tolist()
    hit_list = hits.detach().cpu().tolist()
    metrics['query_count'] += int(y.numel())
    metrics['episode_count'] += 1
    metrics['loss'] = metrics['loss'] + float(cls)
    for i, (true_i, pred_i, hit) in enumerate(zip(y_list, p_list, hit_list)):
        true_name = names[int(true_i)]
        pred_name = names[int(pred_i)]
        slot = metrics['per_class'].setdefault(true_name, [0, 0])
        slot[1] += 1
        slot[0] += int(hit)
        if true_name != pred_name:
            key = '{}|{}'.format(true_name, pred_name)
            metrics['confusion'][key] = metrics['confusion'].get(key, 0) + 1
        metrics['predictions'].append({
            'label': label,
            'episode_id': int(episode['episode_id']),
            'query_index': i,
            'true_class': true_name,
            'pred_class': pred_name,
            'true_index': int(true_i),
            'pred_index': int(pred_i),
            'correct': bool(hit),
            'query_path': episode['query_records'][i]['path'],
        })


def finalize_metrics(metrics):
    n_ep = max(metrics['episode_count'], 1)
    total = max(metrics['query_count'], 1)
    hits = sum(item[0] for item in metrics['per_class'].values())
    out = dict(metrics)
    out['acc'] = hits / total
    out['loss'] = metrics['loss'] / n_ep if metrics['episode_count'] else 0.0
    out['per_class'] = {
        name: {'hit': hit, 'denominator': count, 'recall': hit / max(count, 1)}
        for name, (hit, count) in sorted(metrics['per_class'].items())
    }
    return out


def create_manifest(cfg, manifest_path, seed, tasks):
    hold_cfg, hold_name = holdout_cfg(cfg)
    if holdout_cross(cfg):
        raise ValueError('this entry is for holdout_eval=same; refused to build a cross-source manifest')
    sampler = make_curriculum(hold_cfg, 'test', int(seed))
    episodes = []
    attempts = 0
    limit = max(int(tasks) * 10, int(tasks))
    last_error = None
    while len(episodes) < int(tasks):
        attempts += 1
        if attempts > limit:
            raise RuntimeError(
                'could not collect {} legal holdout episodes (got {}, last error: {})'.format(
                    tasks, len(episodes), last_error))
        try:
            raw = sampler.sample(
                cross_source=False, stage_sources=False, allow_fallback=False, record_paths=True)
        except RuntimeError as exc:
            last_error = exc
            continue
        record = {
            'episode_id': len(episodes),
            'n_way': int(raw['n_way']),
            'k_shot': int(raw['k_shot']),
            'q_query': int(cfg.get('q_query', 8)),
            'query_source': raw['query_source'],
            'cross_source': bool(raw['cross_source']),
            'class_names': list(raw['class_names']),
            'support': list(raw['support_records']),
            'query': list(raw['query_records']),
        }
        # Fail closed: do not relax groups or mix sources to salvage a draw.
        sampler.validate_episode_record(record, expected_source=hold_name)
        episodes.append(record)
    payload = {
        'schema': 'holdout_eval_manifest_v1',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'note': (
            'Independent fixed holdout set. Not a reconstruction of B official '
            'eval episodes (those paths were not logged). Paired A/B numbers on '
            'this manifest must not be compared as if they were B reported holdout acc.'
        ),
        'manifest_seed': int(seed),
        'n_tasks': int(tasks),
        'n_attempts_used': attempts,
        'holdout_source': hold_name,
        'split': 'test',
        'holdout_pool': str(cfg.get('holdout_pool') or ''),
        'holdout_eval': str(cfg.get('holdout_eval') or ''),
        'cross_source': False,
        'strict_groups': bool(cfg.get('strict_groups', False)),
        'q_query': int(cfg.get('q_query', 8)),
        'k_choices': list(cfg.get('k_choices') or [3, 5]),
        'k_probs': list(cfg.get('k_probs') or [0.7, 0.3]),
        'n_way_cap': int(cfg.get('n_way', 5)),
        'data_root': str(Path(cfg['data_root'])),
        'sampler_seed_formula': 'independent manifest_seed (not training seed+3)',
        'episodes': episodes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(manifest_path)
    return payload


def load_and_validate_manifest(manifest_path, cfg):
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    if payload.get('schema') != 'holdout_eval_manifest_v1':
        raise ValueError('unsupported manifest schema: {}'.format(payload.get('schema')))
    hold_cfg, hold_name = holdout_cfg(cfg)
    sampler = make_curriculum(hold_cfg, 'test', 0)
    if str(payload.get('holdout_source')) != hold_name:
        raise ValueError('manifest holdout_source mismatch')
    if bool(payload.get('cross_source')) or holdout_cross(cfg):
        raise ValueError('manifest/config is not holdout_eval=same')
    if int(payload.get('q_query', -1)) != int(cfg.get('q_query', 8)):
        raise ValueError('manifest q_query mismatch')
    episodes = payload.get('episodes') or []
    if not episodes:
        raise ValueError('manifest has no episodes')
    for rec in episodes:
        sampler.validate_episode_record(rec, expected_source=hold_name)
    return payload, sampler


def load_checkpoint_model(cfg, ckpt_path, device):
    try:
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location=device)
    saved_cfg = state.get('config') if isinstance(state, dict) else None
    if not isinstance(saved_cfg, dict):
        saved_cfg = dict(cfg)
    arch_diff = compare_configs(
        {k: cfg.get(k) for k in ARCH_KEYS},
        {k: saved_cfg.get(k) for k in ARCH_KEYS},
        'eval_config', 'checkpoint_config')
    if not arch_diff['ok']:
        raise ValueError('checkpoint architecture/protocol mismatch: {}'.format(
            arch_diff['unexpected_differences']))
    model = build_model(saved_cfg).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model


def evaluate_manifest(model, sampler, payload, device, label):
    metrics = make_empty_metrics()
    model.eval()
    with torch.no_grad():
        for rec in payload['episodes']:
            raw = sampler.episode_from_record(rec)
            raw['episode_id'] = int(rec['episode_id'])
            episode = move_episode(raw, device)
            pred = forward_episode(model, episode, int(episode['n_way']))
            accumulate_episode(metrics, episode, pred, label)
    return finalize_metrics(metrics)


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def paired_report(baseline, candidate):
    base = {(p['episode_id'], p['query_index']): p for p in baseline['predictions']}
    cand = {(p['episode_id'], p['query_index']): p for p in candidate['predictions']}
    if set(base) != set(cand):
        raise ValueError('A/B prediction keys do not cover the same queries')
    counts = Counter()
    disagreements = []
    for key in sorted(base):
        b = base[key]
        a = cand[key]
        if a['correct'] and b['correct']:
            tag = 'both_correct'
        elif (not a['correct']) and (not b['correct']):
            tag = 'both_wrong'
        elif a['correct'] and (not b['correct']):
            tag = 'A_correct_B_wrong'
        else:
            tag = 'B_correct_A_wrong'
        counts[tag] += 1
        if tag in {'A_correct_B_wrong', 'B_correct_A_wrong'}:
            disagreements.append({
                'episode_id': key[0],
                'query_index': key[1],
                'true_class': a['true_class'],
                'A_pred': a['pred_class'],
                'B_pred': b['pred_class'],
                'pattern': tag,
                'query_path': a['query_path'],
            })
    total = sum(counts.values())
    return {
        'query_count': total,
        'both_correct': counts['both_correct'],
        'both_wrong': counts['both_wrong'],
        'A_correct_B_wrong': counts['A_correct_B_wrong'],
        'B_correct_A_wrong': counts['B_correct_A_wrong'],
        'disagreements': disagreements,
        'note': 'Single-seed paired counts only. Do not treat differences as significance.',
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_check_config(args):
    config_path, cfg = load_yaml_config(args.config)
    baseline = project_path(args.baseline, must_exist=True, kind='baseline')
    saved = baseline_saved_config(baseline)
    diff = compare_configs(cfg, saved, 'A_yaml', 'B_config_json')
    print('A config:', config_path)
    print('B config:', baseline.parent / 'config.json')
    print('allowed unique variables:')
    for row in diff['allowed_differences']:
        print('  {} : A={!r}  B={!r}'.format(row['key'], row['A_yaml'], row['B_config_json']))
    if diff['ok']:
        print('unexpected differences: none')
    else:
        print('unexpected differences:')
        for row in diff['unexpected_differences']:
            print('  {} : A={!r}  B={!r}'.format(row['key'], row['A_yaml'], row['B_config_json']))
            raise SystemExit(2)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='A-control yaml (protocol + holdout settings)')
    parser.add_argument('--baseline', required=True, help='existing B last.pt (read-only)')
    parser.add_argument('--candidate', help='A last.pt; default output/cspd/<output_group>/<exp_name>/last.pt')
    parser.add_argument('--manifest', help='fixed episode json; created once if missing, never randomly overwritten')
    parser.add_argument('--output', help='paired eval output directory')
    parser.add_argument('--manifest-seed', type=int, default=20261111,
                        help='used only when creating a new manifest')
    parser.add_argument('--tasks', type=int, default=None,
                        help='episode count when creating a manifest; default eval_tasks from config')
    parser.add_argument('--device', default=None)
    parser.add_argument('--check-config', action='store_true',
                        help='static A vs B config diff only; no model load, no manifest sampling')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.check_config:
        return run_check_config(args)
    config_path, cfg = load_yaml_config(args.config)
    if not bool(cfg.get('hard_pair_protocol', False)):
        raise ValueError('paired eval expects hard_pair_protocol=true (same eval structure)')
    baseline = project_path(args.baseline, must_exist=True, kind='baseline')
    group = str(cfg.get('output_group') or '').strip()
    name = str(cfg.get('exp_name') or '').strip()
    if args.candidate:
        candidate = project_path(args.candidate, kind='candidate')
    else:
        candidate = project_path(ROOT / 'output' / 'cspd' / group / name / 'last.pt', kind='candidate')
    if args.output:
        out_dir = project_path(args.output, kind='output')
    else:
        out_dir = project_path(ROOT / 'output' / 'cspd' / group / 'paired_eval', kind='output')
    if args.manifest:
        manifest_path = project_path(args.manifest, kind='manifest')
    else:
        manifest_path = out_dir / 'manifest.json'
    tasks = int(args.tasks or cfg.get('eval_tasks') or 20)
    diff = compare_configs(cfg, baseline_saved_config(baseline), 'A_yaml', 'B_config_json')
    if not diff['ok']:
        raise ValueError('A/B config has unexpected differences: {}'.format(
            diff['unexpected_differences']))
    created = False
    if manifest_path.is_file():
        payload, sampler = load_and_validate_manifest(manifest_path, cfg)
        print('reused manifest', manifest_path, 'episodes', len(payload['episodes']))
    else:
        payload = create_manifest(cfg, manifest_path, int(args.manifest_seed), tasks)
        _, sampler = load_and_validate_manifest(manifest_path, cfg)
        created = True
        print('created manifest', manifest_path, 'episodes', len(payload['episodes']),
              'attempts', payload['n_attempts_used'])
    if not candidate.is_file():
        raise FileNotFoundError(
            'candidate checkpoint not found: {}. Train A first; manifest is ready at {}'.format(
                candidate, manifest_path))
    device = pick_device(args.device or cfg.get('device') or 'cuda')
    print('evaluate baseline sequentially on', device)
    model_b = load_checkpoint_model(cfg, baseline, device)
    metrics_b = evaluate_manifest(model_b, sampler, payload, device, 'B')
    release_model(model_b)
    print('evaluate candidate sequentially on', device)
    model_a = load_checkpoint_model(cfg, candidate, device)
    metrics_a = evaluate_manifest(model_a, sampler, payload, device, 'A')
    release_model(model_a)
    paired = paired_report(metrics_b, metrics_a)
    result = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'config_path': str(config_path),
        'baseline_ckpt': str(baseline),
        'candidate_ckpt': str(candidate),
        'manifest_path': str(manifest_path),
        'manifest_created_this_run': created,
        'manifest_seed': payload.get('manifest_seed'),
        'n_tasks': len(payload['episodes']),
        'device': str(device),
        'config_diff': diff,
        'B': {k: v for k, v in metrics_b.items() if k != 'predictions'},
        'A': {k: v for k, v in metrics_a.items() if k != 'predictions'},
        'paired': {k: v for k, v in paired.items() if k != 'disagreements'},
        'note': (
            'Evaluated sequentially with torch.no_grad. Loss is mean per-episode '
            'cross-entropy, same formula as evaluate_curriculum. This manifest is '
            'independent of B official holdout episodes. Single seed, no significance claim.'
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'paired_metrics.json', result)
    write_json(out_dir / 'predictions.json', {
        'A': metrics_a['predictions'],
        'B': metrics_b['predictions'],
        'disagreements': paired['disagreements'],
    })
    write_json(out_dir / 'config_diff.json', diff)
    print('A acc {:.5f} loss {:.4f}  B acc {:.5f} loss {:.4f}  queries {}'.format(
        metrics_a['acc'], metrics_a['loss'], metrics_b['acc'], metrics_b['loss'],
        paired['query_count']))
    print('paired A_correct_B_wrong {} B_correct_A_wrong {} both_correct {} both_wrong {}'.format(
        paired['A_correct_B_wrong'], paired['B_correct_A_wrong'],
        paired['both_correct'], paired['both_wrong']))
    print('wrote', out_dir / 'paired_metrics.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
