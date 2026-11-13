#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 diagnostics: common-error trace, support-group distribution, optional geometry.

Read-only on original data / checkpoints / configs. Writes only under
output/cspd/gc10_hp_control_20261111/diagnostics/p0_common_errors/.

Phases:
  data  -- manifest + predictions only (fast)
  sheets -- PIL contact sheets for pre-registered first examples
  geom  -- sequential no_grad feature extraction (A then B)
  all   -- data, sheets, then geom
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

def _project_root(start):
    for path in [start, *start.parents]:
        if (path / 'cspd' / 'data.py').is_file() and (path / 'scripts' / 'train_cspd.py').is_file():
            return path
    raise RuntimeError('project root not found from ' + str(start))


ROOT = _project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / 'output' / 'cspd' / 'gc10_hp_control_20261111' / 'diagnostics' / 'p0_common_errors'
PAIRED = ROOT / 'output' / 'cspd' / 'gc10_hp_control_20261111' / 'paired_eval'
A_CKPT = ROOT / 'output' / 'cspd' / 'gc10_hp_control_20261111' / 'A_GC10_s0' / 'last.pt'
B_CKPT = ROOT / 'output' / 'cspd' / 'hp_b_samplerfix_20261111_v2' / 'B_GC10_s0' / 'last.pt'
A_CFG = ROOT / 'output' / 'cspd' / 'gc10_hp_control_20261111' / 'A_GC10_s0' / 'config.json'

FOCUS = (('oil', 'punch'), ('punch', 'weld_line'), ('rolled_pit', 'fold_wrinkle'))
# Pre-registered extras: largest common-same direction, not cherry-picked by appearance.
EXTRA_FOCUS = (('inclusion', 'silk_spot'),)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ns_group(source, class_name, gid):
    return '{}::{}::{}'.format(source, class_name, gid)


def group_stats(items, class_name, source='GC10'):
    raw = Counter(str(it['group_id']) for it in items)
    n = sum(raw.values())
    shares = {g: (c / n) for g, c in raw.items()} if n else {}
    max_share = max(shares.values()) if shares else 0.0
    n_eff = (1.0 / sum(p * p for p in shares.values())) if shares else 0.0
    sizes = list(raw.values())
    return {
        'n': n,
        'n_groups': len(raw),
        'counts': dict(raw),
        'shares': shares,
        'max_share': max_share,
        'n_eff': n_eff,
        'singleton': len(raw) <= 1,
        'equal_sizes': (len(set(sizes)) == 1) if sizes else True,
        'pooling_can_differ': len(raw) >= 2 and len(set(sizes)) > 1,
        'namespaced': {ns_group(source, class_name, g): c for g, c in raw.items()},
    }


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def index_preds(pred):
    a = {(r['episode_id'], r['query_index']): r for r in pred['A']}
    b = {(r['episode_id'], r['query_index']): r for r in pred['B']}
    if set(a) != set(b):
        raise RuntimeError('A/B prediction keys do not match')
    return a, b


def manifest_query(ep, qidx):
    return ep['query'][qidx]


def data_phase():
    man = load_json(PAIRED / 'manifest.json')
    pred = load_json(PAIRED / 'predictions.json')
    metrics = load_json(PAIRED / 'paired_metrics.json')
    a_map, b_map = index_preds(pred)
    episodes = {e['episode_id']: e for e in man['episodes']}
    data_root = Path(man['data_root'])

    # Support stats for all 100 class-supports (20 ep x 5 classes).
    support_rows = []
    pooling = Counter()
    pooling_by_k = defaultdict(Counter)
    for ep in man['episodes']:
        k = int(ep['k_shot'])
        for name in ep['class_names']:
            items = [it for it in ep['support'] if it['class'] == name]
            st = group_stats(items, name, ep['query_source'])
            if st['singleton']:
                tag = 'singleton'
            elif st['equal_sizes']:
                tag = 'equal_multi'
            else:
                tag = 'uneven_multi'
            pooling[tag] += 1
            pooling_by_k[k][tag] += 1
            support_rows.append({
                'episode_id': ep['episode_id'],
                'k_shot': k,
                'class': name,
                'n_support': st['n'],
                'n_groups': st['n_groups'],
                'max_share': '{:.4f}'.format(st['max_share']),
                'n_eff': '{:.4f}'.format(st['n_eff']),
                'counts': json.dumps(st['counts'], ensure_ascii=False),
                'namespaced': json.dumps(st['namespaced'], ensure_ascii=False),
                'pooling_tag': tag,
                'pooling_can_differ': int(st['pooling_can_differ']),
                'dominant_group': (max(st['counts'], key=st['counts'].get) if st['counts'] else ''),
            })

    # All 800 queries as reference, plus common-error subset.
    query_rows = []
    common_rows = []
    path_freq_all = Counter()
    path_freq_common = Counter()
    common_same = 0
    common_diff = 0
    dir_A = Counter()
    dir_B = Counter()
    dir_common_same = Counter()
    dir_common_diff = Counter()
    for key in sorted(a_map):
        eid, qidx = key
        ep = episodes[eid]
        a, b = a_map[key], b_map[key]
        qitem = manifest_query(ep, qidx)
        # sanity: path field
        if a['query_path'] != qitem['path'] or b['query_path'] != qitem['path']:
            raise RuntimeError('query_path mismatch at {}'.format(key))
        true_c = a['true_class']
        assert true_c == b['true_class'] == qitem['class']
        src = qitem['source']
        gid = str(qitem['group_id'])
        true_items = [it for it in ep['support'] if it['class'] == true_c]
        st_true = group_stats(true_items, true_c, src)
        st_a = group_stats([it for it in ep['support'] if it['class'] == a['pred_class']], a['pred_class'], src)
        st_b = group_stats([it for it in ep['support'] if it['class'] == b['pred_class']], b['pred_class'], src)
        path_freq_all[qitem['path']] += 1
        both_wrong = (not a['correct']) and (not b['correct'])
        same_pred = a['pred_class'] == b['pred_class']
        if not a['correct']:
            dir_A[(true_c, a['pred_class'])] += 1
        if not b['correct']:
            dir_B[(true_c, b['pred_class'])] += 1
        row = {
            'episode_id': eid,
            'query_index': qidx,
            'k_shot': ep['k_shot'],
            'true_class': true_c,
            'query_path': qitem['path'],
            'query_group_raw': gid,
            'query_group_ns': ns_group(src, true_c, gid),
            'query_split': qitem.get('split'),
            'A_pred': a['pred_class'],
            'B_pred': b['pred_class'],
            'A_correct': int(a['correct']),
            'B_correct': int(b['correct']),
            'both_wrong': int(both_wrong),
            'same_pred': int(same_pred),
            'true_support_n_groups': st_true['n_groups'],
            'true_support_max_share': '{:.4f}'.format(st_true['max_share']),
            'true_support_n_eff': '{:.4f}'.format(st_true['n_eff']),
            'true_support_counts': json.dumps(st_true['counts'], ensure_ascii=False),
            'true_support_can_differ': int(st_true['pooling_can_differ']),
            'A_pred_support_n_groups': st_a['n_groups'],
            'A_pred_support_max_share': '{:.4f}'.format(st_a['max_share']),
            'A_pred_support_counts': json.dumps(st_a['counts'], ensure_ascii=False),
            'B_pred_support_n_groups': st_b['n_groups'],
            'B_pred_support_max_share': '{:.4f}'.format(st_b['max_share']),
            'B_pred_support_counts': json.dumps(st_b['counts'], ensure_ascii=False),
        }
        query_rows.append(row)
        if both_wrong:
            path_freq_common[qitem['path']] += 1
            if same_pred:
                common_same += 1
                dir_common_same[(true_c, a['pred_class'])] += 1
                pattern = 'same_wrong_class'
            else:
                common_diff += 1
                dir_common_diff[(true_c, a['pred_class'], b['pred_class'])] += 1
                pattern = 'different_wrong_class'
            row = dict(row)
            row['pattern'] = pattern
            common_rows.append(row)

    freq_dist = Counter(path_freq_common.values())

    def ref_share(subset, field, thresh=0.67):
        vals = [float(r[field]) for r in subset]
        return {
            'n': len(vals),
            'mean_max_share': (sum(vals) / len(vals)) if vals else None,
            'frac_max_share_ge_{}'.format(str(thresh).replace('.', 'p')): (
                sum(v >= thresh for v in vals) / len(vals) if vals else None),
            'frac_singleton_true_support': (
                sum(int(r['true_support_n_groups']) <= 1 for r in subset) / len(subset)
                if subset else None),
        }

    all_q = query_rows
    correct_both = [r for r in all_q if r['A_correct'] and r['B_correct']]
    common_only = common_rows

    focus_counts = {}
    for src, dst in list(FOCUS) + list(EXTRA_FOCUS):
        focus_counts['{}->{}'.format(src, dst)] = {
            'A': dir_A[(src, dst)],
            'B': dir_B[(src, dst)],
            'common_same': dir_common_same[(src, dst)],
            'common_diff_involving': sum(
                v for (t, pa, pb), v in dir_common_diff.items()
                if t == src and (pa == dst or pb == dst)),
        }

    # First pre-registered examples (episode_id, query_index order).
    first_examples = {}
    for src, dst in list(FOCUS) + list(EXTRA_FOCUS):
        hit = None
        for r in common_rows:
            if r['true_class'] == src and r['A_pred'] == dst and r['B_pred'] == dst:
                hit = (r['episode_id'], r['query_index'], r['k_shot'], r['query_path'],
                       r['query_group_ns'], r['true_support_counts'],
                       r['A_pred_support_counts'])
                break
        first_examples['{}->{}'.format(src, dst)] = hit

    # Unique path table
    path_rows = []
    # map path -> groups (should be unique per path)
    path_meta = {}
    for r in common_rows:
        path_meta.setdefault(r['query_path'], {
            'group_ns': r['query_group_ns'],
            'true_class': r['true_class'],
            'times': 0,
            'A_preds': Counter(),
            'B_preds': Counter(),
        })
        path_meta[r['query_path']]['times'] += 1
        path_meta[r['query_path']]['A_preds'][r['A_pred']] += 1
        path_meta[r['query_path']]['B_preds'][r['B_pred']] += 1
    for path, meta in sorted(path_meta.items(), key=lambda kv: (-kv[1]['times'], kv[0])):
        path_rows.append({
            'query_path': path,
            'true_class': meta['true_class'],
            'query_group_ns': meta['group_ns'],
            'times_in_common_errors': meta['times'],
            'A_preds': json.dumps(dict(meta['A_preds']), ensure_ascii=False),
            'B_preds': json.dumps(dict(meta['B_preds']), ensure_ascii=False),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'support_class_stats.csv', support_rows,
              list(support_rows[0].keys()) if support_rows else ['episode_id'])
    write_csv(OUT / 'queries_all.csv', query_rows,
              list(query_rows[0].keys()))
    write_csv(OUT / 'common_errors.csv', common_rows,
              list(common_rows[0].keys()) if common_rows else ['episode_id'])
    write_csv(OUT / 'common_error_unique_paths.csv', path_rows,
              list(path_rows[0].keys()) if path_rows else ['query_path'])

    stats = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'zone': 'Asia/Shanghai (UTC+8)',
        'sources': {
            'manifest': str(PAIRED / 'manifest.json'),
            'predictions': str(PAIRED / 'predictions.json'),
            'paired_metrics': str(PAIRED / 'paired_metrics.json'),
            'baseline_ckpt': str(metrics.get('baseline_ckpt') or B_CKPT),
            'candidate_ckpt': str(metrics.get('candidate_ckpt') or A_CKPT),
            'label_check': 'baseline=B (hardpair on), candidate=A (hardpair off)',
        },
        'group_rule': {
            'GC10': (
                'groups.json IDs are extracted by regex img_\\d+_(\\d+)_\\d+ from the '
                'filename stem (steel_sources_v3.yaml / merge_sources.group_ids_for, mode=regex). '
                'The documented generation method calls this a filename batch number '
                '(批次号), not a 32x32 histogram cluster (that mode is for other sources). '
                'IDs are namespaced as source::class::id in this report; the same numeric '
                'token in two classes is not treated as one batch.'
            ),
            'mask_role': 'Dataset masks are used only to crop around the annotated defect, not as a model-predicted mask.',
        },
        'common_errors': {
            'n_observations': len(common_rows),
            'n_unique_query_paths': len(path_freq_common),
            'repeat_freq_dist': dict(sorted(freq_dist.items())),
            'max_repeat': max(path_freq_common.values()) if path_freq_common else 0,
            'same_wrong_class': common_same,
            'different_wrong_class': common_diff,
            'top_common_same_directions': [
                {'true': t, 'pred': p, 'n': n}
                for (t, p), n in dir_common_same.most_common(15)
            ],
            'common_diff_directions': [
                {'true': t, 'A_pred': pa, 'B_pred': pb, 'n': n}
                for (t, pa, pb), n in dir_common_diff.most_common(20)
            ],
            'focus_counts': focus_counts,
        },
        'pooling_possibility_100_class_supports': {
            'n': 100,
            'counts': dict(pooling),
            'by_k': {str(k): dict(v) for k, v in pooling_by_k.items()},
            'frac_pooling_can_differ': pooling['uneven_multi'] / 100.0,
            'note': (
                'Group-equal-weight equals the ordinary mean when the class support has '
                'one group, or when every group has the same shot count. Only uneven_multi '
                'supports can change under P1. This is a mathematical possibility, not an '
                'effect estimate. P1 was not run.'
            ),
        },
        'true_support_dominance_reference': {
            'threshold_note': 'max_share>=2/3 is a descriptive cutoff, not a causal test.',
            'all_queries': ref_share(all_q, 'true_support_max_share'),
            'both_correct': ref_share(correct_both, 'true_support_max_share'),
            'common_errors': ref_share(common_only, 'true_support_max_share'),
            'by_k_common': {
                str(k): ref_share([r for r in common_only if int(r['k_shot']) == k],
                                  'true_support_max_share')
                for k in (3, 5)
            },
            'by_k_both_correct': {
                str(k): ref_share([r for r in correct_both if int(r['k_shot']) == k],
                                  'true_support_max_share')
                for k in (3, 5)
            },
        },
        'first_pre_registered_examples': {
            key: (None if v is None else {
                'episode_id': v[0], 'query_index': v[1], 'k_shot': v[2],
                'query_path': v[3], 'query_group_ns': v[4],
                'true_support_counts': v[5], 'pred_support_counts': v[6],
            })
            for key, v in first_examples.items()
        },
        'paired_acc_reminder': {
            'A_off': metrics.get('A', {}).get('acc'),
            'B_on': metrics.get('B', {}).get('acc'),
            'both_wrong': metrics.get('paired', {}).get('both_wrong'),
        },
        'geometry': 'not_run_in_data_phase',
    }
    write_json(OUT / 'p0_data_stats.json', stats)
    return stats, man, pred, a_map, b_map, episodes, data_root, first_examples


def _crop_rgb(path, mask_path, data_root):
    from PIL import Image
    from cspd.data import crop_around_mask
    img = Image.open(data_root / path).convert('RGB')
    mask = None
    if mask_path:
        mp = data_root / mask_path
        if mp.is_file():
            mask = Image.open(mp).convert('L')
    if mask is not None:
        img, _m = crop_around_mask(img, mask, margin=0.35, min_side=96, square=True)
    return img


def _cell(img, label, size=140):
    from PIL import Image, ImageDraw, ImageFont
    im = img.copy()
    im.thumbnail((size, size))
    canvas = Image.new('RGB', (size, size + 36), (255, 255, 255))
    ox = (size - im.width) // 2
    oy = (size - im.height) // 2
    canvas.paste(im, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((2, size + 2), label[:42], fill=(0, 0, 0), font=font)
    return canvas


def make_contact_sheet(ep, qidx, a_row, b_row, data_root, out_path, title):
    from PIL import Image, ImageDraw, ImageFont
    qitem = ep['query'][qidx]
    true_c = qitem['class']
    # If A/B same pred, one pred row; else two.
    pred_classes = []
    if a_row['pred_class'] == b_row['pred_class']:
        pred_classes.append(('pred A=B ' + a_row['pred_class'], a_row['pred_class']))
    else:
        pred_classes.append(('pred A ' + a_row['pred_class'], a_row['pred_class']))
        pred_classes.append(('pred B ' + b_row['pred_class'], b_row['pred_class']))
    rows_imgs = []
    qimg = _crop_rgb(qitem['path'], qitem.get('mask_path'), data_root)
    qlab = 'Q {} g={} A={} B={}'.format(
        true_c, qitem['group_id'], a_row['pred_class'], b_row['pred_class'])
    query_cell = _cell(qimg, qlab, size=180)

    def row_for(title_s, cls):
        items = [it for it in ep['support'] if it['class'] == cls]
        cells = []
        for i, it in enumerate(items):
            im = _crop_rgb(it['path'], it.get('mask_path'), data_root)
            cells.append(_cell(im, 'S{} {} g={}'.format(i, cls, it['group_id'])))
        return title_s, cells

    bands = [row_for('support true ' + true_c, true_c)]
    for lab, cls in pred_classes:
        bands.append(row_for('support ' + lab, cls))

    size = 140
    pad = 6
    label_h = 22
    width = max(180 + 20, max(len(cells) for _t, cells in bands) * (size + pad) + pad)
    height = 40 + 180 + 20 + sum(label_h + size + 36 + pad for _t, cells in bands)
    canvas = Image.new('RGB', (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title[:120], fill=(0, 0, 0), font=font)
    canvas.paste(query_cell, (8, 32))
    y = 32 + 180 + 16
    for lab, cells in bands:
        draw.text((8, y), lab + ' (crop around annotated mask, not a model mask)', fill=(0, 0, 0), font=font)
        y += label_h
        x = pad
        for cell in cells:
            canvas.paste(cell, (x, y))
            x += size + pad
        y += size + 36 + pad
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return out_path


def sheets_phase(man, a_map, b_map, episodes, data_root, first_examples):
    sheet_dir = OUT / 'contact_sheets'
    sheet_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, rec in first_examples.items():
        if rec is None:
            written.append({'direction': key, 'path': None, 'note': 'no common-same example'})
            continue
        eid, qidx = rec[0], rec[1]
        ep = episodes[eid]
        title = 'FIRST {}  ep{} q{} k={}  (pre-registered, not cherry-picked)'.format(
            key, eid, qidx, ep['k_shot'])
        fname = 'first_{}_ep{}_q{}.jpg'.format(key.replace('->', '_to_'), eid, qidx)
        path = make_contact_sheet(
            ep, qidx, a_map[(eid, qidx)], b_map[(eid, qidx)],
            data_root, sheet_dir / fname, title)
        written.append({
            'direction': key,
            'episode_id': eid,
            'query_index': qidx,
            'k_shot': ep['k_shot'],
            'path': str(path),
        })
    write_json(OUT / 'contact_sheets' / 'index.json', written)
    return written


def geom_phase(man, a_map, b_map, episodes):
    import torch
    from cspd.data import make_curriculum
    from cspd.model import _unit_rows
    from cspd.runtime import build_model, move_episode, pick_device

    cfg = load_json(A_CFG)
    hold = dict(cfg)
    hold['query_only'] = str(cfg.get('holdout_source') or 'GC10')
    sampler = make_curriculum(hold, 'test', 0)
    device = pick_device(cfg.get('device') or 'cuda')
    beta_fixed = float(cfg.get('fixed_beta', 0.25))

    def extract(model, ep):
        raw = sampler.episode_from_record(ep)
        episode = move_episode(raw, device)
        ways = int(episode['n_way'])
        names = list(episode['class_names'])
        with torch.no_grad():
            s_feat = model._maps(episode['support_x'])
            q_feat = model._maps(episode['query_x'])
            s_core, s_ring = model._core_ring(s_feat, episode['support_mask'])
            q_core, q_ring = model._core_ring(q_feat, episode['query_mask'])
            if model.use_policy:
                raise RuntimeError('P0 geom expects use_policy=false / fixed beta')
            beta = float(model.fixed_beta)
            p_core = []
            p_ring = []
            logits_cols = []
            for c in range(ways):
                sel = episode['support_y'] == c
                pc = _unit_rows(s_core[sel].mean(dim=0, keepdim=True))
                pr = _unit_rows(s_ring[sel].mean(dim=0, keepdim=True))
                p_core.append(pc)
                p_ring.append(pr)
                sc = (_unit_rows(q_core) * pc).sum(dim=-1)
                sr = (_unit_rows(q_ring) * pr).sum(dim=-1)
                logits_cols.append(sc + beta * sr)
            logits = torch.stack(logits_cols, dim=-1)
            pred = logits.argmax(dim=-1)
            # per-class group means on support
            sy = episode['support_y'].detach().cpu().tolist()
            group_offset = []
            support_items = list(ep['support'])
            for c, name in enumerate(names):
                sel = [i for i, y in enumerate(sy) if y == c]
                cores = s_core[sel]
                rings = s_ring[sel]
                class_c = cores.mean(dim=0, keepdim=True)
                class_r = rings.mean(dim=0, keepdim=True)
                gids = [str(support_items[i]['group_id']) for i in sel]
                byg = defaultdict(list)
                for j, gid in enumerate(gids):
                    byg[gid].append(j)
                offsets = []
                for gid, idxs in byg.items():
                    idx_t = torch.tensor(idxs, device=cores.device)
                    gc_ = cores[idx_t].mean(dim=0, keepdim=True)
                    gr_ = rings[idx_t].mean(dim=0, keepdim=True)
                    cos_c = float((_unit_rows(gc_) * _unit_rows(class_c)).sum())
                    cos_r = float((_unit_rows(gr_) * _unit_rows(class_r)).sum())
                    offsets.append({
                        'group_ns': ns_group(ep['query_source'], name, gid),
                        'n': len(idxs),
                        'cos_to_class_mean_core': cos_c,
                        'cos_to_class_mean_ring': cos_r,
                    })
                group_offset.append({'class': name, 'groups': offsets})
            q_rows = []
            qc_u = _unit_rows(q_core)
            qr_u = _unit_rows(q_ring)
            for i in range(int(logits.shape[0])):
                scores = logits[i].detach().cpu().tolist()
                true_i = int(episode['query_y'][i].item())
                pred_i = int(pred[i].item())
                s_true = scores[true_i]
                others = [s for j, s in enumerate(scores) if j != true_i]
                margin = s_true - (max(others) if others else s_true)
                cos_core = [
                    float((qc_u[i] * p_core[c]).sum()) for c in range(ways)
                ]
                cos_ring = [
                    float((qr_u[i] * p_ring[c]).sum()) for c in range(ways)
                ]
                q_rows.append({
                    'query_index': i,
                    'pred_class': names[pred_i],
                    'pred_index': pred_i,
                    'scores': {names[c]: scores[c] for c in range(ways)},
                    'cos_core': {names[c]: cos_core[c] for c in range(ways)},
                    'cos_ring': {names[c]: cos_ring[c] for c in range(ways)},
                    's_true': s_true,
                    'margin': margin,
                    'beta': beta,
                    'score_def': 'cos_core + beta * cos_ring',
                    'margin_def': 's_true - max_other',
                    'cos_core_true': cos_core[true_i],
                    'cos_ring_true': cos_ring[true_i],
                    'cos_core_pred': cos_core[pred_i],
                    'cos_ring_pred': cos_ring[pred_i],
                })
        # free gpu tensors
        del s_feat, q_feat, s_core, s_ring, q_core, q_ring, logits, episode, raw
        return q_rows, group_offset, beta, names

    def run_ckpt(label, ckpt):
        try:
            state = torch.load(str(ckpt), map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(str(ckpt), map_location=device)
        model = build_model(state.get('config') or cfg).to(device)
        model.load_state_dict(state['model'])
        model.eval()
        out = {}
        mismatches = []
        with torch.no_grad():
            for ep in man['episodes']:
                q_rows, group_offset, beta, names = extract(model, ep)
                if abs(beta - beta_fixed) > 1e-8:
                    raise RuntimeError('unexpected beta {} vs {}'.format(beta, beta_fixed))
                saved = a_map if label == 'A' else b_map
                for row in q_rows:
                    key = (ep['episode_id'], row['query_index'])
                    rec = saved[key]
                    if rec['pred_class'] != row['pred_class']:
                        mismatches.append({
                            'label': label,
                            'episode_id': ep['episode_id'],
                            'query_index': row['query_index'],
                            'saved': rec['pred_class'],
                            'recomputed': row['pred_class'],
                        })
                    row['episode_id'] = ep['episode_id']
                    row['saved_pred'] = rec['pred_class']
                    row['match_saved'] = rec['pred_class'] == row['pred_class']
                out[ep['episode_id']] = {
                    'names': names,
                    'k_shot': ep['k_shot'],
                    'beta': beta,
                    'queries': q_rows,
                    'support_group_offsets': group_offset,
                }
        del model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out, mismatches

    print('geom: evaluate B then A sequentially on', device, flush=True)
    # B first or A first — user said A/B sequentially, avoid 4GB. Either order is fine.
    geom_b, mis_b = run_ckpt('B', B_CKPT)
    print('geom B done, mismatches', len(mis_b), flush=True)
    geom_a, mis_a = run_ckpt('A', A_CKPT)
    print('geom A done, mismatches', len(mis_a), flush=True)

    # Flatten query geometry aligned to saved 299 membership (never replaced).
    rows = []
    for key in sorted(a_map):
        eid, qidx = key
        a_s, b_s = a_map[key], b_map[key]
        qa = next(r for r in geom_a[eid]['queries'] if r['query_index'] == qidx)
        qb = next(r for r in geom_b[eid]['queries'] if r['query_index'] == qidx)
        both_wrong = (not a_s['correct']) and (not b_s['correct'])
        rows.append({
            'episode_id': eid,
            'query_index': qidx,
            'k_shot': episodes[eid]['k_shot'],
            'true_class': a_s['true_class'],
            'A_pred_saved': a_s['pred_class'],
            'B_pred_saved': b_s['pred_class'],
            'A_recomputed': qa['pred_class'],
            'B_recomputed': qb['pred_class'],
            'A_match': int(qa['match_saved']),
            'B_match': int(qb['match_saved']),
            'both_wrong_saved': int(both_wrong),
            'A_margin': qa['margin'],
            'B_margin': qb['margin'],
            'A_s_true': qa['s_true'],
            'B_s_true': qb['s_true'],
            'A_cos_core_true': qa['cos_core_true'],
            'A_cos_ring_true': qa['cos_ring_true'],
            'A_cos_core_pred': qa['cos_core_pred'],
            'A_cos_ring_pred': qa['cos_ring_pred'],
            'B_cos_core_true': qb['cos_core_true'],
            'B_cos_ring_true': qb['cos_ring_true'],
            'B_cos_core_pred': qb['cos_core_pred'],
            'B_cos_ring_pred': qb['cos_ring_pred'],
            'beta': qa['beta'],
        })

    write_csv(OUT / 'geometry_queries.csv', rows, list(rows[0].keys()))
    write_json(OUT / 'geometry_mismatches.json', {'A': mis_a, 'B': mis_b})

    def summarize(flag, label, margin_key, core_t, ring_t, core_p, ring_p):
        sub = [r for r in rows if flag(r)]
        def mean(xs):
            return sum(xs) / len(xs) if xs else None
        return {
            'n': len(sub),
            'mean_margin': mean([r[margin_key] for r in sub]),
            'mean_cos_core_true': mean([r[core_t] for r in sub]),
            'mean_cos_ring_true': mean([r[ring_t] for r in sub]),
            'mean_cos_core_pred': mean([r[core_p] for r in sub]),
            'mean_cos_ring_pred': mean([r[ring_p] for r in sub]),
        }

    # Focus geometry for common-same target directions.
    focus_geom = {}
    for src, dst in list(FOCUS) + list(EXTRA_FOCUS):
        sub = [r for r in rows if r['both_wrong_saved'] and r['true_class'] == src
               and r['A_pred_saved'] == dst and r['B_pred_saved'] == dst]
        def mean(xs):
            return sum(xs) / len(xs) if xs else None
        focus_geom['{}->{}'.format(src, dst)] = {
            'n': len(sub),
            'A_mean_margin': mean([r['A_margin'] for r in sub]),
            'B_mean_margin': mean([r['B_margin'] for r in sub]),
            'A_mean_cos_core_true': mean([r['A_cos_core_true'] for r in sub]),
            'A_mean_cos_core_pred': mean([r['A_cos_core_pred'] for r in sub]),
            'B_mean_cos_core_true': mean([r['B_cos_core_true'] for r in sub]),
            'B_mean_cos_core_pred': mean([r['B_cos_core_pred'] for r in sub]),
        }

    # Support group offsets: how far group means sit from class mean, for true vs pred on common-same focus.
    offset_rows = []
    for ep in man['episodes']:
        for label, geom in (('A', geom_a), ('B', geom_b)):
            for block in geom[ep['episode_id']]['support_group_offsets']:
                for g in block['groups']:
                    offset_rows.append({
                        'episode_id': ep['episode_id'],
                        'k_shot': ep['k_shot'],
                        'model': label,
                        'class': block['class'],
                        'group_ns': g['group_ns'],
                        'n': g['n'],
                        'cos_to_class_mean_core': g['cos_to_class_mean_core'],
                        'cos_to_class_mean_ring': g['cos_to_class_mean_ring'],
                    })
    write_csv(OUT / 'geometry_group_offsets.csv', offset_rows, list(offset_rows[0].keys()))

    payload = {
        'score_definition': 's = cos_core + beta * cos_ring, beta=fixed 0.25, prototypes = L2-normalized class means of support core/ring (same as DualRegionBetaDetector.forward)',
        'margin_definition': 's_true - max_{c != true} s_c',
        'mismatches_A': len(mis_a),
        'mismatches_B': len(mis_b),
        'kept_common_error_set': 'saved predictions.json both-wrong membership; not replaced',
        'A_all': summarize(lambda r: True, 'A', 'A_margin', 'A_cos_core_true', 'A_cos_ring_true', 'A_cos_core_pred', 'A_cos_ring_pred'),
        'A_both_correct': summarize(lambda r: r['A_pred_saved'] == r['true_class'] and r['B_pred_saved'] == r['true_class'], 'A', 'A_margin', 'A_cos_core_true', 'A_cos_ring_true', 'A_cos_core_pred', 'A_cos_ring_pred'),
        'A_common_wrong': summarize(lambda r: r['both_wrong_saved'], 'A', 'A_margin', 'A_cos_core_true', 'A_cos_ring_true', 'A_cos_core_pred', 'A_cos_ring_pred'),
        'B_all': summarize(lambda r: True, 'B', 'B_margin', 'B_cos_core_true', 'B_cos_ring_true', 'B_cos_core_pred', 'B_cos_ring_pred'),
        'B_both_correct': summarize(lambda r: r['A_pred_saved'] == r['true_class'] and r['B_pred_saved'] == r['true_class'], 'B', 'B_margin', 'B_cos_core_true', 'B_cos_ring_true', 'B_cos_core_pred', 'B_cos_ring_pred'),
        'B_common_wrong': summarize(lambda r: r['both_wrong_saved'], 'B', 'B_margin', 'B_cos_core_true', 'B_cos_ring_true', 'B_cos_core_pred', 'B_cos_ring_pred'),
        'focus_common_same': focus_geom,
        'note': 'Group-mean cosine to class mean is a geometric offset, not proof that a batch pulled the prototype or caused the error.',
    }
    write_json(OUT / 'p0_geometry_stats.json', payload)
    return payload


def write_conclusion(data_stats, sheet_index, geom_stats):
    focus = data_stats['common_errors']['focus_counts']
    pool = data_stats['pooling_possibility_100_class_supports']
    ce = data_stats['common_errors']
    ref = data_stats['true_support_dominance_reference']
    lines = []
    lines.append('# P0 共同错误与 support group / 原型几何')
    lines.append('')
    lines.append('时间：{} Asia/Shanghai。范围仅 paired_eval 清单与预测，外加只推理几何。未跑 P1/P2，未改网络或训练。'.format(
        data_stats['created_at']))
    lines.append('')
    lines.append('## 已证实（数据级）')
    lines.append('')
    lines.append('- 299 是**共同错误 query 观测次数**，对应 **{} 张唯一 query 路径**（重复频次分布 {}，最多重复 {} 次）。不是 299 张互不重复的图。'.format(
        ce['n_unique_query_paths'], ce['repeat_freq_dist'], ce['max_repeat']))
    lines.append('- 其中 **{} 次 A/B 错成同一类**，**{} 次两边都错但预测类不同**。不能把 299 写成单一错向。'.format(
        ce['same_wrong_class'], ce['different_wrong_class']))
    lines.append('- 共同同错最多的方向是 inclusion→silk_spot **{}**，然后才是用户关注的三条：'.format(
        focus['inclusion->silk_spot']['common_same']))
    lines.append('  - oil→punch：A={} B={} 共同同错={}'.format(
        focus['oil->punch']['A'], focus['oil->punch']['B'], focus['oil->punch']['common_same']))
    lines.append('  - punch→weld_line：A={} B={} 共同同错={}'.format(
        focus['punch->weld_line']['A'], focus['punch->weld_line']['B'], focus['punch->weld_line']['common_same']))
    lines.append('  - rolled_pit→fold_wrinkle：A={} B={} 共同同错={}'.format(
        focus['rolled_pit->fold_wrinkle']['A'], focus['rolled_pit->fold_wrinkle']['B'],
        focus['rolled_pit->fold_wrinkle']['common_same']))
    lines.append('- GC10 的 group id 来自文件名正则 `img_\\d+_(\\d+)_\\d+`（配置与 `merge_sources.group_ids_for` 写明为文件名批次号），不是直方图聚类。本报告用 `source::class::id` 命名空间，不把跨类相同数字当成同一批次。')
    lines.append('- 数据 mask 只用于 crop，不是模型预测出的掩码。')
    lines.append('- 20 episode × 5 类 = 100 个 class-support：singleton={}，等量多组={}，不均等多组（P1 数学上才可能与普通均值不同）={}（占比 {:.0%}）。k=3：{}；k=5：{}。单组或各组张数相等时，组等权与普通均值数学等价，不必为这些格子做新实验。'.format(
        pool['counts'].get('singleton', 0),
        pool['counts'].get('equal_multi', 0),
        pool['counts'].get('uneven_multi', 0),
        pool['frac_pooling_can_differ'],
        pool['by_k'].get('3'),
        pool['by_k'].get('5'),
    ))
    lines.append('- 正确类 support 的 max_share 均值：全部 query {:.3f}；两边都对 {:.3f}；共同错误 {:.3f}。单组占比：全部 {:.3f}；都对 {:.3f}；共同错 {:.3f}。主导组在错误里并不明显更高，不能把共同错误归因于“组主导拉偏”。严格分组协议本身就会产生不少单组 support。'.format(
        ref['all_queries']['mean_max_share'],
        ref['both_correct']['mean_max_share'],
        ref['common_errors']['mean_max_share'],
        ref['all_queries']['frac_singleton_true_support'],
        ref['both_correct']['frac_singleton_true_support'],
        ref['common_errors']['frac_singleton_true_support'],
    ))
    lines.append('')
    lines.append('## 代表 contact sheet（预先固定每方向首个共同同错，不按外观挑选）')
    lines.append('')
    for item in sheet_index:
        lines.append('- {}：{}'.format(item['direction'], item.get('path') or item.get('note')))
    lines.append('照片只能说明可见的纹理/亮度差，不能从图片断言真实生产批次、工况或材质。source_table 工况编码按来源固定，GC10 内部不能据此分工况。')
    lines.append('')
    if geom_stats:
        lines.append('## 原型几何（只推理，membership 仍用原 299）')
        lines.append('')
        lines.append('- 分数定义：`s = cos_core + β·cos_ring`，β=0.25（fixed）。原型是该类 support core/ring 均值后再 L2 归一化，与 `DualRegionBetaDetector.forward` 一致。margin = `s_true - max_other`。')
        lines.append('- 与已保存预测不一致：A {} 条，B {} 条。共同错误集合未因重算而替换。'.format(
            geom_stats['mismatches_A'], geom_stats['mismatches_B']))
        ac = geom_stats['A_both_correct']
        aw = geom_stats['A_common_wrong']
        bc = geom_stats['B_both_correct']
        bw = geom_stats['B_common_wrong']
        lines.append('- A margin 均值：都对 {:.3f}，共同错 {:.3f}；B：都对 {:.3f}，共同错 {:.3f}。'.format(
            ac['mean_margin'], aw['mean_margin'], bc['mean_margin'], bw['mean_margin']))
        lines.append('- A 共同错：cos_core 真类 {:.3f} vs 预测类 {:.3f}；B：真类 {:.3f} vs 预测类 {:.3f}。'.format(
            aw['mean_cos_core_true'], aw['mean_cos_core_pred'],
            bw['mean_cos_core_true'], bw['mean_cos_core_pred']))
        for k, v in geom_stats['focus_common_same'].items():
            lines.append('- {} 共同同错 n={}：A margin {:.3f}（core 真/预 {:.3f}/{:.3f}）；B margin {:.3f}（core 真/预 {:.3f}/{:.3f}）。'.format(
                k, v['n'], v['A_mean_margin'], v['A_mean_cos_core_true'], v['A_mean_cos_core_pred'],
                v['B_mean_margin'], v['B_mean_cos_core_true'], v['B_mean_cos_core_pred']))
        lines.append('- 组均值相对类均值的余弦偏移已写入 geometry_group_offsets.csv。这是几何描述，不能直接说“该批次把原型拉偏并导致错误”。')
    else:
        lines.append('## 原型几何')
        lines.append('')
        lines.append('本文件写于数据阶段，几何尚未跑完或未运行。')
    lines.append('')
    lines.append('## 假说 vs 未做')
    lines.append('')
    lines.append('- 假说（未证实）：不均等多组 support 上，组等权会改善油污/冲孔。P1 的**必要可能性**存在（100 格中 {} 格数学上可能与普通均值不同，k=5 更集中），但主导组在错误与正确参照上没有清楚分离，**不是充分理由断言 P1 会改善**。'.format(
        pool['counts'].get('uneven_multi', 0)))
    lines.append('- P1 建议条件：仅当需要直接检验该操作时再做，且必须同 episode、同特征、只改聚合；单组格子应事先知道结果不变。即使 P1 无改善，也只能说对“组等权这一操作”无证据，不能排除层特征相近或类别定义重叠。')
    lines.append('- P2（按 episode 配对 bootstrap）只应在 P1 真的跑了对比之后，用来约束 20 任务单种子的不确定性；本次未执行。')
    lines.append('- 未执行：组等权评估、训练、改网络、bootstrap。')
    text = '\n'.join(lines) + '\n'
    (OUT / 'p0_conclusion.md').write_text(text, encoding='utf-8')
    return OUT / 'p0_conclusion.md'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['data', 'sheets', 'geom', 'all'], default='all')
    return p.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data_stats = sheet_index = geom_stats = None
    if args.phase in ('data', 'sheets', 'geom', 'all'):
        data_stats, man, pred, a_map, b_map, episodes, data_root, first_examples = data_phase()
        print('data phase written', OUT / 'p0_data_stats.json',
              'common', data_stats['common_errors']['n_observations'],
              'unique', data_stats['common_errors']['n_unique_query_paths'],
              'pooling_can_differ', data_stats['pooling_possibility_100_class_supports']['frac_pooling_can_differ'],
              flush=True)
    if args.phase in ('sheets', 'all'):
        sheet_index = sheets_phase(man, a_map, b_map, episodes, data_root, first_examples)
        print('sheets', sheet_index, flush=True)
    if args.phase in ('geom', 'all'):
        geom_stats = geom_phase(man, a_map, b_map, episodes)
        print('geom mismatches A/B', geom_stats['mismatches_A'], geom_stats['mismatches_B'], flush=True)
    if data_stats is not None:
        path = write_conclusion(data_stats, sheet_index or [], geom_stats)
        print('conclusion', path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
