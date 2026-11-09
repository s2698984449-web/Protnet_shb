"""工业双区域目标域原型：冻结 layer4 + cos(core)+β·cos(ring)。

流程：
1) 铸坯开发集在 β∈{0.25,0.5} 上按 k=3 的 Δhard 选一次；接近则取 0.25；
2) 锁定 β 后在 CR7/GC10/NEU 上报 k=3/5 的缺陷准确率、Δbg、Δhard、混淆；
3) 跨来源支持仅附录。不训练、不调距离、不用查询标签。
"""

import numpy as np
from torchvision import transforms

from .backbone import SpatialResNet18
from .frozen_contrast import (
    classify_branch,
    classify_combo,
    encode_core_context,
)
from .target_shot import (
    MEAN,
    STD,
    assign_batches,
    bootstrap_images,
    collect_target_items,
    power_from_ci,
    sample_support,
    sample_support_cross,
)


def _empty_conf(classes):
    return {t: {p: 0 for p in classes} for t in classes}


def _add_conf(conf, truth, pred):
    for image, y in truth.items():
        p = pred.get(image)
        if p is None or y not in conf or p not in conf[y]:
            continue
        conf[y][p] += 1


def _conf_rate(conf):
    out = {}
    for t, row in conf.items():
        total = sum(row.values())
        out[t] = {p: (round(v / total, 4) if total else 0.0) for p, v in row.items()}
    return out


def evaluate_dual(visual, classes, cfg, k, beta, seed_offset=0, support_pool=None):
    """score = (1-cos_core) + β*(1-cos_ring)；背景/几何支路仍用 cosine / L2。"""
    assign_batches(visual, int(cfg['n_clusters']), int(cfg['seed']) + seed_offset, 'stats')
    if support_pool is not None:
        assign_batches(
            support_pool, int(cfg['n_clusters']), int(cfg['seed']) + 7 + seed_offset, 'stats')
    py_rng = __import__('random').Random(int(cfg['seed']) + 17 + seed_offset + 10 * int(k))
    support_reps = int(cfg.get('support_repeats', 20))
    boot_reps = int(cfg.get('bootstrap', 200))
    beta = float(beta)
    hits = {row['image']: [] for row in visual}
    conf = _empty_conf(classes)
    used = 0

    for _ in range(support_reps):
        if support_pool is None:
            support, query = sample_support(visual, classes, k, py_rng)
        else:
            support, query = sample_support_cross(visual, support_pool, classes, k, py_rng)
        if support is None:
            continue
        # alpha=1, beta=β：距离加权；classify_combo 用 1-cos
        truth_d, pred_d = classify_combo(
            support, query, classes, 'cosine', alpha=1.0, beta=beta)
        truth_b, pred_b = classify_branch(support, query, 'outside', classes, 'cosine')
        truth_g, pred_g = classify_branch(support, query, 'geometry', classes, 'l2')
        _add_conf(conf, truth_d, pred_d)
        for image, y in truth_d.items():
            if image not in pred_b or image not in pred_g:
                continue
            hits[image].append((
                int(pred_d[image] == y),
                int(pred_b[image] == y),
                int(pred_g[image] == y),
            ))
        used += 1

    rows = []
    for image, vals in hits.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rows.append({
            'image': image,
            'defect': float(arr[:, 0].mean()),
            'bg': float(arr[:, 1].mean()),
            'geo': float(arr[:, 2].mean()),
        })
    metrics = bootstrap_images(rows, boot_reps, int(cfg['seed']) + 202 + seed_offset)
    metrics['power_bg'] = power_from_ci(metrics.get('delta_ci'))
    metrics['power_hard'] = power_from_ci(metrics.get('delta_vs_hard_ci'))
    return {
        'beta': beta,
        'k': k,
        'support_mode': 'cross' if support_pool is not None else 'same',
        'support_repeats_used': used,
        'n_eval_images': len(rows),
        'metrics_image_boot': metrics,
        'confusion_counts': conf,
        'confusion_rate': _conf_rate(conf),
    }


def _pack(block):
    m = block.get('metrics_image_boot') or {}
    return {
        'beta': block.get('beta'),
        'k': block.get('k'),
        'support_mode': block.get('support_mode'),
        'defect_acc': m.get('model_acc'),
        'defect_ci': m.get('model_ci'),
        'bg_acc': m.get('outside_acc'),
        'geo_acc': m.get('geometry_acc'),
        'delta_bg': m.get('delta'),
        'delta_bg_ci': m.get('delta_ci'),
        'pass_vs_bg': m.get('pass_vs_bg'),
        'delta_hard': m.get('delta_vs_hard'),
        'delta_hard_ci': m.get('delta_vs_hard_ci'),
        'pass_vs_hard': m.get('pass_vs_hard'),
        'power_bg': m.get('power_bg'),
        'power_hard': m.get('power_hard'),
        'n_eval_images': block.get('n_eval_images'),
        'support_repeats_used': block.get('support_repeats_used'),
        'confusion_rate': block.get('confusion_rate'),
    }


def select_beta(dev_visual, classes, cfg, candidates, select_k=3, check_k=5):
    """铸坯选 β：主看 select_k 的 Δhard；接近则 0.25；并报 check_k 防反转。"""
    candidates = [float(x) for x in candidates]
    prefer = float(cfg.get('beta_prefer_when_close', 0.25))
    close_eps = float(cfg.get('beta_close_eps', 0.02))
    grid = {}
    for i, beta in enumerate(candidates):
        print('.. select CAST k', select_k, 'beta', beta, flush=True)
        ev = evaluate_dual(
            dev_visual, classes, cfg, select_k, beta, seed_offset=10 * i)
        grid[str(beta)] = {'k3': _pack(ev)}
        print('.. select CAST k', check_k, 'beta', beta, flush=True)
        ev5 = evaluate_dual(
            dev_visual, classes, cfg, check_k, beta, seed_offset=100 + 10 * i)
        grid[str(beta)]['k5'] = _pack(ev5)

    scores = {b: (grid[str(b)]['k3'].get('delta_hard') or -1.0) for b in candidates}
    best = max(candidates, key=lambda b: scores[b])
    other = [b for b in candidates if b != best]
    locked = best
    reason = 'higher_delta_hard_k3'
    if other:
        gap = abs(scores[best] - scores[other[0]])
        mde_best = ((grid[str(best)]['k3'].get('power_hard') or {}).get('mde80_paired'))
        close = gap <= close_eps or (mde_best is not None and gap <= float(mde_best))
        if close and prefer in candidates:
            locked = prefer
            reason = 'close_prefer_{}'.format(prefer)

    # k=5 是否相对 k=3 排序反转
    order3 = sorted(candidates, key=lambda b: scores[b], reverse=True)
    scores5 = {b: (grid[str(b)]['k5'].get('delta_hard') or -1.0) for b in candidates}
    order5 = sorted(candidates, key=lambda b: scores5[b], reverse=True)
    reversal = order3 != order5

    return {
        'candidates': candidates,
        'select_k': select_k,
        'check_k': check_k,
        'metric': 'delta_hard',
        'scores_k3': {str(b): scores[b] for b in candidates},
        'scores_k5': {str(b): scores5[b] for b in candidates},
        'locked_beta': locked,
        'reason': reason,
        'gap_k3': round(abs(scores[candidates[0]] - scores[candidates[1]]), 4)
        if len(candidates) == 2 else None,
        'order_reversal_k3_vs_k5': bool(reversal),
        'grid': grid,
    }


def run_industrial_proto(cfg, device):
    model = SpatialResNet18(pretrained=True, freeze=True).to(device).eval()
    dilate_px = int(cfg.get('context_dilate_px', 16))
    select_source = str(cfg.get('select_source', 'CAST'))
    denom = list(cfg.get('denominator') or ['CR7', 'GC10', 'NEU'])
    ks = [int(x) for x in (cfg.get('primary_ks') or [3, 5])]
    candidates = list(cfg.get('beta_candidates') or [0.25, 0.5])
    run_cross = bool(cfg.get('run_cross_appendix', True))
    size = int(cfg['image_size'])
    resize = transforms.Resize((size, size))
    normalize = transforms.Normalize(MEAN, STD)

    encode_list = list(dict.fromkeys([select_source] + denom))
    cache = {}
    for source in encode_list:
        print('.. encode', source, flush=True)
        items, classes, skipped = collect_target_items(cfg, source)
        if len(classes) < 2 or not items:
            cache[source] = {'visual': None, 'classes': classes, 'skipped': skipped}
            continue
        visual = encode_core_context(
            model, items, resize, normalize, device, int(cfg['seed']), dilate_px=dilate_px)
        cache[source] = {'visual': visual, 'classes': classes, 'skipped': skipped}

    report = {
        'protocol': 'industrial_dual_proto_v1',
        'feature': 'frozen_imagenet_layer4_core_ring',
        'score': 'd_core + beta * d_ring  (d=1-cos)',
        'context_dilate_px': dilate_px,
        'primary_metric_select': 'delta_hard @ k=3 on CAST',
        'delivery_metrics': ['defect_acc', 'delta_bg', 'delta_hard', 'confusion_rate'],
        'notes': [
            '不训练；远背景不进类原型与分数；支持/查询按伪批次分离',
            'β 只在铸坯开发集选一次；分母不参与选参',
            '跨来源支持仅附录',
        ],
        'selection': {},
        'delivery': {},
        'appendix_cross': {},
    }

    sel = cache.get(select_source) or {}
    if sel.get('visual') is None:
        report['selection'] = {'error': 'select_source_insufficient', 'skipped': sel.get('skipped')}
        return report

    selection = select_beta(
        sel['visual'], sel['classes'], cfg, candidates,
        select_k=int(cfg.get('select_k', 3)),
        check_k=int(cfg.get('check_k', 5)))
    report['selection'] = selection
    locked = float(selection['locked_beta'])
    print('== locked_beta', locked, selection['reason'], flush=True)

    # 交付表：锁定 β × 分母 × k
    for source in denom:
        pack = cache.get(source) or {}
        visual, classes, skipped = pack.get('visual'), pack.get('classes'), pack.get('skipped')
        if visual is None:
            report['delivery'][source] = {'error': 'insufficient', 'skipped': skipped}
            continue
        block = {
            'role': 'test_denominator',
            'n_images': len(visual),
            'classes': classes,
            'skipped': skipped,
            'locked_beta': locked,
            'by_k': {},
        }
        for i, k in enumerate(ks):
            print('.. delivery', source, 'k', k, 'beta', locked, flush=True)
            ev = evaluate_dual(
                visual, classes, cfg, k, locked, seed_offset=1000 + 10 * i)
            block['by_k'][str(k)] = _pack(ev)
        # 验收摘录
        k3 = block['by_k'].get('3') or {}
        block['acceptance'] = {
            'delta_bg_positive': bool(k3.get('pass_vs_bg')),
            'delta_hard_pass': bool(k3.get('pass_vs_hard')),
            'delta_hard': k3.get('delta_hard'),
            'delta_hard_ci': k3.get('delta_hard_ci'),
            'mde_hard': (k3.get('power_hard') or {}).get('mde80_paired'),
        }
        report['delivery'][source] = block

    # 附录：跨来源支持（锁定 β，k=3）
    if run_cross:
        for source in denom:
            pack = cache.get(source) or {}
            visual, classes = pack.get('visual'), pack.get('classes')
            if visual is None:
                continue
            support_pool = []
            for other in encode_list:
                if other == source:
                    continue
                other_v = (cache.get(other) or {}).get('visual')
                if other_v:
                    support_pool.extend(other_v)
            shared = [
                c for c in classes
                if sum(1 for r in support_pool if r['class'] == c) >= 3
                and sum(1 for r in visual if r['class'] == c) >= 1
            ]
            if len(shared) < 2:
                report['appendix_cross'][source] = {'skipped': True, 'shared': shared}
                continue
            print('.. appendix cross', source, 'shared', shared, flush=True)
            ev = evaluate_dual(
                visual, shared, cfg, 3, locked,
                seed_offset=2000, support_pool=support_pool)
            report['appendix_cross'][source] = {
                'shared': shared,
                'k': 3,
                'locked_beta': locked,
                **_pack(ev),
            }

    # 总验收
    hard_pass = []
    bg_pass = []
    for source in denom:
        acc = (report['delivery'].get(source) or {}).get('acceptance') or {}
        if 'delta_bg_positive' in acc:
            bg_pass.append(bool(acc['delta_bg_positive']))
        if 'delta_hard_pass' in acc:
            hard_pass.append(bool(acc['delta_hard_pass']))
    report['gate'] = {
        'beta_no_reversal': not bool(selection.get('order_reversal_k3_vs_k5')),
        'denom_delta_bg_positive_count': sum(bg_pass),
        'denom_delta_hard_pass_count': sum(hard_pass),
        'need_hard_pass_at_least': 2,
        'hard_pass_sources': {
            s: ((report['delivery'].get(s) or {}).get('acceptance') or {}).get('delta_hard_pass')
            for s in denom
        },
        'neu_hard_ci_crosses_zero': None,
    }
    neu = (report['delivery'].get('NEU') or {}).get('by_k', {}).get('3') or {}
    ci = neu.get('delta_hard_ci')
    if ci and len(ci) == 2:
        report['gate']['neu_hard_ci_crosses_zero'] = bool(ci[0] <= 0 <= ci[1])
    return report
