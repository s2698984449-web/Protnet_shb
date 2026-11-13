"""运行:
  python scripts/train_cspd.py --config config/cspd/cspd.yaml
  python scripts/train_cspd.py --config config/cspd/curriculum_full.yaml

train_mode=curriculum 时读 data/new_split，课程采样 + dual-region β。
输出在 output/<exp_name>/。
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cspd.data import curriculum_rho, describe_config, make_curriculum, make_episodes
from cspd.hard_pairs import (HardPairBuffer, hard_probability, metric_bucket,
                             record_predictions, serial_metrics)
from cspd.runtime import (
    build_model,
    episode_loss,
    evaluate,
    forward_episode,
    move_episode,
    output_dir,
    pick_device,
    set_seed,
    write_config,
    write_structure,
)


def setup_stdio():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


class RunLog:
    def __init__(self, path):
        self.path = path

    def line(self, text):
        print(text, flush=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(text + '\n')


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def _set_policy_trainable(model, enabled):
    if not hasattr(model, 'policy'):
        return
    for param in model.policy.parameters():
        param.requires_grad = bool(enabled)


@torch.no_grad()
def fit_source_distance(model, sampler, device, n_images=8):
    """冻结 layer4 上每个来源的均值，得到来源间余弦距离。不消耗训练随机数。"""
    probe = random.Random(0)
    vectors = {}
    was_training = model.training
    model.eval()
    for src in sampler.sources:
        pool = []
        for groups in sampler.by_class.values():
            pool.extend(groups.get(src, []))
        if not pool:
            continue
        pick = probe.sample(pool, min(n_images, len(pool)))
        batch = torch.stack([sampler._load(path, mask, src)['x'] for path, mask in pick], 0)
        _, layer4 = model.backbone(batch.to(device))
        vectors[src] = layer4.mean(dim=(2, 3)).mean(dim=0)
    if was_training:
        model.train()
    names = list(vectors)
    if len(names) < 2:
        return {}
    mat = torch.stack([vectors[name] for name in names])
    mat = mat / mat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sim = mat @ mat.t()
    dist = {}
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if i != j:
                dist[(left, right)] = float(1.0 - sim[i, j])
    return dist


@torch.no_grad()
def evaluate_curriculum(model, sampler, device, tasks, cross_source):
    """固定同来源或跨来源，不跟训练进度走。"""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    losses = []
    betas = []
    per_class = {}
    confusion = {}
    skipped = 0
    by_source = {}
    skip_reasons = {}
    for _ in range(tasks):
        try:
            episode = move_episode(
                sampler.sample(
                    cross_source=cross_source, stage_sources=False, allow_fallback=False), device)
        except RuntimeError as exc:
            # 例如 v3 留出 CR7 后冲孔只剩一家，跨来源抽不到。
            skipped += 1
            skip_reasons[str(exc)] = skip_reasons.get(str(exc), 0) + 1
            continue
        ways = int(episode['n_way'])
        pred = forward_episode(model, episode, ways)
        cls = F.cross_entropy(pred['logits'], episode['query_y'])
        pred_y = pred['logits'].argmax(dim=-1)
        y = episode['query_y']
        names = list(episode['class_names'])
        source_bucket = by_source.setdefault(episode['query_source'], metric_bucket())
        record_predictions(source_bucket, names, y.detach().cpu().tolist(),
                           pred_y.detach().cpu().tolist())
        hits = pred_y == y
        correct += int(hits.sum().item())
        total += int(y.numel())
        losses.append(float(cls))
        if 'beta' in pred:
            betas.append(float(pred['beta'].detach()))
        for i in range(int(y.numel())):
            true_name = names[int(y[i])]
            pred_name = names[int(pred_y[i])]
            slot = per_class.setdefault(true_name, [0, 0])
            slot[1] += 1
            slot[0] += int(hits[i].item())
            if true_name != pred_name:
                key = (true_name, pred_name)
                confusion[key] = confusion.get(key, 0) + 1
    if was_training:
        model.train()
    return {
        'acc': correct / max(total, 1),
        'loss': sum(losses) / max(len(losses), 1) if losses else 0.0,
        'beta': (sum(betas) / len(betas)) if betas else None,
        'per_class': per_class,
        'confusion': confusion,
        'skipped': skipped,
        'skip_reasons': skip_reasons,
        'query_count': total,
        'by_source': by_source,
        'group_failures': dict(sampler.group_failures),
        'source_attempts': dict(getattr(sampler, 'source_attempts', {})),
        'source_successes': dict(getattr(sampler, 'source_successes', {})),
    }


def _recall_text(per_class):
    parts = []
    for name in sorted(per_class):
        hit, count = per_class[name]
        parts.append('{} {:.2f}'.format(name, hit / max(count, 1)))
    return ' '.join(parts) if parts else 'na'


def _take_hard_pairs(confusion, limit=3):
    ranked = sorted(confusion.items(), key=lambda item: item[1], reverse=True)
    return [pair for pair, count in ranked[:limit] if count >= 2]


def train_curriculum(cfg, log, out, device, seed):
    train_sampler = make_curriculum(cfg, 'train', seed)
    new_protocol = bool(cfg.get('hard_pair_protocol', False))
    buffer = HardPairBuffer(train_sampler.sources, str(cfg.get('holdout_source') or ''))
    if new_protocol:
        # Keep protocol val/buffer logs, but never arm the legacy hard_pairs hook.
        train_sampler.hard_pairs = []
    if new_protocol and (int(cfg['eval_every']) != 5 or bool(cfg.get('early_stop', False))):
        raise ValueError('hard-pair protocol requires eval_every=5 and early_stop=false')
    val_sampler = make_curriculum(cfg, 'val', seed + 2)
    test_cfg = dict(cfg)
    # 测试：若 holdout，查询可含该来源；训练排除已在 sampler 内处理
    test_cfg = dict(cfg)
    test_sampler = make_curriculum(test_cfg, 'test', seed + 1)
    summary = {
        'train_sources': train_sampler.sources,
        'train_classes': train_sampler.classes,
        'cross_ok': sorted(train_sampler.cross_ok),
        'val_sources': val_sampler.sources,
        'test_sources': test_sampler.sources,
        'source_class_counts': {src: len(names) for src, names in train_sampler.source_classes.items()},
        'n_way_cap': train_sampler.n_way_cap,
        'composition_saturated_sources': [src for src, names in train_sampler.source_classes.items()
                                          if len(names) <= train_sampler.n_way_cap],
        'strict_groups': train_sampler.strict_groups,
    }
    (out / 'data_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.line('curriculum sources {}'.format(train_sampler.sources))
    log.line('cross_ok {}'.format(sorted(train_sampler.cross_ok)))

    model = build_model(cfg).to(device)
    write_structure(model, out)
    train_sampler.source_dist = fit_source_distance(model, train_sampler, device)
    if train_sampler.source_dist:
        near = min(train_sampler.source_dist.items(), key=lambda item: item[1])
        far = max(train_sampler.source_dist.items(), key=lambda item: item[1])
        log.line('source dist near {} {:.3f} far {} {:.3f}'.format(
            near[0], near[1], far[0], far[1]))
    freeze_policy_until = float(cfg.get('policy_unfreeze_frac', 0.25))
    use_policy = bool(cfg.get('use_policy', True))
    if use_policy:
        _set_policy_trainable(model, False)
    policy_params = list(model.policy.parameters()) if hasattr(model, 'policy') else []
    head_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith('policy.')
    ]
    lr = float(cfg['lr'])
    groups = [{'params': head_params, 'lr': lr}]
    if policy_params:
        groups.append({
            'params': policy_params,
            'lr': lr * float(cfg.get('policy_lr_ratio', 0.3)),
        })
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=float(cfg['weight_decay']))
    log.line('trainable params {}'.format(sum(p.numel() for p in trainable_parameters(model))))

    iters = int(cfg['iters'])
    meta_batch = int(cfg.get('meta_batch', 4))
    grad_accum = max(1, int(cfg.get('grad_accum', 1)))
    grad_clip = float(cfg.get('grad_clip', 1.0))
    eval_every = int(cfg['eval_every'])
    schedule = cfg.get('rho_schedule')
    early = bool(cfg.get('early_stop', False))
    patience = int(cfg.get('patience', 10))
    early_start = int(cfg.get('early_start', 0))
    min_delta = float(cfg.get('min_delta', 0.0))
    monitor = str(cfg.get('monitor', 'loss'))
    best_value = None
    best_iter = 0
    wait = 0
    policy_on = False
    hard_totals = {}
    validation_history = []

    for it in range(1, iters + 1):
        progress = (it - 1) / max(iters - 1, 1)
        rho = curriculum_rho(progress, schedule)
        train_sampler.hard_probability = (
            hard_probability(it) if (new_protocol and train_sampler.hard_pair_enabled) else 0.0)
        hard_events = []
        if use_policy and (not policy_on) and progress >= freeze_policy_until:
            _set_policy_trainable(model, True)
            policy_on = True
            log.line('policy unfrozen at iter {:04d}'.format(it))
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_intra = []
        loss_cross = []
        accs = []
        betas = []
        used_sources = set()
        for step in range(meta_batch):
            cross = train_sampler.rng.random() < rho
            episode = move_episode(train_sampler.sample(
                cross_source=cross,
                progress=progress,
                avoid_sources=used_sources,
                stage_sources=True,
            ), device)
            source = episode['query_source']
            info = episode.get('hard_pair')
            if info is not None:
                hard_events.append(dict(source=source, **info))
                counts = hard_totals.setdefault(source, dict(episodes=0, requested=0, applied=0,
                                                             changed=0, fallback=0, saturated=0))
                counts['episodes'] += 1
                for key in ('requested', 'applied', 'changed', 'fallback', 'saturated'):
                    counts[key] += int(info[key])
            used_sources.add(source)
            if len(used_sources) >= len(train_sampler.sources):
                used_sources = set()
            ways = int(episode['n_way'])
            pred = forward_episode(model, episode, ways)
            cls = F.cross_entropy(pred['logits'], episode['query_y'])
            (cls / float(meta_batch * grad_accum)).backward()
            if episode['cross_source']:
                loss_cross.append(float(cls))
            else:
                loss_intra.append(float(cls))
            accs.append(float((pred['logits'].argmax(-1) == episode['query_y']).float().mean()))
            if 'beta' in pred:
                betas.append(float(pred['beta'].detach()))
            if (step + 1) % grad_accum == 0 or step + 1 == meta_batch:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
        mean_intra = sum(loss_intra) / len(loss_intra) if loss_intra else 0.0
        mean_cross = sum(loss_cross) / len(loss_cross) if loss_cross else 0.0
        n_ep = len(loss_intra) + len(loss_cross)
        gloss = (sum(loss_intra) + sum(loss_cross)) / n_ep if n_ep else 0.0
        got = len(loss_cross) / float(meta_batch)
        beta_now = sum(betas) / len(betas) if betas else float(cfg.get('fixed_beta', 0.25))
        log.line(
            'iter {:04d} rho {:.2f} got {:.2f} loss {:.4f} intra {:.4f} cross {:.4f} acc {:.3f} beta {:.3f} policy {}'.format(
                it, rho, got, gloss, mean_intra, mean_cross,
                sum(accs) / max(len(accs), 1),
                beta_now,
                int(policy_on and use_policy)))
        if new_protocol:
            with (out / 'hard_pairs.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'iter': it, 'probability': train_sampler.hard_probability,
                                         'episodes': hard_events,
                                         'group_failures': train_sampler.group_failures,
                                         'source_attempts': train_sampler.source_attempts,
                                         'source_successes': train_sampler.source_successes}) + '\n')
        if it % eval_every == 0 or it == iters:
            tasks = int(cfg['eval_tasks'])
            same_m = evaluate_curriculum(model, val_sampler, device, tasks, False)
            cross_m = evaluate_curriculum(model, val_sampler, device, tasks, True)
            if new_protocol:
                report = buffer.update(same_m['by_source'])
                train_sampler.same_hard_pairs = buffer.active if train_sampler.hard_pair_enabled else {}
                row = {'iter': it, 'buffer': report,
                       'same_validation': serial_metrics(same_m)}
                validation_history.append(row)
                with (out / 'hard_pairs.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(serial_metrics(row)) + '\n')
            else:
                train_sampler.hard_pairs = _take_hard_pairs(cross_m['confusion'])
            log.line(
                'eval val iter {:04d} same_acc {:.3f} same_loss {:.4f} cross_acc {:.3f} cross_loss {:.4f}'.format(
                    it, same_m['acc'], same_m['loss'], cross_m['acc'], cross_m['loss']))
            if not early:
                continue
            higher = monitor in ('acc', 'cross_acc', 'same_acc')
            if monitor in ('same_loss', 'same_acc'):
                value = same_m['acc'] if higher else same_m['loss']
            else:
                value = cross_m['acc'] if higher else cross_m['loss']
            if higher:
                improved = best_value is None or value > best_value + min_delta
            else:
                improved = best_value is None or value < best_value - min_delta
            if improved:
                best_value = value
                best_iter = it
                wait = 0
                torch.save(
                    {
                        'model': model.state_dict(),
                        'config': cfg,
                        'iter': it,
                        'metrics': {
                            'same_acc': same_m['acc'],
                            'same_loss': same_m['loss'],
                            'cross_acc': cross_m['acc'],
                            'cross_loss': cross_m['loss'],
                        },
                    },
                    out / 'best.pt')
                log.line('best iter {:04d} {} {:.4f}'.format(it, monitor, value))
            elif it < early_start:
                log.line('early hold until {}'.format(early_start))
            else:
                wait += 1
                log.line('early wait {}/{}'.format(wait, patience))
                if wait >= patience:
                    log.line('early stop at iter {:04d}, best iter {:04d}'.format(it, best_iter))
                    break

    torch.save({'model': model.state_dict(), 'config': cfg}, out / 'last.pt')
    log.line('saved {}'.format(out / 'last.pt'))
    if best_iter:
        log.line('use {}'.format(out / 'best.pt'))
        try:
            state = torch.load(out / 'best.pt', map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(out / 'best.pt', map_location=device)
        model.load_state_dict(state['model'])
    else:
        log.line('no early stop, use last')
    tasks = int(cfg['eval_tasks'])
    same_m = evaluate_curriculum(model, test_sampler, device, tasks, False)
    cross_m = evaluate_curriculum(model, test_sampler, device, tasks, True)
    log.line('test same acc {:.3f} loss {:.4f} recall {}'.format(
        same_m['acc'], same_m['loss'], _recall_text(same_m['per_class'])))
    log.line('test cross acc {:.3f} loss {:.4f} recall {}'.format(
        cross_m['acc'], cross_m['loss'], _recall_text(cross_m['per_class'])))
    result = {'config': cfg, 'train_sources': train_sampler.sources,
              'test_same': serial_metrics(same_m), 'test_cross': serial_metrics(cross_m),
              'hard_sampling': hard_totals, 'validation': validation_history,
              'train_group_failures': train_sampler.group_failures,
              'train_source_attempts': train_sampler.source_attempts,
              'train_source_successes': train_sampler.source_successes,
              'status': 'evaluated', 'holdout': None}
    hold_name = str(cfg.get('holdout_source') or '').strip()
    if hold_name:
        hold_cfg = dict(cfg)
        hold_cfg['query_only'] = hold_name
        hold_cross = str(cfg.get('holdout_eval') or 'cross') != 'same'
        try:
            hold_sampler = make_curriculum(hold_cfg, 'test', seed + 3)
            hold_m = evaluate_curriculum(model, hold_sampler, device, tasks, hold_cross)
            result['holdout'] = serial_metrics(hold_m)
            if new_protocol and not hold_m['query_count']:
                raise RuntimeError('no valid holdout queries under strict groups')
            log.line('test holdout {} acc {:.3f} loss {:.4f} recall {}'.format(
                hold_name, hold_m['acc'], hold_m['loss'], _recall_text(hold_m['per_class'])))
        except RuntimeError as exc:
            log.line('test holdout {} skip {}'.format(hold_name, exc))
            if new_protocol:
                raise
    (out / 'metrics.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def train_legacy(cfg, log, out, device, seed):
    data_report = describe_config(cfg, seed)
    (out / 'data_summary.json').write_text(
        json.dumps(data_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for key, item in data_report.items():
        log.line('{} classes {} images {} mask_ratio {}'.format(
            key,
            len(item['classes']),
            sum(sum(groups.values()) for groups in item['counts'].values()),
            item['episode'].get('support_has_mask_ratio')))

    train_sampler = make_episodes(cfg, 'train_dir', seed)
    test_sampler = make_episodes(cfg, 'test_dir', seed + 1)
    model = build_model(cfg).to(device)
    structure = write_structure(model, out)
    summary = '\n'.join(structure.splitlines()[:1 + len(list(model.named_children()))])
    log.line('structure')
    log.line(summary)
    log.line('full structure {}'.format(out / 'structure.txt'))

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_params.append(param)
        else:
            head_params.append(param)
    lr = float(cfg['lr'])
    if backbone_params and cfg.get('backbone_lr') not in (None, ''):
        groups = [
            {'params': head_params, 'lr': lr},
            {'params': backbone_params, 'lr': float(cfg['backbone_lr'])},
        ]
    else:
        groups = head_params + backbone_params
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=float(cfg['weight_decay']))
    log.line('trainable params {}'.format(sum(p.numel() for p in trainable_parameters(model))))
    log.line('classes {}'.format(train_sampler.classes))

    iters = int(cfg['iters'])
    eval_every = int(cfg['eval_every'])
    grad_clip = float(cfg['grad_clip'])
    loc_w = float(cfg.get('location_loss_weight', 0.0))
    src_w = float(cfg.get('source_proto_weight', 0.0))
    early = bool(cfg.get('early_stop', False))
    patience = int(cfg.get('patience', 10))
    early_start = int(cfg.get('early_start', 0))
    min_delta = float(cfg.get('min_delta', 0.0))
    monitor = str(cfg.get('monitor', 'loss'))
    if monitor not in ('loss', 'acc'):
        raise ValueError('monitor 只能是 loss 或 acc')
    if early:
        if not str(cfg.get('val_dir') or '').strip():
            raise ValueError('early_stop=true 时必须配置独立的 val_dir')
        stop_sampler = make_episodes(cfg, 'val_dir', seed + 2)
        stop_name = 'val_dir'
        log.line('early stop on {} monitor {} patience {} start {}'.format(
            stop_name, monitor, patience, early_start))
    else:
        stop_sampler = test_sampler
        stop_name = 'test_dir'

    best_value = None
    best_iter = 0
    wait = 0
    for it in range(1, iters + 1):
        model.train()
        episode = move_episode(train_sampler.sample(), device)
        pred = forward_episode(model, episode, train_sampler.n_way)
        loss, cls_loss, loc_loss, align_loss = episode_loss(
            pred, episode, loc_w, src_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), grad_clip)
        opt.step()
        acc = (pred['logits'].argmax(-1) == episode['query_y']).float().mean().item()
        log.line(
            'iter {:04d} loss {:.4f} cls {:.4f} align {:.4f} loc {:.4f} acc {:.3f} g {:.3f} alpha {}'.format(
                it, float(loss), float(cls_loss), float(align_loss), float(loc_loss), acc,
                float(pred['gate_g'].mean()),
                [round(float(v), 3) for v in pred['scale_alpha'].detach().mean(0)]))
        if it % eval_every == 0 or it == iters:
            metrics = evaluate(
                model, stop_sampler, device, int(cfg['eval_tasks']), loc_w, src_w)
            line = 'eval {} iter {:04d} acc {:.3f} loss {:.4f} cls {:.4f} align {:.4f} loc {:.4f} g {:.3f}'.format(
                stop_name, it, metrics['acc'], metrics['loss'], metrics['cls_loss'],
                metrics['align_loss'], metrics['loc_loss'], metrics['gate_g'])
            if metrics.get('loc_recall@0.2') is not None:
                line += ' loc@0.2 {:.3f}'.format(metrics['loc_recall@0.2'])
            log.line(line)
            if not early:
                continue
            value = metrics['acc'] if monitor == 'acc' else metrics['loss']
            if monitor == 'acc':
                improved = best_value is None or value > best_value + min_delta
            else:
                improved = best_value is None or value < best_value - min_delta
            if improved:
                best_value = value
                best_iter = it
                wait = 0
                torch.save(
                    {'model': model.state_dict(), 'config': cfg, 'iter': it, 'metrics': metrics},
                    out / 'best.pt')
                log.line('best iter {:04d} {} {:.4f}'.format(it, monitor, value))
            elif it < early_start:
                log.line('early hold until {}'.format(early_start))
            else:
                wait += 1
                log.line('early wait {}/{}'.format(wait, patience))
                if wait >= patience:
                    log.line('early stop at iter {:04d}, best iter {:04d}'.format(it, best_iter))
                    break

    torch.save({'model': model.state_dict(), 'config': cfg}, out / 'last.pt')
    log.line('saved {}'.format(out / 'last.pt'))
    if best_iter:
        log.line('use {}'.format(out / 'best.pt'))
        try:
            state = torch.load(out / 'best.pt', map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(out / 'best.pt', map_location=device)
        model.load_state_dict(state['model'])
        test_metrics = evaluate(model, test_sampler, device, int(cfg['eval_tasks']), loc_w, src_w)
        log.line('test_dir after best acc {:.3f} loss {:.4f} cls {:.4f} align {:.4f}'.format(
            test_metrics['acc'], test_metrics['loss'],
            test_metrics['cls_loss'], test_metrics['align_loss']))


def main():
    setup_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config' / 'cspd' / 'cspd.yaml'))
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--eval-tasks', type=int, default=None)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    if args.iters is not None:
        cfg['iters'] = args.iters
    if args.device is not None:
        cfg['device'] = args.device
    if args.eval_tasks is not None:
        cfg['eval_tasks'] = args.eval_tasks

    # Pretrained downloads, if needed when the user starts training, stay in project.
    torch.hub.set_dir(str(ROOT / '.cache/torch/hub'))
    device = pick_device(cfg.get('device', 'cuda'))
    seed = int(cfg.get('seed', 0))
    set_seed(seed)
    out = output_dir(cfg, ROOT, fresh=True)
    log = RunLog(out / 'train.log')
    log.line('===== {} ====='.format(datetime.now().isoformat(timespec='seconds')))
    config_text = write_config(cfg, out, args.config)
    log.line('config')
    log.line(config_text.rstrip())

    if str(cfg.get('train_mode') or '').strip() == 'curriculum':
        train_curriculum(cfg, log, out, device, seed)
    else:
        train_legacy(cfg, log, out, device, seed)
    (out / 'run_complete.json').write_text(
        json.dumps({'config': cfg, 'status': 'complete'}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')


if __name__ == '__main__':
    main()
