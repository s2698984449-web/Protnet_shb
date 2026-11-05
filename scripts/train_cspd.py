"""运行:
  python scripts/load_data.py --config config/cspd.yaml
  python scripts/train_cspd.py --config config/cspd.yaml
  python scripts/train_cspd.py --config config/cspd.yaml --iters 20 --eval-tasks 5

训练只读配置。换数据只改配置里的绝对路径和可选旁路，不改脚本。
输出在 output/<exp_name>/：config.yaml、config.json、structure.txt、data_summary.json、train.log、best.pt、last.pt。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cspd.data import describe_config, make_episodes
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


def main():
    setup_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config' / 'cspd.yaml'))
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

    device = pick_device(cfg.get('device', 'cuda'))
    seed = int(cfg.get('seed', 0))
    set_seed(seed)
    out = output_dir(cfg, ROOT)
    log = RunLog(out / 'train.log')
    log.line('===== {} ====='.format(datetime.now().isoformat(timespec='seconds')))

    data_report = describe_config(cfg, seed)
    (out / 'data_summary.json').write_text(
        json.dumps(data_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    config_text = write_config(cfg, out, args.config)
    log.line('config')
    log.line(config_text.rstrip())
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

    groups = []
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
    opt = torch.optim.AdamW(
        groups,
        lr=lr,
        weight_decay=float(cfg['weight_decay']))
    n_train = sum(p.numel() for p in trainable_parameters(model))
    log.line('trainable params {}'.format(n_train))
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
            raise ValueError('early_stop=true 时必须配置独立的 val_dir，不能用 test_dir 选模型')
        stop_sampler = make_episodes(cfg, 'val_dir', seed + 2)
        stop_name = 'val_dir'
        log.line('early stop on {} monitor {} patience {} start {}'.format(
            stop_name, monitor, patience, early_start))
        if int(cfg['eval_tasks']) < 20:
            log.line('eval_tasks {} is noisy, early stop may react to one unlucky draw'.format(
                cfg['eval_tasks']))
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

    if not early:
        metrics = evaluate(model, test_sampler, device, int(cfg['eval_tasks']), loc_w, src_w)
        log.line('final test_dir acc {:.3f} loss {:.4f}'.format(metrics['acc'], metrics['loss']))

    ckpt = out / 'last.pt'
    torch.save({'model': model.state_dict(), 'config': cfg}, ckpt)
    log.line('saved {}'.format(ckpt))
    if best_iter:
        log.line('use {}'.format(out / 'best.pt'))
        # 用独立测试集再报一次，不参与选模型
        try:
            state = torch.load(out / 'best.pt', map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(out / 'best.pt', map_location=device)
        model.load_state_dict(state['model'])
        test_metrics = evaluate(model, test_sampler, device, int(cfg['eval_tasks']), loc_w, src_w)
        log.line('test_dir after best acc {:.3f} loss {:.4f} cls {:.4f} align {:.4f}'.format(
            test_metrics['acc'], test_metrics['loss'],
            test_metrics['cls_loss'], test_metrics['align_loss']))


if __name__ == '__main__':
    main()
