"""训练和推理共用的运行函数。"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .model import ConditionStrategyDetector, DualRegionBetaDetector, location_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name):
    if str(name).startswith('cuda') and not torch.cuda.is_available():
        print('cuda unavailable, use cpu')
        name = 'cpu'
    return torch.device(name)


def move_episode(episode, device):
    out = {}
    for key, value in episode.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def forward_episode(model, episode, ways):
    return model(
        episode['support_x'],
        episode['support_y'],
        episode['query_x'],
        episode['support_cond'],
        episode['query_cond'],
        episode['query_y'],
        ways,
        support_cond_mask=episode.get('support_cond_mask'),
        query_cond_mask=episode.get('query_cond_mask'),
        support_x_restored=episode.get('support_x_restored'),
        query_x_restored=episode.get('query_x_restored'),
        support_mask=episode.get('support_mask'),
        query_mask=episode.get('query_mask'),
        support_has_mask=episode.get('support_has_mask'),
        query_has_mask=episode.get('query_has_mask'),
    )


def episode_loss(pred, episode, location_loss_weight=0.0, source_proto_weight=0.0):
    cls = F.cross_entropy(pred['logits'], episode['query_y'])
    align = pred.get('source_proto')
    if align is None:
        align = cls.new_zeros(())
    loc_w = float(location_loss_weight)
    src_w = float(source_proto_weight)
    if loc_w <= 0:
        loc = cls.new_zeros(())
    else:
        loc = location_loss(pred['peak_yx'], pred['gt_yx'], pred['gt_valid'])
    return cls + loc_w * loc + src_w * align, cls, loc, align


def build_model(cfg):
    if str(cfg.get('train_mode') or '').strip() == 'curriculum':
        model = DualRegionBetaDetector(
            feat_dim=int(cfg.get('feat_dim', 128)),
            cond_dim=int(cfg.get('cond_dim', 6)),
            dilate_px=int(cfg.get('context_dilate_px', 16)),
            feature_layer=int(cfg.get('feature_layer', 3)),
            pretrained_backbone=bool(cfg.get('pretrained_backbone', True)),
            freeze_backbone=bool(cfg.get('freeze_backbone', True)),
            use_policy=bool(cfg.get('use_policy', True)),
            fixed_beta=float(cfg.get('fixed_beta', 0.25)),
            beta_center=float(cfg.get('beta_center', 0.25)),
            beta_scale=float(cfg.get('beta_scale', 0.25)),
        )
        if bool(cfg.get('freeze_policy', False)):
            for param in model.policy.parameters():
                param.requires_grad = False
        return model
    model = ConditionStrategyDetector(
        feat_dim=cfg['feat_dim'],
        cond_dim=cfg['cond_dim'],
        film_rho=cfg['film_rho'],
        policy_rho=cfg['policy_rho'],
        topk_ratio=cfg['topk_ratio'],
        adapt_steps=cfg['adapt_steps'],
        adapt_lr=cfg['adapt_lr'],
        pretrained_backbone=cfg['pretrained_backbone'],
        freeze_backbone=cfg['freeze_backbone'],
        use_mask_proto=bool(cfg.get('use_mask_proto', True)),
        gauss_sigma_ratio=float(cfg.get('gauss_sigma_ratio', 0.15)),
    )
    apply_freeze(model, cfg.get('freeze') or {})
    return model


def apply_freeze(model, freeze):
    """freeze 里值为 true 的模块不更新。缺省的键保持构建时的状态。"""
    flags = dict(freeze or {})
    if 'proj' in flags:
        flags.setdefault('proj3', flags['proj'])
        flags.setdefault('proj4', flags['proj'])
    if 'policy' in flags:
        for name in ('policy_trunk', 'g_head', 'alpha_head', 'gate_head', 'tau_head'):
            flags.setdefault(name, flags['policy'])
    groups = {
        'backbone': model.backbone,
        'proj3': model.proj3,
        'proj4': model.proj4,
        'cond_encoder': model.cond_encoder,
        'part_encoder': model.part_encoder,
        'film': model.film,
        'policy_trunk': model.policy.trunk,
        'g_head': model.policy.g_head,
        'alpha_head': model.policy.alpha_head,
        'gate_head': model.policy.gate_head,
        'tau_head': model.policy.tau_head,
    }
    for name, module in groups.items():
        if name not in flags:
            continue
        frozen = bool(flags[name])
        for param in module.parameters():
            param.requires_grad = not frozen
        if name == 'backbone':
            model.backbone.frozen = frozen
            if frozen:
                model.backbone.eval()
                model.backbone.train_layer4 = False
    if flags.get('layer4') is False:
        model.backbone.train_layer4 = True
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True


def output_dir(cfg, project_root, *, fresh=False):
    name = str(cfg.get('exp_name', '')).strip()
    group = str(cfg.get('output_group') or '').strip()
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError('exp_name 必须是一级目录名')
    if group and (Path(group).name != group or group in {'.', '..'}):
        raise ValueError('output_group 必须是一级目录名，例如 v3')
    out = Path(project_root) / 'output' / 'cspd'
    if group:
        out = out / group
    out = out / name
    # Training uses fresh=True; legacy read/inference helpers retain their API.
    out.mkdir(parents=True, exist_ok=not fresh)
    return out


def write_config(cfg, out_dir, source):
    payload = dict(cfg)
    payload['config_path'] = str(Path(source).resolve())
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (out_dir / 'config.json').write_text(text + '\n', encoding='utf-8')
    with (out_dir / 'config.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return text


def structure_text(model):
    rows = ['module\tparams\ttrainable']
    for name, module in model.named_children():
        params = list(module.parameters())
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        rows.append('{}\t{}\t{}'.format(name, total, trainable))
    rows.append('')
    rows.append(str(model))
    return '\n'.join(rows) + '\n'


def write_structure(model, out_dir):
    text = structure_text(model)
    (out_dir / 'structure.txt').write_text(text, encoding='utf-8')
    return text


@torch.no_grad()
def evaluate(model, sampler, device, tasks, location_loss_weight=0.0, source_proto_weight=0.0):
    model.eval()
    correct = 0
    total = 0
    losses = []
    cls_losses = []
    align_losses = []
    loc_losses = []
    g_values = []
    loc_hits = 0
    loc_total = 0
    for _ in range(tasks):
        raw = sampler.sample()
        episode = move_episode(raw, device)
        ways = int(episode.get('n_way', sampler.n_way))
        pred = forward_episode(model, episode, ways)
        loss, cls, loc, align = episode_loss(
            pred, episode, location_loss_weight, source_proto_weight)
        hits = (pred['logits'].argmax(dim=-1) == episode['query_y']).sum().item()
        correct += hits
        total += episode['query_y'].numel()
        losses.append(float(loss))
        cls_losses.append(float(cls))
        align_losses.append(float(align))
        loc_losses.append(float(loc))
        g_values.append(float(pred['gate_g'].mean()))
        valid = pred['gt_valid']
        if bool(valid.any()):
            err = (pred['peak_yx'][valid] - pred['gt_yx'][valid]).abs().sum(dim=-1)
            loc_hits += int((err < 0.2).sum().item())
            loc_total += int(valid.sum().item())
    model.train()
    return {
        'acc': correct / max(total, 1),
        'loss': sum(losses) / max(len(losses), 1),
        'cls_loss': sum(cls_losses) / max(len(cls_losses), 1),
        'align_loss': sum(align_losses) / max(len(align_losses), 1),
        'loc_loss': sum(loc_losses) / max(len(loc_losses), 1),
        'loc_recall@0.2': loc_hits / max(loc_total, 1) if loc_total else None,
        'gate_g': sum(g_values) / max(len(g_values), 1),
    }
