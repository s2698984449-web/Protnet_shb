"""MAML inner/outer loop helpers for multimodal aviation defect detection."""

import numpy as np
import torch
import torch.nn.functional as F
from learn2learn.utils import update_module
from torch.autograd import grad

from .multimodal_model import compute_stop_rl_loss, supervised_loss


def split_support_query(batch, shots, ways, device):
    """Split task batch into support/query multimodal tensors."""
    (rgb, thermal, ultrasonic), labels = batch
    rgb = rgb.to(device).float()
    thermal = thermal.to(device).float()
    ultrasonic = ultrasonic.to(device).float()
    labels = labels.to(device)

    n = labels.size(0)
    adapt_mask = np.zeros(n, dtype=bool)
    adapt_mask[np.arange(shots * ways) * 2] = True
    eval_mask = ~adapt_mask

    def _select(mask):
        return (
            rgb[mask], thermal[mask], ultrasonic[mask], labels[mask],
        )

    s_rgb, s_th, s_us, s_y = _select(adapt_mask)
    q_rgb, q_th, q_us, q_y = _select(eval_mask)
    return (s_rgb, s_th, s_us, s_y), (q_rgb, q_th, q_us, q_y)


def _maml_adapt(learner, loss, param_list, lr, first_order):
    params = [p for p in param_list if p.requires_grad]
    second_order = not first_order
    gradients = grad(
        loss, params,
        retain_graph=second_order,
        create_graph=second_order,
        allow_unused=True,
    )
    for p, g in zip(params, gradients):
        if g is not None:
            p.update = -lr * g
    learner.module = update_module(learner.module)


def adapt_supervised(learner, support, domain_meta, adapt_steps, param_list=None):
    if param_list is None:
        param_list = list(learner.module.parameters())
    s_rgb, s_th, s_us, s_y = support
    for _ in range(adapt_steps):
        loss = supervised_loss(learner.module, s_rgb, s_th, s_us, s_y, domain_meta)
        _maml_adapt(learner, loss, param_list, learner.lr, learner.first_order)


def evaluate_query(learner, query, domain_meta):
    q_rgb, q_th, q_us, q_y = query
    with torch.no_grad():
        fused, _, _ = learner.module.forward_logits(q_rgb, q_th, q_us, domain_meta)
        preds = fused.argmax(dim=-1)
        acc = (preds == q_y).float().mean().item()
        loss = F.cross_entropy(fused, q_y).item()
    return loss, acc


def meta_task_loss(learner, query, domain_meta, value_net, w_rl):
    q_rgb, q_th, q_us, q_y = query
    sl = supervised_loss(learner.module, q_rgb, q_th, q_us, q_y, domain_meta)
    if w_rl <= 0:
        return sl, sl.item(), evaluate_query(learner, query, domain_meta)[1], 1.0
    rl, mean_depth = compute_stop_rl_loss(
        learner.module, value_net, q_rgb, q_th, q_us, q_y, domain_meta)
    total = sl + w_rl * rl
    acc = evaluate_query(learner, query, domain_meta)[1]
    return total, sl.item(), acc, mean_depth


def curriculum_w_rl(iteration, stage1_end=100, stage2_end=200, w_rl_max=0.5):
    if iteration <= stage1_end:
        return 0.0
    if iteration <= stage2_end:
        progress = (iteration - stage1_end) / max(stage2_end - stage1_end, 1)
        return 0.3 * progress
    return w_rl_max


def curriculum_adapt_steps(iteration, default_steps=5, stage3_start=200):
    if iteration > stage3_start:
        return max(default_steps - 2, 1)
    return default_steps
