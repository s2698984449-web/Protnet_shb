"""Asymmetric multimodal fusion with KB prior + confidence + task embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFusion(nn.Module):
    """
    Independent per-modality classifiers + dynamic fusion weights.

    w = softmax(alpha * w_kb + beta * w_conf + gamma * w_task)
    """

    def __init__(self, embed_dim=128, num_classes=7, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.cls_rgb = nn.Linear(embed_dim, num_classes)
        self.cls_thermal = nn.Linear(embed_dim, num_classes)
        self.cls_us = nn.Linear(embed_dim, num_classes)
        self.task_to_weights = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, f_rgb, f_thermal, f_us, kb_prior=None, task_emb=None):
        logits_rgb = self.cls_rgb(f_rgb)
        logits_thermal = self.cls_thermal(f_thermal)
        logits_us = self.cls_us(f_us)

        conf_rgb = F.softmax(logits_rgb, dim=-1).max(dim=-1).values
        conf_thermal = F.softmax(logits_thermal, dim=-1).max(dim=-1).values
        conf_us = F.softmax(logits_us, dim=-1).max(dim=-1).values
        conf_stack = torch.stack([conf_rgb, conf_thermal, conf_us], dim=-1)
        w_conf = F.softmax(conf_stack, dim=-1)

        score = self.beta * w_conf
        if kb_prior is not None:
            if kb_prior.dim() == 1:
                kb_prior = kb_prior.unsqueeze(0).expand_as(w_conf)
            score = score + self.alpha * kb_prior
        if task_emb is not None:
            w_task = F.softmax(self.task_to_weights(task_emb), dim=-1)
            if w_task.shape[0] == 1 and score.shape[0] > 1:
                w_task = w_task.expand_as(score)
            score = score + self.gamma * w_task

        dynamic_w = F.softmax(score, dim=-1)
        fused = (
            dynamic_w[:, 0:1] * logits_rgb +
            dynamic_w[:, 1:2] * logits_thermal +
            dynamic_w[:, 2:3] * logits_us
        )
        return fused, dynamic_w, (logits_rgb, logits_thermal, logits_us)
