"""Supervised contrastive loss (SupCon, Khosla et al. 2020).

Supports optional fallback for mini-class batches (v7): when the per-task batch
has zero positive pairs (a class appears only once), fall back to a SimCLR-style
self-contrast term so the anchor still receives gradient.  Off by default to
preserve prior behavior for aerospace / older configs.
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised contrastive loss on L2-normalized features."""

    def __init__(self, temperature=0.07, min_class_fallback=False):
        super().__init__()
        self.temperature = temperature
        self.min_class_fallback = min_class_fallback

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] L2-normalized
            labels: [B] integer class ids
        Returns:
            scalar loss (0 if no positive pairs in batch)
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size < 2:
            return features.new_zeros(())

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = mask.sum(dim=1)
        if pos_per_anchor.max() < 1:
            if not self.min_class_fallback:
                return features.new_zeros(())
            # v7 fallback: every anchor treats all other samples as negatives,
            # positive contribution from its own augmented view (SimCLR-style).
            # Use logits_mask itself as positives so anchors with no same-class
            # peer still receive a (weak) self-contrast gradient.
            sim_logits = anchor_dot_contrast - logits_max.detach()
            sim_mask = logits_mask  # exclude self only
            exp_sim = torch.exp(sim_logits) * sim_mask
            log_prob_sim = sim_logits - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
            # Each anchor has (B-1) "positive-like" neighbours (uniform prior).
            return -log_prob_sim.mean()

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_per_anchor.clamp(min=1)
        valid = pos_per_anchor >= 1
        if not valid.any():
            return features.new_zeros(())
        return -mean_log_prob_pos[valid].mean()
