"""Shared YOLO26 backbone + multi-task heads for Phase 0 pretraining."""

import logging
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import YOLO26BackboneEncoder
from .supcon_loss import SupConLoss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (Lin et al., 2017)."""

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # can be a tensor of per-class weights
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha.to(logits.device) if self.alpha is not None else None, reduction='none')
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        focal = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss (Cao et al., 2019).

    For class c with training-sample count n_c, define margin
        gamma_c = s / n_c^(1/4)
    The loss adds -gamma_c to logit[c] for the ground-truth class only, so
    under-represented classes get a larger decision boundary margin.  Designed
    for long-tailed recognition; combined with class-balanced sampling it
    substantially improves tail-class accuracy on imbalanced classification.

    Reference: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware
    Margin Loss", NeurIPS 2019.
    """

    def __init__(self, cls_counts, max_margin=0.3, weight=None, reduction='mean'):
        super().__init__()
        cls_counts = torch.as_tensor(cls_counts, dtype=torch.float32).clamp(min=1)
        # Normalize counts -> mean=1 for stability.
        cls_counts = cls_counts / cls_counts.mean()
        margins = max_margin / cls_counts.pow(0.25)
        # Don't register as Parameter (these are derived from data; not learnable).
        self.register_buffer('margins', margins)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # Build a one-hot mask of the ground-truth class and subtract margin.
        idx = torch.arange(logits.size(1), device=logits.device)
        one_hot = (idx.unsqueeze(0) == targets.unsqueeze(1)).float()
        adjusted = logits - one_hot * self.margins.to(logits.device)
        return F.cross_entropy(
            adjusted, targets, weight=self.weight,
            reduction=self.reduction)


class PretrainMultitaskModel(nn.Module):
    """
    Shared YOLO26 encoder with one linear head per pretrain task.

    Aerospace default: blade (5-way) + aebad (binary).
    Automotive: cardd (4-way) + neu (6-way) + asde (binary).
    """

    def __init__(self, yolo_weights='', yolo_imgsz=640, feat_dim=256, task_specs=None,
                 loss_mode='ce', proj_dim=64, supcon_temperature=0.07,
                 label_namespace='per_task', temperature_adapt=False,
                 aux_scratch_head=False, supcon_min_class_fallback=False):
        """
        Args:
            label_namespace:
                'per_task' (default) – each task computes SupCon independently with
                    its own 0..K-1 labels. Safe and original behaviour.
                'global' – tasks contribute features into a single SupCon batch with
                    globally remapped labels (offset per task). This is the P0 v5
                    upgrade and enables cross-task class clustering.
            temperature_adapt:
                If True, dynamically scale temperature by per-batch feature norm.
                Recommended for tasks where intra-class variance is large
                (industrial defects).  See forward docs.
            aux_scratch_head:
                If True, attach an auxiliary binary head (64-dim proj → 1) that
                predicts is_scratch (0/1) across all tasks.  This teaches the
                shared encoder that cardd.scratch / neu.scratches / asde.defect
                are semantically related — the "层次 1" scratch fusion described in
                the CMPL v5 plan.  The BCE loss is small (weight=0.1) so it does
                not interfere with the main CE+SupCon objectives.
        """
        super().__init__()
        self.task_specs = task_specs or [('blade', 5), ('aebad', 2)]
        self.loss_mode = loss_mode
        self.label_namespace = label_namespace
        self.temperature_adapt = bool(temperature_adapt)
        self.aux_scratch_head = bool(aux_scratch_head)
        # Cumulative label offsets for global namespace (computed lazily once num
        # classes per task are known).
        self._label_offsets = None
        self.encoder = YOLO26BackboneEncoder(
            out_dim=feat_dim, imgsz=yolo_imgsz, weights_path=yolo_weights)
        self.feat_dim = feat_dim
        self.heads = nn.ModuleDict({
            name: nn.Linear(feat_dim, num_classes)
            for name, num_classes in self.task_specs
        })
        self._template_weights = yolo_weights
        self.projection = None
        self.supcon = None
        if loss_mode in ('supcon_ce', 'supcon_ce_v5'):
            self.projection = nn.Sequential(
                nn.Linear(feat_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim),
            )
            self.supcon = SupConLoss(temperature=supcon_temperature,
                                      min_class_fallback=supcon_min_class_fallback)
        self._aux_scratch_head = None
        if self.aux_scratch_head:
            # Share projection: reuse the 64-dim projection output for scratch prediction.
            # The projection already ends with ReLU+Linear, so we only need a Linear head here.
            # CRITICAL: do NOT add another ReLU(inplace=True) — it would modify the shared
            # projection output in-place, breaking the autograd graph when supcon also
            # needs those features in the same step.
            self._aux_scratch_head = nn.Linear(proj_dim, 1)

    @property
    def head_blade(self):
        return self.heads['blade']

    @property
    def head_aebad(self):
        return self.heads['aebad']

    def backbone_modules(self):
        det_idx = len(self.encoder.det_model.model) - 1
        return self.encoder.det_model.model[:det_idx]

    def configure_backbone(self, mode='frozen', unfreeze_last_n=2, freeze_indices=None):
        """mode: frozen | partial | full | indices

        indices: freeze YOLO blocks listed in freeze_indices (Ultralytics model[i]),
        train the remaining backbone blocks before Detect. Prefer indices over
        ResNet-style stem/stage names (YOLO26n has none).
        """
        det_idx = len(self.encoder.det_model.model) - 1
        freeze_set = set(freeze_indices or ())
        for i in range(det_idx):
            if mode == 'full':
                trainable = True
            elif mode == 'partial':
                trainable = i >= det_idx - unfreeze_last_n
            elif mode == 'indices':
                trainable = i not in freeze_set
            else:
                trainable = False
            for param in self.encoder.det_model.model[i].parameters():
                param.requires_grad = trainable
        if mode == 'partial':
            status = 'partial(last={})'.format(unfreeze_last_n)
        elif mode == 'indices':
            status = 'indices(freeze={})'.format(sorted(freeze_set))
        else:
            status = mode
        trainable_blocks = [
            i for i in range(det_idx)
            if any(p.requires_grad for p in self.encoder.det_model.model[i].parameters())
        ]
        logging.info('Backbone train mode: {}; trainable_blocks={}.'.format(
            status, trainable_blocks))

    def encode(self, x):
        return self.encoder.encode(x)

    def forward_task(self, task_name, x):
        return self.heads[task_name](self.encode(x))

    def forward_blade(self, x):
        return self.forward_task('blade', x)

    def forward_aebad(self, x):
        return self.forward_task('aebad', x)

    def task_loss(self, task_name, x, y, label_smoothing=0.0):
        return F.cross_entropy(
            self.forward_task(task_name, x), y, label_smoothing=label_smoothing)

    def project(self, feats):
        if self.projection is None:
            raise RuntimeError('Projection head missing (loss_mode != supcon_ce).')
        return F.normalize(self.projection(feats), dim=1)

    # ------------------------------------------------------------------
    # Auxiliary scratch binary head (层次 1 fusion)
    # ------------------------------------------------------------------
    def scratch_binary_logits(self, proj_feats):
        """Return logits for is_scratch binary classification.

        Args:
            proj_feats: [B, proj_dim] L2-normalized projection features.
        Returns:
            [B, 1] raw logits (no sigmoid).
        Raises RuntimeError if aux_scratch_head is not enabled.
        """
        if self._aux_scratch_head is None:
            raise RuntimeError('aux_scratch_head not enabled on this model.')
        return self._aux_scratch_head(proj_feats)

    def scratch_binary_loss(self, proj_feats, is_scratch_labels):
        """Compute BCE loss for auxiliary scratch binary head.

        Args:
            proj_feats: [B, proj_dim] L2-normalized projection features.
            is_scratch_labels: [B] int tensor (0 or 1).
        Returns:
            Scalar BCEWithLogitsLoss.
        """
        logits = self.scratch_binary_logits(proj_feats)
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            is_scratch_labels.float(),
            reduction='mean',
        )

    def _resolve_label_offsets(self):
        """Lazily build cumulative offsets [0, K0, K0+K1, ...] for global namespace."""
        if self._label_offsets is not None:
            return self._label_offsets
        offsets = {}
        cum = 0
        for name, k in self.task_specs:
            offsets[name] = cum
            cum += k
        self._label_offsets = offsets
        return offsets

    def task_loss_supcon_ce(self, task_name, x, y, supcon_weight=0.5, ce_weight=0.5,
                            label_smoothing=0.0, class_weight=None,
                            use_focal=False, focal_gamma=2.0,
                            use_ldam=False, ldam_cls_counts=None, ldam_max_margin=0.3):
        feats = self.encode(x)
        if use_focal:
            logits = self.heads[task_name](feats)
            ce = FocalLoss(gamma=focal_gamma, alpha=class_weight)(logits, y)
        elif use_ldam:
            logits = self.heads[task_name](feats)
            ldam = LDAMLoss(cls_counts=ldam_cls_counts, max_margin=ldam_max_margin,
                            weight=class_weight)
            ce = ldam(logits, y)
        else:
            ce = F.cross_entropy(
                self.heads[task_name](feats), y, label_smoothing=label_smoothing,
                weight=class_weight)
        if self.projection is None or feats.size(0) < 2:
            return ce, ce  # (total_loss, ce_component) for uncertainty weighting
        proj = self.project(feats)
        if self.temperature_adapt:
            # Temperature adaptive to feature-norm variance (Wen et al., 2022).
            # When intra-class variance is high, lower tau sharpens; when low, raise.
            with torch.no_grad():
                std = proj.norm(dim=1).std().clamp(min=1e-3)
                adapt_tau = float(self.supcon.temperature) * float(std.detach().item())
                adapt_tau = max(min(adapt_tau, 0.2), 0.03)
            logits_sc = torch.matmul(proj, proj.T) / adapt_tau
            mask = torch.eq(y, y.T).float().to(proj.device)
            logits_mask = 1.0 - torch.eye(proj.size(0), device=proj.device)
            mask = mask * logits_mask
            logits_max, _ = logits_sc.max(dim=1, keepdim=True)
            logits_sc = logits_sc - logits_max.detach()
            exp_logits = torch.exp(logits_sc) * logits_mask
            log_prob = logits_sc - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
            pos = mask.sum(dim=1)
            valid = pos >= 1
            if not valid.any():
                if not self.supcon.min_class_fallback:
                    return ce_weight * ce, ce
                # v7 fallback: SimCLR-style self-contrast so anchor still gets gradient.
                supcon_val = -log_prob.mean()
            else:
                supcon_val = -(mask * log_prob).sum(dim=1)[valid].mean()
        else:
            supcon_val = self.supcon(proj, y)
        total = ce_weight * ce + supcon_weight * supcon_val
        return total, ce

    def compute_task_loss(self, task_name, x, y, supcon_weight=0.5, ce_weight=0.5,
                          label_smoothing=0.0, class_weight=None,
                          use_focal=False, focal_gamma=2.0,
                          use_ldam=False, ldam_cls_counts=None, ldam_max_margin=0.3):
        if self.loss_mode in ('supcon_ce', 'supcon_ce_v5'):
            return self.task_loss_supcon_ce(
                task_name, x, y, supcon_weight=supcon_weight, ce_weight=ce_weight,
                label_smoothing=label_smoothing, class_weight=class_weight,
                use_focal=use_focal, focal_gamma=focal_gamma,
                use_ldam=use_ldam, ldam_cls_counts=ldam_cls_counts,
                ldam_max_margin=ldam_max_margin)
        ce = self.task_loss(task_name, x, y, label_smoothing=label_smoothing)
        return ce, ce

    def blade_loss(self, x, y):
        return self.task_loss('blade', x, y)

    def aebad_loss(self, x, y):
        return self.task_loss('aebad', x, y)


def _unwrap_encoder(model):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module.encoder
    return model.encoder


def export_pretrained_yolo(model, template_weights, out_path):
    """Write ultralytics-compatible .pt with fine-tuned backbone weights."""
    encoder = _unwrap_encoder(model)
    template = os.path.abspath(template_weights)
    out_abs = os.path.abspath(out_path)
    if not os.path.isfile(template):
        raise FileNotFoundError('Template YOLO weights not found: {}'.format(template))

    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    if template != out_abs:
        shutil.copy2(template, out_abs)

    ckpt = torch.load(out_abs, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        target = ckpt['model']
        if hasattr(target, 'load_state_dict'):
            target.load_state_dict(encoder.det_model.state_dict(), strict=False)
        else:
            raise RuntimeError('Unsupported YOLO checkpoint format in {}.'.format(template))
        torch.save(ckpt, out_abs)
    else:
        torch.save(encoder.det_model.state_dict(), out_abs)

    logging.info('Saved pretrained YOLO backbone to {}.'.format(out_abs))
