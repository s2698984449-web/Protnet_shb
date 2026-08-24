"""Multi-source domain pretraining model with ResNet18 backbone.

Architecture:
    RGB → ResNet18 (layer1-4) ─┬→ GAP → proj_head → feat (128-d)
                                │
                                ├── FiLM modulated by ConditionEncoder
                                └── Feature used by per-task classification heads

Three source datasets (MVTEC / GC10 / NEU) each have their own CE head.
One shared SupCon projection + global SupCon across all samples.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ResNet18Encoder
from .condition_encoder import (
    ConditionEncoder, ProductEncoder, FiLMGate, DomainReliability,
)
from .supcon_loss import SupConLoss


class MultiDomainPretrainModel(nn.Module):
    """ResNet18 backbone + FiLM + per-task heads + global SupCon projection.

    Args:
        task_specs: [(task_name, num_classes), ...] e.g.
            [('mvtec', 15), ('gc10', 8), ('neu', 6), ('cardd', 4),
             ('carparts', 8), ('syndcar', 4)]
        n_products: total product slots (默认 20 = 15+1+1+3 个 car-paint).
            PID 0~14 = MVTEC, 15 = GC10, 16 = NEU,
            17 = cardd, 18 = carparts, 19 = syndcar.
        n_conditions: 工况维度数 (默认 7).
        d_cond: 工况 embedding dim.
        d_prod: product embedding dim.
        feat_dim: backbone 投影输出 dim (= classifier input dim).
        proj_dim: SupCon 投影 head 输出 dim.
        film_layer: 在 ResNet18 的哪一层后插入 FiLM ('layer3' 推荐).
        film_channels: 该层输出通道数 (layer3=128, layer4=256).
        freeze_backbone: 是否冻结 backbone (只训 proj + FiLM + heads).
    """

    def __init__(
        self,
        task_specs=None,
        n_products: int = 20,
        n_conditions: int = 7,
        d_cond: int = 128,
        d_prod: int = 128,
        feat_dim: int = 128,
        proj_dim: int = 64,
        supcon_temperature: float = 0.07,
        film_layer: str = 'layer4',
        film_channels: int = 512,
        film_rho: float = 0.5,
        film_gate_clamp: bool = True,
        film_gate_clamp_std: float = 5.0,
        freeze_backbone: bool = True,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        if task_specs is None:
            # mvtec: 15-way product classification
            # gc10 : 8-way defect classification
            # neu  : 6-way defect classification
            task_specs = [('mvtec', 15), ('gc10', 8), ('neu', 6)]
        self.task_specs = task_specs
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.film_layer = film_layer

        # ── Backbone ─────────────────────────────────────────
        self.backbone = ResNet18Encoder(
            out_dim=feat_dim,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )

        # ── Condition / Product / FiLM ──────────────────────
        self.cond_enc = ConditionEncoder(n_conditions=n_conditions, d=d_cond)
        self.prod_enc = ProductEncoder(n_products=n_products, d=d_prod)
        self.film = FiLMGate(d_cond=d_cond, d_feat=film_channels,
                             rho=film_rho,
                             gate_clamp=film_gate_clamp,
                             gate_clamp_std=film_gate_clamp_std)
        # 把 product embedding (d_prod) 投影到 backbone feat_dim, 让 ProductEncoder
        # 能影响最终表示 (cross-domain 时很关键: 新产品 cold-start 通过这个注入)
        self._prod_proj = nn.Sequential(
            nn.Linear(d_prod, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.Tanh(),  # 输出 [-1, 1], 防止 product injection 主导 backbone
        )

        # ── Per-task CE heads ────────────────────────────────
        self.heads = nn.ModuleDict({
            name: nn.Linear(feat_dim, num_classes)
            for name, num_classes in task_specs
        })

        # ── SupCon projection head ───────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.supcon = SupConLoss(temperature=supcon_temperature)

        # ── Domain reliability (for cross-domain episode) ────
        self.reliability = DomainReliability(
            feat_dim=feat_dim, n_domains=len(task_specs), hidden=64,
        )

        # Save FiLM target channel for forward()
        self._film_channels = film_channels

        logging.info(
            'MultiDomainPretrainModel ready: tasks={}, feat_dim={}, '
            'proj_dim={}, film_layer={}, film_rho={}, film_gate_clamp={}, '
            'freeze_backbone={}.'.format(
                task_specs, feat_dim, proj_dim, film_layer, film_rho,
                film_gate_clamp, freeze_backbone))
        logging.info('  film.gamma.weight: %s', list(self.film.gamma.weight.shape))

    # ── Backbone with FiLM modulation ──────────────────────
    def _backbone_with_film(self, x, cond_emb):
        """ResNet18 forward with FiLM modulation on layer4 output (512 ch)."""
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        # ── FiLM modulate after layer4, before GAP ──
        x = self.film(x, cond_emb)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # (B, 512)

    def encode(self, x, cond_vec):
        """Encode RGB + condition → 128-d feature.

        Args:
            x: (B, 3, H, W) RGB tensor (会被 ResNet18 内置 resize).
            cond_vec: (B, n_conditions) one-hot/multi-hot condition.
        Returns:
            feat: (B, feat_dim)
        """
        # ── 处理 [1, 20, 3, 224, 224] 等多余 leading batch dim ──
        if x.dim() == 5:
            assert x.size(0) == 1, (
                f'encode: 5D x with batch>1 not supported yet: {x.shape}'
            )
            x = x.squeeze(0)  # → [N, 3, H, W]
        # 只有真正的 3D 图 (1,3,H,W) 才需要加 batch 维; 已经是 4D 就跳过
        if x.dim() == 3:
            x = x.unsqueeze(0)  # → [1, 3, H, W]
        assert x.dim() == 4, f'after normalize: x={x.dim()}D {x.shape}'
        B = x.size(0)
        if cond_vec is None:
            cond_vec = torch.zeros(B, self.cond_enc.n_conditions, device=x.device)
        cond_emb = self.cond_enc(cond_vec)
        feat_512 = self._backbone_with_film(x, cond_emb)
        return self.backbone.proj(feat_512)

    def encode_with_product(self, x, cond_vec, product_id):
        """Encode with condition AND product_id injection (cross-domain / meta-test).

        Product embedding 通过 prod_proj 投影到 feat_dim 空间, 然后加到 backbone 输出上.
        这样 ProductEncoder 把 product 信息注入表示 → 同 backbone 不同产品学出不同特征空间.

        Args:
            x: (B, 3, H, W)
            cond_vec: (B, n_conditions) — 用于 FiLM 调制
            product_id: (B,) long — ProductEncoder slot
        Returns:
            feat: (B, feat_dim)
        """
        feat = self.encode(x, cond_vec)
        prod_emb = self.prod_enc(product_id)  # (B, d_prod)
        # 投影 product embedding 到 feat_dim, 残差注入
        prod_feat = self._prod_proj(prod_emb)  # (B, feat_dim)
        return feat + prod_feat

    def project(self, feats):
        """L2-normalized SupCon projection."""
        return F.normalize(self.projection(feats), dim=1)

    # ── Forward: per-task CE ──────────────────────────────
    def forward_task(self, task_name, x, cond_vec):
        feat = self.encode(x, cond_vec)
        return self.heads[task_name](feat), feat

    # ── Combined loss (CE + SupCon) for one task ──────────
    def compute_loss(
        self,
        task_name,
        x,
        y,
        cond_vec,
        supcon_weight=0.5,
        ce_weight=0.5,
        label_smoothing=0.0,
        class_weight=None,
    ):
        """Compute CE + SupCon for one task batch.

        Returns:
            total_loss, ce_only, feats
        """
        logits, feats = self.forward_task(task_name, x, cond_vec)
        ce = F.cross_entropy(
            logits, y, label_smoothing=label_smoothing, weight=class_weight,
        )
        if feats.size(0) < 2:
            return ce_weight * ce, ce, feats

        proj = self.project(feats)
        supcon_val = self.supcon(proj, y)
        total = ce_weight * ce + supcon_weight * supcon_val
        return total, ce, feats

    # ── Combined loss across all tasks (global SupCon) ────
    def compute_combined_loss(
        self,
        batches,  # dict {task_name: (x, y)}
        cond_batches,  # dict {task_name: cond_vec}
        supcon_weight=0.5,
        ce_weight=0.5,
        label_smoothing=0.0,
        global_supcon=True,
    ):
        """Sum CE per task + SupCon across all samples together.

        用于 combined mix_strategy: 一次性 forward 多个源域样本,
        SupCon 用全局 label namespace (task offset) 让跨源域同类聚类.
        """
        all_feats, all_labels, total_ce = [], [], 0.0
        offsets = {}
        cum = 0
        for name, _ in self.task_specs:
            offsets[name] = cum
            cum += self.heads[name].out_features

        for name, (x, y) in batches.items():
            cond = cond_batches[name]
            logits, feats = self.forward_task(name, x, cond)
            ce = F.cross_entropy(
                logits, y, label_smoothing=label_smoothing,
            )
            total_ce = total_ce + ce
            all_feats.append(feats)
            if global_supcon:
                all_labels.append(y + offsets[name])
            else:
                all_labels.append(y)

        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if all_feats.size(0) < 2:
            return total_ce, total_ce, all_feats

        proj = self.project(all_feats)
        supcon_val = self.supcon(proj, all_labels)
        total = ce_weight * total_ce + supcon_weight * supcon_val
        return total, total_ce, all_feats

    # ── Reliability scoring for cross-domain episode ──────
    def reliability_score(self, query_emb, domain_ids):
        """Score per-sample reliability for a cross-domain episode.

        Args:
            query_emb: (N, D)
            domain_ids: (N,) long
        Returns:
            reliability: (N,) ∈ (0, 1)
        """
        n_dom = self.reliability.n_domains
        one_hot = F.one_hot(domain_ids.long(), n_dom).float()
        return self.reliability(query_emb, one_hot)

    # ── For inference / saving ────────────────────────────
    def backbone_state_dict(self):
        return self.backbone.state_dict()

    def head_params(self):
        """Trainable head parameters (everything except backbone)."""
        params = []
        for name, p in self.named_parameters():
            if 'backbone.conv' in name or 'backbone.bn' in name or \
               'backbone.layer' in name or 'backbone.relu' in name or \
               'backbone.maxpool' in name or 'backbone.avgpool' in name:
                continue
            params.append(p)
        return params

    def backbone_params(self):
        return [p for n, p in self.named_parameters()
                if ('backbone.conv' in n or 'backbone.bn' in n or
                    'backbone.layer' in n or 'backbone.avgpool' in n)]