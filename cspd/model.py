"""工况策略检测器（通用闭环）。

数据流:
  原图 / 可选还原图 -> 空间骨干 -> 1x1 投影
       -> 还原图特征按工况槽位做 FiLM（缺测维不参与）
       -> 策略按类给出混合、尺度、通道、距离倍数
       -> 支持集按类建原型（有掩码时用高斯权重）
       -> 查询空间距离图 -> 类别 / 峰值位置

内循环只更新两个 1x1 投影。策略和 FiLM 留在外循环。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .annotate import (
    gaussian_weight_from_mask,
    mask_centroid_yx,
    resize_mask,
    weighted_spatial_mean,
)
from .backbone import SpatialResNet18
from .condition import FiLM, MLPEncoder
from .condition_io import apply_cond_mask
from .policy import EpisodeBetaPolicy, PolicyNet


def _conv1x1(x, weight, bias):
    return F.conv2d(x, weight, bias)


def topk_mean(score, ratio):
    _, _, h, w = score.shape
    k = max(1, int(round(float(ratio) * h * w)))
    vals, _ = score.flatten(2).topk(k, dim=-1)
    return vals.mean(dim=-1)


def peak_yx(score, pred):
    n, _, h, w = score.shape
    chosen = score[torch.arange(n, device=score.device), pred]
    idx = chosen.view(n, -1).argmax(dim=-1)
    y = (idx // w).float() / max(h - 1, 1)
    x = (idx % w).float() / max(w - 1, 1)
    return torch.stack([y, x], dim=-1)


def location_loss(peak, gt_yx, valid):
    """峰值与掩码质心的 L1。无有效掩码时损失为 0。"""
    if valid.dtype != torch.bool:
        valid = valid.bool()
    if not bool(valid.any()):
        return peak.new_zeros(())
    diff = (peak[valid] - gt_yx[valid]).abs().mean()
    return diff


def loo_pooled_logits(raw3, film3, raw4, film4, labels, ways, g, alpha, gate, tau):
    p3 = raw3.mean(dim=(2, 3))
    f3 = film3.mean(dim=(2, 3))
    p4 = raw4.mean(dim=(2, 3))
    f4 = film4.mean(dim=(2, 3))
    n = labels.shape[0]
    rows = []
    for i in range(n):
        dists = []
        for c in range(ways):
            m3 = gate[c].view(1, -1) * (g[c] * f3 + (1.0 - g[c]) * p3)
            m4 = gate[c].view(1, -1) * (g[c] * f4 + (1.0 - g[c]) * p4)
            keep = labels == c
            keep = keep.clone()
            keep[i] = False
            if not bool(keep.any()):
                proto3 = m3.mean(dim=0)
                proto4 = m4.mean(dim=0)
            else:
                proto3 = m3[keep].mean(dim=0)
                proto4 = m4[keep].mean(dim=0)
            d3 = -((m3[i] - proto3) ** 2).sum()
            d4 = -((m4[i] - proto4) ** 2).sum()
            dists.append((alpha[c, 0] * d3 + alpha[c, 1] * d4) / tau[c])
        rows.append(torch.stack(dists))
    return torch.stack(rows, dim=0)


class ConditionStrategyDetector(nn.Module):
    def __init__(
        self,
        feat_dim=128,
        cond_dim=6,
        film_rho=0.5,
        policy_rho=0.5,
        topk_ratio=0.1,
        adapt_steps=1,
        adapt_lr=0.1,
        pretrained_backbone=True,
        freeze_backbone=True,
        use_mask_proto=True,
        gauss_sigma_ratio=0.15,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.cond_dim = int(cond_dim)
        self.topk_ratio = float(topk_ratio)
        self.adapt_steps = int(adapt_steps)
        self.adapt_lr = float(adapt_lr)
        self.use_mask_proto = bool(use_mask_proto)
        self.gauss_sigma_ratio = float(gauss_sigma_ratio)
        self.backbone = SpatialResNet18(
            pretrained=pretrained_backbone, freeze=freeze_backbone)
        self.proj3 = nn.Conv2d(256, feat_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(512, feat_dim, kernel_size=1)
        self.cond_encoder = MLPEncoder(cond_dim, feat_dim)
        self.part_encoder = MLPEncoder(feat_dim, feat_dim)
        self.film = FiLM(d_cond=feat_dim, d_feat=feat_dim, rho=film_rho)
        self.policy = PolicyNet(
            state_dim=feat_dim + feat_dim + cond_dim + 2,
            feat_dim=feat_dim,
            n_scales=2,
            rho=policy_rho,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.backbone.frozen:
            self.backbone.eval()
        return self

    def _project(self, layer3, layer4, w3, b3, w4, b4):
        return _conv1x1(layer3, w3, b3), _conv1x1(layer4, w4, b4)

    def _restore(self, raw3, raw4, cond, cond_mask):
        emb = self.cond_encoder(apply_cond_mask(cond, cond_mask))
        return self.film(raw3, emb), self.film(raw4, emb)

    def _mix_class(self, raw, restored, g, gate):
        mixed = g * restored + (1.0 - g) * raw
        return mixed * gate.view(1, -1, 1, 1)

    def _proto_weight(self, masks, has_mask, size_hw):
        n = masks.shape[0]
        device = masks.device
        if (not self.use_mask_proto) or has_mask is None or (not bool(has_mask.any())):
            h, w = size_hw
            return torch.ones(n, 1, h, w, device=device) / float(h * w)
        resized = resize_mask(masks, size_hw)
        weight = gaussian_weight_from_mask(resized, self.gauss_sigma_ratio)
        uniform = torch.ones_like(weight) / float(size_hw[0] * size_hw[1])
        has = has_mask.view(n, 1, 1, 1).float()
        return weight * has + uniform * (1.0 - has)

    def _class_state(self, raw3, raw4, film3, film4, support_y, support_cond, support_cond_mask, ways):
        pooled = raw4.mean(dim=(2, 3))
        gap3 = (film3 - raw3).abs().mean(dim=(1, 2, 3))
        gap4 = (film4 - raw4).abs().mean(dim=(1, 2, 3))
        gap = 0.5 * (gap3 + gap4)
        z_part = self.part_encoder(pooled.mean(dim=0))
        rows = []
        for c in range(ways):
            sel = support_y == c
            shots = pooled[sel]
            if shots.shape[0] == 0:
                raise RuntimeError('class {} missing in support'.format(c))
            appearance = shots.mean(dim=0)
            if shots.shape[0] == 1:
                disp = shots.new_zeros(())
            else:
                disp = shots.std(dim=0, unbiased=False).mean()
            cond = support_cond[sel].mean(dim=0)
            # 缺测维在类均值里仍为 0，与 mask 语义一致
            cond = cond * support_cond_mask[sel].mean(dim=0).clamp(0, 1)
            rows.append(torch.cat([
                z_part,
                appearance,
                cond,
                disp.reshape(1),
                gap[sel].mean().reshape(1),
            ], dim=0))
        return torch.stack(rows, dim=0)

    def _knobs(self, raw3, raw4, film3, film4, support_y, support_cond, support_cond_mask, ways):
        state = self._class_state(
            raw3, raw4, film3, film4, support_y, support_cond, support_cond_mask, ways)
        return self.policy(state)

    def _encode_pair(self, layer3, layer4, rest3, rest4, cond, cond_mask, w3, b3, w4, b4):
        raw3, raw4 = self._project(layer3, layer4, w3, b3, w4, b4)
        rest_p3, rest_p4 = self._project(rest3, rest4, w3, b3, w4, b4)
        film3, film4 = self._restore(rest_p3, rest_p4, cond, cond_mask)
        return raw3, raw4, film3, film4

    def _score_with_knobs(
            self, s_raw3, s_film3, s_raw4, s_film4,
            q_raw3, q_film3, q_raw4, q_film4,
            support_y, ways, g, alpha, gate, tau,
            support_mask, support_has_mask, query_y, query_mask, query_has_mask):
        maps = []
        align_terms = []
        for c in range(ways):
            s3 = self._mix_class(s_raw3, s_film3, g[c], gate[c])
            s4 = self._mix_class(s_raw4, s_film4, g[c], gate[c])
            q3 = self._mix_class(q_raw3, q_film3, g[c], gate[c])
            q4 = self._mix_class(q_raw4, q_film4, g[c], gate[c])
            sel = support_y == c
            w3 = self._proto_weight(support_mask[sel], support_has_mask[sel], s3.shape[-2:])
            w4 = self._proto_weight(support_mask[sel], support_has_mask[sel], s4.shape[-2:])
            proto3 = weighted_spatial_mean(s3[sel], w3).mean(dim=0)
            proto4 = weighted_spatial_mean(s4[sel], w4).mean(dim=0)
            map3 = -((q3 - proto3.view(1, -1, 1, 1)) ** 2).sum(dim=1)
            map4 = -((q4 - proto4.view(1, -1, 1, 1)) ** 2).sum(dim=1)
            map4 = F.interpolate(
                map4.unsqueeze(1), size=map3.shape[-2:],
                mode='bilinear', align_corners=False).squeeze(1)
            maps.append((alpha[c, 0] * map3 + alpha[c, 1] * map4) / tau[c])
            qsel = query_y == c
            if bool(qsel.any()):
                qw3 = self._proto_weight(query_mask[qsel], query_has_mask[qsel], q3.shape[-2:])
                qw4 = self._proto_weight(query_mask[qsel], query_has_mask[qsel], q4.shape[-2:])
                q_mean3 = weighted_spatial_mean(q3[qsel], qw3).mean(dim=0)
                q_mean4 = weighted_spatial_mean(q4[qsel], qw4).mean(dim=0)
                same = 0.5 * (
                    ((q_mean3 - proto3) ** 2).mean() + ((q_mean4 - proto4) ** 2).mean())
                align_terms.append(same)
        score = torch.stack(maps, dim=1)
        logits = topk_mean(score, self.topk_ratio)
        if align_terms:
            align = torch.stack(align_terms).mean()
        else:
            align = score.new_zeros(())
        return logits, score, align

    def _query_scores(self, w3, b3, w4, b4, bundles, ways):
        (s_l3, s_l4, s_r3, s_r4, q_l3, q_l4, q_r3, q_r4,
         support_y, support_cond, support_cond_mask, query_cond, query_cond_mask,
         support_mask, support_has_mask, query_y, query_mask, query_has_mask) = bundles
        s_raw3, s_raw4, s_film3, s_film4 = self._encode_pair(
            s_l3, s_l4, s_r3, s_r4, support_cond, support_cond_mask, w3, b3, w4, b4)
        q_raw3, q_raw4, q_film3, q_film4 = self._encode_pair(
            q_l3, q_l4, q_r3, q_r4, query_cond, query_cond_mask, w3, b3, w4, b4)
        g, alpha, gate, tau = self._knobs(
            s_raw3, s_raw4, s_film3, s_film4,
            support_y, support_cond, support_cond_mask, ways)
        logits, score, align = self._score_with_knobs(
            s_raw3, s_film3, s_raw4, s_film4,
            q_raw3, q_film3, q_raw4, q_film4,
            support_y, ways, g, alpha, gate, tau,
            support_mask, support_has_mask, query_y, query_mask, query_has_mask)
        return logits, score, g, alpha, gate, tau, align

    def _support_logits(
            self, w3, b3, w4, b4, s_l3, s_l4, s_r3, s_r4,
            support_y, support_cond, support_cond_mask, ways):
        raw3, raw4, film3, film4 = self._encode_pair(
            s_l3, s_l4, s_r3, s_r4, support_cond, support_cond_mask, w3, b3, w4, b4)
        g, alpha, gate, tau = self._knobs(
            raw3, raw4, film3, film4, support_y, support_cond, support_cond_mask, ways)
        return loo_pooled_logits(
            raw3, film3, raw4, film4, support_y, ways, g, alpha, gate, tau)

    def forward(
            self,
            support_x,
            support_y,
            query_x,
            support_cond,
            query_cond,
            query_y,
            ways,
            support_cond_mask=None,
            query_cond_mask=None,
            support_x_restored=None,
            query_x_restored=None,
            support_mask=None,
            query_mask=None,
            support_has_mask=None,
            query_has_mask=None,
    ):
        if support_cond_mask is None:
            support_cond_mask = torch.ones_like(support_cond)
        if query_cond_mask is None:
            query_cond_mask = torch.ones_like(query_cond)
        if support_x_restored is None:
            support_x_restored = support_x
        if query_x_restored is None:
            query_x_restored = query_x
        if support_mask is None:
            support_mask = support_x.new_zeros(
                support_x.shape[0], 1, support_x.shape[2], support_x.shape[3])
        if query_mask is None:
            query_mask = query_x.new_zeros(
                query_x.shape[0], 1, query_x.shape[2], query_x.shape[3])
        if support_has_mask is None:
            support_has_mask = torch.zeros(
                support_x.shape[0], dtype=torch.bool, device=support_x.device)
        if query_has_mask is None:
            query_has_mask = torch.zeros(
                query_x.shape[0], dtype=torch.bool, device=query_x.device)

        s_l3, s_l4 = self.backbone(support_x)
        q_l3, q_l4 = self.backbone(query_x)
        s_r3, s_r4 = self.backbone(support_x_restored)
        q_r3, q_r4 = self.backbone(query_x_restored)

        fast = (self.proj3.weight, self.proj3.bias, self.proj4.weight, self.proj4.bias)
        trainable = [p.requires_grad for p in fast]
        create_graph = bool(self.training and self.adapt_steps > 0)
        if self.adapt_steps > 0 and any(trainable):
            with torch.enable_grad():
                for _ in range(self.adapt_steps):
                    s_logits = self._support_logits(
                        *fast, s_l3, s_l4, s_r3, s_r4,
                        support_y, support_cond, support_cond_mask, ways)
                    loss_s = F.cross_entropy(s_logits, support_y)
                    active = [p for p, flag in zip(fast, trainable) if flag]
                    grads = torch.autograd.grad(
                        loss_s, active, create_graph=create_graph, allow_unused=False)
                    step = iter(grads)
                    fast = tuple(
                        p - self.adapt_lr * next(step) if flag else p
                        for p, flag in zip(fast, trainable))

        bundles = (
            s_l3, s_l4, s_r3, s_r4, q_l3, q_l4, q_r3, q_r4,
            support_y, support_cond, support_cond_mask, query_cond, query_cond_mask,
            support_mask, support_has_mask, query_y, query_mask, query_has_mask,
        )
        logits, score, g, alpha, gate, tau, align = self._query_scores(*fast, bundles, ways)
        pred = logits.argmax(dim=-1)
        peak = peak_yx(score, pred)
        gt_yx, gt_valid = mask_centroid_yx(query_mask)
        gt_valid = gt_valid & query_has_mask
        if not self.training:
            logits = logits.detach()
            score = score.detach()
            peak = peak.detach()
            g = g.detach()
            alpha = alpha.detach()
            gate = gate.detach()
            tau = tau.detach()
            align = align.detach()
            gt_yx = gt_yx.detach()
            gt_valid = gt_valid.detach()
        return {
            'logits': logits,
            'score_map': score,
            'pred': pred if self.training else pred.detach(),
            'peak_yx': peak,
            'gt_yx': gt_yx,
            'gt_valid': gt_valid,
            'gate_g': g,
            'scale_alpha': alpha,
            'channel_gate': gate,
            'distance_tau': tau,
            'source_proto': align,
        }


def _masked_mean(feat, mask):
    """feat (N,C,H,W), mask (N,H,W) bool/float → (N,C)。空掩码退回全局均值。"""
    w = mask.float().unsqueeze(1)
    denom = w.sum(dim=(2, 3)).clamp_min(1.0)
    filled = (feat * w).sum(dim=(2, 3)) / denom
    empty = mask.flatten(1).sum(dim=1) < 0.5
    if bool(empty.any()):
        gap = feat.mean(dim=(2, 3))
        filled = torch.where(empty.view(-1, 1), gap, filled)
    return filled


def _unit_rows(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class DualRegionBetaDetector(nn.Module):
    """课程学习：冻结骨干 + layer3 投影 + episode 级 β + core/ring cosine。"""

    def __init__(
            self,
            feat_dim=128,
            cond_dim=6,
            dilate_px=16,
            pretrained_backbone=True,
            freeze_backbone=True,
            use_policy=True,
            fixed_beta=0.25,
            beta_center=0.25,
            beta_scale=0.25,
            feature_layer=3,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.cond_dim = int(cond_dim)
        self.dilate_px = int(dilate_px)
        self.use_policy = bool(use_policy)
        self.fixed_beta = float(fixed_beta)
        self.backbone = SpatialResNet18(
            pretrained=pretrained_backbone, freeze=freeze_backbone)
        self.feature_layer = int(feature_layer)
        if self.feature_layer not in (3, 4):
            raise ValueError('feature_layer must be 3 or 4')
        # Default keeps old layer3 checkpoint keys unchanged.
        if self.feature_layer == 3:
            self.proj3 = nn.Conv2d(256, self.feat_dim, kernel_size=1)
        else:
            self.proj4 = nn.Conv2d(512, self.feat_dim, kernel_size=1)
        # 低维统计：6 标量 + 工况均值 + 工况 mask
        self.policy = EpisodeBetaPolicy(
            state_dim=6 + 2 * self.cond_dim,
            beta_center=beta_center,
            beta_scale=beta_scale,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.backbone.frozen:
            self.backbone.eval()
        return self

    def _maps(self, x):
        layer3, layer4 = self.backbone(x)
        if self.feature_layer == 3:
            return self.proj3(layer3)
        return self.proj4(layer4)

    def _core_ring(self, feat, masks):
        """core 为掩码落到特征图的格子；ring 先在输入分辨率按 dilate_px 像素膨胀，再落到特征图，与工业 encode_core_context 一致。
        空核时用膨胀区作为 core，不再退回整图均值。
        """
        n, _, h, w = feat.shape
        del n
        mask_f = masks.float()
        if mask_f.dim() == 3:
            mask_f = mask_f.unsqueeze(1)
        core_m = F.interpolate(mask_f, size=(h, w), mode='nearest').squeeze(1) >= 0.5
        if self.dilate_px > 0:
            k = 2 * int(self.dilate_px) + 1
            dil_hi = F.max_pool2d(
                mask_f, kernel_size=k, stride=1, padding=int(self.dilate_px))
            dil_m = F.interpolate(dil_hi, size=(h, w), mode='nearest').squeeze(1) >= 0.5
        else:
            dil_m = core_m
        ring_m = dil_m & (~core_m)
        has_core = core_m.flatten(1).any(dim=1).view(-1, 1, 1)
        core_m = torch.where(has_core, core_m, dil_m)
        has_ring = ring_m.flatten(1).any(dim=1).view(-1, 1, 1)
        ring_m = torch.where(has_ring, ring_m, dil_m)
        return _masked_mean(feat, core_m), _masked_mean(feat, ring_m)

    def support_stats(self, core, ring, y, ways, cond, cond_mask):
        device = core.device
        stats = []
        stats.append(core.var(dim=0).mean().reshape(1))
        stats.append(ring.var(dim=0).mean().reshape(1))
        stats.append((_unit_rows(core) * _unit_rows(ring)).sum(dim=-1).mean().reshape(1))
        intra = []
        protos = []
        for c in range(ways):
            sel = y == c
            if not bool(sel.any()):
                continue
            shots = core[sel]
            protos.append(shots.mean(dim=0))
            if shots.shape[0] > 1:
                intra.append(shots.std(dim=0, unbiased=False).mean())
        if intra:
            stats.append(torch.stack(intra).mean().reshape(1))
        else:
            stats.append(core.new_zeros(1))
        if len(protos) >= 2:
            p = torch.stack(protos)
            p = _unit_rows(p)
            sim = p @ p.t()
            eye = torch.eye(sim.shape[0], device=device, dtype=torch.bool)
            stats.append(((1.0 - sim)[~eye]).mean().reshape(1))
        else:
            stats.append(core.new_zeros(1))
        stats.append(torch.tensor([float(core.shape[0])], device=device))
        stats.append(cond.mean(dim=0))
        stats.append(cond_mask.mean(dim=0))
        return torch.cat(stats, dim=0)

    def forward(
            self,
            support_x,
            support_y,
            query_x,
            support_cond,
            query_cond,
            query_y,
            ways,
            support_cond_mask=None,
            query_cond_mask=None,
            support_mask=None,
            query_mask=None,
            support_has_mask=None,
            query_has_mask=None,
            **_unused,
    ):
        del query_cond, query_cond_mask, support_has_mask, query_has_mask, query_y, _unused
        if support_cond_mask is None:
            support_cond_mask = torch.ones_like(support_cond)
        if support_mask is None or query_mask is None:
            raise RuntimeError('DualRegionBetaDetector 需要支持集和查询集掩码')
        s_feat = self._maps(support_x)
        q_feat = self._maps(query_x)
        s_core, s_ring = self._core_ring(s_feat, support_mask)
        q_core, q_ring = self._core_ring(q_feat, query_mask)
        if self.use_policy:
            state = self.support_stats(
                s_core, s_ring, support_y, ways, support_cond, support_cond_mask)
            beta = self.policy(state)[0]
        else:
            beta = s_core.new_tensor(self.fixed_beta)
        logits = []
        for c in range(ways):
            sel = support_y == c
            p_core = _unit_rows(s_core[sel].mean(dim=0, keepdim=True))
            p_ring = _unit_rows(s_ring[sel].mean(dim=0, keepdim=True))
            s_c = (_unit_rows(q_core) * p_core).sum(dim=-1)
            s_r = (_unit_rows(q_ring) * p_ring).sum(dim=-1)
            logits.append(s_c + beta * s_r)
        logits = torch.stack(logits, dim=-1)
        return {
            'logits': logits,
            'beta': beta.detach() if not self.training else beta,
            'gate_g': beta.detach().expand(ways),
            'scale_alpha': logits.new_zeros(ways, 2),
            'source_proto': logits.new_zeros(()),
            'peak_yx': logits.new_zeros(logits.shape[0], 2),
            'gt_yx': logits.new_zeros(logits.shape[0], 2),
            'gt_valid': torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device),
        }
