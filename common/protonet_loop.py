"""Prototypical Networks episode loss / inference (compatible with multimodal encoder).

改进点（方案 C）:
  ① Trim mean pooling   — 去除 support 中异常样本（最高/最低 10%）后再平均
  ② Mahalanobis 距离   — 考虑特征协方差，比欧氏/余弦更判别性
  ③ Mean-Shift 原型聚类 — 每类多个聚类中心表示，不用单一均值
  ④ Adaptive Prototype — prototype 与 query 实时交互（cross-attention）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from sklearn.cluster import MeanShift, estimate_bandwidth
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

_EPS = 1e-8


def _normalize(x):
    return x / (x.norm(p=2, dim=1, keepdim=True) + _EPS)


# ---------------------------------------------------------------------------
# ① Trim mean pooling：去除 support 中 10% 最高/最低值后取平均
# ---------------------------------------------------------------------------
def _trim_mean(values, trim_ratio=0.1):
    """Trim mean: remove top/bottom trim_ratio fraction before taking mean."""
    if values.numel() <= 2:
        return values.mean(dim=0)
    n = values.size(0)
    k = max(1, int(n * trim_ratio))
    sorted_vals, _ = values.sort(dim=0)
    return sorted_vals[k:n - k].mean(dim=0)


# ---------------------------------------------------------------------------
# 原型构建（支持多种策略）
# ---------------------------------------------------------------------------
def build_prototypes(support_emb, support_labels, ways,
                     prototype_type='mean',
                     trim_ratio=0.1,
                     bandwidth=None):
    """Build class prototypes.

    Args:
        support_emb:    (N, D) support embedding
        support_labels: (N,)  class labels
        ways:           number of classes
        prototype_type: 'mean' | 'trim_mean' | 'mean_shift'
        trim_ratio:     fraction to trim for trim_mean
        bandwidth:      bandwidth for MeanShift (auto if None)
    Returns:
        prototypes: (ways, D) or dict for mean_shift
    """
    prototypes = []
    for cls in range(ways):
        mask = support_labels == cls
        if not mask.any():
            # 空类 → 全局均值作为 fallback
            prototypes.append(support_emb.mean(dim=0))
            continue

        emb = support_emb[mask]  # (N_c, D)

        if prototype_type == 'trim_mean':
            p = _trim_mean(emb, trim_ratio=trim_ratio)
            prototypes.append(p)

        elif prototype_type == 'mean_shift':
            if not _HAS_SKLEARN:
                prototypes.append(emb.mean(dim=0))
                continue
            if bandwidth is None:
                bandwidth = _estimate_bandwidth(emb)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            try:
                ms.fit(emb.cpu().numpy())
                centers = torch.from_numpy(ms.cluster_centers_).to(emb.device)
            except Exception:
                centers = emb.mean(dim=0, keepdim=True)
            prototypes.append(centers)

        else:  # 'mean'
            prototypes.append(emb.mean(dim=0))

    if prototype_type == 'mean_shift':
        # 返回 (ways, None) 的列表，每项是 (K_c, D) 的聚类中心
        return prototypes
    return torch.stack(prototypes, dim=0)  # (ways, D)


# ---------------------------------------------------------------------------
# Mahalanobis 距离（带可学习缩放因子）
# ---------------------------------------------------------------------------
class MahalanobisHead(torch.nn.Module):
    """Mahalanobis 距离头: d(x,y) = sqrt((x-y)^T Σ^{-1} (x-y))
    用对角协方差近似（稳定且省内存），训练时自动学习每类的 Σ。"""
    def __init__(self, feat_dim, ways, init_reg=1e-3):
        super().__init__()
        self.feat_dim = feat_dim
        self.ways = ways
        # 每类的对数方差（稳定求逆）
        self.log_vars = torch.nn.Parameter(
            torch.zeros(ways, feat_dim).fill_(init_reg).log())

    def forward(self, query_emb, prototypes, support_labels):
        """Compute Mahalanobis logits.

        Args:
            query_emb:    (N, D)
            prototypes:   (ways, D)  mean prototypes
            support_labels: (N_s,)  only used to accumulate per-class stats at train time
        Returns:
            logits: (N, ways)
        """
        vars_ = self.log_vars.exp() + _EPS  # (ways, D)
        diff = query_emb.unsqueeze(1) - prototypes.unsqueeze(0)  # (N, ways, D)
        # Mahalanobis: sum over D: diff^2 / var
        mahal_sq = (diff ** 2) / vars_.unsqueeze(0)  # (N, ways, D)
        mahal = -mahal_sq.sum(dim=2)  # (N, ways)  越大越近
        return mahal


# ---------------------------------------------------------------------------
# ④ Adaptive Prototype：prototype 与 query 实时交互（cross-attention）
# ---------------------------------------------------------------------------
class AdaptivePrototype(nn.Module):
    """Cross-attention based prototype refinement.

    Idea (from EFINet): static mean prototypes miss query-specific patterns.
    This module lets query features attend to class prototypes and produce
    query-aware adjusted prototypes before computing distances.

    Flow:
        prototypes (ways, D)  ──key/value──▶  MultiheadAttention
        query_emb   (N, D)    ──query  ───▶  adjusted_prototypes (N, ways, D)
        Then: distance(query_emb, adjusted_prototype) per query sample.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} % num_heads {num_heads} != 0"
        self.head_dim = embed_dim // num_heads

        # Project prototypes to key/value, query comes from query_emb directly
        self.proj_kv = nn.Linear(embed_dim, embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim)  # optional: transform query
        self.attn_dropout = nn.Dropout(dropout)
        # Output projection to restore prototype dimension
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # gating: residual-like blend between original prototype and attended output
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, prototypes, query_emb):
        """Adjust prototypes conditioned on query embeddings.

        Args:
            prototypes:   (ways, D)   class prototypes from support
            query_emb:    (N, D)      query embeddings
        Returns:
            adjusted:    (N, ways, D) adjusted prototypes per query sample
        """
        # prototypes:  (ways, D) → (ways, 1, D) → (ways, N, D) via broadcasting below
        # query_emb:   (N, D) → (N, D) → (N, 1, D) for residual

        # Self-attention: query=query_emb, key/value=prototypes
        #   shape: (N, heads, ways, head_dim) after rearrange
        # Standard: q_proj = query_emb (N, D), kv_proj = prototypes (ways, D)
        q = self.proj_q(query_emb)                        # (N, D)
        kv = self.proj_kv(prototypes)                     # (ways, D)

        # Multihead attention: q unsqueeze(1) → (N, 1, D), kv unsqueeze(0) → (1, ways, D)
        # PyTorch MHADoes not support cross-attn directly, so we use scaled_dot_product_attention
        q = q.unsqueeze(1)                               # (N, 1, D)
        kv = kv.unsqueeze(0)                             # (1, ways, D)

        # scaled_dot_product_attention returns just the attention output (no weights) when
# need_weights=False (default in recent PyTorch).
        attn_out = F.scaled_dot_product_attention(
            q, kv, kv,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )                                                 # (N, 1, D)
        attn_out = attn_out.squeeze(1)                   # (N, D)
        attn_out = self.out_proj(attn_out)               # (N, D)

        # Gate: blend original prototype with attended output
        g = torch.sigmoid(self.gate)                     # scalar in (0,1)
        # For each query, blend its attended proto with original
        # attn_out is per-query; expand to (N, ways, D)
        adjusted = g * attn_out.unsqueeze(1) + (1 - g) * prototypes.unsqueeze(0)
        return adjusted  # (N, ways, D)


def adaptive_prototypical_logits(query_emb, support_emb, support_labels, ways,
                                  prototype_type='mean', trim_ratio=0.1,
                                  adaptive_proto_module=None):
    """Adaptive Prototype logits: first build mean/trim_mean prototypes,
    then refine them with cross-attention on query_emb before distance.

    Args:
        query_emb:      (N, D)
        support_emb:    (N_s, D)
        support_labels: (N_s,)
        ways:           int
        prototype_type: 'mean' | 'trim_mean'
        trim_ratio:     float
        adaptive_proto_module: AdaptivePrototype or None (skip adaptation)
    Returns:
        logits: (N, ways)
    """
    # 1. Build static prototypes from support
    static_protos = build_prototypes(
        support_emb, support_labels, ways,
        prototype_type=prototype_type, trim_ratio=trim_ratio)  # (ways, D)

    if adaptive_proto_module is None:
        # Fallback to plain euclidean
        n, m = query_emb.size(0), static_protos.size(0)
        q = query_emb.unsqueeze(1).expand(n, m, -1)
        p = static_protos.unsqueeze(0).expand(n, m, -1)
        return -((q - p) ** 2).sum(dim=2)

    # 2. Adjust prototypes via cross-attention
    adjusted_protos = adaptive_proto_module(static_protos, query_emb)  # (N, ways, D)

    # 3. Distance: euclidean between each query and its adjusted prototype
    n = query_emb.size(0)
    q_expanded = query_emb.unsqueeze(1)                              # (N, 1, D)
    logits = -((q_expanded - adjusted_protos) ** 2).sum(dim=2)     # (N, ways)
    return logits


# ---------------------------------------------------------------------------
# 核心 logit 函数（支持多种距离 + 原型策略组合）
# ---------------------------------------------------------------------------
def prototypical_logits(query_emb, support_emb, support_labels, ways,
                        distance='euclidean',
                        prototype_type='mean',
                        trim_ratio=0.1,
                        mahal_head=None):
    """Nearest-prototype logits: higher = closer to class prototype.

    Args:
        query_emb:      (N, D)
        support_emb:    (N_s, D)
        support_labels: (N_s,)
        ways:           int
        distance:        'euclidean' | 'cosine' | 'mahalanobis'
        prototype_type: 'mean' | 'trim_mean' | 'mean_shift'
        trim_ratio:     float, fraction to trim for trim_mean
        mahal_head:      MahalanobisHead or None
    Returns:
        logits: (N, ways)  (for mean_shift, returns (N, total_K) concatenated logits)
    """
    # ---- 构建原型 ----
    if prototype_type == 'mean_shift':
        prototype_list = build_prototypes(
            support_emb, support_labels, ways,
            prototype_type='mean_shift', bandwidth=None)
        # prototype_list[cls] = (K_c, D)
        all_centers = torch.cat(prototype_list, dim=0)   # (total_K, D)
        cls_sizes = [p.size(0) for p in prototype_list]
        cls_offsets = [0] + list(torch.tensor(cls_sizes).cumsum(0).tolist())

        if distance == 'cosine':
            all_centers = _normalize(all_centers)
            q = _normalize(query_emb)
            raw_logits = torch.mm(q, all_centers.t())  # (N, total_K)
        elif distance == 'mahalanobis' and mahal_head is not None:
            raw_logits = mahal_head(query_emb, all_centers, support_labels)
        else:  # euclidean
            n, m = query_emb.size(0), all_centers.size(0)
            q = query_emb.unsqueeze(1).expand(n, m, -1)
            c = all_centers.unsqueeze(0).expand(n, m, -1)
            raw_logits = -((q - c) ** 2).sum(dim=2)

        # 把同一类的多个聚类中心的 logits 取 max（聚类投票）
        logits = torch.zeros(query_emb.size(0), ways, device=query_emb.device)
        for cls in range(ways):
            start, end = cls_offsets[cls], cls_offsets[cls + 1]
            logits[:, cls] = raw_logits[:, start:end].max(dim=1).values
        return logits

    # ---- 普通原型（单一均值） ----
    prototypes = build_prototypes(
        support_emb, support_labels, ways,
        prototype_type=prototype_type, trim_ratio=trim_ratio)

    if distance == 'cosine':
        query_emb = _normalize(query_emb)
        prototypes = _normalize(prototypes)
        return torch.mm(query_emb, prototypes.t())

    if distance == 'mahalanobis' and mahal_head is not None:
        return mahal_head(query_emb, prototypes, support_labels)

    # 欧氏距离
    n, m = query_emb.size(0), prototypes.size(0)
    q = query_emb.unsqueeze(1).expand(n, m, -1)
    p = prototypes.unsqueeze(0).expand(n, m, -1)
    return -((q - p) ** 2).sum(dim=2)


# ---------------------------------------------------------------------------
# Episode loss / evaluation
# ---------------------------------------------------------------------------
def protonet_episode_loss(model, support, query, domain_meta,
                           distance='euclidean',
                           prototype_type='mean',
                           mahal_head=None,
                           adaptive_proto=None):
    """ProtoNet episode loss with optional Adaptive Prototype (cross-attn).

    Args:
        model:           the model (must have model.embed and model.adaptive_proto)
        support:         (s_rgb, s_th, s_us, s_y) 5-D or 3-D tensors
        query:           (q_rgb, q_th, q_us, q_y)
        domain_meta:     domain metadata
        distance:        'euclidean' | 'cosine' | 'mahalanobis'
        prototype_type:  'mean' | 'trim_mean'
        mahal_head:      MahalanobisHead or None
        adaptive_proto:  AdaptivePrototype or None (skip if None)
    Returns:
        loss, acc
    """
    from .protonet_loop import (
        prototypical_logits, adaptive_prototypical_logits,
        build_prototypes as _build_protos, _normalize
    )
    s_rgb, s_th, s_us, s_y = support
    q_rgb, q_th, q_us, q_y = query
    if s_rgb.dim() == 5:
        s_rgb = s_rgb.reshape(s_rgb.shape[0] * s_rgb.shape[1], *s_rgb.shape[2:])
    elif s_rgb.dim() == 3:
        s_rgb = s_rgb.unsqueeze(0)
    if q_rgb.dim() == 5:
        q_rgb = q_rgb.reshape(q_rgb.shape[0] * q_rgb.shape[1], *q_rgb.shape[2:])
    elif q_rgb.dim() == 3:
        q_rgb = q_rgb.unsqueeze(0)
    if s_rgb.dim() == 3:
        s_rgb = s_rgb.unsqueeze(0)
    if q_rgb.dim() == 3:
        q_rgb = q_rgb.unsqueeze(0)
    support = (s_rgb, s_th, s_us, s_y)
    query = (q_rgb, q_th, q_us, q_y)
    s_emb = model.embed(s_rgb, s_th, s_us, domain_meta)
    q_emb = model.embed(q_rgb, q_th, q_us, domain_meta)
    ways = int(max(s_y.max().item(), q_y.max().item())) + 1

    if adaptive_proto is not None:
        logits = adaptive_prototypical_logits(
            q_emb, s_emb, s_y, ways,
            prototype_type=prototype_type,
            adaptive_proto_module=adaptive_proto)
    else:
        logits = prototypical_logits(
            q_emb, s_emb, s_y, ways,
            distance=distance, prototype_type=prototype_type,
            mahal_head=mahal_head)
    loss = F.cross_entropy(logits, q_y)
    acc = (logits.argmax(dim=-1) == q_y).float().mean().item()
    return loss, acc


def evaluate_protonet_query(model, support, query, domain_meta,
                             distance='euclidean',
                             prototype_type='mean',
                             mahal_head=None,
                             adaptive_proto=None):
    s_rgb, s_th, s_us, s_y = support
    q_rgb, q_th, q_us, q_y = query
    with torch.no_grad():
        s_emb = model.embed(s_rgb, s_th, s_us, domain_meta)
        q_emb = model.embed(q_rgb, q_th, q_us, domain_meta)
        ways = int(max(s_y.max().item(), q_y.max().item())) + 1
        if adaptive_proto is not None:
            from .protonet_loop import (
                adaptive_prototypical_logits,
                prototypical_logits as _proto_logits,
            )
            logits = adaptive_prototypical_logits(
                q_emb, s_emb, s_y, ways,
                prototype_type=prototype_type,
                adaptive_proto_module=adaptive_proto)
        else:
            logits = prototypical_logits(
                q_emb, s_emb, s_y, ways,
                distance=distance, prototype_type=prototype_type,
                mahal_head=mahal_head)
        acc = (logits.argmax(dim=-1) == q_y).float().mean().item()
        loss = F.cross_entropy(logits, q_y).item()
    return loss, acc


# ---------------------------------------------------------------------------
# 训练时累积 Mahalanobis 协方差（每类独立）
# ---------------------------------------------------------------------------
class MahalStatistics:
    """训练时收集每个类的 embedding 统计量，用于初始化 MahalanobisHead。"""
    def __init__(self, feat_dim, ways, device):
        self.feat_dim = feat_dim
        self.ways = ways
        self.sum = torch.zeros(ways, feat_dim, device=device)
        self.sum_sq = torch.zeros(ways, feat_dim, device=device)
        self.count = torch.zeros(ways, device=device)

    def update(self, emb, labels):
        for cls in range(self.ways):
            mask = labels == cls
            if mask.any():
                self.sum[cls] += emb[mask].sum(dim=0)
                self.sum_sq[cls] += (emb[mask] ** 2).sum(dim=0)
                self.count[cls] += mask.sum().item()

    def init_mahal_head(self, reg=1e-3):
        vars_ = torch.zeros(self.ways, self.feat_dim, device=self.sum.device)
        for cls in range(self.ways):
            if self.count[cls] > 1:
                mean = self.sum[cls] / self.count[cls]
                var = (self.sum_sq[cls] / self.count[cls]) - (mean ** 2)
                var = torch.clamp(var, min=reg)
            else:
                var = torch.ones_like(self.sum[cls]) * reg
            vars_[cls] = var
        log_vars = torch.log(vars_)
        head = MahalanobisHead(self.feat_dim, self.ways)
        head.log_vars.data = log_vars
        return head


# ---------------------------------------------------------------------------
# 多任务评估入口（供 eval_loop.py 调用）
# ---------------------------------------------------------------------------
def evaluate_tasks_protonet(
        model, test_tasks, domain_meta, shots, ways, n_tasks, device,
        distance='euclidean', prototype_type='mean', mahal_head=None,
        adaptive_proto=None,
        _split_support_query=None):
    """
    Evaluate a ProtoNet on n_tasks test episodes.

    Args:
        model:          the trained ProtoNet model (forward() returns embeddings)
        test_tasks:     TaskDataset yielding episodes
        domain_meta:     per-class domain metadata (labels → domain_id)
        shots:          number of support shots per class
        ways:           number of classes per episode
        n_tasks:        number of episodes to evaluate
        device:         torch device
        distance:       'euclidean' | 'mahalanobis'
        prototype_type: 'mean' | 'trim_mean'
        mahal_head:     optional MahalanobisHead (required if distance='mahalanobis')
        _split_support_query: internal; if None, imported lazily to avoid circular import

    Returns:
        mean_acc (float), std_acc (float), list of per-episode accuracies
    """
    global split_support_query
    try:
        from .maml_loop import split_support_query as _ssq
        split_support_query = _ssq
    except ImportError:
        pass
    if _split_support_query is not None:
        _ssq_fn = _split_support_query
    else:
        _ssq_fn = split_support_query

    accs = []
    for _ in range(n_tasks):
        batch = test_tasks.sample()
        support, query = _ssq_fn(batch, shots, ways, device)
        with torch.no_grad():
            _, acc = evaluate_protonet_query(
                model, support, query, domain_meta,
                distance=distance, prototype_type=prototype_type,
                mahal_head=mahal_head,
                adaptive_proto=adaptive_proto)
        accs.append(acc)
    arr = np.array(accs, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), accs
