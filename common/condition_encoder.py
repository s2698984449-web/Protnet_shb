"""Condition encoder + FiLM gating for multi-source domain pretraining.

工况 (condition) 描述的是影响成像的环境变量 (光照、噪声、对比度...),
而非产品种类 (产品种类由 product_id 表达). 两路编码后通过 FiLM
(Feature-wise Linear Modulation) 调制 ResNet18 backbone 的中间特征.

设计原则:
  - ConditionEncoder 与 ProductEncoder 独立, 避免 product leak 干扰工况表达.
  - FiLM 调制 layer3 之后 (语义最丰富的中间层), 不在 GAP 之后.
  - 当 condition 输入缺失或为零时, FiLM γ=β=0 → 等价 identity (safe fallback).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionEncoder(nn.Module):
    """Encode a n_conditions-dim condition vector (multi-hot) into a dense embedding.

    工况 7 维 (默认, 见 config/common.yaml 的 CONDITION_SPEC):
      c0 = bright       (高亮)
      c1 = dim          (低亮)
      c2 = low_contrast (低对比度)
      c3 = noisy        (高噪声)
      c4 = blur         (运动模糊)
      c5 = color_cast   (色偏)
      c6 = view         (视角占位 / 标注 one-hot)

    当 source 数据无工况标签时, 全部填 0 (FiLM 退化为 identity, 配合方案 B 限幅).

    Args:
        n_conditions: 条件数 (默认 7, 与 CONDITION_SPEC 对齐).
        d: 输出 embedding 维度.
    """

    def __init__(self, n_conditions: int = 7, d: int = 128):
        super().__init__()
        self.n_conditions = n_conditions
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(n_conditions, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, n_conditions) one-hot/multi-hot or (B,) long index."""
        if c.dim() == 1:
            c = F.one_hot(c.long(), self.n_conditions).float()
        elif c.dim() == 2 and c.size(1) != self.n_conditions:
            raise ValueError(
                f'ConditionEncoder expects (B, {self.n_conditions}), got {tuple(c.shape)}')
        return self.net(c)


class ProductEncoder(nn.Module):
    """Encode a product id (long tensor) into a dense embedding.

    跨源域共用一个 product 字典:
      MVTec 15 类 (0-14) + GC10 8 类 (15-22) + NEU 6 类 (23-28)
      → 总共 29 个 unique product slot.

    Args:
        n_products: product 总数.
        d: 输出 embedding 维度.
    """

    def __init__(self, n_products: int = 29, d: int = 128):
        super().__init__()
        self.n_products = n_products
        self.d = d
        self.embed = nn.Embedding(n_products, d)

    def forward(self, pid: torch.Tensor) -> torch.Tensor:
        """pid: (B,) long product id."""
        return self.embed(pid.long())


class FiLMGate(nn.Module):
    """FiLM 调制: F' = clip(1 + gamma(c), 1-ρ, 1+ρ) ⊙ F + clip(beta(c), -ρ, +ρ).

    Args:
        d_cond:  工况 embedding 维度 (来自 ConditionEncoder).
        d_feat: backbone 特征通道数.
        rho:    FiLM 限幅半径.  1+ρ = γ 上限 (避免训练崩溃);
                ρ 也是 β 的上限 (避免 shift 主导 backbone 输出).
                默认 0.5 (即 γ ∈ [-0.5, +0.5], β ∈ [-0.5*σ_F, +0.5*σ_F]).
        gate_clamp:    是否对最终 feature 再次按 σ 限幅 (方案 B 增强, 防 outlier).
        gate_clamp_std:限幅 σ 倍数 (e.g. 5 → ±5σ).
    """

    def __init__(self, d_cond: int = 128, d_feat: int = 256,
                 rho: float = 0.5, gate_clamp: bool = True,
                 gate_clamp_std: float = 5.0):
        super().__init__()
        self.gamma = nn.Linear(d_cond, d_feat)
        self.beta = nn.Linear(d_cond, d_feat)
        # 零初始化 → FiLM 输出 = 1*F + 0 = F (identity baseline)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        self.d_feat = d_feat
        self.rho = float(rho)
        self.gate_clamp = gate_clamp
        self.gate_clamp_std = float(gate_clamp_std)

    def forward(self, feat: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W); cond_emb: (B, d_cond)."""
        if cond_emb.size(-1) != self.gamma.in_features:
            raise ValueError(
                f'FiLMGate: cond dim {cond_emb.size(-1)} != expected {self.gamma.in_features}')
        g = self.gamma(cond_emb).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(cond_emb).unsqueeze(-1).unsqueeze(-1)
        # ── 方案 B: 限幅 ──
        # γ ∈ [-ρ, +ρ] → 1+γ ∈ [1-ρ, 1+ρ]  (防止乘性放大)
        # β ∈ [-ρ, +ρ]  (防止加性偏置主导)
        g = torch.clamp(g, -self.rho, self.rho)
        b = torch.clamp(b, -self.rho, self.rho)
        out = feat * (1.0 + g) + b
        if self.gate_clamp:
            # 按 σ 自适应限幅 (per-sample), 防止 outlier 破坏下游
            mu = out.mean(dim=(2, 3), keepdim=True)
            sd = out.std(dim=(2, 3), keepdim=True) + 1e-6
            out = torch.clamp(out, mu - self.gate_clamp_std * sd,
                                  mu + self.gate_clamp_std * sd)
        return out


class DomainReliability(nn.Module):
    """源域可靠性学习 — 当一个 episode 来自多源域混合时, 学习每域的可信度.

    输入: query 特征 (B, D) + domain_id (B,)
    输出: scalar reliability ∈ (0, 1) per sample (B,)

    训练信号: 通过 ProtoNet 距离匹配的 softmax 概率反向传播.
    设计: 一个非常轻量的 2 层 MLP, 输出 sigmoid ∈ (0,1).
    """

    def __init__(self, feat_dim: int = 128, n_domains: int = 3, hidden: int = 64):
        super().__init__()
        self.n_domains = n_domains
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim + n_domains, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, query_emb: torch.Tensor, domain_one_hot: torch.Tensor) -> torch.Tensor:
        """query_emb: (B, D); domain_one_hot: (B, n_domains)."""
        h = torch.cat([query_emb, domain_one_hot], dim=-1)
        return torch.sigmoid(self.net(h)).squeeze(-1)  # (B,)