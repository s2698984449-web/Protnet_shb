"""条件化模块知识迁移决策：选卡、Null、迁移强度 ρ。

相似度只负责召回；MLP 结合任务签名差与卡片历史效用选卡。
Null 与 ρ 分离：训练/正式评价均为 soft (1-Null)*ρ。
ψ 用 Query 反事实效用 U=L_Null-L_Borrow 更新，不再使用一阶梯度代理。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.retrieval_memory import MODULES, load_retrieval_config

_EPS = 1e-12
POLICY_KIND = "conditional_transfer_v4"
UTILITY_FEATURE_DIM = 4  # mean, std, log1p(count), neg_rate
FEATURE_MODULES = ("Backbone", "RoiFeature")

# Backbone → Where → RoiFeature → Class / Box
UPSTREAM_OF: dict[str, tuple[str, ...]] = {
    "Backbone": (),
    "Where": ("Backbone",),
    "RoiFeature": ("Backbone", "Where"),
    "Box": ("Backbone", "Where", "RoiFeature"),
    "Class": ("Backbone", "Where", "RoiFeature"),
}
MAX_UPSTREAM = 3

# Query 外循环仍可按职责拆梯度；ψ 不再用其做代理损失。
MODULE_ROLE_LOSS: dict[str, tuple[str, ...]] = {
    "Backbone": ("total",),
    "Where": ("where",),
    "RoiFeature": ("roi_total",),
    "Box": ("localization",),
    "Class": ("positive", "ranking", "hard_negative"),
}


def upstream_slot_dim(key_dim: int) -> int:
    """每个上游模块：selected_key_embed[key_dim] + ρ + null_weight。"""
    return int(key_dim) + 2


def upstream_state_dim(key_dim: int) -> int:
    return MAX_UPSTREAM * upstream_slot_dim(key_dim)


def smooth_utility_features(
    stats: dict | None,
    *,
    prior_count: float = 8.0,
    prior_mean: float = 0.0,
    prior_neg_rate: float = 0.5,
) -> torch.Tensor:
    """少量记录时向先验收缩：均值、标准差、log(1+count)、负迁移率。"""
    stats = stats or {}
    count = float(stats.get("count") or 0.0)
    raw_mean = float(stats.get("ema") if stats.get("ema") is not None else stats.get("mean") or 0.0)
    raw_var = float(stats.get("variance") or 0.0)
    raw_neg = float(stats.get("neg_rate") or prior_neg_rate)
    strength = count / (count + float(prior_count))
    mean = (1.0 - strength) * float(prior_mean) + strength * raw_mean
    if count > 1.0:
        sample_var = max(raw_var / max(count - 1.0, 1.0), 0.0)
    else:
        sample_var = 0.0
    std = math.sqrt((1.0 - strength) * 1.0 + strength * sample_var)
    neg = (1.0 - strength) * float(prior_neg_rate) + strength * raw_neg
    return torch.tensor(
        [mean, std, math.log1p(count), neg],
        dtype=torch.float32,
    )


def mix_selected_key_embed(
    card_weights: torch.Tensor,
    candidate_keys: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """用 Softmax 选卡权重对候选签名做软混合，得到本模块实际选用的卡片嵌入。"""
    keys = candidate_keys.detach().float()
    weights = card_weights.float().reshape(-1)
    if keys.ndim != 2:
        raise ValueError("candidate_keys 必须是 [K, key_dim]")
    if int(weights.numel()) != int(keys.shape[0]):
        raise ValueError("选卡权重与候选签名数量不一致")
    if int(mask.sum()) == 0:
        return torch.zeros(keys.shape[-1], dtype=torch.float32)
    return (weights.unsqueeze(1) * keys).sum(dim=0)


def encode_upstream_state(
    prior_plans: dict[str, dict],
    module: str,
    *,
    key_dim: int = 64,
) -> torch.Tensor:
    """上游已选卡片签名嵌入 + 实际 ρ/Null，供下游条件化。"""
    slot = upstream_slot_dim(key_dim)
    parts: list[torch.Tensor] = []
    for name in UPSTREAM_OF[module]:
        plan = prior_plans.get(name) or {}
        embed = plan.get("selected_key_embed")
        if embed is None:
            embed = torch.zeros(key_dim, dtype=torch.float32)
        else:
            embed = embed.detach().float().reshape(-1)
            if int(embed.numel()) > key_dim:
                embed = embed[:key_dim]
            elif int(embed.numel()) < key_dim:
                embed = F.pad(embed, (0, key_dim - int(embed.numel())))
        rho = plan.get("rho")
        rho_v = float(rho.detach()) if torch.is_tensor(rho) else float(plan.get("rho") or 0.0)
        null_w = plan.get("null_weight")
        if torch.is_tensor(null_w):
            null_v = float(null_w.detach())
        else:
            null_v = float(plan.get("null_probability") or 1.0)
        parts.append(torch.cat([embed, torch.tensor([rho_v, null_v], dtype=torch.float32)]))
    while len(parts) < MAX_UPSTREAM:
        parts.append(torch.zeros(slot, dtype=torch.float32))
    return torch.cat(parts[:MAX_UPSTREAM])


class ConditionalTransferMLP(nn.Module):
    """共享主干 + 模块嵌入；输出选卡 logits、独立 Null logit、标量 ρ。"""

    def __init__(self, key_dim: int = 64, hidden: int = 128, max_candidates: int = 4):
        super().__init__()
        self.key_dim = int(key_dim)
        self.max_candidates = int(max_candidates)
        self.utility_dim = UTILITY_FEATURE_DIM
        self.upstream_dim = upstream_state_dim(self.key_dim)
        self.module_emb = nn.Embedding(len(MODULES), 16)
        # task_key + emb + sim + norm + key_diff + utility[4] + upstream
        in_dim = self.key_dim + 16 + 2 + 1 + self.utility_dim + self.upstream_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Linear(hidden, 1)
        self.rho_head = nn.Linear(hidden, 1)
        self.null_head = nn.Linear(hidden, 1)

    def _fit_key(self, key: torch.Tensor) -> torch.Tensor:
        key = key.detach().float().reshape(-1)
        if int(key.numel()) == self.key_dim:
            return key
        if int(key.numel()) > self.key_dim:
            return key[: self.key_dim]
        return F.pad(key, (0, self.key_dim - int(key.numel())))

    def forward(
        self,
        task_key: torch.Tensor,
        candidate_sims: torch.Tensor,
        candidate_norms: torch.Tensor,
        candidate_key_diffs: torch.Tensor,
        candidate_utilities: torch.Tensor,
        module: str,
        candidate_mask: torch.Tensor,
        upstream_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (card_logits[K], null_logit, rho_raw)。"""
        if module not in MODULES:
            raise ValueError(f"未知模块：{module}")
        module_index = MODULES.index(module)
        key = self._fit_key(task_key)
        emb = self.module_emb.weight[module_index]
        sims = candidate_sims.detach().float().reshape(-1)
        norms = candidate_norms.detach().float().reshape(-1)
        diffs = candidate_key_diffs.detach().float().reshape(-1)
        utilities = candidate_utilities.detach().float()
        if utilities.ndim == 1:
            utilities = utilities.reshape(self.max_candidates, self.utility_dim)
        mask = candidate_mask.bool().reshape(-1)
        upstream = upstream_state.detach().float().reshape(-1)
        if int(upstream.numel()) != self.upstream_dim:
            if int(upstream.numel()) > self.upstream_dim:
                upstream = upstream[: self.upstream_dim]
            else:
                upstream = F.pad(upstream, (0, self.upstream_dim - int(upstream.numel())))
        if sims.numel() != self.max_candidates:
            raise ValueError("候选相似度长度必须等于 max_candidates")
        if int(utilities.shape[0]) != self.max_candidates or int(utilities.shape[-1]) != self.utility_dim:
            raise ValueError("候选效用特征形状必须是 [K, 4]")
        scores = []
        pooled = []
        for index in range(self.max_candidates):
            features = torch.cat(
                [
                    key, emb,
                    sims[index].reshape(1), norms[index].reshape(1), diffs[index].reshape(1),
                    utilities[index].reshape(-1), upstream,
                ],
                dim=0,
            )
            hidden = self.trunk(features)
            pooled.append(hidden)
            logit = self.score_head(hidden).reshape(())
            if not bool(mask[index]):
                logit = logit - 1e6
            scores.append(logit)
        stacked = torch.stack(pooled)
        card_logits = torch.stack(scores)
        context = stacked.mean(dim=0)
        null_logit = self.null_head(context).reshape(())
        rho_raw = self.rho_head(context).reshape(())
        return card_logits, null_logit, rho_raw


class TransferPolicyController:
    """Top-K 召回后由 MLP 决策；Null 与 ρ 分离。"""

    def __init__(self, config: dict | None = None):
        config = load_retrieval_config() if config is None else config
        self.rho_max = float(config.get("rho_max", 0.3))
        if not 0 < self.rho_max <= 1:
            raise ValueError("ρ 上限必须在 (0, 1]")
        floors = config.get("rho_floor") or {}
        self.rho_floor = {module: float(floors.get(module, 0.0)) for module in MODULES}
        for module, floor in self.rho_floor.items():
            if not 0 <= floor < self.rho_max:
                raise ValueError(f"{module} 的 ρ 下限必须在 [0, rho_max)")
        self.utility_prior_count = float(config.get("utility_prior_count", 8.0))
        self.null_eval_threshold = float(config.get("null_eval_threshold", 0.5))
        boosts = config.get("policy_strength_boost") or {}
        self.strength_boost = {module: float(boosts.get(module, 1.0)) for module in MODULES}
        for module, boost in self.strength_boost.items():
            if boost <= 0:
                raise ValueError(f"{module} 的 policy_strength_boost 必须为正")
        self.key_dim = int(config.get("pca_dim", 64))
        self.max_candidates = int(config.get("max_top_k", 4))
        self.policy = ConditionalTransferMLP(
            key_dim=self.key_dim,
            hidden=int(config.get("policy_hidden", 128)),
            max_candidates=self.max_candidates,
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=float(config.get("policy_adam_lr", 0.001)),
        )

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_optimizer_state": self.optimizer.state_dict(),
            "rho_max": self.rho_max,
            "rho_floor": dict(self.rho_floor),
            "key_dim": self.key_dim,
            "kind": POLICY_KIND,
        }

    def load_state_dict(self, payload: dict) -> None:
        kind = payload.get("kind")
        if kind is not None and kind != POLICY_KIND:
            raise ValueError(f"决策网络须为 {POLICY_KIND}，收到 {kind}，请重新训练")
        if "key_dim" in payload and int(payload["key_dim"]) != self.key_dim:
            raise ValueError("决策网络 key_dim 与当前配置不一致")
        self.policy.load_state_dict(payload["policy"])
        opt_state = payload.get("policy_optimizer_state") or payload.get("optimizer")
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        if "rho_max" in payload:
            self.rho_max = float(payload["rho_max"])
        if payload.get("rho_floor"):
            for module, value in payload["rho_floor"].items():
                if module in self.rho_floor:
                    self.rho_floor[module] = float(value)

    def effective_rho(self, raw: torch.Tensor, module: str) -> torch.Tensor:
        floor = float(self.rho_floor.get(module, 0.0))
        return floor + (self.rho_max - floor) * torch.sigmoid(raw)


def pad_candidates(
    vectors: list[torch.Tensor],
    sims: list[float],
    width: int,
    max_candidates: int,
    key_diffs: list[float] | None = None,
    candidate_keys: list[torch.Tensor] | None = None,
    candidate_utilities: list[torch.Tensor] | None = None,
    key_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 units、sims、norms、mask、key_diffs、candidate_keys、utilities。"""
    units = []
    sim_values = []
    norms = []
    mask = []
    diffs = []
    keys = []
    utilities = []
    for index in range(max_candidates):
        if index < len(vectors) and vectors[index] is not None:
            vector = vectors[index].detach().float().reshape(-1)
            if int(vector.numel()) != width:
                raise AssertionError("候选向量宽度不一致")
            norm = float(torch.linalg.vector_norm(vector))
            units.append(vector if norm >= _EPS else torch.zeros(width))
            sim_values.append(float(sims[index]) if index < len(sims) else 0.0)
            norms.append(norm)
            mask.append(norm >= _EPS)
            diffs.append(float(key_diffs[index]) if key_diffs is not None and index < len(key_diffs) else 0.0)
            if candidate_keys is not None and index < len(candidate_keys) and candidate_keys[index] is not None:
                key = candidate_keys[index].detach().float().reshape(-1)
                if int(key.numel()) > key_dim:
                    key = key[:key_dim]
                elif int(key.numel()) < key_dim:
                    key = F.pad(key, (0, key_dim - int(key.numel())))
                keys.append(key)
            else:
                keys.append(torch.zeros(key_dim))
            if candidate_utilities is not None and index < len(candidate_utilities) and candidate_utilities[index] is not None:
                util = candidate_utilities[index].detach().float().reshape(-1)
                if int(util.numel()) != UTILITY_FEATURE_DIM:
                    raise AssertionError("效用特征维度必须为 4")
                utilities.append(util)
            else:
                utilities.append(smooth_utility_features(None))
        else:
            units.append(torch.zeros(width))
            sim_values.append(0.0)
            norms.append(0.0)
            mask.append(False)
            diffs.append(0.0)
            keys.append(torch.zeros(key_dim))
            utilities.append(smooth_utility_features(None))
    return (
        torch.stack(units),
        torch.tensor(sim_values, dtype=torch.float32),
        torch.tensor(norms, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(diffs, dtype=torch.float32),
        torch.stack(keys),
        torch.stack(utilities),
    )


def mix_cards_from_logits(
    card_logits: torch.Tensor,
    units: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """仅对真实候选 Softmax；返回单位 mix 与 card_weights（和为 1）。

    Null 不进入该 Softmax。软借用时有效卡片质量为 (1-Null)*w_i，
    不对非 Null 卡片再归一化。
    """
    logits = card_logits.clone()
    if int(mask.sum()) == 0:
        weights = torch.zeros_like(logits)
        return torch.zeros(units.shape[-1]), weights
    logits = torch.where(mask, logits, torch.full_like(logits, -1e6))
    weights = torch.softmax(logits, dim=0)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    total = weights.sum().clamp_min(_EPS)
    weights = weights / total
    mix = (weights.unsqueeze(1) * units).sum(dim=0)
    return mix, weights


def null_probability_from_logit(null_logit: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(null_logit)


def decide_inject(
    *,
    allow_borrow: bool,
    retrieved_valid: bool,
    null_weight: torch.Tensor,
    applied_l2: float,
    step: float,
    discrete_null: bool,
    null_threshold: float,
) -> bool:
    """训练连续稀释；评价时 Null 可变成真正的离散不借。"""
    if not allow_borrow or not retrieved_valid or step <= 0.0:
        return False
    if discrete_null and float(null_weight.detach()) >= float(null_threshold):
        return False
    return applied_l2 > 1e-8


def counterfactual_policy_loss(
    plans: list[dict],
    utility: float,
    strength_boost: dict[str, float] | None = None,
) -> torch.Tensor:
    """用 stopgrad(U) 经 soft 注入幅度回传选卡/Null/ρ。U=L_Null-L_Borrow。"""
    boost = strength_boost or {}
    total = None
    for plan in plans:
        if not plan.get("valid") or not plan.get("inject"):
            continue
        delta = plan.get("delta_vector")
        if delta is None or not torch.is_tensor(delta):
            continue
        if not bool(delta.requires_grad):
            continue
        weight = float(boost.get(plan["module"], 1.0))
        term = weight * torch.linalg.vector_norm(delta.float().reshape(-1))
        total = term if total is None else total + term
    if total is None:
        raise ValueError("没有可微的迁移决策")
    return -float(utility) * total
