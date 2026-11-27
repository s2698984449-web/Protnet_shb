"""条件化模块知识迁移 v5.1：任务条件 Null + LOO/Add-One + 上游组合。

相似度只负责召回；MLP 选卡、硬门控 z、强度 ρ。
动作 a_m=(z_m, c_m, ρ_m)：z_m=0 时注入严格为 0。
ψ 用模块条件贡献 U_m（Leave-One-Out / Add-One）更新，不再广播整任务 U。
Null 头用有效候选 mask-mean，并直连 task∥module∥upstream，避免模块固定偏置。
AP50/AP75 效用单独预测，与损失效用分账。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.retrieval_memory import MODULES, load_retrieval_config

_EPS = 1e-12
POLICY_KIND = "conditional_transfer_v51"
UTILITY_FEATURE_DIM = 4  # mean, std, log1p(count), neg_rate

# Backbone → Where → RoiFeature → Class / Box
UPSTREAM_OF: dict[str, tuple[str, ...]] = {
    "Backbone": (),
    "Where": ("Backbone",),
    "RoiFeature": ("Backbone", "Where"),
    "Box": ("Backbone", "Where", "RoiFeature"),
    "Class": ("Backbone", "Where", "RoiFeature"),
}
MAX_UPSTREAM = 3


def upstream_slot_dim(key_dim: int) -> int:
    """每个上游模块：selected_key_embed[key_dim] + ρ + z。"""
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
    """上游已选卡片签名嵌入 + ρ + 硬门控 z，供下游条件化。"""
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
        z = plan.get("borrow_gate")
        if torch.is_tensor(z):
            z_v = float(z.detach())
        else:
            z_v = float(plan.get("z_hard") or 0.0)
        parts.append(torch.cat([embed, torch.tensor([rho_v, z_v], dtype=torch.float32)]))
    while len(parts) < MAX_UPSTREAM:
        parts.append(torch.zeros(slot, dtype=torch.float32))
    return torch.cat(parts[:MAX_UPSTREAM])


def hard_borrow_gate(
    null_logit: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (z_ste, z_hard, p_null)。前向硬门控，反向走 (1-p_null)。"""
    p_null = torch.sigmoid(null_logit)
    z_soft = 1.0 - p_null
    z_hard = (p_null < float(threshold)).to(dtype=z_soft.dtype)
    z_ste = z_hard + z_soft - z_soft.detach()
    return z_ste, z_hard, p_null


class ConditionalTransferMLP(nn.Module):
    """共享主干 + 模块嵌入；输出选卡 logits、独立 Null logit、标量 ρ。"""

    def __init__(self, key_dim: int = 64, hidden: int = 128, max_candidates: int = 4):
        super().__init__()
        self.key_dim = int(key_dim)
        self.max_candidates = int(max_candidates)
        self.utility_dim = UTILITY_FEATURE_DIM
        self.upstream_dim = upstream_state_dim(self.key_dim)
        self.module_emb = nn.Embedding(len(MODULES), 16)
        in_dim = self.key_dim + 16 + 2 + 1 + self.utility_dim + self.upstream_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Linear(hidden, 1)
        self.rho_head = nn.Linear(hidden, 1)
        # Null：有效候选 mask-mean 与 task∥module∥upstream 直连摘要融合，避免 pad 稀释与模块塌缩。
        null_direct_dim = self.key_dim + 16 + self.upstream_dim
        self.null_direct = nn.Sequential(
            nn.Linear(null_direct_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.null_fuse = nn.Linear(hidden * 2, hidden)
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
        mask_f = mask.to(dtype=stacked.dtype).unsqueeze(1)
        mask_sum = mask_f.sum().clamp_min(1.0)
        cand_context = (stacked * mask_f).sum(dim=0) / mask_sum
        if int(mask.sum()) == 0:
            cand_context = torch.zeros_like(cand_context)
        direct = self.null_direct(torch.cat([key, emb, upstream], dim=0))
        null_context = F.relu(self.null_fuse(torch.cat([cand_context, direct], dim=0)))
        null_logit = self.null_head(null_context).reshape(())
        rho_raw = self.rho_head(cand_context).reshape(())
        return card_logits, null_logit, rho_raw


class ApUtilityHead(nn.Module):
    """由各模块决策摘要预测 AP50/AP75 差及其不确定性（与损失效用分账）。"""

    def __init__(self, module_count: int = 5, hidden: int = 64):
        super().__init__()
        in_dim = int(module_count) * 4  # rho, z, p_null, sim_max
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),  # mu50, log_sigma50, mu75, log_sigma75
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(features.float().reshape(1, -1)).reshape(-1)
        return {
            "mu_ap50": raw[0],
            "log_sigma_ap50": raw[1],
            "mu_ap75": raw[2],
            "log_sigma_ap75": raw[3],
        }


def plan_decision_features(plans: list[dict]) -> torch.Tensor:
    """拼装 AP 效用头输入；不含产品名。"""
    rows = []
    by_module = {plan["module"]: plan for plan in plans}
    for module in MODULES:
        plan = by_module.get(module) or {}
        rho = plan.get("rho")
        rho_v = float(rho.detach()) if torch.is_tensor(rho) else float(plan.get("rho") or 0.0)
        z_v = float(plan.get("z_hard") or 0.0)
        p_null = float(plan.get("null_probability") or 1.0)
        sim = float(plan.get("sim_max") or 0.0)
        rows.extend([rho_v, z_v, p_null, sim])
    return torch.tensor(rows, dtype=torch.float32)


def ap_utility_nll(
    prediction: dict[str, torch.Tensor],
    target_ap50: float,
    target_ap75: float | None = None,
) -> torch.Tensor:
    """高斯负对数似然；缺 AP75 时只监督 AP50。"""
    loss = _gaussian_nll(prediction["mu_ap50"], prediction["log_sigma_ap50"], float(target_ap50))
    if target_ap75 is not None:
        loss = loss + _gaussian_nll(prediction["mu_ap75"], prediction["log_sigma_ap75"], float(target_ap75))
    return loss


def _gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, target: float) -> torch.Tensor:
    sigma = F.softplus(log_sigma) + 1e-3
    err = mu - float(target)
    return 0.5 * (err * err) / (sigma * sigma) + torch.log(sigma)


def risk_constrained_score(
    mu: float,
    log_sigma: float,
    *,
    risk_lambda: float = 1.0,
) -> float:
    """Score = μ − λσ；不足覆盖风险时调用方选 Null。"""
    sigma = float(F.softplus(torch.tensor(float(log_sigma))) + 1e-3)
    return float(mu) - float(risk_lambda) * sigma


class TransferPolicyController:
    """Top-K 召回后由 MLP 决策；硬 z 与 ρ 分离。"""

    def __init__(self, config: dict | None = None):
        config = load_retrieval_config() if config is None else config
        self.rho_max = float(config.get("rho_max", 0.3))
        if not 0 < self.rho_max <= 1:
            raise ValueError("ρ 上限必须在 (0, 1]")
        self.utility_prior_count = float(config.get("utility_prior_count", 8.0))
        self.null_eval_threshold = float(config.get("null_eval_threshold", 0.5))
        self.key_dim = int(config.get("pca_dim", 64))
        self.max_candidates = int(config.get("max_top_k", 4))
        loo = config.get("module_loo") or {}
        self.loo_enabled = bool(loo.get("enabled", True))
        self.loo_modules_per_step = max(1, int(loo.get("modules_per_step", 1)))
        self.force_close_prob = float(loo.get("force_close_prob", 0.25))
        if not 0.0 <= self.force_close_prob <= 1.0:
            raise ValueError("module_loo.force_close_prob 必须在 [0, 1]")
        probes = loo.get("rho_probes")
        if probes is None:
            probes = [0.1, 0.2, 0.3]
        self.rho_probes = [float(value) for value in probes]
        for value in self.rho_probes:
            if not 0.0 < value <= self.rho_max + 1e-8:
                raise ValueError(f"module_loo.rho_probes 须在 (0, rho_max]，收到 {value}")
        self.rho_pairwise = bool(loo.get("rho_pairwise", True))
        self.card_probe = bool(loo.get("card_probe", True))
        self.card_pairwise = bool(loo.get("card_pairwise", True))
        self._loo_cursor = 0
        self._rho_probe_cursor = 0
        self._rho_pair_cursor = 0
        self._card_probe_cursor = 0
        self._card_pair_cursor = 0
        ap_cfg = config.get("ap_utility") or {}
        # AP Head 默认隔离：不进主决策损失；every_n=0 关闭训练步。
        self.ap_utility_enabled = bool(ap_cfg.get("enabled", False))
        self.ap_utility_every_n_steps = max(0, int(ap_cfg.get("every_n_steps", 0)))
        self.policy = ConditionalTransferMLP(
            key_dim=self.key_dim,
            hidden=int(config.get("policy_hidden", 128)),
            max_candidates=self.max_candidates,
        )
        self.ap_head = ApUtilityHead(module_count=len(MODULES), hidden=int(ap_cfg.get("hidden", 64)))
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.ap_head.parameters()),
            lr=float(config.get("policy_adam_lr", 0.001)),
        )

    def next_loo_modules(self) -> list[str]:
        """轮换抽出本步要做 Leave-One-Out 的模块。"""
        if not self.loo_enabled:
            return []
        count = min(self.loo_modules_per_step, len(MODULES))
        chosen = []
        for _ in range(count):
            chosen.append(MODULES[self._loo_cursor % len(MODULES)])
            self._loo_cursor += 1
        return chosen

    def next_rho_probe(self) -> float | None:
        """轮换一档 ρ 探针；无配置则跳过。"""
        if not self.rho_probes:
            return None
        value = float(self.rho_probes[self._rho_probe_cursor % len(self.rho_probes)])
        self._rho_probe_cursor += 1
        return value

    def rho_probe_pairs(self) -> list[tuple[float, float]]:
        """同任务配对：(0.1,0.2)、(0.2,0.3)、(0.1,0.3)。"""
        values = list(self.rho_probes)
        if len(values) < 2:
            return []
        pairs: list[tuple[float, float]] = []
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                pairs.append((float(values[left]), float(values[right])))
        # 固定轮换顺序：相邻优先，再跨档
        ordered: list[tuple[float, float]] = []
        for index in range(len(values) - 1):
            ordered.append((float(values[index]), float(values[index + 1])))
        for pair in pairs:
            if pair not in ordered and (pair[1], pair[0]) not in ordered:
                ordered.append(pair)
        return ordered

    def next_rho_pair(self) -> tuple[float, float] | None:
        """轮换一对 ρ；不足两档则退回单档 (ρ, None 用单探针路径)。"""
        if not self.rho_pairwise or len(self.rho_probes) < 2:
            single = self.next_rho_probe()
            return None if single is None else (float(single), float(single))
        pairs = self.rho_probe_pairs()
        if not pairs:
            return None
        pair = pairs[self._rho_pair_cursor % len(pairs)]
        self._rho_pair_cursor += 1
        return float(pair[0]), float(pair[1])

    def next_card_probe_index(self, mask: torch.Tensor) -> int | None:
        """在有效候选中轮换一张卡；无有效候选则跳过。"""
        if not self.card_probe:
            return None
        valid = [index for index, flag in enumerate(mask.bool().reshape(-1).tolist()) if flag]
        if not valid:
            return None
        choice = valid[self._card_probe_cursor % len(valid)]
        self._card_probe_cursor += 1
        return int(choice)

    def next_card_probe_pair(self, mask: torch.Tensor) -> tuple[int, int] | None:
        """轮换两张有效卡做配对；不足两张则返回 None（退回单卡）。"""
        if not self.card_probe or not self.card_pairwise:
            return None
        valid = [index for index, flag in enumerate(mask.bool().reshape(-1).tolist()) if flag]
        if len(valid) < 2:
            return None
        pairs = [(valid[i], valid[j]) for i in range(len(valid)) for j in range(i + 1, len(valid))]
        choice = pairs[self._card_pair_cursor % len(pairs)]
        self._card_pair_cursor += 1
        return int(choice[0]), int(choice[1])

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "ap_head": self.ap_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_optimizer_state": self.optimizer.state_dict(),
            "rho_max": self.rho_max,
            "key_dim": self.key_dim,
            "loo_cursor": int(self._loo_cursor),
            "rho_probe_cursor": int(self._rho_probe_cursor),
            "rho_pair_cursor": int(self._rho_pair_cursor),
            "card_probe_cursor": int(self._card_probe_cursor),
            "card_pair_cursor": int(self._card_pair_cursor),
            "kind": POLICY_KIND,
        }

    def load_state_dict(self, payload: dict) -> None:
        kind = payload.get("kind")
        if kind is not None and kind != POLICY_KIND:
            raise ValueError(f"决策网络须为 {POLICY_KIND}，收到 {kind}，请重新训练")
        if "key_dim" in payload and int(payload["key_dim"]) != self.key_dim:
            raise ValueError("决策网络 key_dim 与当前配置不一致")
        self.policy.load_state_dict(payload["policy"])
        if payload.get("ap_head") is not None:
            self.ap_head.load_state_dict(payload["ap_head"])
        opt_state = payload.get("policy_optimizer_state") or payload.get("optimizer")
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        if "rho_max" in payload:
            self.rho_max = float(payload["rho_max"])
        if "loo_cursor" in payload:
            self._loo_cursor = int(payload["loo_cursor"])
        if "rho_probe_cursor" in payload:
            self._rho_probe_cursor = int(payload["rho_probe_cursor"])
        if "rho_pair_cursor" in payload:
            self._rho_pair_cursor = int(payload["rho_pair_cursor"])
        if "card_probe_cursor" in payload:
            self._card_probe_cursor = int(payload["card_probe_cursor"])
        if "card_pair_cursor" in payload:
            self._card_pair_cursor = int(payload["card_pair_cursor"])

    def effective_rho(self, raw: torch.Tensor, module: str) -> torch.Tensor:
        del module
        return self.rho_max * torch.sigmoid(raw)


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
    """仅对真实候选 Softmax；Null/z 不进入该 Softmax。"""
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


def decide_inject(
    *,
    allow_borrow: bool,
    retrieved_valid: bool,
    z_hard: float,
    applied_l2: float,
    step: float,
) -> bool:
    """z_hard=0 时绝不注入。"""
    if not allow_borrow or not retrieved_valid or step <= 0.0:
        return False
    if float(z_hard) < 0.5:
        return False
    return applied_l2 > 1e-8


def counterfactual_policy_loss(
    plans: list[dict],
    module_utilities: dict[str, float],
) -> torch.Tensor:
    """按模块条件贡献 U_m 更新选卡/z/ρ；不得把整任务 U 广播到所有模块。

    幅度项 = z_ste · ρ · step · ||mix||。
    z=0 时前向为 0，反向仍经 STE 到 null_logit，并经 mix 到 card logits，
    从而支持 add-one 效用下的重新开启。
    """
    total = None
    for plan in plans:
        module = plan["module"]
        if module not in module_utilities:
            continue
        if not plan.get("valid"):
            continue
        utility = float(module_utilities[module])
        gate = plan.get("borrow_gate")
        rho = plan.get("rho")
        mix = plan.get("vector")
        if not torch.is_tensor(gate) or not torch.is_tensor(rho) or not torch.is_tensor(mix):
            continue
        if not (bool(gate.requires_grad) or bool(rho.requires_grad) or bool(mix.requires_grad)):
            continue
        step = float(plan.get("support_step_l2") or 1.0)
        if step <= 0.0:
            continue
        amplitude = gate * rho * step * torch.linalg.vector_norm(mix.float().reshape(-1))
        weighted = (-utility) * amplitude
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("没有可微的模块条件贡献")
    return total


def rho_probe_policy_loss(plan: dict, utility: float, rho_probe: float) -> torch.Tensor:
    """单档强度探针（配对不可用时的回退）：U>0 拉向 ρ_probe，U<0 拉向 0。"""
    rho = plan.get("rho")
    if not torch.is_tensor(rho) or not bool(rho.requires_grad):
        raise ValueError("ρ 探针需要可微的 rho")
    target = float(rho_probe) if float(utility) > 0.0 else 0.0
    return (rho - rho.new_tensor(target)) ** 2


def rho_pair_policy_loss(
    plan: dict,
    rho_a: float,
    utility_a: float,
    rho_b: float,
    utility_b: float,
    eps: float,
) -> torch.Tensor:
    """同任务双档排序：|ΔU|>ε 时把 ρ 拉向优胜档；死区内不监督。"""
    rho = plan.get("rho")
    if not torch.is_tensor(rho) or not bool(rho.requires_grad):
        raise ValueError("ρ 配对需要可微的 rho")
    delta = float(utility_a) - float(utility_b)
    if abs(delta) <= float(eps):
        raise ValueError("ρ 配对落在噪声死区")
    if abs(float(rho_a) - float(rho_b)) <= 1e-12:
        raise ValueError("ρ 配对两档相同")
    winner = float(rho_a) if delta > 0.0 else float(rho_b)
    return (rho - rho.new_tensor(winner)) ** 2


def card_probe_policy_loss(plan: dict, utility: float, card_index: int) -> torch.Tensor:
    """单卡探针：只更新对应 card_logit，正 U 抬高、负 U 压低。"""
    logits = plan.get("card_logits")
    if not torch.is_tensor(logits) or not bool(logits.requires_grad):
        raise ValueError("卡探针需要可微的 card_logits")
    index = int(card_index)
    if index < 0 or index >= int(logits.numel()):
        raise ValueError("卡探针下标越界")
    return (-float(utility)) * logits[index]


def card_pair_policy_loss(
    plan: dict,
    index_a: int,
    utility_a: float,
    index_b: int,
    utility_b: float,
    eps: float,
) -> torch.Tensor:
    """单卡绝对效用 + 配对排序：死区内只保留绝对项。"""
    logits = plan.get("card_logits")
    if not torch.is_tensor(logits) or not bool(logits.requires_grad):
        raise ValueError("卡配对需要可微的 card_logits")
    a = int(index_a)
    b = int(index_b)
    if a < 0 or b < 0 or a >= int(logits.numel()) or b >= int(logits.numel()):
        raise ValueError("卡配对下标越界")
    if a == b:
        raise ValueError("卡配对下标相同")
    loss = (-float(utility_a)) * logits[a] + (-float(utility_b)) * logits[b]
    delta = float(utility_a) - float(utility_b)
    if abs(delta) > float(eps):
        # 抬高优胜卡、压低劣势卡
        if delta > 0.0:
            loss = loss + torch.nn.functional.softplus(logits[b] - logits[a])
        else:
            loss = loss + torch.nn.functional.softplus(logits[a] - logits[b])
    return loss


def utility_ap_sign_agreement(
    utility: float,
    delta_ap: float | None,
    *,
    eps_u: float = 0.0,
    eps_ap: float = 0.0,
) -> dict | None:
    """诊断：Query 效用与 ΔAP 符号是否一致（排除近零噪声）。"""
    if delta_ap is None:
        return None
    u_value = float(utility)
    ap_value = float(delta_ap)
    if abs(u_value) <= float(eps_u) or abs(ap_value) <= float(eps_ap):
        return {"comparable": False, "agree": None, "utility": u_value, "delta_ap": ap_value}
    agree = (u_value > 0.0) == (ap_value > 0.0)
    return {"comparable": True, "agree": bool(agree), "utility": u_value, "delta_ap": ap_value}
