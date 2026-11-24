"""检索式 FOMAML 的任务签名、聚类、TIES 和借用幅度。

这里只处理 CPU 上的描述子和参数卡片。检测器训练循环仍在 theta0.py。
第一版投影固定为 PCA，不训练 W。
"""

from __future__ import annotations

from math import ceil, log, sqrt

import torch
import torch.nn.functional as F

from common.io import load_config, named_config_path


MODULES = ("Feature", "Where", "Box", "Class")
KEY_FOR_MODULE = {
    "Feature": "Appearance",
    "Where": "Morphology",
    "Box": "Morphology",
    "Class": "Difference",
}
_EPS = 1e-12


def load_retrieval_config() -> dict:
    path = named_config_path("retrieval_fomaml.yaml")
    config = load_config(path)
    if config.get("train_projection"):
        raise ValueError("第一版 PCA 固定，train_projection 必须为 false")
    if int(config["min_members_for_ties"]) < 2:
        raise ValueError("TIES 最少成员必须不小于 2")
    if not 0 < float(config["no_borrow_probability"]) < 1:
        raise ValueError("不借用概率必须在 (0, 1) 内")
    return config


def expected_members_per_bin(families: int, tasks_per_family: int, clusters: int, groups: int) -> float:
    if min(families, tasks_per_family, clusters, groups) < 1:
        raise ValueError("产品族、任务数、簇数和组数都必须为正")
    return families * tasks_per_family / (clusters * groups)


def family_task_weights(family_ids: list[str]) -> torch.Tensor:
    """每个产品族的总权重相同，族内任务均分。族内任务数相同时退化为等权。"""
    if not family_ids:
        raise ValueError("没有可加权的任务")
    counts: dict[str, int] = {}
    for family_id in family_ids:
        counts[family_id] = counts.get(family_id, 0) + 1
    return torch.tensor([1.0 / counts[family_id] for family_id in family_ids], dtype=torch.float32)


def box_weighted_mean(features: torch.Tensor, box_counts: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("特征必须是二维")
    weights = box_counts.to(dtype=features.dtype).reshape(-1)
    if weights.numel() != features.shape[0]:
        raise ValueError("框数和特征条数不一致")
    total = weights.sum()
    if float(total) <= 0:
        raise ValueError("缺陷原型至少需要一个框")
    return (features * (weights / total).unsqueeze(1)).sum(dim=0)


def appearance_key(reference_features: torch.Tensor) -> torch.Tensor:
    """R 组图像特征的均值。不使用 Query。"""
    if reference_features.ndim != 2 or reference_features.shape[0] < 1:
        raise ValueError("正常参照特征不能为空")
    return reference_features.mean(dim=0)


def difference_key(defect_prototype: torch.Tensor, normal_prototype: torch.Tensor) -> torch.Tensor:
    if defect_prototype.shape != normal_prototype.shape:
        raise ValueError("缺陷原型和正常原型维度不一致")
    return defect_prototype - normal_prototype


def geometry_descriptor(per_image_boxes: list[torch.Tensor], image_hw: tuple[int, int]) -> torch.Tensor:
    """log 面积均值/标准差、log 长宽比均值/标准差、log 相对面积均值、每图框数对数。"""
    if image_hw[0] < 1 or image_hw[1] < 1:
        raise ValueError("图像高宽必须为正")
    image_area = float(image_hw[0] * image_hw[1])
    areas = []
    aspects = []
    relative = []
    counts = []
    for boxes in per_image_boxes:
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("框必须是 [n, 4]")
        count = int(boxes.shape[0])
        counts.append(log(count) if count else 0.0)
        if count == 0:
            continue
        widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(_EPS)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(_EPS)
        area = widths * heights
        areas.append(area.log())
        aspects.append((widths / heights).log())
        relative.append((area / image_area).log())
    if not areas:
        return torch.zeros(6)
    area_values = torch.cat(areas)
    aspect_values = torch.cat(aspects)
    relative_values = torch.cat(relative)
    count_values = torch.tensor(counts, dtype=area_values.dtype)
    return torch.stack([
        area_values.mean(),
        area_values.std(unbiased=False),
        aspect_values.mean(),
        aspect_values.std(unbiased=False),
        relative_values.mean(),
        count_values.mean(),
    ])


def morphology_key(defect_prototype: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    if geometry.numel() != 6:
        raise ValueError("几何描述子必须是 6 维")
    scale = sqrt(2.0)
    return torch.cat([defect_prototype / scale, geometry.to(dtype=defect_prototype.dtype) / scale])


def fit_fixed_pca(keys: torch.Tensor, dim: int) -> dict[str, torch.Tensor]:
    if keys.ndim != 2 or keys.shape[0] < 2:
        raise ValueError("PCA 至少需要两条任务签名")
    if dim < 1:
        raise ValueError("PCA 维度必须为正")
    mean = keys.mean(dim=0)
    centered = keys - mean
    components = torch.zeros(dim, keys.shape[1], dtype=keys.dtype)
    if float(torch.linalg.vector_norm(centered)) >= _EPS:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        rank = min(dim, vh.shape[0])
        components[:rank] = vh[:rank]
    return {"mean": mean, "components": components}


def project_l2(vector: torch.Tensor, pca: dict[str, torch.Tensor]) -> torch.Tensor:
    projected = pca["components"] @ (vector - pca["mean"])
    norm = torch.linalg.vector_norm(projected)
    if float(norm) < _EPS:
        return torch.zeros_like(projected)
    return projected / norm


def spherical_kmeans(
    vectors: torch.Tensor, family_ids: list[str], clusters: int, seed: int, steps: int = 30,
) -> dict:
    """球面 K-means。样本按产品族等权。空簇标 invalid，不改配置里的 K。"""
    if vectors.ndim != 2 or vectors.shape[0] != len(family_ids):
        raise ValueError("聚类样本和产品族数量不一致")
    if clusters < 1 or steps < 1:
        raise ValueError("簇数和迭代次数必须为正")
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(_EPS)
    points = vectors / norms
    count = points.shape[0]
    used = min(clusters, count)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    pick = torch.randperm(count, generator=generator)[:used]
    centers = torch.zeros(clusters, points.shape[1], dtype=points.dtype)
    valid = torch.zeros(clusters, dtype=torch.bool)
    centers[:used] = points[pick]
    valid[:used] = True
    weights = family_task_weights(family_ids)
    assignment = torch.full((count,), -1, dtype=torch.long)
    for _ in range(steps):
        scores = points @ centers.T
        scores[:, ~valid] = -torch.inf
        updated = scores.argmax(dim=1)
        if torch.equal(updated, assignment):
            break
        assignment = updated
        next_centers = torch.zeros_like(centers)
        next_valid = torch.zeros_like(valid)
        for cluster_id in range(clusters):
            chosen = assignment == cluster_id
            if not bool(chosen.any()):
                continue
            mass = weights[chosen].sum().clamp_min(_EPS)
            center = (points[chosen] * weights[chosen].unsqueeze(1)).sum(dim=0) / mass
            norm = torch.linalg.vector_norm(center)
            if float(norm) < _EPS:
                continue
            next_centers[cluster_id] = center / norm
            next_valid[cluster_id] = True
        centers = next_centers
        valid = next_valid
        if not bool(valid.any()):
            break
    return {
        "centers": centers,
        "valid": valid,
        "assignment": assignment,
        "clusters_requested": clusters,
        "clusters_used": int(valid.sum()),
    }


def _median(values: torch.Tensor) -> torch.Tensor:
    ordered = values.reshape(-1).sort().values
    count = int(ordered.numel())
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def median_anchor(family_norms: list[torch.Tensor]) -> torch.Tensor:
    """先在每个产品族内取中位数，再对产品族取中位数。"""
    if not family_norms:
        raise ValueError("没有范数可以计算锚定尺度")
    return _median(torch.stack([_median(part) for part in family_norms]))


def clip_large_delta(delta: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """只缩小超过锚定尺度的增量，不把小增量放大成单位向量。"""
    norm = torch.linalg.vector_norm(delta)
    scale = torch.minimum(torch.ones((), dtype=delta.dtype), anchor.to(dtype=delta.dtype) / (norm + _EPS))
    return delta * scale, scale


def trim_vector(delta: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if not 0 < keep_ratio <= 1:
        raise ValueError("保留比例必须在 (0, 1] 内")
    flat = delta.reshape(-1)
    if keep_ratio == 1 or flat.numel() == 1:
        return delta.clone()
    keep = max(1, ceil(keep_ratio * flat.numel()))
    if keep >= flat.numel():
        return delta.clone()
    index = torch.topk(flat.abs(), keep, largest=True, sorted=False).indices
    kept = torch.zeros_like(flat)
    kept[index] = flat[index]
    return kept.reshape(delta.shape)


def _coherence(deltas: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    total = weights.sum().clamp_min(_EPS)
    mixed = (deltas * (weights / total).unsqueeze(1)).sum(dim=0)
    denominator = ((weights / total) * torch.linalg.vector_norm(deltas, dim=1)).sum() + _EPS
    return torch.linalg.vector_norm(mixed) / denominator


def merge_cluster_members(
    deltas: torch.Tensor, weights: torch.Tensor, keep_ratio: float, min_members: int = 5,
) -> dict:
    """n≥5 时做 TIES；不足时用加权均值。输出单位方向，零向量标 invalid。"""
    if deltas.ndim != 2 or weights.numel() != deltas.shape[0]:
        raise ValueError("成员增量和权重数量不一致")
    if float(weights.sum()) <= 0:
        raise ValueError("成员权重之和必须为正")
    # keep_ratio=1 时不再整表 clone，避免 Feature 级 40M 维在 assemble 峰值再翻倍。
    if float(keep_ratio) >= 1.0 - 1e-12:
        trimmed = deltas
    else:
        trimmed = torch.stack([trim_vector(deltas[row], keep_ratio) for row in range(deltas.shape[0])])
    low_sample = deltas.shape[0] < min_members
    if low_sample:
        card = (trimmed * (weights / weights.sum()).unsqueeze(1)).sum(dim=0)
    else:
        total = weights.sum()
        elect = torch.sign((trimmed * weights.unsqueeze(1)).sum(dim=0) / total)
        match = (torch.sign(trimmed) == elect.unsqueeze(0)) & (elect.unsqueeze(0) != 0) & (trimmed != 0)
        matched_weight = weights.unsqueeze(1) * match.to(dtype=weights.dtype)
        denominator = matched_weight.sum(dim=0)
        numerator = (matched_weight * trimmed).sum(dim=0)
        card = torch.where(denominator > 0, numerator / denominator.clamp_min(_EPS), torch.zeros_like(numerator))
        card = torch.where(elect == 0, torch.zeros_like(card), card)
    norm = torch.linalg.vector_norm(card)
    valid = float(norm) >= _EPS
    return {
        "card": card,
        "unit": card / norm if valid else torch.zeros_like(card),
        "valid": valid,
        "low_sample": low_sample,
        "coherence": _coherence(deltas, weights),
    }


def merge_groups(units: list[torch.Tensor], member_counts: list[int], coherences: list[torch.Tensor], excluded_group: int | None) -> dict:
    """跨组只做加权线性平均，不再单位归一化。训练任务排除自己所在组。"""
    if not (len(units) == len(member_counts) == len(coherences)):
        raise ValueError("组卡片、成员数和一致性数量不一致")
    mixed = None
    weight_sum = torch.zeros(())
    used = 0
    for index, unit in enumerate(units):
        if excluded_group is not None and index == excluded_group:
            continue
        if unit is None or float(torch.linalg.vector_norm(unit)) < _EPS or member_counts[index] < 1:
            continue
        weight = torch.as_tensor(member_counts[index], dtype=unit.dtype) * coherences[index].to(dtype=unit.dtype)
        if float(weight) <= 0:
            continue
        mixed = weight * unit if mixed is None else mixed + weight * unit
        weight_sum = weight_sum + weight
        used += 1
    if mixed is None or float(weight_sum) <= 0:
        return {"vector": None, "valid": False, "groups": 0}
    return {"vector": mixed / weight_sum, "valid": True, "groups": used}


def retrieval_distribution(similarities: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    if similarities.ndim != 1:
        raise ValueError("相似度必须是一维")
    if similarities.numel() == 0:
        empty = similarities.new_empty(0)
        return empty, similarities.new_zeros(())
    deviation = similarities.std(unbiased=False)
    if similarities.numel() == 1 or float(deviation) < _EPS:
        probabilities = torch.full_like(similarities, 1.0 / similarities.numel())
    else:
        standardized = (similarities - similarities.mean()) / deviation
        probabilities = torch.softmax(standardized / temperature, dim=0)
    return probabilities, similarities.max()


def retrieval_concentration(probabilities: torch.Tensor) -> float:
    count = int(probabilities.numel())
    if count <= 1:
        return 1.0
    entropy = -(probabilities * probabilities.clamp_min(_EPS).log()).sum()
    return float(1 - entropy / log(count))


def adaptive_topk(
    similarities: torch.Tensor, mass_threshold: float = 0.8, max_k: int = 4, temperature: float = 1.0,
) -> dict:
    probabilities, sim_max = retrieval_distribution(similarities, temperature)
    count = int(probabilities.numel())
    concentration = retrieval_concentration(probabilities) if count else 0.0
    if count == 0:
        return {"indices": [], "weights": [], "probabilities": probabilities, "sim_max": sim_max, "concentration": 0.0}
    order = torch.argsort(probabilities, descending=True)
    if count == 1:
        chosen = order[:1]
    else:
        primary = order[:min(2, count)]
        if count <= 2 or float(probabilities[primary].sum()) >= mass_threshold:
            chosen = primary
        else:
            chosen = order[:min(max_k, count)]
    selected = probabilities[chosen]
    weights = selected / selected.sum()
    return {
        "indices": [int(index) for index in chosen],
        "weights": weights,
        "probabilities": probabilities,
        "sim_max": sim_max,
        "concentration": concentration,
    }


def mix_cards(cards: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    if len(cards) != int(weights.numel()):
        raise ValueError("卡片和检索权重数量不一致")
    mixed = torch.zeros_like(cards[0])
    for card, weight in zip(cards, weights):
        mixed = mixed + weight * card
    return mixed


def borrowing_scale(
    gain: torch.Tensor, beta: torch.Tensor, sim_max: torch.Tensor, baseline: torch.Tensor,
    concentration: float, mix_norm: float, *, gamma_max: float = 1.0, has_card: bool = True,
) -> torch.Tensor:
    """gamma = gamma_max * sigmoid(g) * sigmoid(beta * (sim_max - s0)) * c_ret。门控失败时为 0。"""
    raw = gamma_max * torch.sigmoid(gain) * torch.sigmoid(beta * (sim_max - baseline)) * float(concentration)
    open_gate = has_card and mix_norm > 0 and concentration > 0
    return raw * (1.0 if open_gate else 0.0)


def positive_beta(raw_beta: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw_beta)


def initial_gamma_parameters(g_init: float = -1.0, beta_init: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    gain = torch.tensor(float(g_init), dtype=torch.float32, requires_grad=True)
    raw_beta = torch.tensor(log(torch.expm1(torch.tensor(float(beta_init))).item()), dtype=torch.float32, requires_grad=True)
    return gain, raw_beta


def gamma_proxy_loss(
    gammas: list[torch.Tensor], anchors: list[torch.Tensor], query_grads: list[torch.Tensor], mixes: list[torch.Tensor],
) -> torch.Tensor:
    """L = Σ gamma * A * <g_Q, B_mix>。点积为负表示一阶上有利于降低 Query 损失。"""
    if not gammas:
        raise ValueError("没有可计算的模块借用")
    total = gammas[0].new_zeros(())
    for gamma, anchor, query_grad, mix in zip(gammas, anchors, query_grads, mixes):
        total = total + gamma * anchor * torch.dot(query_grad.detach().reshape(-1), mix.detach().reshape(-1))
    return total


def borrow_mask(count: int, probability: float, seed: int) -> torch.Tensor:
    """整条任务一起决定。True 表示这条任务允许借用。"""
    if count < 1:
        raise ValueError("任务数必须为正")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.rand(count, generator=generator) >= float(probability)


class RollingMedian:
    """Meta-Train 相似度的滚动中位数。空窗口基线按 0 计算，决定之后才写入本条任务。"""

    def __init__(self, window: int):
        if int(window) < 1:
            raise ValueError("滚动窗口必须为正")
        self.window = int(window)
        self.values: list[float] = []

    def baseline(self) -> float:
        if not self.values:
            return 0.0
        return float(_median(torch.tensor(self.values, dtype=torch.float32)))

    def observe(self, value: float) -> None:
        self.values.append(float(value))
        if len(self.values) > self.window:
            del self.values[:-self.window]

    def state(self) -> list[float]:
        return list(self.values)

    def load(self, values) -> None:
        self.values = [float(value) for value in values][-self.window:]


def split_half_cosine(vectors: torch.Tensor, splits: int = 3) -> torch.Tensor:
    count = int(vectors.shape[0])
    if count < 2 or splits < 1:
        raise ValueError("分半至少需要两个成员")
    scores = []
    for split in range(splits):
        left = [index for index in range(count) if (index + split) % 2 == 0]
        right = [index for index in range(count) if (index + split) % 2 == 1]
        left_mean = vectors[left].mean(dim=0)
        right_mean = vectors[right].mean(dim=0)
        scores.append(F.cosine_similarity(left_mean, right_mean, dim=0, eps=_EPS))
    return torch.stack(scores).mean()


def select_keep_ratio(
    bins_by_ratio: dict[float, list[torch.Tensor]], min_members: int = 8, splits: int = 3, default: float = 1.0,
) -> dict:
    """在成员数足够的 bin 上，选平均分半余弦最大的保留比例。没有合格 bin 时不修剪。"""
    best_ratio = default
    best_score = None
    rows = []
    for ratio, bins in bins_by_ratio.items():
        usable = [item for item in bins if item.shape[0] >= min_members]
        if not usable:
            rows.append({"ratio": float(ratio), "bins": 0, "score": None})
            continue
        score = torch.stack([split_half_cosine(item, splits) for item in usable]).mean()
        rows.append({"ratio": float(ratio), "bins": len(usable), "score": float(score)})
        if best_score is None or float(score) > best_score:
            best_score = float(score)
            best_ratio = float(ratio)
    return {"keep_ratio": best_ratio, "scores": rows, "calibrated": best_score is not None}
