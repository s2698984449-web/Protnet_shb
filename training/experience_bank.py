"""检索式经验库：建库、落盘、检索，以及 FOMAML 单条任务的借用。

投影、聚类、TIES、Top-K 和 γ 都调用已经测过的 retrieval_memory。
θ₀ 训练循环在传入经验库目录时调用这里的单步和按轮刷新，不另写一套检索。
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from collections import OrderedDict
from math import ceil, log
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from common.io import (
    file_sha256, json_sha256, load_config, load_verified_episode_groups, load_verified_group_split,
    load_verified_research_splits, named_config_path, project_root, write_csv, write_json,
)
from training.checkpoints import _atomic_output, load_checkpoint, save_checkpoint
from training.fomaml_roi import MODULE_PREFIXES, validate_detection_loss_settings
from training.retrieval_memory import (
    KEY_FOR_MODULE, MODULES, _coherence, adaptive_topk, appearance_key, box_weighted_mean,
    difference_key, family_task_weights, fit_fixed_pca, load_retrieval_config, merge_cluster_members,
    merge_groups, mix_cards, morphology_key, project_l2, spherical_kmeans, trim_vector,
)


_KEY_NAMES = ("Appearance", "Morphology", "Difference")


def detection_settings() -> dict:
    """建库用的检测损失和内层步长。数值来自现有配置，不在这里另设一套。"""
    baseline = load_config(named_config_path("baseline.yaml"))
    fomaml = load_config(named_config_path("fomaml.yaml"))
    training = fomaml["training"]
    loss = {**training["loss"], "weights": dict(training["loss"]["weights"])}
    loss.setdefault("defect_score", "margin")
    validate_detection_loss_settings(loss)
    inner_lr = float(training["inner_lr"])
    if inner_lr <= 0:
        raise ValueError("内层学习率必须为正")
    return {
        "loss": loss,
        "where_loss_weight": float(training["where_loss_weight"]),
        "ignore_iou": float(baseline["evaluation"]["ignore_iou"]),
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "views_per_group": int(training["views_per_group"]),
        "inner_lr": inner_lr,
    }


def assign_experience_groups(family_ids: list[str], groups: int, same_object: list[list[str]]) -> dict[str, int]:
    """把源族分进 G 个经验组。已知同物体跨数据集的族放在同一组，其余按族编号轮转。"""
    if groups < 1:
        raise ValueError("经验组数必须为正")
    present = set(family_ids)
    if len(present) != len(family_ids):
        raise ValueError("产品族编号重复")
    used = set()
    atoms = []
    for bundle in same_object:
        members = sorted({family for family in bundle if family in present})
        if not members:
            continue
        if set(members) & used:
            raise ValueError(f"同一个产品族出现在多组同物体名单里：{members}")
        atoms.append(members)
        used.update(members)
    atoms.extend([family] for family in sorted(present - used))
    atoms.sort(key=lambda members: members[0])
    loads = [0 for _ in range(groups)]
    assignment = {}
    for members in atoms:
        index = min(range(groups), key=lambda item: (loads[item], item))
        for family in members:
            assignment[family] = index
        loads[index] += len(members)
    if set(assignment) != present:
        raise AssertionError("经验组没有覆盖全部源族")
    return assignment


def select_bank_tasks(episodes: list[dict], family_ids: list[str], tasks_per_family: int) -> tuple[list[dict], list[dict]]:
    """每个源族按 Episode 编号取不同的 Support 组组合。不够 M 条时保留实际条数。"""
    if tasks_per_family < 1:
        raise ValueError("每族任务数必须为正")
    by_family: dict[str, list[dict]] = {}
    for episode in episodes:
        by_family.setdefault(episode["family_id"], []).append(episode)
    selected, report = [], []
    for family in family_ids:
        pool = sorted(by_family.get(family, []), key=lambda item: item["episode_id"])
        chosen, seen = [], set()
        for episode in pool:
            signature = tuple(sorted(unit["physical_group_id"] for unit in episode["support"]))
            if signature in seen:
                continue
            seen.add(signature)
            chosen.append(episode)
            if len(chosen) == tasks_per_family:
                break
        report.append({
            "family_id": family, "available": len(pool), "requested": tasks_per_family, "selected": len(chosen),
        })
        selected.extend(chosen)
    return selected, report


def limit_role_views(records: list[dict], limit: int) -> list[dict]:
    """每个物理组只留 limit 个视角。缺陷组优先留有框的视角，组本身不被拆成多个样本。"""
    if limit < 1:
        raise ValueError("每组视角数必须为正")
    buckets: dict[tuple, list[dict]] = {}
    order = []
    for row in records:
        key = (row["split_role"], row["physical_group_id"])
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(row)
    selected = []
    for key in order:
        rows = buckets[key]
        if key[0] == "SUPPORT":
            rows = sorted(rows, key=lambda row: (0 if row.get("annotations") else 1, str(row["image_id"])))
        else:
            rows = sorted(rows, key=lambda row: str(row["image_id"]))
        selected.extend(rows[:limit])
    return selected


def views_geometry(per_image_boxes: list[torch.Tensor], image_sizes: list[tuple[int, int]]) -> torch.Tensor:
    """与 geometry_descriptor 同一套六维统计。每张图用自己的面积，因此可以混合不同分辨率。"""
    if not per_image_boxes or len(per_image_boxes) != len(image_sizes):
        raise ValueError("几何统计需要每张图的框和高宽")
    areas, aspects, relative, counts = [], [], [], []
    for boxes, hw in zip(per_image_boxes, image_sizes):
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("框必须是 [n, 4]")
        if int(hw[0]) < 1 or int(hw[1]) < 1:
            raise ValueError("图像高宽必须为正")
        count = int(boxes.shape[0])
        counts.append(log(count) if count else 0.0)
        if count == 0:
            continue
        widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-12)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-12)
        area = widths * heights
        areas.append(area.log())
        aspects.append((widths / heights).log())
        relative.append((area / float(hw[0] * hw[1])).log())
    if not areas:
        return torch.zeros(6)
    area_values = torch.cat(areas)
    aspect_values = torch.cat(aspects)
    relative_values = torch.cat(relative)
    count_values = torch.tensor(counts, dtype=area_values.dtype)
    return torch.stack([
        area_values.mean(), area_values.std(unbiased=False),
        aspect_values.mean(), aspect_values.std(unbiased=False),
        relative_values.mean(), count_values.mean(),
    ])


def group_prototypes(views: list[dict]) -> list[dict]:
    """先按物理组聚合视角，再交给任务签名。"""
    grouped: dict[str, list[dict]] = {}
    order = []
    for view in views:
        group_id = view["physical_group_id"]
        if group_id not in grouped:
            order.append(group_id)
            grouped[group_id] = []
        grouped[group_id].append(view)
    prototypes = []
    for group_id in order:
        members = grouped[group_id]
        features = [member["f_def"] for member in members if member["f_def"] is not None and member["f_def"].numel()]
        if features:
            stacked = torch.cat(features, dim=0)
            defect = box_weighted_mean(stacked, torch.ones(stacked.shape[0]))
            count = int(stacked.shape[0])
        else:
            defect = None
            count = 0
        prototypes.append({
            "physical_group_id": group_id,
            "f_img": torch.stack([member["f_img"] for member in members]).mean(dim=0),
            "f_def": defect,
            "box_count": count,
            "boxes": [member["boxes"] for member in members],
            "hw": [member["hw"] for member in members],
        })
    return prototypes


def task_keys_from_groups(support_groups: list[dict], reference_groups: list[dict]) -> dict[str, torch.Tensor]:
    """Support 提供缺陷原型和几何，R 提供正常外观。Query 不进入这个函数。"""
    if not reference_groups or not support_groups:
        raise ValueError("任务签名需要 Support 和正常参照")
    proto_norm = appearance_key(torch.stack([group["f_img"] for group in reference_groups]))
    usable = [group for group in support_groups if group["f_def"] is not None and group["box_count"] > 0]
    if not usable:
        raise ValueError("Support 没有缺陷框特征")
    proto_def = box_weighted_mean(
        torch.stack([group["f_def"] for group in usable]),
        torch.tensor([group["box_count"] for group in usable], dtype=torch.float32),
    )
    boxes, sizes = [], []
    for group in support_groups:
        boxes.extend(group["boxes"])
        sizes.extend(group["hw"])
    geometry = views_geometry(boxes, sizes)
    appearance = proto_norm
    morphology = morphology_key(proto_def, geometry)
    difference = difference_key(proto_def, proto_norm)
    if appearance.numel() != proto_def.numel() or difference.numel() != proto_def.numel():
        raise AssertionError("外观、差异和缺陷原型维度不一致")
    if morphology.numel() != proto_def.numel() + 6:
        raise AssertionError("形态签名维度不正确")
    return {
        "Appearance": appearance, "Morphology": morphology, "Difference": difference,
        "proto_def": proto_def, "proto_norm": proto_norm, "geometry": geometry,
    }


def query_exposure(index_rows: list[dict], episodes: list[dict]) -> dict:
    """检查监督 θ₀ 的训练索引有没有覆盖这些源族的 Query 物理组。"""
    indexed = {(row["family_id"], row["physical_group_id"]) for row in index_rows}
    query = set()
    for episode in episodes:
        for role in ("query_defect", "query_normal"):
            for unit in episode[role]:
                query.add((episode["family_id"], unit["physical_group_id"]))
    if not query:
        status = "unknown"
    else:
        status = "true" if query & indexed else "false"
    return {
        "query_seen_by_theta0": status,
        "query_groups": len(query),
        "overlap_groups": len(query & indexed),
    }


def assert_sources_allowed(family_ids: list[str], scope: dict) -> None:
    forbidden = set(scope["oof_target_family_ids"]) | set(scope["meta_val_family_ids"]) | set(scope["outer_test_family_ids"])
    leaked = sorted(set(family_ids) & forbidden)
    if leaked:
        raise AssertionError(f"经验库混入了留出、验证或外层测试族：{leaked}")
    allowed = set(scope["theta0_train_family_ids"])
    if not set(family_ids) <= allowed:
        raise AssertionError("经验库源族超出了当前折允许的 Meta-Train 源族")


def module_layout(named: dict[str, torch.Tensor]) -> list[dict]:
    offset = 0
    layout = []
    for name, value in named.items():
        numel = int(value.numel())
        layout.append({"name": name, "shape": list(value.shape), "offset": offset, "numel": numel})
        offset += numel
    return layout


def pack_named(named: dict[str, torch.Tensor], layout: list[dict]) -> torch.Tensor:
    parts = []
    for item in layout:
        value = named[item["name"]].detach().reshape(-1).float().cpu()
        if int(value.numel()) != int(item["numel"]):
            raise AssertionError(f"参数长度和布局不一致：{item['name']}")
        parts.append(value)
    if not parts:
        raise ValueError("模块没有参数")
    return torch.cat(parts)


def unpack_named(flat: torch.Tensor, layout: list[dict]) -> dict[str, torch.Tensor]:
    named = {}
    for item in layout:
        start = int(item["offset"])
        stop = start + int(item["numel"])
        named[item["name"]] = flat[start:stop].reshape(item["shape"])
    return named


def trim_packed(vector: torch.Tensor, layout: list[dict], keep_ratio: float) -> torch.Tensor:
    """按 layout 分段做幅度裁剪，避免 unpack/pack 再分配一整份命名字典。"""
    ratio = float(keep_ratio)
    if ratio >= 1.0 - 1e-12:
        return vector
    parts = []
    for item in layout:
        start = int(item["offset"])
        stop = start + int(item["numel"])
        chunk = vector[start:stop].reshape(item["shape"])
        parts.append(trim_vector(chunk, ratio).reshape(-1))
    return torch.cat(parts)


def flat_sketch_indices(layout: list[dict], indices: list[dict]) -> torch.Tensor:
    """把分层草图坐标折成 packed 向量上的一维下标。"""
    by_name = {item["name"]: item for item in layout}
    pieces = []
    for item in indices:
        entry = by_name[item["name"]]
        pieces.append(int(entry["offset"]) + item["index"].long())
    if not pieces:
        raise ValueError("草图坐标为空")
    return torch.cat(pieces)


def layered_sketch_indices(layout: list[dict], dim: int, seed: int) -> list[dict]:
    """按参数张量的元素数分配草图坐标，固定随机种子。"""
    total = sum(int(item["numel"]) for item in layout)
    if total < 1 or dim < 1:
        raise ValueError("草图维度和参数量必须为正")
    width = min(int(dim), total)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if width == total:
        return [
            {"name": item["name"], "index": torch.arange(int(item["numel"]))}
            for item in layout if int(item["numel"])
        ]
    raw = [width * int(item["numel"]) / total for item in layout]
    counts = [int(value) for value in raw]
    leftover = width - sum(counts)
    order = sorted(range(len(layout)), key=lambda index: (raw[index] - counts[index], int(layout[index]["numel"])), reverse=True)
    for index in order:
        if leftover <= 0:
            break
        room = int(layout[index]["numel"]) - counts[index]
        if room <= 0:
            continue
        take = min(room, leftover)
        counts[index] += take
        leftover -= take
    if sum(counts) != width:
        raise AssertionError("草图坐标没有分配满")
    selected = []
    for item, count in zip(layout, counts):
        if count <= 0:
            continue
        draw = torch.randperm(int(item["numel"]), generator=generator)[:count]
        selected.append({"name": item["name"], "index": draw.sort().values})
    return selected


def sketch_values(named: dict[str, torch.Tensor], indices: list[dict], keep_ratio: float) -> torch.Tensor:
    pieces = []
    for item in indices:
        values = named[item["name"]].detach().reshape(-1).float().cpu()
        local = item["index"]
        if keep_ratio < 1:
            keep = max(1, ceil(keep_ratio * int(values.numel())))
            if keep < int(values.numel()):
                chosen = torch.topk(values.abs(), keep, largest=True, sorted=False).indices
                # 只对草图下标做成员判定，避免为 Feature 级千万维分配整表 bool mask。
                ordered = chosen.sort().values
                positions = torch.searchsorted(ordered, local.long()).clamp(max=max(int(ordered.numel()) - 1, 0))
                matched = ordered[positions] == local.long() if int(ordered.numel()) else torch.zeros_like(local, dtype=torch.bool)
                picked = values[local]
                pieces.append(torch.where(matched, picked, torch.zeros_like(picked)))
                continue
        pieces.append(values[local])
    if not pieces:
        raise ValueError("草图坐标为空")
    return torch.cat(pieces)


def sketch_packed(vector: torch.Tensor, layout: list[dict], indices: list[dict], keep_ratio: float) -> torch.Tensor:
    """在 packed 向量上按与 sketch_values 相同的分段保留规则取草图。"""
    ratio = float(keep_ratio)
    if ratio >= 1.0 - 1e-12:
        return vector.detach().float().cpu().reshape(-1)[flat_sketch_indices(layout, indices)]
    trimmed = trim_packed(vector.detach().float().cpu().reshape(-1), layout, ratio)
    return trimmed[flat_sketch_indices(layout, indices)]


def _storage_bytes(payload: dict) -> int:
    if payload["format"] == "zeros":
        return 0
    if payload["format"] == "dense_fp16":
        return int(payload["values"].numel()) * 2
    if payload["format"] == "sparse_fp16":
        return int(payload["index"].numel()) * 6
    if payload["format"] == "dense_fp32":
        return int(payload["values"].numel()) * 4
    raise ValueError(f"未知卡片格式：{payload['format']}")


def decode_vector(payload: dict) -> torch.Tensor:
    kind = payload["format"]
    if kind == "zeros":
        return torch.zeros(int(payload["numel"]))
    if kind == "dense_fp16":
        return payload["values"].float().reshape(-1)
    if kind == "dense_fp32":
        return payload["values"].float().reshape(-1)
    if kind == "sparse_fp16":
        restored = torch.zeros(int(payload["numel"]))
        index = payload["index"].long()
        if int(index.numel()):
            restored[index] = payload["values"].float()
        return restored
    raise ValueError(f"未知卡片格式：{kind}")


def _restores(original: torch.Tensor, restored: torch.Tensor) -> tuple[bool, float, float]:
    if int(original.numel()) != int(restored.numel()):
        return False, 0.0, float("inf")
    if float(torch.linalg.vector_norm(original)) < 1e-12:
        exact = bool(torch.equal(original, restored))
        return exact, 1.0 if exact else 0.0, 0.0 if exact else float("inf")
    cosine = float(F.cosine_similarity(original, restored, dim=0, eps=1e-12))
    relative = float(torch.linalg.vector_norm(original - restored) / torch.linalg.vector_norm(original))
    return cosine >= 0.999 and relative <= 1e-2, cosine, relative


def choose_vector_storage(vector: torch.Tensor) -> dict:
    """在能恢复方向的格式里选择更小的一个。密文 FP16 恢复失败或截零过多时保留 FP32。"""
    flat = vector.detach().float().cpu().reshape(-1).contiguous()
    if float(torch.linalg.vector_norm(flat)) < 1e-12:
        return {"format": "zeros", "numel": int(flat.numel()), "bytes": 0, "cosine": 1.0, "relative_error": 0.0}
    half = flat.half()
    zero_fraction = float((half == 0).sum().item()) / float(flat.numel())
    limit = float(load_retrieval_config()["fp16_zero_fraction_limit"])
    candidates = []
    if zero_fraction <= limit:
        candidates.append({"format": "dense_fp16", "values": half})
    # Feature 级千万维上构造 sparse index 会再吃掉数百 MB，只对中小向量尝试。
    if int(flat.numel()) <= 1_000_000:
        nonzero_count = int(torch.count_nonzero(flat))
        if nonzero_count * 6 < int(flat.numel()) * 2:
            nonzero = torch.nonzero(flat, as_tuple=False).reshape(-1)
            candidates.append({
                "format": "sparse_fp16", "numel": int(flat.numel()),
                "index": nonzero.to(torch.int32), "values": flat[nonzero].half(),
            })
    accepted = []
    for payload in candidates:
        ok, cosine, relative = _restores(flat, decode_vector(payload))
        if not ok:
            continue
        payload = {**payload, "bytes": _storage_bytes(payload), "cosine": cosine, "relative_error": relative}
        accepted.append(payload)
    if not accepted:
        payload = {"format": "dense_fp32", "values": flat}
        ok, cosine, relative = _restores(flat, decode_vector(payload))
        if not ok:
            raise AssertionError("FP32 卡片不能恢复")
        payload.update({"bytes": _storage_bytes(payload), "cosine": cosine, "relative_error": relative})
        return payload
    accepted.sort(key=lambda item: (item["bytes"], item["format"]))
    return accepted[0]


def clip_packed(vector: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    from training.retrieval_memory import clip_large_delta

    clipped, _scale = clip_large_delta(vector, anchor)
    return clipped


def _persist_card_file(cards_dir, module: str, cluster: int, group: int, card: dict) -> dict:
    """编码并落盘一张卡片，返回索引行。调用方随后可释放 unit。"""
    from pathlib import Path

    cards_dir = Path(cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)
    encoded = choose_vector_storage(card["unit"] if card["valid"] else torch.zeros_like(card["unit"]))
    name = f"{module}_c{int(cluster)}_g{int(group)}.pt"
    save_checkpoint(
        cards_dir / name,
        {"encoded": {k: v for k, v in encoded.items() if k not in {"bytes", "cosine", "relative_error"}}},
    )
    return {
        "module": module, "cluster": int(cluster), "group": int(group), "layer": module,
        "file": f"cards/{name}",
        "format": encoded["format"], "bytes": int(encoded["bytes"]), "cosine": float(encoded["cosine"]),
        "relative_error": float(encoded["relative_error"]), "valid": bool(card["valid"]),
        "low_sample": bool(card["low_sample"]), "coherence": float(card["coherence"]),
        "member_count": int(card["member_count"]), "families": list(card["families"]),
        "task_ids": list(card["task_ids"]),
        "utility": dict(card.get("utility") or _empty_utility()),
        "compatibility": dict(card.get("compatibility") or {}),
        "theta_version": str(card.get("theta_version") or card.get("theta_hash") or ""),
        "theta_hash": str(card.get("theta_hash") or card.get("theta_version") or ""),
        "use_count": int((card.get("utility") or {}).get("count") or 0),
        "neg_rate": float((card.get("utility") or {}).get("neg_rate") or 0.0),
    }


def _empty_utility() -> dict:
    return {
        "mean": 0.0, "variance": 0.0, "count": 0, "neg_rate": 0.0, "ema": 0.0,
    }


def _accumulate_utility(stats: dict, signal: float) -> dict:
    """单卡片效用的在线均值/方差/负迁移率；不写入 AP。"""
    count = int(stats.get("count") or 0) + 1
    prev = float(stats.get("ema") or 0.0)
    ema = signal if count == 1 else 0.9 * prev + 0.1 * signal
    mean = float(stats.get("mean") or 0.0)
    delta = signal - mean
    mean = mean + delta / count
    variance = float(stats.get("variance") or 0.0) + delta * (signal - mean)
    neg = float(stats.get("neg_rate") or 0.0)
    neg = ((neg * (count - 1)) + float(signal < 0.0)) / count
    stats.update({"mean": mean, "variance": variance, "count": count, "neg_rate": neg, "ema": ema})
    return stats


def _candidate_utility_stats(bank: dict, module: str, candidate: dict) -> dict:
    """把一个检索候选下各组卡片的历史效用聚成平滑输入；无身份字段。"""
    from training.transfer_policy import smooth_utility_features

    controller_prior = float(load_retrieval_config().get("utility_prior_count", 8.0))
    groups = list(candidate.get("groups") or [])
    cluster_id = int(candidate.get("cluster_id", -1))
    means, counts, negs, variances = [], [], [], []
    for group in groups:
        card = _lookup_card(bank, module, cluster_id, int(group))
        if card is None:
            continue
        stats = card.get("utility") or _empty_utility()
        means.append(float(stats.get("ema") if stats.get("ema") is not None else stats.get("mean") or 0.0))
        counts.append(float(stats.get("count") or 0.0))
        negs.append(float(stats.get("neg_rate") or 0.0))
        variances.append(float(stats.get("variance") or 0.0))
    if not counts:
        return {
            "stats": _empty_utility(),
            "features": smooth_utility_features(None, prior_count=controller_prior),
        }
    total = sum(counts)
    weight = [count / total if total > 0 else 1.0 / len(counts) for count in counts]
    merged = {
        "mean": sum(value * w for value, w in zip(means, weight)),
        "ema": sum(value * w for value, w in zip(means, weight)),
        "variance": sum(value * w for value, w in zip(variances, weight)),
        "count": int(round(sum(counts))),
        "neg_rate": sum(value * w for value, w in zip(negs, weight)),
    }
    return {
        "stats": merged,
        "features": smooth_utility_features(merged, prior_count=controller_prior),
    }


def update_module_contributions(bank: dict, module_utilities: dict[str, float], plans: list[dict]) -> None:
    """记录模块条件贡献与共现；不得把 U_m 广播成单卡独立效用。"""
    contribution_log = bank.setdefault("module_contribution_log", [])
    injected = [plan["module"] for plan in plans if plan.get("inject")]
    contribution_log.append({
        "modules": list(injected),
        "utilities": {module: float(value) for module, value in module_utilities.items()},
        "kind": "module_loo",
    })
    if len(contribution_log) > 256:
        del contribution_log[:-256]
    combo_log = bank.setdefault("combo_utility_log", [])
    if injected:
        combo_log.append({
            "modules": list(injected),
            "utilities": {module: float(module_utilities.get(module, 0.0)) for module in injected},
            "kind": "module_combo_loo",
        })
        if len(combo_log) > 256:
            del combo_log[:-256]
    index = bank.get("card_index") or {}
    for plan in plans:
        if not plan.get("inject"):
            continue
        module = plan["module"]
        if module not in module_utilities:
            continue
        partners = [name for name in injected if name != module]
        metas = list(plan.get("candidate_meta") or [])
        for meta in metas[:1]:
            for card_id in (meta.get("card_ids") or [])[:1]:
                parts = str(card_id).split("/")
                if len(parts) != 3:
                    continue
                key = _card_key(module, int(parts[1][1:]), int(parts[2][1:]))
                row = index.get(key)
                if row is None:
                    continue
                compat = row.setdefault("compatibility", {})
                for partner in partners:
                    bucket = compat.setdefault(partner, {"count": 0, "ema": 0.0})
                    bucket["count"] = int(bucket.get("count") or 0) + 1
                    u_m = float(module_utilities[module])
                    prev = float(bucket.get("ema") or 0.0)
                    bucket["ema"] = u_m if bucket["count"] == 1 else 0.9 * prev + 0.1 * u_m


def update_card_utilities(bank: dict, plans: list[dict], module_signals: dict[str, float]) -> None:
    """兼容旧调用名：转发为模块贡献记账，不再把模块 U 写成单卡效用。"""
    utilities = {module: -float(signal) for module, signal in module_signals.items()}
    update_module_contributions(bank, utilities, plans)



def assemble_group_cards(
    tasks: list[dict],
    load_packed,
    layouts: dict,
    anchors: dict,
    keep_ratio: dict,
    min_members: int,
    on_card=None,
) -> dict:
    """按簇和经验组调用已有 TIES。跨组不在这里合并。每次只把一个 bin、一个模块读进内存。"""
    cards = {}
    for module in MODULES:
        key_name = KEY_FOR_MODULE[module]
        bins: dict[tuple, list[dict]] = {}
        for task in tasks:
            bins.setdefault((int(task["cluster"][key_name]), int(task["group"])), []).append(task)
        print(
            f"经验库拼卡 module={module} bins={len(bins)} keep={float(keep_ratio[module]):.2f}",
            flush=True,
        )
        for (cluster, group), members in sorted(bins.items()):
            clipped_rows = []
            for task in members:
                packed = load_packed(task["task_id"])
                vector = packed[module].detach().float().cpu().contiguous()
                del packed
                clipped_rows.append(clip_packed(vector, anchors[module]))
                del vector
            clipped = torch.stack(clipped_rows)
            del clipped_rows
            weights = family_task_weights([task["family_id"] for task in members])
            ratio = float(keep_ratio[module])
            if ratio >= 1.0 - 1e-12:
                trimmed = clipped
            else:
                trimmed = torch.stack([trim_packed(row, layouts[module], ratio) for row in clipped])
            merged = merge_cluster_members(trimmed, weights, 1.0, min_members)
            coherence = float(_coherence(clipped, weights))
            del clipped, trimmed
            entry = {
                "unit": merged["unit"],
                "card": merged["card"],
                "valid": bool(merged["valid"]),
                "low_sample": bool(merged["low_sample"]),
                "coherence": coherence,
                "member_count": len(members),
                "families": sorted({task["family_id"] for task in members}),
                "task_ids": [task["task_id"] for task in members],
                "utility": _empty_utility(),
                "compatibility": {},
                "theta_version": str(task.get("theta_version") or task.get("theta_sha256") or ""),
                "theta_hash": str(task.get("theta_hash") or task.get("theta_sha256") or ""),
                "layer": module,
            }
            del merged
            if on_card is not None:
                on_card((module, cluster, group), entry)
            cards[(module, cluster, group)] = entry
            gc.collect()
    return cards


def _card_key(module: str, cluster: int, group: int) -> str:
    return f"{module}/{int(cluster)}/{int(group)}"


def _empty_retrieval(module: str, sim_max: float = 0.0, concentration: float = 0.0) -> dict:
    return {
        "valid": False, "module": module, "vector": None, "named": {}, "candidates": [],
        "source_groups": [], "cluster_ids": [], "weights": [],
        "sim_max": float(sim_max), "concentration": float(concentration),
    }


def retrieve_module_experience(
    bank: dict, module: str, key: torch.Tensor, excluded_group: int | None,
    allowed_families: set[str] | None = None,
) -> dict:
    """用任务签名检索一个模块。excluded_group 是训练任务自己的经验组；外部目标传 None。

    allowed_families 给出这条任务可以使用的源族。卡片里只要有一个成员不在其中，整张卡片不用。
    """
    if module not in KEY_FOR_MODULE:
        raise ValueError(f"未知模块：{module}")
    key_name = KEY_FOR_MODULE[module]
    projected = project_l2(key.detach().float().cpu(), bank["pca"][key_name])
    valid = bank["center_valid"][key_name]
    centers = bank["centers"][key_name]
    usable = torch.where(valid)[0]
    if int(usable.numel()) == 0 or float(torch.linalg.vector_norm(projected)) < 1e-12:
        return _empty_retrieval(module)
    similarities = centers[usable] @ projected
    picked = adaptive_topk(
        similarities, float(bank["top_mass_threshold"]), int(bank["max_top_k"]), float(bank["temperature"]),
    )
    cluster_ids = [int(usable[index]) for index in picked["indices"]]
    raw_weights = [float(value) for value in picked["weights"]]
    vectors, weights, used_groups, candidates = [], [], [], []
    width = int(bank["layouts"][module][-1]["offset"] + bank["layouts"][module][-1]["numel"]) if bank["layouts"][module] else 0
    groups = int(bank["experience_groups"])
    center_sims = {int(usable[index]): float(similarities[index]) for index in range(int(usable.numel()))}
    for cluster_id, weight in zip(cluster_ids, raw_weights):
        units, counts, coherences, present = [], [], [], []
        for group in range(groups):
            card = _lookup_card(bank, module, cluster_id, group)
            if excluded_group is not None and group == int(excluded_group):
                units.append(torch.zeros(width))
                counts.append(0)
                coherences.append(torch.zeros(()))
                continue
            if card is None or not card["valid"] or card["member_count"] < 1:
                units.append(torch.zeros(width))
                counts.append(0)
                coherences.append(torch.zeros(()))
                continue
            families = {str(family) for family in card.get("families") or []}
            if allowed_families is not None and (not families or not families <= set(allowed_families)):
                units.append(torch.zeros(width))
                counts.append(0)
                coherences.append(torch.zeros(()))
                continue
            units.append(card["unit"])
            counts.append(int(card["member_count"]))
            coherences.append(torch.tensor(float(card["coherence"])))
            present.append(group)
        merged = merge_groups(units, counts, coherences, None)
        if not merged["valid"]:
            continue
        if excluded_group is not None and int(excluded_group) in present:
            raise AssertionError("检索结果包含了被排除的经验组")
        vectors.append(merged["vector"])
        weights.append(weight)
        used_groups.extend(present)
        candidates.append({
            "vector": merged["vector"],
            "sim": float(center_sims.get(int(cluster_id), 0.0)),
            "cluster_id": int(cluster_id),
            "groups": list(present),
            "card_ids": [f"{module}/c{cluster_id}/g{group}" for group in present],
        })
    if not vectors:
        empty = _empty_retrieval(module, float(picked["sim_max"]), float(picked["concentration"]))
        empty["candidates"] = []
        return empty
    scale = sum(weights)
    mix = mix_cards(vectors, torch.tensor(weights) / scale)
    return {
        "valid": True, "module": module, "vector": mix,
        "named": unpack_named(mix, bank["layouts"][module]),
        "candidates": candidates,
        "source_groups": sorted(set(used_groups)), "cluster_ids": cluster_ids, "weights": [value / scale for value in weights],
        "sim_max": float(picked["sim_max"]), "concentration": float(picked["concentration"]),
    }


def _lookup_card(bank: dict, module: str, cluster: int, group: int) -> dict | None:
    memory = bank.get("cards_memory") or {}
    key = (module, int(cluster), int(group))
    if key in memory:
        return memory[key]
    index = bank.get("card_index") or {}
    meta = index.get(_card_key(module, cluster, group))
    if meta is None:
        return None
    cache = bank.setdefault("_cache", OrderedDict())
    cache_key = _card_key(module, cluster, group)
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]
    payload = load_checkpoint(Path_of(bank) / meta["file"])
    card = {**meta, "unit": decode_vector(payload["encoded"]), "valid": bool(meta["valid"])}
    cache[cache_key] = card
    while len(cache) > 8:
        cache.popitem(last=False)
    return card


def Path_of(bank: dict):
    from pathlib import Path

    return Path(bank["directory"])


def save_experience_bank(directory, bank: dict, card_rows: list[dict] | None = None) -> dict:
    """写索引、清单和卡片分片。完整卡片不进入 GPU。"""
    from pathlib import Path

    directory = Path(directory)
    cards_dir = directory / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    if card_rows is None:
        card_rows = []
        for key in sorted(bank.get("cards_memory") or {}, key=lambda item: (item[0], item[1], item[2])):
            module, cluster, group = key
            card = bank["cards_memory"][key]
            row = _persist_card_file(cards_dir, module, cluster, group, card)
            card_rows.append(row)
            card["unit"] = torch.empty(0)
            card["card"] = torch.empty(0)
            gc.collect()
    else:
        card_rows = list(card_rows)
    manifest = bank["manifest"]
    manifest_hash = json_sha256(manifest)
    write_json(directory / "experience_manifest.json", manifest)
    write_csv(
        directory / "near_duplicate_candidates.csv",
        ["family_a", "family_b", "cosine", "same_experience_group", "merged"],
        bank.get("near_duplicates") or [],
    )
    index = {
        "kind": "experience_bank",
        "theta0_sha256": bank["theta0_sha256"],
        "theta_sha256": bank["theta_sha256"],
        "preprocess_sha256": bank["preprocess_sha256"],
        "split_sha256": bank["split_sha256"],
        "manifest_sha256": manifest_hash,
        "experience_groups": int(bank["experience_groups"]),
        "clusters_requested": int(bank["clusters_requested"]),
        "top_mass_threshold": float(bank["top_mass_threshold"]),
        "max_top_k": int(bank["max_top_k"]),
        "temperature": float(bank["temperature"]),
        "groups": dict(bank["groups"]),
        "layouts": bank["layouts"],
        "pca": bank["pca"],
        "centers": bank["centers"],
        "center_valid": bank["center_valid"],
        "anchors": {name: float(value) for name, value in bank["anchors"].items()},
        "keep_ratio": {name: float(value) for name, value in bank["keep_ratio"].items()},
        "next_keep_ratio": bank["next_keep_ratio"],
        "trim_calibration": bank["trim_calibration"],
        "cards": card_rows,
        "signatures": bank["signatures"],
        "assignments": bank["assignments"],
    }
    save_checkpoint(directory / "module_memory.pt", index)
    bank["card_index"] = {_card_key(row["module"], row["cluster"], row["group"]): row for row in card_rows}
    bank["directory"] = str(directory)
    bank["manifest_sha256"] = manifest_hash
    return index


def latest_experience_bank(fold_id: int, root=None):
    """当前折里最新一份检索式经验库。不需要在命令里再指定目录。"""
    from pathlib import Path

    if root is None:
        root = project_root() / load_retrieval_config()["output_dir"]
    fold = Path(root) / f"fold_{int(fold_id)}"
    if not fold.is_dir():
        raise FileNotFoundError(f"当前折还没有经验库：{fold}")
    found = []
    for path in fold.iterdir():
        index = path / "module_memory.pt"
        if not path.is_dir() or not index.is_file():
            continue
        payload = load_checkpoint(index)
        if payload.get("kind") == "experience_bank":
            found.append((index.stat().st_mtime, path.name, path))
    if not found:
        raise FileNotFoundError(f"当前折还没有检索式经验库：{fold}")
    found.sort()
    return found[-1][2]


def load_experience_bank(directory, theta_sha256: str | None = None) -> dict:
    """只加载索引、签名和聚类。卡片在检索时按需读取。"""
    from pathlib import Path

    directory = Path(directory)
    index = load_checkpoint(directory / "module_memory.pt")
    if index.get("kind") != "experience_bank":
        raise ValueError(f"不是检索式经验库：{directory}")
    layout_modules = set((index.get("layouts") or {}).keys())
    if layout_modules != set(MODULES):
        raise ValueError(
            f"经验库模块布局是 {sorted(layout_modules)}，当前协议需要 {list(MODULES)}；"
            "请用五功能组协议重建经验库"
        )
    if theta_sha256 is not None and index["theta_sha256"] != theta_sha256:
        raise ValueError("经验库的参数版本和当前 θ 不一致")
    manifest = json.loads((directory / "experience_manifest.json").read_text(encoding="utf-8"))
    if json_sha256(manifest) != index["manifest_sha256"]:
        raise ValueError("经验库清单和索引哈希不一致")
    bank = {
        "directory": str(directory),
        "theta0_sha256": index["theta0_sha256"],
        "theta_sha256": index["theta_sha256"],
        "preprocess_sha256": index["preprocess_sha256"],
        "split_sha256": index["split_sha256"],
        "manifest_sha256": index["manifest_sha256"],
        "experience_groups": int(index["experience_groups"]),
        "clusters_requested": int(index["clusters_requested"]),
        "top_mass_threshold": float(index["top_mass_threshold"]),
        "max_top_k": int(index["max_top_k"]),
        "temperature": float(index["temperature"]),
        "groups": {str(key): int(value) for key, value in index["groups"].items()},
        "layouts": index["layouts"],
        "pca": index["pca"],
        "centers": index["centers"],
        "center_valid": index["center_valid"],
        "anchors": {str(key): float(value) for key, value in index["anchors"].items()},
        "keep_ratio": {str(key): float(value) for key, value in index["keep_ratio"].items()},
        "next_keep_ratio": index["next_keep_ratio"],
        "trim_calibration": index["trim_calibration"],
        "signatures": index["signatures"],
        "assignments": index["assignments"],
        "card_index": {_card_key(row["module"], row["cluster"], row["group"]): row for row in index["cards"]},
        "manifest": manifest,
    }
    return bank


def _state_snapshot(module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _state_difference(module, snapshot: dict[str, torch.Tensor]) -> tuple[float, str]:
    current = module.state_dict()
    if set(current) != set(snapshot):
        raise AssertionError("建库改变了参数或 buffer 的名称")
    worst, owner = 0.0, ""
    for name, tensor in current.items():
        diff = float((tensor.detach().cpu() - snapshot[name]).abs().max())
        if diff > worst:
            worst, owner = diff, name
    return worst, owner


def _state_sha256(snapshot: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        tensor = snapshot[name].contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _task_seed(bank_seed: int, episode_id: str) -> int:
    digest = hashlib.sha256(f"{bank_seed}:{episode_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _preprocess_sha256(detector, version: str) -> str:
    transform = detector.transform
    payload = {
        "version": version,
        "min_size": [int(value) for value in transform.min_size],
        "max_size": int(transform.max_size),
        "size_divisible": int(transform.size_divisible),
        "mean": [float(value) for value in transform.image_mean],
        "std": [float(value) for value in transform.image_std],
        "b0_v3_sha256": file_sha256(named_config_path("b0_v3.yaml")),
    }
    return json_sha256(payload)


def _cache_name(image_id: str) -> str:
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest() + ".pt"


def _extract_view(detector, record: dict, device) -> dict:
    from training.fomaml_roi import _transformed_view

    detector.eval()
    _original, images, targets = _transformed_view(detector, record, device)
    pyramid = detector.backbone(images.tensors)
    means = []
    for name in sorted(pyramid):
        level = pyramid[name]
        if int(level.shape[1]) != 256:
            raise AssertionError(f"FPN 通道不是 256：{name} {tuple(level.shape)}")
        means.append(level.mean(dim=(2, 3)).reshape(-1))
    f_img = torch.stack(means).mean(dim=0).detach().float().cpu()
    boxes = targets[0]["boxes"]
    if int(boxes.shape[0]):
        pooled = detector.roi_heads.box_roi_pool(pyramid, [boxes], images.image_sizes)
        if tuple(pooled.shape[1:]) != (256, 7, 7):
            raise AssertionError(f"RoIAlign 输出不是 256×7×7：{tuple(pooled.shape)}")
        f_def = pooled.mean(dim=(2, 3)).detach().float().cpu()
    else:
        f_def = torch.zeros(0, 256)
    del pyramid, images, targets
    original_boxes = torch.tensor(
        [annotation["bbox"] for annotation in record.get("annotations") or []], dtype=torch.float32,
    ).reshape(-1, 4)
    return {
        "image_id": record["image_id"],
        "physical_group_id": record["physical_group_id"],
        "view_id": record.get("view_id", ""),
        "split_role": record["split_role"],
        "f_img": f_img,
        "f_def": f_def,
        "boxes": original_boxes,
        "hw": [int(record["height"]), int(record["width"])],
    }


def _support_deltas(detector, records: list[dict], device, settings: dict) -> tuple[float, dict[str, dict[str, torch.Tensor]]]:
    from training.theta0 import _mean_module_grads

    detector.zero_grad(set_to_none=True)
    loss, grads = _mean_module_grads(detector, records, device, settings)
    named = [(name, parameter) for name, parameter in detector.named_parameters() if parameter.requires_grad]
    if len(named) != len(grads):
        raise AssertionError("梯度数量和可训练参数不一致")
    grouped = {label: {} for label in MODULE_PREFIXES}
    for (name, parameter), grad in zip(named, grads):
        owners = [label for label, prefixes in MODULE_PREFIXES.items() if name.startswith(prefixes)]
        if len(owners) != 1:
            raise AssertionError(f"参数没有唯一模块：{name}")
        value = torch.zeros(parameter.shape) if grad is None else grad.detach().float().cpu()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"模块梯度不是有限值：{name}")
        snapshot = settings.get("inner_lr_snapshot")
        scale = float(snapshot[owners[0]]) if snapshot else float(settings["inner_lr"])
        grouped[owners[0]][name] = value.mul(-scale)
    del grads
    detector.zero_grad(set_to_none=True)
    detector.eval()
    return float(loss), grouped


def _fit_keys(raw: dict[str, torch.Tensor], families: list[str], clusters: int, seed: int) -> dict:
    fitted = {}
    for offset, name in enumerate(_KEY_NAMES):
        pca = fit_fixed_pca(raw[name], 64)
        projected = torch.stack([project_l2(row, pca) for row in raw[name]])
        clustering = spherical_kmeans(projected, families, clusters, seed + offset)
        fitted[name] = {"pca": pca, "clustering": clustering, "projected": projected}
    return fitted


def _directory_bytes(directory) -> int:
    from pathlib import Path

    return sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())


def build_experience_bank(theta0, run_dir, *, max_families: int | None = None, tasks_per_family: int | None = None) -> dict:
    """冻结 θ₀ 做签名和聚类，再在同一份 θ 上计算一步梯度卡片。不更新 θ。"""
    from pathlib import Path

    from data_proc.baseline_data import build_b0_index
    from data_proc.fomaml_episodes import InsufficientEpisodeAnnotations, load_training_episodes
    from models.detector import b0_model_config, build_faster_rcnn
    from training.checkpoints import load_checkpoint as read_weights
    from training.theta0 import _open_modules
    from training.train_b0 import _check_runtime

    started = perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print("经验库开始：校验冻结划分并读取 θ₀", flush=True)
    load_verified_group_split()
    load_verified_research_splits()
    load_verified_episode_groups()
    config = load_retrieval_config()
    for required in ("cluster_seed", "bank_seed", "signature_version", "same_object_families", "near_duplicate_cosine"):
        if required not in config:
            raise ValueError(f"retrieval_fomaml.yaml 缺少 {required}")
    if config.get("train_projection"):
        raise ValueError("第一版不训练投影")
    settings = detection_settings()
    theta0 = Path(theta0).resolve(strict=True)
    run_dir = Path(run_dir)
    theta0_file_hash = file_sha256(theta0)
    parent = read_weights(theta0)
    if file_sha256(theta0) != theta0_file_hash:
        raise RuntimeError("监督 θ₀ 在读取时发生变化")
    source = parent["provenance"]
    if source.get("mode") != "fold_train_full" or source.get("theta0_training_method"):
        raise ValueError("经验库需要完成的正式折内监督 θ₀")
    adapter, episodes, scope = load_training_episodes(source["fold_id"], source["train_family_ids"])
    print(
        f"经验库数据分组 {scope.get('data_profile')}，源族 {len(scope['train_family_ids'])}，"
        f"候选 Episode {len(episodes)}",
        flush=True,
    )
    b0_model_config(source["settings"]["model"])
    families = list(scope["train_family_ids"])
    if max_families is not None:
        if int(max_families) < 1:
            raise ValueError("max-families 必须为正")
        families = families[: int(max_families)]
    if tasks_per_family is not None:
        requested_tasks = int(tasks_per_family)
    elif scope.get("bank_tasks_per_family") is not None:
        requested_tasks = int(scope["bank_tasks_per_family"])
    else:
        requested_tasks = int(config["tasks_per_family"])
    assert_sources_allowed(families, scope)
    groups = assign_experience_groups(families, int(config["experience_groups"]), config["same_object_families"])
    tasks, task_report = select_bank_tasks(episodes, families, requested_tasks)
    if len(tasks) < 2:
        raise ValueError("PCA 至少需要两条成功的源任务，当前选出的 Bank 任务不足")

    index_rows = build_b0_index(adapter, load_verified_episode_groups(), set(families), load_verified_group_split())
    exposure = query_exposure(index_rows, [episode for episode in episodes if episode["family_id"] in set(families)])
    device = _check_runtime(source["settings"])
    detector = build_faster_rcnn(source["settings"]["model"], initialize_backbone=False)
    detector.load_state_dict(parent["model"], strict=True)
    del parent
    flags = {name: parameter.requires_grad for name, parameter in detector.named_parameters()}
    _open_modules(detector)
    snapshot = _state_snapshot(detector)
    theta_hash = _state_sha256(snapshot)
    detector.to(device).eval()
    preprocess_hash = _preprocess_sha256(detector, config["signature_version"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    cache_dir = run_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_started = perf_counter()
    prepared = []
    query_ids_seen = set()
    for episode in tasks:
        query_ids = {
            image_id
            for role in ("query_defect", "query_normal")
            for unit in episode[role]
            for image_id in unit["image_ids"]
        }
        query_ids_seen.update(query_ids)
        records = adapter.build_episode_records(episode, roles=("reference", "support"))
        roles = {record["split_role"] for record in records}
        if roles - {"R", "SUPPORT"}:
            raise AssertionError(f"签名读取了 Support 和 R 以外的角色：{sorted(roles)}")
        if any(record["image_id"] in query_ids for record in records):
            raise AssertionError(f"任务签名混入 Query 图像：{episode['episode_id']}")
        support_ids = {unit["physical_group_id"] for unit in episode["support"]}
        reference_ids = {unit["physical_group_id"] for unit in episode["reference"]}
        query_group_ids = {
            unit["physical_group_id"]
            for role in ("query_defect", "query_normal")
            for unit in episode[role]
        }
        if support_ids & reference_ids or support_ids & query_group_ids or reference_ids & query_group_ids:
            raise AssertionError(f"R、Support、Query 物理组重叠：{episode['episode_id']}")
        limited = limit_role_views(records, settings["views_per_group"])
        views = []
        for record in limited:
            if record["split_role"] == "R" and (int(record["normal_flag"]) != 1 or record["annotations"]):
                raise AssertionError(f"正常参照不是纯正常图：{record['image_id']}")
            cached = cache_dir / _cache_name(record["image_id"])
            view = load_checkpoint(cached) if cached.is_file() else None
            if view is None or view.get("theta0_sha256") != theta0_file_hash or view.get("preprocess_sha256") != preprocess_hash:
                view = _extract_view(detector, record, device)
                view.update({
                    "theta0_sha256": theta0_file_hash,
                    "preprocess_sha256": preprocess_hash,
                    "signature_version": config["signature_version"],
                    "view_id": str(record.get("view_id") or ""),
                })
                save_checkpoint(cached, view)
            views.append(view)
        support_views = [view for view in views if view["split_role"] == "SUPPORT"]
        reference_views = [view for view in views if view["split_role"] == "R"]
        try:
            keys = task_keys_from_groups(group_prototypes(support_views), group_prototypes(reference_views))
        except ValueError as error:
            print(f"经验库跳过签名 {episode['episode_id']}：{error}", flush=True)
            prepared.append({"task_id": episode["episode_id"], "skipped": str(error)})
            continue
        prepared.append({
            "episode": episode,
            "task_id": episode["episode_id"],
            "keys": keys,
            "support_records": [record for record in limited if record["split_role"] == "SUPPORT"],
            "support_image_ids": [view["image_id"] for view in support_views],
            "reference_image_ids": [view["image_id"] for view in reference_views],
            "support_group_ids": sorted({view["physical_group_id"] for view in support_views}),
            "reference_group_ids": sorted({view["physical_group_id"] for view in reference_views}),
        })
        print(f"经验库签名 {episode['episode_id']} support={len(support_views)} reference={len(reference_views)}", flush=True)
    feature_seconds = perf_counter() - feature_started
    signature_skipped = [item for item in prepared if item.get("skipped")]
    prepared = [item for item in prepared if not item.get("skipped")]
    if query_ids_seen & {image_id for item in prepared for image_id in item["support_image_ids"] + item["reference_image_ids"]}:
        raise AssertionError("Query 图像进入了特征缓存")

    gradient_started = perf_counter()
    delta_dir = run_dir / "task_deltas"
    delta_dir.mkdir(parents=True, exist_ok=True)
    succeeded = []
    skipped = []
    layouts = None
    for item in prepared:
        seed = _task_seed(int(config["bank_seed"]), item["task_id"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        try:
            loss, deltas = _support_deltas(detector, item["support_records"], device, settings)
        except InsufficientEpisodeAnnotations as error:
            skipped.append({"task_id": item["task_id"], "reason": str(error)})
            difference, owner = _state_difference(detector, snapshot)
            if difference != 0.0:
                raise AssertionError(f"失败任务改变了参数 {owner}，差值 {difference}") from error
            continue
        difference, owner = _state_difference(detector, snapshot)
        if difference != 0.0 or any(parameter.grad is not None for parameter in detector.parameters()):
            raise AssertionError(f"建库梯度改变了共享参数或留下梯度：{owner} {difference}")
        if layouts is None:
            layouts = {module: module_layout(deltas[module]) for module in MODULES}
        item_packed = {module: pack_named(deltas[module], layouts[module]) for module in MODULES}
        norms = {module: float(torch.linalg.vector_norm(item_packed[module])) for module in MODULES}
        safe_name = item["task_id"].replace(":", "_") + ".pt"
        save_checkpoint(delta_dir / safe_name, {
            "theta_sha256": theta_hash, "task_id": item["task_id"], "loss": loss, "norms": norms, "delta": item_packed,
        })
        item.update({"seed": seed, "loss": loss, "norms": norms, "delta_file": f"task_deltas/{safe_name}"})
        succeeded.append(item)
        print(
            f"经验库梯度 {item['task_id']} loss={loss:.4f} "
            + " ".join(f"{module}={norms[module]:.4g}" for module in MODULES),
            flush=True,
        )
        del deltas, item_packed
    gradient_seconds = perf_counter() - gradient_started
    for name, parameter in detector.named_parameters():
        parameter.requires_grad_(flags[name])
    final_difference, final_owner = _state_difference(detector, snapshot)
    if final_difference != 0.0 or _state_sha256(_state_snapshot(detector)) != theta_hash:
        raise AssertionError(f"建库结束时 θ 已变化：{final_owner} {final_difference}")
    if len(succeeded) < 2:
        raise ValueError("有效梯度任务少于 2 条，不能拟合 PCA")

    raw = {name: torch.stack([item["keys"][name] for item in succeeded]) for name in _KEY_NAMES}
    families_for_cluster = [item["episode"]["family_id"] for item in succeeded]
    fitted = _fit_keys(raw, families_for_cluster, int(config["clusters_per_module"]), int(config["cluster_seed"]))
    for position, item in enumerate(succeeded):
        item["family_id"] = item["episode"]["family_id"]
        item["cluster"] = {}
        for name in _KEY_NAMES:
            cluster_id = int(fitted[name]["clustering"]["assignment"][position])
            if cluster_id < 0 or not bool(fitted[name]["clustering"]["valid"][cluster_id]):
                raise AssertionError(f"{item['task_id']} 的 {name} 簇无效：{cluster_id}")
            item["cluster"][name] = cluster_id
        item["group"] = int(groups[item["episode"]["family_id"]])
    family_norms = {module: [] for module in MODULES}
    by_family: dict[str, list] = {}
    for item in succeeded:
        by_family.setdefault(item["episode"]["family_id"], []).append(item)
    for family, members in by_family.items():
        for module in MODULES:
            family_norms[module].append(torch.tensor([member["norms"][module] for member in members]))
    from training.retrieval_memory import median_anchor

    anchors = {module: median_anchor(family_norms[module]).float() for module in MODULES}
    keep_ratio = {module: 1.0 for module in MODULES}
    sketch_started = perf_counter()
    sketches = {module: {ratio: [] for ratio in config["trim_ratios"]} for module in MODULES}
    sketch_indices = {
        module: layered_sketch_indices(layouts[module], int(config["trim_sketch_dim"]), int(config["cluster_seed"]) + 17 + offset)
        for offset, module in enumerate(MODULES)
    }
    def load_packed(task_id: str) -> dict[str, torch.Tensor]:
        payload = load_checkpoint(delta_dir / f"{task_id.replace(':', '_')}.pt")
        if payload["theta_sha256"] != theta_hash:
            raise AssertionError("任务梯度不是当前 θ 的快照")
        return payload["delta"]

    for item in succeeded:
        loaded_delta = load_packed(item["task_id"])
        for module in MODULES:
            clipped = clip_packed(loaded_delta[module], anchors[module])
            for ratio in config["trim_ratios"]:
                sketches[module][float(ratio)].append((
                    item["cluster"][KEY_FOR_MODULE[module]], item["group"],
                    sketch_packed(clipped, layouts[module], sketch_indices[module], float(ratio)),
                ))
            del clipped
        del loaded_delta
        gc.collect()
    from training.retrieval_memory import select_keep_ratio

    calibration = {}
    next_keep = {}
    for module in MODULES:
        bins_by_ratio = {}
        for ratio, rows in sketches[module].items():
            bins: dict[tuple, list] = {}
            for cluster, group, vector in rows:
                bins.setdefault((cluster, group), []).append(vector)
            bins_by_ratio[float(ratio)] = [torch.stack(members) for members in bins.values()]
        choice = select_keep_ratio(
            bins_by_ratio, int(config["trim_calibration_min_members"]), int(config["trim_splits"]), 1.0,
        )
        calibration[module] = {
            "keep_ratio": float(choice["keep_ratio"]),
            "calibrated": bool(choice["calibrated"]),
            "scores": [
                {
                    "ratio": float(row["ratio"]), "bins": int(row["bins"]),
                    "score": -1.0 if row["score"] is None else float(row["score"]),
                    "score_missing": row["score"] is None,
                }
                for row in choice["scores"]
            ],
        }
        next_keep[module] = float(choice["keep_ratio"])
    sketch_seconds = perf_counter() - sketch_started
    for item in succeeded:
        item["theta_sha256"] = theta_hash
        item["theta_hash"] = theta_hash
        item["theta_version"] = theta_hash
    cards = assemble_group_cards(
        succeeded, load_packed, layouts, anchors, keep_ratio, int(config["min_members_for_ties"]),
    )

    appearance_means = {}
    for item in succeeded:
        appearance_means.setdefault(item["episode"]["family_id"], []).append(item["keys"]["Appearance"])
    family_vectors = {family: torch.stack(rows).mean(dim=0) for family, rows in appearance_means.items()}
    near_rows = []
    ordered = sorted(family_vectors)
    threshold = float(config["near_duplicate_cosine"])
    for left_index, left in enumerate(ordered):
        left_vector = family_vectors[left]
        left_vector = left_vector / left_vector.norm().clamp_min(1e-12)
        for right in ordered[left_index + 1:]:
            right_vector = family_vectors[right]
            right_vector = right_vector / right_vector.norm().clamp_min(1e-12)
            cosine = float(torch.dot(left_vector, right_vector))
            if cosine >= threshold:
                near_rows.append({
                    "family_a": left, "family_b": right, "cosine": f"{cosine:.6f}",
                    "same_experience_group": int(groups[left] == groups[right]), "merged": 0,
                })

    assignments = []
    for item in succeeded:
        assignments.append({
            "task_id": item["task_id"], "family_id": item["episode"]["family_id"], "group": item["group"],
            "cluster": item["cluster"], "seed": item["seed"], "loss": item["loss"], "norms": item["norms"],
            "delta_file": item["delta_file"], "support_group_ids": item["support_group_ids"],
            "reference_group_ids": item["reference_group_ids"], "support_image_ids": item["support_image_ids"],
            "reference_image_ids": item["reference_image_ids"],
        })
    manifest = json.loads(json.dumps({
        "fold_id": int(scope["fold_id"]),
        "theta0_file_sha256": theta0_file_hash,
        "theta_sha256": theta_hash,
        "preprocess_sha256": preprocess_hash,
        "split_sha256": scope["fold_sha256"],
        "signature_version": config["signature_version"],
        "requested_tasks_per_family": requested_tasks,
        "families": task_report,
        "experience_groups": {family: int(groups[family]) for family in families},
        "skipped": signature_skipped + skipped,
        "assignments": assignments,
        "query_exposure": exposure,
        "views_per_group": settings["views_per_group"],
        "inner_lr": settings["inner_lr"],
        **inner_lr_manifest_fields(settings),
        "same_object_families": config["same_object_families"],
        "config_sha256": file_sha256(named_config_path("retrieval_fomaml.yaml")),
        "fomaml_config_sha256": file_sha256(named_config_path("fomaml.yaml")),
        "data_profile": scope.get("data_profile"),
        "episode_source": scope.get("episode_source"),
        "profile_dataset_ids": scope.get("profile_dataset_ids"),
        "replaces_three_dataset_protocol": bool(scope.get("replaces_three_dataset_protocol")),
    }))
    bank = {
        "manifest": manifest,
        "near_duplicates": near_rows,
        "theta0_sha256": theta0_file_hash,
        "theta_sha256": theta_hash,
        "preprocess_sha256": preprocess_hash,
        "split_sha256": scope["fold_sha256"],
        "experience_groups": int(config["experience_groups"]),
        "clusters_requested": int(config["clusters_per_module"]),
        "top_mass_threshold": float(config["top_mass_threshold"]),
        "max_top_k": int(config["max_top_k"]),
        "temperature": float(config["temperature_after_zscore"]),
        "groups": {family: int(groups[family]) for family in families},
        "layouts": layouts,
        "pca": {name: fitted[name]["pca"] for name in _KEY_NAMES},
        "centers": {name: fitted[name]["clustering"]["centers"] for name in _KEY_NAMES},
        "center_valid": {name: fitted[name]["clustering"]["valid"] for name in _KEY_NAMES},
        "anchors": {module: float(anchors[module]) for module in MODULES},
        "keep_ratio": keep_ratio,
        "next_keep_ratio": next_keep,
        "trim_calibration": calibration,
        "signatures": {name: raw[name] for name in _KEY_NAMES},
        "assignments": assignments,
        "cards_memory": cards,
    }
    # save 会清空内存 unit 以省内存；落盘前先记下检索快照，再与重新加载结果对照。
    probe = succeeded[0]
    before_by_module = {
        module: retrieve_module_experience(
            bank, module, probe["keys"][KEY_FOR_MODULE[module]], probe["group"],
        )
        for module in MODULES
    }
    save_experience_bank(run_dir, bank)
    bank["cards_memory"] = {}
    loaded = load_experience_bank(run_dir, theta_hash)
    reload_checks = []
    for module in MODULES:
        before = before_by_module[module]
        after = retrieve_module_experience(loaded, module, probe["keys"][KEY_FOR_MODULE[module]], probe["group"])
        if before["valid"] != after["valid"]:
            raise AssertionError(f"重新加载后 {module} 的检索有效性不一致")
        if probe["group"] in before["source_groups"] or probe["group"] in after["source_groups"]:
            raise AssertionError(f"{module} 检索读到了任务自己的经验组")
        cosine = 1.0
        if before["valid"]:
            cosine = float(F.cosine_similarity(before["vector"], after["vector"], dim=0, eps=1e-12))
            if cosine < 0.999:
                raise AssertionError(f"{module} 重新加载后的检索方向不一致：{cosine}")
        external = retrieve_module_experience(loaded, module, probe["keys"][KEY_FOR_MODULE[module]], None)
        reload_checks.append({
            "module": module, "excluded_valid": bool(after["valid"]), "external_valid": bool(external["valid"]),
            "cosine": cosine, "source_groups": after["source_groups"],
        })
        del before, after, external
    forbidden = set(scope["oof_target_family_ids"]) | set(scope["meta_val_family_ids"]) | set(scope["outer_test_family_ids"])
    if set(loaded["groups"]) & forbidden:
        raise AssertionError("落盘后的经验库含有目标族")
    for row in loaded["card_index"].values():
        if set(row["families"]) & forbidden:
            raise AssertionError(f"卡片含有目标族：{row['families']}")
    member_counts = {}
    format_counts = {}
    valid_cards = 0
    for row in loaded["card_index"].values():
        member_counts[str(row["member_count"])] = member_counts.get(str(row["member_count"]), 0) + 1
        format_counts[row["format"]] = format_counts.get(row["format"], 0) + 1
        valid_cards += int(bool(row["valid"]))
    _discard_task_deltas(run_dir)
    elapsed = perf_counter() - started
    report = {
        "run_dir": str(run_dir),
        "fold_id": int(scope["fold_id"]),
        "families": len(families),
        "requested_tasks_per_family": requested_tasks,
        "task_report": task_report,
        "succeeded": len(succeeded),
        "skipped": signature_skipped + skipped,
        "cards": len(loaded["card_index"]),
        "valid_cards": valid_cards,
        "member_counts": member_counts,
        "formats": format_counts,
        "anchors": {module: float(anchors[module]) for module in MODULES},
        "clusters_used": {name: int(fitted[name]["clustering"]["clusters_used"]) for name in _KEY_NAMES},
        "clusters_requested": int(config["clusters_per_module"]),
        "signature_dims": {name: int(raw[name].shape[1]) for name in _KEY_NAMES},
        "theta_max_abs_diff": final_difference,
        "theta_sha256": theta_hash,
        "theta0_file_sha256": theta0_file_hash,
        "query_exposure": exposure,
        "reload_checks": reload_checks,
        "near_duplicates": len(near_rows),
        "feature_seconds": feature_seconds,
        "gradient_seconds": gradient_seconds,
        "sketch_seconds": sketch_seconds,
        "seconds": elapsed,
        "disk_bytes": _directory_bytes(run_dir),
        "next_keep_ratio": next_keep,
        "code_sha256": file_sha256(project_root() / "training" / "experience_bank.py"),
    }
    write_json(run_dir / "bank_report.json", report)
    print(json.dumps({key: report[key] for key in ("succeeded", "cards", "valid_cards", "formats", "seconds", "disk_bytes", "theta_max_abs_diff", "query_exposure")}, ensure_ascii=False), flush=True)
    return report


def allowed_source_families(episode: dict) -> set[str]:
    """来源组掩码里的可用族。当前任务自己的族不参与借用。"""
    action_ids = episode.get("source_available_action_ids")
    if not isinstance(action_ids, list) or not action_ids:
        raise ValueError(f"任务缺少可用源掩码：{episode.get('episode_id')}")
    allowed = {str(family) for family in action_ids if family != "Null"}
    family = episode.get("family_id")
    if family in allowed:
        allowed.remove(family)
    return allowed


def excluded_experience_group(bank: dict, family_id: str) -> int | None:
    """源族排除自己的经验组。库外目标没有经验组，可以使用全部来源组。"""
    if family_id in bank["groups"]:
        return int(bank["groups"][family_id])
    return None


class GammaController:
    """条件化迁移决策网络包装（选卡 + Null + ρ）。保留类名以兼容训练循环调用点。"""

    def __init__(self, config: dict | None = None):
        from training.transfer_policy import TransferPolicyController

        self.policy = TransferPolicyController(config)
        self.rho_max = self.policy.rho_max

    def state_dict(self) -> dict:
        from training.transfer_policy import POLICY_KIND

        payload = self.policy.state_dict()
        return {
            "policy_state": payload,
            "policy_optimizer_state": payload.get("policy_optimizer_state") or payload.get("optimizer"),
            "kind": POLICY_KIND,
            "modules": list(MODULES),
        }

    def load_state_dict(self, payload: dict) -> None:
        from training.transfer_policy import POLICY_KIND

        kind = payload.get("kind")
        if kind != POLICY_KIND or "policy_state" not in payload:
            raise ValueError(f"决策权重须为 {POLICY_KIND}，请用五模块条件化协议重新训练")
        if payload.get("modules") and list(payload["modules"]) != list(MODULES):
            raise ValueError(
                f"决策网络模块是 {payload['modules']}，当前协议需要 {list(MODULES)}"
            )
        self.policy.load_state_dict(payload["policy_state"])
        self.rho_max = self.policy.rho_max


class InnerLRController:
    """各模块各自的 Support 学习率；与迁移强度 ρ 分开。"""

    def __init__(self, config: dict | None = None):
        from training.retrieval_memory import load_retrieval_config

        config = load_retrieval_config() if config is None else config
        if config.get("inner_lr_optimizer", "Adam") != "Adam":
            raise ValueError("内循环学习率目前只使用 Adam")
        self.minimum = float(config["inner_lr_min"])
        self.maximum = float(config["inner_lr_max"])
        initial = float(config["inner_lr_init"])
        if not self.minimum < initial < self.maximum:
            raise ValueError("内循环学习率初值必须落在上下界之间")
        raw_value = float(torch.log(torch.expm1(torch.tensor(initial))))
        self.raw = {
            module: torch.tensor(raw_value, dtype=torch.float32, requires_grad=True) for module in MODULES
        }
        self.optimizer = torch.optim.Adam(list(self.raw.values()), lr=float(config["inner_lr_meta_lr"]))

    def effective(self, module: str) -> torch.Tensor:
        return F.softplus(self.raw[module]).clamp(self.minimum, self.maximum)

    def current(self) -> tuple[dict[str, float], dict[str, str]]:
        values, hits = {}, {}
        for module in MODULES:
            positive = float(F.softplus(self.raw[module].detach()))
            if positive < self.minimum:
                hits[module] = "min"
            elif positive > self.maximum:
                hits[module] = "max"
            else:
                hits[module] = ""
            values[module] = float(min(self.maximum, max(self.minimum, positive)))
        return values, hits

    def state_dict(self) -> dict:
        values, hits = self.current()
        return {
            "raw": {module: self.raw[module].detach().cpu().clone() for module in MODULES},
            "optimizer": self.optimizer.state_dict(),
            "values": values,
            "bound_hits": hits,
        }

    def load_state_dict(self, payload: dict) -> None:
        with torch.no_grad():
            for module in MODULES:
                self.raw[module].copy_(payload["raw"][module])
        self.optimizer.load_state_dict(payload["optimizer"])


def plan_module_borrow(
    bank: dict, module: str, key: torch.Tensor, excluded_group: int | None,
    allowed_families: set[str], controller: GammaController, allow_borrow: bool,
    *, support_step_l2: float | None = None, projected_key: torch.Tensor | None = None,
    upstream_state: torch.Tensor | None = None, prior_plans: dict[str, dict] | None = None,
) -> dict:
    """相似度只召回；MLP 结合历史效用选卡；硬 z 与 ρ 分离。"""
    from training.retrieval_memory import project_l2
    from training.transfer_policy import (
        decide_inject, encode_upstream_state, hard_borrow_gate, mix_cards_from_logits,
        mix_selected_key_embed, pad_candidates,
    )

    retrieved = retrieve_module_experience(
        bank, module, key, excluded_group, allowed_families=allowed_families,
    )
    anchor = torch.tensor(float(bank["anchors"][module]), dtype=torch.float32)
    width = int(bank["layouts"][module][-1]["offset"] + bank["layouts"][module][-1]["numel"]) if bank["layouts"][module] else 0
    candidates = list(retrieved.get("candidates") or [])
    vectors = [row["vector"] for row in candidates]
    sims = [float(row["sim"]) for row in candidates]
    key_name = KEY_FOR_MODULE[module]
    key_dim = int(controller.policy.key_dim)
    task_key = projected_key
    if task_key is None:
        task_key = project_l2(key.detach().float().cpu(), bank["pca"][key_name])
    centers = bank["centers"][key_name]
    key_diffs = []
    candidate_keys = []
    candidate_utilities = []
    candidate_meta = []
    for row in candidates:
        cluster_id = int(row.get("cluster_id", -1))
        util_info = _candidate_utility_stats(bank, module, row)
        candidate_utilities.append(util_info["features"])
        candidate_meta.append({
            "cluster_id": cluster_id,
            "card_ids": list(row.get("card_ids") or []),
            "groups": list(row.get("groups") or []),
            "sim": float(row.get("sim") or 0.0),
            "utility": dict(util_info["stats"]),
            "utility_features": [float(value) for value in util_info["features"].tolist()],
        })
        if 0 <= cluster_id < int(centers.shape[0]) and bool(bank["center_valid"][key_name][cluster_id]):
            center = centers[cluster_id].detach().float().reshape(-1)
            candidate_keys.append(center)
            denom = float(torch.linalg.vector_norm(center) * torch.linalg.vector_norm(task_key))
            cosine = 0.0 if denom < 1e-12 else float(torch.dot(center, task_key.reshape(-1)[: center.numel()]) / denom)
            key_diffs.append(1.0 - cosine)
        else:
            candidate_keys.append(torch.zeros(key_dim))
            key_diffs.append(1.0 - float(row.get("sim") or 0.0))
    units, sim_tensor, norm_tensor, mask, diff_tensor, key_tensor, util_tensor = pad_candidates(
        vectors, sims, width, controller.policy.max_candidates,
        key_diffs=key_diffs, candidate_keys=candidate_keys,
        candidate_utilities=candidate_utilities, key_dim=key_dim,
    )
    if upstream_state is None:
        upstream_state = encode_upstream_state(prior_plans or {}, module, key_dim=key_dim)
    card_logits, null_logit, rho_raw = controller.policy.policy(
        task_key, sim_tensor, norm_tensor, diff_tensor, util_tensor, module, mask, upstream_state,
    )
    mix, card_weights = mix_cards_from_logits(
        card_logits, units, mask, temperature=float(controller.policy.card_temperature),
    )
    selected_key_embed = mix_selected_key_embed(card_weights, key_tensor, mask)
    threshold = float(controller.policy.null_eval_threshold)
    # a_m=(z,c,ρ)；硬门控 + STE，z=0 时 named 增量严格为 0。
    borrow_gate, z_hard_t, null_weight = hard_borrow_gate(null_logit, threshold=threshold)
    z_hard = float(z_hard_t.detach())
    null_probability = float(null_weight.detach())
    rho = controller.policy.effective_rho(rho_raw, module)
    step = 1.0 if support_step_l2 is None else float(support_step_l2)
    delta_vector = borrow_gate * rho * float(step) * mix
    # 候选增量（不含 z）：供 z=0 时 add-one 反事实；正式 named 仍严格乘 z_hard。
    candidate_delta = rho.detach() * float(step) * mix.detach()
    applied_l2 = float(torch.linalg.vector_norm(z_hard_t.detach() * candidate_delta))
    inject = decide_inject(
        allow_borrow=bool(allow_borrow),
        retrieved_valid=bool(retrieved["valid"]),
        z_hard=z_hard,
        applied_l2=applied_l2,
        step=float(step),
    )
    candidate_named = unpack_named(candidate_delta, bank["layouts"][module]) if float(torch.linalg.vector_norm(candidate_delta)) > 1e-8 else {}
    named = unpack_named(z_hard_t.detach() * candidate_delta, bank["layouts"][module]) if inject else {}
    card_weight_list = [float(value) for value in card_weights.detach().reshape(-1).tolist()]
    effective_card_weights = [z_hard * weight for weight in card_weight_list]
    reject = "" if retrieved["valid"] else "empty"
    selection_rows = []
    for offset, meta in enumerate(candidate_meta):
        weight = float(card_weights[offset].detach()) if offset < int(card_weights.numel()) else 0.0
        if weight < 1e-6 and offset >= len(candidates):
            continue
        selection_rows.append({
            **meta,
            "weight": weight,
            "why": (
                f"sim={meta['sim']:.3f}, util_mean={meta['utility_features'][0]:.3f}, "
                f"util_std={meta['utility_features'][1]:.3f}, count={meta['utility'].get('count', 0)}, "
                f"neg_rate={meta['utility_features'][3]:.3f}"
            ),
        })
    selection_rows.sort(key=lambda row: row["weight"], reverse=True)
    return {
        "module": module,
        "gamma": rho,
        "rho": rho,
        "rho_raw": rho_raw,
        "logits": card_logits,
        "card_logits": card_logits,
        "null_logit": null_logit,
        "card_weights": card_weights,
        "card_weight_list": card_weight_list,
        "effective_card_weights": effective_card_weights,
        "null_weight": null_weight,
        "null_probability": null_probability,
        "borrow_gate": borrow_gate,
        "z_hard": float(z_hard),
        "null_decision": "null" if not inject else "borrow",
        "selected_key_embed": selected_key_embed,
        "selection": selection_rows,
        "candidate_meta": candidate_meta,
        "inject": inject,
        "named": named,
        "candidate_named": candidate_named,
        "vector": mix,
        "units": units.detach().clone(),
        "candidate_mask": mask.detach().clone(),
        "delta_vector": delta_vector,
        "upstream_state": upstream_state.detach().clone(),
        "anchor": anchor,
        "sim_max": float(retrieved["sim_max"]),
        "utility_reject": reject,
        "concentration": float(retrieved["concentration"]),
        "valid": bool(retrieved["valid"]),
        "source_groups": list(retrieved["source_groups"]),
        "cluster_ids": list(retrieved["cluster_ids"]),
        "candidates": candidates,
        "candidate_card_ids": [cid for row in candidates for cid in row.get("card_ids") or []],
        "excluded_group": None if excluded_group is None else int(excluded_group),
        "support_step_l2": step,
        "candidate_borrow_l2": applied_l2,
        "applied_borrow_l2": applied_l2 if inject else 0.0,
        "borrow_over_step": None if step <= 0 else (applied_l2 / step if inject else 0.0),
    }


def plan_modules_in_graph_order(
    bank: dict, keys: dict, excluded_group: int | None, allowed_families: set[str],
    controller: GammaController, allow_borrow: bool, step_l2: dict[str, float],
) -> list[dict]:
    """按 Backbone→Where→RoiFeature→Class/Box 顺序决策，下游读取上游卡片签名嵌入。"""
    from training.transfer_policy import encode_upstream_state

    decided: dict[str, dict] = {}
    plans = []
    key_dim = int(controller.policy.key_dim)
    for module in MODULES:
        upstream = encode_upstream_state(decided, module, key_dim=key_dim)
        plan = plan_module_borrow(
            bank, module, keys[KEY_FOR_MODULE[module]], excluded_group, allowed_families,
            controller, allow_borrow, support_step_l2=float(step_l2.get(module, 0.0)),
            upstream_state=upstream, prior_plans=decided,
        )
        decided[module] = plan
        plans.append(plan)
    return plans


def force_close_plan(plan: dict) -> None:
    """训练探索：正式注入关掉；gate 前向为 0，反向经 STE 到 null_logit，供 Add-One 重开。"""
    plan["inject"] = False
    plan["named"] = {}
    plan["z_hard"] = 0.0
    plan["null_decision"] = "null"
    plan["applied_borrow_l2"] = 0.0
    plan["borrow_over_step"] = 0.0
    weights = list(plan.get("card_weight_list") or [])
    plan["effective_card_weights"] = [0.0 for _ in weights]
    gate = plan.get("borrow_gate")
    null_w = plan.get("null_weight")
    if torch.is_tensor(null_w) and bool(null_w.requires_grad):
        z_soft = 1.0 - null_w
        plan["borrow_gate"] = z_soft - z_soft.detach()
    elif torch.is_tensor(gate) and bool(gate.requires_grad):
        plan["borrow_gate"] = gate * 0.0
    plan["force_closed"] = True


def apply_force_close_and_replan(
    plans: list[dict],
    close_modules: list[str],
    *,
    bank: dict,
    keys: dict,
    excluded_group: int | None,
    allowed_families: set[str],
    controller: GammaController,
    allow_borrow: bool,
    step_l2: dict[str, float],
) -> list[str]:
    """关闭指定模块后，仅重规划依赖它们的下游，使 upstream z 可见。"""
    from training.transfer_policy import UPSTREAM_OF, encode_upstream_state

    if not close_modules:
        return []
    closed = set()
    by_module = {plan["module"]: plan for plan in plans}
    for module in close_modules:
        plan = by_module.get(module)
        if plan is None or not plan.get("inject"):
            continue
        if not (plan.get("candidate_named") or plan.get("valid")):
            continue
        force_close_plan(plan)
        closed.add(module)
    if not closed:
        return []
    key_dim = int(controller.policy.key_dim)
    decided = {plan["module"]: plan for plan in plans}
    for module in MODULES:
        upstream_names = UPSTREAM_OF.get(module) or ()
        if not any(name in closed for name in upstream_names):
            continue
        if module in closed:
            continue
        upstream = encode_upstream_state(decided, module, key_dim=key_dim)
        plan = plan_module_borrow(
            bank, module, keys[KEY_FOR_MODULE[module]], excluded_group, allowed_families,
            controller, allow_borrow, support_step_l2=float(step_l2.get(module, 0.0)),
            upstream_state=upstream, prior_plans=decided,
        )
        decided[module] = plan
        for index, old in enumerate(plans):
            if old["module"] == module:
                plans[index] = plan
                break
    return sorted(closed)


def support_packed_grads(detector, records, device, settings: dict, layouts: dict) -> dict[str, torch.Tensor]:
    """在共享 θ 上求 Support 梯度，不更新参数。决定借用之后，适应步仍在注入后的点上重算梯度。"""
    from training.theta0 import _mean_module_grads

    _loss, grads = _mean_module_grads(detector, records, device, settings)
    packed = {module: _pack_module_grad(detector, grads, module, layouts[module]) for module in MODULES}
    detector.zero_grad(set_to_none=True)
    detector.eval()
    return packed


def query_total_packed_grads(
    detector,
    records,
    device,
    settings: dict,
    layouts: dict,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Query 总损失与打包梯度；ψ 用模块 LOO/Add-One，不再做职责分项反传。"""
    from training.fomaml_roi import live_detection_loss
    from training.theta0 import InsufficientEpisodeAnnotations

    import gc

    trainable = [parameter for parameter in detector.parameters() if parameter.requires_grad]
    total_accum = [torch.zeros_like(parameter) for parameter in trainable]
    total_loss = None
    used = 0
    detector.train()
    for record in records:
        total = live_detection_loss(
            detector, record, device, settings["loss"], settings["where_loss_weight"], settings["ignore_iou"],
        )
        total_grads = torch.autograd.grad(total, trainable, allow_unused=True)
        for slot, grad in zip(total_accum, total_grads):
            if grad is not None:
                slot.add_(grad.detach())
        total_loss = total.detach() if total_loss is None else total_loss + total.detach()
        used += 1
        del total, total_grads
        gc.collect()
    if used == 0 or total_loss is None:
        raise InsufficientEpisodeAnnotations("Support 或 Query 没有可用的检测监督")
    scale = float(used)
    query_grads = [grad / scale for grad in total_accum]
    packed_query = {
        module: _pack_module_grad(detector, query_grads, module, layouts[module]) for module in MODULES
    }
    detector.zero_grad(set_to_none=True)
    detector.eval()
    del total_accum
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return total_loss / scale, query_grads, packed_query


def gate_borrow_by_support(plans: list[dict], module_grads: dict[str, torch.Tensor]) -> None:
    """只记录 Support 方向诊断，不再硬拒绝注入。迁移决策交给 MLP + Null。"""
    for plan in plans:
        module = plan["module"]
        grad = module_grads[module].detach().float().reshape(-1)
        grad_norm = float(torch.linalg.vector_norm(grad))
        plan["support_grad_l2"] = grad_norm
        mix = plan.get("vector")
        scale = float(plan.get("applied_borrow_l2") or plan.get("candidate_borrow_l2") or 0.0)
        if mix is None:
            plan["support_dot"] = 0.0
            continue
        mix_flat = mix.detach().float().reshape(-1)
        if int(mix_flat.numel()) != int(grad.numel()):
            raise AssertionError(f"{module} 的借用向量和 Support 梯度维度不一致")
        mix_norm = float(torch.linalg.vector_norm(mix_flat))
        direction = mix_flat if mix_norm < 1e-12 else mix_flat / mix_norm
        plan["support_dot"] = float(torch.dot(grad, direction)) * scale


def attach_first_order_ratio(
    plans: list[dict],
    support_grads: dict[str, torch.Tensor],
    query_grads: dict[str, torch.Tensor],
    alphas: dict[str, float],
) -> None:
    """在共享 θ 上记 R = ⟨g_Q, ρ ‖α g_S‖ û⟩ / ⟨g_Q, α g_S⟩，以及 ‖ρ B‖、‖α g_S‖。"""
    for plan in plans:
        module = plan["module"]
        g_q = query_grads[module].detach().float().reshape(-1)
        g_s = support_grads[module].detach().float().reshape(-1)
        alpha = float(alphas[module])
        mix = plan.get("vector")
        step_l2 = abs(alpha) * float(torch.linalg.vector_norm(g_s))
        scale = float(plan["rho"].detach()) * step_l2
        step_dot = float(torch.dot(g_q, g_s)) * alpha
        if mix is None:
            mix_norm = 0.0
            borrow_l2 = 0.0
            borrow_dot = 0.0
        else:
            mix_flat = mix.detach().float().reshape(-1)
            if int(mix_flat.numel()) != int(g_q.numel()):
                raise AssertionError(f"{module} 的借用向量和 Query 梯度维度不一致")
            mix_norm = float(torch.linalg.vector_norm(mix_flat))
            direction = mix_flat if mix_norm < 1e-12 else mix_flat / mix_norm
            borrow_l2 = abs(scale)
            borrow_dot = float(torch.dot(g_q, direction)) * scale
        plan["retrieved_mix_l2"] = mix_norm
        plan["retrieved_b_l2"] = mix_norm
        plan["candidate_borrow_l2"] = borrow_l2
        plan["alpha_grad_l2"] = step_l2
        plan["query_dot_borrow"] = borrow_dot
        plan["query_dot_step"] = step_dot
        plan["first_order_ratio"] = None if abs(step_dot) < 1e-12 else borrow_dot / step_dot


def gamma_runtime_values(controller: GammaController) -> dict:
    """条件化决策网络运行时读数：ρ 上限与 Null 头偏置。"""
    null_bias = float(controller.policy.policy.null_head.bias.detach().reshape(-1)[0])
    return {
        "rho_max": float(controller.rho_max),
        "null_bias": null_bias,
        "kind": str(controller.state_dict().get("kind")),
        "force_close_prob": float(getattr(controller.policy, "force_close_prob", 0.0)),
        "null_eval_threshold": float(controller.policy.null_eval_threshold),
    }


def bank_card_attenuation(bank: dict) -> dict:
    """卡片侧衰减读数：合并后单位方向范数恒为 1，幅度由锚和成员数体现。成员 Δ 已删则标未测。"""
    from collections import Counter

    by_module = {}
    for module in MODULES:
        rows = [row for row in bank["card_index"].values() if row["module"] == module and row["valid"]]
        members = Counter(int(row["member_count"]) for row in rows)
        by_module[module] = {
            "valid_cards": len(rows),
            "member_counts": {str(key): int(value) for key, value in sorted(members.items())},
            "singleton_cards": int(members.get(1, 0)),
            "mean_coherence": None if not rows else float(sum(float(row["coherence"]) for row in rows) / len(rows)),
            "anchor": float(bank["anchors"][module]),
            "unit_l2": 1.0,
            "member_delta_l2": None,
            "member_delta_l2_note": "task_deltas 拼卡后已删除，成员 Δ 范数未测",
        }
    return by_module


def module_utility_report(plans: list[dict], alphas: dict[str, float]) -> dict:
    """留出和训练日志共用的模块读数。拒绝借用的块保留候选幅度，注入幅度记 0。"""
    report = {}
    for plan in plans:
        module = plan["module"]
        grad_l2 = float(plan.get("support_grad_l2") or 0.0)
        step_l2 = float(plan.get("alpha_grad_l2") if plan.get("alpha_grad_l2") is not None else alphas[module] * grad_l2)
        applied = float(plan.get("applied_borrow_l2") or 0.0) if plan["inject"] else 0.0
        top = (plan.get("selection") or [{}])[:3]
        card_weights = list(plan.get("card_weight_list") or [])
        if not card_weights and plan.get("card_weights") is not None:
            weights = plan["card_weights"]
            card_weights = [float(value) for value in weights.detach().reshape(-1).tolist()]
        null_p = float(plan.get("null_probability") or 0.0)
        effective = list(plan.get("effective_card_weights") or [(1.0 - null_p) * weight for weight in card_weights])
        report[module] = {
            "gamma": float(plan["gamma"].detach()),
            "rho": float(plan["rho"].detach()) if plan.get("rho") is not None else float(plan["gamma"].detach()),
            "null_probability": null_p,
            "null_decision": plan.get("null_decision") or ("borrow" if plan["inject"] else "null"),
            "card_weights": card_weights,
            "effective_card_weights": effective,
            "sim_max": float(plan["sim_max"]),
            "inject": bool(plan["inject"]),
            "utility_reject": plan.get("utility_reject") or "",
            "selected_cards": top,
            "support_dot": None if "support_dot" not in plan else float(plan["support_dot"]),
            "support_grad_l2": grad_l2,
            "alpha": float(alphas[module]),
            "alpha_grad_l2": step_l2,
            "retrieved_mix_l2": None if "retrieved_mix_l2" not in plan else float(plan["retrieved_mix_l2"]),
            "retrieved_b_l2": None if "retrieved_b_l2" not in plan else float(plan["retrieved_b_l2"]),
            "candidate_borrow_l2": float(plan.get("candidate_borrow_l2") or 0.0),
            "applied_borrow_l2": applied,
            "borrow_over_step": (
                plan["borrow_over_step"] if "borrow_over_step" in plan
                else (None if step_l2 <= 0.0 else applied / step_l2)
            ),
            "query_dot_borrow": None if "query_dot_borrow" not in plan else float(plan["query_dot_borrow"]),
            "query_dot_step": None if "query_dot_step" not in plan else float(plan["query_dot_step"]),
            "first_order_ratio": None if "first_order_ratio" not in plan else plan["first_order_ratio"],
        }
    return report


def module_utility_summary(episodes: dict) -> dict:
    """按块汇总留出 AP50 差。这是读数，不是全产品放行门槛。"""
    summary = {}
    for module in MODULES:
        injected, refused = [], []
        for row in episodes.values():
            info = row["modules"][module]
            bucket = injected if info["inject"] else refused
            bucket.append(float(row["ap50_delta"]))
        summary[module] = {
            "injected_episodes": len(injected),
            "refused_episodes": len(refused),
            "mean_ap50_delta_when_injected": None if not injected else float(sum(injected) / len(injected)),
            "mean_ap50_delta_when_refused": None if not refused else float(sum(refused) / len(refused)),
        }
    return summary


def mean_query_detection_loss(detector, records, device, settings: dict) -> float:
    """只算 Query 检测损失均值，不反传、不打包梯度。"""
    from training.fomaml_roi import live_detection_loss
    from training.theta0 import InsufficientEpisodeAnnotations

    total = 0.0
    used = 0
    detector.train()
    for record in records:
        loss = live_detection_loss(
            detector, record, device, settings["loss"], settings["where_loss_weight"], settings["ignore_iou"],
        )
        total += float(loss.detach())
        used += 1
        del loss
    detector.zero_grad(set_to_none=True)
    detector.eval()
    if used == 0:
        raise InsufficientEpisodeAnnotations("Support 或 Query 没有可用的检测监督")
    return total / float(used)


def apply_counterfactual_policy(
    controller: GammaController,
    plans: list[dict],
    module_utilities: dict[str, float],
    *,
    ap_targets: dict | None = None,
    rho_probe: dict | None = None,
    card_probe: dict | None = None,
    task_utility: float | None = None,
) -> dict:
    """按模块条件贡献 U_m 更新 ψ；可选 ρ/卡相对排序探针。AP Head 默认不进主损失。"""
    from training.transfer_policy import (
        ap_utility_nll, card_pair_policy_loss, card_probe_policy_loss, counterfactual_policy_loss,
        plan_decision_features, rho_pair_policy_loss, rho_probe_policy_loss,
        utility_ap_sign_agreement,
    )

    by_module = {plan["module"]: plan for plan in plans}
    policy = controller.policy
    try:
        loss = counterfactual_policy_loss(
            plans,
            module_utilities,
            null_sign_refuse_weight=float(policy.null_sign_refuse_weight),
            null_sign_accept_weight=float(policy.null_sign_accept_weight),
            null_sign_margin=float(policy.null_sign_margin),
        )
    except ValueError:
        loss = None
    if rho_probe and rho_probe.get("module") in by_module:
        plan = by_module[rho_probe["module"]]
        try:
            if rho_probe.get("pairwise"):
                term = rho_pair_policy_loss(
                    plan,
                    float(rho_probe["rho_a"]),
                    float(rho_probe["utility_a"]),
                    float(rho_probe["rho_b"]),
                    float(rho_probe["utility_b"]),
                    float(rho_probe.get("eps", 0.0)),
                )
            else:
                term = rho_probe_policy_loss(
                    plan, float(rho_probe["utility"]), float(rho_probe["rho"]),
                )
            loss = term if loss is None else loss + term
        except ValueError:
            pass
    if card_probe and card_probe.get("module") in by_module:
        plan = by_module[card_probe["module"]]
        try:
            if card_probe.get("pairwise"):
                term = card_pair_policy_loss(
                    plan,
                    int(card_probe["index_a"]),
                    float(card_probe["utility_a"]),
                    int(card_probe["index_b"]),
                    float(card_probe["utility_b"]),
                    float(card_probe.get("eps", 0.0)),
                )
            else:
                term = card_probe_policy_loss(
                    plan, float(card_probe["utility"]), int(card_probe["index"]),
                )
            weight = float(policy.card_pair_loss_weight)
            if weight != 1.0:
                term = weight * term
            loss = term if loss is None else loss + term
        except ValueError:
            pass
    ap_loss = None
    prediction = None
    sign_agree = None
    # 仅当显式打开且 every_n>0 时才把 AP 写入损失。
    train_ap = (
        ap_targets is not None
        and controller.policy.ap_utility_enabled
        and int(controller.policy.ap_utility_every_n_steps) > 0
    )
    if ap_targets is not None and task_utility is not None:
        sign_agree = {
            "ap50": utility_ap_sign_agreement(
                float(task_utility),
                None if ap_targets.get("ap50") is None else float(ap_targets["ap50"]),
            ),
            "ap75": utility_ap_sign_agreement(
                float(task_utility),
                None if ap_targets.get("ap75") is None else float(ap_targets["ap75"]),
            ),
        }
    if train_ap:
        features = plan_decision_features(plans)
        prediction = controller.policy.ap_head(features)
        ap_loss = ap_utility_nll(
            prediction,
            float(ap_targets["ap50"]),
            None if ap_targets.get("ap75") is None else float(ap_targets["ap75"]),
        )
        loss = ap_loss if loss is None else loss + ap_loss
    if loss is None:
        return {
            "loss": 0.0,
            "module_utilities": {module: float(module_utilities.get(module, 0.0)) for module in MODULES},
            "dots": {module: -float(module_utilities.get(module, 0.0)) for module in MODULES},
            "gain_grads": {module: 0.0 for module in MODULES},
            "ap_loss": None,
            "ap_prediction": None,
            "sign_agreement": sign_agree,
            "rho_probe": rho_probe,
            "card_probe": card_probe,
        }
    controller.policy.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    controller.policy.optimizer.step()
    return {
        "loss": float(loss.detach()),
        "module_utilities": {module: float(module_utilities.get(module, 0.0)) for module in MODULES},
        "dots": {module: -float(module_utilities.get(module, 0.0)) for module in MODULES},
        "gain_grads": {module: 0.0 for module in MODULES},
        "ap_loss": None if ap_loss is None else float(ap_loss.detach()),
        "ap_prediction": None if prediction is None else {
            "mu_ap50": float(prediction["mu_ap50"].detach()),
            "mu_ap75": float(prediction["mu_ap75"].detach()),
            "log_sigma_ap50": float(prediction["log_sigma_ap50"].detach()),
            "log_sigma_ap75": float(prediction["log_sigma_ap75"].detach()),
        },
        "sign_agreement": sign_agree,
        "rho_probe": rho_probe,
        "card_probe": card_probe,
    }


def apply_alpha_proxy(
    controller: InnerLRController, support_grads: dict[str, torch.Tensor], query_grads: dict[str, torch.Tensor],
) -> dict:
    """dL_Q/dα = -⟨g_Q, g_S⟩。只更新四个学习率，不再跑检测器。"""
    loss = controller.raw[MODULES[0]].new_zeros(())
    dots = {}
    for module in MODULES:
        dot = torch.dot(
            query_grads[module].detach().float().reshape(-1),
            support_grads[module].detach().float().reshape(-1),
        )
        dots[module] = float(dot)
        loss = loss - controller.effective(module) * dot
    controller.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    controller.optimizer.step()
    values, hits = controller.current()
    return {"dots": dots, "values": values, "bound_hits": hits, "loss": float(loss.detach())}


def own_update_from_borrowed_init(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], borrow: dict[str, torch.Tensor]) -> dict:
    """内循环差相对借用后的初始化计算，借来的增量留在外面。"""
    from training.theta0 import split_parameter_delta

    borrowed = {}
    for name, old in before.items():
        extra = borrow.get(name)
        borrowed[name] = old if extra is None else old + extra.detach().cpu().reshape(old.shape)
    return split_parameter_delta(borrowed, after)


def _parameter_snapshot(module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def _borrow_leak(before: dict, after: dict, borrow: dict, delta: dict) -> float:
    worst = 0.0
    for name, old in before.items():
        owners = [label for label, prefixes in MODULE_PREFIXES.items() if name.startswith(prefixes)]
        learned = delta[owners[0]][name]
        extra = borrow.get(name)
        added = torch.zeros_like(old) if extra is None else extra.detach().cpu().reshape(old.shape)
        leaked = (after[name] - old - added - learned).abs().max()
        worst = max(worst, float(leaked))
    return worst


def _cache_directories(bank: dict) -> list:
    from pathlib import Path

    directories = []
    manifest = bank.get("manifest") or {}
    for item in manifest.get("feature_cache_dirs") or []:
        directories.append(Path(item))
    local = Path(bank["directory"]) / "feature_cache"
    if local not in directories:
        directories.append(local)
    return directories


def _load_cached_view(bank: dict, image_id: str, theta0_file_hash: str, extra_caches=None) -> dict | None:
    from pathlib import Path

    name = _cache_name(image_id)
    directories = [Path(item) for item in (extra_caches or [])]
    directories.extend(_cache_directories(bank))
    for directory in directories:
        path = directory / name
        if not path.is_file():
            continue
        view = load_checkpoint(path)
        if view.get("theta0_sha256") == theta0_file_hash:
            return view
    return None


def task_signature(
    adapter, episode: dict, bank: dict, detector, device, settings: dict,
    theta0_state: dict | None = None, writable_cache=None, read_only: bool = False,
    extra_caches=None,
) -> dict:
    """只用正常参照和 Support 的 θ₀ 特征。Query 图像不能进入描述子。"""
    from pathlib import Path

    query_ids = {
        image_id
        for role in ("query_defect", "query_normal")
        for unit in episode[role]
        for image_id in unit["image_ids"]
    }
    records = adapter.build_episode_records(episode, roles=("reference", "support"))
    roles = {record["split_role"] for record in records}
    if roles - {"R", "SUPPORT"}:
        raise AssertionError(f"经验检索读取了 Support 和 R 以外的角色：{sorted(roles)}")
    if any(record["image_id"] in query_ids for record in records):
        raise AssertionError(f"经验检索混入 Query 图像：{episode['episode_id']}")
    support_ids = {unit["physical_group_id"] for unit in episode["support"]}
    reference_ids = {unit["physical_group_id"] for unit in episode["reference"]}
    query_group_ids = {
        unit["physical_group_id"] for role in ("query_defect", "query_normal") for unit in episode[role]
    }
    if support_ids & reference_ids or support_ids & query_group_ids or reference_ids & query_group_ids:
        raise AssertionError(f"R、Support、Query 物理组重叠：{episode['episode_id']}")
    limited = limit_role_views(records, int(settings["views_per_group"]))
    theta0_file_hash = bank["theta0_sha256"]
    views = []
    missing = []
    for record in limited:
        if record["split_role"] == "R" and (int(record["normal_flag"]) != 1 or record["annotations"]):
            raise AssertionError(f"正常参照不是纯正常图：{record['image_id']}")
        view = _load_cached_view(bank, record["image_id"], theta0_file_hash, extra_caches)
        views.append(view)
        if view is None:
            missing.append(record)
    if missing:
        if read_only or theta0_state is None:
            raise FileNotFoundError("θ₀ 特征缓存缺少检索所需视角，不能临时改用当前共享参数")
        current = _state_snapshot(detector)
        try:
            detector.load_state_dict(theta0_state, strict=True)
            detector.eval()
            writable = None if writable_cache is None else Path(writable_cache)
            if writable is not None:
                writable.mkdir(parents=True, exist_ok=True)
            version = (bank.get("manifest") or {}).get("signature_version", "")
            for record in missing:
                view = _extract_view(detector, record, device)
                view.update({
                    "theta0_sha256": theta0_file_hash,
                    "preprocess_sha256": bank["preprocess_sha256"],
                    "signature_version": version,
                    "view_id": str(record.get("view_id") or ""),
                })
                if writable is not None:
                    save_checkpoint(writable / _cache_name(record["image_id"]), view)
                views[limited.index(record)] = view
        finally:
            detector.load_state_dict(current, strict=True)
    if any(view is None for view in views):
        raise AssertionError("经验检索仍有视角没有 θ₀ 特征")
    support_views = [view for view in views if view["split_role"] == "SUPPORT"]
    reference_views = [view for view in views if view["split_role"] == "R"]
    keys = task_keys_from_groups(group_prototypes(support_views), group_prototypes(reference_views))
    return {
        "keys": keys,
        "support_records": [record for record in limited if record["split_role"] == "SUPPORT"],
        "support_image_ids": [view["image_id"] for view in support_views],
        "reference_image_ids": [view["image_id"] for view in reference_views],
    }


def module_inner_lrs(settings: dict, controller: InnerLRController | None = None) -> tuple[dict[str, float], dict[str, str]]:
    if controller is not None:
        return controller.current()
    value = float(settings["inner_lr"])
    return {module: value for module in MODULES}, {module: "" for module in MODULES}


def inner_lr_manifest_fields(settings: dict) -> dict:
    """卡片版本要么是固定 inner_lr，要么是同一组 α 快照。两种定义不能混称同一版本。"""
    snapshot = settings.get("inner_lr_snapshot")
    if snapshot:
        return {
            "inner_lr_snapshot": {module: float(snapshot[module]) for module in MODULES},
            "inner_lr_source": "module_snapshot",
            "delta_definition": "delta = -alpha_m_snapshot * mean_support_grad; clip by A_m; PCA and clusters copied",
        }
    return {
        "inner_lr_snapshot": None,
        "inner_lr_source": "fixed_inner_lr",
        "delta_definition": "delta = -inner_lr * mean_support_grad; clip by A_m; PCA and clusters copied",
    }


def assert_compatible_inner_lr(bank: dict, snapshot: dict, *, strict: bool = True) -> None:
    """旧的固定 0.01 卡片不能和新的模块学习率快照宣称是同一版本。

    每轮开始写卡时 α 与卡片一致；轮内 α 仍会更新，最佳断点上的 α 可与该轮卡片有微小差别。
    留出加载传 strict=False，只核对同属可学习 α 体系。
    """
    manifest = bank.get("manifest") or {}
    recorded = manifest.get("inner_lr_snapshot")
    if manifest.get("inner_lr_source") != "module_snapshot" or not recorded:
        raise ValueError("固定学习率卡片不能与可学习 α 快照混用")
    if not snapshot:
        raise ValueError("缺少可学习 α 快照")
    if set(recorded) != set(MODULES) or set(snapshot) != set(MODULES):
        raise ValueError("α 快照模块不完整")
    if not strict:
        return
    for module in MODULES:
        if abs(float(recorded[module]) - float(snapshot[module])) > 1e-8:
            raise ValueError(f"{module} 的学习率快照和当前 α 不一致")


def apply_module_inner_step(detector, grads, alphas: dict[str, float], max_norm: float):
    from training.theta0 import _clip_grads

    grads = _clip_grads(grads, max_norm)
    trainable = [(name, parameter) for name, parameter in detector.named_parameters() if parameter.requires_grad]
    if len(trainable) != len(grads):
        raise AssertionError("内循环梯度数量和可训练参数不一致")
    with torch.no_grad():
        for (name, parameter), grad in zip(trainable, grads):
            owners = [label for label, prefixes in MODULE_PREFIXES.items() if name.startswith(prefixes)]
            if len(owners) != 1:
                raise AssertionError(f"参数没有唯一模块：{name}")
            parameter.add_(-float(alphas[owners[0]]) * grad)
    return grads


def retrieval_support_adapt(
    detector, support_records, device, settings: dict, alphas: dict[str, float], borrow_named: dict,
    layouts: dict | None = None,
) -> dict:
    """训练和留出共用的借用注入与 Support 一步。这里不读取 Query。"""
    from training.theta0 import _add_named_delta, _mean_module_grads, split_parameter_delta

    if borrow_named:
        _add_named_delta(detector, borrow_named)
    borrowed_init = _parameter_snapshot(detector)
    losses = []
    packed = None
    grads = None
    for _ in range(int(settings["inner_steps"])):
        loss, grads = _mean_module_grads(detector, support_records, device, settings)
        losses.append(float(loss))
        grads = apply_module_inner_step(detector, grads, alphas, float(settings["grad_clip_norm"]))
    if grads is None:
        raise AssertionError("Support 适应没有执行")
    if layouts is not None:
        packed = {module: _pack_module_grad(detector, grads, module, layouts[module]) for module in MODULES}
    after = _parameter_snapshot(detector)
    return {
        "support_losses": losses,
        "support_grads": packed,
        "module_delta": split_parameter_delta(borrowed_init, after),
        "borrowed_init": borrowed_init,
        "after": after,
    }


def counterfactual_adapt_query_loss(
    detector,
    snapshot: dict,
    support_records,
    query_records,
    device,
    settings: dict,
    alphas: dict[str, float],
    borrow_named: dict,
    *,
    rng_cpu,
    rng_cuda,
) -> float:
    """反事实臂：恢复快照 → 注入 → Support 一步 → Query 损失。不打包梯度、不留参数快照。"""
    import gc

    from training.theta0 import _add_named_delta, _mean_module_grads

    detector.load_state_dict(snapshot, strict=True)
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state(rng_cuda, device)
    if borrow_named:
        _add_named_delta(detector, borrow_named)
    ran = False
    for _ in range(int(settings["inner_steps"])):
        loss, grads = _mean_module_grads(detector, support_records, device, settings)
        apply_module_inner_step(detector, grads, alphas, float(settings["grad_clip_norm"]))
        del loss, grads
        ran = True
    if not ran:
        raise AssertionError("Support 适应没有执行")
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state(rng_cuda, device)
    value = float(mean_query_detection_loss(detector, query_records, device, settings))
    detector.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value


def _pack_module_grad(detector, grads: list[torch.Tensor], module: str, layout: list[dict]) -> torch.Tensor:
    trainable = [(name, parameter) for name, parameter in detector.named_parameters() if parameter.requires_grad]
    if len(trainable) != len(grads):
        raise AssertionError("Query 梯度和可训练参数数量不一致")
    prefixes = MODULE_PREFIXES[module]
    named = {}
    for (name, parameter), grad in zip(trainable, grads):
        if name.startswith(prefixes):
            named[name] = torch.zeros(parameter.shape) if grad is None else grad.detach().float().cpu()
    return pack_named(named, layout)


def module_leave_one_out_utilities(
    detector,
    snapshot: dict,
    plans: list[dict],
    support_records,
    query_records,
    device,
    settings: dict,
    alphas: dict[str, float],
    layouts: dict,
    loss_all: float,
    loo_modules: list[str],
    *,
    rng_cpu,
    rng_cuda,
) -> tuple[dict[str, float], dict[str, str], dict[str, float]]:
    """模块条件贡献。已注入：U_m=L_{-m}-L_all；未注入：U_m=L_all-L_{all+m}（add-one，供重开）。

    第三项 `loss_without_by_module`：该模块关闭时的 Query 损失，供 ρ/卡探针复用，避免再跑一轮适配。
    """
    del layouts  # 反事实轻路径不打包梯度
    utilities: dict[str, float] = {}
    kinds: dict[str, str] = {}
    loss_without_by_module: dict[str, float] = {}
    by_module = {plan["module"]: plan for plan in plans}

    def _adapt_and_loss(borrow: dict) -> float:
        return counterfactual_adapt_query_loss(
            detector, snapshot, support_records, query_records, device, settings, alphas, borrow,
            rng_cpu=rng_cpu, rng_cuda=rng_cuda,
        )

    for module in loo_modules:
        plan = by_module.get(module)
        if plan is None or not plan.get("valid"):
            utilities[module] = 0.0
            kinds[module] = "skip"
            continue
        if plan.get("inject"):
            borrow = {}
            for other in plans:
                if other["module"] == module or not other.get("inject"):
                    continue
                borrow.update(other["named"])
            loss_without = _adapt_and_loss(borrow)
            loss_without_by_module[module] = float(loss_without)
            utilities[module] = float(loss_without - float(loss_all))
            kinds[module] = "leave_one_out"
            continue
        # 未注入：正式臂已不含该模块，loss_all 即「无该模块」基线。
        loss_without_by_module[module] = float(loss_all)
        candidate = plan.get("candidate_named") or {}
        if not candidate:
            utilities[module] = 0.0
            kinds[module] = "closed_empty"
            continue
        borrow = {}
        for other in plans:
            if other.get("inject"):
                borrow.update(other["named"])
        borrow.update(candidate)
        loss_with = _adapt_and_loss(borrow)
        utilities[module] = float(float(loss_all) - loss_with)
        kinds[module] = "add_one"
    detector.load_state_dict(snapshot, strict=True)
    return utilities, kinds, loss_without_by_module


def module_rho_and_card_probes(
    detector,
    snapshot: dict,
    plans: list[dict],
    support_records,
    query_records,
    device,
    settings: dict,
    alphas: dict[str, float],
    layouts: dict,
    loo_modules: list[str],
    controller: GammaController,
    *,
    rng_cpu,
    rng_cuda,
    loss_without_by_module: dict[str, float] | None = None,
) -> dict:
    """对 LOO 目标模块做同任务 ρ 配对 + 单卡/卡配对；相对「无该模块」算 Query 效用。"""
    by_module = {plan["module"]: plan for plan in plans}
    baselines = dict(loss_without_by_module or {})
    rho_probe = None
    card_probe = None
    shared_eps = float(getattr(controller.policy, "_probe_noise_eps", 0.0) or 0.0)
    deadzone_rel = float(getattr(controller.policy, "rho_deadzone_rel", 0.05) or 0.0)
    # 交替：偶数步只探 ρ，奇数步只探卡，避免同一步 5 次适配。
    phase = int(getattr(controller.policy, "_probe_phase", 0) or 0)
    do_rho = True
    do_card = True
    if bool(getattr(controller.policy, "probe_alternate", True)):
        do_rho = (phase % 2 == 0)
        do_card = not do_rho
        controller.policy._probe_phase = phase + 1

    def _adapt_and_loss(borrow: dict) -> float:
        return counterfactual_adapt_query_loss(
            detector, snapshot, support_records, query_records, device, settings, alphas, borrow,
            rng_cpu=rng_cpu, rng_cuda=rng_cuda,
        )

    def _pair_eps(utility_a: float, utility_b: float) -> float:
        nonlocal shared_eps
        scale = max(abs(float(utility_a)), abs(float(utility_b)), 1e-6)
        eps = max(float(shared_eps), float(deadzone_rel) * float(scale))
        shared_eps = 0.9 * float(shared_eps) + 0.1 * float(eps)
        controller.policy._probe_noise_eps = float(shared_eps)
        return float(eps)

    def _others_borrow(module: str) -> dict:
        borrow = {}
        for other in plans:
            if other["module"] == module or not other.get("inject"):
                continue
            borrow.update(other["named"])
        return borrow

    def _rho_named(module: str, rho_value: float, step: float, mix: torch.Tensor) -> dict | None:
        probe_flat = float(rho_value) * step * mix.detach().float().reshape(-1)
        if float(torch.linalg.vector_norm(probe_flat)) <= 1e-8:
            return None
        named = dict(_others_borrow(module))
        named.update(unpack_named(probe_flat, layouts[module]))
        return named

    def _card_named(module: str, rho_card: float, step: float, unit: torch.Tensor) -> dict | None:
        card_flat = float(rho_card) * step * unit.detach().float().reshape(-1)
        if float(torch.linalg.vector_norm(card_flat)) <= 1e-8:
            return None
        named = dict(_others_borrow(module))
        named.update(unpack_named(card_flat, layouts[module]))
        return named

    for module in loo_modules:
        plan = by_module.get(module)
        if plan is None or not plan.get("valid"):
            continue
        step = float(plan.get("support_step_l2") or 0.0)
        mix = plan.get("vector")
        if step <= 0.0 or not torch.is_tensor(mix):
            continue
        if module in baselines:
            loss_without = float(baselines[module])
        else:
            loss_without = _adapt_and_loss(_others_borrow(module))
            baselines[module] = float(loss_without)
        if do_rho and rho_probe is None:
            pair = controller.policy.next_rho_pair()
            if pair is not None:
                rho_a, rho_b = float(pair[0]), float(pair[1])
                if abs(rho_a - rho_b) <= 1e-12:
                    named = _rho_named(module, rho_a, step, mix)
                    if named is not None:
                        utility = float(loss_without - _adapt_and_loss(named))
                        rho_probe = {
                            "module": module, "pairwise": False,
                            "rho": float(rho_a), "utility": utility,
                        }
                else:
                    named_a = _rho_named(module, rho_a, step, mix)
                    named_b = _rho_named(module, rho_b, step, mix)
                    if named_a is not None and named_b is not None:
                        # 同 RNG 下重复评价 ε 恒为 0，只评两档；死区用相对阈值。
                        utility_a = float(loss_without - _adapt_and_loss(named_a))
                        utility_b = float(loss_without - _adapt_and_loss(named_b))
                        eps = _pair_eps(utility_a, utility_b)
                        delta = utility_a - utility_b
                        winner = None
                        if abs(delta) > eps:
                            winner = rho_a if delta > 0.0 else rho_b
                        rho_probe = {
                            "module": module,
                            "pairwise": True,
                            "rho_a": rho_a,
                            "rho_b": rho_b,
                            "utility_a": utility_a,
                            "utility_b": utility_b,
                            "eps": float(eps),
                            "delta_u": float(delta),
                            "winner": None if winner is None else float(winner),
                        }
        mask = plan.get("candidate_mask")
        units = plan.get("units")
        if do_card and card_probe is None and torch.is_tensor(mask) and torch.is_tensor(units):
            rho_card = float(plan["rho"].detach()) if torch.is_tensor(plan.get("rho")) else float(plan.get("rho") or 0.0)
            if rho_card <= 1e-8 and rho_probe is not None:
                if rho_probe.get("pairwise") and rho_probe.get("winner") is not None:
                    rho_card = float(rho_probe["winner"])
                elif rho_probe.get("pairwise"):
                    rho_card = 0.5 * (float(rho_probe["rho_a"]) + float(rho_probe["rho_b"]))
                else:
                    rho_card = float(rho_probe.get("rho") or 0.1)
            if rho_card <= 1e-8:
                rho_card = float(controller.policy.rho_probes[0]) if controller.policy.rho_probes else 0.1
            card_pair = controller.policy.next_card_probe_pair(mask)
            if card_pair is not None:
                index_a, index_b = int(card_pair[0]), int(card_pair[1])
                named_a = _card_named(module, rho_card, step, units[index_a])
                named_b = _card_named(module, rho_card, step, units[index_b])
                if named_a is not None and named_b is not None:
                    utility_a = float(loss_without - _adapt_and_loss(named_a))
                    utility_b = float(loss_without - _adapt_and_loss(named_b))
                    eps = _pair_eps(utility_a, utility_b)
                    card_probe = {
                        "module": module,
                        "pairwise": True,
                        "index_a": index_a,
                        "index_b": index_b,
                        "rho": float(rho_card),
                        "utility_a": utility_a,
                        "utility_b": utility_b,
                        "eps": float(eps),
                        "delta_u": float(utility_a - utility_b),
                    }
            else:
                card_index = controller.policy.next_card_probe_index(mask)
                if card_index is not None:
                    named = _card_named(module, rho_card, step, units[int(card_index)])
                    if named is not None:
                        card_probe = {
                            "module": module,
                            "pairwise": False,
                            "index": int(card_index),
                            "rho": float(rho_card),
                            "utility": float(loss_without - _adapt_and_loss(named)),
                        }
        if (not do_rho or rho_probe is not None) and (not do_card or card_probe is not None):
            break
    detector.load_state_dict(snapshot, strict=True)
    return {"rho_probe": rho_probe, "card_probe": card_probe}


def retrieval_fomaml_step(
    detector, optimizer, support_records, query_records, episode: dict, bank: dict,
    controller: GammaController, device, settings: dict, *, allow_borrow: bool,
    inner_lr_controller: InnerLRController | None = None,
    theta0_state: dict | None = None, adapter=None, read_only_cache: bool = False, writable_cache=None,
    extra_caches=None,
) -> dict:
    """检索、临时注入、Support 一步；Query 反事实效用更新 ψ；共享 θ 用借用臂梯度外更新。"""
    from time import perf_counter

    if adapter is None:
        raise ValueError("经验检索需要 Episode 适配器来读取正常参照和 Support")
    started = perf_counter()
    signature = task_signature(
        adapter, episode, bank, detector, device, settings, theta0_state, writable_cache, read_only_cache,
        extra_caches,
    )
    allowed = allowed_source_families(episode)
    excluded = excluded_experience_group(bank, str(episode["family_id"]))
    alphas, bound_before = module_inner_lrs(settings, inner_lr_controller)
    gate_grads = support_packed_grads(detector, support_records, device, settings, bank["layouts"])
    step_l2 = {
        module: float(alphas[module]) * float(torch.linalg.vector_norm(gate_grads[module].reshape(-1)))
        for module in MODULES
    }
    plans = plan_modules_in_graph_order(
        bank, signature["keys"], excluded, allowed, controller, allow_borrow, step_l2,
    )
    loo_modules: list[str] = []
    force_closed: list[str] = []
    if allow_borrow:
        loo_modules = controller.policy.next_loo_modules()
        prob = float(getattr(controller.policy, "force_close_prob", 0.0) or 0.0)
        to_close = []
        if prob > 0.0 and loo_modules:
            for module in loo_modules:
                if float(torch.rand(()).item()) < prob:
                    to_close.append(module)
        if to_close:
            force_closed = apply_force_close_and_replan(
                plans, to_close,
                bank=bank, keys=signature["keys"], excluded_group=excluded,
                allowed_families=allowed, controller=controller, allow_borrow=allow_borrow,
                step_l2=step_l2,
            )
    retrieval_seconds = perf_counter() - started
    gate_borrow_by_support(plans, gate_grads)
    snapshot = _state_snapshot(detector)
    before = _parameter_snapshot(detector)
    borrow = {}
    for plan in plans:
        if plan["inject"]:
            borrow.update(plan["named"])
    inner_started = perf_counter()
    adapted = retrieval_support_adapt(
        detector, support_records, device, settings, alphas, borrow, bank["layouts"],
    )
    inner_seconds = perf_counter() - inner_started
    support_losses = adapted["support_losses"]
    module_delta = adapted["module_delta"]
    leak = _borrow_leak(before, adapted["after"], borrow, module_delta)
    if leak > 1e-5:
        raise AssertionError(f"本任务增量混入了借用参数，最大绝对差 {leak}")
    inner_change = max(float(value.abs().max()) for block in module_delta.values() for value in block.values())
    if inner_change <= 0:
        raise AssertionError("Support 适配没有更新检测器")
    import gc

    outer_started = perf_counter()
    outer_parts = {"query_grad": 0.0, "null": 0.0, "loo": 0.0, "probes": 0.0, "sgd": 0.0}
    # 多轮反事实前先回收碎片，避免读大图时 CPU MemoryError。
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    # 借用臂：Query 损失 + θ 外循环梯度
    part_t = perf_counter()
    query_loss, query_grads, packed_query = query_total_packed_grads(
        detector, query_records, device, settings, bank["layouts"],
    )
    loss_borrow = float(query_loss.detach()) if torch.is_tensor(query_loss) else float(query_loss)
    outer_parts["query_grad"] = perf_counter() - part_t
    detector.load_state_dict(snapshot, strict=True)
    # 对照臂：相同 Support/Query/RNG，无借用（轻路径，不打包梯度）
    part_t = perf_counter()
    loss_null = counterfactual_adapt_query_loss(
        detector, snapshot, support_records, query_records, device, settings, alphas, {},
        rng_cpu=rng_cpu, rng_cuda=rng_cuda,
    )
    outer_parts["null"] = perf_counter() - part_t
    utility = float(loss_null - loss_borrow)
    rho_probe = None
    card_probe = None
    if allow_borrow and loo_modules:
        part_t = perf_counter()
        module_utilities, loo_kinds, loo_baselines = module_leave_one_out_utilities(
            detector, snapshot, plans, support_records, query_records, device, settings, alphas,
            bank["layouts"], loss_borrow, loo_modules, rng_cpu=rng_cpu, rng_cuda=rng_cuda,
        )
        outer_parts["loo"] = perf_counter() - part_t
        part_t = perf_counter()
        probes = module_rho_and_card_probes(
            detector, snapshot, plans, support_records, query_records, device, settings, alphas,
            bank["layouts"], loo_modules, controller, rng_cpu=rng_cpu, rng_cuda=rng_cuda,
            loss_without_by_module=loo_baselines,
        )
        outer_parts["probes"] = perf_counter() - part_t
        rho_probe = probes.get("rho_probe")
        card_probe = probes.get("card_probe")
        del loo_baselines, probes
    else:
        module_utilities, loo_kinds = {}, {}
    detector.load_state_dict(snapshot, strict=True)
    restored, owner = _state_difference(detector, snapshot)
    if restored != 0.0:
        raise AssertionError(f"外更新前没有恢复到借用前的共享参数：{owner} {restored}")
    part_t = perf_counter()
    optimizer.zero_grad(set_to_none=True)
    trainable = [parameter for parameter in detector.parameters() if parameter.requires_grad]
    if len(trainable) != len(query_grads):
        raise AssertionError("Query 梯度和共享可训练参数数量不一致")
    for parameter, grad in zip(trainable, query_grads):
        parameter.grad = grad
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable, float(settings["grad_clip_norm"]), error_if_nonfinite=True,
    ).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    outer_parts["sgd"] = perf_counter() - part_t
    outer_seconds = perf_counter() - outer_started
    updated = _parameter_snapshot(detector)
    outer_change = max(float((updated[name] - before[name]).abs().max()) for name in before)
    if grad_norm <= 0 or outer_change <= 0:
        raise AssertionError("Query 外循环没有更新共享检测器")
    ap_targets = None
    step_id = int(getattr(controller.policy, "_loo_cursor", 0))
    every = int(controller.policy.ap_utility_every_n_steps)
    if controller.policy.ap_utility_enabled and every > 0 and adapter is not None and (step_id % every == 0):
        from training.evaluation import evaluate_detector
        eval_cfg = {
            "iou_thresholds": list(settings.get("eval_iou_thresholds") or [0.5, 0.75]),
            "working_iou": float(settings.get("working_iou", 0.5)),
            "ignore_iou": float(settings["ignore_iou"]),
            "score_threshold": float(settings.get("score_threshold", 0.05)),
            "batch_size": 1,
            "prefetch_images": False,
        }
        detector.load_state_dict(snapshot, strict=True)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda, device)
        retrieval_support_adapt(
            detector, support_records, device, settings, alphas,
            {k: v for plan in plans if plan.get("inject") for k, v in plan["named"].items()},
            bank["layouts"],
        )
        borrow_ap = evaluate_detector(detector, [episode], adapter, device, eval_cfg)
        detector.load_state_dict(snapshot, strict=True)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda, device)
        retrieval_support_adapt(detector, support_records, device, settings, alphas, {}, bank["layouts"])
        null_ap = evaluate_detector(detector, [episode], adapter, device, eval_cfg)
        detector.load_state_dict(snapshot, strict=True)
        ap_targets = {
            "ap50": float(borrow_ap["product_macro"]["ap50"]) - float(null_ap["product_macro"]["ap50"]),
            "ap75": (
                None if borrow_ap["product_macro"].get("ap75") is None or null_ap["product_macro"].get("ap75") is None
                else float(borrow_ap["product_macro"]["ap75"]) - float(null_ap["product_macro"]["ap75"])
            ),
        }
        for name, parameter in detector.named_parameters():
            parameter.data.copy_(updated[name].to(device=parameter.device, dtype=parameter.dtype))
    if allow_borrow and (module_utilities or rho_probe or card_probe or ap_targets is not None):
        proxy = apply_counterfactual_policy(
            controller, plans, module_utilities, ap_targets=ap_targets,
            rho_probe=rho_probe, card_probe=card_probe, task_utility=utility,
        )
        if module_utilities:
            update_module_contributions(bank, module_utilities, plans)
    else:
        proxy = {
            "loss": 0.0,
            "module_utilities": {},
            "dots": {module: 0.0 for module in MODULES},
            "gain_grads": {module: 0.0 for module in MODULES},
            "ap_loss": None,
            "ap_prediction": None,
            "sign_agreement": None,
            "rho_probe": None,
            "card_probe": None,
        }
    alpha_report = None
    if inner_lr_controller is not None:
        alpha_report = apply_alpha_proxy(inner_lr_controller, adapted["support_grads"], packed_query)
    borrow_applied = any(bool(plan["inject"]) for plan in plans)
    detector.eval()
    choices = {}
    weights = {}
    borrow_l2 = {}
    for plan in plans:
        total = 0.0
        for value in plan["named"].values():
            total += float(torch.sum(value.detach().float() ** 2))
        borrow_l2[plan["module"]] = total ** 0.5
    for plan in plans:
        module = plan["module"]
        if not allow_borrow:
            choices[module] = "NoBorrow"
        elif plan.get("utility_reject") == "empty":
            choices[module] = "Empty"
        elif plan["inject"]:
            choices[module] = "Borrow"
        else:
            choices[module] = "Null"
        weights[module] = float(plan["rho"].detach()) * float(plan.get("z_hard") or 0.0)
    del before, adapted, query_grads, packed_query
    return {
        "support_losses": support_losses,
        "query_detection_loss": loss_borrow,
        "query_loss_borrow": loss_borrow,
        "query_loss_null": loss_null,
        "counterfactual_utility": utility,
        "module_utilities": module_utilities,
        "loo_modules": list(loo_modules),
        "loo_kinds": loo_kinds,
        "force_closed": list(force_closed),
        "rho_probe": rho_probe,
        "card_probe": card_probe,
        "sign_agreement": proxy.get("sign_agreement"),
        "ap_targets": ap_targets,
        "inner_max_weight_change": inner_change,
        "outer_max_weight_change": outer_change,
        "query_grad_norm": grad_norm,
        "module_delta": module_delta,
        "inner_seconds": inner_seconds,
        "outer_seconds": outer_seconds,
        "outer_parts": {key: float(value) for key, value in outer_parts.items()},
        "retrieval_seconds": retrieval_seconds,
        "module_choices": choices,
        "reuse_weights": weights,
        "allow_borrow": bool(allow_borrow),
        "injected": {plan["module"]: bool(plan["inject"]) for plan in plans},
        "source_groups": {plan["module"]: list(plan["source_groups"]) for plan in plans},
        "sim_max": {plan["module"]: float(plan["sim_max"]) for plan in plans},
        "proxy_dots": proxy["dots"],
        "gain_grads": proxy["gain_grads"],
        "proxy_loss": proxy["loss"],
        "restore_max_abs": restored,
        "borrow_leak_max_abs": leak,
        "borrow_l2": borrow_l2,
        "gamma": weights,
        "module_utility": module_utility_report(plans, alphas),
        "borrow_applied": borrow_applied,
        "gamma_proxy_meaning": "query_counterfactual_utility",
        "inner_lr_before": alphas,
        "inner_lr_after": None if alpha_report is None else alpha_report["values"],
        "inner_lr_bound_hits": bound_before if alpha_report is None else alpha_report["bound_hits"],
        "alpha_dots": None if alpha_report is None else alpha_report["dots"],
    }


def holdout_borrow_plan(
    bank: dict, episode: dict, keys: dict, controller: GammaController, records=None,
    *, step_l2: dict[str, float] | None = None,
    null_eval_threshold: float | None = None,
) -> list[dict]:
    """留出只读检索：硬门控；阈值可覆盖配置。不更新 θ/库/决策网。"""
    if records is not None:
        roles = {row["split_role"] for row in records}
        if roles - {"R", "SUPPORT"}:
            raise AssertionError(f"留出检索不能读取 Query：{sorted(roles)}")
    unexpected = set(keys) - {"Appearance", "Morphology", "Difference"}
    if unexpected:
        raise AssertionError(f"留出检索键只能来自外观、形态和差异：{sorted(unexpected)}")
    if null_eval_threshold is not None:
        controller.policy.null_eval_threshold = float(null_eval_threshold)
    family_id = str(episode["family_id"])
    allowed = allowed_source_families(episode)
    excluded = excluded_experience_group(bank, family_id)
    scales = step_l2 or {module: 1.0 for module in MODULES}
    return plan_modules_in_graph_order(bank, keys, excluded, allowed, controller, True, scales)


def retrieval_holdout_pair(
    detector, support_records, episode: dict, bank: dict, keys: dict, gamma: GammaController,
    alpha: InnerLRController, device, settings: dict,
) -> dict:
    """同一 Support 上比较无借用 A 和检索借用 B。α 与决策网络冻结，并且这里不读取 Query。"""
    before_alpha = {module: float(alpha.raw[module].detach()) for module in MODULES}
    before_policy = {name: float(param.detach().sum()) for name, param in gamma.policy.policy.named_parameters()}
    values, _hits = alpha.current()
    support_grads = support_packed_grads(detector, support_records, device, settings, bank["layouts"])
    step_l2 = {
        module: float(values[module]) * float(torch.linalg.vector_norm(support_grads[module].reshape(-1)))
        for module in MODULES
    }
    plans = holdout_borrow_plan(bank, episode, keys, gamma, support_records, step_l2=step_l2)
    snapshot = _state_snapshot(detector)
    retrieval_support_adapt(detector, support_records, device, settings, values, {})
    detector.load_state_dict(snapshot, strict=True)
    borrow = {}
    for plan in plans:
        if plan["inject"]:
            borrow.update(plan["named"])
    retrieval_support_adapt(detector, support_records, device, settings, values, borrow)
    detector.load_state_dict(snapshot, strict=True)
    after_alpha = {module: float(alpha.raw[module].detach()) for module in MODULES}
    after_policy = {name: float(param.detach().sum()) for name, param in gamma.policy.policy.named_parameters()}
    if after_alpha != before_alpha or after_policy != before_policy:
        raise AssertionError("留出适应改动了冻结的 α 或 γ")
    return {
        "adapt": "retrieval_support_adapt",
        "query_used_for_selection": False,
        "alpha_frozen": True,
        "gamma_frozen": True,
        "arm_a_injected": False,
        "arm_b_injected": {plan["module"]: bool(plan["inject"]) for plan in plans},
    }


def frozen_retrieval_structure(bank: dict) -> dict:
    """沿用投影、簇中心和簇分配。有效任务数按清单原样记录，不复制任务去补齐 M。"""
    assignments = []
    seen = set()
    for row in bank["assignments"]:
        task_id = str(row["task_id"])
        if task_id in seen:
            raise AssertionError(f"经验任务重复，不能把同一条 Episode 复制成多个任务：{task_id}")
        seen.add(task_id)
        assignments.append({
            "task_id": task_id,
            "family_id": str(row["family_id"]),
            "group": int(row["group"]),
            "cluster": {name: int(row["cluster"][name]) for name in _KEY_NAMES},
            "seed": int(row["seed"]),
            "support_image_ids": list(row.get("support_image_ids") or []),
            "reference_group_ids": list(row.get("reference_group_ids") or []),
            "support_group_ids": list(row.get("support_group_ids") or []),
            "reference_image_ids": list(row.get("reference_image_ids") or []),
        })
    if len(assignments) < 1:
        raise ValueError("经验库没有可刷新的任务")
    requested = int((bank.get("manifest") or {}).get("requested_tasks_per_family") or 0)
    return {
        "pca": bank["pca"],
        "centers": bank["centers"],
        "center_valid": bank["center_valid"],
        "groups": {str(family): int(group) for family, group in bank["groups"].items()},
        "assignments": assignments,
        "layouts": bank["layouts"],
        "effective_tasks": len(assignments),
        "requested_tasks_per_family": requested,
        "experience_groups": int(bank["experience_groups"]),
    }


def _saved_near_duplicates(bank: dict, source_dir) -> list[dict]:
    from pathlib import Path

    from common.io import read_csv

    if bank.get("near_duplicates"):
        return list(bank["near_duplicates"])
    path = Path(source_dir) / "near_duplicate_candidates.csv"
    if not path.is_file():
        return []
    return read_csv(path)


def refresh_deltas_complete(output_dir, assignments: list[dict]) -> bool:
    """经验库刷新目录是否已写齐全部任务增量，可跳过重算直接拼卡。"""
    from pathlib import Path

    delta_dir = Path(output_dir) / "task_deltas"
    if not delta_dir.is_dir():
        return False
    for row in assignments:
        if not (delta_dir / f"{row['task_id'].replace(':', '_')}.pt").is_file():
            return False
    return True


def _discard_task_deltas(directory) -> None:
    """卡片和索引核对通过后删掉全量任务差。检索只读卡片，下一轮会按新的 θ 重算。"""
    import shutil
    from pathlib import Path

    delta_dir = Path(directory) / "task_deltas"
    if not delta_dir.is_dir():
        return
    shutil.rmtree(delta_dir)
    print(f"经验库已删除拼卡后的全量任务差：{delta_dir}", flush=True)


def refresh_experience_cards(
    source_dir, detector, output_dir, adapter, episodes, device, settings: dict, *, resume_deltas: bool = False,
) -> dict:
    """用当前共享参数重算全部卡片。投影、簇中心和簇分配保持原样。"""
    from pathlib import Path

    from training.retrieval_memory import median_anchor, select_keep_ratio

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    bank = load_experience_bank(source_dir)
    structure = frozen_retrieval_structure(bank)
    by_id = {episode["episode_id"]: episode for episode in episodes}
    snapshot = _state_snapshot(detector)
    theta_hash = _state_sha256(snapshot)
    started = perf_counter()
    delta_dir = output_dir / "task_deltas"
    resume_ready = resume_deltas and refresh_deltas_complete(output_dir, structure["assignments"])
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_ready:
        raise FileExistsError(f"经验库版本目录已存在：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(exist_ok=True)
    refreshed = []
    if resume_ready:
        print(
            f"经验库刷新续跑：复用已有 {len(structure['assignments'])} 个任务增量，跳过重算",
            flush=True,
        )
        for row in structure["assignments"]:
            safe_name = row["task_id"].replace(":", "_") + ".pt"
            payload = load_checkpoint(delta_dir / safe_name)
            if payload["theta_sha256"] != theta_hash:
                raise AssertionError(f"续跑增量与当前 θ 不一致：{row['task_id']}")
            if payload.get("task_id") not in (None, row["task_id"]):
                raise AssertionError(f"续跑增量任务编号不匹配：{row['task_id']}")
            refreshed.append({
                **row,
                "loss": float(payload["loss"]),
                "norms": {module: float(payload["norms"][module]) for module in MODULES},
                "delta_file": f"task_deltas/{safe_name}",
                "support_image_ids": list(row.get("support_image_ids") or []),
            })
            del payload
            gc.collect()
    else:
        total = len(structure["assignments"])
        for index, row in enumerate(structure["assignments"], start=1):
            episode = by_id.get(row["task_id"])
            if episode is None:
                raise KeyError(f"刷新卡片时找不到原任务：{row['task_id']}")
            query_ids = {
                image_id
                for role in ("query_defect", "query_normal")
                for unit in episode[role]
                for image_id in unit["image_ids"]
            }
            records = adapter.build_episode_records(episode, roles=("support",))
            roles = {record["split_role"] for record in records}
            if roles - {"SUPPORT"}:
                raise AssertionError(f"刷新卡片读取了 Support 以外的角色：{sorted(roles)}")
            if any(record["image_id"] in query_ids for record in records):
                raise AssertionError(f"刷新卡片混入 Query 图像：{episode['episode_id']}")
            support = limit_role_views(records, int(settings["views_per_group"]))
            chosen = [record["image_id"] for record in support]
            if row["support_image_ids"] and set(chosen) != set(row["support_image_ids"]):
                raise AssertionError(f"刷新卡片的 Support 视角和原任务不一致：{row['task_id']}")
            seed = int(row["seed"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            print(f"经验库刷新增量 {index}/{total} task={row['task_id']}", flush=True)
            loss, deltas = _support_deltas(detector, support, device, settings)
            difference, owner = _state_difference(detector, snapshot)
            if difference != 0.0:
                raise AssertionError(f"刷新卡片改变了共享参数：{owner} {difference}")
            packed = {module: pack_named(deltas[module], structure["layouts"][module]) for module in MODULES}
            norms = {module: float(torch.linalg.vector_norm(packed[module])) for module in MODULES}
            safe_name = row["task_id"].replace(":", "_") + ".pt"
            save_checkpoint(delta_dir / safe_name, {
                "theta_sha256": theta_hash, "task_id": row["task_id"], "loss": loss, "norms": norms, "delta": packed,
            })
            refreshed.append({
                **row, "loss": loss, "norms": norms, "delta_file": f"task_deltas/{safe_name}",
                "support_image_ids": chosen,
            })
            del deltas, packed
            gc.collect()
    if [item["cluster"] for item in refreshed] != [item["cluster"] for item in structure["assignments"]]:
        raise AssertionError("刷新卡片改变了簇分配")
    by_family: dict[str, list] = {}
    for item in refreshed:
        by_family.setdefault(item["family_id"], []).append(item)
    family_norms = {module: [] for module in MODULES}
    for members in by_family.values():
        for module in MODULES:
            family_norms[module].append(torch.tensor([member["norms"][module] for member in members]))
    anchors = {module: median_anchor(family_norms[module]).float() for module in MODULES}
    keep_ratio = {module: float(bank["next_keep_ratio"][module]) for module in MODULES}
    config = load_retrieval_config()
    sketches = {module: {float(ratio): [] for ratio in config["trim_ratios"]} for module in MODULES}
    sketch_indices = {
        module: layered_sketch_indices(structure["layouts"][module], int(config["trim_sketch_dim"]), int(config["cluster_seed"]) + 17 + offset)
        for offset, module in enumerate(MODULES)
    }

    def load_packed(task_id: str) -> dict[str, torch.Tensor]:
        payload = load_checkpoint(delta_dir / f"{task_id.replace(':', '_')}.pt")
        if payload["theta_sha256"] != theta_hash:
            raise AssertionError("刷新后的任务梯度不是当前 θ 的快照")
        return payload["delta"]

    print(f"经验库刷新草图校准 tasks={len(refreshed)}", flush=True)
    for index, item in enumerate(refreshed, start=1):
        loaded_delta = load_packed(item["task_id"])
        for module in MODULES:
            clipped = clip_packed(loaded_delta[module], anchors[module])
            for ratio in config["trim_ratios"]:
                sketches[module][float(ratio)].append((
                    item["cluster"][KEY_FOR_MODULE[module]], item["group"],
                    sketch_packed(clipped, structure["layouts"][module], sketch_indices[module], float(ratio)),
                ))
            del clipped
        del loaded_delta
        if index == 1 or index == len(refreshed) or index % 5 == 0:
            print(f"经验库刷新草图 {index}/{len(refreshed)}", flush=True)
        gc.collect()
    calibration = {}
    next_keep = {}
    for module in MODULES:
        bins_by_ratio = {}
        for ratio, rows in sketches[module].items():
            bins: dict[tuple, list] = {}
            for cluster, group, vector in rows:
                bins.setdefault((cluster, group), []).append(vector)
            bins_by_ratio[float(ratio)] = [torch.stack(members) for members in bins.values()]
        choice = select_keep_ratio(
            bins_by_ratio, int(config["trim_calibration_min_members"]), int(config["trim_splits"]), 1.0,
        )
        calibration[module] = {
            "keep_ratio": float(choice["keep_ratio"]),
            "calibrated": bool(choice["calibrated"]),
            "scores": [
                {
                    "ratio": float(score["ratio"]), "bins": int(score["bins"]),
                    "score": -1.0 if score["score"] is None else float(score["score"]),
                    "score_missing": score["score"] is None,
                }
                for score in choice["scores"]
            ],
        }
        next_keep[module] = float(choice["keep_ratio"])
    del sketches
    gc.collect()
    cards_dir = output_dir / "cards"
    if cards_dir.exists():
        for path in cards_dir.glob("*.pt"):
            path.unlink()
    card_rows = []

    def on_card(key, entry):
        module, cluster, group = key
        card_rows.append(_persist_card_file(cards_dir, module, cluster, group, entry))
        entry["unit"] = torch.empty(0)
        entry["card"] = torch.empty(0)
        gc.collect()

    print(f"经验库刷新拼卡 keep={keep_ratio}", flush=True)
    for item in refreshed:
        item["theta_sha256"] = theta_hash
        item["theta_hash"] = theta_hash
        item["theta_version"] = theta_hash
    cards = assemble_group_cards(
        refreshed, load_packed, structure["layouts"], anchors, keep_ratio, int(config["min_members_for_ties"]),
        on_card=on_card,
    )
    caches = _cache_directories(bank)
    manifest = json.loads(json.dumps({
        **bank["manifest"],
        "theta_sha256": theta_hash,
        "requested_tasks_per_family": structure["requested_tasks_per_family"],
        "effective_tasks": structure["effective_tasks"],
        "assignments": refreshed,
        "refresh_parent": str(source_dir),
        "feature_cache_dirs": [str(path) for path in caches if path.is_dir()],
        **inner_lr_manifest_fields(settings),
    }))
    refreshed_bank = {
        "manifest": manifest,
        "near_duplicates": _saved_near_duplicates(bank, source_dir),
        "theta0_sha256": bank["theta0_sha256"],
        "theta_sha256": theta_hash,
        "preprocess_sha256": bank["preprocess_sha256"],
        "split_sha256": bank["split_sha256"],
        "experience_groups": structure["experience_groups"],
        "clusters_requested": int(bank["clusters_requested"]),
        "top_mass_threshold": float(bank["top_mass_threshold"]),
        "max_top_k": int(bank["max_top_k"]),
        "temperature": float(bank["temperature"]),
        "groups": structure["groups"],
        "layouts": structure["layouts"],
        "pca": structure["pca"],
        "centers": structure["centers"],
        "center_valid": structure["center_valid"],
        "anchors": {module: float(anchors[module]) for module in MODULES},
        "keep_ratio": keep_ratio,
        "next_keep_ratio": next_keep,
        "trim_calibration": calibration,
        "signatures": bank["signatures"],
        "assignments": refreshed,
        "cards_memory": cards,
    }
    save_experience_bank(output_dir, refreshed_bank, card_rows=card_rows)
    loaded = load_experience_bank(output_dir, theta_hash)
    if loaded["assignments"] != refreshed_bank["assignments"]:
        raise AssertionError("落盘后的任务清单和刷新结果不一致")
    for name in _KEY_NAMES:
        if not torch.equal(loaded["centers"][name], bank["centers"][name]):
            raise AssertionError(f"刷新后的 {name} 簇中心发生了变化")
        if not torch.equal(loaded["pca"][name]["components"], bank["pca"][name]["components"]):
            raise AssertionError(f"刷新后的 {name} 投影发生了变化")
    final_difference, final_owner = _state_difference(detector, snapshot)
    if final_difference != 0.0 or _state_sha256(_state_snapshot(detector)) != theta_hash:
        raise AssertionError(f"刷新结束时共享参数已变化：{final_owner} {final_difference}")
    _discard_task_deltas(output_dir)
    return {
        "run_dir": str(output_dir),
        "theta_sha256": theta_hash,
        "effective_tasks": structure["effective_tasks"],
        "requested_tasks_per_family": structure["requested_tasks_per_family"],
        "assignments_unchanged": True,
        "pca_unchanged": True,
        "centers_unchanged": True,
        "resumed_deltas": bool(resume_ready),
        "seconds": perf_counter() - started,
        "disk_bytes": _directory_bytes(output_dir),
    }


def check_retrieval_wiring(bank_dir, theta0_path) -> dict:
    """用最小真实经验库跑一条 Episode，并按更新后的 θ 刷新一轮卡片。不启动完整训练。"""
    import tempfile
    from pathlib import Path

    from data_proc.fomaml_episodes import load_training_episodes
    from models.detector import b0_model_config, build_faster_rcnn
    from training.theta0 import _open_modules, _usable_phase_views, episode_records
    from training.train_b0 import _check_runtime

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    load_verified_group_split()
    load_verified_research_splits()
    load_verified_episode_groups()
    bank_dir = Path(bank_dir).resolve(strict=True)
    theta0_path = Path(theta0_path).resolve(strict=True)
    memory_hash = file_sha256(bank_dir / "module_memory.pt")
    manifest_hash = file_sha256(bank_dir / "experience_manifest.json")
    bank = load_experience_bank(bank_dir)
    parent = load_checkpoint(theta0_path)
    source = parent["provenance"]
    adapter, episodes, scope = load_training_episodes(
        source["fold_id"], source["train_family_ids"], profile="three_dataset_oof_v1",
    )
    if set(scope["theta0_train_family_ids"]) & set(scope["meta_val_family_ids"]):
        raise AssertionError("Meta-Val 进入了 θ₀ 训练集")
    if set(scope["theta0_train_family_ids"]) & set(scope["outer_test_family_ids"]):
        raise AssertionError("Outer Test 进入了 θ₀ 训练集")
    episode = next(item for item in episodes if item["episode_id"] == "oof:mvtec:bottle:000")
    b0_model_config(source["settings"]["model"])
    device = _check_runtime(source["settings"])
    detector = build_faster_rcnn(source["settings"]["model"], initialize_backbone=False)
    detector.load_state_dict(parent["model"], strict=True)
    del parent
    _open_modules(detector)
    detector.to(device).eval()
    settings = detection_settings()
    fomaml = load_config(named_config_path("fomaml.yaml"))
    settings["inner_steps"] = int(fomaml["training"]["inner_steps"])
    settings["grad_clip_norm"] = float(source["settings"]["grad_clip_norm"])
    settings["fallback_views_per_group"] = int(fomaml["training"]["fallback_views_per_group"])
    support_records, query_records = episode_records(adapter, episode)
    support_records = _usable_phase_views(
        detector, support_records, device, settings["ignore_iou"], settings["views_per_group"],
        settings["fallback_views_per_group"],
    )
    query_records = _usable_phase_views(
        detector, query_records, device, settings["ignore_iou"], settings["views_per_group"],
        settings["fallback_views_per_group"],
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in detector.parameters() if parameter.requires_grad],
        lr=float(fomaml["training"]["outer_lr"]), momentum=float(fomaml["training"]["outer_momentum"]),
    )
    print("检索接线检查：bottle:000 借用、内循环、Query 梯度", flush=True)
    step = retrieval_fomaml_step(
        detector, optimizer, support_records, query_records, episode, bank, GammaController(), device, settings,
        allow_borrow=True, adapter=adapter, read_only_cache=True,
    )
    produced = step.pop("module_delta")
    delta_l2 = {}
    for module, block in produced.items():
        flat = torch.cat([value.reshape(-1).float() for value in block.values()])
        delta_l2[module] = float(torch.linalg.vector_norm(flat))
    del produced
    excluded = int(bank["groups"][episode["family_id"]])
    for module, groups in step["source_groups"].items():
        if excluded in groups:
            raise AssertionError(f"{module} 检索到了任务自己的经验组")
    sign_checked = []
    sign_ok = True
    for module, grad in step["gain_grads"].items():
        if grad == 0:
            continue
        sign_checked.append(module)
        dot = step["proxy_dots"][module]
        if dot == 0 or (grad > 0) != (dot > 0):
            sign_ok = False
    print("检索接线检查：用更新后的共享参数刷新卡片", flush=True)
    with tempfile.TemporaryDirectory(prefix="retrieval-refresh-") as temporary:
        refresh = refresh_experience_cards(
            bank_dir, detector, Path(temporary) / "epoch", adapter, episodes, device, settings,
        )
        current_hash = _state_sha256(_state_snapshot(detector))
        refreshed = load_experience_bank(refresh["run_dir"], current_hash)
        same_clusters = len(refreshed["assignments"]) == len(bank["assignments"]) and all(
            left["task_id"] == right["task_id"] and left["cluster"] == right["cluster"]
            for left, right in zip(bank["assignments"], refreshed["assignments"])
        )
        refresh_summary = {
            "theta_sha256": refreshed["theta_sha256"],
            "matches_detector": refreshed["theta_sha256"] == current_hash,
            "differs_from_source": refreshed["theta_sha256"] != bank["theta_sha256"],
            "assignments_unchanged": same_clusters,
            "effective_tasks": refresh["effective_tasks"],
            "requested_tasks_per_family": refresh["requested_tasks_per_family"],
            "seconds": refresh["seconds"],
            "disk_bytes": refresh["disk_bytes"],
            "deleted": True,
        }
        del refreshed
    if file_sha256(bank_dir / "module_memory.pt") != memory_hash or file_sha256(bank_dir / "experience_manifest.json") != manifest_hash:
        raise AssertionError("接线检查改写了原来的最小经验库")
    report = {
        "episode_id": episode["episode_id"],
        "forced_allow_borrow": True,
        "injected": step["injected"],
        "borrow_l2": step["borrow_l2"],
        "delta_l2": delta_l2,
        "gamma": step["gamma"],
        "source_groups": step["source_groups"],
        "restore_max_abs": step["restore_max_abs"],
        "borrow_leak_max_abs": step["borrow_leak_max_abs"],
        "proxy_dots": step["proxy_dots"],
        "gain_grads": step["gain_grads"],
        "proxy_sign_matches": sign_ok,
        "proxy_sign_checked": sign_checked,
        "query_seen_by_theta0": bank["manifest"]["query_exposure"]["query_seen_by_theta0"],
        "meta_val_disjoint_from_theta0": True,
        "outer_test_disjoint_from_theta0": True,
        "source_bank_unchanged": True,
        "refresh": refresh_summary,
        "full_training": False,
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["restore_max_abs"] != 0.0 or report["borrow_leak_max_abs"] > 1e-5:
        raise AssertionError("借用没有在外更新前从共享参数里撤出")
    if not any(report["injected"].values()):
        raise AssertionError("这条真实任务没有注入任何模块，借用恢复未被实际观察到")
    if not sign_checked or not sign_ok:
        raise AssertionError("Query 梯度没有按一阶代理更新 γ")
    if not refresh_summary["matches_detector"] or not refresh_summary["differs_from_source"]:
        raise AssertionError("刷新后的经验库和当前 θ 不是同一版本")
    if not refresh_summary["assignments_unchanged"]:
        raise AssertionError("刷新卡片改变了簇分配")
    if refresh_summary["effective_tasks"] != len(bank["assignments"]):
        raise AssertionError("刷新卡片改变了有效任务数")
    return report


def check_alpha_gamma_theta(bank_dir, theta0_path) -> dict:
    """用一条真实 Episode 核对 α、γ、θ 三条更新互不影响。不刷新经验库，也不启动完整训练。"""
    from pathlib import Path

    from data_proc.fomaml_episodes import load_training_episodes
    from models.detector import b0_model_config, build_faster_rcnn
    from training.theta0 import _open_modules, _usable_phase_views, episode_records
    from training.train_b0 import _check_runtime

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    load_verified_group_split()
    load_verified_research_splits()
    load_verified_episode_groups()
    bank_dir = Path(bank_dir).resolve(strict=True)
    theta0_path = Path(theta0_path).resolve(strict=True)
    memory_hash = file_sha256(bank_dir / "module_memory.pt")
    manifest_hash = file_sha256(bank_dir / "experience_manifest.json")
    bank = load_experience_bank(bank_dir)
    parent = load_checkpoint(theta0_path)
    source = parent["provenance"]
    adapter, episodes, _scope = load_training_episodes(
        source["fold_id"], source["train_family_ids"], profile="three_dataset_oof_v1",
    )
    episode = next(item for item in episodes if item["episode_id"] == "oof:mvtec:bottle:000")
    b0_model_config(source["settings"]["model"])
    device = _check_runtime(source["settings"])
    detector = build_faster_rcnn(source["settings"]["model"], initialize_backbone=False)
    detector.load_state_dict(parent["model"], strict=True)
    del parent
    _open_modules(detector)
    detector.to(device).eval()
    settings = detection_settings()
    fomaml = load_config(named_config_path("fomaml.yaml"))
    settings["inner_steps"] = int(fomaml["training"]["inner_steps"])
    settings["grad_clip_norm"] = float(source["settings"]["grad_clip_norm"])
    settings["fallback_views_per_group"] = int(fomaml["training"]["fallback_views_per_group"])
    support_records, query_records = episode_records(adapter, episode)
    support_records = _usable_phase_views(
        detector, support_records, device, settings["ignore_iou"], settings["views_per_group"],
        settings["fallback_views_per_group"],
    )
    query_records = _usable_phase_views(
        detector, query_records, device, settings["ignore_iou"], settings["views_per_group"],
        settings["fallback_views_per_group"],
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in detector.parameters() if parameter.requires_grad],
        lr=float(fomaml["training"]["outer_lr"]), momentum=float(fomaml["training"]["outer_momentum"]),
    )
    alpha = InnerLRController()
    gamma = GammaController()
    alpha_ids = {id(alpha.raw[module]) for module in MODULES}
    policy_ids = {id(parameter) for parameter in gamma.policy.policy.parameters()}
    theta_ids = {id(parameter) for parameter in detector.parameters()}
    if alpha_ids & theta_ids or policy_ids & theta_ids or alpha_ids & policy_ids:
        raise AssertionError("α、迁移决策网络和 θ 不是三套独立参数")
    alpha_before, _hits = alpha.current()
    policy_before = {name: float(param.detach().sum()) for name, param in gamma.policy.policy.named_parameters()}
    theta_before = _state_sha256(_state_snapshot(detector))
    if any(abs(value - 0.01) > 1e-6 for value in alpha_before.values()):
        raise AssertionError(f"模块学习率初值不是 0.01：{alpha_before}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = perf_counter()
    step = retrieval_fomaml_step(
        detector, optimizer, support_records, query_records, episode, bank, gamma, device, settings,
        allow_borrow=True, inner_lr_controller=alpha, adapter=adapter, read_only_cache=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    peak_mib = 0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**20
    produced = step.pop("module_delta")
    del produced
    alpha_after, hits = alpha.current()
    policy_after = {name: float(param.detach().sum()) for name, param in gamma.policy.policy.named_parameters()}
    theta_after = _state_sha256(_state_snapshot(detector))
    alpha_sign_ok = True
    alpha_checked = []
    for module, dot in step["alpha_dots"].items():
        if abs(dot) < 1e-8:
            continue
        alpha_checked.append(module)
        moved_up = alpha_after[module] > alpha_before[module]
        if (dot > 0) != moved_up:
            alpha_sign_ok = False
    if file_sha256(bank_dir / "module_memory.pt") != memory_hash or file_sha256(bank_dir / "experience_manifest.json") != manifest_hash:
        raise AssertionError("学习率检查改写了原来的最小经验库")
    policy_changed = {name: policy_after[name] != policy_before[name] for name in policy_before}
    report = {
        "episode_id": episode["episode_id"],
        "alpha_before": alpha_before,
        "alpha_after": alpha_after,
        "alpha_dots": step["alpha_dots"],
        "alpha_sign_matches": alpha_sign_ok,
        "alpha_sign_checked": alpha_checked,
        "bound_hits": hits,
        "policy_changed": any(policy_changed.values()),
        "theta_changed": theta_before != theta_after,
        "restore_max_abs": step["restore_max_abs"],
        "borrow_leak_max_abs": step["borrow_leak_max_abs"],
        "borrow_applied": step["borrow_applied"],
        "gamma_proxy_meaning": step["gamma_proxy_meaning"],
        "parameters_independent": True,
        "peak_cuda_allocated_mib": peak_mib,
        "elapsed_seconds": elapsed,
        "source_bank_unchanged": True,
        "full_training": False,
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not alpha_checked or not alpha_sign_ok:
        raise AssertionError("α 没有按 Query 与 Support 梯度的点积更新")
    if not report["policy_changed"]:
        raise AssertionError("条件化迁移决策网络没有独立更新")
    if not report["theta_changed"] or report["restore_max_abs"] != 0.0 or report["borrow_leak_max_abs"] > 1e-5:
        raise AssertionError("θ 的外循环更新和借用恢复不正确")
    return report
