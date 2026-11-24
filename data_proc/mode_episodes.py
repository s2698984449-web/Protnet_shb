"""从冻结物理组池按缺陷模式构造 Episode。不改划分，也不覆盖固定 Episode 文件。"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from common.io import (
    file_sha256, load_config, load_verified_episode_groups, load_verified_group_split,
    load_verified_research_splits, named_config_path, project_root, write_json, write_jsonl,
)
from data_proc.episodes import _source_mask, _stable_seed


def load_mode_episode_config() -> dict:
    config = load_config(named_config_path("mode_episode.yaml"))
    same = float(config["same_query_ratio"])
    other = float(config["other_query_ratio"])
    if abs(same + other - 1.0) > 1e-8 or same < 0 or other < 0:
        raise ValueError("同模式与其他模式的 Query 比例之和必须为 1")
    for key in ("k_shot", "q_defect_groups", "n_reference_groups", "q_normal_groups", "max_episodes_per_mode"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} 必须为正")
    return config


def _labels(row: dict) -> list[str]:
    return [part for part in str(row.get("defect_labels") or "").split("|") if part]


def _shuffle(groups: list[dict], seed: int, *parts: str) -> list[dict]:
    ranked = list(groups)
    random.Random(_stable_seed(seed, *parts)).shuffle(ranked)
    return ranked


def _unique(groups: list[dict]) -> list[dict]:
    seen = set()
    chosen = []
    for group in groups:
        group_id = group["physical_group_id"]
        if group_id in seen:
            continue
        seen.add(group_id)
        chosen.append(group)
    return chosen


def _take(groups: list[dict], count: int, used: set[str], seed: int, *parts: str) -> tuple[list[dict], bool]:
    available = [group for group in _unique(groups) if group["physical_group_id"] not in used]
    short = len(available) < count
    return _shuffle(available, seed, *parts)[: min(count, len(available))], short


def _other_groups(groups: list[dict], mode: str, used: set[str], count: int, seed: int, *parts: str) -> tuple[list[dict], bool]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for group in _unique(groups):
        if group["physical_group_id"] in used or mode in _labels(group):
            continue
        others = [label for label in _labels(group) if label != mode]
        if not others:
            continue
        buckets[sorted(others)[0]].append(group)
    if not buckets or count < 1:
        return [], count > 0
    order = _shuffle(sorted(buckets), seed, *parts, "order")
    picked = []
    picked_ids = set()
    while len(picked) < count:
        progressed = False
        for label in order:
            pool = [
                group for group in buckets[label]
                if group["physical_group_id"] not in used and group["physical_group_id"] not in picked_ids
            ]
            if not pool:
                continue
            choice = _shuffle(pool, seed, *parts, label, str(len(picked)))[0]
            picked.append(choice)
            picked_ids.add(choice["physical_group_id"])
            progressed = True
            if len(picked) >= count:
                break
        if not progressed:
            break
    return picked, len(picked) < count


def _unit(group: dict) -> dict:
    image_ids = str(group.get("image_ids") or "").split("|")
    image_ids = [image_id for image_id in image_ids if image_id]
    return {
        "physical_group_id": group["physical_group_id"],
        "image_ids": image_ids,
        "defect_labels": _labels(group),
        "episode_pool_role": group["episode_pool_role"],
    }


def _ratio(part: int, total: int) -> float | None:
    if total <= 0:
        return None
    return part / total


def build_mode_episodes(rows: list[dict], protocol: dict, dataset_id: str | None = None) -> tuple[list[dict], list[dict]]:
    """按产品族和缺陷模式生成 Episode。数量不够时降级并记录，不复制物理组。"""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if dataset_id is not None and row["dataset_id"] != dataset_id:
            continue
        grouped[(row["dataset_id"], row["product_id"])].append(row)
    episodes, skipped = [], []
    seed = int(protocol["seed"])
    same_target = int(round(int(protocol["q_defect_groups"]) * float(protocol["same_query_ratio"])))
    other_target = int(protocol["q_defect_groups"]) - same_target
    for product_key in sorted(grouped):
        rows_for_product = grouped[product_key]
        roles = defaultdict(list)
        for row in rows_for_product:
            roles[row["episode_pool_role"]].append(row)
        modes = sorted({label for row in roles["SUPPORT"] for label in _labels(row)})
        used: set[str] = set()
        if not modes:
            skipped.append({
                "dataset_id": product_key[0], "product_id": product_key[1],
                "reason": "no_support_mode", "n_modes": 0,
            })
            continue
        produced = {mode: 0 for mode in modes}
        limit = int(protocol["max_episodes_per_mode"])
        while any(produced[mode] < limit for mode in modes):
            progressed = False
            for mode in modes:
                if produced[mode] >= limit:
                    continue
                episode, skip = _one_mode_episode(
                    rows_for_product[0], roles, mode, len(modes), used, produced[mode], protocol, seed,
                    same_target, other_target,
                )
                if episode is None:
                    skipped.append(skip)
                    produced[mode] = limit
                    continue
                for group_id in (
                    episode["support_group_ids"] + episode["query_defect_group_ids"]
                    + episode["query_normal_group_ids"] + episode["reference_group_ids"]
                ):
                    used.add(group_id)
                episodes.append(episode)
                produced[mode] += 1
                progressed = True
            if not progressed:
                break
    return episodes, skipped


def _one_mode_episode(meta, roles, mode, n_modes, used, index, protocol, seed, same_target, other_target):
    dataset_id = meta["dataset_id"]
    product_id = meta["product_id"]
    parts = ("mode", dataset_id, product_id, mode, str(index))
    reasons = []
    support_pool = [row for row in roles["SUPPORT"] if mode in _labels(row)]
    support, support_short = _take(support_pool, int(protocol["k_shot"]), used, seed, *parts, "support")
    if not support:
        return None, {
            "dataset_id": dataset_id, "product_id": product_id, "support_mode": mode,
            "episode_index": index, "reason": "insufficient_support", "available_support": 0,
        }
    if support_short:
        reasons.append("insufficient_support")
    taken = {group["physical_group_id"] for group in support} | set(used)
    same_pool = [row for row in roles["QUERY_DEFECT"] if mode in _labels(row)]
    same, same_short = _take(same_pool, same_target, taken, seed, *parts, "same")
    taken.update(group["physical_group_id"] for group in same)
    if same_short:
        reasons.append("insufficient_same")
    other, other_short = _other_groups(roles["QUERY_DEFECT"], mode, taken, other_target, seed, *parts, "other")
    if n_modes < 2:
        reasons.append("single_mode")
    elif other_short:
        reasons.append("insufficient_other")
    taken.update(group["physical_group_id"] for group in other)
    reference, reference_short = _take(roles["R"], int(protocol["n_reference_groups"]), taken, seed, *parts, "reference")
    if reference_short:
        reasons.append("insufficient_reference")
    taken.update(group["physical_group_id"] for group in reference)
    normal, normal_short = _take(
        roles["QUERY_NORMAL"], int(protocol["q_normal_groups"]), taken, seed, *parts, "normal",
    )
    if normal_short:
        reasons.append("insufficient_normal")
    query_defect = same + other
    ids = [group["physical_group_id"] for group in support + query_defect + reference + normal]
    images = [image_id for group in support + query_defect + reference + normal for image_id in _unit(group)["image_ids"]]
    if len(ids) != len(set(ids)) or len(images) != len(set(images)):
        raise AssertionError(f"{dataset_id}:{product_id}:{mode} 复制了物理组或图像")
    for role, groups in (
        ("SUPPORT", support), ("QUERY_DEFECT", query_defect), ("R", reference), ("QUERY_NORMAL", normal),
    ):
        for group in groups:
            if group["episode_pool_role"] != role:
                raise AssertionError(f"模式采样越过了冻结角色池：{group['physical_group_id']}")
    query_modes = [mode] * len(same)
    for group in other:
        labels = [label for label in _labels(group) if label != mode]
        query_modes.append(sorted(labels)[0] if labels else "")
    total = len(query_defect)
    episode_seed = _stable_seed(seed, dataset_id, product_id, mode, str(index))
    return {
        "episode_id": f"mode:{dataset_id}:{product_id}:{mode}:{index:03d}",
        "protocol": protocol.get("version", "mode_v1"),
        "scope": protocol.get("scope", "controlled_validation"),
        "dataset_id": dataset_id,
        "family_id": meta["family_id"],
        "product_id": product_id,
        "support_mode": mode,
        "n_modes_available": n_modes,
        "query_modes": dict(Counter(query_modes)),
        "support_group_ids": [group["physical_group_id"] for group in support],
        "query_defect_group_ids": [group["physical_group_id"] for group in query_defect],
        "query_defect_modes": query_modes,
        "query_normal_group_ids": [group["physical_group_id"] for group in normal],
        "reference_group_ids": [group["physical_group_id"] for group in reference],
        "support": [_unit(group) for group in support],
        "query_defect": [_unit(group) for group in query_defect],
        "query_normal": [_unit(group) for group in normal],
        "reference": [_unit(group) for group in reference],
        "seed": episode_seed,
        "requested_same_ratio": float(protocol["same_query_ratio"]),
        "requested_other_ratio": float(protocol["other_query_ratio"]),
        "actual_same_ratio": _ratio(len(same), total),
        "actual_other_ratio": _ratio(len(other), total),
        "degraded": bool(reasons),
        "degradation_reason": ",".join(reasons),
    }, None


def query_mode_slices(episode: dict) -> dict:
    """评价时按清单拆 Query。这一步不参与检索，也不改 α、γ。"""
    mode = episode["support_mode"]
    same, other = [], []
    for group_id, label in zip(episode["query_defect_group_ids"], episode["query_defect_modes"]):
        if label == mode:
            same.append(group_id)
        else:
            other.append(group_id)
    normal = list(episode["query_normal_group_ids"])
    return {
        "same_mode_group_ids": same,
        "other_mode_group_ids": other,
        "normal_group_ids": normal,
        "overall_group_ids": same + other + normal,
    }


def write_mode_manifest(config: dict | None = None) -> dict:
    """写出模式 Episode 清单。写完后核对冻结分配文件没有变化。"""
    config = load_mode_episode_config() if config is None else config
    load_verified_group_split()
    _outer, _meta, inner = load_verified_research_splits()
    split_config = load_config(named_config_path("split.yaml"))
    assignment = project_root() / split_config["output_dir"] / split_config["artifacts"]["episode_groups"]
    before = file_sha256(assignment)
    rows = load_verified_episode_groups()
    usable = [row for row in rows if row["evaluation_stage"] != "outer_test"]
    episodes, skipped = build_mode_episodes(usable, config, dataset_id=config.get("dataset_id") or None)
    by_product = {(row["dataset_id"], row["product_id"]): row for row in rows}
    for episode in episodes:
        sample = by_product[(episode["dataset_id"], episode["product_id"])]
        universe, mask, available = _source_mask({
            "evaluation_stage": sample["evaluation_stage"],
            "inner_target_fold": sample["inner_target_fold"],
        }, inner)
        episode["evaluation_stage"] = sample["evaluation_stage"]
        episode["inner_target_fold"] = sample["inner_target_fold"]
        episode["source_universe_action_ids"] = list(universe)
        episode["source_available_mask"] = [int(bit) for bit in mask]
        episode["source_available_action_ids"] = list(available)
        episode["training_eligible"] = sample["evaluation_stage"] == "inner_oof"
        episode["official_unsupervised_protocol"] = False
    after = file_sha256(assignment)
    if before != after:
        raise AssertionError("模式采样改写了冻结物理组分配")
    output = project_root() / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / config["manifest_name"], episodes)
    selected = config.get("dataset_id")

    def _matches(row: dict) -> bool:
        return selected is None or row["dataset_id"] == selected

    def _mode_keys(source: list[dict], role: str | None = None) -> set[tuple[str, str]]:
        keys = set()
        for row in source:
            if not _matches(row) or (role is not None and row["episode_pool_role"] != role):
                continue
            for label in _labels(row):
                keys.add((row["product_id"], label))
        return keys

    reasons: dict[str, int] = defaultdict(int)
    for episode in episodes:
        for reason in str(episode["degradation_reason"]).split(","):
            if reason:
                reasons[reason] += 1
    for item in skipped:
        reasons[str(item["reason"])] += 1
    families = sorted({row["family_id"] for row in rows if _matches(row)})
    open_families = sorted({row["family_id"] for row in usable if _matches(row)})
    report = {
        "version": config["version"],
        "scope": config["scope"],
        "dataset_id": selected,
        "families": len(families),
        "families_excluding_outer_test": len(open_families),
        "defect_modes": len(_mode_keys(rows)),
        "defect_modes_excluding_outer_test": len(_mode_keys(usable)),
        "support_modes_excluding_outer_test": len(_mode_keys(usable, "SUPPORT")),
        "episodes": len(episodes),
        "exact_episodes": sum(int(not episode["degraded"]) for episode in episodes),
        "training_eligible_episodes": sum(int(episode["training_eligible"]) for episode in episodes),
        "skipped": len(skipped),
        "degraded": sum(int(episode["degraded"]) for episode in episodes),
        "degradation_reasons": dict(sorted(reasons.items())),
        "assignment_sha256": before,
        "assignment_unchanged": True,
        "outer_test_excluded": True,
        "official_unsupervised_protocol": False,
        "replaces_three_dataset_protocol": False,
        "manifest": str(output / config["manifest_name"]),
    }
    write_json(output / config["report_name"], report)
    return report
