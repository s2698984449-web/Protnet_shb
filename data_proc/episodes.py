"""A1/A2：从冻结物理组池构造固定 Episode Bank，并执行边界校验。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.io import (
    file_sha256,
    load_config,
    load_verified_episode_groups,
    load_verified_group_split,
    load_verified_research_splits,
    named_config_path,
    project_root,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join((str(seed), *parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _split_pipe(value: str) -> list[str]:
    return [part for part in value.split("|") if part]


def _group_unit(row: dict, episode_classes: set[str]) -> dict:
    labels = set(_split_pipe(row["defect_labels"]))
    return {
        "physical_group_id": row["physical_group_id"],
        "image_ids": _split_pipe(row["image_ids"]),
        "image_paths": _split_pipe(row["image_paths"]),
        "mask_paths": _split_pipe(row["mask_paths"]),
        "visible_defect_labels": sorted(labels),
        "ignore_labels": sorted(labels - episode_classes),
        "group_kind": row["group_kind"],
    }


def _index_group_pools(rows: list[dict]) -> dict[tuple[str, str], dict]:
    pools: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "meta": None,
            "roles": defaultdict(list),
        }
    )
    for row in rows:
        key = (row["dataset_id"], row["product_id"])
        slot = pools[key]
        meta = {
            "dataset_id": row["dataset_id"],
            "family_id": row["family_id"],
            "product_id": row["product_id"],
            "outer_split": row["outer_split"],
            "meta_split": row["meta_split"],
            "evaluation_stage": row["evaluation_stage"],
            "inner_target_fold": row["inner_target_fold"],
            "source_mask_key": row["source_mask_key"],
        }
        if slot["meta"] is not None and slot["meta"] != meta:
            raise AssertionError(f"{key} 层级元数据不一致")
        slot["meta"] = meta
        parsed = dict(row)
        parsed["labels"] = set(_split_pipe(row["defect_labels"]))
        slot["roles"][row["episode_pool_role"]].append(parsed)
    return pools


def _class_counts(slot: dict) -> tuple[dict[str, int], dict[str, int]]:
    support = Counter(
        label
        for group in slot["roles"]["SUPPORT"]
        for label in group["labels"]
    )
    query = Counter(
        label
        for group in slot["roles"]["QUERY_DEFECT"]
        for label in group["labels"]
    )
    return dict(support), dict(query)


def build_episode_statistics(
    pools: dict[tuple[str, str], dict],
    protocol: dict,
) -> tuple[list[dict], dict[tuple[str, str], list[str]]]:
    rows = []
    eligible_by_product: dict[tuple[str, str], list[str]] = {}
    n_way = int(protocol["n_way"])
    k_shot = int(protocol["k_shot"])
    q_min = int(protocol["q_defect_per_class"])
    n_reference = int(protocol["n_reference_groups"])
    q_normal = int(protocol["q_normal_groups"])
    for product_key, slot in sorted(pools.items()):
        support_counts, query_counts = _class_counts(slot)
        labels = sorted(set(support_counts) | set(query_counts))
        eligible = [
            label
            for label in labels
            if support_counts.get(label, 0) >= k_shot
            and query_counts.get(label, 0) >= q_min
        ]
        normal_ok = (
            len(slot["roles"]["R"]) >= n_reference
            and len(slot["roles"]["QUERY_NORMAL"]) >= q_normal
        )
        eligible_by_product[product_key] = eligible if normal_ok else []
        for label in labels:
            rows.append(
                {
                    "dataset_id": product_key[0],
                    "family_id": slot["meta"]["family_id"],
                    "product_id": product_key[1],
                    "evaluation_stage": slot["meta"]["evaluation_stage"],
                    "label": label,
                    "n_r_groups": len(slot["roles"]["R"]),
                    "n_support_groups": support_counts.get(label, 0),
                    "n_query_normal_groups": len(
                        slot["roles"]["QUERY_NORMAL"]
                    ),
                    "n_query_defect_groups": query_counts.get(label, 0),
                    "main_protocol_eligible_class": int(label in eligible and normal_ok),
                    "main_task_qualified": int(
                        len(eligible) >= n_way and normal_ok
                    ),
                    "max_balanced_kq": min(
                        support_counts.get(label, 0),
                        query_counts.get(label, 0),
                    ),
                }
            )
    return rows, eligible_by_product


def _feasibility_summary(
    pools: dict[tuple[str, str], dict],
    grid: dict,
) -> list[dict]:
    result = []
    for n_way in grid["n_way"]:
        for k_shot in grid["k_shot"]:
            for q_min in grid["q_min"]:
                qualified = 0
                by_dataset = Counter()
                for product_key, slot in pools.items():
                    support, query = _class_counts(slot)
                    eligible = sum(
                        support.get(label, 0) >= int(k_shot)
                        and query.get(label, 0) >= int(q_min)
                        for label in set(support) | set(query)
                    )
                    if eligible >= int(n_way):
                        qualified += 1
                        by_dataset[product_key[0]] += 1
                result.append(
                    {
                        "n_way": int(n_way),
                        "k_shot": int(k_shot),
                        "q_min": int(q_min),
                        "qualified_products": qualified,
                        "mvtec": by_dataset["mvtec"],
                        "visa": by_dataset["visa"],
                        "realiad": by_dataset["realiad"],
                    }
                )
    return result


def _select_pilot_products(
    pools: dict[tuple[str, str], dict],
    eligible_by_product: dict[tuple[str, str], list[str]],
    max_per_dataset: int,
    seed: int,
) -> list[tuple[str, str]]:
    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for product_key, slot in pools.items():
        if (
            slot["meta"]["evaluation_stage"] == "meta_validation"
            and eligible_by_product[product_key]
        ):
            candidates[product_key[0]].append(product_key)
    selected = []
    for dataset_id, products in sorted(candidates.items()):
        ranked = sorted(
            products,
            key=lambda key: _stable_seed(
                seed, "pilot_product", dataset_id, key[1]
            ),
        )
        selected.extend(ranked[:max_per_dataset])
    return selected


def _sample_groups(
    groups: list[dict],
    count: int,
    seed: int,
    *parts: str,
) -> list[dict]:
    if len(groups) < count:
        raise AssertionError(
            f"可用物理组 {len(groups)} 少于请求数量 {count}：{parts}"
        )
    ranked = list(groups)
    random.Random(_stable_seed(seed, *parts)).shuffle(ranked)
    return ranked[:count]


def _source_mask(
    meta: dict,
    inner: dict,
) -> tuple[list[str], list[int], list[str]]:
    universe = inner["source_universe_action_ids"]
    if meta["evaluation_stage"] == "inner_oof":
        fold_id = int(meta["inner_target_fold"])
        fold = next(fold for fold in inner["folds"] if fold["fold_id"] == fold_id)
        mask = fold["source_available_mask"]
    else:
        mask = [1] * len(universe)
    available = [
        action for action, enabled in zip(universe, mask) if int(enabled) == 1
    ]
    return universe, mask, available


def _build_task_episodes(
    bank_name: str,
    product_key: tuple[str, str],
    slot: dict,
    eligible_classes: list[str],
    protocol: dict,
    inner: dict,
    seed: int,
) -> list[dict]:
    n_way = int(protocol["n_way"])
    k_shot = int(protocol["k_shot"])
    q_defect = int(protocol["q_defect_per_class"])
    n_reference = int(protocol["n_reference_groups"])
    q_normal = int(protocol["q_normal_groups"])
    n_episodes = int(protocol["episodes_per_task"])
    class_rank = sorted(
        eligible_classes,
        key=lambda label: _stable_seed(
            seed, "class_rank", product_key[0], product_key[1], label
        ),
    )
    universe, source_mask, available = _source_mask(slot["meta"], inner)
    prior_usage: Counter[str] = Counter()
    episodes = []
    for episode_index in range(n_episodes):
        start = (episode_index * n_way) % len(class_rank)
        selected_classes = [
            class_rank[(start + offset) % len(class_rank)]
            for offset in range(n_way)
        ]
        class_set = set(selected_classes)
        reference = _sample_groups(
            slot["roles"]["R"],
            n_reference,
            seed,
            bank_name,
            product_key[0],
            product_key[1],
            str(episode_index),
            "R",
        )
        query_normal = _sample_groups(
            slot["roles"]["QUERY_NORMAL"],
            q_normal,
            seed,
            bank_name,
            product_key[0],
            product_key[1],
            str(episode_index),
            "Q_NORMAL",
        )
        support_by_class = {}
        query_by_class = {}
        for label in selected_classes:
            support_candidates = [
                group
                for group in slot["roles"]["SUPPORT"]
                if label in group["labels"]
            ]
            query_candidates = [
                group
                for group in slot["roles"]["QUERY_DEFECT"]
                if label in group["labels"]
            ]
            support_by_class[label] = _sample_groups(
                support_candidates,
                k_shot,
                seed,
                bank_name,
                product_key[0],
                product_key[1],
                str(episode_index),
                "S",
                label,
            )
            query_by_class[label] = _sample_groups(
                query_candidates,
                q_defect,
                seed,
                bank_name,
                product_key[0],
                product_key[1],
                str(episode_index),
                "Q_DEFECT",
                label,
            )
        support = {
            group["physical_group_id"]: group
            for groups in support_by_class.values()
            for group in groups
        }
        query_defect = {
            group["physical_group_id"]: group
            for groups in query_by_class.values()
            for group in groups
        }
        r_ids = {group["physical_group_id"] for group in reference}
        s_ids = set(support)
        q_ids = {
            *(group["physical_group_id"] for group in query_normal),
            *query_defect,
        }
        if r_ids & s_ids or r_ids & q_ids or s_ids & q_ids:
            raise AssertionError(f"{product_key} Episode 物理组发生角色泄漏")
        used = sorted(r_ids | s_ids | q_ids)
        reused = sorted(group_id for group_id in used if prior_usage[group_id] > 0)
        prior_usage.update(used)
        episode_seed = _stable_seed(
            seed, bank_name, product_key[0], product_key[1], str(episode_index)
        )
        episodes.append(
            {
                "episode_id": (
                    f"{bank_name}:{product_key[0]}:{product_key[1]}:"
                    f"{episode_index:03d}"
                ),
                "bank": bank_name,
                "dataset_id": product_key[0],
                "family_id": slot["meta"]["family_id"],
                "product_id": product_key[1],
                "evaluation_stage": slot["meta"]["evaluation_stage"],
                "inner_target_fold": (
                    int(slot["meta"]["inner_target_fold"])
                    if slot["meta"]["inner_target_fold"] != ""
                    else None
                ),
                "seed": episode_seed,
                "n_way": n_way,
                "k_shot": k_shot,
                "q_defect_per_class": q_defect,
                "n_reference_groups": n_reference,
                "q_normal_groups": q_normal,
                "episode_classes": selected_classes,
                "ignore_policy": protocol["ignore_policy"],
                "view_policy": protocol["view_policy"],
                "reference": [
                    _group_unit(group, class_set) for group in reference
                ],
                "support": [
                    _group_unit(group, class_set)
                    for group in support.values()
                ],
                "support_group_ids_by_class": {
                    label: [
                        group["physical_group_id"]
                        for group in support_by_class[label]
                    ]
                    for label in selected_classes
                },
                "query_normal": [
                    _group_unit(group, class_set) for group in query_normal
                ],
                "query_defect": [
                    _group_unit(group, class_set)
                    for group in query_defect.values()
                ],
                "query_group_ids_by_class": {
                    label: [
                        group["physical_group_id"]
                        for group in query_by_class[label]
                    ]
                    for label in selected_classes
                },
                "source_universe_action_ids": universe,
                "source_available_mask": source_mask,
                "source_available_action_ids": available,
                "missing_utility_policy": "masked_not_zero",
                "reused_group_ids_from_prior_episodes": reused,
            }
        )
    return episodes


def validate_episode_bank(
    episodes: list[dict],
    assignment_rows: list[dict],
    outer: dict,
    meta: dict,
    inner: dict,
) -> None:
    role_by_group = {
        (
            row["dataset_id"],
            row["product_id"],
            row["physical_group_id"],
        ): row["episode_pool_role"]
        for row in assignment_rows
    }
    outer_test = set(outer["outer_test_family_ids"])
    meta_train = set(meta["meta_train_family_ids"])
    meta_val = set(meta["meta_val_family_ids"])
    episode_ids = set()
    for episode in episodes:
        if episode["episode_id"] in episode_ids:
            raise AssertionError(f"Episode ID 重复：{episode['episode_id']}")
        episode_ids.add(episode["episode_id"])
        family_id = episode["family_id"]
        if episode["bank"] == "oof":
            if family_id not in meta_train:
                raise AssertionError("OOF Episode 目标不属于 Meta-Train")
        elif episode["bank"] == "meta_val":
            if family_id not in meta_val:
                raise AssertionError("Meta-Validation Episode 目标边界错误")
        else:
            raise AssertionError(f"未知 Episode Bank：{episode['bank']}")
        if family_id in outer_test:
            raise AssertionError("Episode Bank 混入 Outer Test")

        role_units = {
            "R": episode["reference"],
            "SUPPORT": episode["support"],
            "QUERY_NORMAL": episode["query_normal"],
            "QUERY_DEFECT": episode["query_defect"],
        }
        ids_by_role = {
            role: {unit["physical_group_id"] for unit in units}
            for role, units in role_units.items()
        }
        roles = list(ids_by_role)
        for left_index, left in enumerate(roles):
            for right in roles[left_index + 1 :]:
                if ids_by_role[left] & ids_by_role[right]:
                    raise AssertionError(
                        f"{episode['episode_id']} {left}/{right} 物理组重叠"
                    )
        for role, units in role_units.items():
            for unit in units:
                key = (
                    episode["dataset_id"],
                    episode["product_id"],
                    unit["physical_group_id"],
                )
                if role_by_group.get(key) != role:
                    raise AssertionError(
                        f"{episode['episode_id']} 绕过冻结角色池：{key}"
                    )
        if len(episode["reference"]) != int(episode["n_reference_groups"]):
            raise AssertionError("正常参照物理组数量不满足")
        if len(episode["query_normal"]) != int(episode["q_normal_groups"]):
            raise AssertionError("正常 Query 物理组数量不满足")
        for label in episode["episode_classes"]:
            if len(episode["support_group_ids_by_class"][label]) != int(
                episode["k_shot"]
            ):
                raise AssertionError("Support K 不满足")
            if len(episode["query_group_ids_by_class"][label]) != int(
                episode["q_defect_per_class"]
            ):
                raise AssertionError("Query Q 不满足")
        universe = episode["source_universe_action_ids"]
        mask = episode["source_available_mask"]
        if len(universe) != len(mask) or universe[0] != "Null" or mask[0] != 1:
            raise AssertionError("Episode source_available_mask 无效")
        available = [
            action
            for action, enabled in zip(universe, mask)
            if int(enabled) == 1
        ]
        if available != episode["source_available_action_ids"]:
            raise AssertionError("Episode 源候选动作与 mask 不一致")


def _git_version(root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty = True
    return {"git_commit": commit, "git_worktree_dirty": dirty}


def _verify_frozen_episode_artifacts(
    config_file: Path,
    output_dir: Path,
    config: dict,
) -> bool:
    hash_path = output_dir / config["artifacts"]["hashes"]
    if not hash_path.is_file():
        return False
    with hash_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload["config_sha256"] != file_sha256(config_file):
        raise RuntimeError(
            "Episode v1 已冻结，但当前 episode.yaml 与冻结版本不同。"
            "请提高版本并使用新的 output_dir，禁止覆盖 v1。"
        )
    for filename, expected_hash in payload["artifacts"].items():
        artifact_path = output_dir / filename
        if not artifact_path.is_file():
            raise RuntimeError(f"冻结 Episode 产物缺失：{artifact_path}")
        if file_sha256(artifact_path) != expected_hash:
            raise RuntimeError(f"冻结 Episode 产物哈希不匹配：{artifact_path}")
    print(f"frozen_episode_verified {hash_path}")
    return True


def _render_report(
    path: Path,
    config: dict,
    feasibility: list[dict],
    statistics: list[dict],
    meta_val_episodes: list[dict],
    oof_episodes: list[dict],
) -> None:
    protocol = config["protocol"]
    main_row = next(
        row
        for row in feasibility
        if row["n_way"] == int(protocol["n_way"])
        and row["k_shot"] == int(protocol["k_shot"])
        and row["q_min"] == int(protocol["q_defect_per_class"])
    )
    qualified_products = {
        (row["dataset_id"], row["product_id"])
        for row in statistics
        if int(row["main_task_qualified"]) == 1
    }
    lines = [
        "# 固定 Episode 协议 v1",
        "",
        f"日期：{date.today().isoformat()}",
        "",
        "## 主协议",
        "",
        f"- N-way：{protocol['n_way']}",
        f"- K-shot：{protocol['k_shot']}，按独立物理组计数",
        f"- 每类缺陷 Query：{protocol['q_defect_per_class']} 个物理组",
        f"- 正常参照 R：{protocol['n_reference_groups']} 个纯正常物理组",
        f"- 正常 Query：{protocol['q_normal_groups']} 个纯正常物理组",
        f"- 视角：`{protocol['view_policy']}`",
        f"- Episode 外缺陷：`{protocol['ignore_policy']}`",
        "",
        "选择 1-way 作为首轮统一主协议，是因为 VisA 当前索引只有通用 anomaly 标签；先验证二分类缺陷定位。K=5、Q=5 在三个数据域均有合格任务，未合格产品不会强行补齐。",
        "",
        f"该协议合格产品：{main_row['qualified_products']}/57；"
        f"MVTec {main_row['mvtec']}，VisA {main_row['visa']}，Real-IAD {main_row['realiad']}。",
        f"类别级统计覆盖 {len(statistics)} 行，合格产品 {len(qualified_products)} 个。",
        "",
        "## Episode Bank",
        "",
        f"- Meta-Validation Pilot：{len(meta_val_episodes)} 条 Episode。",
        f"- OOF Training：{len(oof_episodes)} 条 Episode。",
        "- Meta-Validation Bank 不参与选择器梯度训练。",
        "- 每条 OOF Episode 保存对应折的完整 `source_available_mask`；缺失 U 按 mask 缺失，不补0。",
        "- 同一物理组的全部视角随组进入同一角色。混合物理组不会进入 R 或正常 Query。",
        "",
        "## 验收",
        "",
        "- R、Support、Query-Normal、Query-Defect 物理组两两不相交。",
        "- 每类 Support 数量等于 K，每类缺陷 Query 数量等于 Q。",
        "- Episode 只从冻结的角色池抽样，Outer Test 未进入当前两个 Bank。",
        "- Episode ID、种子、物理组、图像路径、Mask、Ignore 类和源候选掩码均已保存。",
        "",
        "下一步使用这两套固定 Bank 开发统一数据适配器和 Faster R-CNN 二分类定位基线；仍不启动 Δ、U 或选择器。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = project_root()
    config_file = named_config_path("episode.yaml")
    config = load_config(config_file)
    split_dir = root / config["split_dir"]
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.get("freeze_artifacts", False) and _verify_frozen_episode_artifacts(
        config_file, output_dir, config
    ):
        return

    load_verified_group_split()
    outer, meta, inner = load_verified_research_splits()
    assignment_rows = load_verified_episode_groups()
    pools = _index_group_pools(assignment_rows)
    statistics, eligible_by_product = build_episode_statistics(
        pools, config["protocol"]
    )
    feasibility = _feasibility_summary(pools, config["feasibility_grid"])

    pilot_config = config["meta_validation_pilot"]
    pilot_products = _select_pilot_products(
        pools,
        eligible_by_product,
        int(pilot_config["max_products_per_dataset"]),
        int(config["seed"]),
    )
    meta_val_episodes = []
    for product_key in pilot_products:
        meta_val_episodes.extend(
            _build_task_episodes(
                "meta_val",
                product_key,
                pools[product_key],
                eligible_by_product[product_key],
                config["protocol"],
                inner,
                int(config["seed"]),
            )
        )

    oof_products = sorted(
        product_key
        for product_key, slot in pools.items()
        if slot["meta"]["evaluation_stage"] == "inner_oof"
        and eligible_by_product[product_key]
    )
    oof_episodes = []
    for product_key in oof_products:
        oof_episodes.extend(
            _build_task_episodes(
                "oof",
                product_key,
                pools[product_key],
                eligible_by_product[product_key],
                config["protocol"],
                inner,
                int(config["seed"]),
            )
        )

    validate_episode_bank(
        [*meta_val_episodes, *oof_episodes],
        assignment_rows,
        outer,
        meta,
        inner,
    )
    artifacts = config["artifacts"]
    meta_path = output_dir / artifacts["meta_validation_bank"]
    oof_path = output_dir / artifacts["oof_bank"]
    statistics_path = output_dir / artifacts["statistics"]
    feasibility_path = output_dir / artifacts["feasibility"]
    write_jsonl(meta_path, meta_val_episodes)
    write_jsonl(oof_path, oof_episodes)
    statistic_fields = list(statistics[0].keys())
    write_csv(statistics_path, statistic_fields, statistics)
    write_csv(
        feasibility_path,
        list(feasibility[0].keys()),
        feasibility,
    )

    # 写后重读，保证磁盘序列化结果仍能通过相同校验。
    validate_episode_bank(
        [*read_jsonl(meta_path), *read_jsonl(oof_path)],
        assignment_rows,
        outer,
        meta,
        inner,
    )
    split_hash_path = split_dir / "split_artifact_hashes.json"
    hash_payload = {
        "version": config["version"],
        "status": "frozen",
        "seed": int(config["seed"]),
        "config_sha256": file_sha256(config_file),
        "split_artifact_hashes_sha256": file_sha256(split_hash_path),
        "code_version": {
            **_git_version(root),
            "generator_sha256": file_sha256(Path(__file__).resolve()),
        },
        "artifacts": {
            artifacts["meta_validation_bank"]: file_sha256(meta_path),
            artifacts["oof_bank"]: file_sha256(oof_path),
            artifacts["statistics"]: file_sha256(statistics_path),
            artifacts["feasibility"]: file_sha256(feasibility_path),
        },
    }
    write_json(output_dir / artifacts["hashes"], hash_payload)
    _render_report(
        root / config["report_path"],
        config,
        feasibility,
        statistics,
        meta_val_episodes,
        oof_episodes,
    )
    print(
        f"meta_val_episodes={len(meta_val_episodes)} "
        f"oof_episodes={len(oof_episodes)} "
        f"qualified_oof_products={len(oof_products)}"
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
