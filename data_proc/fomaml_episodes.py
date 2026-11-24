"""从冻结的折内训练族准备 FOMAML Support 与 Query。"""

from __future__ import annotations

from common.io import (
    file_sha256, json_sha256, load_config, load_verified_episode_groups, load_verified_group_split,
    load_verified_research_splits, named_config_path, project_root, read_jsonl, verify_frozen_episode_artifacts,
)
from data_proc.baseline_data import build_b0_index
from data_proc.detection import UnifiedDetectionAdapter
from data_proc.episodes import validate_episode_bank


_ROLES = ("support", "query_normal", "query_defect")


class InsufficientEpisodeAnnotations(ValueError):
    """共享 Mask 筛选后，至少一个正物理组失去了全部有效缺陷框。"""


def _training_episode(episode, assignment, by_image, excluded, group_counts) -> tuple[dict, dict]:
    if (
        episode["n_way"] != 1 or len(episode["episode_classes"]) != 1
        or episode["k_shot"] != group_counts["support"]
        or episode["q_defect_per_class"] != group_counts["query_defect"]
        or episode["q_normal_groups"] != group_counts["query_normal"]
    ):
        raise ValueError("B0 二分类 FOMAML 需要 1-way 及与冻结协议一致的 Support、Query 预算")

    result = dict(episode)
    removed, incomplete = [], []
    selected = episode["episode_classes"][0]
    seen_groups, seen_images = set(), set()
    for role in _ROLES:
        units = episode[role]
        if len(units) != group_counts[role]:
            raise AssertionError(f"Episode {episode['episode_id']} {role} 组数不符合协议")
        result[role] = []
        for unit in units:
            group_id = unit["physical_group_id"]
            image_ids = unit["image_ids"]
            key = (episode["dataset_id"], episode["product_id"], group_id)
            frozen = assignment[key]
            if (
                frozen["family_id"] != episode["family_id"] or frozen["image_ids"].split("|") != image_ids
                or len(set(image_ids)) != len(image_ids) or group_id in seen_groups or seen_images.intersection(image_ids)
            ):
                raise AssertionError(f"Episode {episode['episode_id']} {role} 物理组成员异常：{group_id}")
            seen_groups.add(group_id)
            seen_images.update(image_ids)
            removed.extend(image_id for image_id in image_ids if image_id in excluded)
            rows = [by_image[image_id] for image_id in image_ids if image_id not in excluded]
            if not rows:
                incomplete.append({"role": role, "physical_group_id": group_id, "reason": "all_views_excluded"})
                continue
            if role == "query_normal" and any(not row["normal_flag"] for row in rows):
                raise AssertionError(f"Episode {episode['episode_id']} 正常 Query 混入缺陷视角")
            if role != "query_normal" and not any(selected in row["defect_labels"].split("|") for row in rows):
                incomplete.append({"role": role, "physical_group_id": group_id, "reason": "no_selected_defect_view"})
                continue
            labels = {label for row in rows for label in row["defect_labels"].split("|") if label}
            result[role].append({
                **unit,
                "image_ids": [row["image_id"] for row in rows],
                "image_paths": sorted(row["image_path"] for row in rows),
                "mask_paths": sorted(row["mask_path"] for row in rows if row["mask_path"]),
                "visible_defect_labels": sorted(labels), "ignore_labels": sorted(labels - {selected}),
            })
    return result, {"excluded_image_ids": removed, "incomplete_groups": incomplete}


def load_fomaml_episodes(fold_id: int, train_family_ids: list[str]):
    """校验折级边界和冻结 Bank，返回适配器、可训练 Episode 及来源记录。"""
    group_split = load_verified_group_split()
    outer, meta, inner = load_verified_research_splits()
    groups = load_verified_episode_groups()
    hashes = verify_frozen_episode_artifacts()
    folds = [fold for fold in inner["folds"] if fold["fold_id"] == fold_id]
    if len(folds) != 1:
        raise ValueError(f"找不到唯一的冻结 OOF 折 {fold_id}")
    fold = folds[0]
    frozen_train = set(fold["theta0_train_family_ids"])
    target = set(fold["target_family_ids"])
    meta_val = set(meta["meta_val_family_ids"])
    outer_test = set(outer["outer_test_family_ids"])
    requested = set(train_family_ids)
    if (
        not requested or len(requested) != len(train_family_ids) or not requested <= frozen_train
        or frozen_train != set(fold["train_family_ids"]) or target != set(fold["val_family_ids"])
        or frozen_train & (target | meta_val | outer_test)
    ):
        raise ValueError(f"FOMAML 折 {fold_id} 训练族与冻结 θ₀ 或留出边界不一致")

    root = project_root()
    episode_config = load_config(named_config_path("episode.yaml"))
    protocol = episode_config["protocol"]
    group_counts = {"support": protocol["k_shot"], "query_normal": protocol["q_normal_groups"],
                    "query_defect": protocol["q_defect_per_class"]}
    episode_dir = root / episode_config["output_dir"]
    bank_name = episode_config["artifacts"]["oof_bank"]
    episodes = read_jsonl(episode_dir / bank_name)
    validate_episode_bank(episodes, groups, outer, meta, inner)
    family_folds = {
        family: part["fold_id"] for part in inner["folds"] for family in part["target_family_ids"]
    }
    adapter = UnifiedDetectionAdapter()
    index = build_b0_index(adapter, groups, requested, group_split)
    by_image = {row["image_id"]: row for row in index}
    assignment = {
        (row["dataset_id"], row["product_id"], row["physical_group_id"]): row for row in groups
    }
    excluded = (
        set(group_split["excluded_official_train_image_ids"])
        if group_split["conflict_policy"] == "preserve_official_test" else set()
    )
    eligible, rejected, excluded_images = [], [], set()
    excluded_occurrences = 0
    for episode in sorted(episodes, key=lambda item: item["episode_id"]):
        family = episode["family_id"]
        if episode["bank"] != "oof" or episode["inner_target_fold"] != family_folds.get(family):
            raise AssertionError(f"OOF Episode 与冻结折不一致：{episode['episode_id']}")
        if family not in requested:
            continue
        if episode["inner_target_fold"] == fold_id:
            raise AssertionError(f"FOMAML Episode 混入当前折目标族：{family}")
        prepared, audit = _training_episode(episode, assignment, by_image, excluded, group_counts)
        excluded_images.update(audit["excluded_image_ids"])
        excluded_occurrences += len(audit["excluded_image_ids"])
        if audit["incomplete_groups"]:
            rejected.append({"episode_id": episode["episode_id"], "groups": audit["incomplete_groups"]})
        else:
            eligible.append(prepared)
    covered = {episode["family_id"] for episode in eligible}
    if covered != requested:
        raise ValueError(f"FOMAML 训练族没有可用的完整 Episode：{sorted(requested - covered)}")

    data_config_path = root / adapter.config["data_config"]
    data_config = load_config(data_config_path)
    audit_dir = root / data_config["output_dir"]
    scope = {
        "fold_id": fold_id, "fold_sha256": json_sha256(fold),
        "train_family_ids": sorted(requested), "theta0_train_family_ids": sorted(frozen_train),
        "delta_source_family_ids": sorted(fold["delta_source_family_ids"]), "oof_target_family_ids": sorted(target),
        "meta_val_family_ids": sorted(meta_val), "outer_test_family_ids": sorted(outer_test),
        "data_config_sha256": file_sha256(data_config_path), "manifest_sha256": file_sha256(audit_dir / "manifest.csv"),
        "group_split_sha256": file_sha256(audit_dir / data_config["physical_group"]["artifacts"]["group_split"]),
        "train_index_sha256": json_sha256(index), "split_manifest_sha256": hashes["split_artifact_hashes_sha256"],
        "episode_manifest_sha256": file_sha256(episode_dir / episode_config["artifacts"]["hashes"]),
        "episode_bank_sha256": hashes["artifacts"][bank_name], "eligible_episode_sha256": json_sha256(eligible),
        "eligible_episode_count": len(eligible), "eligible_episode_ids": [episode["episode_id"] for episode in eligible],
        "conflict_filter": {
            "policy": group_split["conflict_policy"], "excluded_image_ids": sorted(excluded_images),
            "excluded_view_occurrences": excluded_occurrences, "rejected_episodes": rejected,
        },
    }
    return adapter, eligible, scope


def episode_records(adapter, episode) -> tuple[list[dict], list[dict]]:
    """按需读取共享检测标注，保住全部有意义的 Support 与 Query 视角。"""
    records = adapter.build_episode_records(episode, roles=_ROLES)
    present = {
        (record["split_role"], record["physical_group_id"])
        for record in records if record["annotations"]
    }
    required = {
        (role.upper(), unit["physical_group_id"])
        for role in ("support", "query_defect") for unit in episode[role]
    }
    if missing := required - present:
        raise InsufficientEpisodeAnnotations(
            f"Episode {episode['episode_id']} 正物理组在 Mask 筛选后无有效框：{sorted(missing)}"
        )
    usable = [
        record for record in records
        if record["normal_flag"] or record["annotations"] or record["ignore_annotations"]
    ]
    support = [record for record in usable if record["split_role"] == "SUPPORT"]
    query = [record for record in usable if record["split_role"] != "SUPPORT"]
    for key in ("image_id", "physical_group_id"):
        if {record[key] for record in support}.intersection(record[key] for record in query):
            raise AssertionError(f"Episode {episode['episode_id']} Support 与 Query 的 {key} 重叠")
    return support, query
