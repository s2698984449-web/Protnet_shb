"""从冻结的折内训练族准备 FOMAML Support 与 Query。

训练实际读哪一组 Episode 由 config/fomaml_data.yaml 的 active_profile 决定。
"""

from __future__ import annotations

from common.io import (
    file_sha256, json_sha256, load_config, load_verified_episode_groups, load_verified_group_split,
    load_verified_research_splits, named_config_path, project_root, read_jsonl, verify_frozen_episode_artifacts,
)
from data_proc.baseline_data import build_b0_index
from data_proc.detection import UnifiedDetectionAdapter
from data_proc.episodes import validate_episode_bank


_ROLES = ("support", "query_normal", "query_defect")
_MODE_ROLES = ("reference", "support", "query_normal", "query_defect")


def load_fomaml_data_config(profile: str | None = None) -> dict:
    """读取数据分组配置。profile 为空时使用 active_profile。"""
    root = load_config(named_config_path("fomaml_data.yaml"))
    name = profile or root["active_profile"]
    profiles = root["profiles"]
    if name not in profiles:
        raise KeyError(f"未知的 FOMAML 数据分组：{name}")
    selected = dict(profiles[name])
    selected["profile"] = name
    selected["active_profile"] = root["active_profile"]
    datasets = selected.get("dataset_ids")
    if datasets is not None:
        selected["dataset_ids"] = [str(item) for item in datasets]
        if not selected["dataset_ids"]:
            raise ValueError(f"数据分组 {name} 的 dataset_ids 不能为空列表")
    return selected


def filter_family_ids(family_ids: list[str], profile: dict) -> list[str]:
    """按配置里的数据集过滤产品族，顺序保持不变。"""
    datasets = profile.get("dataset_ids")
    if datasets is None:
        return list(family_ids)
    allowed = set(datasets)
    return [family for family in family_ids if family.split(":", 1)[0] in allowed]


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


def _prepare_mode_training_episode(episode, assignment, by_image, excluded) -> tuple[dict | None, dict]:
    """模式 Episode 允许 Query 含其他缺陷；Support 必须含当前模式。"""
    if not episode.get("support") or not episode.get("support_mode"):
        return None, {"excluded_image_ids": [], "incomplete_groups": [{"role": "support", "reason": "empty"}]}
    selected = episode["support_mode"]
    # 二分类检测：Query 的同模式与异模式缺陷都算正样本，不能只保留 Support 模式。
    positive = {selected}
    for unit in list(episode.get("support") or []) + list(episode.get("query_defect") or []):
        positive.update(label for label in (unit.get("defect_labels") or []) if label)
    result = {
        **episode,
        "bank": "mode",
        "n_way": 1,
        "episode_classes": sorted(positive),
        "support_mode": selected,
        "k_shot": len(episode["support"]),
        "q_defect_per_class": len(episode.get("query_defect") or []),
        "q_normal_groups": len(episode.get("query_normal") or []),
        "n_reference_groups": len(episode.get("reference") or []),
    }
    removed, incomplete = [], []
    seen_groups, seen_images = set(), set()
    for role in _MODE_ROLES:
        units = episode.get(role) or []
        result[role] = []
        for unit in units:
            group_id = unit["physical_group_id"]
            image_ids = list(unit["image_ids"])
            key = (episode["dataset_id"], episode["product_id"], group_id)
            frozen = assignment[key]
            if (
                frozen["family_id"] != episode["family_id"]
                or frozen["image_ids"].split("|") != image_ids
                or len(set(image_ids)) != len(image_ids)
                or group_id in seen_groups
                or seen_images.intersection(image_ids)
            ):
                raise AssertionError(f"模式 Episode {episode['episode_id']} {role} 物理组成员异常：{group_id}")
            expected = "R" if role == "reference" else role.upper()
            if frozen["episode_pool_role"] != expected:
                raise AssertionError(f"模式 Episode 越过角色池：{group_id}")
            seen_groups.add(group_id)
            seen_images.update(image_ids)
            removed.extend(image_id for image_id in image_ids if image_id in excluded)
            rows = [by_image[image_id] for image_id in image_ids if image_id not in excluded]
            if not rows:
                incomplete.append({"role": role, "physical_group_id": group_id, "reason": "all_views_excluded"})
                continue
            if role in ("query_normal", "reference") and any(not row["normal_flag"] for row in rows):
                raise AssertionError(f"模式 Episode {episode['episode_id']} {role} 混入缺陷视角")
            labels = {label for row in rows for label in row["defect_labels"].split("|") if label}
            if role == "support" and selected not in labels:
                incomplete.append({"role": role, "physical_group_id": group_id, "reason": "no_support_mode"})
                continue
            if role == "query_defect" and not labels:
                incomplete.append({"role": role, "physical_group_id": group_id, "reason": "no_defect_view"})
                continue
            result[role].append({
                **unit,
                "image_ids": [row["image_id"] for row in rows],
                "image_paths": sorted(row["image_path"] for row in rows),
                "mask_paths": sorted(row["mask_path"] for row in rows if row["mask_path"]),
                "visible_defect_labels": sorted(labels),
                "ignore_labels": sorted(labels - positive),
            })
    if not result["support"] or not result["query_defect"]:
        return None, {"excluded_image_ids": removed, "incomplete_groups": incomplete or [{"reason": "empty_support_or_query"}]}
    return result, {"excluded_image_ids": removed, "incomplete_groups": incomplete}


def load_mode_training_episodes(fold_id: int, train_family_ids: list[str], profile: dict):
    """从模式清单加载当前折可训练 Episode。只使用配置允许的数据集和源族。"""
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
        or frozen_train & (target | meta_val | outer_test) or requested & (target | meta_val | outer_test)
    ):
        raise ValueError(f"模式 FOMAML 折 {fold_id} 训练族与冻结边界不一致")

    mode_name = profile.get("mode_config") or "mode_episode.yaml"
    mode_config = load_config(named_config_path(mode_name))
    root = project_root()
    manifest_path = root / mode_config["output_dir"] / mode_config["manifest_name"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"模式 Episode 清单不存在，请先运行 prepare.py mode-episodes：{manifest_path}")
    raw = read_jsonl(manifest_path)
    adapter = UnifiedDetectionAdapter()
    # 与监督 θ₀ 核对时用整折源族索引；实际训练只读当前分组的族。
    fold_index = build_b0_index(adapter, groups, frozen_train, group_split)
    index = [row for row in fold_index if row["family_id"] in requested]
    by_image = {row["image_id"]: row for row in index}
    assignment = {
        (row["dataset_id"], row["product_id"], row["physical_group_id"]): row for row in groups
    }
    excluded = (
        set(group_split["excluded_official_train_image_ids"])
        if group_split["conflict_policy"] == "preserve_official_test" else set()
    )
    datasets = set(profile["dataset_ids"] or [])
    eligible, rejected, excluded_images = [], [], set()
    excluded_occurrences = 0
    for episode in sorted(raw, key=lambda item: item["episode_id"]):
        family = episode["family_id"]
        if family not in requested:
            continue
        if datasets and episode["dataset_id"] not in datasets:
            continue
        if profile.get("require_training_eligible", True) and not episode.get("training_eligible", True):
            continue
        if profile.get("only_exact") and episode.get("degraded"):
            continue
        if not profile.get("allow_degraded", True) and episode.get("degraded"):
            continue
        if family in target or family in meta_val or family in outer_test:
            raise AssertionError(f"模式 Episode 混入了禁止训练的族：{family}")
        prepared, audit = _prepare_mode_training_episode(episode, assignment, by_image, excluded)
        excluded_images.update(audit["excluded_image_ids"])
        excluded_occurrences += len(audit["excluded_image_ids"])
        if prepared is None or audit["incomplete_groups"]:
            rejected.append({"episode_id": episode["episode_id"], "groups": audit["incomplete_groups"]})
            continue
        eligible.append(prepared)
    covered = {episode["family_id"] for episode in eligible}
    if not eligible or covered != requested:
        raise ValueError(f"模式训练族没有可用 Episode：缺少 {sorted(requested - covered)}")

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
        "train_index_sha256": json_sha256(fold_index), "split_manifest_sha256": hashes["split_artifact_hashes_sha256"],
        "episode_manifest_sha256": file_sha256(manifest_path),
        "episode_bank_sha256": file_sha256(manifest_path),
        "eligible_episode_sha256": json_sha256(eligible),
        "eligible_episode_count": len(eligible), "eligible_episode_ids": [episode["episode_id"] for episode in eligible],
        "data_profile": profile["profile"],
        "episode_source": profile["episode_source"],
        "profile_train_index_sha256": json_sha256(index),
        "replaces_three_dataset_protocol": bool(profile.get("replaces_three_dataset_protocol")),
        "official_unsupervised_protocol": bool(profile.get("official_unsupervised_protocol")),
        "conflict_filter": {
            "policy": group_split["conflict_policy"], "excluded_image_ids": sorted(excluded_images),
            "excluded_view_occurrences": excluded_occurrences, "rejected_episodes": rejected,
        },
    }
    return adapter, eligible, scope


def load_training_episodes(fold_id: int, train_family_ids: list[str], profile: str | None = None):
    """统一训练 Episode 入口。分组由 fomaml_data.yaml 决定。"""
    selected = load_fomaml_data_config(profile)
    families = filter_family_ids(train_family_ids, selected)
    if not families:
        raise ValueError(f"数据分组 {selected['profile']} 在折 {fold_id} 没有可训练产品族")
    source = selected["episode_source"]
    if source == "frozen_oof_v1":
        adapter, episodes, scope = load_fomaml_episodes(fold_id, families)
    elif source == "mode_manifest":
        adapter, episodes, scope = load_mode_training_episodes(fold_id, families, selected)
    else:
        raise ValueError(f"未知的 episode_source：{source}")
    scope["data_profile"] = selected["profile"]
    scope["episode_source"] = source
    scope["profile_dataset_ids"] = selected.get("dataset_ids")
    scope["replaces_three_dataset_protocol"] = bool(selected.get("replaces_three_dataset_protocol"))
    scope["official_unsupervised_protocol"] = bool(selected.get("official_unsupervised_protocol"))
    scope["bank_tasks_per_family"] = selected.get("bank_tasks_per_family")
    return adapter, episodes, scope


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
