"""S1：按产品族生成外层留出和内层 OOF，不读取图像。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.io import (
    assert_disjoint_groups,
    file_sha256,
    load_config,
    load_verified_episode_groups,
    load_verified_group_split,
    load_verified_research_splits,
    named_config_path,
    project_root,
    write_csv,
    write_json,
)


def _stable_rank(seed: int, *parts: str) -> str:
    payload = "|".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_family_rows(product_rows: list[dict], config: dict) -> list[dict]:
    explicit: dict[tuple[str, str], str] = {}
    for dataset_id, families in config.get("product_family_map", {}).items():
        for family_name, products in families.items():
            for product_id in products:
                key = (dataset_id, product_id)
                if key in explicit:
                    raise ValueError(f"产品被多个产品族重复声明：{key}")
                explicit[key] = family_name

    known_products = {
        (row["dataset_id"], row["product_id"]) for row in product_rows
    }
    unknown = sorted(set(explicit) - known_products)
    if unknown:
        raise ValueError(f"产品族配置引用不存在的产品：{unknown}")

    rows = []
    for product in sorted(
        product_rows, key=lambda row: (row["dataset_id"], row["product_id"])
    ):
        dataset_id = product["dataset_id"]
        product_id = product["product_id"]
        family_name = explicit.get((dataset_id, product_id), product_id)
        rows.append(
            {
                "dataset_id": dataset_id,
                "product_id": product_id,
                "family_name": family_name,
                "family_id": f"{dataset_id}:{family_name}",
                "n_images": int(product["n_images"]),
                "n_normal_images": int(product["n_normal_images"]),
                "n_defect_images": int(product["n_defect_images"]),
                "n_defect_classes": int(product["n_defect_classes"]),
            }
        )
    return rows


def _select_stratified_holdout(
    family_ids_by_dataset: dict[str, list[str]],
    seed: int,
    fraction: float,
    minimum: int,
    scope: str,
) -> set[str]:
    selected: set[str] = set()
    for dataset_id, family_ids in sorted(family_ids_by_dataset.items()):
        if len(family_ids) < 2:
            raise ValueError(f"{dataset_id} 产品族不足2个，不能做 {scope} 留出")
        n_test = max(minimum, math.floor(len(family_ids) * fraction + 0.5))
        n_test = min(n_test, len(family_ids) - 1)
        ranked = sorted(
            family_ids,
            key=lambda family_id: _stable_rank(
                seed, scope, dataset_id, family_id
            ),
        )
        selected.update(ranked[:n_test])
    return selected


def _assign_inner_folds(
    outer_train_by_dataset: dict[str, list[str]],
    seed: int,
    n_folds: int,
) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("inner folds 必须至少为2")
    assignment: dict[str, int] = {}
    total_counts = [0] * n_folds
    dataset_counts: dict[str, list[int]] = {}
    for dataset_id, family_ids in sorted(outer_train_by_dataset.items()):
        if len(family_ids) < n_folds:
            raise ValueError(
                f"{dataset_id} 外层训练产品族 {len(family_ids)} 个，少于 {n_folds} 折"
            )
        dataset_counts[dataset_id] = [0] * n_folds
        ranked = sorted(
            family_ids,
            key=lambda family_id: _stable_rank(
                seed, "inner", dataset_id, family_id
            ),
        )
        for family_id in ranked:
            fold_id = min(
                range(n_folds),
                key=lambda fold: (
                    dataset_counts[dataset_id][fold],
                    total_counts[fold],
                    _stable_rank(seed, "fold_tie", family_id, str(fold)),
                ),
            )
            assignment[family_id] = fold_id
            dataset_counts[dataset_id][fold_id] += 1
            total_counts[fold_id] += 1
    return assignment


def _validate_realiad_groups(
    family_rows: list[dict],
    outer_test_family_ids: set[str],
    group_split: dict,
) -> dict[str, int]:
    product_to_family = {
        row["product_id"]: row["family_id"]
        for row in family_rows
        if row["dataset_id"] == "realiad"
    }
    by_product = group_split["by_product"]
    if set(product_to_family) != set(by_product):
        missing = sorted(set(product_to_family) - set(by_product))
        extra = sorted(set(by_product) - set(product_to_family))
        raise AssertionError(
            f"Real-IAD 产品与物理组映射不一致，missing={missing}, extra={extra}"
        )

    outer_train_groups: set[str] = set()
    outer_test_groups: set[str] = set()
    all_groups: set[str] = set()
    for product_id, split_slot in by_product.items():
        product_groups = set(split_slot["train_group_ids"]) | set(
            split_slot["test_group_ids"]
        )
        if all_groups & product_groups:
            raise AssertionError("Real-IAD 物理组出现在多个产品中")
        all_groups.update(product_groups)
        family_id = product_to_family[product_id]
        if family_id in outer_test_family_ids:
            outer_test_groups.update(product_groups)
        else:
            outer_train_groups.update(product_groups)
    assert_disjoint_groups(outer_train_groups, outer_test_groups)
    return {
        "n_realiad_physical_groups": len(all_groups),
        "n_outer_train_physical_groups": len(outer_train_groups),
        "n_outer_test_physical_groups": len(outer_test_groups),
    }


def _make_statistics(
    family_rows: list[dict],
    outer_test: set[str],
    meta_val: set[str],
    inner_assignment: dict[str, int],
    n_folds: int,
) -> list[dict]:
    datasets = sorted({row["dataset_id"] for row in family_rows})
    rows = []
    for dataset_id in [*datasets, "all"]:
        subset = [
            row
            for row in family_rows
            if dataset_id == "all" or row["dataset_id"] == dataset_id
        ]
        families = {row["family_id"] for row in subset}
        products = {
            f"{row['dataset_id']}:{row['product_id']}" for row in subset
        }
        train_families = families - outer_test
        test_families = families & outer_test
        meta_val_families = train_families & meta_val
        meta_train_families = train_families - meta_val
        stat = {
            "dataset_id": dataset_id,
            "n_products": len(products),
            "n_families": len(families),
            "n_outer_train_families": len(train_families),
            "n_outer_test_families": len(test_families),
            "n_outer_train_products": sum(
                row["family_id"] in train_families for row in subset
            ),
            "n_outer_test_products": sum(
                row["family_id"] in test_families for row in subset
            ),
            "n_meta_train_families": len(meta_train_families),
            "n_meta_val_families": len(meta_val_families),
            "n_meta_train_products": sum(
                row["family_id"] in meta_train_families for row in subset
            ),
            "n_meta_val_products": sum(
                row["family_id"] in meta_val_families for row in subset
            ),
        }
        for fold_id in range(n_folds):
            stat[f"inner_fold_{fold_id}_val_families"] = sum(
                inner_assignment.get(family_id) == fold_id
                for family_id in meta_train_families
            )
        rows.append(stat)
    return rows


def _group_id(row: dict) -> str:
    if row["dataset_id"] == "realiad":
        return f"{row['product_id']}_{row['physical_sample_id']}"
    return row["image_id"]


def _build_episode_group_assignments(
    manifest_rows: list[dict],
    family_rows: list[dict],
    inner_payload: dict,
    episode_config: dict,
) -> tuple[list[dict], list[dict]]:
    """把每个产品内部的完整物理组分到 R/Support/Query 池。"""
    hierarchy = {
        (row["dataset_id"], row["product_id"]): row for row in family_rows
    }
    groups: dict[tuple[str, str, str], dict] = {}
    for row in manifest_rows:
        key = (row["dataset_id"], row["product_id"], _group_id(row))
        slot = groups.setdefault(
            key,
            {
                "dataset_id": row["dataset_id"],
                "product_id": row["product_id"],
                "physical_group_id": key[2],
                "image_ids": [],
                "image_paths": [],
                "mask_paths": [],
                "normal": False,
                "labels": set(),
            },
        )
        slot["image_ids"].append(row["image_id"])
        slot["image_paths"].append(row["image_path"])
        if row["mask_path"]:
            slot["mask_paths"].append(row["mask_path"])
        slot["normal"] = slot["normal"] or row["normal_flag"] == "1"
        slot["labels"].update(
            label for label in row["defect_labels"].split("|") if label
        )

    by_product: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for group in groups.values():
        by_product[(group["dataset_id"], group["product_id"])].append(group)

    seed = int(episode_config["seed"])
    min_side = int(episode_config["min_groups_per_side"])
    assignment_rows: list[dict] = []
    statistics: list[dict] = []
    fold_by_family = {
        family_id: fold["fold_id"]
        for fold in inner_payload["folds"]
        for family_id in fold["val_family_ids"]
    }

    for product_key, product_groups in sorted(by_product.items()):
        hierarchy_row = hierarchy[product_key]
        family_id = hierarchy_row["family_id"]
        normal_only = [
            group for group in product_groups if group["normal"] and not group["labels"]
        ]
        defect_groups = [group for group in product_groups if group["labels"]]
        if len(normal_only) < 2:
            raise AssertionError(f"{product_key} 纯正常物理组不足2个，无法隔离 R 与正常 Query")
        if len(defect_groups) < 2:
            raise AssertionError(
                f"{product_key} 缺陷物理组不足2个，无法隔离 Support 与 Query"
            )

        normal_ranked = sorted(
            normal_only,
            key=lambda group: _stable_rank(
                seed, "normal_pool", family_id, product_key[1], group["physical_group_id"]
            ),
        )
        n_reference = round(
            len(normal_ranked) * float(episode_config["normal_reference_fraction"])
        )
        n_reference = min(
            len(normal_ranked) - min_side, max(min_side, n_reference)
        )
        reference_ids = {
            group["physical_group_id"] for group in normal_ranked[:n_reference]
        }

        defect_ranked = sorted(
            defect_groups,
            key=lambda group: _stable_rank(
                seed, "defect_pool", family_id, product_key[1], group["physical_group_id"]
            ),
        )
        n_support = round(
            len(defect_ranked) * float(episode_config["defect_support_fraction"])
        )
        n_support = min(
            len(defect_ranked) - min_side, max(min_side, n_support)
        )
        support_ids = {
            group["physical_group_id"] for group in defect_ranked[:n_support]
        }

        labels = sorted(
            {label for group in defect_groups for label in group["labels"]}
        )
        for _ in range(max(1, len(labels) * 2)):
            changed = False
            for label in labels:
                label_groups = [
                    group for group in defect_ranked if label in group["labels"]
                ]
                if len(label_groups) < 2:
                    continue
                support_for_label = [
                    group
                    for group in label_groups
                    if group["physical_group_id"] in support_ids
                ]
                query_for_label = [
                    group
                    for group in label_groups
                    if group["physical_group_id"] not in support_ids
                ]
                if not support_for_label:
                    support_ids.add(query_for_label[0]["physical_group_id"])
                    changed = True
                elif not query_for_label:
                    support_ids.remove(support_for_label[-1]["physical_group_id"])
                    changed = True
            if not changed:
                break

        for label in labels:
            label_ids = {
                group["physical_group_id"]
                for group in defect_ranked
                if label in group["labels"]
            }
            if len(label_ids) >= 2:
                if not label_ids & support_ids or not label_ids - support_ids:
                    raise AssertionError(
                        f"{product_key}/{label} 无法同时覆盖 Support 与 Query"
                    )

        if hierarchy_row["outer_split"] == "test":
            stage = "outer_test"
            source_mask_key = "meta_train_full"
        elif hierarchy_row["meta_split"] == "validation":
            stage = "meta_validation"
            source_mask_key = "meta_train_full"
        else:
            stage = "inner_oof"
            source_mask_key = f"fold_{fold_by_family[family_id]}"

        role_ids: dict[str, set[str]] = {
            "R": reference_ids,
            "QUERY_NORMAL": {
                group["physical_group_id"]
                for group in normal_only
                if group["physical_group_id"] not in reference_ids
            },
            "SUPPORT": support_ids,
            "QUERY_DEFECT": {
                group["physical_group_id"]
                for group in defect_groups
                if group["physical_group_id"] not in support_ids
            },
        }
        role_names = list(role_ids)
        for left_index, left in enumerate(role_names):
            for right in role_names[left_index + 1 :]:
                assert_disjoint_groups(role_ids[left], role_ids[right])
        assigned = set().union(*role_ids.values())
        expected = {group["physical_group_id"] for group in product_groups}
        if assigned != expected:
            raise AssertionError(f"{product_key} 存在未分配或重复物理组")

        role_by_id = {
            group_id: role for role, group_ids in role_ids.items() for group_id in group_ids
        }
        support_class_counts = {
            label: sum(
                group["physical_group_id"] in support_ids and label in group["labels"]
                for group in defect_groups
            )
            for label in labels
        }
        query_class_counts = {
            label: sum(
                group["physical_group_id"] not in support_ids and label in group["labels"]
                for group in defect_groups
            )
            for label in labels
        }
        statistics.append(
            {
                "dataset_id": product_key[0],
                "product_id": product_key[1],
                "family_id": family_id,
                "evaluation_stage": stage,
                "source_mask_key": source_mask_key,
                "n_groups": len(product_groups),
                "n_r_groups": len(role_ids["R"]),
                "n_support_groups": len(role_ids["SUPPORT"]),
                "n_query_normal_groups": len(role_ids["QUERY_NORMAL"]),
                "n_query_defect_groups": len(role_ids["QUERY_DEFECT"]),
                "n_defect_classes": len(labels),
                "min_support_groups_per_class": min(support_class_counts.values()),
                "min_query_groups_per_class": min(query_class_counts.values()),
            }
        )
        for group in sorted(
            product_groups, key=lambda item: item["physical_group_id"]
        ):
            if group["labels"] and group["normal"]:
                group_kind = "mixed_normal_defect"
            elif group["labels"]:
                group_kind = "defect_only"
            else:
                group_kind = "normal_only"
            assignment_rows.append(
                {
                    "dataset_id": group["dataset_id"],
                    "family_id": family_id,
                    "product_id": group["product_id"],
                    "physical_group_id": group["physical_group_id"],
                    "outer_split": hierarchy_row["outer_split"],
                    "meta_split": hierarchy_row["meta_split"],
                    "evaluation_stage": stage,
                    "inner_target_fold": hierarchy_row["inner_val_fold"],
                    "source_mask_key": source_mask_key,
                    "episode_pool_role": role_by_id[group["physical_group_id"]],
                    "group_kind": group_kind,
                    "defect_labels": "|".join(sorted(group["labels"])),
                    "n_images": len(group["image_ids"]),
                    "n_masks": len(group["mask_paths"]),
                    "image_ids": "|".join(sorted(group["image_ids"])),
                    "image_paths": "|".join(sorted(group["image_paths"])),
                    "mask_paths": "|".join(sorted(group["mask_paths"])),
                }
            )
    return assignment_rows, statistics


def render_split_report(
    path: Path,
    config: dict,
    family_rows: list[dict],
    outer: dict,
    meta: dict,
    inner: dict,
    statistics: list[dict],
    episode_statistics: list[dict],
) -> None:
    family_to_products: dict[str, list[str]] = defaultdict(list)
    for row in family_rows:
        family_to_products[row["family_id"]].append(row["product_id"])
    lines = [
        "# v1 外层产品族留出与内层 OOF",
        "",
        f"日期：{date.today().isoformat()}",
        f"随机种子：{config['seed']}",
        "",
        "## 协议",
        "",
        "外层按数据集分层，以稳定哈希选择约20%的产品族作为完全封存测试组。选择过程不读取模型结果、缺陷难度或测试指标。",
        "",
        "VisA 保守合并 `pcb1–4` 为一个产品族，`macaroni1/2` 为一个产品族。其他对象各自成族。Real-IAD 整个产品类别随外层产品族移动，类别内 `category + Sxxxx` 物理组绝不拆分。",
        "",
        f"外层训练产品族：{len(outer['outer_train_family_ids'])}；外层测试产品族：{len(outer['outer_test_family_ids'])}。",
        f"Meta-Train：{len(meta['meta_train_family_ids'])} 个产品族；Meta-Validation：{len(meta['meta_val_family_ids'])} 个产品族。",
        f"内层：{config['inner']['folds']} 折；每个 Meta-Train 产品族恰好作为一次OOF目标折。",
        f"候选源池主方案：`{config['candidate_pool_policy']}`。",
        "",
        "## 外层封存产品族",
        "",
    ]
    for family_id in outer["outer_test_family_ids"]:
        lines.append(
            f"- `{family_id}`：{', '.join(sorted(family_to_products[family_id]))}"
        )
    lines.extend(["", "## Meta-Validation 产品族", ""])
    for family_id in meta["meta_val_family_ids"]:
        lines.append(
            f"- `{family_id}`：{', '.join(sorted(family_to_products[family_id]))}"
        )
    lines.extend(["", "## 内层验证折", ""])
    for fold in inner["folds"]:
        lines.append(
            f"- fold {fold['fold_id']}：{len(fold['val_family_ids'])} 个验证产品族；"
            f"{len(fold['train_family_ids'])} 个训练产品族。"
        )
    lines.extend(
        [
            "",
            "## 硬性校验",
            "",
            "- 外层训练/测试产品族交集为0。",
            "- Meta-Train/Meta-Validation 产品族交集为0，二者完整覆盖 Outer Train。",
            "- Meta-Validation 不参与 θ₀、Δ、选择器梯度和内层OOF。",
            "- 外层测试产品族未进入任何内层折。",
            "- 每个 Meta-Train 产品族恰好作为一次内层OOF目标。",
            "- 每个内层折的训练/验证产品族交集为0。",
            "- 每折 `source_available_mask` 与真实源任务池一致；缺失U标签不补0。",
            "- Real-IAD 外层训练/测试物理组交集为0。",
            "- 每个产品内 R、Support、Query-Normal、Query-Defect 四个物理组池两两不相交并完整覆盖。",
            "",
            f"已为 {len(episode_statistics)} 个产品生成物理组级Episode池。当前文件冻结池边界，不等于具体K-shot Episode。",
            "",
            "下一步从这些池中生成小规模固定Episode，并把对应折的 `source_available_mask` 原样写入每条Episode。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_code_version(root: Path) -> dict:
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
    return {
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "generator_sha256": file_sha256(Path(__file__).resolve()),
    }


def _verify_frozen_artifacts(
    root: Path,
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
            "划分协议已冻结，但当前 split.yaml 与冻结版本不同。"
            "请提高 version 并使用新的 output_dir，禁止覆盖 v1。"
        )
    for filename, expected_hash in payload["artifacts"].items():
        artifact_path = output_dir / filename
        if not artifact_path.is_file():
            raise RuntimeError(f"冻结划分产物缺失：{artifact_path}")
        if file_sha256(artifact_path) != expected_hash:
            raise RuntimeError(f"冻结划分产物哈希不匹配：{artifact_path}")
    print(f"frozen_split_verified {hash_path}")
    return True


def _freeze_artifacts(
    root: Path,
    config_file: Path,
    output_dir: Path,
    config: dict,
) -> None:
    artifacts = config["artifacts"]
    frozen_keys = [
        "family_manifest",
        "outer",
        "meta",
        "inner",
        "episode_groups",
        "episode_statistics",
    ]
    hashes = {
        artifacts[key]: file_sha256(output_dir / artifacts[key])
        for key in frozen_keys
    }
    payload = {
        "version": config["version"],
        "status": "frozen",
        "seed": int(config["seed"]),
        "created_on": date.today().isoformat(),
        "config_path": str(config_file.relative_to(root)).replace("\\", "/"),
        "config_sha256": file_sha256(config_file),
        "code_version": _git_code_version(root),
        "artifacts": hashes,
        "rule": (
            "修改数据划分时必须提高版本并写入新的 output_dir；"
            "禁止覆盖本清单约束的文件。"
        ),
    }
    write_json(output_dir / artifacts["hashes"], payload)


def main() -> None:
    root = project_root()
    config_file = named_config_path("split.yaml")
    config = load_config(config_file)
    audit_dir = root / config["input_dir"]
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.get("freeze_artifacts", False) and _verify_frozen_artifacts(
        root, config_file, output_dir, config
    ):
        return

    product_rows = _read_csv(audit_dir / "product_counts.csv")
    family_rows = _build_family_rows(product_rows, config)
    family_ids_by_dataset: dict[str, list[str]] = defaultdict(list)
    for row in family_rows:
        family_ids_by_dataset[row["dataset_id"]].append(row["family_id"])
    family_ids_by_dataset = {
        dataset_id: sorted(set(family_ids))
        for dataset_id, family_ids in family_ids_by_dataset.items()
    }

    seed = int(config["seed"])
    outer_config = config["outer"]
    outer_test = _select_stratified_holdout(
        family_ids_by_dataset,
        seed,
        float(outer_config["test_fraction"]),
        int(outer_config["min_test_families_per_dataset"]),
        "outer",
    )
    all_families = {
        family_id
        for family_ids in family_ids_by_dataset.values()
        for family_id in family_ids
    }
    outer_train = all_families - outer_test
    assert_disjoint_groups(outer_train, outer_test)

    outer_train_by_dataset = {
        dataset_id: sorted(set(family_ids) - outer_test)
        for dataset_id, family_ids in family_ids_by_dataset.items()
    }
    meta_config = config["meta_validation"]
    meta_val = _select_stratified_holdout(
        outer_train_by_dataset,
        seed,
        float(meta_config["fraction"]),
        int(meta_config["min_families_per_dataset"]),
        "meta_validation",
    )
    meta_train = outer_train - meta_val
    assert_disjoint_groups(meta_train, meta_val)
    if meta_train | meta_val != outer_train:
        raise AssertionError("Meta 划分未完整覆盖 Outer Train")

    meta_train_by_dataset = {
        dataset_id: sorted(set(family_ids) - meta_val)
        for dataset_id, family_ids in outer_train_by_dataset.items()
    }
    n_folds = int(config["inner"]["folds"])
    inner_assignment = _assign_inner_folds(
        meta_train_by_dataset, seed, n_folds
    )

    for row in family_rows:
        row["outer_split"] = (
            "test" if row["family_id"] in outer_test else "train"
        )
        row["meta_split"] = (
            ""
            if row["family_id"] in outer_test
            else ("validation" if row["family_id"] in meta_val else "train")
        )
        row["inner_val_fold"] = (
            ""
            if row["family_id"] not in meta_train
            else inner_assignment[row["family_id"]]
        )

    group_split_path = audit_dir / "group_split.json"
    group_split = load_verified_group_split(group_split_path)
    physical_counts = _validate_realiad_groups(
        family_rows, outer_test, group_split
    )

    outer_payload = {
        "version": config["version"],
        "status": "frozen",
        "seed": seed,
        "selection": outer_config["selection"],
        "test_fraction": float(outer_config["test_fraction"]),
        "config_hash": file_sha256(config_file),
        "physical_group_split_hash": file_sha256(group_split_path),
        "outer_train_family_ids": sorted(outer_train),
        "outer_test_family_ids": sorted(outer_test),
        "outer_train_product_ids": sorted(
            f"{row['dataset_id']}:{row['product_id']}"
            for row in family_rows
            if row["family_id"] in outer_train
        ),
        "outer_test_product_ids": sorted(
            f"{row['dataset_id']}:{row['product_id']}"
            for row in family_rows
            if row["family_id"] in outer_test
        ),
        "by_dataset": {
            dataset_id: {
                "train_family_ids": sorted(set(family_ids) - outer_test),
                "test_family_ids": sorted(set(family_ids) & outer_test),
            }
            for dataset_id, family_ids in sorted(family_ids_by_dataset.items())
        },
        "physical_group_counts": physical_counts,
    }
    meta_payload = {
        "version": config["version"],
        "status": "frozen",
        "seed": seed,
        "selection": meta_config["selection"],
        "validation_fraction": float(meta_config["fraction"]),
        "outer_test_family_ids": sorted(outer_test),
        "meta_train_family_ids": sorted(meta_train),
        "meta_val_family_ids": sorted(meta_val),
        "meta_train_product_ids": sorted(
            f"{row['dataset_id']}:{row['product_id']}"
            for row in family_rows
            if row["family_id"] in meta_train
        ),
        "meta_val_product_ids": sorted(
            f"{row['dataset_id']}:{row['product_id']}"
            for row in family_rows
            if row["family_id"] in meta_val
        ),
        "usage": {
            "meta_train": "theta0_delta_utility_selector_training",
            "meta_validation": "model_selection_null_calibration_hyperparameters",
            "outer_test": "sealed_final_evaluation",
        },
    }
    source_universe = ["Null", *sorted(meta_train)]
    inner_folds = []
    for fold_id in range(n_folds):
        val_families = {
            family_id
            for family_id, assigned_fold in inner_assignment.items()
            if assigned_fold == fold_id
        }
        train_families = meta_train - val_families
        assert_disjoint_groups(train_families, val_families)
        source_mask = [
            int(action == "Null" or action in train_families)
            for action in source_universe
        ]
        inner_folds.append(
            {
                "fold_id": fold_id,
                "theta0_train_family_ids": sorted(train_families),
                "delta_source_family_ids": sorted(train_families),
                "target_family_ids": sorted(val_families),
                "train_family_ids": sorted(train_families),
                "val_family_ids": sorted(val_families),
                "train_product_ids": sorted(
                    f"{row['dataset_id']}:{row['product_id']}"
                    for row in family_rows
                    if row["family_id"] in train_families
                ),
                "val_product_ids": sorted(
                    f"{row['dataset_id']}:{row['product_id']}"
                    for row in family_rows
                    if row["family_id"] in val_families
                ),
                "candidate_source_family_ids": sorted(train_families),
                "source_available_mask": source_mask,
                "missing_utility_policy": "masked_not_zero",
            }
        )
    inner_payload = {
        "version": config["version"],
        "status": "frozen",
        "seed": seed,
        "n_folds": n_folds,
        "assignment": config["inner"]["assignment"],
        "candidate_pool_policy": config["candidate_pool_policy"],
        "outer_test_family_ids": sorted(outer_test),
        "meta_val_family_ids": sorted(meta_val),
        "source_universe_action_ids": source_universe,
        "folds": inner_folds,
    }

    artifacts = config["artifacts"]
    family_fields = [
        "dataset_id",
        "product_id",
        "family_name",
        "family_id",
        "outer_split",
        "meta_split",
        "inner_val_fold",
        "n_images",
        "n_normal_images",
        "n_defect_images",
        "n_defect_classes",
    ]
    statistics = _make_statistics(
        family_rows, outer_test, meta_val, inner_assignment, n_folds
    )
    manifest_rows = _read_csv(audit_dir / "manifest.csv")
    episode_rows, episode_statistics = _build_episode_group_assignments(
        manifest_rows,
        family_rows,
        inner_payload,
        config["episode_groups"],
    )
    write_csv(
        output_dir / artifacts["family_manifest"],
        family_fields,
        family_rows,
    )
    write_json(output_dir / artifacts["outer"], outer_payload)
    write_json(output_dir / artifacts["meta"], meta_payload)
    write_json(output_dir / artifacts["inner"], inner_payload)
    write_csv(
        output_dir / artifacts["statistics"],
        list(statistics[0].keys()),
        statistics,
    )
    episode_fields = [
        "dataset_id",
        "family_id",
        "product_id",
        "physical_group_id",
        "outer_split",
        "meta_split",
        "evaluation_stage",
        "inner_target_fold",
        "source_mask_key",
        "episode_pool_role",
        "group_kind",
        "defect_labels",
        "n_images",
        "n_masks",
        "image_ids",
        "image_paths",
        "mask_paths",
    ]
    write_csv(
        output_dir / artifacts["episode_groups"],
        episode_fields,
        episode_rows,
    )
    write_csv(
        output_dir / artifacts["episode_statistics"],
        list(episode_statistics[0].keys()),
        episode_statistics,
    )
    load_verified_episode_groups(output_dir / artifacts["episode_groups"])
    load_verified_research_splits(
        output_dir / artifacts["outer"],
        output_dir / artifacts["meta"],
        output_dir / artifacts["inner"],
    )
    render_split_report(
        root / config["report_path"],
        config,
        family_rows,
        outer_payload,
        meta_payload,
        inner_payload,
        statistics,
        episode_statistics,
    )
    if config.get("freeze_artifacts", False):
        _freeze_artifacts(root, config_file, output_dir, config)
    print(
        f"families={len(all_families)} outer_train={len(outer_train)} "
        f"outer_test={len(outer_test)} meta_train={len(meta_train)} "
        f"meta_val={len(meta_val)} inner_folds={n_folds}"
    )
    print(
        "realiad_physical_groups={n_realiad_physical_groups} "
        "outer_overlap=0".format(**physical_counts)
    )
    print(
        f"episode_pool_products={len(episode_statistics)} "
        f"episode_physical_groups={len(episode_rows)} role_overlap=0"
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
