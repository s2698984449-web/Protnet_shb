"""B0 对应的冻结训练索引、分层抽样和按需读取。"""

from __future__ import annotations

from collections import defaultdict
from copy import copy
from itertools import zip_longest
from random import Random

from data_proc.detection import (
    UnifiedDetectionAdapter,
    image_to_tensor,
    target_for_model,
)


def build_b0_index(
    adapter: UnifiedDetectionAdapter,
    episode_groups: list[dict],
    family_ids: set[str],
    group_split: dict,
) -> list[dict]:
    """只从选定的 Meta-Train 产品族建立训练索引，不打开图像或 Mask。"""
    if not family_ids:
        raise ValueError("B0 训练产品族不能为空")
    excluded_images = set()
    if group_split["conflict_policy"] == "preserve_official_test":
        excluded_images = set(group_split["excluded_official_train_image_ids"])

    index = []
    seen_images = set()
    seen_families = set()
    for group in episode_groups:
        family_id = group["family_id"]
        if family_id not in family_ids:
            continue
        if (group["outer_split"], group["meta_split"], group["evaluation_stage"]) != (
            "train", "train", "inner_oof"
        ):
            raise AssertionError(f"B0 训练索引混入非 Meta-Train 产品族：{family_id}")

        ids = group["image_ids"].split("|")
        if not all(ids) or len(ids) != int(group["n_images"]) or len(set(ids)) != len(ids):
            raise AssertionError(f"冻结物理组图像清单不一致：{group['physical_group_id']}")
        rows = [adapter.manifest[image_id] for image_id in ids]
        image_paths = sorted(row["image_path"] for row in rows)
        mask_paths = sorted(row["mask_path"] for row in rows if row["mask_path"])
        if (
            "|".join(image_paths) != group["image_paths"]
            or "|".join(mask_paths) != group["mask_paths"]
        ):
            raise AssertionError(f"Manifest 路径与冻结物理组不一致：{group['physical_group_id']}")

        for row in rows:
            image_id = row["image_id"]
            dataset_id = row["dataset_id"]
            product_id = row["product_id"]
            key = image_id
            if dataset_id == group_split["dataset_id"]:
                key = group_split["group_key"].format(
                    category=product_id, sample_id=row["physical_sample_id"]
                )
            if (dataset_id, product_id, key) != (
                group["dataset_id"], group["product_id"], group["physical_group_id"]
            ):
                raise AssertionError(f"Manifest 记录越过冻结产品或物理组边界：{image_id}")
            if image_id in seen_images:
                raise AssertionError(f"B0 图像被重复分配到物理组：{image_id}")
            seen_images.add(image_id)
            if image_id in excluded_images:
                continue
            normal_flag = int(row["normal_flag"])
            if normal_flag not in (0, 1) or (
                not normal_flag and not (row["mask_path"] and row["defect_labels"])
            ):
                raise AssertionError(f"B0 Manifest 缺少有效图像标注：{image_id}")
            index.append(
                {
                    "image_id": image_id,
                    "dataset_id": dataset_id,
                    "product_id": product_id,
                    "family_id": family_id,
                    "physical_group_id": key,
                    "normal_flag": normal_flag,
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "defect_labels": row["defect_labels"],
                }
            )
            seen_families.add(family_id)

    if missing := family_ids - seen_families:
        raise AssertionError(f"B0 训练产品族无可用图像：{sorted(missing)}")
    return sorted(index, key=lambda row: row["image_id"])


def _round_robin(families: dict[str, list[list[int]]], rng: Random) -> list[list[int]]:
    pools = [families[family_id] for family_id in sorted(families)]
    for pool in pools:
        rng.shuffle(pool)
    rng.shuffle(pools)
    return [group for cycle in zip_longest(*pools) for group in cycle if group is not None]


def sample_b0_epoch(
    index: list[dict], *, seed: int, epoch: int, sampling: dict
) -> list[int]:
    """每域按正负物理组分配配额，桶内在各产品族之间轮流抽组。"""
    quota = int(sampling["groups_per_dataset"])
    fraction = float(sampling["positive_group_fraction"])
    max_views = int(sampling["max_views_per_group"])
    if quota < 1 or max_views < 1 or not 0.0 <= fraction <= 1.0:
        raise ValueError("B0 物理组配额和视角上限必须为正，正组比例必须在 [0, 1] 内")

    rng = Random(f"{seed}:{epoch}")
    by_dataset = defaultdict(lambda: defaultdict(list))
    for row_index, row in enumerate(index):
        key = (row["family_id"], row["product_id"], row["physical_group_id"])
        by_dataset[row["dataset_id"]][key].append(row_index)

    selection = []
    for dataset_id in sorted(by_dataset):
        positive = defaultdict(list)
        negative = defaultdict(list)
        for (family_id, _, _), group in sorted(by_dataset[dataset_id].items()):
            has_defect = any(not index[i]["normal_flag"] for i in group)
            pools = positive if has_defect else negative
            pools[family_id].append(group)
        positive_groups = _round_robin(positive, rng)
        negative_groups = _round_robin(negative, rng)
        n_groups = min(quota, len(positive_groups) + len(negative_groups))
        n_positive = min(len(positive_groups), round(n_groups * fraction))
        n_negative = min(len(negative_groups), n_groups - n_positive)
        n_positive = n_groups - n_negative

        for group in positive_groups[:n_positive]:
            first = rng.choice([i for i in group if not index[i]["normal_flag"]])
            other_views = [i for i in group if i != first]
            selection.append(first)
            selection.extend(
                rng.sample(other_views, min(max_views - 1, len(other_views)))
            )
        for group in negative_groups[:n_negative]:
            selection.extend(rng.sample(group, min(max_views, len(group))))

    rng.shuffle(selection)
    return selection


def configure_detection_worker(_: int) -> None:
    import cv2

    cv2.setNumThreads(1)


class B0DetectionDataset:
    def __init__(self, adapter: UnifiedDetectionAdapter, index: list[dict]) -> None:
        # 工作者按已验证的 index 取图，Windows spawn 时无需复制完整标注索引。
        self.adapter = copy(adapter)
        self.adapter.manifest = {}
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, position: int):
        row = self.index[position]
        record = self.adapter.build_b0_record(row)
        if record is None:
            return None, {"image_id": row["image_id"], "dataset_id": row["dataset_id"]}
        target = target_for_model(record)
        target["image_id_str"] = row["image_id"]
        return image_to_tensor(record["image_path"]), target
