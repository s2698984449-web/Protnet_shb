"""统一二分类检测记录、Mask 连通域转框与 Torch 数据集。"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.io import (
    load_config,
    named_config_path,
    project_root,
    read_jsonl,
    write_csv,
)


def _threshold_mask(mask: np.ndarray, threshold: str | int) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    if threshold == "nonzero":
        return mask > 0
    return mask > int(threshold)


def mask_to_boxes(
    mask: np.ndarray,
    threshold: str | int,
    min_component_area: int,
    connectivity: int = 8,
) -> tuple[list[list[float]], dict]:
    """按连通区域生成实例框；不把整张 Mask 合并成一个框。"""
    binary = _threshold_mask(mask, threshold).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )
    boxes: list[list[float]] = []
    removed = 0
    component_areas = []
    for component in range(1, component_count):
        x, y, width, height, area = [
            int(value) for value in stats[component]
        ]
        component_areas.append(area)
        if area < min_component_area:
            removed += 1
            continue
        if width <= 0 or height <= 0:
            removed += 1
            continue
        boxes.append([float(x), float(y), float(x + width), float(y + height)])
    return boxes, {
        "foreground_pixels": int(binary.sum()),
        "raw_components": component_count - 1,
        "kept_components": len(boxes),
        "removed_small_components": removed,
        "min_raw_component_area": min(component_areas) if component_areas else 0,
        "max_raw_component_area": max(component_areas) if component_areas else 0,
    }


def _load_manifest(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = {}
    for row in rows:
        if row["image_id"] in manifest:
            raise AssertionError(f"manifest image_id 重复：{row['image_id']}")
        manifest[row["image_id"]] = row
    return manifest


class UnifiedDetectionAdapter:
    """把三套数据转换为统一二分类检测记录。"""

    def __init__(self, baseline_config: dict | None = None) -> None:
        self.root = project_root()
        self.config = baseline_config or load_config(
            named_config_path("baseline.yaml")
        )
        data_config = load_config(
            self.root / self.config["data_config"]
        )
        self.dataset_roots = {
            dataset_id: Path(dataset["root"])
            for dataset_id, dataset in data_config["datasets"].items()
        }
        self.manifest = _load_manifest(
            self.root / data_config["output_dir"] / "manifest.csv"
        )
        self.mask_config = self.config["mask_to_box"]
        self._annotation_cache = {}

    def _annotation(
        self,
        row: dict,
        selected_classes: set[str],
    ) -> tuple[list[dict], list[dict], dict]:
        labels = {
            label for label in row["defect_labels"].split("|") if label
        }
        cache_key = (row["image_id"], tuple(sorted(selected_classes)))
        cached = self._annotation_cache.get(cache_key)
        if cached is not None:
            return cached
        positive = bool(labels & selected_classes)
        ignored = bool(labels - selected_classes)
        if int(row["normal_flag"]) == 1:
            result = ([], [], {
                "foreground_pixels": 0,
                "raw_components": 0,
                "kept_components": 0,
                "removed_small_components": 0,
                "min_raw_component_area": 0,
                "max_raw_component_area": 0,
            })
            self._annotation_cache[cache_key] = result
            return result
        if not row["mask_path"]:
            raise AssertionError(f"缺陷图缺少 Mask：{row['image_id']}")
        dataset_id = row["dataset_id"]
        mask_path = self.dataset_roots[dataset_id] / row["mask_path"]
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"))
        boxes, diagnostics = mask_to_boxes(
            mask,
            self.mask_config["thresholds"][dataset_id],
            int(self.mask_config["min_component_area"]),
            int(self.mask_config["connectivity"]),
        )
        annotations = [
            {
                "bbox": tuple(box),
                "category_id": int(self.config["classes"]["defect"]),
                "area": float((box[2] - box[0]) * (box[3] - box[1])),
                "iscrowd": 0,
            }
            for box in boxes
        ]
        if positive:
            result = (annotations, [], diagnostics)
        elif ignored:
            result = ([], annotations, diagnostics)
        else:
            raise AssertionError(f"缺陷图没有可解释的类别：{row['image_id']}")
        self._annotation_cache[cache_key] = result
        return result
        raise AssertionError(f"缺陷图没有可解释的类别：{row['image_id']}")

    def build_b0_record(self, row: dict) -> dict | None:
        """按需生成训练记录，无法保留任何有效框的缺陷图返回空记录。"""
        selected_classes = set(row["defect_labels"].split("|")) - {""}
        annotations, _, _ = self._annotation(row, selected_classes)
        if int(row["normal_flag"]) == 0 and not annotations:
            return None
        return {
            "image_id": row["image_id"],
            "image_path": str(
                self.dataset_roots[row["dataset_id"]] / row["image_path"]
            ),
            "annotations": annotations,
        }

    def build_episode_records(
        self,
        episode: dict,
        roles: Iterable[str] = (
            "reference",
            "support",
            "query_normal",
            "query_defect",
        ),
    ) -> list[dict]:
        selected_classes = set(episode["episode_classes"])
        records = []
        role_map = {
            "reference": "R",
            "support": "SUPPORT",
            "query_normal": "QUERY_NORMAL",
            "query_defect": "QUERY_DEFECT",
        }
        for role_name in roles:
            for unit in episode[role_name]:
                for image_id in unit["image_ids"]:
                    row = self.manifest[image_id]
                    if row["product_id"] != episode["product_id"]:
                        raise AssertionError(
                            f"Episode 图像越过产品边界：{image_id}"
                        )
                    image_path = self.dataset_roots[row["dataset_id"]] / row[
                        "image_path"
                    ]
                    with Image.open(image_path) as image:
                        width, height = image.size
                    annotations, ignore_annotations, diagnostics = (
                        self._annotation(row, selected_classes)
                    )
                    records.append(
                        {
                            "episode_id": episode["episode_id"],
                            "image_id": image_id,
                            "dataset": row["dataset_id"],
                            "product_id": row["product_id"],
                            "family_id": episode["family_id"],
                            "physical_group_id": unit["physical_group_id"],
                            "view_id": row["camera_id"],
                            "split_role": role_map[role_name],
                            "image_path": str(image_path),
                            "mask_path": (
                                str(
                                    self.dataset_roots[row["dataset_id"]]
                                    / row["mask_path"]
                                )
                                if row["mask_path"]
                                else ""
                            ),
                            "width": width,
                            "height": height,
                            "normal_flag": int(row["normal_flag"]),
                            "annotations": annotations,
                            "ignore_annotations": ignore_annotations,
                            "mask_diagnostics": diagnostics,
                        }
                    )
        return records


def image_to_tensor(path: str):
    """读图为 CHW float32；先走 uint8 再转 float，避免直接 float32 大数组峰值翻倍。"""
    import torch

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(dtype=torch.float32)
    tensor.div_(255.0)
    del rgb
    return tensor


def target_for_model(record: dict) -> dict:
    import torch

    boxes = [annotation["bbox"] for annotation in record["annotations"]]
    if boxes:
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
    else:
        box_tensor = torch.zeros((0, 4), dtype=torch.float32)
    return {
        "boxes": box_tensor,
        "labels": torch.ones((len(boxes),), dtype=torch.int64),
        "image_id": torch.tensor(
            [int(hashlib_sha1(record["image_id"])[:15], 16)], dtype=torch.int64
        ),
        "area": (
            (box_tensor[:, 2] - box_tensor[:, 0])
            * (box_tensor[:, 3] - box_tensor[:, 1])
        ),
        "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
    }


def hashlib_sha1(value: str) -> str:
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()


class TorchDetectionDataset:
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return image_to_tensor(record["image_path"]), target_for_model(record)


def load_episode_records(
    bank_path: Path,
    max_episodes: int | None = None,
) -> list[dict]:
    adapter = UnifiedDetectionAdapter()
    episodes = read_jsonl(bank_path)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return [
        record
        for episode in episodes
        for record in adapter.build_episode_records(episode)
    ]


def audit_annotation_pilot() -> None:
    root = project_root()
    config = load_config(named_config_path("baseline.yaml"))
    adapter = UnifiedDetectionAdapter(config)
    pilot = config["annotation_pilot"]
    episodes = read_jsonl(root / pilot["bank"])
    limit = int(pilot["max_unique_images_per_dataset"])
    seen: dict[str, set[str]] = defaultdict(set)
    audit_rows = []
    for episode in episodes:
        for record in adapter.build_episode_records(episode):
            dataset_id = record["dataset"]
            if (
                len(seen[dataset_id]) >= limit
                or record["image_id"] in seen[dataset_id]
            ):
                continue
            seen[dataset_id].add(record["image_id"])
            for annotation in [
                *record["annotations"],
                *record["ignore_annotations"],
            ]:
                x1, y1, x2, y2 = annotation["bbox"]
                if not (
                    0 <= x1 < x2 <= record["width"]
                    and 0 <= y1 < y2 <= record["height"]
                ):
                    raise AssertionError(
                        f"检测框越界：{record['image_id']} {annotation['bbox']}"
                    )
            if record["normal_flag"] == 1 and (
                record["annotations"] or record["ignore_annotations"]
            ):
                raise AssertionError(f"正常图出现检测框：{record['image_id']}")
            diagnostics = record["mask_diagnostics"]
            if (
                record["normal_flag"] == 0
                and diagnostics["kept_components"] == 0
            ):
                raise AssertionError(
                    f"缺陷 Mask 没有保留连通区域：{record['image_id']}"
                )
            audit_rows.append(
                {
                    "dataset_id": dataset_id,
                    "episode_id": record["episode_id"],
                    "product_id": record["product_id"],
                    "physical_group_id": record["physical_group_id"],
                    "image_id": record["image_id"],
                    "split_role": record["split_role"],
                    "normal_flag": record["normal_flag"],
                    "width": record["width"],
                    "height": record["height"],
                    "positive_boxes": len(record["annotations"]),
                    "ignore_boxes": len(record["ignore_annotations"]),
                    **diagnostics,
                    "image_path": record["image_path"],
                    "mask_path": record["mask_path"],
                }
            )
        if all(
            len(seen[dataset_id]) >= limit
            for dataset_id in adapter.dataset_roots
        ):
            break

    output_path = root / config["annotation_output"]
    write_csv(output_path, list(audit_rows[0].keys()), audit_rows)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in audit_rows:
        by_dataset[row["dataset_id"]].append(row)
    lines = [
        "# 固定 Episode 标注适配审计",
        "",
        f"日期：{date.today().isoformat()}",
        "",
        "本轮只读取 Meta-Validation 固定 Episode 中每个数据集最多"
        f"{limit}张唯一图像，不扫描原始目录。",
        "",
        "Mask 转框采用连通区域，不把所有前景合并成一个大框。"
        f"最小连通域面积为 {config['mask_to_box']['min_component_area']} 像素。",
        "",
    ]
    for dataset_id in sorted(by_dataset):
        rows = by_dataset[dataset_id]
        lines.append(
            f"- {dataset_id}：{len(rows)}张；正常"
            f"{sum(int(row['normal_flag']) == 1 for row in rows)}张；"
            f"正框{sum(int(row['positive_boxes']) for row in rows)}个；"
            f"Ignore框{sum(int(row['ignore_boxes']) for row in rows)}个；"
            f"移除小连通域"
            f"{sum(int(row['removed_small_components']) for row in rows)}个。"
        )
    lines.extend(
        [
            "",
            "三套数据均能转换为统一二分类检测记录；正常图保留为空标注负样本。"
            "Real-IAD 无可见缺陷的视角保持负样本，不因所属物理组为NG而强制标正。",
            "",
        ]
    )
    report_path = root / config["annotation_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        "annotation_images="
        + ",".join(
            f"{dataset_id}:{len(rows)}"
            for dataset_id, rows in sorted(by_dataset.items())
        )
    )
    print(f"wrote {output_path}")
