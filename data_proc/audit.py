"""S0 审计：从索引生成 manifest、产品计数和合格任务候选。

只读三套数据的已知索引，不递归扫盘，不打开图像像素。
MVTec 只列各类别的 train、test，并核对约定掩码文件名。
VisA 主清单只读 split_csv/1cls.csv。2cls 文件只做行数对照，不并入 manifest。
Real-IAD 只读 realiad_jsons/*.json。图像根目录约定为
realiad_1024/<类别>/<类别>/<json 内相对路径>。
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.io import (
    assert_disjoint_groups,
    config_path,
    file_sha256,
    load_verified_group_split,
    load_config,
    percentile,
    posix,
    project_root,
    write_csv,
    write_json,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SAMPLE_DIR = re.compile(r"S\d+")
CAMERA_IN_NAME = re.compile(r"_C(\d+)_", re.IGNORECASE)
SAMPLE_IN_NAME = re.compile(r"_(\d+)_(?:OK|NG)_", re.IGNORECASE)
TIMESTAMP_IN_NAME = re.compile(r"_C\d+_(\d{14})\.", re.IGNORECASE)

MANIFEST_FIELDS = [
    "image_id",
    "dataset_id",
    "product_id",
    "product_family_id",
    "physical_sample_id",
    "normal_flag",
    "image_role",
    "defect_labels",
    "bbox_path",
    "mask_path",
    "image_path",
    "split",
    "camera_id",
    "source_index",
]


def parse_sample_token(relative_path: str) -> tuple[str, str, str]:
    """从路径目录名提取 Sxxxx，从文件名提取相机位。"""
    parts = [part for part in relative_path.split("/") if part]
    sample_ids = [part for part in parts if SAMPLE_DIR.fullmatch(part)]
    camera_match = CAMERA_IN_NAME.search(parts[-1] if parts else "")
    camera_id = f"C{camera_match.group(1)}" if camera_match else ""
    if len(sample_ids) == 1:
        return sample_ids[0], camera_id, ""
    if not sample_ids:
        return "", camera_id, "sample_id_unparsed"
    return sample_ids[0], camera_id, "multiple_sample_tokens"


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def collect_mvtec(root: Path, normal_dirname: str) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    flags: list[dict] = []
    categories = sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    for category in categories:
        for split in ("train", "test"):
            split_dir = category / split
            if not split_dir.is_dir():
                flags.append(_flag("missing_split_dir", "mvtec", category.name, split, str(split_dir)))
                continue
            class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
            for class_dir in class_dirs:
                label = class_dir.name
                normal = label == normal_dirname
                for entry in sorted(class_dir.iterdir(), key=lambda item: item.name):
                    if entry.is_dir():
                        flags.append(
                            _flag(
                                "unexpected_subdir",
                                "mvtec",
                                category.name,
                                posix(entry.relative_to(root)),
                                "未下钻",
                            )
                        )
                        continue
                    if not is_image(entry):
                        continue
                    mask_rel = ""
                    if not normal:
                        mask_path = category / "ground_truth" / label / f"{entry.stem}_mask.png"
                        if mask_path.is_file():
                            mask_rel = posix(mask_path.relative_to(root))
                        else:
                            flags.append(
                                _flag(
                                    "missing_mask",
                                    "mvtec",
                                    category.name,
                                    posix(entry.relative_to(root)),
                                    posix(mask_path.relative_to(root)),
                                )
                            )
                    records.append(
                        _raw(
                            dataset_id="mvtec",
                            product_id=category.name,
                            split=split,
                            image_rel=posix(entry.relative_to(root)),
                            mask_rel=mask_rel,
                            normal=normal,
                            label="" if normal else label,
                            sample_id="",
                            camera_id="",
                            source_index="mvtec_train_test_layout",
                        )
                    )
    return records, flags


def collect_visa(root: Path, index_rel: str) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    flags: list[dict] = []
    index_path = root / index_rel
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            image_rel = posix(row["image"].strip())
            if not image_rel:
                flags.append(_flag("empty_image_path", "visa", row.get("object", ""), str(line_no), index_rel))
                continue
            label = row["label"].strip().lower()
            normal = label == "normal"
            mask_rel = posix(row["mask"].strip())
            records.append(
                _raw(
                    dataset_id="visa",
                    product_id=row["object"].strip(),
                    split=row["split"].strip(),
                    image_rel=image_rel,
                    mask_rel=mask_rel,
                    normal=normal,
                    label="" if normal else label,
                    sample_id="",
                    camera_id="",
                    source_index=posix(index_rel),
                )
            )
    return records, flags


def summarize_side_index(root: Path, index_rel: str) -> dict:
    index_path = root / index_rel
    objects: set[str] = set()
    labels: set[str] = set()
    splits: set[str] = set()
    rows = 0
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            objects.add(row.get("object", "").strip())
            labels.add(row.get("label", "").strip())
            splits.add(row.get("split", "").strip())
    return {
        "index": posix(index_rel),
        "n_rows": rows,
        "n_objects": len(objects),
        "objects": "|".join(sorted(objects)),
        "labels": "|".join(sorted(labels)),
        "splits": "|".join(sorted(splits)),
        "in_manifest": 0,
    }


def collect_realiad(root: Path, index_dir: str, image_subdir: str) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    flags: list[dict] = []
    index_root = root / index_dir
    for json_path in sorted(index_root.glob("*.json")):
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        expected_product = json_path.stem
        normal_name = str(payload.get("meta", {}).get("normal_class", "OK"))
        prefix = str(payload.get("meta", {}).get("prefix", ""))
        if prefix != f"{expected_product}/":
            flags.append(
                _flag("prefix_mismatch", "realiad", expected_product, prefix, f"期望 {expected_product}/")
            )
        for split in ("train", "test"):
            for row in payload.get(split, []):
                product_id = str(row.get("category", "")).strip() or expected_product
                if product_id != expected_product:
                    flags.append(
                        _flag(
                            "category_mismatch",
                            "realiad",
                            expected_product,
                            product_id,
                            json_path.name,
                        )
                    )
                image_rel_in_json = posix(str(row.get("image_path", "")).strip())
                if not image_rel_in_json:
                    flags.append(_flag("empty_image_path", "realiad", product_id, split, json_path.name))
                    continue
                image_rel = posix(Path(image_subdir) / product_id / product_id / image_rel_in_json)
                sample_id, camera_id, sample_issue = parse_sample_token(image_rel_in_json)
                if sample_issue:
                    flags.append(_flag(sample_issue, "realiad", product_id, image_rel, image_rel_in_json))
                anomaly = str(row.get("anomaly_class", "")).strip()
                normal = anomaly == normal_name
                mask_value = row.get("mask_path")
                mask_rel = ""
                if mask_value not in (None, "", "None"):
                    mask_rel = posix(Path(image_subdir) / product_id / product_id / posix(str(mask_value)))
                records.append(
                    _raw(
                        dataset_id="realiad",
                        product_id=product_id,
                        split=split,
                        image_rel=image_rel,
                        mask_rel=mask_rel,
                        normal=normal,
                        label="" if normal else anomaly,
                        sample_id=sample_id,
                        camera_id=camera_id,
                        source_index=f"realiad_jsons/{json_path.name}",
                    )
                )
    return records, flags


def _raw(
    dataset_id: str,
    product_id: str,
    split: str,
    image_rel: str,
    mask_rel: str,
    normal: bool,
    label: str,
    sample_id: str,
    camera_id: str,
    source_index: str,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "product_id": product_id,
        "product_family_id": product_id,
        "split": split,
        "image_path": image_rel,
        "mask_path": mask_rel,
        "normal_flag": "1" if normal else "0",
        "image_role": infer_image_role(dataset_id, image_rel, normal),
        "defect_label": label,
        "physical_sample_id": sample_id,
        "camera_id": camera_id,
        "source_index": source_index,
    }


def infer_image_role(dataset_id: str, image_rel: str, normal: bool) -> str:
    """区分官方图像标签和 Real-IAD 的拍摄目录。"""
    if dataset_id == "realiad":
        parts = set(image_rel.split("/"))
        if "OK" in parts:
            return "ok_reference"
        if normal:
            return "ng_view_ok"
        return "ng_defect"
    if dataset_id == "visa":
        return "normal" if normal else "anomaly"
    return "good" if normal else "defect"


def _flag(flag_type: str, dataset_id: str, product_id: str, key: str, detail: str) -> dict:
    return {
        "flag_type": flag_type,
        "dataset_id": dataset_id,
        "product_id": product_id,
        "key": key,
        "detail": detail,
    }


def merge_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset_id"], record["image_path"])].append(record)

    merged: list[dict] = []
    flags: list[dict] = []
    for (dataset_id, image_path), group in grouped.items():
        product_id = group[0]["product_id"]
        if len(group) > 1:
            flags.append(
                _flag("duplicate_image_path", dataset_id, product_id, image_path, str(len(group)))
            )
        labels = sorted({item["defect_label"] for item in group if item["defect_label"]})
        normals = {item["normal_flag"] for item in group}
        splits = sorted({item["split"] for item in group})
        sample_ids = sorted({item["physical_sample_id"] for item in group if item["physical_sample_id"]})
        cameras = sorted({item["camera_id"] for item in group if item["camera_id"]})
        masks = sorted({item["mask_path"] for item in group if item["mask_path"]})
        roles = sorted({item["image_role"] for item in group})
        if len(roles) > 1:
            flags.append(_flag("image_role_conflict", dataset_id, product_id, image_path, "|".join(roles)))
        if len(normals) > 1 or (labels and "1" in normals):
            flags.append(_flag("label_conflict_same_path", dataset_id, product_id, image_path, "|".join(labels)))
        if len(splits) > 1:
            flags.append(_flag("split_conflict_same_path", dataset_id, product_id, image_path, "|".join(splits)))
        if len(sample_ids) > 1:
            flags.append(
                _flag("sample_id_conflict_same_path", dataset_id, product_id, image_path, "|".join(sample_ids))
            )
        normal_flag = "0" if labels else "1"
        merged.append(
            {
                "image_id": f"{dataset_id}:{image_path}",
                "dataset_id": dataset_id,
                "product_id": product_id,
                "product_family_id": product_id,
                "physical_sample_id": sample_ids[0] if sample_ids else "",
                "normal_flag": normal_flag,
                "image_role": roles[0] if len(roles) == 1 else "|".join(roles),
                "defect_labels": "|".join(labels),
                "bbox_path": "",
                "mask_path": masks[0] if masks else "",
                "image_path": image_path,
                "split": splits[0] if len(splits) == 1 else "conflict",
                "camera_id": cameras[0] if len(cameras) == 1 else "|".join(cameras),
                "source_index": group[0]["source_index"],
            }
        )
    merged.sort(key=lambda row: (row["dataset_id"], row["product_id"], row["split"], row["image_path"]))
    return merged, flags


def sample_uid(row: dict) -> str:
    sample_id = row["physical_sample_id"] or row["image_id"]
    return f"{row['dataset_id']}:{row['product_id']}:{sample_id}"


def build_tables(
    manifest: list[dict],
    k_values: list[int],
    q_values: list[int],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    by_product: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in manifest:
        by_product[(row["dataset_id"], row["product_id"])].append(row)

    product_rows: list[dict] = []
    class_rows: list[dict] = []
    task_rows: list[dict] = []
    sample_flags: list[dict] = []
    quota_rows: list[dict] = []
    large_sample_rows: list[dict] = []

    for dataset_id, product_id in sorted(by_product):
        rows = by_product[(dataset_id, product_id)]
        samples: dict[str, dict] = {}
        for row in rows:
            uid = sample_uid(row)
            slot = samples.setdefault(
                uid,
                {
                    "sample_id": row["physical_sample_id"] or row["image_id"],
                    "splits": set(),
                    "normal": False,
                    "labels": set(),
                    "n_images": 0,
                    "cameras": set(),
                },
            )
            slot["n_images"] += 1
            slot["splits"].add(row["split"])
            if row["camera_id"]:
                slot["cameras"].add(row["camera_id"])
            if row["normal_flag"] == "1":
                slot["normal"] = True
            if row["defect_labels"]:
                slot["labels"].update(label for label in row["defect_labels"].split("|") if label)

        both_states = [item for item in samples.values() if item["normal"] and item["labels"]]
        multi_split = [item for item in samples.values() if len(item["splits"]) > 1]

        class_splits = _class_split_counts(rows)
        defect_labels = sorted(label for label in class_splits if not label.startswith("__normal__"))
        classes_test_only = []
        classes_train_only = []
        classes_both = []
        for label in defect_labels:
            train_n = class_splits[label]["train"]["samples"]
            test_n = class_splits[label]["test"]["samples"]
            if train_n and test_n:
                classes_both.append(label)
            elif test_n:
                classes_test_only.append(label)
            elif train_n:
                classes_train_only.append(label)
            for split_name in ("all", "train", "test"):
                bucket = class_splits[label][split_name]
                class_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "product_id": product_id,
                        "label": label,
                        "state": "defect",
                        "split": split_name,
                        "n_images": bucket["images"],
                        "n_samples": bucket["samples"],
                        "n_with_mask": bucket["masks"],
                    }
                )
        normal_key = "__normal__"
        for split_name in ("all", "train", "test"):
            bucket = class_splits[normal_key][split_name]
            if bucket["images"] == 0 and split_name != "all":
                continue
            class_rows.append(
                {
                    "dataset_id": dataset_id,
                    "product_id": product_id,
                    "label": "normal",
                    "state": "normal",
                    "split": split_name,
                    "n_images": bucket["images"],
                    "n_samples": bucket["samples"],
                    "n_with_mask": bucket["masks"],
                }
            )

        defect_images = [row for row in rows if row["normal_flag"] == "0"]
        defect_with_mask = sum(1 for row in defect_images if row["mask_path"])
        normal_samples = sum(1 for item in samples.values() if item["normal"])
        defect_samples = sum(1 for item in samples.values() if item["labels"])
        product_rows.append(
            {
                "dataset_id": dataset_id,
                "product_id": product_id,
                "product_family_id": product_id,
                "n_images": len(rows),
                "n_normal_images": sum(1 for row in rows if row["normal_flag"] == "1"),
                "n_defect_images": len(defect_images),
                "n_defect_classes": len(defect_labels),
                "n_normal_samples": normal_samples,
                "n_defect_samples": defect_samples,
                "n_physical_samples": len(samples),
                "n_samples_in_multiple_splits": len(multi_split),
                "n_samples_both_states": len(both_states),
                "defect_with_mask": defect_with_mask,
                "defect_missing_mask": len(defect_images) - defect_with_mask,
                "classes_test_only": "|".join(classes_test_only),
                "classes_train_only": "|".join(classes_train_only),
                "classes_in_both_splits": "|".join(classes_both),
                "normal_ref_ok": int(normal_samples >= 1),
                "loo_calibration_ok": int(normal_samples >= 2),
            }
        )

        all_counts = {label: class_splits[label]["all"]["samples"] for label in defect_labels}
        train_counts = {label: class_splits[label]["train"]["samples"] for label in defect_labels}
        test_counts = {label: class_splits[label]["test"]["samples"] for label in defect_labels}
        for k_shot in k_values:
            for q_min in q_values:
                eligible = sorted(label for label, count in all_counts.items() if count >= k_shot + q_min)
                official = sorted(
                    label
                    for label in defect_labels
                    if train_counts[label] >= k_shot and test_counts[label] >= q_min
                )
                if eligible:
                    limiting = min(eligible, key=lambda label: (all_counts[label], label))
                    limiting_n = all_counts[limiting]
                    disjoint = limiting_n // (k_shot + q_min)
                else:
                    limiting = ""
                    limiting_n = 0
                    disjoint = 0
                task_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "product_id": product_id,
                        "k_shot": k_shot,
                        "q_min": q_min,
                        "n_way": len(eligible),
                        "eligible_classes": "|".join(eligible),
                        "limiting_class": limiting,
                        "limiting_samples": limiting_n,
                        "disjoint_episodes": disjoint,
                        "normal_samples": normal_samples,
                        "det_qualified": int(len(eligible) >= 1 and normal_samples >= 1),
                        "loo_calibration_ok": int(normal_samples >= 2),
                        "n_way_official_split": len(official),
                    }
                )

        if dataset_id == "realiad":
            quota, large_rows = _realiad_quota(dataset_id, product_id, samples)
            quota_rows.append(quota)
            large_sample_rows.extend(large_rows)

    return product_rows, class_rows, task_rows, sample_flags, quota_rows, large_sample_rows


def _empty_bucket() -> dict[str, set | int]:
    return {"images": 0, "masks": 0, "sample_ids": set()}


def _class_split_counts(rows: list[dict]) -> dict[str, dict[str, dict]]:
    counts: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(_empty_bucket))
    for row in rows:
        labels = ["__normal__"] if row["normal_flag"] == "1" else []
        labels.extend(label for label in row["defect_labels"].split("|") if label)
        uid = sample_uid(row)
        has_mask = 1 if row["mask_path"] else 0
        for label in labels:
            for split_name in ("all", row["split"]):
                if split_name == "conflict":
                    continue
                bucket = counts[label][split_name]
                bucket["images"] += 1
                bucket["masks"] += has_mask
                bucket["sample_ids"].add(uid)
    normalized: dict[str, dict[str, dict]] = {}
    for label, splits in counts.items():
        normalized[label] = {}
        for split_name in ("all", "train", "test"):
            bucket = splits.get(split_name, _empty_bucket())
            sample_ids = bucket["sample_ids"]
            normalized[label][split_name] = {
                "images": bucket["images"],
                "masks": bucket["masks"],
                "samples": len(sample_ids) if isinstance(sample_ids, set) else 0,
            }
    if "__normal__" not in normalized:
        normalized["__normal__"] = {
            split_name: {"images": 0, "masks": 0, "samples": 0} for split_name in ("all", "train", "test")
        }
    return normalized


def _realiad_quota(dataset_id: str, product_id: str, samples: dict[str, dict]) -> tuple[dict, list[dict]]:
    sizes = sorted(item["n_images"] for item in samples.values())
    median = percentile(sizes, 0.5)
    p95 = percentile(sizes, 0.95)
    maximum = sizes[-1] if sizes else 0
    total_images = sum(sizes)
    largest = max(samples.values(), key=lambda item: item["n_images"])
    share = (largest["n_images"] / total_images) if total_images else 0.0
    # 单样本占比超过 2% 才视为会主导该产品。5 的倍数来自多相机和多状态，不按这个设配额。
    dominates = share > 0.02
    cap = int(math.ceil(p95))
    large_rows: list[dict] = []
    if dominates:
        for item in samples.values():
            if item["n_images"] > p95:
                large_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "product_id": product_id,
                        "physical_sample_id": item["sample_id"],
                        "n_images": item["n_images"],
                        "n_cameras": len(item["cameras"]),
                        "splits": "|".join(sorted(item["splits"])),
                        "labels": "|".join(sorted(item["labels"])) or "normal",
                        "suggested_cap": cap,
                    }
                )
    quota = {
        "dataset_id": dataset_id,
        "product_id": product_id,
        "n_samples": len(samples),
        "n_images": total_images,
        "min_images_per_sample": sizes[0] if sizes else 0,
        "median_images_per_sample": f"{median:.1f}",
        "p95_images_per_sample": f"{p95:.1f}",
        "max_images_per_sample": maximum,
        "max_sample_id": largest["sample_id"],
        "max_sample_share": f"{share:.4f}",
        "distribution": "sample_dominates" if dominates else "no_sample_dominates",
        "suggested_cap": cap if dominates else "",
        "n_samples_over_p95": len(large_rows),
    }
    return quota, large_rows


def summarize_tasks(task_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in task_rows:
        grouped[(row["dataset_id"], int(row["k_shot"]), int(row["q_min"]))].append(row)
    summary = []
    for (dataset_id, k_shot, q_min), rows in sorted(grouped.items()):
        ways = sorted(int(row["n_way"]) for row in rows)
        summary.append(
            {
                "dataset_id": dataset_id,
                "k_shot": k_shot,
                "q_min": q_min,
                "n_products": len(rows),
                "n_det_qualified": sum(int(row["det_qualified"]) for row in rows),
                "n_with_loo": sum(int(row["loo_calibration_ok"]) for row in rows if int(row["det_qualified"])),
                "median_n_way": f"{percentile(ways, 0.5):.1f}",
                "n_products_nway_ge_2": sum(way >= 2 for way in ways),
                "n_products_nway_ge_5": sum(way >= 5 for way in ways),
                "min_n_way": ways[0] if ways else 0,
                "max_n_way": ways[-1] if ways else 0,
            }
        )
    return summary


def existence_checks(manifest: list[dict], roots: dict[str, Path]) -> list[dict]:
    picked: dict[tuple[str, str, str], dict] = {}
    for row in manifest:
        product_key = (row["dataset_id"], row["product_id"])
        role = "normal" if row["normal_flag"] == "1" else "defect"
        picked.setdefault((*product_key, role), row)
        if row["mask_path"] and row["normal_flag"] == "0":
            picked.setdefault((*product_key, "mask"), row)
    checks = []
    for (dataset_id, product_id, role), row in sorted(picked.items()):
        relative = row["mask_path"] if role == "mask" else row["image_path"]
        absolute = roots[dataset_id] / relative
        checks.append(
            {
                "dataset_id": dataset_id,
                "product_id": product_id,
                "role": role,
                "relative_path": relative,
                "exists": int(absolute.is_file()),
            }
        )
    return checks


def family_rows(hypotheses: list[dict], visa_products: set[str]) -> list[dict]:
    rows = []
    for item in hypotheses:
        members = item["members"]
        missing = [name for name in members if name not in visa_products]
        rows.append(
            {
                "hypothesis_id": item["id"],
                "members": "|".join(members),
                "missing_members": "|".join(missing),
                "status": "not_applied",
                "note": "主清单仍按对象分开；并组只作审计假设",
            }
        )
    return rows


def realiad_alignment(manifest: list[dict]) -> list[dict]:
    """同一 Sxxxx 在 OK 目录与 NG 目录中的重合。按同一物理件解释，划分前需确认。"""
    per: dict[str, dict] = defaultdict(
        lambda: {
            "ok_train": set(),
            "ok_test": set(),
            "ng": set(),
            "n_ok": 0,
            "n_ng_view_ok": 0,
            "n_ng_defect": 0,
            "n_ng_defect_train": 0,
        }
    )
    for row in manifest:
        if row["dataset_id"] != "realiad":
            continue
        slot = per[row["product_id"]]
        sample_id = row["physical_sample_id"]
        role = row["image_role"]
        if role == "ok_reference" and row["split"] == "train":
            slot["ok_train"].add(sample_id)
            slot["n_ok"] += 1
        elif role == "ok_reference":
            slot["ok_test"].add(sample_id)
            slot["n_ok"] += 1
        elif role == "ng_view_ok":
            slot["ng"].add(sample_id)
            slot["n_ng_view_ok"] += 1
        elif role == "ng_defect":
            slot["ng"].add(sample_id)
            slot["n_ng_defect"] += 1
            if row["split"] == "train":
                slot["n_ng_defect_train"] += 1
    rows = []
    for product_id, slot in sorted(per.items()):
        ng_ids = slot["ng"]
        rows.append(
            {
                "product_id": product_id,
                "n_ok_train_samples": len(slot["ok_train"]),
                "n_ok_test_samples": len(slot["ok_test"]),
                "n_ok_in_both_splits": len(slot["ok_train"] & slot["ok_test"]),
                "n_ng_samples": len(ng_ids),
                "n_ng_also_ok_train": len(ng_ids & slot["ok_train"]),
                "n_ng_also_ok_test": len(ng_ids & slot["ok_test"]),
                "n_ng_without_ok": len(ng_ids - slot["ok_train"] - slot["ok_test"]),
                "n_ok_images": slot["n_ok"],
                "n_ng_view_ok_images": slot["n_ng_view_ok"],
                "n_ng_defect_images": slot["n_ng_defect"],
                "n_ng_defect_train_images": slot["n_ng_defect_train"],
            }
        )
    return rows


def physical_group_id(row: dict) -> str:
    """Real-IAD 物理组主键。禁止跨产品合并相同 Sxxxx。"""
    return f"{row['product_id']}_{row['physical_sample_id']}"


def _realiad_condition(row: dict) -> tuple[str, str]:
    parts = row["image_path"].split("/")
    if "OK" in parts:
        return "OK", ""
    if "NG" in parts:
        ng_index = parts.index("NG")
        defect_subdir = parts[ng_index + 1] if ng_index + 1 < len(parts) else ""
        return "NG", defect_subdir
    return "unknown", ""


def _filename_metadata(row: dict) -> tuple[str, str]:
    filename = Path(row["image_path"]).name
    number_match = SAMPLE_IN_NAME.search(filename)
    timestamp_match = TIMESTAMP_IN_NAME.search(filename)
    filename_sample = f"S{int(number_match.group(1)):04d}" if number_match else ""
    timestamp = timestamp_match.group(1) if timestamp_match else ""
    return filename_sample, timestamp


def build_physical_group_audit(
    manifest: list[dict],
    group_config: dict,
    config_hash: str,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """按 category + Sxxxx 聚合 Real-IAD，并冻结官方基准的无泄漏分组。"""
    policy = group_config["conflict_policy"]
    if policy not in {"preserve_official_test", "full_group_to_test"}:
        raise ValueError(f"不支持的物理组冲突策略：{policy}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        if row["dataset_id"] == "realiad":
            groups[physical_group_id(row)].append(row)

    physical_rows: list[dict] = []
    conflict_rows: list[dict] = []
    statistics: dict[str, dict[str, int | str]] = {}
    train_group_ids: set[str] = set()
    test_group_ids: set[str] = set()
    excluded_train_image_ids: list[str] = []
    by_product_split: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"train_group_ids": [], "test_group_ids": []}
    )

    for group_id, rows in sorted(groups.items()):
        product_id = rows[0]["product_id"]
        sample_id = rows[0]["physical_sample_id"]
        official_splits = sorted({row["split"] for row in rows})
        conflict = len(official_splits) > 1
        assignment = "test" if "test" in official_splits else "train"
        if assignment == "train":
            train_group_ids.add(group_id)
            by_product_split[product_id]["train_group_ids"].append(group_id)
        else:
            test_group_ids.add(group_id)
            by_product_split[product_id]["test_group_ids"].append(group_id)

        labels = sorted(
            {
                label
                for row in rows
                for label in row["defect_labels"].split("|")
                if label
            }
        )
        cameras = sorted({row["camera_id"] for row in rows if row["camera_id"]})
        conditions = {_realiad_condition(row)[0] for row in rows}
        defect_subdirs = sorted(
            {subdir for row in rows for _, subdir in [_realiad_condition(row)] if subdir}
        )
        timestamps = [
            timestamp
            for row in rows
            for _, timestamp in [_filename_metadata(row)]
            if timestamp
        ]
        n_train_images = sum(row["split"] == "train" for row in rows)
        n_test_images = sum(row["split"] == "test" for row in rows)
        if conflict:
            excluded_train_image_ids.extend(
                row["image_id"] for row in rows if row["split"] == "train"
            )
            conflict_rows.append(
                {
                    "physical_group_id": group_id,
                    "category": product_id,
                    "sample_id": sample_id,
                    "official_splits": "|".join(official_splits),
                    "n_images": len(rows),
                    "n_train_images": n_train_images,
                    "n_test_images": n_test_images,
                    "n_ok_reference_images": sum(
                        row["image_role"] == "ok_reference" for row in rows
                    ),
                    "n_ng_view_ok_images": sum(
                        row["image_role"] == "ng_view_ok" for row in rows
                    ),
                    "n_defect_images": sum(row["normal_flag"] == "0" for row in rows),
                    "defect_subdirs": "|".join(defect_subdirs),
                    "defect_labels": "|".join(labels),
                    "camera_ids": "|".join(cameras),
                    "first_timestamp": min(timestamps) if timestamps else "",
                    "last_timestamp": max(timestamps) if timestamps else "",
                    "frozen_assignment": "test",
                    "train_side_action": (
                        "exclude_from_official_train"
                        if policy == "preserve_official_test"
                        else "move_into_group_test"
                    ),
                }
            )

        slot = statistics.setdefault(
            product_id,
            {
                "dataset_id": "realiad",
                "product_id": product_id,
                "n_physical_groups": 0,
                "n_normal_groups": 0,
                "n_defect_groups": 0,
                "n_mixed_normal_defect_groups": 0,
                "n_official_train_only_groups": 0,
                "n_official_test_only_groups": 0,
                "n_cross_split_conflict_groups": 0,
                "n_frozen_train_groups": 0,
                "n_frozen_test_groups": 0,
                "n_excluded_train_images": 0,
            },
        )
        has_normal = any(row["normal_flag"] == "1" for row in rows)
        has_defect = any(row["normal_flag"] == "0" for row in rows)
        slot["n_physical_groups"] += 1
        slot["n_normal_groups"] += int(has_normal)
        slot["n_defect_groups"] += int(has_defect)
        slot["n_mixed_normal_defect_groups"] += int(has_normal and has_defect)
        slot["n_official_train_only_groups"] += int(official_splits == ["train"])
        slot["n_official_test_only_groups"] += int(official_splits == ["test"])
        slot["n_cross_split_conflict_groups"] += int(conflict)
        slot["n_frozen_train_groups"] += int(assignment == "train")
        slot["n_frozen_test_groups"] += int(assignment == "test")
        slot["n_excluded_train_images"] += n_train_images if conflict else 0

        for row in sorted(rows, key=lambda item: item["image_path"]):
            condition, defect_subdir = _realiad_condition(row)
            filename_sample, timestamp = _filename_metadata(row)
            if conflict and row["split"] == "train":
                membership = (
                    "excluded_from_official_train"
                    if policy == "preserve_official_test"
                    else "test"
                )
            else:
                membership = assignment
            physical_rows.append(
                {
                    "physical_group_id": group_id,
                    "category": product_id,
                    "sample_id": sample_id,
                    "filename_sample_id": filename_sample,
                    "sample_id_matches_filename": int(filename_sample == sample_id),
                    "official_split": row["split"],
                    "official_group_splits": "|".join(official_splits),
                    "cross_split_conflict": int(conflict),
                    "frozen_group_assignment": assignment,
                    "effective_membership": membership,
                    "condition_dir": condition,
                    "defect_subdir": defect_subdir,
                    "normal_flag": row["normal_flag"],
                    "image_role": row["image_role"],
                    "defect_labels": row["defect_labels"],
                    "camera_id": row["camera_id"],
                    "capture_timestamp": timestamp,
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "source_index": row["source_index"],
                }
            )

    assert_disjoint_groups(train_group_ids, test_group_ids)
    split_payload = {
        "version": 1,
        "status": "frozen_physical_group_boundary",
        "dataset_id": "realiad",
        "group_key": group_config["realiad_key_format"],
        "conflict_policy": policy,
        "config_hash": config_hash,
        "train_group_ids": sorted(train_group_ids),
        "test_group_ids": sorted(test_group_ids),
        "excluded_official_train_image_ids": sorted(excluded_train_image_ids),
        "by_product": {
            product_id: {
                "train_group_ids": sorted(slot["train_group_ids"]),
                "test_group_ids": sorted(slot["test_group_ids"]),
            }
            for product_id, slot in sorted(by_product_split.items())
        },
        "counts": {
            "n_groups": len(groups),
            "n_train_groups": len(train_group_ids),
            "n_test_groups": len(test_group_ids),
            "n_cross_split_conflict_groups": len(conflict_rows),
            "n_excluded_official_train_images": len(excluded_train_image_ids),
        },
        "note": (
            "本文件冻结物理组边界和官方基准冲突处理，不是 Group K-Fold。"
            "同一 category + Sxxxx 不得跨训练与测试。"
        ),
    }
    return physical_rows, conflict_rows, list(statistics.values()), split_payload


def _sample_conflict_groups(
    conflict_rows: list[dict], sample_size: int, seed: int
) -> list[dict]:
    by_product: dict[str, list[dict]] = defaultdict(list)
    for row in conflict_rows:
        by_product[row["category"]].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    remaining: list[dict] = []
    for product_id in sorted(by_product):
        candidates = sorted(
            by_product[product_id], key=lambda item: item["physical_group_id"]
        )
        first = rng.choice(candidates)
        selected.append(first)
        remaining.extend(row for row in candidates if row is not first)
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    return sorted(selected[:sample_size], key=lambda item: item["physical_group_id"])


def _rigid_feature_inliers(
    image_a: Path,
    image_b: Path,
    nfeatures: int,
    lowe_ratio: float,
    reproj_threshold: float,
) -> int:
    """图像核验诊断项；失败返回 -1，不参与物理组硬判定。"""
    try:
        import cv2
    except ImportError:
        return -1
    gray_a = cv2.imread(str(image_a), cv2.IMREAD_GRAYSCALE)
    gray_b = cv2.imread(str(image_b), cv2.IMREAD_GRAYSCALE)
    if gray_a is None or gray_b is None:
        return -1
    orb = cv2.ORB_create(nfeatures=nfeatures)
    key_a, desc_a = orb.detectAndCompute(gray_a, None)
    key_b, desc_b = orb.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < lowe_ratio * pair[1].distance
    ]
    if len(good) < 6:
        return 0
    points_a = [key_a[match.queryIdx].pt for match in good]
    points_b = [key_b[match.trainIdx].pt for match in good]
    import numpy as np

    _, mask = cv2.estimateAffinePartial2D(
        np.float32(points_a).reshape(-1, 1, 2),
        np.float32(points_b).reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
    )
    return int(mask.sum()) if mask is not None else 0


def verify_realiad_pairs(
    physical_rows: list[dict],
    conflict_rows: list[dict],
    verification_config: dict,
    realiad_root: Path,
    read_images: bool,
) -> list[dict]:
    """固定抽样核验文件名、五视角、时间戳及可选图像刚体对应。"""
    sampled = _sample_conflict_groups(
        conflict_rows,
        int(verification_config["sample_size"]),
        int(verification_config["seed"]),
    )
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in physical_rows:
        by_group[row["physical_group_id"]].append(row)
    by_product: dict[str, list[str]] = defaultdict(list)
    for group_id, rows in by_group.items():
        if any(row["condition_dir"] == "OK" for row in rows):
            by_product[rows[0]["category"]].append(group_id)

    output: list[dict] = []
    camera = str(verification_config["camera_for_inliers"])
    for sampled_row in sampled:
        group_id = sampled_row["physical_group_id"]
        rows = by_group[group_id]
        ok_rows = [row for row in rows if row["condition_dir"] == "OK"]
        ng_rows = [row for row in rows if row["condition_dir"] == "NG"]
        ok_timestamps = sorted(
            {row["capture_timestamp"] for row in ok_rows if row["capture_timestamp"]}
        )
        ng_timestamps = sorted(
            {row["capture_timestamp"] for row in ng_rows if row["capture_timestamp"]}
        )
        camera_ids = sorted({row["camera_id"] for row in rows if row["camera_id"]})
        labels = sorted(
            {
                label
                for row in rows
                for label in row["defect_labels"].split("|")
                if label
            }
        )
        filename_match = all(
            int(row["sample_id_matches_filename"]) == 1 for row in rows
        )
        same_inliers = ""
        control_inliers = ""
        control_group_id = ""
        if read_images:
            ok_camera = next(
                (row for row in ok_rows if row["camera_id"] == camera), None
            )
            ng_camera = next(
                (
                    row
                    for row in ng_rows
                    if row["camera_id"] == camera and row["normal_flag"] == "0"
                ),
                None,
            )
            controls = [
                candidate
                for candidate in sorted(by_product[rows[0]["category"]])
                if candidate != group_id
            ]
            control_group_id = controls[0] if controls else ""
            control_camera = next(
                (
                    row
                    for row in by_group.get(control_group_id, [])
                    if row["condition_dir"] == "OK" and row["camera_id"] == camera
                ),
                None,
            )
            if ok_camera and ng_camera:
                same_inliers = _rigid_feature_inliers(
                    realiad_root / ok_camera["image_path"],
                    realiad_root / ng_camera["image_path"],
                    int(verification_config["orb_nfeatures"]),
                    float(verification_config["lowe_ratio"]),
                    float(verification_config["ransac_reproj_threshold"]),
                )
            if ok_camera and control_camera:
                control_inliers = _rigid_feature_inliers(
                    realiad_root / ok_camera["image_path"],
                    realiad_root / control_camera["image_path"],
                    int(verification_config["orb_nfeatures"]),
                    float(verification_config["lowe_ratio"]),
                    float(verification_config["ransac_reproj_threshold"]),
                )
        output.append(
            {
                "physical_group_id": group_id,
                "category": rows[0]["category"],
                "sample_id": rows[0]["sample_id"],
                "n_images": len(rows),
                "n_ok_images": len(ok_rows),
                "n_ng_images": len(ng_rows),
                "n_cameras": len(camera_ids),
                "camera_ids": "|".join(camera_ids),
                "n_ng_defect_subdirs": len(
                    {row["defect_subdir"] for row in ng_rows if row["defect_subdir"]}
                ),
                "defect_labels": "|".join(labels),
                "all_filename_ids_match": int(filename_match),
                "all_timestamps_parsed": int(
                    all(row["capture_timestamp"] for row in rows)
                ),
                "first_ok_timestamp": ok_timestamps[0] if ok_timestamps else "",
                "first_ng_timestamp": ng_timestamps[0] if ng_timestamps else "",
                "ok_captured_before_ng": int(
                    bool(ok_timestamps and ng_timestamps)
                    and ok_timestamps[0] < ng_timestamps[0]
                ),
                "image_check_enabled": int(read_images),
                "camera_checked": camera if read_images else "",
                "same_group_rigid_inliers": same_inliers,
                "control_group_id": control_group_id,
                "control_rigid_inliers": control_inliers,
                "manual_review_status": "pending",
            }
        )
    return output


def render_physical_group_report(
    path: Path,
    split_payload: dict,
    conflict_rows: list[dict],
    verification_rows: list[dict],
) -> None:
    counts = split_payload["counts"]
    n_filename_ok = sum(
        int(row["all_filename_ids_match"]) for row in verification_rows
    )
    n_timestamp_ok = sum(
        int(row["all_timestamps_parsed"]) for row in verification_rows
    )
    n_ok_before_ng = sum(
        int(row["ok_captured_before_ng"]) for row in verification_rows
    )
    image_rows = [
        row for row in verification_rows if int(row["image_check_enabled"]) == 1
    ]
    image_better = sum(
        int(row["same_group_rigid_inliers"]) > int(row["control_rigid_inliers"])
        for row in image_rows
        if row["same_group_rigid_inliers"] != ""
        and row["control_rigid_inliers"] != ""
    )
    lines = [
        "# Real-IAD 物理样本组审计",
        "",
        f"日期：{date.today().isoformat()}",
        "",
        "## 冻结规则",
        "",
        "同一产品类别下的 `category + Sxxxx` 是 v1 物理分组主键。OK、所有 NG 缺陷子目录、五个相机视角及 Mask 必须随组管理；不跨产品合并相同 Sxxxx。",
        "",
        "官方基准采用 `preserve_official_test`：跨官方 train/test 的完整物理组归入测试组，训练侧正常图从官方训练输入中剔除。`group_split.json` 固化组边界；它不是 Group K-Fold。",
        "",
        "## 统计",
        "",
        f"- 物理组：{counts['n_groups']}",
        f"- 冻结训练组：{counts['n_train_groups']}",
        f"- 冻结测试组：{counts['n_test_groups']}",
        f"- 跨官方划分冲突组：{counts['n_cross_split_conflict_groups']}",
        f"- 从官方训练输入剔除的冲突图像：{counts['n_excluded_official_train_images']}",
        "",
        "训练/测试物理组零交集断言已经通过。断言失败时训练入口必须停止。",
        "",
        "## 固定抽样核验",
        "",
        f"固定抽取 {len(verification_rows)} 个冲突组，覆盖全部产品后再按固定种子补样。",
        f"- 文件名编号与目录 Sxxxx 一致：{n_filename_ok}/{len(verification_rows)}",
        f"- 时间戳可解析：{n_timestamp_ok}/{len(verification_rows)}",
        f"- 首个 OK 时间早于首个 NG：{n_ok_before_ng}/{len(verification_rows)}",
    ]
    if image_rows:
        lines.extend(
            [
                f"- 单视角刚体特征对照已运行：{len(image_rows)} 组；同组内点多于同产品对照组：{image_better}/{len(image_rows)}",
                "- 刚体特征内点只作辅助诊断。视角、照明、缺陷制造和物体重摆会降低该数值，不能据此拆分物理组。",
            ]
        )
    else:
        lines.append("- 本次未读取图像；可用 `python scripts/prepare.py audit --verify-images` 执行有限图像诊断。")
    lines.extend(
        [
            "",
            "元数据与抽样图像支持保守合并，但官方没有明确声明 Sxxxx 的跨状态全局唯一性。因此结论仍表述为 v1 防泄漏规则，不宣称官方逐件对应关系。",
            "",
            "## 产物",
            "",
            "- `output/audit/physical_group_manifest.csv`：逐图像物理组清单，含 Mask、官方划分和有效成员资格。",
            "- `output/audit/cross_split_conflicts.csv`：跨官方划分冲突组。",
            "- `output/audit/group_statistics.csv`：产品级物理组统计。",
            "- `output/audit/group_split.json`：冻结物理组边界和零交集划分。",
            "- `output/audit/realiad_pair_verification.csv`：固定抽样核验结果。",
            "",
            "下一步才是基于冻结组边界设计外层产品留出和内层 OOF，不允许直接按图像执行 Group K-Fold。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _count_flags(flags: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for flag in flags:
        counts[flag["flag_type"]] += 1
    return dict(sorted(counts.items()))


def render_report(
    path: Path,
    config_path: Path,
    config_hash: str,
    manifest: list[dict],
    product_rows: list[dict],
    class_rows: list[dict],
    task_summary: list[dict],
    flags: list[dict],
    quota_rows: list[dict],
    side_rows: list[dict],
    checks: list[dict],
    hypotheses: list[dict],
    alignment_rows: list[dict],
) -> None:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in product_rows:
        by_dataset[row["dataset_id"]].append(row)
    flag_counts = _count_flags(flags)
    failed_checks = [row for row in checks if int(row["exists"]) == 0]
    tight_classes = [
        row
        for row in class_rows
        if row["state"] == "defect" and row["split"] == "all" and int(row["n_samples"]) < 11
    ]
    lines = [
        "# S0 数据审计",
        "",
        f"日期：{date.today().isoformat()}",
        f"配置：`{posix(config_path)}`",
        f"配置哈希：`{config_hash}`",
        "",
        "本轮只读索引。MVTec 只列各类别的 `train`、`test` 和约定掩码文件名；VisA 主清单只读 `split_csv/1cls.csv`；Real-IAD 只读 `realiad_jsons`。没有递归扫盘，没有读取像素，没有做近重复比对。",
        "",
        "合格规则：一个缺陷类在全部独立样本上计数，样本数不少于 `K + Q_min` 才进入该预算的 N-way。MVTec 和 VisA 没有物理样本号，一张图算一个样本。Real-IAD 用路径里的 `Sxxxx`，同一产品下的同一编号算一个样本，多相机位不重复计数。产品至少有一个正常样本，才记为检测任务合格。正常样本不少于 2，才记为可以做留一标定。不预设 5-way。`normal_flag` 是官方图像标签。Real-IAD 另有 `image_role`：`ok_reference` 才是 OK 目录里的正常参照，`ng_view_ok` 是缺陷件上未显出缺陷的视角。",
        "",
        "官方 train/test 另列在 `n_way_official_split`：训练集样本不少于 K，且测试集样本不少于 Q_min。这不是本轮的主合格规则。MVTec 的缺陷图都在 test，所以这一列会是 0。",
        "",
        "## 规模",
        "",
        "| 数据集 | 产品数 | 图像数 | 正常图 | 缺陷图 | 缺陷类数 | 正常样本 | 缺陷样本 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id in ("mvtec", "visa", "realiad"):
        rows = by_dataset.get(dataset_id, [])
        lines.append(
            "| {dataset} | {products} | {images} | {normal} | {defect} | {classes} | {normal_samples} | {defect_samples} |".format(
                dataset=dataset_id,
                products=len(rows),
                images=sum(int(row["n_images"]) for row in rows),
                normal=sum(int(row["n_normal_images"]) for row in rows),
                defect=sum(int(row["n_defect_images"]) for row in rows),
                classes=sum(int(row["n_defect_classes"]) for row in rows),
                normal_samples=sum(int(row["n_normal_samples"]) for row in rows),
                defect_samples=sum(int(row["n_defect_samples"]) for row in rows),
            )
        )
    lines.extend(["", "## 合格任务", ""])
    lines.append("| 数据集 | K | Q_min | 产品数 | 检测合格 | 其中可留一标定 | N-way 中位数 | N>=2 | N>=5 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in task_summary:
        lines.append(
            "| {dataset_id} | {k_shot} | {q_min} | {n_products} | {n_det_qualified} | {n_with_loo} | {median_n_way} | {n_products_nway_ge_2} | {n_products_nway_ge_5} |".format(
                **row
            )
        )
    lines.extend(["", "## 需要注意的产品", ""])
    notes = []
    for row in product_rows:
        if int(row["normal_ref_ok"]) == 0:
            notes.append(f"- `{row['dataset_id']}/{row['product_id']}` 没有正常样本。")
        if int(row["defect_missing_mask"]):
            notes.append(
                f"- `{row['dataset_id']}/{row['product_id']}` 有 {row['defect_missing_mask']} 张缺陷图缺少掩码路径。"
            )
    if notes:
        lines.extend(notes)
    else:
        lines.append("没有产品缺少正常样本，也没有缺陷图缺少掩码路径。")
    mvtec_test_only = [
        row["product_id"]
        for row in product_rows
        if row["dataset_id"] == "mvtec" and row["classes_test_only"] and not row["classes_in_both_splits"] and not row["classes_train_only"]
    ]
    if mvtec_test_only:
        lines.append(
            f"- MVTec 的 {len(mvtec_test_only)} 个产品缺陷类都只出现在 test，官方 train 只有正常图。主合格数仍按全部缺陷样本计算。"
        )

    lines.extend(["", "## 独立样本少于 11 的缺陷类", ""])
    lines.append("这些类做不到 K=1 且 Q_min=10。")
    if not tight_classes:
        lines.append("没有。")
    else:
        lines.append(f"共 {len(tight_classes)} 个。完整列表在 `class_counts.csv`。")
        for row in tight_classes[:40]:
            lines.append(
                f"- `{row['dataset_id']}/{row['product_id']}/{row['label']}`：{row['n_samples']} 个样本，{row['n_images']} 张图。"
            )
        if len(tight_classes) > 40:
            lines.append(f"- 其余 {len(tight_classes) - 40} 个见 `class_counts.csv`。")

    lines.extend(["", "## VisA 产品族假设", ""])
    lines.append("主清单的 `product_family_id` 等于对象名，假设没有并进去。")
    for row in hypotheses:
        lines.append(
            f"- `{row['hypothesis_id']}`：{row['members']}。缺失成员：{row['missing_members'] or '无'}。状态：{row['status']}。"
        )
    lines.extend(["", "未并入主清单的 VisA 对照索引：", ""])
    for row in side_rows:
        lines.append(
            f"- `{row['index']}`：{row['n_rows']} 行，{row['n_objects']} 个对象，标签 {row['labels']}，划分 {row['splits']}。"
        )

    lines.extend(["", "## Real-IAD 编号", ""])
    lines.append(
        "v1 已冻结保守规则：同一产品下，OK 目录和各个 NG 子类中的相同 `Sxxxx` 合并为一个物理样本组。各状态、缺陷子类、相机视角和 Mask 不得跨训练与测试。该规则用于防泄漏，不宣称官方已经确认逐件对应关系。"
    )
    if alignment_rows:
        ng_samples = sum(int(row["n_ng_samples"]) for row in alignment_rows)
        ng_train = sum(int(row["n_ng_also_ok_train"]) for row in alignment_rows)
        ng_test = sum(int(row["n_ng_also_ok_test"]) for row in alignment_rows)
        ng_none = sum(int(row["n_ng_without_ok"]) for row in alignment_rows)
        ok_both = sum(int(row["n_ok_in_both_splits"]) for row in alignment_rows)
        ok_images = sum(int(row["n_ok_images"]) for row in alignment_rows)
        ng_view_ok = sum(int(row["n_ng_view_ok_images"]) for row in alignment_rows)
        ng_defect = sum(int(row["n_ng_defect_images"]) for row in alignment_rows)
        ng_defect_train = sum(int(row["n_ng_defect_train_images"]) for row in alignment_rows)
        lines.append(
            f"OK 目录 {ok_images} 张，其中 train/test 共用编号 {ok_both} 个。NG 缺陷图 {ng_defect} 张，其中 train {ng_defect_train} 张。另有 {ng_view_ok} 张在 NG 路径下但官方标成正常，角色是 `ng_view_ok`，不能直接进入正常参照库。"
        )
        lines.append(
            f"NG 编号 {ng_samples} 个：{ng_train} 个在 OK train 里也出现，{ng_test} 个在 OK test 里也出现，{ng_none} 个没有 OK 目录。若编号确实是同一物理件，官方划分把这 {ng_train} 个缺陷件的正常图放进了 train。"
        )
    lines.extend(["", "## Real-IAD 样本配额", ""])
    dominates = [row for row in quota_rows if row["distribution"] == "sample_dominates"]
    quiet = [row for row in quota_rows if row["distribution"] == "no_sample_dominates"]
    lines.append(
        f"{len(quiet)} 个产品的最大样本占比不超过 2%，本轮不设配额。单样本图像数的中位数是 5，多出来的张数是同一编号下的多相机和多状态。"
    )
    for row in dominates:
        lines.append(
            f"- `{row['product_id']}`：最大 {row['max_images_per_sample']} 张（{row['max_sample_id']}，占比 {row['max_sample_share']}），建议上限 {row['suggested_cap']}。"
        )

    lines.extend(["", "## 标志", ""])
    if not flag_counts:
        lines.append("没有路径重复、样本号解析失败或同图标签冲突。")
    else:
        for name, count in flag_counts.items():
            lines.append(f"- `{name}`：{count}")

    lines.extend(["", "## 存在性抽查", ""])
    lines.append(f"每个产品抽 1 张正常图、1 张缺陷图，以及 1 条缺陷掩码路径（索引里有掩码时）。共 {len(checks)} 次，失败 {len(failed_checks)} 次。")
    for row in failed_checks:
        lines.append(
            f"- 失败 `{row['dataset_id']}/{row['product_id']}` {row['role']}：`{row['relative_path']}`"
        )
    lines.extend(
        [
            "",
            "## 本轮没有做",
            "",
            "像素近重复、同图裁切比对、外层/内层划分、Episode 抽样。外部集不在当前配置的三套数据目录里。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def self_check() -> None:
    sample_id, camera_id, issue = parse_sample_token(
        "OK/S0004/audiojack_0004_OK_C1_20231021130716.jpg"
    )
    assert (sample_id, camera_id, issue) == ("S0004", "C1", "")
    sample_id, camera_id, issue = parse_sample_token(
        "NG/HS/S1013/audiojack_1013_NG_HS_C2_20231021130901.jpg"
    )
    assert (sample_id, camera_id, issue) == ("S1013", "C2", "")


def main(verify_images_override: bool = False) -> None:
    self_check()
    root = project_root()
    source = config_path()
    config = load_config(source)
    output_dir = root / config["output_dir"]
    report_file = root / config["report_path"]
    group_config = config["physical_group"]
    group_report_file = root / group_config["report_path"]
    group_artifacts = group_config["artifacts"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    mvtec_cfg = config["datasets"]["mvtec"]
    visa_cfg = config["datasets"]["visa"]
    realiad_cfg = config["datasets"]["realiad"]

    mvtec_records, mvtec_flags = collect_mvtec(Path(mvtec_cfg["root"]), mvtec_cfg["normal_dirname"])
    visa_records, visa_flags = collect_visa(Path(visa_cfg["root"]), visa_cfg["primary_index"])
    realiad_records, realiad_flags = collect_realiad(
        Path(realiad_cfg["root"]),
        realiad_cfg["index_dir"],
        realiad_cfg["image_subdir"],
    )
    print(
        f"raw records mvtec={len(mvtec_records)} visa={len(visa_records)} realiad={len(realiad_records)}"
    )

    manifest, merge_flags = merge_records(mvtec_records + visa_records + realiad_records)
    k_values = [int(value) for value in config["task_candidates"]["k_shot"]]
    q_values = [int(value) for value in config["task_candidates"]["q_min"]]
    product_rows, class_rows, task_rows, sample_flags, quota_rows, large_rows = build_tables(
        manifest, k_values, q_values
    )
    flags = mvtec_flags + visa_flags + realiad_flags + merge_flags + sample_flags
    task_summary = summarize_tasks(task_rows)
    side_rows = [summarize_side_index(Path(visa_cfg["root"]), rel) for rel in visa_cfg["side_indexes"]]
    visa_products = {row["product_id"] for row in product_rows if row["dataset_id"] == "visa"}
    hypotheses = family_rows(visa_cfg["family_hypotheses"], visa_products)
    alignment_rows = realiad_alignment(manifest)
    roots = {
        "mvtec": Path(mvtec_cfg["root"]),
        "visa": Path(visa_cfg["root"]),
        "realiad": Path(realiad_cfg["root"]),
    }
    checks = existence_checks(manifest, roots)
    config_hash = file_sha256(source)
    physical_rows, conflict_rows, group_statistics, split_payload = (
        build_physical_group_audit(manifest, group_config, config_hash)
    )
    verify_images = bool(
        verify_images_override
        or group_config["verification"].get("check_visual_inliers", False)
    )
    verification_rows = verify_realiad_pairs(
        physical_rows,
        conflict_rows,
        group_config["verification"],
        Path(realiad_cfg["root"]),
        verify_images,
    )

    write_csv(output_dir / "manifest.csv", MANIFEST_FIELDS, manifest)
    write_csv(output_dir / "product_counts.csv", list(product_rows[0].keys()) if product_rows else ["dataset_id"], product_rows)
    write_csv(output_dir / "class_counts.csv", list(class_rows[0].keys()) if class_rows else ["dataset_id"], class_rows)
    write_csv(output_dir / "task_candidates.csv", list(task_rows[0].keys()) if task_rows else ["dataset_id"], task_rows)
    write_csv(output_dir / "task_summary.csv", list(task_summary[0].keys()) if task_summary else ["dataset_id"], task_summary)
    write_csv(output_dir / "family_hypothesis.csv", list(hypotheses[0].keys()) if hypotheses else ["hypothesis_id"], hypotheses)
    write_csv(output_dir / "visa_side_indexes.csv", list(side_rows[0].keys()) if side_rows else ["index"], side_rows)
    write_csv(output_dir / "realiad_sample_quota.csv", list(quota_rows[0].keys()) if quota_rows else ["dataset_id"], quota_rows)
    write_csv(
        output_dir / "realiad_id_alignment.csv",
        list(alignment_rows[0].keys()) if alignment_rows else ["product_id"],
        alignment_rows,
    )
    large_fields = [
        "dataset_id",
        "product_id",
        "physical_sample_id",
        "n_images",
        "n_cameras",
        "splits",
        "labels",
        "suggested_cap",
    ]
    write_csv(output_dir / "realiad_large_samples.csv", large_fields, large_rows)
    flag_fields = ["flag_type", "dataset_id", "product_id", "key", "detail"]
    write_csv(output_dir / "audit_flags.csv", flag_fields, flags)
    write_csv(
        output_dir / "existence_sample.csv",
        ["dataset_id", "product_id", "role", "relative_path", "exists"],
        checks,
    )
    physical_fields = [
        "physical_group_id",
        "category",
        "sample_id",
        "filename_sample_id",
        "sample_id_matches_filename",
        "official_split",
        "official_group_splits",
        "cross_split_conflict",
        "frozen_group_assignment",
        "effective_membership",
        "condition_dir",
        "defect_subdir",
        "normal_flag",
        "image_role",
        "defect_labels",
        "camera_id",
        "capture_timestamp",
        "image_id",
        "image_path",
        "mask_path",
        "source_index",
    ]
    write_csv(
        output_dir / group_artifacts["physical_group_manifest"],
        physical_fields,
        physical_rows,
    )
    conflict_fields = [
        "physical_group_id",
        "category",
        "sample_id",
        "official_splits",
        "n_images",
        "n_train_images",
        "n_test_images",
        "n_ok_reference_images",
        "n_ng_view_ok_images",
        "n_defect_images",
        "defect_subdirs",
        "defect_labels",
        "camera_ids",
        "first_timestamp",
        "last_timestamp",
        "frozen_assignment",
        "train_side_action",
    ]
    write_csv(
        output_dir / group_artifacts["cross_split_conflicts"],
        conflict_fields,
        conflict_rows,
    )
    group_stat_fields = list(group_statistics[0].keys()) if group_statistics else [
        "dataset_id",
        "product_id",
    ]
    write_csv(
        output_dir / group_artifacts["group_statistics"],
        group_stat_fields,
        group_statistics,
    )
    verification_fields = (
        list(verification_rows[0].keys())
        if verification_rows
        else ["physical_group_id"]
    )
    write_csv(
        output_dir / group_artifacts["pair_verification"],
        verification_fields,
        verification_rows,
    )
    split_path = output_dir / group_artifacts["group_split"]
    write_json(split_path, split_payload)
    verified_split = load_verified_group_split(split_path)
    assert verified_split["counts"] == split_payload["counts"]
    render_report(
        report_file,
        source,
        config_hash,
        manifest,
        product_rows,
        class_rows,
        task_summary,
        flags,
        quota_rows,
        side_rows,
        checks,
        hypotheses,
        alignment_rows,
    )
    render_physical_group_report(
        group_report_file,
        split_payload,
        conflict_rows,
        verification_rows,
    )
    failed = sum(1 for row in checks if int(row["exists"]) == 0)
    print(f"manifest={len(manifest)} products={len(product_rows)} flags={len(flags)} existence_failed={failed}")
    print(
        "physical_groups={groups} conflicts={conflicts} excluded_train_images={excluded}".format(
            groups=split_payload["counts"]["n_groups"],
            conflicts=split_payload["counts"]["n_cross_split_conflict_groups"],
            excluded=split_payload["counts"]["n_excluded_official_train_images"],
        )
    )
    print("group_split_disjoint=passed")
    print(f"tables {output_dir}")
    print(f"report {report_file}")
    print(f"group_report {group_report_file}")


if __name__ == "__main__":
    main()
