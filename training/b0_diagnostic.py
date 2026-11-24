"""B0 的短跑样本、RPN 诊断、RoI 诊断和候选预算对照。"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from random import Random
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import box_iou
from common.io import file_sha256, load_config, named_config_path, project_root, write_json
from data_proc.baseline_data import B0DetectionDataset, configure_detection_worker, sample_b0_epoch
from models.detector import build_faster_rcnn
from training.evaluation import _box_iou, _product_macro, _QueryMatches, _score_product
from training.train_b0 import _check_runtime, _prepare_data, _train_epoch, _training_settings
import json
from copy import copy
from pathlib import Path
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.ops import box_iou, clip_boxes_to_image, remove_small_boxes
from common.io import file_sha256, load_config, named_config_path, read_json, write_json
from training.checkpoints import load_checkpoint
from training.train_b0 import _check_runtime, _prepare_data
from collections import defaultdict
from torchvision.ops import box_iou, clip_boxes_to_image
from common.io import file_sha256, load_config, named_config_path, write_json
from data_proc.detection import image_to_tensor
from training.evaluation import evaluate_detector


# b0_short.py

def _group_key(row: dict) -> tuple:
    return row["dataset_id"], row["product_id"], row["physical_group_id"]


def _select_samples(index: list[dict], settings: dict, short: dict) -> tuple[list, list]:
    train_count = short["train_groups_per_dataset"]
    holdout_count = short["holdout_groups_per_dataset"]
    if min(train_count, holdout_count) < 2 or train_count % 2 or holdout_count % 2:
        raise ValueError("每域训练与留出物理组配额均须为正偶数")
    positions = sample_b0_epoch(
        index, seed=settings["seed"], epoch=0,
        sampling={"groups_per_dataset": train_count + holdout_count,
                  "positive_group_fraction": 0.5, "max_views_per_group": 1},
    )
    buckets = defaultdict(list)
    for position in positions:
        row = index[position]
        buckets[row["dataset_id"], row["normal_flag"]].append(row)
    domains = sorted({row["dataset_id"] for row in index})
    train, holdout = [], []
    for domain in domains:
        for normal in (0, 1):
            rows = buckets[domain, normal]
            if len(rows) != (train_count + holdout_count) // 2:
                raise ValueError(f"{domain} 的正常或缺陷物理组不足以构建短跑对照")
            train.extend(rows[:train_count // 2])
            holdout.extend(rows[train_count // 2:])
    groups = [_group_key(row) for row in train + holdout]
    if len(set(groups)) != len(groups):
        raise AssertionError("短跑中同一物理组被重复选择或进入两侧")
    return train, holdout


def _compact_samples(batch):
    # 以 uint8 缓存原图，跨进程传输减少到浮点图像的四分之一。
    return [(image.mul(255).round_().to(torch.uint8), target) if image is not None
            else (None, target) for image, target in batch]


class _CachedSamples:
    def __init__(self, rows: list[dict], decoded: list[tuple]) -> None:
        self.items = [(row, image, target) for row, (image, target) in zip(rows, decoded)
                      if image is not None]
        self.skipped = [target for image, target in decoded if image is None]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, position: int) -> tuple:
        _, image, target = self.items[position]
        # 正式训练 collate 会取出并删除本次批次的字符串 ID。
        return image.float().div_(255), target.copy()


def _prepare_samples(adapter, train_rows, holdout_rows, workers: int, batch_size: int):
    rows = train_rows + holdout_rows
    loader = DataLoader(
        B0DetectionDataset(adapter, rows), batch_size=batch_size, num_workers=workers,
        collate_fn=_compact_samples, worker_init_fn=configure_detection_worker,
    )
    decoded = [sample for batch in loader for sample in batch]
    if len(decoded) != len(rows):
        raise AssertionError("短跑解码数量与抽样数量不一致")
    train = _CachedSamples(train_rows, decoded[:len(train_rows)])
    holdout = _CachedSamples(holdout_rows, decoded[len(train_rows):])
    for sample in (train, holdout):
        expected = {(row["dataset_id"], row["normal_flag"]) for row in rows}
        actual = {(row["dataset_id"], row["normal_flag"]) for row, _, _ in sample.items}
        if actual != expected:
            raise ValueError("过滤空缺陷标注后，短跑缺少某个数据域的正常或缺陷图")
    return train, holdout


@torch.inference_mode()
def _evaluate(model, samples: _CachedSamples, device, config: dict) -> dict:
    model.eval()
    queries = defaultdict(list)
    totals, covered = Counter(), Counter()
    batch_size = config["batch_size"]
    for offset in range(0, len(samples), batch_size):
        batch = samples.items[offset:offset + batch_size]
        images = [image.to(device).float().div_(255) for _, image, _ in batch]
        original_sizes = [image.shape[-2:] for image in images]
        targets = [{"boxes": target["boxes"].to(device)} for _, _, target in batch]
        images, transformed_targets = model.transform(images, targets)
        features = model.backbone(images.tensors)
        proposals, _ = model.rpn(images, features)
        detections, _ = model.roi_heads(features, proposals, images.image_sizes)
        detections = model.transform.postprocess(detections, images.image_sizes, original_sizes)
        for (row, _, target), scaled, proposal, detection in zip(
            batch, transformed_targets, proposals, detections
        ):
            domain = row["dataset_id"]
            gt = scaled["boxes"]
            totals[domain] += len(gt)
            if len(gt) and len(proposal):
                covered[domain] += int((box_iou(proposal[:50], gt).max(dim=0).values >= 0.5).sum())
            boxes = detection["boxes"].cpu().numpy()
            scores = detection["scores"].cpu().numpy()
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise FloatingPointError("B0 短跑返回非有限预测")
            order = np.argsort(-scores, kind="stable")
            annotations = [{"bbox": box} for box in target["boxes"].tolist()]
            queries[f"{domain}:{row['product_id']}"].append(_QueryMatches(
                scores=scores[order], positive_iou=_box_iou(boxes[order], annotations),
                ignore_overlap=np.zeros(len(boxes), dtype=bool), pure_normal=bool(row["normal_flag"]),
            ))
    per_product = {name: _score_product(items, config) for name, items in queries.items()}
    return {
        "images": len(samples),
        "rpn_recall_at_50": sum(covered.values()) / sum(totals.values()),
        "product_macro": _product_macro(list(per_product.values())),
        "per_product": per_product,
        "rpn_by_dataset": {name: {"matched_boxes": covered[name], "gt_boxes": total}
                           for name, total in sorted(totals.items())},
    }


def run_short(run_name: str, variant: str, workers: int | None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run-name 只能使用字母、数字、连字符和下划线")
    start = perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    baseline_path = named_config_path("baseline.yaml")
    short_path = named_config_path("b0_short.yaml")
    baseline, short = load_config(baseline_path), load_config(short_path)
    if variant not in short["variants"]:
        raise ValueError(f"未知短跑组：{variant}")
    sizes = short["variants"][variant]
    if len(sizes) != 5 or any(size <= 0 for size in sizes):
        raise ValueError("FPN 的五层各需配置一个正的锚框尺度")
    workers = short["workers"] if workers is None else workers
    if workers < 0 or min(short[key] for key in (
        "epochs", "evaluate_every", "decode_batch_size", "cpu_threads", "eval_batch_size"
    )) < 1:
        raise ValueError("短跑轮数、间隔、批大小和线程数必须为正，解码进程数可为零")
    settings = _training_settings(baseline, False)
    settings.update(families_per_dataset=short["families_per_dataset"], num_workers=0)
    evaluation = {**baseline["evaluation"], "batch_size": short["eval_batch_size"]}
    device = _check_runtime(settings)
    torch.set_num_threads(short["cpu_threads"])
    configure_detection_worker(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    adapter, index, _, provenance = _prepare_data(baseline, settings, pilot=True)
    train_rows, holdout_rows = _select_samples(index, settings, short)
    print(f"variant={variant} workers={workers} train_groups={len(train_rows)} "
          f"holdout_groups={len(holdout_rows)}", flush=True)
    decode_start = perf_counter()
    train, holdout = _prepare_samples(adapter, train_rows, holdout_rows, workers, short["decode_batch_size"])
    decode_seconds = perf_counter() - decode_start

    torch.manual_seed(settings["seed"])
    model = build_faster_rcnn(settings["model"])
    ratios = model.rpn.anchor_generator.aspect_ratios
    model.rpn.anchor_generator = AnchorGenerator(tuple((size,) for size in sizes), ratios)
    initialization = hashlib.sha256()
    for tensor in model.state_dict().values():
        initialization.update(tensor.numpy().tobytes())
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"],
        weight_decay=settings["weight_decay"],
    )
    directory = project_root() / short["output_dir"] / run_name
    directory.mkdir(parents=True, exist_ok=False)
    result = {
        "variant": variant, "anchor_sizes": sizes, "seed": settings["seed"],
        "initialization_sha256": initialization.hexdigest(),
        "config_sha256": {"baseline": file_sha256(baseline_path), "short": file_sha256(short_path)},
        "provenance": provenance, "workers": workers, "eval_batch_size": evaluation["batch_size"],
        "training_batch_size": settings["batch_size"], "decode_seconds": decode_seconds,
        "samples": {name: {"image_ids": [row["image_id"] for row, _, _ in sample.items],
                           "skipped": sample.skipped}
                    for name, sample in (("train", train), ("holdout", holdout))},
        "history": [], "complete": False,
    }
    global_step = 0
    for epoch in range(1, short["epochs"] + 1):
        positions = list(range(len(train)))
        Random(f"{settings['seed']}:{epoch}").shuffle(positions)
        tick = perf_counter()
        training = _train_epoch(model, optimizer, train, positions, device, settings, epoch, global_step)
        global_step += training["steps"]
        entry = {"epoch": epoch, "training": training, "train_seconds": perf_counter() - tick}
        if epoch % short["evaluate_every"] == 0 or epoch == short["epochs"]:
            tick = perf_counter()
            entry["train_evaluation"] = _evaluate(model, train, device, evaluation)
            entry["holdout_evaluation"] = _evaluate(model, holdout, device, evaluation)
            entry["evaluation_seconds"] = perf_counter() - tick
            holdout_metrics = entry["holdout_evaluation"]
            print(f"epoch={epoch} steps={global_step} loss={training['loss']['total']:.6f} "
                  f"holdout_rpn50={holdout_metrics['rpn_recall_at_50']:.4f} "
                  f"holdout_ap50={holdout_metrics['product_macro']['ap50']:.4f}", flush=True)
        result["history"].append(entry)
        result["complete"] = epoch == short["epochs"]
        result["duration_seconds"] = perf_counter() - start
        write_json(directory / "metrics.json", result)
    return result

# b0_rpn_diagnostic.py

POST_NMS_BUDGETS = (50, 100, 300)
STAGES = ("anchors", "decoded", "pre_nms", "post_nms_all") + tuple(
    f"post_nms_top{budget}" for budget in POST_NMS_BUDGETS
)
IOU_THRESHOLD = 0.5


def _valid_boxes(boxes, image_size, min_size):
    boxes = clip_boxes_to_image(boxes, image_size)
    return boxes, remove_small_boxes(boxes, min_size)


def _rpn_stages(rpn, images, features):
    feature_list = list(features.values())
    level_scores, level_deltas = rpn.head(feature_list)
    level_counts = [level[0].numel() for level in level_scores]
    anchors = rpn.anchor_generator(images, feature_list)
    objectness, deltas = concat_box_prediction_layers(level_scores, level_deltas)
    proposals = rpn.box_coder.decode(deltas, anchors).reshape(len(anchors), -1, 4)
    scores = objectness.reshape(len(anchors), -1)
    top = rpn._get_top_n_idx(scores, level_counts)
    # 复用原版筛选和 NMS，只在诊断用的副本上去掉最终数量上限。
    uncapped_rpn = copy(rpn)
    uncapped_rpn._post_nms_top_n = {**rpn._post_nms_top_n, "testing": top.shape[1]}
    after_nms, _ = uncapped_rpn.filter_proposals(
        proposals, objectness, images.image_sizes, level_counts
    )

    for original, decoded, logits, order, boxes, image_size in zip(
        anchors, proposals, scores, top, after_nms, images.image_sizes
    ):
        valid_decoded, keep_all = _valid_boxes(decoded, image_size, rpn.min_size)
        pre_nms, keep_size = _valid_boxes(decoded[order], image_size, rpn.min_size)
        keep_score = logits[order][keep_size].sigmoid() >= rpn.score_thresh
        yield (original, valid_decoded[keep_all], pre_nms[keep_size[keep_score]],
               boxes, *(boxes[:budget] for budget in POST_NMS_BUDGETS))


@torch.inference_mode()
def _coverage(model, samples, device, batch_size: int) -> dict:
    model.eval()
    domains = defaultdict(Counter)
    for start in range(0, len(samples), batch_size):
        batch = samples.items[start:start + batch_size]
        images = [image.to(device).float().div_(255) for _, image, _ in batch]
        targets = [{"boxes": target["boxes"].to(device)} for _, _, target in batch]
        images, targets = model.transform(images, targets)
        features = model.backbone(images.tensors)
        layers = _rpn_stages(model.rpn, images, features)
        for (row, _, _), target, stages in zip(batch, targets, layers):
            gt = target["boxes"]
            counts = domains[row["dataset_id"]]
            counts["images"] += 1
            counts["gt_boxes"] += len(gt)
            for name, boxes in zip(STAGES, stages):
                counts[f"{name}_candidates"] += len(boxes)
                if len(gt) and len(boxes):
                    matches = box_iou(boxes, gt).max(dim=0).values >= IOU_THRESHOLD
                    counts[name] += int(matches.sum())

    def summarize(counts: Counter) -> dict:
        total, images = counts["gt_boxes"], counts["images"]
        return {
            "images": images, "gt_boxes": total,
            "stages": {
                name: {"covered": counts[name], "recall": counts[name] / total if total else None,
                       "mean_candidates_per_image": counts[f"{name}_candidates"] / images if images else None}
                for name in STAGES
            },
        }

    combined = sum(domains.values(), Counter())
    return {"overall": summarize(combined),
            "by_dataset": {domain: summarize(counts) for domain, counts in sorted(domains.items())}}


def diagnose_rpn(checkpoint: Path, samples: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"诊断输出已存在：{output}")
    model_state = load_checkpoint(checkpoint)
    checkpoint_source = model_state["provenance"]
    settings = checkpoint_source["settings"]
    device = _check_runtime(settings)
    short = load_config(named_config_path("b0_short.yaml"))
    torch.set_num_threads(short["cpu_threads"])
    sample_metrics = read_json(samples)
    if not sample_metrics["complete"]:
        raise ValueError("请使用完整的短跑样本清单")
    baseline_path = named_config_path("baseline.yaml")
    if file_sha256(baseline_path) != checkpoint_source["config_sha256"]:
        raise ValueError("当前 baseline.yaml 与 checkpoint 记录的配置不一致")
    baseline = load_config(baseline_path)
    adapter, index, _, current = _prepare_data(baseline, settings, pilot=False)
    for key in ("manifest_sha256", "split_manifest_sha256", "episode_manifest_sha256"):
        if any(source[key] != current[key] for source in (checkpoint_source, sample_metrics["provenance"])):
            raise ValueError(f"诊断数据来源不一致：{key}")

    rows = {row["image_id"]: row for row in index}
    train_rows, holdout_rows = (
        [rows[image_id] for image_id in sample_metrics["samples"][split]["image_ids"]]
        for split in ("train", "holdout")
    )
    keys = [_group_key(row) for row in train_rows + holdout_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("短跑样本有重复物理组")
    train, holdout = _prepare_samples(adapter, train_rows, holdout_rows, 0, short["decode_batch_size"])
    if train.skipped or holdout.skipped:
        raise ValueError("诊断重新解码时出现了短跑期间没有跳过的样本")

    model = build_faster_rcnn(settings["model"], initialize_backbone=False)
    model.load_state_dict(model_state["model"])
    model.to(device)
    model.eval()
    batch_size = checkpoint_source["evaluation"]["batch_size"]
    result = {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_epoch": model_state["epoch"],
        "samples": str(samples.resolve()), "iou_threshold": IOU_THRESHOLD, "batch_size": batch_size,
        "rpn": {"pre_nms_top_n_per_fpn_level": model.rpn.pre_nms_top_n(),
                "score_threshold": model.rpn.score_thresh, "nms_iou_threshold": model.rpn.nms_thresh,
                "checkpoint_post_nms_top_n": model.rpn.post_nms_top_n()},
    }
    for name, dataset in (("short_train", train), ("short_holdout", holdout)):
        result[name] = _coverage(model, dataset, device, batch_size)
        print(name, json.dumps(result[name]["overall"], ensure_ascii=False), flush=True)
    write_json(output, result)
    print(f"诊断结果 {output}", flush=True)
    return result

# b0_roi_diagnostic.py

ROLES = ("query_defect", "query_normal")
PROBES = ("oracle_gt", "jittered_gt", "matched_rpn_top50", "matched_rpn_top300", "normal_rpn_top50")
JITTER_FRACTION = 0.2


def _sample_queries(adapter, episodes, limit: int) -> list[dict]:
    """固定 Episode 顺序抽样，每个产品的每种角色优先覆盖不同物理组。"""
    chosen = defaultdict(list)
    seen_images = set()
    seen_groups = set()
    products = set()
    for episode in sorted(episodes, key=lambda item: item["episode_id"]):
        product = (episode["dataset_id"], episode["product_id"])
        products.add(product)
        classes = set(episode["episode_classes"])
        for role in ROLES:
            records = chosen[product, role]
            for unit in sorted(episode[role], key=lambda item: item["physical_group_id"]):
                if len(records) == limit:
                    break
                group_key = (*product, unit["physical_group_id"])
                if group_key in seen_groups:
                    continue
                for image_id in sorted(unit["image_ids"]):
                    row = adapter.manifest[image_id]
                    normal = int(row["normal_flag"]) == 1
                    labels = set(row["defect_labels"].split("|"))
                    eligible = normal if role == "query_normal" else not normal and bool(labels & classes)
                    if image_id in seen_images or not eligible:
                        continue
                    probe_episode = {**episode, role: [{**unit, "image_ids": [image_id]}]}
                    record, = adapter.build_episode_records(probe_episode, roles=(role,))
                    if role == "query_defect" and not record["annotations"]:
                        continue
                    if role == "query_normal" and (record["annotations"] or record["ignore_annotations"]):
                        raise AssertionError(f"正常 Query 中出现标注：{image_id}")
                    records.append(record)
                    seen_images.add(image_id)
                    seen_groups.add(group_key)
                    break
    missing = [(product, role) for product in sorted(products) for role in ROLES if not chosen[product, role]]
    if missing:
        raise ValueError(f"诊断样本缺少产品或角色：{missing}")
    return [record for key in sorted(chosen) for record in chosen[key]]


def _jitter_boxes(gt, image_size):
    """把真值框水平移向图像内部，生成已知初始 IoU 的可控提案。"""
    _, width = image_size
    right_room = width - gt[:, 2]
    left_room = gt[:, 0]
    distance = (gt[:, 2] - gt[:, 0]) * JITTER_FRACTION
    shift = torch.where(right_room >= left_room, distance, -distance)
    jittered = gt.clone()
    jittered[:, 0] += shift
    jittered[:, 2] += shift
    return clip_boxes_to_image(jittered, image_size)


def _foreground_predictions(roi_heads, features, boxes, image_size, defect_id):
    if not len(boxes):
        return boxes.new_empty(0), boxes
    pooled = roi_heads.box_roi_pool(features, [boxes], [image_size])
    logits, deltas = roi_heads.box_predictor(roi_heads.box_head(pooled))
    scores = logits.softmax(dim=1)[:, defect_id]
    regressed = roi_heads.box_coder.decode(deltas, [boxes])[:, defect_id]
    regressed = clip_boxes_to_image(regressed, image_size)
    if not (torch.isfinite(scores).all() and torch.isfinite(regressed).all()):
        raise FloatingPointError("RoI 预测包含非有限值")
    return scores, regressed


@torch.inference_mode()
def _probe_image(model, record, device, defect_id: int) -> dict:
    image = image_to_tensor(record["image_path"]).to(device)
    gt = image.new_tensor([item["bbox"] for item in record["annotations"]]).reshape(-1, 4)
    images, targets = model.transform([image], [{"boxes": gt}])
    image_size, gt = images.image_sizes[0], targets[0]["boxes"]
    features = model.backbone(images.tensors)
    result = {key: record[key] for key in (
        "dataset", "product_id", "image_id", "physical_group_id", "episode_id", "split_role"
    )}
    result.update(gt_boxes=len(gt), gt_short_sides_px=(gt[:, 2:] - gt[:, :2]).min(dim=1).values.tolist(),
                  rpn_top50_gt_covered=0, rpn_top300_gt_covered=0, probes={})
    if not len(gt):
        proposals, _ = model.rpn(images, features)
        scores, _ = _foreground_predictions(model.roi_heads, features, proposals[0][:50], image_size, defect_id)
        result["probes"]["normal_rpn_top50"] = {"foreground_scores": scores.tolist()}
        return result

    stages, = _rpn_stages(model.rpn, images, features)
    top50 = stages[STAGES.index("post_nms_top50")]
    top300 = stages[STAGES.index("post_nms_top300")]
    for cap, boxes in ((50, top50), (300, top300)):
        if len(boxes):
            result[f"rpn_top{cap}_gt_covered"] = int((box_iou(boxes, gt).max(dim=0).values >= IOU_THRESHOLD).sum())
    scores, matches = box_iou(top300, gt).max(dim=1)
    valid = scores >= IOU_THRESHOLD
    candidates = {
        "oracle_gt": (gt, gt),
        "jittered_gt": (_jitter_boxes(gt, image_size), gt),
        "matched_rpn_top300": (top300[valid], gt[matches[valid]]),
    }
    counts = [len(boxes) for boxes, _ in candidates.values()]
    foreground, regressed = _foreground_predictions(
        model.roi_heads, features, torch.cat([boxes for boxes, _ in candidates.values()]), image_size, defect_id
    )
    for (kind, (proposals, matched_gt)), confidence, prediction in zip(
        candidates.items(), foreground.split(counts), regressed.split(counts)
    ):
        result["probes"][kind] = {
            "foreground_scores": confidence.tolist(),
            "before_iou": box_iou(proposals, matched_gt).diagonal().tolist(),
            "after_iou": box_iou(prediction, matched_gt).diagonal().tolist(),
        }
    is_top50 = (valid.nonzero().flatten() < len(top50)).tolist()
    result["probes"]["matched_rpn_top50"] = {
        key: [value for value, keep in zip(values, is_top50) if keep]
        for key, values in result["probes"]["matched_rpn_top300"].items()
    }
    return result


def _summary(rows: list[dict], threshold: float) -> dict:
    result = {
        "defect_images": sum(row["split_role"] == "QUERY_DEFECT" for row in rows),
        "normal_images": sum(row["split_role"] == "QUERY_NORMAL" for row in rows),
        "gt_boxes": sum(row["gt_boxes"] for row in rows),
        "rpn_top50_gt_covered": sum(row["rpn_top50_gt_covered"] for row in rows),
        "rpn_top300_gt_covered": sum(row["rpn_top300_gt_covered"] for row in rows),
        "probes": {},
    }
    for kind in PROBES:
        entries = [row["probes"][kind] for row in rows if kind in row["probes"]]
        scores = _probe_scores(entries)
        group = {"boxes": int(scores.size)}
        if scores.size:
            group.update(foreground_score_median=float(np.median(scores)),
                         foreground_score_p90=float(np.percentile(scores, 90)),
                         foreground_scores_at_threshold=int(np.count_nonzero(scores >= threshold)),
                         foreground_fraction_at_threshold=float(np.mean(scores >= threshold)))
            if kind != "normal_rpn_top50":
                before = np.asarray([value for entry in entries for value in entry["before_iou"]])
                after = np.asarray([value for entry in entries for value in entry["after_iou"]])
                delta = after - before
                group.update(before_iou_median=float(np.median(before)),
                             after_iou_median=float(np.median(after)),
                             mean_iou_change=float(np.mean(delta)),
                             fraction_improved=float(np.mean(delta > 1e-6)),
                             fraction_after_iou_at_least_0_5=float(np.mean(after >= IOU_THRESHOLD)))
        if kind == "normal_rpn_top50":
            group["images_at_threshold"] = sum(
                any(score >= threshold for score in entry["foreground_scores"]) for entry in entries
            )
        result["probes"][kind] = group
    short_sides = [size for row in rows for size in row["gt_short_sides_px"]]
    result["gt_short_side_px_median"] = float(np.median(short_sides)) if short_sides else None
    positive = _probe_scores([row["probes"]["oracle_gt"] for row in rows if "oracle_gt" in row["probes"]])
    negative = np.sort(_probe_scores([
        row["probes"]["normal_rpn_top50"] for row in rows if "normal_rpn_top50" in row["probes"]
    ]))
    if positive.size and negative.size:
        ranks = (np.searchsorted(negative, positive, side="left")
                 + np.searchsorted(negative, positive, side="right")) / 2
        result["oracle_vs_normal_roi_auroc"] = float(ranks.mean() / negative.size)
    return result


def _probe_scores(entries: list[dict]) -> np.ndarray:
    return np.asarray([score for entry in entries for score in entry["foreground_scores"]])


def _probe_summary(model, records, device, defect_id: int, threshold: float) -> dict:
    rows = [_probe_image(model, record, device, defect_id) for record in records]
    by_product, by_dataset = defaultdict(list), defaultdict(list)
    for row in rows:
        by_product[f"{row['dataset']}:{row['product_id']}"].append(row)
        by_dataset[row["dataset"]].append(row)
    return {
        "overall": _summary(rows, threshold),
        "by_dataset": {name: _summary(items, threshold) for name, items in sorted(by_dataset.items())},
        "by_product": {name: _summary(items, threshold) for name, items in sorted(by_product.items())},
        "samples": rows,
    }


def diagnose_roi(
    checkpoint: Path, output: Path, images_per_role_per_product: int,
    dataset: str | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"诊断输出已存在：{output}")
    if images_per_role_per_product < 1:
        raise ValueError("每个产品各角色的抽样量须为正整数")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    state = load_checkpoint(checkpoint)
    source = state["provenance"]
    settings, evaluation = source["settings"], source["evaluation"]
    device = _check_runtime(settings)
    torch.set_num_threads(load_config(named_config_path("b0_short.yaml"))["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    baseline_path = named_config_path("baseline.yaml")
    if file_sha256(baseline_path) != source["config_sha256"]:
        raise ValueError("当前 baseline.yaml 与 checkpoint 记录的配置不一致")
    adapter, _, episodes, current = _prepare_data(load_config(baseline_path), settings, pilot=False)
    for key in ("manifest_sha256", "split_manifest_sha256", "episode_manifest_sha256"):
        if source[key] != current[key]:
            raise ValueError(f"诊断数据来源不一致：{key}")
    if dataset:
        episodes = [episode for episode in episodes if episode["dataset_id"] == dataset]
        if not episodes:
            raise ValueError(f"Meta-Val 没有数据域：{dataset}")
    records = _sample_queries(adapter, episodes, images_per_role_per_product)
    print(f"RoI 诊断使用 {len(records)} 张固定 Meta-Val Query 图像", flush=True)

    model = build_faster_rcnn(settings["model"], initialize_backbone=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    defect_id = int(adapter.config["classes"]["defect"])
    threshold = evaluation["score_threshold"]
    result = {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_epoch": state["epoch"],
        "episode_manifest_sha256": current["episode_manifest_sha256"],
        "dataset": dataset,
        "images_per_role_per_product_cap": images_per_role_per_product,
        "score_threshold": threshold, "iou_threshold": IOU_THRESHOLD,
        "horizontal_jitter_box_width_fraction": JITTER_FRACTION,
    }
    result.update(_probe_summary(model, records, device, defect_id, threshold))
    write_json(output, result)
    print(json.dumps(result["overall"], ensure_ascii=False), flush=True)
    print(f"RoI 诊断结果 {output}", flush=True)
    return result

# b0_rpn_budget.py

WARMUP_STEPS = 5


class _TimedDetector(torch.nn.Module):
    """只计模型前向，排除读图、输入传输和指标计算。"""

    def __init__(self, model, device) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.latency_ms = []

    def forward(self, images):
        torch.cuda.synchronize(self.device)
        start = perf_counter()
        predictions = self.model(images)
        torch.cuda.synchronize(self.device)
        self.latency_ms.append((perf_counter() - start) * 1000 / len(images))
        return predictions


def sweep_rpn_budgets(checkpoint: Path, output: Path, budgets) -> dict:
    if output.exists():
        raise FileExistsError(f"对照输出已存在：{output}")
    if not budgets or min(budgets) < 1 or len(set(budgets)) != len(budgets):
        raise ValueError("候选预算必须是互不重复的正整数")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    state = load_checkpoint(checkpoint)
    source = state["provenance"]
    settings, evaluation = source["settings"], source["evaluation"]
    device = _check_runtime(settings)
    short = load_config(named_config_path("b0_short.yaml"))
    torch.set_num_threads(short["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    baseline_path = named_config_path("baseline.yaml")
    if file_sha256(baseline_path) != source["config_sha256"]:
        raise ValueError("当前 baseline.yaml 与 checkpoint 记录的配置不一致")
    adapter, _, episodes, current = _prepare_data(load_config(baseline_path), settings, pilot=False)
    for key in ("manifest_sha256", "split_manifest_sha256", "episode_manifest_sha256"):
        if source[key] != current[key]:
            raise ValueError(f"诊断数据来源不一致：{key}")

    model = build_faster_rcnn(settings["model"], initialize_backbone=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    warmup = adapter.build_episode_records(episodes[0], roles=("query_normal",))
    warmup_images = [image_to_tensor(row["image_path"]).to(device)
                     for row in warmup[:evaluation["batch_size"]]]
    result = {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_epoch": state["epoch"],
        "episode_manifest_sha256": current["episode_manifest_sha256"],
        "evaluation": evaluation, "warmup_steps_per_budget": WARMUP_STEPS,
        "budgets": list(budgets), "runs": [], "complete": False,
    }
    reference_counts = None
    for budget in budgets:
        model.rpn._post_nms_top_n["testing"] = budget
        print(f"开始 Top-{budget} 评价，使用 {len(episodes)} 条 Meta-Val Episode", flush=True)
        with torch.inference_mode():
            for _ in range(WARMUP_STEPS):
                model(warmup_images)
            torch.cuda.synchronize(device)
        timed = _TimedDetector(model, device).eval()
        start = perf_counter()
        metrics = evaluate_detector(timed, episodes, adapter, device, evaluation)
        eval_seconds = perf_counter() - start
        if reference_counts is not None and metrics["counts"] != reference_counts:
            raise AssertionError("候选预算对照的 Query 样本数不一致")
        reference_counts = metrics["counts"]
        result["runs"].append({
            "rpn_post_nms_top_n": budget,
            "metrics": metrics,
            "latency": {"mean_ms_per_image": float(np.mean(timed.latency_ms)),
                        "median_ms_per_image": float(np.median(timed.latency_ms)),
                        "p95_ms_per_image": float(np.percentile(timed.latency_ms, 95)),
                        "model_calls": len(timed.latency_ms), "evaluation_seconds": eval_seconds},
        })
        result["complete"] = len(result["runs"]) == len(budgets)
        write_json(output, result)
        summary = metrics["product_macro"]
        print(json.dumps({"top_k": budget, **summary, **result["runs"][-1]["latency"]},
                         ensure_ascii=False), flush=True)
    print(f"候选预算对照结果 {output}", flush=True)
    return result
