"""固定 Episode 的二分类缺陷框评价。"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from statistics import mean

import numpy as np
import torch

from data_proc.detection import image_to_tensor


@dataclass
class _QueryMatches:
    scores: np.ndarray
    positive_iou: np.ndarray
    ignore_overlap: np.ndarray
    pure_normal: bool


def _box_iou(boxes: np.ndarray, targets: list[dict]) -> np.ndarray:
    target_boxes = np.asarray([target["bbox"] for target in targets], dtype=float).reshape(-1, 4)
    left = np.maximum(boxes[:, None, :2], target_boxes[None, :, :2])
    right = np.minimum(boxes[:, None, 2:], target_boxes[None, :, 2:])
    intersection = np.maximum(right - left, 0).prod(axis=2)
    box_area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    target_area = (target_boxes[:, 2:] - target_boxes[:, :2]).prod(axis=1)
    union = box_area[:, None] + target_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _match(query: _QueryMatches, threshold: float) -> np.ndarray:
    """按置信度依次匹配，每个有效真值最多匹配一次。"""
    outcomes = np.where(query.ignore_overlap, 0, -1)
    unmatched = np.ones(query.positive_iou.shape[1], dtype=bool)
    for index, ious in enumerate(query.positive_iou):
        candidates = np.flatnonzero(unmatched & (ious >= threshold))
        if candidates.size:
            target = candidates[ious[candidates].argmax()]
            unmatched[target] = False
            outcomes[index] = 1
    return outcomes


def _average_precision(outcomes: np.ndarray, positives: int) -> float:
    scored = outcomes[outcomes != 0]
    if not scored.size:
        return 0.0
    cumulative_tp = np.cumsum(scored == 1)
    recall = cumulative_tp / positives
    precision = cumulative_tp / np.arange(1, len(scored) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_steps = np.linspace(0, 1, 101)
    indices = np.searchsorted(recall, recall_steps)
    return float(np.append(envelope, 0)[indices].mean())


def _score_product(queries: list[_QueryMatches], config: dict) -> dict:
    positives = sum(query.positive_iou.shape[1] for query in queries)
    normal_images = sum(query.pure_normal for query in queries)
    scores = np.concatenate([query.scores for query in queries])
    score_order = np.argsort(-scores, kind="stable")
    working_iou = config["working_iou"]
    working_outcomes = [_match(query, working_iou) for query in queries]
    outcomes = np.concatenate(working_outcomes)
    at_working_point = outcomes[scores >= config["score_threshold"]]
    true_positives = np.count_nonzero(at_working_point == 1)
    false_positives = np.count_nonzero(at_working_point == -1)
    normal_fp = sum(
        np.count_nonzero((matches == -1) & (query.scores >= config["score_threshold"]))
        for query, matches in zip(queries, working_outcomes)
        if query.pure_normal
    )
    ap_by_iou = {}
    if positives:
        for iou in config["iou_thresholds"]:
            matches = outcomes
            if iou != working_iou:
                matches = np.concatenate([_match(query, iou) for query in queries])
            ap_by_iou[iou] = _average_precision(matches[score_order], positives)
    return {
        "ap50": ap_by_iou.get(0.5),
        "ap": mean(ap_by_iou.values()) if ap_by_iou else None,
        "precision": float(true_positives / max(true_positives + false_positives, 1)),
        "recall": float(true_positives / positives) if positives else None,
        "normal_fp_per_image": float(normal_fp / normal_images) if normal_images else None,
        "query_images": len(queries),
        "normal_query_images": normal_images,
        "positive_boxes": positives,
        "normal_false_positives": int(normal_fp),
    }


def recall_at_normal_fp(queries: list[_QueryMatches], working_iou: float, fp_per_image_budget) -> dict:
    """在不超过参照正常图误报率的最高置信度前缀上计算缺陷召回。"""
    positives = sum(query.positive_iou.shape[1] for query in queries)
    normal_images = sum(bool(query.pure_normal) for query in queries)
    if not positives or not normal_images or fp_per_image_budget is None:
        return {"recall": None, "normal_fp_per_image": None, "score_threshold": None}
    detections = []
    for query in queries:
        outcomes = _match(query, working_iou)
        for score, outcome in zip(query.scores, outcomes):
            if outcome == 0:
                continue
            detections.append((float(score), int(outcome), bool(query.pure_normal)))
    detections.sort(key=lambda item: -item[0])
    true_positives = 0
    normal_fp = 0
    chosen = {"recall": 0.0, "normal_fp_per_image": 0.0, "score_threshold": None}
    for score, outcome, pure_normal in detections:
        next_fp = normal_fp + int(pure_normal and outcome < 0)
        if next_fp / normal_images > float(fp_per_image_budget) + 1e-8:
            break
        normal_fp = next_fp
        true_positives += int(outcome > 0)
        chosen = {
            "recall": true_positives / positives,
            "normal_fp_per_image": normal_fp / normal_images,
            "score_threshold": score,
        }
    return chosen


def operating_point_comparison(reference: dict, candidate: dict, config: dict) -> dict:
    """按产品比较 AP50，以及不超过参照正常图误报率时的缺陷召回。"""
    if reference.keys() != candidate.keys():
        raise AssertionError("配对产品不一致")
    points = {}
    recall_deltas = []
    ap_deltas = []
    for key in sorted(reference):
        ref = _score_product(reference[key], config)
        cand = _score_product(candidate[key], config)
        matched_ref = recall_at_normal_fp(
            reference[key], config["working_iou"], ref["normal_fp_per_image"],
        )
        matched_cand = recall_at_normal_fp(
            candidate[key], config["working_iou"], ref["normal_fp_per_image"],
        )
        recall_delta = None
        if matched_ref["recall"] is not None and matched_cand["recall"] is not None:
            recall_delta = matched_cand["recall"] - matched_ref["recall"]
            recall_deltas.append(recall_delta)
        ap_delta = cand["ap50"] - ref["ap50"]
        ap_deltas.append(ap_delta)
        dataset, product = key
        points[f"{dataset}:{product}"] = {
            "ap50_delta": ap_delta,
            "reference_recall_at_normal_fp": matched_ref["recall"],
            "candidate_recall_at_normal_fp": matched_cand["recall"],
            "recall_delta_at_reference_normal_fp": recall_delta,
            "reference_normal_fp_per_image": ref["normal_fp_per_image"],
            "candidate_normal_fp_per_image": matched_cand["normal_fp_per_image"],
        }
    return {
        "products": points,
        "product_count": len(points),
        "ap50_positive_products": sum(value > 0 for value in ap_deltas),
        "ap50_product_min": min(ap_deltas) if ap_deltas else None,
        "ap50_product_max": max(ap_deltas) if ap_deltas else None,
        "matched_recall_positive_products": sum(value > 0 for value in recall_deltas),
        "matched_recall_product_min": min(recall_deltas) if recall_deltas else None,
        "matched_recall_product_max": max(recall_deltas) if recall_deltas else None,
    }


def _product_macro(products: list[dict]) -> dict:
    metrics = {}
    for key in ("ap50", "ap", "precision", "recall", "normal_fp_per_image"):
        values = [product[key] for product in products if product[key] is not None]
        metrics[key] = mean(values) if values else None
    return {"products": len(products), **metrics}


def _read_images(records: list[dict]) -> list[torch.Tensor]:
    return [image_to_tensor(record["image_path"]) for record in records]


def _image_batches(records: list[dict], batch_size: int, io_pool):
    batches = (records[start : start + batch_size] for start in range(0, len(records), batch_size))
    if io_pool is None:
        for batch in batches:
            yield batch, _read_images(batch)
        return

    batch = next(batches, None)
    if batch is None:
        return
    pending = io_pool.submit(_read_images, batch)
    for following in batches:
        images = pending.result()
        pending = io_pool.submit(_read_images, following)
        yield batch, images
        batch = following
    yield batch, pending.result()


def _predict_batch(model, images: list[torch.Tensor], device, defect_id: int) -> list[tuple]:
    predictions = []
    for output in model([image.to(device) for image in images]):
        defect = output["labels"] == defect_id
        boxes = output["boxes"][defect].cpu().numpy()
        scores = output["scores"][defect].cpu().numpy()
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise FloatingPointError("检测器返回非有限预测")
        rank = np.argsort(-scores, kind="stable")
        predictions.append((boxes[rank], scores[rank]))
    return predictions


def _build_episode_queries(episode: dict, adapter) -> list[dict]:
    return adapter.build_episode_records(episode, roles=("query_normal", "query_defect"))


def prepare_episode_queries(episodes: list[dict], adapter) -> list[list[dict]]:
    """一次性构造固定 Query 的标注记录，无需保存图像张量。"""
    return [_build_episode_queries(episode, adapter) for episode in episodes]


def evaluate_detector(
    model, episodes: list[dict], adapter, device, config: dict, *,
    prepared_queries: list[list[dict]] | None = None,
) -> dict:
    """对 Query 直接推理，每个 Episode 保留独立的真值与 Ignore 语境。"""
    thresholds = config["iou_thresholds"]
    batch_size = config.get("batch_size", 1)
    ious = [*thresholds, config["working_iou"], config["ignore_iou"]]
    if not episodes or batch_size < 1 or 0.5 not in thresholds:
        raise ValueError("评价需要非空 Episode、正批量大小和 AP50 所需的 IoU=0.5")
    if len(set(thresholds)) != len(thresholds) or any(not 0 < iou <= 1 for iou in ious):
        raise ValueError("评价 IoU 阈值不合法或重复")
    if not 0 <= config["score_threshold"] <= 1:
        raise ValueError("评价置信度阈值必须在 [0, 1] 内")
    if prepared_queries is not None and len(prepared_queries) != len(episodes):
        raise ValueError("缓存的 Query 记录数与 Episode 数不一致")

    product_queries: dict[tuple[str, str], list[_QueryMatches]] = defaultdict(list)
    predictions: dict[str, tuple] = {}
    defect_id = int(adapter.config["classes"]["defect"])
    was_training = model.training
    model.eval()
    io_context = ThreadPoolExecutor(max_workers=1) if config.get("prefetch_images", False) else nullcontext()
    try:
        with torch.inference_mode(), io_context as io_pool:
            for index, episode in enumerate(episodes):
                records = (
                    _build_episode_queries(episode, adapter)
                    if prepared_queries is None else prepared_queries[index]
                )
                image_ids = {record["image_id"] for record in records}
                if not records or len(image_ids) != len(records):
                    raise AssertionError(f"Episode Query 为空或含重复图像：{episode['episode_id']}")
                missing = [record for record in records if record["image_id"] not in predictions]
                for batch, images in _image_batches(missing, batch_size, io_pool):
                    batch_predictions = _predict_batch(model, images, device, defect_id)
                    if len(batch_predictions) != len(batch):
                        raise AssertionError("检测器预测数与图像数不一致")
                    for record, prediction in zip(batch, batch_predictions):
                        predictions[record["image_id"]] = prediction
                for record in records:
                    boxes, scores = predictions[record["image_id"]]
                    pure_normal = record["split_role"] == "QUERY_NORMAL"
                    if pure_normal and record["normal_flag"] != 1:
                        raise AssertionError(f"正常 Query 混入缺陷图：{record['image_id']}")
                    key = (record["dataset"], record["product_id"])
                    ignore_iou = _box_iou(boxes, record["ignore_annotations"])
                    product_queries[key].append(
                        _QueryMatches(
                            scores=scores,
                            positive_iou=_box_iou(boxes, record["annotations"]),
                            ignore_overlap=(ignore_iou >= config["ignore_iou"]).any(axis=1),
                            pure_normal=pure_normal,
                        )
                    )
    finally:
        if was_training:
            model.train()

    per_product = {}
    dataset_products: dict[str, list[dict]] = defaultdict(list)
    for (dataset, product), queries in sorted(product_queries.items()):
        metrics = _score_product(queries, config)
        per_product[f"{dataset}:{product}"] = metrics
        dataset_products[dataset].append(metrics)
    return {
        "product_macro": _product_macro(list(per_product.values())),
        "per_product": per_product,
        "per_dataset": {
            dataset: _product_macro(products) for dataset, products in dataset_products.items()
        },
        "counts": {
            "episodes": len(episodes),
            "unique_query_images": len(predictions),
            "query_image_occurrences": sum(len(rows) for rows in product_queries.values()),
        },
    }
