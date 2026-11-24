from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops import box_iou, generalized_box_iou_loss

from data_proc.detection import image_to_tensor, target_for_model


MODULE_PREFIXES = {
    # RoI 的共享 fc6/fc7 提取分类与回归共同使用的特征，仅划入 Feature。
    "Feature": ("backbone.", "roi_heads.box_head."),
    "Where": ("rpn.",),
    "Box": ("roi_heads.box_predictor.bbox_pred.",),
    "Class": ("roi_heads.box_predictor.cls_score.",),
}

SKETCH_DIM = 4096
SKETCH_SEED = 20261124
_SKETCH_PRIME = 2147483647
_SKETCH_COEFFS: tuple[int, int, int, int] | None = None


def _sketch_coefficients() -> tuple[int, int, int, int]:
    global _SKETCH_COEFFS
    if _SKETCH_COEFFS is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SKETCH_SEED)
        draws = torch.randint(1, _SKETCH_PRIME, (4,), generator=generator, dtype=torch.int64)
        _SKETCH_COEFFS = tuple(int(value) for value in draws)
    return _SKETCH_COEFFS


def module_count_sketch(named: dict[str, torch.Tensor], dim: int = SKETCH_DIM) -> torch.Tensor:
    """把一个模块的参数按名称顺序压成草图。默认 4096 维。"""
    if int(dim) < 1:
        raise ValueError("草图维度必须为正")
    dim = int(dim)
    present = {name: value for name, value in named.items() if value is not None}
    if not present:
        return torch.zeros(dim, dtype=torch.float16)
    device = next(iter(present.values())).device
    sketch = torch.zeros(dim, dtype=torch.float32, device=device)
    prime = _SKETCH_PRIME
    slope, bias, sign_slope, sign_bias = _sketch_coefficients()
    offset = 0
    for name in sorted(present):
        flat = present[name].detach().reshape(-1)
        count = int(flat.numel())
        if count == 0:
            continue
        if flat.device != device:
            flat = flat.to(device)
        index = torch.arange(offset, offset + count, device=device, dtype=torch.int64)
        folded = torch.remainder(index, prime)
        buckets = torch.remainder(torch.remainder(slope * folded + bias, prime), dim)
        signs = torch.remainder(sign_slope * folded + sign_bias, prime).bitwise_and(1)
        signs = signs.to(dtype=torch.float32).mul_(-2).add_(1)
        sketch.scatter_add_(0, buckets, signs * flat.float())
        offset += count
    if not torch.isfinite(sketch).all():
        raise FloatingPointError("模块草图出现非有限值")
    return sketch.detach().half().cpu()


def cosine_neg_grad(grad_sketch: torch.Tensor, delta_sketch: torch.Tensor) -> float:
    """cos(-梯度草图, 参数差草图)。越大越像沿当前损失往下走。"""
    left = grad_sketch.detach().float().reshape(-1)
    right = delta_sketch.detach().float().reshape(-1)
    if left.numel() != right.numel() or left.numel() == 0:
        raise ValueError("草图维度不一致")
    denom = left.norm() * right.norm()
    if float(denom) == 0.0:
        return -1.0
    value = float(torch.dot(-left, right) / denom)
    if not isfinite(value):
        raise FloatingPointError("草图余弦不是有限值")
    return value


@dataclass(frozen=True)
class ROIExample:
    """One image's sampled RoIs and all candidates needed by the detector loss."""

    features: torch.Tensor
    labels: torch.Tensor
    regression_targets: torch.Tensor
    sampled_boxes: torch.Tensor
    proposal_features: torch.Tensor
    proposal_boxes: torch.Tensor
    target_boxes: torch.Tensor
    ignore_boxes: torch.Tensor
    normal_flag: bool


def validate_detection_loss_settings(settings: dict) -> dict:
    required_weights = ("roi", "positive", "ranking", "hard_negative", "localization")
    weights = settings.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(required_weights):
        raise ValueError("Detection loss weights must define all five loss terms")
    if any(not isfinite(float(weights[name])) or float(weights[name]) < 0 for name in required_weights):
        raise ValueError("Detection loss weights must be finite and non-negative")
    if float(weights["roi"]) <= 0:
        raise ValueError("The RoI loss weight must be positive")
    for key in ("positive_iou", "negative_iou"):
        if not 0 < float(settings[key]) <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if float(settings["negative_iou"]) < float(settings["positive_iou"]):
        raise ValueError("negative_iou must be at least positive_iou")
    if not isfinite(float(settings["working_logit"])):
        raise ValueError("working_logit must be finite")
    if not isfinite(float(settings["ranking_margin"])) or float(settings["ranking_margin"]) <= 0:
        raise ValueError("ranking_margin must be positive and finite")
    if int(settings["hard_negative_top_k"]) < 1:
        raise ValueError("hard_negative_top_k must be positive")
    score = settings.get("defect_score", "class_logit")
    if score not in ("margin", "class_logit"):
        raise ValueError("defect_score must be margin or class_logit")
    return settings


def _defect_scores(logits: torch.Tensor, settings: dict) -> torch.Tensor:
    """Match the binary softmax probability, which depends on the logit margin."""
    mode = settings.get("defect_score", "class_logit")
    if mode == "margin":
        return logits[:, 1] - logits[:, 0]
    if mode == "class_logit":
        return logits[:, 1]
    raise ValueError(f"defect_score must be margin or class_logit: {mode}")


def _eligible_proposal_indices(proposals, positive_boxes, ignore_boxes, fg_iou, ignore_iou):
    count = proposals.shape[0]
    if count == 0 or not ignore_boxes.numel():
        return torch.arange(count, device=proposals.device)
    excluded = (box_iou(proposals, ignore_boxes) >= ignore_iou).any(dim=1)
    foreground = (box_iou(proposals, positive_boxes) >= fg_iou).any(dim=1)
    return torch.where(~excluded | foreground)[0]


def _remove_ignored_proposals(proposals, positive_boxes, ignore_boxes, fg_iou, ignore_iou):
    return proposals[_eligible_proposal_indices(
        proposals, positive_boxes, ignore_boxes, fg_iou, ignore_iou,
    )]


def _record_boxes(record: dict, key: str, original_size, valid_size, device) -> torch.Tensor:
    boxes = device.new_tensor([ann["bbox"] for ann in record.get(key, [])]).reshape(-1, 4)
    return resize_boxes(boxes, original_size, valid_size)


def _build_roi_example(
    roi_heads, pyramid, sampled, labels, regression_targets,
    eligible, target_boxes, ignore_boxes, images, normal_flag,
):
    sampled_boxes = sampled[0]
    sampled_features = roi_heads.box_head(
        roi_heads.box_roi_pool(pyramid, [sampled_boxes], images.image_sizes)
    )
    proposal_features = roi_heads.box_head(
        roi_heads.box_roi_pool(pyramid, [eligible], images.image_sizes)
    )
    regression = torch.where(labels[0][:, None] > 0, regression_targets[0], 0.0)
    return ROIExample(
        sampled_features.cpu(),
        labels[0].cpu(),
        regression.cpu(),
        sampled_boxes.cpu(),
        proposal_features.cpu(),
        eligible.cpu(),
        target_boxes.cpu(),
        ignore_boxes.cpu(),
        bool(normal_flag),
    )


@torch.no_grad()
def prepare_roi_example(detector, record: dict, device, ignore_iou: float) -> ROIExample | None:
    """Build one image example from frozen RPN candidates and RoI features."""
    if detector.training or detector.rpn.training:
        raise ValueError("RoI sample preparation requires an eval detector and RPN")
    if not 0 < ignore_iou <= 1:
        raise ValueError("Ignore IoU must be in (0, 1]")

    cached = record.get("image_tensor")
    image = cached.to(device) if cached is not None else image_to_tensor(record["image_path"]).to(device)
    original_size = tuple(image.shape[-2:])
    target = {key: value.to(device) for key, value in target_for_model(record).items()}
    images, targets = detector.transform([image], [target])
    pyramid = detector.backbone(images.tensors)
    proposals, _ = detector.rpn(images, pyramid)
    proposal = proposals[0]
    roi_heads = detector.roi_heads
    target_boxes = targets[0]["boxes"]
    ignore_boxes = _record_boxes(
        record, "ignore_annotations", original_size, images.image_sizes[0], proposal
    )
    eligible = _remove_ignored_proposals(
        proposal,
        target_boxes,
        ignore_boxes,
        roi_heads.proposal_matcher.high_threshold,
        ignore_iou,
    )
    sampled, _, labels, regression_targets = roi_heads.select_training_samples([eligible], targets)
    if not labels[0].numel() or not eligible.numel():
        return None
    return _build_roi_example(
        roi_heads,
        pyramid,
        sampled,
        labels,
        regression_targets,
        eligible,
        target_boxes,
        ignore_boxes,
        images,
        record["normal_flag"],
    )


def _loss_from_predictions(
    sampled_logits, sampled_regression, logits, box_regression,
    labels, regression_targets, proposal_boxes, target_boxes,
    box_coder, settings, device,
):
    """检测对齐损失的计算本体。预测由调用方提供，打分和训练共用这一套公式。"""
    roi_classifier, roi_regressor = fastrcnn_loss(
        sampled_logits,
        sampled_regression,
        [labels.to(device)],
        [regression_targets.to(device)],
    )
    roi_loss = roi_classifier + roi_regressor

    scores = _defect_scores(logits, settings)
    proposal_boxes = proposal_boxes.to(device)
    target_boxes = target_boxes.to(device)
    if target_boxes.numel():
        overlaps = box_iou(proposal_boxes, target_boxes)
        max_iou = overlaps.max(dim=1).values
    else:
        overlaps = scores.new_zeros((proposal_boxes.shape[0], 0))
        max_iou = scores.new_zeros((proposal_boxes.shape[0],))

    positive_threshold = float(settings["positive_iou"])
    negative_threshold = float(settings["negative_iou"])
    negative_candidates = max_iou < negative_threshold

    positive_scores = []
    positive_pairs = []
    if target_boxes.numel():
        for target_index in range(target_boxes.shape[0]):
            candidates = overlaps[:, target_index] >= positive_threshold
            if candidates.any():
                candidate_indices = torch.where(candidates)[0]
                best = candidate_indices[scores[candidate_indices].argmax()]
                positive_pairs.append((best, target_index))
                positive_scores.append(scores[best])
    positive_scores = torch.stack(positive_scores) if positive_scores else scores.new_empty(0)
    if positive_scores.numel():
        positive_loss = torch.nn.functional.softplus(
            float(settings["working_logit"]) - positive_scores
        ).mean()
    else:
        positive_loss = scores.new_zeros(())

    negative_scores = scores[negative_candidates]
    top_k = min(int(settings["hard_negative_top_k"]), negative_scores.numel())
    if top_k:
        hard_negative = negative_scores.topk(top_k).values
        negative_loss = torch.nn.functional.softplus(
            hard_negative - float(settings["working_logit"])
        ).mean()
    else:
        hard_negative = scores.new_empty(0)
        negative_loss = scores.new_zeros(())

    if positive_scores.numel() and hard_negative.numel():
        ranking_loss = torch.nn.functional.softplus(
            float(settings["ranking_margin"])
            - positive_scores[:, None]
            + hard_negative[None, :]
        ).mean()
    else:
        ranking_loss = scores.new_zeros(())

    if positive_pairs:
        indices = torch.tensor([proposal for proposal, _ in positive_pairs], device=device)
        target_indices = torch.tensor([target for _, target in positive_pairs], device=device)
        selected_regression = box_regression[indices, 4:8]
        predicted_boxes = box_coder.decode(selected_regression, [proposal_boxes[indices]]).reshape(-1, 4)
        matched_boxes = target_boxes[target_indices]
        localization_loss = generalized_box_iou_loss(
            predicted_boxes,
            matched_boxes,
            reduction="mean",
        )
    else:
        localization_loss = scores.new_zeros(())

    return {
        "roi": roi_loss,
        "positive": positive_loss,
        "ranking": ranking_loss,
        "hard_negative": negative_loss,
        "localization": localization_loss,
    }


def _loss_components(predictor, example: ROIExample, box_coder, settings, device):
    sampled_logits, sampled_regression = predictor(example.features.to(device))
    logits, box_regression = predictor(example.proposal_features.to(device))
    return _loss_from_predictions(
        sampled_logits,
        sampled_regression,
        logits,
        box_regression,
        example.labels,
        example.regression_targets,
        example.proposal_boxes,
        example.target_boxes,
        box_coder,
        settings,
        device,
    )


def _weighted_detection_loss(components, settings) -> torch.Tensor:
    weights = settings["weights"]
    total = sum(
        components[name] * float(weights[name])
        for name in ("roi", "positive", "ranking", "hard_negative", "localization")
    )
    if not torch.isfinite(total):
        raise FloatingPointError("Detection-aligned RoI loss is not finite")
    return total


def roi_detection_loss(predictor, example: ROIExample, box_coder, settings, device) -> torch.Tensor:
    """Compute a detector-aligned loss for one image and its RPN candidates."""
    return _weighted_detection_loss(
        _loss_components(predictor, example, box_coder, settings, device), settings,
    )


def reuse_weight(base: float, candidate: float, cap: float, *, higher_is_better: bool) -> float:
    """把支持集收益变成小于 1 的复用权重。没有正收益时权重为 0。"""
    cap = float(cap)
    if not 0.0 < cap < 1.0 or not isfinite(cap):
        raise ValueError("复用权重上限必须在 (0, 1) 内")
    if not isfinite(float(base)) or not isfinite(float(candidate)):
        return 0.0
    if higher_is_better:
        benefit = float(candidate) - float(base)
        scale = benefit
    else:
        if float(base) <= 0:
            return 0.0
        benefit = float(base) - float(candidate)
        scale = benefit / float(base)
    if benefit <= 0:
        return 0.0
    return float(min(cap, scale))


def _transformed_view(detector, record: dict, device):
    cached = record.get("image_tensor")
    image = cached.to(device) if cached is not None else image_to_tensor(record["image_path"]).to(device)
    original_size = tuple(image.shape[-2:])
    target = {key: value.to(device) for key, value in target_for_model(record).items()}
    images, targets = detector.transform([image], [target])
    return original_size, images, targets


def _pyramid(detector, tensors):
    levels = detector.backbone(tensors)
    return levels


def _live_roi_loss(detector, pyramid, proposals, targets, record, original_size, images, loss_settings, ignore_iou, device):
    proposal = proposals[0]
    roi_heads = detector.roi_heads
    target_boxes = targets[0]["boxes"]
    ignore_boxes = _record_boxes(
        record, "ignore_annotations", original_size, images.image_sizes[0], proposal
    )
    eligible = _remove_ignored_proposals(
        proposal, target_boxes, ignore_boxes, roi_heads.proposal_matcher.high_threshold, ignore_iou,
    )
    sampled, _, labels, regression_targets = roi_heads.select_training_samples([eligible], targets)
    anchor = pyramid[next(iter(pyramid))]
    if not labels[0].numel() or not eligible.numel():
        return anchor.new_zeros(())
    sampled_boxes = sampled[0]
    example = ROIExample(
        roi_heads.box_head(roi_heads.box_roi_pool(pyramid, [sampled_boxes], images.image_sizes)),
        labels[0],
        torch.where(labels[0][:, None] > 0, regression_targets[0], 0.0),
        sampled_boxes,
        roi_heads.box_head(roi_heads.box_roi_pool(pyramid, [eligible], images.image_sizes)),
        eligible,
        target_boxes,
        ignore_boxes,
        bool(record["normal_flag"]),
    )
    return roi_detection_loss(roi_heads.box_predictor, example, roi_heads.box_coder, loss_settings, device)


def live_detection_loss(
    detector, record: dict, device, loss_settings, where_loss_weight: float, ignore_iou: float,
    checkpoint_graph: bool = False,
) -> torch.Tensor:
    """一张图的检测损失。梯度同时到达骨干、RPN、RoI 共享层和分类、框头。"""
    if checkpoint_graph:
        cached = record.get("image_tensor")
        image = cached.detach() if cached is not None else image_to_tensor(record["image_path"]).detach()

        def run(image_tensor):
            local = dict(record)
            local["image_tensor"] = image_tensor
            return live_detection_loss(
                detector, local, device, loss_settings, where_loss_weight, ignore_iou, False,
            )

        from torch.utils.checkpoint import checkpoint
        return checkpoint(run, image.to(device), use_reentrant=False)
    if not detector.training or not detector.rpn.training:
        raise ValueError("模块元训练需要训练模式的检测器和 RPN")
    where_loss_weight = float(where_loss_weight)
    if not isfinite(where_loss_weight) or where_loss_weight < 0:
        raise ValueError("Where 损失权重必须为非负有限值")
    if not 0 < float(ignore_iou) <= 1:
        raise ValueError("Ignore IoU must be in (0, 1]")
    original_size, images, targets = _transformed_view(detector, record, device)
    pyramid = _pyramid(detector, images.tensors)
    proposals, rpn_losses = detector.rpn(images, pyramid, targets)
    where = rpn_losses["loss_objectness"] + rpn_losses["loss_rpn_box_reg"]
    roi = _live_roi_loss(
        detector, pyramid, proposals, targets, record, original_size, images, loss_settings, ignore_iou, device,
    )
    total = roi + where_loss_weight * where
    if not torch.isfinite(total):
        raise FloatingPointError(f"模块检测损失不是有限值：{record.get('image_id')}")
    return total


@torch.no_grad()
def eval_support_view(detector, record: dict, device, ignore_iou: float, *, keep_pyramid: bool = False) -> dict:
    """在当前权重下取一张支持图的固定候选。默认不缓存整塔特征，避免上百张图把内存顶满。"""
    if detector.training or detector.rpn.training:
        raise ValueError("支持集打分需要 eval 检测器")
    if not 0 < float(ignore_iou) <= 1:
        raise ValueError("Ignore IoU must be in (0, 1]")
    original_size, images, targets = _transformed_view(detector, record, device)
    pyramid = detector.backbone(images.tensors)
    proposals, _ = detector.rpn(images, pyramid)
    view = {
        "image_id": record["image_id"],
        "record": record,
        "original_size": original_size,
        "tensors": images.tensors.detach().cpu(),
        "image_sizes": list(images.image_sizes),
        "targets": {key: value.detach().cpu() for key, value in targets[0].items()},
        "proposals": proposals[0].detach().cpu(),
    }
    if keep_pyramid:
        view["pyramid"] = {key: value.detach().cpu() for key, value in pyramid.items()}
    return view


def fixed_proposal_detection_loss(detector, view: dict, device, loss_settings, ignore_iou: float) -> torch.Tensor:
    """只替换特征，候选框保持打分开始时的那一批。"""
    from torchvision.models.detection.image_list import ImageList

    images = ImageList(view["tensors"].to(device), view["image_sizes"])
    targets = [{key: value.to(device) for key, value in view["targets"].items()}]
    pyramid = detector.backbone(images.tensors)
    return _live_roi_loss(
        detector, pyramid, [view["proposals"].to(device)], targets, view["record"],
        view["original_size"], images, loss_settings, ignore_iou, device,
    )


def proposal_recall(detector, view: dict, device, iou_threshold: float = 0.5) -> float | None:
    """在固定骨干特征上只跑 RPN，读 top-50 对真值框的覆盖。正常图不参与。"""
    from torchvision.models.detection.image_list import ImageList

    gt = view["targets"]["boxes"]
    if gt.numel() == 0:
        return None
    images = ImageList(view["tensors"].to(device), view["image_sizes"])
    if "pyramid" in view:
        pyramid = {key: value.to(device) for key, value in view["pyramid"].items()}
    else:
        pyramid = detector.backbone(images.tensors)
    proposals, _ = detector.rpn(images, pyramid)
    top = proposals[0][:50]
    if top.numel() == 0:
        return 0.0
    covered = box_iou(top, gt.to(device)).max(dim=0).values >= iou_threshold
    return float(covered.float().mean())


def _mean_optional(values):
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _predict_heads(predictor, features, *, cls: bool, bbox: bool):
    """与 FastRCNNPredictor.forward 使用同一套线性层。只算本次权重真正变化的那一头。"""
    if cls and bbox:
        return predictor(features)
    flat = features.flatten(start_dim=1) if features.dim() == 4 else features
    logits = predictor.cls_score(flat) if cls else None
    deltas = predictor.bbox_pred(flat) if bbox else None
    return logits, deltas


class SupportScoreCache:
    """无梯度模块评分的分层缓存。

    只在一次初始化选择内有效。Feature 令牌变化后丢弃骨干特征、候选和 RoI 特征；
    Where 令牌变化后丢弃候选和依赖候选的 RoI 特征，骨干特征可以继续用。
    Box 只重算框回归，Class 只重算分类。采样仍然每次调用正负样本采样器。
    骨干特征留在 CPU，打分时按张图搬回显存，避免 4GB 卡上同时放下整组特征塔。
    """

    def __init__(self, detector, device, settings):
        ignore_iou = float(settings["ignore_iou"])
        if not 0 < ignore_iou <= 1:
            raise ValueError("Ignore IoU must be in (0, 1]")
        self.detector = detector
        self.device = device
        self.loss_settings = settings["loss"]
        self.ignore_iou = ignore_iou
        self.feature_token = ("shared",)
        self.where_token = ("shared",)
        self.box_token = ("shared",)
        self.class_token = ("shared",)
        self._prepared = {}
        self._pyramids = {}
        self._proposals = {}
        self._roi = {}
        self._logits = {}
        self._deltas = {}
        self._pyramid_token = None
        self._proposal_token = None
        self._roi_token = None
        self._logits_token = None
        self._deltas_token = None

    def close(self) -> None:
        self._prepared.clear()
        self._pyramids.clear()
        self._proposals.clear()
        self._roi.clear()
        self._logits.clear()
        self._deltas.clear()
        self._pyramid_token = None
        self._proposal_token = None
        self._roi_token = None
        self._logits_token = None
        self._deltas_token = None

    def token(self, module_name: str):
        return {
            "Feature": self.feature_token,
            "Where": self.where_token,
            "Box": self.box_token,
            "Class": self.class_token,
        }[module_name]

    def mark(self, module_name: str, token) -> None:
        if module_name == "Feature":
            self.feature_token = token
        elif module_name == "Where":
            self.where_token = token
        elif module_name == "Box":
            self.box_token = token
        elif module_name == "Class":
            self.class_token = token
        else:
            raise ValueError(f"未知模块：{module_name}")

    def score(self, module_name: str, records) -> float | None:
        if module_name == "Feature":
            return self.score_feature(records)
        if module_name == "Where":
            return self.score_where(records)
        if module_name in ("Box", "Class"):
            return self.score_head(records)
        raise ValueError(f"未知模块：{module_name}")

    def score_feature(self, records) -> float | None:
        scores = []
        for record in records:
            prepared, pyramid, proposals = self._proposals_for(record, keep_pyramid=True)
            loss = _live_roi_loss(
                self.detector, pyramid, [proposals], prepared["targets"], record,
                prepared["original_size"], prepared["images"],
                self.loss_settings, self.ignore_iou, self.device,
            )
            scores.append(float(loss))
        return _mean_optional(scores)

    def score_where(self, records) -> float | None:
        scores = []
        for record in records:
            prepared, _, proposals = self._proposals_for(record, keep_pyramid=False)
            gt = prepared["targets"][0]["boxes"]
            if gt.numel() == 0:
                scores.append(None)
                continue
            top = proposals[:50]
            if top.numel() == 0:
                scores.append(0.0)
                continue
            covered = box_iou(top, gt).max(dim=0).values >= 0.5
            scores.append(float(covered.float().mean()))
        return _mean_optional(scores)

    def score_head(self, records) -> float | None:
        losses = []
        roi_heads = self.detector.roi_heads
        predictor = roi_heads.box_predictor
        for record in records:
            prepared, _, proposals = self._proposals_for(record, keep_pyramid=False)
            upstream = self._eligible_for(record, prepared, proposals)
            sampled, _, labels, regression_targets = roi_heads.select_training_samples(
                [upstream["eligible"]], prepared["targets"],
            )
            if not labels[0].numel() or not upstream["eligible"].numel():
                continue
            key = self._key(record)
            proposal_features = self._proposal_features(key, upstream, prepared)
            sampled_features = self._sampled_features(key, upstream, sampled[0], prepared)
            regression = torch.where(labels[0][:, None] > 0, regression_targets[0], 0.0)
            sampled_logits, sampled_deltas = predictor(sampled_features)
            proposal_logits, proposal_deltas = self._proposal_predictions(key, proposal_features)
            components = _loss_from_predictions(
                sampled_logits, sampled_deltas, proposal_logits, proposal_deltas,
                labels[0], regression, upstream["eligible"], upstream["target_boxes"],
                roi_heads.box_coder, self.loss_settings, self.device,
            )
            losses.append(float(_weighted_detection_loss(components, self.loss_settings)))
        return _mean_optional(losses)

    def _guard(self) -> None:
        if torch.is_grad_enabled():
            raise RuntimeError("模块评分缓存只能在无梯度打分中使用")
        if self.detector.training or self.detector.rpn.training:
            raise RuntimeError("模块评分缓存需要 eval 检测器")

    def _key(self, record):
        return (record.get("image_id"), id(record))

    def _input(self, record):
        key = self._key(record)
        found = self._prepared.get(key)
        if found is None:
            original_size, images, targets = _transformed_view(self.detector, record, self.device)
            found = {"original_size": original_size, "images": images, "targets": targets}
            self._prepared[key] = found
        return found

    def _drop_from_pyramid(self) -> None:
        self._pyramids.clear()
        self._proposals.clear()
        self._roi.clear()
        self._logits.clear()
        self._deltas.clear()
        self._pyramid_token = None
        self._proposal_token = None
        self._roi_token = None
        self._logits_token = None
        self._deltas_token = None

    def _drop_from_proposals(self) -> None:
        self._proposals.clear()
        self._roi.clear()
        self._logits.clear()
        self._deltas.clear()
        self._proposal_token = None
        self._roi_token = None
        self._logits_token = None
        self._deltas_token = None

    def _store_pyramid(self, pyramid):
        stored = {}
        cuda = False
        for level, tensor in pyramid.items():
            if tensor.is_cuda:
                mirror = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
                mirror.copy_(tensor.detach())
                stored[level] = mirror
                cuda = True
            else:
                stored[level] = tensor
        if cuda:
            torch.cuda.synchronize(self.device)
        return stored

    def _on_device(self, pyramid):
        return {level: tensor.to(self.device) for level, tensor in pyramid.items()}

    def _proposals_for(self, record, *, keep_pyramid: bool):
        self._guard()
        if self.feature_token != self._pyramid_token:
            self._drop_from_pyramid()
            self._pyramid_token = self.feature_token
        prepared = self._input(record)
        key = self._key(record)
        pyramid = None
        if key not in self._pyramids:
            pyramid = self.detector.backbone(prepared["images"].tensors)
            self._pyramids[key] = self._store_pyramid(pyramid)
        proposal_token = (self.feature_token, self.where_token)
        if proposal_token != self._proposal_token:
            self._drop_from_proposals()
            self._proposal_token = proposal_token
        proposals = self._proposals.get(key)
        if proposals is None:
            if pyramid is None:
                pyramid = self._on_device(self._pyramids[key])
            proposals, _ = self.detector.rpn(prepared["images"], pyramid)
            proposals = proposals[0]
            self._proposals[key] = proposals
        elif keep_pyramid and pyramid is None:
            pyramid = self._on_device(self._pyramids[key])
        if not keep_pyramid:
            pyramid = None
        return prepared, pyramid, proposals

    def _roi_features(self, pyramid, boxes, prepared):
        roi_heads = self.detector.roi_heads
        return roi_heads.box_head(
            roi_heads.box_roi_pool(pyramid, [boxes], prepared["images"].image_sizes)
        )

    def _rows_for_boxes(self, boxes, pool_boxes, pool_features):
        if pool_features is None or boxes.numel() == 0 or pool_boxes.numel() == 0:
            return None
        same = (boxes[:, None, :] == pool_boxes[None, :, :]).all(dim=-1)
        if not bool(same.any(dim=1).all()):
            return None
        return pool_features.index_select(0, same.to(dtype=torch.int64).argmax(dim=1))

    def _sampled_features(self, key, upstream, sampled_boxes, prepared):
        gathered = self._rows_for_boxes(
            sampled_boxes, upstream["eligible"], upstream["proposal_features"],
        )
        if gathered is not None:
            return gathered
        target_boxes = upstream["target_boxes"]
        pyramid = None
        if target_boxes.numel() and upstream.get("gt_features") is None:
            pyramid = self._on_device(self._pyramids[key])
            upstream["gt_features"] = self._roi_features(pyramid, target_boxes, prepared)
        if target_boxes.numel() and upstream.get("gt_features") is not None:
            if upstream["eligible"].numel():
                pool_boxes = torch.cat([upstream["eligible"], target_boxes], dim=0)
                pool_features = torch.cat([upstream["proposal_features"], upstream["gt_features"]], dim=0)
            else:
                pool_boxes = target_boxes
                pool_features = upstream["gt_features"]
            gathered = self._rows_for_boxes(sampled_boxes, pool_boxes, pool_features)
            if gathered is not None:
                return gathered
        if pyramid is None:
            pyramid = self._on_device(self._pyramids[key])
        return self._roi_features(pyramid, sampled_boxes, prepared)

    def _sync_roi_token(self) -> None:
        token = (self.feature_token, self.where_token)
        if token == self._roi_token:
            return
        self._roi.clear()
        self._logits.clear()
        self._deltas.clear()
        self._roi_token = token
        self._logits_token = None
        self._deltas_token = None

    def _eligible_for(self, record, prepared, proposals):
        self._sync_roi_token()
        key = self._key(record)
        found = self._roi.get(key)
        if found is not None:
            return found
        target_boxes = prepared["targets"][0]["boxes"]
        ignore_boxes = _record_boxes(
            record, "ignore_annotations", prepared["original_size"],
            prepared["images"].image_sizes[0], proposals,
        )
        eligible = _remove_ignored_proposals(
            proposals, target_boxes, ignore_boxes,
            self.detector.roi_heads.proposal_matcher.high_threshold, self.ignore_iou,
        )
        found = {
            "eligible": eligible, "target_boxes": target_boxes,
            "proposal_features": None, "gt_features": None,
        }
        self._roi[key] = found
        return found

    def _proposal_features(self, key, upstream, prepared):
        if upstream["proposal_features"] is not None:
            return upstream["proposal_features"]
        features = self._roi_features(
            self._on_device(self._pyramids[key]), upstream["eligible"], prepared,
        )
        upstream["proposal_features"] = features
        return features

    def _proposal_predictions(self, key, features):
        logits_token = (self._roi_token, self.class_token)
        deltas_token = (self._roi_token, self.box_token)
        if logits_token != self._logits_token:
            self._logits.clear()
            self._logits_token = logits_token
        if deltas_token != self._deltas_token:
            self._deltas.clear()
            self._deltas_token = deltas_token
        logits = self._logits.get(key)
        deltas = self._deltas.get(key)
        if logits is None or deltas is None:
            new_logits, new_deltas = _predict_heads(
                self.detector.roi_heads.box_predictor, features,
                cls=logits is None, bbox=deltas is None,
            )
            if logits is None:
                logits = new_logits
                self._logits[key] = logits
            if deltas is None:
                deltas = new_deltas
                self._deltas[key] = deltas
        return logits, deltas

