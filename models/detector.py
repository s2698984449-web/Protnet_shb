"""B0 二分类 Faster R-CNN + FPN 构造。"""

from __future__ import annotations

import re
from copy import deepcopy
from functools import partial
from math import isfinite
from pathlib import Path
from urllib.parse import urlparse

from common.io import file_sha256, load_config, named_config_path, project_root


def _backbone_weights(model_config: dict):
    from torchvision.models import ResNet50_Weights

    name = model_config.get("backbone_weights")
    if name is None:
        if model_config.get("backbone_checkpoint") or model_config.get("pretrained_weights"):
            raise ValueError("载入预训练骨干时必须显式指定 backbone_weights")
        return None
    if name not in {weights.name for weights in ResNet50_Weights}:
        raise ValueError(f"未知 ResNet-50 骨干权重：{name}")
    return ResNet50_Weights[name]


def backbone_weight_manifest(model_config: dict) -> dict | None:
    """验证本地 ImageNet 权重并记录来源，禁止隐式下载。"""
    import torch

    weights = _backbone_weights(model_config)
    if weights is None:
        return None
    filename = Path(urlparse(weights.url).path).name
    expected = re.search(r"-([0-9a-f]{8,64})\.pth$", filename)
    if expected is None:
        raise ValueError(f"TorchVision 权重地址缺少 SHA-256 前缀：{weights.url}")
    configured = model_config.get("backbone_checkpoint")
    checkpoint = (project_root() / Path(configured).expanduser() if configured else
                  Path(torch.hub.get_dir()) / "checkpoints" / filename)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"缺少 ResNet-50 预训练权重：{checkpoint}，官方地址：{weights.url}")
    sha256 = file_sha256(checkpoint)
    if not sha256.startswith(expected.group(1)):
        raise ValueError(f"ResNet-50 预训练权重 SHA-256 不匹配：{checkpoint}")
    return {"id": str(weights), "sha256": sha256, "url": weights.url,
            "checkpoint": str(checkpoint.resolve())}


def _gaussian_box_parameters(boxes):
    import torch

    left_top, right_bottom = boxes.to(dtype=torch.float32).split(2, dim=1)
    return torch.cat(((left_top + right_bottom) * 0.5, (right_bottom - left_top) * 0.5), dim=1)


def _iou_nwd_similarity(ground_truth, anchors, *, constant: float):
    import torch
    from torchvision.ops import box_iou

    iou = box_iou(ground_truth, anchors)
    distance = torch.cdist(
        _gaussian_box_parameters(ground_truth), _gaussian_box_parameters(anchors),
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    nwd = distance.div_(-constant).exp_()
    return torch.maximum(iou, nwd)


def b0_model_config(model_config: dict) -> dict:
    """将当前 B0 协议应用于各个调用入口，拒绝旧协议或临时覆盖。"""
    protocol = load_config(named_config_path("b0_v3.yaml"))["model"]
    for key, expected in protocol.items():
        if key in model_config and model_config[key] != expected:
            raise ValueError(f"B0 模型设置 {key} 与当前 V3 协议不一致")
    matching = protocol["rpn_matching"]
    if matching["mode"] != "iou_nwd" or protocol["resize_mode"] != "shrink_only":
        raise ValueError("当前 B0 协议必须使用共享预处理与 IoU/NWD 混合匹配")
    constant = matching["nwd_constant"]
    if type(constant) not in (int, float) or not isfinite(constant) or constant <= 0:
        raise ValueError("NWD 的 nwd_constant 必须为正且有限")
    return {**model_config, **protocol}


def build_faster_rcnn(model_config: dict, *, initialize_backbone: bool = True):
    architecture = model_config["architecture"]
    if architecture != "fasterrcnn_resnet50_fpn":
        raise ValueError(f"不支持的检测器：{architecture}")
    from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn

    kwargs = deepcopy(b0_model_config(model_config))
    kwargs.pop("architecture")
    kwargs.pop("pretrained_weights", None)
    kwargs.pop("backbone_weights", None)
    kwargs.pop("backbone_checkpoint", None)
    kwargs.pop("resize_mode")
    matching = kwargs.pop("rpn_matching")
    if _backbone_weights(model_config) is None:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, **kwargs)
        return _configure_detector(model, matching["nwd_constant"])

    import torch
    from torchvision.models import resnet50
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.ops.misc import FrozenBatchNorm2d

    trainable_layers = kwargs.pop("trainable_backbone_layers", None)
    if trainable_layers is None:
        trainable_layers = 3
    if type(trainable_layers) is not int or not 0 <= trainable_layers <= 5:
        raise ValueError("trainable_backbone_layers 必须为 0 到 5 的整数")
    backbone = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    if initialize_backbone:
        manifest = backbone_weight_manifest(model_config)
        state = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
    model = FasterRCNN(_resnet_fpn_extractor(backbone, trainable_layers), **kwargs)
    return _configure_detector(model, matching["nwd_constant"])


def _configure_detector(model, nwd_constant: float):
    from data_proc.detection_geometry import ShrinkOnlyDetectionTransform, image_geometry

    transform = model.transform
    if len(transform.min_size) != 1 or transform.fixed_size is not None:
        raise ValueError("共享预处理需要唯一的短边上限和长边上限")
    short_limit = transform.min_size[0]
    image_geometry((1, 1), short_limit, transform.max_size, transform.size_divisible)
    model.transform = ShrinkOnlyDetectionTransform(
        short_limit, transform.max_size, transform.image_mean, transform.image_std, transform.size_divisible,
    )
    model.rpn.box_similarity = partial(_iou_nwd_similarity, constant=nwd_constant)
    return model


def finite_loss_sum(losses: dict):
    import torch

    if not losses:
        raise AssertionError("检测器没有返回训练损失")
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise FloatingPointError(f"检测器损失非有限值：{losses}")
    return total
