"""共享检测输入尺度、有效区域和检测框坐标映射。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.functional import interpolate
from torchvision.models.detection.transform import (
    GeneralizedRCNNTransform,
    resize_boxes,
    resize_keypoints,
)
from torchvision.ops import clip_boxes_to_image


@dataclass(frozen=True)
class DetectionGeometry:
    """尺寸按 (高, 宽) 表示，padding 保持左上角不动。"""

    original_size: tuple[int, int]
    valid_size: tuple[int, int]
    padded_size: tuple[int, int]

    def to_valid_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        return resize_boxes(boxes, self.original_size, self.valid_size)

    def to_original_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        valid_boxes = clip_boxes_to_image(boxes, self.valid_size)
        return resize_boxes(valid_boxes, self.valid_size, self.original_size)


def image_geometry(
    original_size: tuple[int, int], short_limit: int, long_limit: int, stride: int = 32,
) -> DetectionGeometry:
    """只缩小超限图像，计算单图所需的步长对齐尺寸。"""
    short_side, long_side = sorted(original_size)
    if short_side < 1 or not 0 < short_limit <= long_limit or stride < 1:
        raise ValueError("图像尺寸、尺度上限和补齐步长必须为正，且短边上限不得超过长边上限")
    scale = min(1.0, short_limit / short_side, long_limit / long_side)
    height, width = original_size
    valid_height, valid_width = max(1, round(height * scale)), max(1, round(width * scale))
    padded_height = (valid_height + stride - 1) // stride * stride
    padded_width = (valid_width + stride - 1) // stride * stride
    return DetectionGeometry((height, width), (valid_height, valid_width), (padded_height, padded_width))


class ShrinkOnlyDetectionTransform(GeneralizedRCNNTransform):
    """用共享几何规则缩放，沿用 TorchVision 的归一化和右下批量补齐。"""

    def resize(self, image, target=None):
        geometry = image_geometry(image.shape[-2:], self.min_size[0], self.max_size, self.size_divisible)
        if geometry.valid_size == geometry.original_size:
            return image, target
        image = interpolate(image[None], size=geometry.valid_size, mode="bilinear", align_corners=False)[0]
        if target is None:
            return image, target

        boxes = geometry.to_valid_boxes(target["boxes"])
        target["boxes"] = boxes
        if "area" in target:
            target["area"] = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
        if "masks" in target:
            masks = target["masks"]
            target["masks"] = interpolate(masks[:, None].float(), size=geometry.valid_size)[:, 0].to(masks.dtype)
        if "keypoints" in target:
            target["keypoints"] = resize_keypoints(target["keypoints"], geometry.original_size, geometry.valid_size)
        return image, target

    def postprocess(self, result, image_shapes, original_image_sizes):
        if not self.training:
            for prediction, valid_size in zip(result, image_shapes):
                prediction["boxes"] = clip_boxes_to_image(prediction["boxes"], valid_size)
        return super().postprocess(result, image_shapes, original_image_sizes)
