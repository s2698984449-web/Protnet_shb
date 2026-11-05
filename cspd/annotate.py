"""通用位置标注：掩码加载、高斯权重、质心。

掩码根目录和文件名模板由配置给出，不绑定数据集：
  图像:  <split_root>/<rel>
  掩码:  <mask_root>/<rel 的父目录>/<mask_pattern 格式化后的文件名>

mask_pattern 可用 {stem} 和 {suffix}，例如 {stem}_mask.png。
mask_root 为空则不加载掩码。
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def resolve_mask_path(image_path, split_root, mask_root, mask_pattern='{stem}_mask.png'):
    if not mask_root:
        return None
    image_path = Path(image_path)
    split_root = Path(split_root)
    mask_root = Path(mask_root)
    rel = image_path.relative_to(split_root)
    name = mask_pattern.format(stem=image_path.stem, suffix=image_path.suffix)
    return mask_root / rel.parent / name


def rasterize_boxes(width, height, boxes):
    """把像素框画成二值图。boxes 为 (xmin, ymin, xmax, ymax)，坐标是原图像素。"""
    from PIL import ImageDraw
    canvas = Image.new('L', (int(width), int(height)), 0)
    draw = ImageDraw.Draw(canvas)
    for xmin, ymin, xmax, ymax in boxes:
        x0 = max(0, min(int(width) - 1, int(round(xmin))))
        y0 = max(0, min(int(height) - 1, int(round(ymin))))
        x1 = max(0, min(int(width) - 1, int(round(xmax))))
        y1 = max(0, min(int(height) - 1, int(round(ymax))))
        if x1 > x0 and y1 > y0:
            draw.rectangle([x0, y0, x1, y1], fill=255)
    return canvas


def load_mask(path, size):
    """返回 (1, H, W)，值在 [0, 1]。文件不存在则返回 None。"""
    if path is None or not Path(path).is_file():
        return None
    img = Image.open(path).convert('L')
    img = transforms.Resize((int(size), int(size)))(img)
    mask = transforms.ToTensor()(img)
    if float(mask.max()) > 0:
        mask = (mask > 0.5).float()
    return mask


def gaussian_weight_from_mask(mask, sigma_ratio=0.15):
    """mask: (N, 1, H, W) 或 (1, H, W)。在掩码上做高斯平滑，用作原型权重。"""
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    n, _, h, w = mask.shape
    sigma = max(float(sigma_ratio) * max(h, w), 1e-3)
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
    kernel = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
    weight = F.conv2d(mask, kernel, padding=radius) + mask
    flat = weight.flatten(1)
    denom = flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (flat / denom).view(n, 1, h, w)


def mask_centroid_yx(mask):
    """mask: (N, 1, H, W) -> (N, 2) 归一化到 0-1。全零掩码返回 (-1, -1) 表示无效。"""
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    n, _, h, w = mask.shape
    ys = torch.linspace(0, 1, h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1)
    xs = torch.linspace(0, 1, w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w)
    mass = mask.flatten(1).sum(dim=1).clamp_min(0.0)
    valid = mass > 0
    y = (mask * ys).flatten(1).sum(dim=1) / mass.clamp_min(1e-6)
    x = (mask * xs).flatten(1).sum(dim=1) / mass.clamp_min(1e-6)
    out = torch.stack([y, x], dim=-1)
    out = torch.where(valid.unsqueeze(-1), out, out.new_full(out.shape, -1.0))
    return out, valid


def resize_mask(mask, size_hw):
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return F.interpolate(mask, size=size_hw, mode='bilinear', align_corners=False)


def weighted_spatial_mean(tokens, weight):
    """tokens (N, D, H, W), weight (N, 1, H, W) -> (N, D)。"""
    weight = weight / weight.flatten(1).sum(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (tokens * weight).sum(dim=(2, 3))
