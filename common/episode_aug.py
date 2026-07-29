"""Episode-level RGB augmentation for MAML meta-train (support + query, same policy)."""

import math
import random

import torch
import torch.nn.functional as F

# Match AviationDefectDataset RGB normalize
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_NUM_POLICIES = 5


def _mean_std(device, dtype):
    mean = torch.tensor(_IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def _denormalize(rgb):
    mean, std = _mean_std(rgb.device, rgb.dtype)
    return (rgb * std + mean).clamp(0.0, 1.0)


def _normalize(rgb):
    mean, std = _mean_std(rgb.device, rgb.dtype)
    return (rgb - mean) / std


def _policy_hflip(rgb):
    return torch.flip(rgb, dims=[3])


def _policy_hflip_rotate(rgb):
    out = torch.flip(rgb, dims=[3])
    angle = random.uniform(-15.0, 15.0) * math.pi / 180.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
        device=rgb.device, dtype=rgb.dtype).unsqueeze(0)
    theta = theta.expand(rgb.size(0), -1, -1)
    grid = F.affine_grid(theta, rgb.size(), align_corners=False)
    return F.grid_sample(rgb, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def _policy_color_jitter(rgb):
    x = _denormalize(rgb)
    brightness = 1.0 + random.uniform(-0.25, 0.25)
    contrast = 1.0 + random.uniform(-0.25, 0.25)
    saturation = 1.0 + random.uniform(-0.15, 0.15)
    x = (x * brightness).clamp(0.0, 1.0)
    mu = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - mu) * contrast + mu).clamp(0.0, 1.0)
    gray = x.mean(dim=1, keepdim=True).expand_as(x)
    x = (x * saturation + gray * (1.0 - saturation)).clamp(0.0, 1.0)
    return _normalize(x)


def _policy_random_erasing(rgb):
    x = _denormalize(rgb)
    n, c, h, w = x.shape
    for i in range(n):
        if random.random() > 0.5:
            continue
        eh = max(1, int(h * random.uniform(0.02, 0.12)))
        ew = max(1, int(w * random.uniform(0.02, 0.12)))
        y0 = random.randint(0, max(0, h - eh))
        x0 = random.randint(0, max(0, w - ew))
        x[i, :, y0:y0 + eh, x0:x0 + ew] = random.uniform(0.0, 1.0)
    return _normalize(x)


def _policy_illumination_aug(rgb):
    """
    Simulate industrial illumination variation: gamma correction, brightness
    offset, and local specular highlight.

    Operates on denormalized [0,1] RGB, independently per image in the batch
    (unlike the failed color_jitter which was effectively a batch-uniform transform).
    Applied to both support and query independently to avoid trivially reducing
    task difficulty.
    """
    x = _denormalize(rgb)
    n, c, h, w = x.shape

    # --- gamma correction: 0.7 (darker) ~ 1.3 (brighter) ---
    gamma = random.uniform(0.7, 1.3)
    x = x ** gamma

    # --- brightness offset: ±20% ---
    brightness = 1.0 + random.uniform(-0.2, 0.2)
    x = x * brightness

    # --- per-channel gain for slight color temperature shift ---
    gain = torch.empty(n, c, 1, 1, device=x.device, dtype=x.dtype).uniform_(0.92, 1.08)
    x = x * gain

    # --- local specular highlight (5% chance per image) ---
    if random.random() < 0.05:
        y_c = random.randint(h // 4, 3 * h // 4)
        x_c = random.randint(w // 4, 3 * w // 4)
        rh, rw = h // 8, w // 8
        yy = torch.arange(h, device=x.device, dtype=torch.float32).view(1, 1, h, 1)
        xx = torch.arange(w, device=x.device, dtype=torch.float32).view(1, 1, 1, w)
        dist = ((yy - y_c) ** 2 / (rh ** 2) + (xx - x_c) ** 2 / (rw ** 2)).clamp(0, 1)
        spot = (1 - dist).clamp(0, 1) * random.uniform(0.15, 0.35)
        x = x + spot.unsqueeze(1)  # broadcast across channels

    return _normalize(x.clamp(0.0, 1.0))


_POLICIES = (
    _policy_hflip,
    _policy_hflip_rotate,
    _policy_color_jitter,
    _policy_random_erasing,
)


def augment_rgb(rgb, policy_id=None):
    """Apply one episode policy to a normalized RGB batch (B,3,H,W)."""
    if policy_id is None:
        policy_id = random.randint(0, _NUM_POLICIES - 1)
    policy_id = int(policy_id) % _NUM_POLICIES
    return _POLICIES[policy_id](rgb), policy_id


def augment_support_query(support, query, policy_id=None):
    """
    Apply the same augmentation to support and query RGB tensors.

    support/query: (rgb, thermal, us, labels)
    """
    s_rgb, s_th, s_us, s_y = support
    q_rgb, q_th, q_us, q_y = query
    aug_rgb = torch.cat([s_rgb, q_rgb], dim=0)
    aug_rgb, policy_id = augment_rgb(aug_rgb, policy_id=policy_id)
    s_rgb = aug_rgb[: s_rgb.size(0)]
    q_rgb = aug_rgb[s_rgb.size(0):]
    return (s_rgb, s_th, s_us, s_y), (q_rgb, q_th, q_us, q_y), policy_id
