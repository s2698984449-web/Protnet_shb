"""ROI crop utilities for aviation dataset conversion."""

import json
import os

import numpy as np
from PIL import Image

from .aviation_taxonomy import DEFAULT_CROP_POLICY


def merge_crop_policy(overrides=None):
    policy = dict(DEFAULT_CROP_POLICY)
    if overrides:
        policy.update(overrides)
    return policy


def _clip_bbox(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), img_w))
    y2 = max(y1 + 1, min(int(round(y2)), img_h))
    return x1, y1, x2, y2


def yolo_box_to_pixels(cx, cy, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    x1 = cx * img_w - bw / 2.0
    y1 = cy * img_h - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh
    return _clip_bbox(x1, y1, x2, y2, img_w, img_h)


def mask_to_bbox(mask_img):
    """Return pixel bbox (x1,y1,x2,y2) from grayscale/binary mask."""
    arr = np.array(mask_img.convert('L'))
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    return _clip_bbox(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, arr.shape[1], arr.shape[0])


def expand_bbox(x1, y1, x2, y2, img_w, img_h, expand_ratio):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * expand_ratio
    h = (y2 - y1) * expand_ratio
    return _clip_bbox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, img_w, img_h)


def _bbox_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersect_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return _bbox_area(x1, y1, x2, y2)


def _defect_ratio(defect_box, crop_box):
    defect_area = _bbox_area(*defect_box)
    if defect_area <= 0:
        return 0.0
    return _intersect_area(defect_box, crop_box) / float(defect_area)


def _scale_bbox_short_side(x1, y1, x2, y2, img_w, img_h, target_short):
    w = x2 - x1
    h = y2 - y1
    short = min(w, h)
    if short <= 0:
        return x1, y1, x2, y2
    scale = target_short / float(short)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nw = w * scale
    nh = h * scale
    return _clip_bbox(cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0, img_w, img_h)


def compute_crop_box(defect_box, img_w, img_h, policy=None):
    """
    Build ROI crop box around defect_box (x1,y1,x2,y2) in pixel coords.
    Returns (crop_box, crop_mode).
    """
    policy = merge_crop_policy(policy)
    expand_ratio = policy['expand_ratio']
    min_ratio = policy['min_defect_ratio']
    min_short = policy['min_short_side']
    max_short = policy['max_short_side']

    crop_box = expand_bbox(*defect_box, img_w, img_h, expand_ratio)
    crop_mode = 'roi'

    for _ in range(12):
        if _defect_ratio(defect_box, crop_box) >= min_ratio:
            break
        cx = (crop_box[0] + crop_box[2]) / 2.0
        cy = (crop_box[1] + crop_box[3]) / 2.0
        w = (crop_box[2] - crop_box[0]) * 1.25
        h = (crop_box[3] - crop_box[1]) * 1.25
        crop_box = _clip_bbox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, img_w, img_h)

    if _defect_ratio(defect_box, crop_box) < min_ratio:
        crop_box = (0, 0, img_w, img_h)
        crop_mode = 'full_image'

    short_side = min(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    if short_side < min_short:
        crop_box = _scale_bbox_short_side(*crop_box, img_w, img_h, min_short)
    elif short_side > max_short:
        crop_box = _scale_bbox_short_side(*crop_box, img_w, img_h, max_short)

    return crop_box, crop_mode


def crop_image(image, crop_box):
    x1, y1, x2, y2 = crop_box
    return image.crop((x1, y1, x2, y2))


def build_roi_record(defect_box, crop_box, crop_mode, img_w, img_h, source_dataset, extra=None):
    record = {
        'bbox_defect': list(defect_box),
        'bbox_crop': list(crop_box),
        'defect_ratio': round(_defect_ratio(defect_box, crop_box), 4),
        'crop_mode': crop_mode,
        'image_size': [img_w, img_h],
        'crop_size': [crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]],
        'source_dataset': source_dataset,
    }
    if extra:
        record.update(extra)
    return record


def save_rgb_jpg(image, path, quality=95):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(path, format='JPEG', quality=quality)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_aviation_sample(
    dst_dir,
    rgb_crop,
    source_meta=None,
    roi_meta=None,
    rgb_full=None,
    save_full=True,
):
    """Write one aviation sample folder (rgb.jpg + optional sidecars)."""
    os.makedirs(dst_dir, exist_ok=True)
    save_rgb_jpg(rgb_crop, os.path.join(dst_dir, 'rgb.jpg'))
    if save_full and rgb_full is not None:
        save_rgb_jpg(rgb_full, os.path.join(dst_dir, 'rgb_full.jpg'))
    if roi_meta is not None:
        write_json(os.path.join(dst_dir, 'roi.json'), roi_meta)
    if source_meta is not None:
        write_json(os.path.join(dst_dir, 'source.json'), source_meta)


def save_normal_sample(dst_dir, image, source_meta=None):
    """Normal samples: full image as rgb.jpg (no crop)."""
    save_aviation_sample(
        dst_dir,
        rgb_crop=image,
        source_meta=source_meta,
        roi_meta=None,
        rgb_full=None,
        save_full=False,
    )


def defect_crop_from_box(image, defect_box, source_dataset, policy=None, extra=None):
    """Return (crop_image, full_rgb, roi_meta) from PIL image and defect pixel bbox."""
    img_w, img_h = image.size
    crop_box, crop_mode = compute_crop_box(defect_box, img_w, img_h, policy)
    crop_img = crop_image(image, crop_box)
    roi_meta = build_roi_record(
        defect_box, crop_box, crop_mode, img_w, img_h, source_dataset, extra=extra)
    return crop_img, image.convert('RGB'), roi_meta


def defect_crop_from_mask(image, mask_img, source_dataset, policy=None, extra=None):
    defect_box = mask_to_bbox(mask_img)
    if defect_box is None:
        return None, None, None
    return defect_crop_from_box(image, defect_box, source_dataset, policy, extra)


def save_flat_jpg(image, path, quality=95):
    """Save a single JPEG for pretrain/ flat layouts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_rgb_jpg(image, path, quality=quality)
