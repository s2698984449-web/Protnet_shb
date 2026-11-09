"""双区域特征：core / ring 池化与 cosine / L2 原型分类。供工业交付复用。"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .target_shot import (
    encode_layer4,
    image_stats,
    load_mask,
    load_rgb,
    paste_same_area_random_aspect,
)


def dilate_bool(mask, radius):
    if radius <= 0:
        return mask.clone()
    m = mask.float().view(1, 1, *mask.shape)
    k = 2 * int(radius) + 1
    out = F.max_pool2d(m, kernel_size=k, stride=1, padding=int(radius))
    return out[0, 0] >= 0.5


def resize_mask(mask, h, w):
    m = mask.float().view(1, 1, *mask.shape)
    m = F.interpolate(m, size=(h, w), mode='nearest')
    return m[0, 0] >= 0.5


def masked_mean(feat, mask):
    mask = mask.to(device=feat.device)
    if not bool(mask.any()):
        return feat.detach().cpu().mean(dim=(1, 2)).float()
    w = mask.float()
    vec = (feat * w.unsqueeze(0)).sum(dim=(1, 2)) / w.sum().clamp_min(1.0)
    return vec.detach().cpu().float()


@torch.no_grad()
def encode_core_context(model, items, resize, normalize, device, seed, dilate_px=16):
    """整图 layer4；core=mask 内；ring=膨胀环；outside=贴块 GAP。"""
    records = []
    size = resize.size[0] if isinstance(resize.size, (tuple, list)) else resize.size
    rng = np.random.RandomState(seed)
    for row in items:
        rgb = load_rgb(row['image'], resize)
        mask = load_mask(row['mask'], size)
        defect_mask = mask[0] >= 0.5
        if not bool(defect_mask.any()):
            continue
        x = normalize(rgb).unsqueeze(0)
        layer4 = encode_layer4(model, x, device)[0]
        _, h, w = layer4.shape
        core_m = resize_mask(defect_mask, h, w)
        dil_m = resize_mask(dilate_bool(defect_mask, dilate_px), h, w)
        ctx_m = dil_m & (~core_m)
        if not bool(ctx_m.any()):
            ctx_m = dil_m
        core = masked_mean(layer4, core_m)
        context = masked_mean(layer4, ctx_m)
        bg_img = normalize(paste_same_area_random_aspect(rgb, defect_mask, rng)).unsqueeze(0)
        outside = encode_layer4(model, bg_img, device)[0].cpu().mean(dim=(1, 2)).float()
        records.append({
            'class': row['class'],
            'source': row['source'],
            'image': str(row['image']),
            'stem': Path(row['image']).stem,
            'core': core,
            'context': context,
            'outside': outside,
            'geometry': row['geometry'].float(),
            'stats': image_stats(rgb).float(),
            'box_frac': row['box_frac'],
            'batch': 0,
        })
    return records


def _unit(v):
    return v / v.norm().clamp_min(1e-6)


def pairwise_dist(query_vec, protos, metric):
    if metric == 'l2':
        return torch.norm(protos - query_vec.unsqueeze(0), dim=1)
    if metric == 'cosine':
        q = _unit(query_vec)
        p = protos / protos.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return 1.0 - (p * q.unsqueeze(0)).sum(dim=1)
    raise ValueError(metric)


def classify_branch(support, query, field, classes, metric, inv_std=None):
    del inv_std  # 基线不再使用 whitening
    protos, labels = [], []
    for name in classes:
        vecs = [row[field] for row in support if row['class'] == name]
        if not vecs:
            continue
        protos.append(torch.stack(vecs).mean(0))
        labels.append(name)
    if len(labels) < 2:
        return {}, {}
    cand = torch.stack(protos)
    pred, truth = {}, {}
    for row in query:
        if row['class'] not in labels:
            continue
        dist = pairwise_dist(row[field], cand, metric)
        pred[row['image']] = labels[int(dist.argmin())]
        truth[row['image']] = row['class']
    return truth, pred


def classify_combo(support, query, classes, metric, alpha, beta,
                   inv_std_core=None, inv_std_ctx=None):
    del inv_std_core, inv_std_ctx
    core_p, ctx_p, labels = [], [], []
    for name in classes:
        cvecs = [row['core'] for row in support if row['class'] == name]
        xvecs = [row['context'] for row in support if row['class'] == name]
        if not cvecs or not xvecs:
            continue
        core_p.append(torch.stack(cvecs).mean(0))
        ctx_p.append(torch.stack(xvecs).mean(0))
        labels.append(name)
    if len(labels) < 2:
        return {}, {}
    core_p = torch.stack(core_p)
    ctx_p = torch.stack(ctx_p)
    pred, truth = {}, {}
    for row in query:
        if row['class'] not in labels:
            continue
        d_c = pairwise_dist(row['core'], core_p, metric)
        d_x = pairwise_dist(row['context'], ctx_p, metric)
        dist = float(alpha) * d_c + float(beta) * d_x
        pred[row['image']] = labels[int(dist.argmin())]
        truth[row['image']] = row['class']
    return truth, pred
