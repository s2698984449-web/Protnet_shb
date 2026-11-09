"""目标域数据与评测工具：读图、伪批次、支持/查询划分、图像自助。"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGE_SUFFIX = {'.jpg', '.jpeg', '.png', '.bmp'}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def list_images(root):
    return sorted(
        p for p in Path(root).rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIX)


def load_rgb(path, resize):
    img = Image.open(path).convert('RGB')
    return transforms.ToTensor()(resize(img))


def load_mask(path, size):
    if path is None or not Path(path).is_file():
        return None
    img = Image.open(path).convert('L')
    img = transforms.Resize(
        (size, size), interpolation=transforms.InterpolationMode.NEAREST)(img)
    return transforms.ToTensor()(img)


def mask_path_for(image_path, merged_dir, mask_dir):
    rel = Path(image_path).relative_to(merged_dir)
    return Path(mask_dir) / rel.parent / (Path(image_path).stem + '.png')


@torch.no_grad()
def encode_layer4(model, batch, device):
    _, layer4 = model(batch.to(device))
    return layer4


def gray_fill():
    return torch.tensor(MEAN).view(3, 1, 1)


def geometry_from_mask(mask):
    if mask is None:
        return None, 0.0
    defect = mask[0] >= 0.5
    if not bool(defect.any()):
        return None, 0.0
    ys, xs = torch.where(defect)
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    area = float(defect.float().mean())
    aspect = float(h / max(w, 1))
    cy = float(ys.float().mean() / max(mask.shape[-2] - 1, 1))
    cx = float(xs.float().mean() / max(mask.shape[-1] - 1, 1))
    return torch.tensor([area, aspect, cx, cy], dtype=torch.float32), area


def image_stats(rgb):
    gray = rgb.mean(dim=0)
    hist = torch.histc(gray, bins=8, min=0.0, max=1.0)
    hist = hist / hist.sum().clamp_min(1e-6)
    return torch.cat([gray.mean().view(1), gray.std(unbiased=False).view(1), hist])


def paint_gray(rgb, defect_mask, keep_defect):
    fill = gray_fill().expand_as(rgb)
    keep = defect_mask if keep_defect else ~defect_mask
    keep = keep.unsqueeze(0).float()
    return rgb * keep + fill * (1.0 - keep)


def paste_same_area_random_aspect(rgb, defect_mask, rng):
    """盖住原框：灰填缺陷后贴同面积随机长宽比纹理。"""
    ys, xs = torch.where(defect_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    area = max(1, (y1 - y0) * (x1 - x0))
    H, W = rgb.shape[-2], rgb.shape[-1]
    out = paint_gray(rgb, defect_mask, keep_defect=False)
    aspect = float(rng.uniform(0.35, 2.8))
    nh = int(max(1, min(H, round((area / aspect) ** 0.5))))
    nw = int(max(1, min(W, round(area / max(nh, 1)))))
    nh = min(nh, H)
    nw = min(nw, W)
    if nh >= H or nw >= W:
        return out
    for _ in range(80):
        yy = int(rng.randint(0, H - nh))
        xx = int(rng.randint(0, W - nw))
        if yy + nh <= y0 or yy >= y1 or xx + nw <= x0 or xx >= x1:
            patch = rgb[:, yy:yy + nh, xx:xx + nw].clone()
            ty = int(rng.randint(0, H - nh))
            tx = int(rng.randint(0, W - nw))
            out[:, ty:ty + nh, tx:tx + nw] = patch
            return out
    return out


def collect_target_items(cfg, source):
    merged = Path(cfg['merged_dir'])
    mask_root = Path(cfg['mask_dir'])
    source_cfg = (cfg.get('source_specs') or {}).get(source, {})
    requested = list(source_cfg.get('classes') or cfg.get('classes') or [])
    drop = set(source_cfg.get('drop_classes') or [])
    max_box = source_cfg.get('max_box_frac')
    classes = [c for c in requested if c not in drop]
    items = []
    skipped = {
        'no_mask': 0,
        'box_frac': 0,
        'class_drop': {c: 0 for c in drop},
        'drop_classes': sorted(drop),
    }
    for split in cfg.get('splits', ['test']):
        for cls in drop:
            folder = merged / split / cls / source
            if folder.is_dir():
                skipped['class_drop'][cls] = len(list_images(folder))
        for cls in classes:
            folder = merged / split / cls / source
            if not folder.is_dir():
                continue
            for image in list_images(folder):
                mask_path = mask_path_for(image, merged, mask_root)
                mask = load_mask(mask_path, int(cfg['image_size']))
                if mask is None:
                    skipped['no_mask'] += 1
                    continue
                geo, area = geometry_from_mask(mask)
                if geo is None:
                    skipped['no_mask'] += 1
                    continue
                if max_box is not None and area > float(max_box):
                    skipped['box_frac'] += 1
                    continue
                items.append({
                    'class': cls,
                    'source': source,
                    'image': image,
                    'mask': mask_path,
                    'geometry': geo,
                    'box_frac': area,
                })
    return items, classes, skipped


def power_from_ci(ci, alpha=0.05, power=0.8):
    if not ci or len(ci) < 2 or ci[0] is None or ci[1] is None:
        return None
    z_alpha = 1.959964
    z_beta = 0.841621
    half = abs(float(ci[1]) - float(ci[0])) / 2.0
    se = half / z_alpha
    mde = (z_alpha + z_beta) * se
    return {
        'se_approx': round(se, 4),
        'need_point_for_ci_lower_gt0': round(z_alpha * se, 4),
        'mde80_paired': round(mde, 4),
        'mde80_indep_two_methods': round(mde * (2 ** 0.5), 4),
        'note': '配对比较同一批图像时用 mde80_paired；两套独立重跑用 mde80_indep',
    }


def sample_support_cross(query_pool, support_pool, classes, k, rng):
    by_s = {name: [r for r in support_pool if r['class'] == name] for name in classes}
    by_q = {name: [r for r in query_pool if r['class'] == name] for name in classes}
    for name in classes:
        if len(by_s[name]) < k or len(by_q[name]) < 1:
            return None, None
    support = []
    for name in classes:
        support.extend(rng.sample(by_s[name], k))
    support_ids = {r['image'] for r in support}
    query = [r for r in query_pool if r['image'] not in support_ids]
    if not query or any(sum(1 for r in query if r['class'] == name) == 0 for name in classes):
        return None, None
    return support, query


def kmeans_labels(vectors, n_clusters, seed):
    data = torch.stack(vectors).numpy().astype(np.float64)
    n = data.shape[0]
    k = max(2, min(int(n_clusters), n // 2 if n >= 4 else max(1, n)))
    rng = np.random.RandomState(seed)
    centers = data[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(40):
        dist = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(k):
            chosen = data[labels == index]
            if len(chosen):
                centers[index] = chosen.mean(axis=0)
    return labels.tolist(), k


def assign_batches(records, n_clusters, seed, scheme):
    if scheme == 'random':
        rng = np.random.RandomState(seed)
        k = max(2, min(n_clusters, max(1, len(records) // 2)))
        labels = rng.randint(0, k, size=len(records))
        for row, label in zip(records, labels):
            row['batch'] = int(label)
        return k
    labels, k = kmeans_labels([row['stats'] for row in records], n_clusters, seed)
    for row, label in zip(records, labels):
        row['batch'] = int(label)
    return k


def sample_support(records, classes, k, rng):
    by_class = {name: [row for row in records if row['class'] == name] for name in classes}
    for name in classes:
        if len(by_class[name]) < k + 1:
            return None, None
    support = []
    support_batches = set()
    for name in classes:
        chosen = rng.sample(by_class[name], k)
        support.extend(chosen)
        support_batches.update(row['batch'] for row in chosen)
    query = [row for row in records if row['batch'] not in support_batches]
    if not query or any(sum(1 for row in query if row['class'] == name) == 0 for name in classes):
        support_ids = {row['image'] for row in support}
        query = [row for row in records if row['image'] not in support_ids]
    if not query:
        return None, None
    return support, query


def quantiles(values):
    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def bootstrap_images(rows, repeats, seed, cluster_ids=None):
    if not rows:
        return {}
    rng = np.random.RandomState(seed)
    n = len(rows)
    if cluster_ids is not None:
        clusters = {}
        for row, cid in zip(rows, cluster_ids):
            clusters.setdefault(cid, []).append(row)
        keys = list(clusters)

        def draw():
            chosen = []
            pick = rng.choice(keys, size=len(keys), replace=True)
            for key in pick:
                chosen.extend(clusters[key])
            return chosen
    else:
        def draw():
            idx = rng.randint(0, n, size=n)
            return [rows[i] for i in idx]

    defect_accs, bg_accs, geo_accs = [], [], []
    delta_bg, delta_hard = [], []
    for _ in range(repeats):
        sample = draw()
        m = len(sample)
        d = sum(r['defect'] for r in sample) / m
        b = sum(r['bg'] for r in sample) / m
        g = sum(r['geo'] for r in sample) / m
        defect_accs.append(d)
        bg_accs.append(b)
        geo_accs.append(g)
        delta_bg.append(d - b)
        delta_hard.append(d - max(b, g))

    def pack(vals):
        mean, lo, hi = quantiles(vals)
        return round(mean, 4), [round(lo, 4), round(hi, 4)]

    d_mean, d_ci = pack(defect_accs)
    b_mean, b_ci = pack(bg_accs)
    g_mean, g_ci = pack(geo_accs)
    db_mean, db_ci = pack(delta_bg)
    dh_mean, dh_ci = pack(delta_hard)
    return {
        'n_images': n,
        'model_acc': d_mean,
        'model_ci': d_ci,
        'outside_acc': b_mean,
        'outside_ci': b_ci,
        'geometry_acc': g_mean,
        'geometry_ci': g_ci,
        'delta': db_mean,
        'delta_ci': db_ci,
        'delta_vs_hard': dh_mean,
        'delta_vs_hard_ci': dh_ci,
        'pass_vs_bg': bool(db_ci[0] > 0),
        'pass_vs_hard': bool(dh_ci[0] > 0),
        'chance': None,
    }
