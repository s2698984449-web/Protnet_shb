"""
Convert car_parts_damage polygon annotations -> Phase 0 + CMPL meta-train pool.

Output:
  data/车/processed/pretrain/task_c_carparts_damage/<class>/<id>.jpg
  data/车/processed/meta_train/roidb/<defect_class>/<source>__<id>.jpg
  data/车/processed/meta_train/pool.json     # {defect_class: [path,...]}
  data/车/processed/meta_train/manifest.csv  # 每张 ROI 的 (defect, work_condition, source)

8 defect classes (car_parts native):
  - Scratch, Dent, Cracked, Paint chip, Flaking, Broken part, Missing part, Corrosion

Work condition mapping (car_parts 全部来自车间集中光):
  - Scratch / Dent / Cracked: wc_garage (mid-shot)
  - Paint chip / Flaking / Corrosion: wc_closeup (micro)
  - Broken part / Missing part: wc_part_view (cropped part)

Usage (project root):
  python scripts/convert_carparts_damage.py --all --check
  python scripts/convert_carparts_damage.py --pretrain
  python scripts/convert_carparts_damage.py --meta_pool
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.crop_utils import (
    defect_crop_from_box,
    expand_bbox,
    merge_crop_policy,
    save_rgb_jpg,
    _clip_bbox,
)
from common.project_paths import AUTOMOTIVE_PROCESSED

# 三个新数据集都直接在 data/ 下，不在 data/车/raw/ 下
DATA_ROOT = os.path.abspath(os.path.join(AUTOMOTIVE_PROCESSED, '..', '..'))
CARPARTS_ROOT = os.path.join(DATA_ROOT, 'car_parts_damage', 'Car parts dataset')

# 8 原生类 (car_parts_damage 自带)
CARPARTS_CLASS_TITLES = (
    'Scratch', 'Dent', 'Cracked', 'Paint chip',
    'Flaking', 'Broken part', 'Missing part', 'Corrosion',
)

# 全局 defect_type 命名 (用于 meta_train pool 跨数据集对齐)
CARPARTS_TITLE_TO_DEFECT = {
    'Scratch': 'cp_scratch',     # carparts 命名空间 (避免与 syndcar.scratch 冲突)
    'Dent': 'cp_dent',
    'Cracked': 'cp_cracked',
    'Paint chip': 'cp_paint_chip',
    'Flaking': 'cp_flaking',
    'Broken part': 'cp_broken_part',
    'Missing part': 'cp_missing_part',
    'Corrosion': 'cp_corrosion',
}

# 工况映射 — 基于图像视觉距离
CARPARTS_TITLE_TO_WC = {
    'Scratch': 'wc_garage',
    'Dent': 'wc_garage',
    'Cracked': 'wc_garage',
    'Paint chip': 'wc_closeup',
    'Flaking': 'wc_closeup',
    'Corrosion': 'wc_closeup',
    'Broken part': 'wc_part_view',
    'Missing part': 'wc_part_view',
}

# Phase 0 multi-class: 沿用原 8 类作为 phase0 分类目标
CARPARTS_PRETRAIN_FOLDERS = CARPARTS_CLASS_TITLES


def _polygon_to_bbox(points_exterior):
    """Return (x1, y1, x2, y2) pixel bbox from polygon exterior points."""
    xs = [p[0] for p in points_exterior]
    ys = [p[1] for p in points_exterior]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_to_mask(points_exterior, img_w, img_h):
    """Rasterize polygon to binary mask (PIL.Image, mode 'L')."""
    mask = Image.new('L', (img_w, img_h), 0)
    if not points_exterior:
        return mask
    draw = ImageDraw.Draw(mask)
    # PIL polygon 接受 list[tuple]
    pts = [(float(x), float(y)) for x, y in points_exterior]
    draw.polygon(pts, fill=255)
    return mask


def _iter_ann_files(root):
    """Yield (img_path, ann_path) for every (img, ann) pair under root/..."""
    if not os.path.isdir(root):
        return
    # 1. 如果当前 root 直接含 ann/ + img/，则只遍历当前 root
    if os.path.isdir(os.path.join(root, 'ann')) and os.path.isdir(os.path.join(root, 'img')):
        yield from _iter_pair_dir(root)
        return
    # 2. 否则遍历一级子目录（File1, File2, ...）
    for sub in sorted(os.listdir(root)):
        sub_path = os.path.join(root, sub)
        if not os.path.isdir(sub_path) or sub.startswith('@') or sub.startswith('.'):
            continue
        # Case A: 直接含 ann + img（当前层次就是配对目录）
        if os.path.isdir(os.path.join(sub_path, 'ann')) and os.path.isdir(os.path.join(sub_path, 'img')):
            yield from _iter_pair_dir(sub_path)
            continue
        # Case B: 配对目录在下一层 (File1/ann, File1/img)
        for sub2 in os.listdir(sub_path):
            sub2_path = os.path.join(sub_path, sub2)
            if not os.path.isdir(sub2_path) or sub2.startswith('@') or sub2.startswith('.'):
                continue
            # 看 sub2 是否同时含 ann + img
            if (os.path.isdir(os.path.join(sub2_path, 'ann'))
                    and os.path.isdir(os.path.join(sub2_path, 'img'))):
                yield from _iter_pair_dir(sub2_path)


def _iter_pair_dir(pair_dir):
    """Yield (img_path, ann_path) from one (ann/, img/) pair directory."""
    ann_dir = os.path.join(pair_dir, 'ann')
    img_dir = os.path.join(pair_dir, 'img')
    if not (os.path.isdir(ann_dir) and os.path.isdir(img_dir)):
        return
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith('.json'):
            continue
        # base = 'Car damages 101.png' (剥掉 .json)
        base = fname[:-5]
        # car_parts 命名格式: ann/Car damages 101.png.json 对应 img/Car damages 101.png
        # base 已经包含扩展名, 直接在 img_dir 里找, 不需要再拼 .png/.jpg
        img_path = os.path.join(img_dir, base)
        if not os.path.isfile(img_path):
            # 兜底: 也许 json 里只写了 basename 不含扩展, 试 .png/.jpg/.jpeg
            stem = os.path.splitext(base)[0]
            for ext in ('.png', '.jpg', '.jpeg'):
                cand = os.path.join(img_dir, stem + ext)
                if os.path.isfile(cand):
                    img_path = cand
                    break
            else:
                continue
        yield img_path, os.path.join(ann_dir, fname)


def convert_carparts_pretrain(out_root, crop_policy):
    """
    输出:
      pretrain/task_c_carparts_damage/<ClassTitle>/<specimen>_<obj_idx>.jpg
    """
    dest = os.path.join(out_root, 'pretrain', 'task_c_carparts_damage')
    counts = defaultdict(int)
    skipped = 0

    for img_path, ann_path in _iter_ann_files(CARPARTS_ROOT):
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
            objects = ann.get('objects', [])
            if not objects:
                skipped += 1
                continue
            img = Image.open(img_path).convert('RGB')
            specimen = os.path.splitext(os.path.basename(img_path))[0]
            specimen_safe = specimen.replace(' ', '_').replace('.', '_')
            for oi, obj in enumerate(objects):
                class_title = obj.get('classTitle', '')
                if class_title not in CARPARTS_PRETRAIN_FOLDERS:
                    continue
                pts = obj.get('points', {}).get('exterior', [])
                if not pts:
                    continue
                bbox = _polygon_to_bbox(pts)
                crop, _, _ = defect_crop_from_box(
                    img, bbox, 'carparts_damage', crop_policy,
                    extra={'polygon_pts': len(pts), 'class_title': class_title},
                )
                fname = '{}_{}.jpg'.format(specimen_safe, oi)
                out_path = os.path.join(dest, class_title, fname)
                save_rgb_jpg(crop, out_path)
                counts[class_title] += 1
        except Exception as e:
            print('[skip] {}: {}'.format(ann_path, e))
            skipped += 1
    return dict(counts), skipped


def convert_carparts_meta_pool(out_root, crop_policy):
    """
    输出:
      meta_train/roidb/<defect_class>/<source>__<specimen>__<obj_idx>.jpg
      meta_train/pool.json  {defect_class: [abs_path, ...]}
      meta_train/manifest.csv  (path, defect_type, work_condition, source_dataset, specimen_id)
    """
    roidb_dir = os.path.join(out_root, 'meta_train', 'roidb')
    pool = defaultdict(list)
    manifest = []

    for img_path, ann_path in _iter_ann_files(CARPARTS_ROOT):
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
            objects = ann.get('objects', [])
            if not objects:
                continue
            img = Image.open(img_path).convert('RGB')
            specimen = os.path.splitext(os.path.basename(img_path))[0]
            specimen_safe = specimen.replace(' ', '_').replace('.', '_')
            for oi, obj in enumerate(objects):
                class_title = obj.get('classTitle', '')
                if class_title not in CARPARTS_TITLE_TO_DEFECT:
                    continue
                pts = obj.get('points', {}).get('exterior', [])
                if not pts:
                    continue
                bbox = _polygon_to_bbox(pts)
                crop, _, _ = defect_crop_from_box(
                    img, bbox, 'carparts_damage', crop_policy,
                    extra={'polygon_pts': len(pts), 'class_title': class_title},
                )
                defect = CARPARTS_TITLE_TO_DEFECT[class_title]
                wc = CARPARTS_TITLE_TO_WC[class_title]
                fname = 'carparts__{}__{}.jpg'.format(specimen_safe, oi)
                out_dir = os.path.join(roidb_dir, defect)
                out_path = os.path.join(out_dir, fname)
                save_rgb_jpg(crop, out_path)
                pool[defect].append(out_path)
                manifest.append({
                    'path': out_path,
                    'defect_type': defect,
                    'work_condition': wc,
                    'source_dataset': 'carparts_damage',
                    'specimen_id': specimen_safe,
                })
        except Exception as e:
            print('[skip] {}: {}'.format(ann_path, e))

    # 合并写 pool.json (carparts 命名空间 cp_* 不会与 syndcar/md 冲突)
    _merge_pool(out_root, pool)
    _append_manifest(out_root, manifest)

    return {k: len(v) for k, v in pool.items()}, len(manifest)


def _merge_pool(out_root, new_entries):
    """Merge new_entries into meta_train/pool.json without overwriting."""
    pool_path = os.path.join(out_root, 'meta_train', 'pool.json')
    existing = {}
    if os.path.isfile(pool_path):
        with open(pool_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    for k, v in new_entries.items():
        # 命名空间隔离: cp_*, md_*, sd_* 互不冲突, 可直接合并
        existing[k] = sorted(set(existing.get(k, []) + v))
    os.makedirs(os.path.dirname(pool_path), exist_ok=True)
    with open(pool_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def _append_manifest(out_root, rows):
    manifest_path = os.path.join(out_root, 'meta_train', 'manifest.csv')
    write_header = not os.path.isfile(manifest_path)
    with open(manifest_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['path', 'defect_type', 'work_condition', 'source_dataset', 'specimen_id'])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def check(out_root):
    pretrain_root = os.path.join(out_root, 'pretrain', 'task_c_carparts_damage')
    pool_path = os.path.join(out_root, 'meta_train', 'pool.json')
    print('=== carparts pretrain counts ===')
    for cls in CARPARTS_PRETRAIN_FOLDERS:
        d = os.path.join(pretrain_root, cls)
        n = len([f for f in os.listdir(d) if f.endswith('.jpg')]) if os.path.isdir(d) else 0
        print('  {:<14s}: {:>6d}'.format(cls, n))
    print('=== carparts meta_train pool ===')
    if os.path.isfile(pool_path):
        with open(pool_path, 'r', encoding='utf-8') as f:
            pool = json.load(f)
        for k, v in sorted(pool.items()):
            print('  {:<14s}: {:>6d}'.format(k, len(v)))
    else:
        print('  pool.json not found, run --meta_pool first')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--meta_pool', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--crop_policy_json', default=None)
    args = parser.parse_args()

    out_root = AUTOMOTIVE_PROCESSED
    policy = merge_crop_policy()

    if args.all or args.pretrain:
        print('[1/2] converting carparts -> pretrain/task_c_carparts_damage ...')
        counts, skipped = convert_carparts_pretrain(out_root, policy)
        print('  done, skipped={}'.format(skipped))
        for k, v in sorted(counts.items()):
            print('    {}: {}'.format(k, v))

    if args.all or args.meta_pool:
        print('[2/2] converting carparts -> meta_train/pool.json + manifest.csv ...')
        pool_counts, total = convert_carparts_meta_pool(out_root, policy)
        print('  done, total ROIs={}'.format(total))
        for k, v in sorted(pool_counts.items()):
            print('    {}: {}'.format(k, v))

    if args.check:
        check(out_root)


if __name__ == '__main__':
    main()
