"""
Convert minor-dent YOLO bbox -> Phase 0 + CMPL meta-train pool.

Output:
  data/车/processed/pretrain/task_b_minor_dent/dent/<id>.jpg
  data/车/processed/meta_train/roidb/dent/<source>__<specimen>__<obj_idx>.jpg
  data/车/processed/meta_train/pool.json     # 追加 dent 类
  data/车/processed/meta_train/manifest.csv  # 追加 (defect_type=dent, wc=wc_parking)

Work condition: wc_parking (停车场+夜间/低光, 多车辆, 复杂背景)
1 defect type: dent (single-class minor dent)

Usage (project root):
  python scripts/convert_minor_dent.py --all --check
  python scripts/convert_minor_dent.py --pretrain
  python scripts/convert_minor_dent.py --meta_pool
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.crop_utils import (
    defect_crop_from_box,
    merge_crop_policy,
    save_rgb_jpg,
    yolo_box_to_pixels,
)
from common.project_paths import AUTOMOTIVE_PROCESSED

DATA_ROOT = os.path.abspath(os.path.join(AUTOMOTIVE_PROCESSED, '..', '..'))
MINORDENT_ROOT = os.path.join(
    DATA_ROOT, 'minor-dent-detection-dataset-v1.0',
    'danbaig-minor-dent-detection-dataset-21db078',
)

# 1 类
DEFECT_TYPE = 'md_dent'  # minor-dent 命名空间
WORK_CONDITION = 'wc_parking'
SOURCE = 'minor_dent'

# Phase 0 multi-class: dent 单类
PRETRAIN_CLASS = 'dent'


def _iter_split(root, split):
    """Yield (img_path, label_path) pairs from minor-dent <split>."""
    img_dir = os.path.join(root, split, 'images')
    lab_dir = os.path.join(root, split, 'labels')
    if not (os.path.isdir(img_dir) and os.path.isdir(lab_dir)):
        return
    for fname in sorted(os.listdir(lab_dir)):
        if not fname.endswith('.txt'):
            continue
        base = os.path.splitext(fname)[0]
        # minor-dent 标注用 .txt；图可能是 png/jpg
        img_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = os.path.join(img_dir, base + ext)
            if os.path.isfile(cand):
                img_path = cand
                break
        if img_path is None:
            continue
        yield img_path, os.path.join(lab_dir, fname)


def _read_yolo_labels(label_path):
    """Yield (cls_id, cx, cy, w, h) from YOLO txt; only cls==0 kept."""
    rows = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                continue
            if cls != 0:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            rows.append((cls, cx, cy, w, h))
    return rows


def convert_minordent_pretrain(out_root, crop_policy, splits=('train', 'valid', 'test')):
    dest = os.path.join(out_root, 'pretrain', 'task_b_minor_dent', PRETRAIN_CLASS)
    counts = defaultdict(int)
    skipped = 0

    for split in splits:
        for img_path, label_path in _iter_split(MINORDENT_ROOT, split):
            try:
                labels = _read_yolo_labels(label_path)
                if not labels:
                    continue
                img = Image.open(img_path).convert('RGB')
                img_w, img_h = img.size
                specimen = os.path.splitext(os.path.basename(img_path))[0]
                for oi, (cls, cx, cy, w, h) in enumerate(labels):
                    bbox = yolo_box_to_pixels(cx, cy, w, h, img_w, img_h)
                    crop, _, _ = defect_crop_from_box(
                        img, bbox, 'minor_dent', crop_policy,
                        extra={'split': split, 'yolo_box': (cx, cy, w, h)},
                    )
                    fname = '{}_{}_{}.jpg'.format(split, specimen, oi)
                    out_path = os.path.join(dest, fname)
                    save_rgb_jpg(crop, out_path)
                    counts[split] += 1
            except Exception as e:
                print('[skip] {}: {}'.format(label_path, e))
                skipped += 1
    return dict(counts), skipped


def convert_minordent_meta_pool(out_root, crop_policy, splits=('train', 'valid', 'test')):
    roidb_dir = os.path.join(out_root, 'meta_train', 'roidb', DEFECT_TYPE)
    os.makedirs(roidb_dir, exist_ok=True)
    pool = []
    manifest = []

    for split in splits:
        for img_path, label_path in _iter_split(MINORDENT_ROOT, split):
            try:
                labels = _read_yolo_labels(label_path)
                if not labels:
                    continue
                img = Image.open(img_path).convert('RGB')
                img_w, img_h = img.size
                specimen = os.path.splitext(os.path.basename(img_path))[0]
                for oi, (cls, cx, cy, w, h) in enumerate(labels):
                    bbox = yolo_box_to_pixels(cx, cy, w, h, img_w, img_h)
                    crop, _, _ = defect_crop_from_box(
                        img, bbox, 'minor_dent', crop_policy,
                        extra={'split': split, 'yolo_box': (cx, cy, w, h)},
                    )
                    fname = 'minor__{}__{}__{}.jpg'.format(split, specimen, oi)
                    out_path = os.path.join(roidb_dir, fname)
                    save_rgb_jpg(crop, out_path)
                    pool.append(out_path)
                    manifest.append({
                        'path': out_path,
                        'defect_type': DEFECT_TYPE,
                        'work_condition': WORK_CONDITION,
                        'source_dataset': SOURCE,
                        'specimen_id': '{}_{}'.format(split, specimen),
                    })
            except Exception as e:
                print('[skip] {}: {}'.format(label_path, e))

    _merge_pool(out_root, {DEFECT_TYPE: pool})
    _append_manifest(out_root, manifest)

    return len(pool), len(manifest)


def _merge_pool(out_root, new_entries):
    """Merge without overwrite."""
    pool_path = os.path.join(out_root, 'meta_train', 'pool.json')
    existing = {}
    if os.path.isfile(pool_path):
        with open(pool_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    for k, v in new_entries.items():
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
    pretrain_dir = os.path.join(out_root, 'pretrain', 'task_b_minor_dent', PRETRAIN_CLASS)
    print('=== minor_dent pretrain counts ===')
    if os.path.isdir(pretrain_dir):
        print('  dent: {}'.format(len([f for f in os.listdir(pretrain_dir) if f.endswith('.jpg')])))
    else:
        print('  not generated yet')
    print('=== minor_dent in meta_train/pool.json ===')
    pool_path = os.path.join(out_root, 'meta_train', 'pool.json')
    if os.path.isfile(pool_path):
        with open(pool_path, 'r', encoding='utf-8') as f:
            pool = json.load(f)
        print('  dent: {}'.format(len(pool.get('dent', []))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--meta_pool', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--splits', default='train,valid,test')
    args = parser.parse_args()

    out_root = AUTOMOTIVE_PROCESSED
    policy = merge_crop_policy()
    splits = tuple(s.strip() for s in args.splits.split(',') if s.strip())

    if args.all or args.pretrain:
        print('[1/2] converting minor_dent -> pretrain/task_b_minor_dent/dent ...')
        counts, skipped = convert_minordent_pretrain(out_root, policy, splits)
        print('  done, skipped={}'.format(skipped))
        for k, v in sorted(counts.items()):
            print('    split {}: {}'.format(k, v))

    if args.all or args.meta_pool:
        print('[2/2] converting minor_dent -> meta_train/pool.json + manifest.csv ...')
        n_pool, n_manifest = convert_minordent_meta_pool(out_root, policy, splits)
        print('  done, pool ROIs={}, manifest rows={}'.format(n_pool, n_manifest))

    if args.check:
        check(out_root)


if __name__ == '__main__':
    main()
