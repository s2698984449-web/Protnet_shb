"""
Convert SYNDCAR YOLO bbox -> Phase 0 + CMPL meta-test pool.

Output:
  data/车/processed/pretrain/task_e_syndcar/<class>/<id>.jpg   (含全 4 类)
  data/车/processed/meta_train/roidb/<defect_class>/<source>__<specimen>__<obj_idx>.jpg
  data/车/processed/meta_test/syndcar_official/<class>/<id>.jpg
  data/车/processed/meta_train/pool.json     # 追加 broken_glass / crack / scratch / broken_light
  data/车/processed/meta_train/manifest.csv

SYNDCAR 4 类 (从 data_damage.yaml 读):
  0: Broken_Glass   -> broken_glass  (合成 3D 渲染, 含玻璃特征)
  1: Cracks        -> crack         (合成 3D 渲染, 折线纹理)
  2: Scratches     -> scratch       (合成 3D 渲染, 线状)
  3: Broken_Lights -> broken_light  (合成 3D 渲染, 灯具破坏)

Work condition: wc_synthetic (3D 渲染均匀光)
Source: syndcar

注意: SYNDCAR 在元训练中只占小比例 (0.05 pretrain), 且放到 meta_test 作为
OOD 鲁棒性测试集 — 因为元测试要求 unseen 类，但这里用同域 synthetic->synthetic。

Usage (project root):
  python scripts/convert_syndcar.py --all --check
  python scripts/convert_syndcar.py --pretrain
  python scripts/convert_syndcar.py --meta_pool
  python scripts/convert_syndcar.py --meta_test
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
SYNDCAR_ROOT = os.path.join(
    DATA_ROOT, 'SYNDCAR Synthetic Vehicle Damage Detection and Par',
    'SYNDCAR Synthetic Vehicle Damage Detection and Par',
)

# SYNDCAR 4 类 ID -> 全局 defect_type
SYNDCAR_CLS_TO_DEFECT = {
    0: 'sd_broken_glass',   # syndcar 命名空间
    1: 'sd_crack',
    2: 'sd_scratch',
    3: 'sd_broken_light',
}

WORK_CONDITION = 'wc_synthetic'
SOURCE = 'syndcar'


def _iter_split(root, split_dir='images', label_dir='labels_damage'):
    img_dir = os.path.join(root, split_dir)
    lab_dir = os.path.join(root, label_dir)
    if not (os.path.isdir(img_dir) and os.path.isdir(lab_dir)):
        return
    for fname in sorted(os.listdir(lab_dir)):
        if not fname.endswith('.txt'):
            continue
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, base + '.png')
        if not os.path.isfile(img_path):
            # 也尝试 jpg
            for ext in ('.jpg', '.jpeg'):
                cand = os.path.join(img_dir, base + ext)
                if os.path.isfile(cand):
                    img_path = cand
                    break
        if not os.path.isfile(img_path):
            continue
        yield img_path, os.path.join(lab_dir, fname)


def _read_yolo_labels(label_path):
    """Yield (cls_id, cx, cy, w, h) from YOLO txt; skip unknown cls."""
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
            if cls not in SYNDCAR_CLS_TO_DEFECT:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            rows.append((cls, cx, cy, w, h))
    return rows


def _convert(out_root, crop_policy, mode):
    """
    mode: 'pretrain' | 'meta_pool' | 'meta_test'
    """
    assert mode in ('pretrain', 'meta_pool', 'meta_test')

    if mode == 'pretrain':
        base = os.path.join(out_root, 'pretrain', 'task_e_syndcar')
    elif mode == 'meta_test':
        base = os.path.join(out_root, 'meta_test', 'syndcar_official')
    else:  # meta_pool
        base = None  # 不共享单一 root，单独写 roidb

    pool_entries = defaultdict(list)
    manifest = []
    counts = defaultdict(int)
    skipped = 0

    # 切分: 前 70% train-pool / 后 30% test
    all_items = list(_iter_split(SYNDCAR_ROOT))
    if mode in ('pretrain', 'meta_pool'):
        items = all_items  # 全量
    else:
        # meta_test: 取后 30% 作为 OOD 测试
        n = len(all_items)
        items = all_items[int(n * 0.7):]

    for img_path, label_path in items:
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
                    img, bbox, 'syndcar', crop_policy,
                    extra={'yolo_box': (cx, cy, w, h), 'cls_id': cls},
                )
                defect = SYNDCAR_CLS_TO_DEFECT[cls]

                if mode in ('pretrain', 'meta_test'):
                    fname = '{}__{}.jpg'.format(specimen, oi)
                    out_path = os.path.join(base, defect, fname)
                    save_rgb_jpg(crop, out_path)
                    counts[defect] += 1
                else:  # meta_pool
                    fname = 'syndcar__{}__{}.jpg'.format(specimen, oi)
                    out_dir = os.path.join(
                        out_root, 'meta_train', 'roidb', defect)
                    out_path = os.path.join(out_dir, fname)
                    save_rgb_jpg(crop, out_path)
                    pool_entries[defect].append(out_path)
                    manifest.append({
                        'path': out_path,
                        'defect_type': defect,
                        'work_condition': WORK_CONDITION,
                        'source_dataset': SOURCE,
                        'specimen_id': specimen,
                    })
        except Exception as e:
            print('[skip] {}: {}'.format(label_path, e))
            skipped += 1

    if mode == 'meta_pool':
        # 合并写 pool.json (命名空间隔离, 不覆盖)
        pool_path = os.path.join(out_root, 'meta_train', 'pool.json')
        existing = {}
        if os.path.isfile(pool_path):
            with open(pool_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        for k, v in pool_entries.items():
            existing[k] = sorted(set(existing.get(k, []) + v))
        with open(pool_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        # 追加 manifest
        manifest_path = os.path.join(out_root, 'meta_train', 'manifest.csv')
        write_header = not os.path.isfile(manifest_path)
        with open(manifest_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=['path', 'defect_type', 'work_condition', 'source_dataset', 'specimen_id'])
            if write_header:
                writer.writeheader()
            writer.writerows(manifest)

        return dict({k: len(v) for k, v in pool_entries.items()}), len(manifest)
    else:
        return dict(counts), skipped


def convert_syndcar_pretrain(out_root, crop_policy):
    return _convert(out_root, crop_policy, 'pretrain')


def convert_syndcar_meta_pool(out_root, crop_policy):
    return _convert(out_root, crop_policy, 'meta_pool')


def convert_syndcar_meta_test(out_root, crop_policy):
    return _convert(out_root, crop_policy, 'meta_test')


def check(out_root):
    pretrain_root = os.path.join(out_root, 'pretrain', 'task_e_syndcar')
    metatest_root = os.path.join(out_root, 'meta_test', 'syndcar_official')
    pool_path = os.path.join(out_root, 'meta_train', 'pool.json')

    print('=== syndcar pretrain counts ===')
    for cls in SYNDCAR_CLS_TO_DEFECT.values():
        d = os.path.join(pretrain_root, cls)
        n = len([f for f in os.listdir(d) if f.endswith('.jpg')]) if os.path.isdir(d) else 0
        print('  {:<14s}: {:>6d}'.format(cls, n))

    print('=== syndcar meta_test counts ===')
    for cls in SYNDCAR_CLS_TO_DEFECT.values():
        d = os.path.join(metatest_root, cls)
        n = len([f for f in os.listdir(d) if f.endswith('.jpg')]) if os.path.isdir(d) else 0
        print('  {:<14s}: {:>6d}'.format(cls, n))

    print('=== syndcar in meta_train/pool.json ===')
    if os.path.isfile(pool_path):
        with open(pool_path, 'r', encoding='utf-8') as f:
            pool = json.load(f)
        for k in SYNDCAR_CLS_TO_DEFECT.values():
            print('  {:<14s}: {:>6d}'.format(k, len(pool.get(k, []))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--meta_pool', action='store_true')
    parser.add_argument('--meta_test', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    out_root = AUTOMOTIVE_PROCESSED
    policy = merge_crop_policy()

    if args.all or args.pretrain:
        print('[1/3] converting syndcar -> pretrain/task_e_syndcar ...')
        counts, skipped = convert_syndcar_pretrain(out_root, policy)
        print('  done, skipped={}'.format(skipped))
        for k, v in sorted(counts.items()):
            print('    {}: {}'.format(k, v))

    if args.all or args.meta_pool:
        print('[2/3] converting syndcar -> meta_train/pool.json + manifest.csv ...')
        pool_counts, total = convert_syndcar_meta_pool(out_root, policy)
        print('  done, total ROIs={}'.format(total))
        for k, v in sorted(pool_counts.items()):
            print('    {}: {}'.format(k, v))

    if args.all or args.meta_test:
        print('[3/3] converting syndcar -> meta_test/syndcar_official ...')
        counts, skipped = convert_syndcar_meta_test(out_root, policy)
        print('  done, skipped={}'.format(skipped))
        for k, v in sorted(counts.items()):
            print('    {}: {}'.format(k, v))

    if args.check:
        check(out_root)


if __name__ == '__main__':
    main()
