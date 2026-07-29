"""
Convert automotive raw datasets -> Phase 0 pretrain + CMPL meta-learning layout.

Output (under data/车/processed):
  pretrain/cardd_cls/{train,val}/{scratch,dent,...}/*.jpg   # CarDD train/val official split
  pretrain/neu_cls/{train,val}/{6 classes}/*.jpg
  pretrain/asde_binary/{train,val}/{normal,scratch}/*.jpg
  pretrain/mvtec_tex/{train,val}/{normal,defect}/*.jpg      # Class-split Phase0b
  pretrain/mvtec_obj/{train,val}/{normal,defect}/*.jpg      # stand-in until DS-DAGM LFS
  cmpl/domain_0/...   # ASDE meta-train (scratch vs normal-as-proxy-dent)
  cmpl/domain_3/...   # CarDD test holdout only (Track 2)

Usage (project root):
  python scripts/convert_cmpl_datasets.py --all --check
  python scripts/convert_cmpl_datasets.py --pretrain --meta --meta_policy track2
  python scripts/convert_cmpl_datasets.py --pretrain_classsplit --check
  python scripts/convert_cmpl_datasets.py --pretrain --legacy_kolektor
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict

from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.cmpl_taxonomy import (
    CARDD_COCO_ID_TO_META_FOLDER,
    CARDD_COCO_TO_FOLDER,
    CARDD_PRETRAIN_CLASSES,
    CMPL_META_TWO_WAY,
    DEFAULT_CROP_POLICY,
    NEU_FILENAME_PREFIX_TO_FOLDER,
    NEU_PRETRAIN_CLASSES,
)
from common.crop_utils import (
    defect_crop_from_box, defect_crop_from_mask, merge_crop_policy,
    save_aviation_sample, save_flat_jpg,
)
from common.project_paths import AUTOMOTIVE_PROCESSED, AUTOMOTIVE_RAW, DATA_ROOT

CARDD_ROOT = os.path.join(AUTOMOTIVE_RAW, 'CarDD', 'CarDD_release')
CARDD_COCO = os.path.join(CARDD_ROOT, 'CarDD_COCO', 'annotations')
CARDD_IMAGE_DIRS = {
    'train': os.path.join(CARDD_ROOT, 'CarDD_SOD', 'CarDD-TR', 'CarDD-TR-Image'),
    'val': os.path.join(CARDD_ROOT, 'CarDD_SOD', 'CarDD-VAL', 'CarDD-VAL-Image'),
    'test': os.path.join(CARDD_ROOT, 'CarDD_SOD', 'CarDD-TE', 'CarDD-TE-Image'),
}
CARDD_DOMAIN = {'train': 0, 'val': 2, 'test': 3}  # legacy mapping; Track2 meta uses test->3 only

ASDE_ROOT = os.path.join(AUTOMOTIVE_RAW, 'ASDE', 'image')
NEU_ROOT = os.path.join(AUTOMOTIVE_RAW, 'NEU-DET', 'NEU-DET')
NEU_IMAGE_DIR = os.path.join(NEU_ROOT, 'IMAGES')
KOLEKTOR_ROOT = os.path.join(AUTOMOTIVE_RAW, 'KolektorSDD2')
MVTEC_AD_ROOT = os.path.join(AUTOMOTIVE_RAW, 'MVTec_AD')
SPECTRUM_ROOT = os.path.join(DATA_ROOT, 'Defect_Spectrum')

# Class-split Phase0b industrial sources (no car/CarDD)
MVTEC_TEXTURE_CATS = ('carpet', 'grid', 'leather', 'tile', 'wood')
MVTEC_OBJECT_CATS = (
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
)


def _load_coco(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _coco_boxes_by_image(ann_json):
    by_image = defaultdict(list)
    cats = {c['id']: c['name'] for c in ann_json['categories']}
    for ann in ann_json['annotations']:
        if ann.get('iscrowd', 0):
            continue
        cat = cats.get(ann['category_id'], '')
        x, y, w, h = ann['bbox']
        by_image[ann['image_id']].append({
            'category_id': ann['category_id'],
            'category_name': cat,
            'bbox_xywh': (x, y, w, h),
        })
    id_to_file = {img['id']: img['file_name'] for img in ann_json['images']}
    return by_image, id_to_file, cats


def _resolve_image(split, file_name):
    for candidate in (
        CARDD_IMAGE_DIRS.get(split),
        os.path.join(CARDD_ROOT, 'CarDD_COCO', 'val2017'),
    ):
        if not candidate:
            continue
        path = os.path.join(candidate, file_name)
        if os.path.isfile(path):
            return path
    return None


def _bbox_xywh_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return x, y, x + w, y + h


def convert_cardd_pretrain(out_root, crop_policy, seed=42):
    """CarDD train/val COCO -> pretrain/cardd_cls (official split preserved)."""
    random.seed(seed)
    dest = os.path.join(out_root, 'pretrain', 'cardd_cls')
    counts = defaultdict(int)
    for split in ('train', 'val'):
        ann_path = os.path.join(CARDD_COCO, 'instances_{}2017.json'.format(split))
        if not os.path.isfile(ann_path):
            continue
        by_image, id_to_file, _ = _coco_boxes_by_image(_load_coco(ann_path))
        for image_id, boxes in by_image.items():
            file_name = id_to_file[image_id]
            img_path = _resolve_image(split, file_name)
            if not img_path:
                continue
            img = Image.open(img_path).convert('RGB')
            for bi, box in enumerate(boxes):
                folder = CARDD_COCO_TO_FOLDER.get(box['category_name'])
                if folder not in CARDD_PRETRAIN_CLASSES:
                    continue
                x1, y1, x2, y2 = _bbox_xywh_to_xyxy(box['bbox_xywh'])
                crop, _, roi_meta = defect_crop_from_box(
                    img, (x1, y1, x2, y2), 'CarDD', crop_policy)
                sub = split  # official COCO train->train, val->val
                sample_name = '{}_{}_{}'.format(split, file_name.replace('.jpg', ''), bi)
                out_path = os.path.join(dest, sub, folder, sample_name + '.jpg')
                save_flat_jpg(crop, out_path)
                counts[(sub, folder)] += 1
    return dict(counts)


def convert_cardd_meta(out_root, crop_policy, splits=('test',)):
    """CarDD 2-way scratch vs dent -> cmpl/domain_* (Track2 default: test->domain_3 only)."""
    meta_root = os.path.join(out_root, 'cmpl')
    counts = defaultdict(int)
    for split in splits:
        ann_path = os.path.join(CARDD_COCO, 'instances_{}2017.json'.format(split))
        if not os.path.isfile(ann_path):
            continue
        domain_id = CARDD_DOMAIN[split]
        by_image, id_to_file, _ = _coco_boxes_by_image(_load_coco(ann_path))
        for image_id, boxes in by_image.items():
            file_name = id_to_file[image_id]
            img_path = _resolve_image(split, file_name)
            if not img_path:
                continue
            img = Image.open(img_path).convert('RGB')
            specimen = file_name.replace('.jpg', '')
            for bi, box in enumerate(boxes):
                class_folder = CARDD_COCO_ID_TO_META_FOLDER.get(box['category_id'])
                if class_folder not in CMPL_META_TWO_WAY:
                    continue
                x1, y1, x2, y2 = _bbox_xywh_to_xyxy(box['bbox_xywh'])
                crop, _, roi_meta = defect_crop_from_box(
                    img, (x1, y1, x2, y2), 'CarDD', crop_policy)
                sample_id = '{}_{}_{}'.format(specimen, class_folder.split('_')[-1], bi)
                sample_dir = os.path.join(
                    meta_root, 'domain_{}'.format(domain_id), class_folder, sample_id)
                save_aviation_sample(
                    sample_dir, crop,
                    source_meta={
                        'dataset': 'CarDD',
                        'original_path': file_name,
                        'specimen_id': specimen,
                        'split': split,
                        'coco_category': box['category_name'],
                    },
                    roi_meta=roi_meta,
                )
                counts[(domain_id, class_folder)] += 1
        _write_cmpl_domain_meta(meta_root, domain_id, 'CarDD', cardd_split=split)
    return dict(counts)


def _write_cmpl_domain_meta(meta_root, domain_id, source, cardd_split=None):
    path = os.path.join(meta_root, 'domain_{}'.format(domain_id), 'meta.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    split_policy = {
        'strategy': 'track2_asde_train_cardd_test' if source == 'ASDE' else 'track2_cardd_test_holdout',
        'meta_train_domains': [0],
        'meta_test_domain': 3,
        'meta_two_way': list(CMPL_META_TWO_WAY),
    }
    if cardd_split:
        split_policy['cardd_split'] = cardd_split
    if source == 'ASDE':
        split_policy['note'] = (
            'ASDE scratch->0_scratch, intact->3_dent (proxy 2nd way for episode construction; '
            'eval on CarDD test uses real dent)')
    meta = {
        'material_id': 'AUTOMOTIVE_PAINT' if source == 'CarDD' else 'ASDE_LINE',
        'material_params': [0.0] * 12,
        'process_params': [0.0] * 8,
        'defect_history': [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'fusion_prior': [0.45, 0.30, 0.25],
        'source': source,
        'split_policy': split_policy,
        'crop_policy': DEFAULT_CROP_POLICY,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def convert_asde_meta(out_root):
    """ASDE -> cmpl/domain_0 for meta-train (cross-domain source)."""
    meta_root = os.path.join(out_root, 'cmpl')
    domain_id = 0
    folder_map = {
        '1_scratch': '0_scratch',
        '0_intact': '3_dent',
    }
    counts = defaultdict(int)
    for src_folder, class_folder in folder_map.items():
        src = os.path.join(ASDE_ROOT, src_folder)
        if not os.path.isdir(src):
            continue
        for fname in sorted(os.listdir(src)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = Image.open(os.path.join(src, fname)).convert('RGB')
            sample_id = 'asde_{}_{}'.format(class_folder.split('_')[-1], os.path.splitext(fname)[0])
            sample_dir = os.path.join(meta_root, 'domain_{}'.format(domain_id), class_folder, sample_id)
            save_aviation_sample(
                sample_dir, img,
                source_meta={
                    'dataset': 'ASDE',
                    'original_path': fname,
                    'asde_folder': src_folder,
                    'class_folder': class_folder,
                },
                roi_meta={'full_image': True},
            )
            counts[(domain_id, class_folder)] += 1
    _write_cmpl_domain_meta(meta_root, domain_id, 'ASDE')
    return dict(counts)


def _neu_class_from_filename(fname):
    prefix = fname.split('_')[0].lower()
    return NEU_FILENAME_PREFIX_TO_FOLDER.get(prefix)


def convert_neu_pretrain(out_root, seed=42, val_ratio=0.2):
    """NEU-DET 6-class -> pretrain/neu_cls (stratified train/val)."""
    random.seed(seed)
    dest = os.path.join(out_root, 'pretrain', 'neu_cls')
    if not os.path.isdir(NEU_IMAGE_DIR):
        return {'error': 'NEU IMAGES dir missing: {}'.format(NEU_IMAGE_DIR)}
    # Fresh split for Class-split Phase0 (overwrite previous 90/10 if present)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    by_class = defaultdict(list)
    for fname in sorted(os.listdir(NEU_IMAGE_DIR)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        folder = _neu_class_from_filename(fname)
        if folder not in NEU_PRETRAIN_CLASSES:
            continue
        by_class[folder].append(os.path.join(NEU_IMAGE_DIR, fname))
    counts = defaultdict(int)
    for folder, paths in by_class.items():
        paths = list(paths)
        random.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_ratio)))
        for i, path in enumerate(paths):
            sub = 'val' if i < n_val else 'train'
            out_path = os.path.join(dest, sub, folder, os.path.basename(path))
            save_flat_jpg(Image.open(path).convert('RGB'), out_path)
            counts[(sub, folder)] += 1
    return dict(counts)


def _asde_collect():
    samples = []
    mapping = {'0_intact': 'normal', '1_scratch': 'scratch'}
    for folder, label in mapping.items():
        src = os.path.join(ASDE_ROOT, folder)
        if not os.path.isdir(src):
            continue
        for fname in sorted(os.listdir(src)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append((os.path.join(src, fname), label))
    return samples


def convert_asde_pretrain(out_root, seed=42):
    dest = os.path.join(out_root, 'pretrain', 'asde_binary')
    random.seed(seed)
    counts = defaultdict(int)
    items = _asde_collect()
    random.shuffle(items)
    n_val = max(1, int(len(items) * 0.1))
    for i, (path, label) in enumerate(items):
        sub = 'val' if i < n_val else 'train'
        out_path = os.path.join(dest, sub, label, os.path.basename(path))
        save_flat_jpg(Image.open(path).convert('RGB'), out_path)
        counts[(sub, label)] += 1
    return dict(counts)


def _kolektor_has_defect(gt_path):
    gt = Image.open(gt_path).convert('L')
    return max(gt.getextrema()) > 0


def convert_kolektor_pretrain(out_root, seed=42):
    dest = os.path.join(out_root, 'pretrain', 'kolektor_binary')
    random.seed(seed)
    counts = defaultdict(int)
    for split in ('train', 'test'):
        src_dir = os.path.join(KOLEKTOR_ROOT, split)
        if not os.path.isdir(src_dir):
            continue
        for fname in sorted(os.listdir(src_dir)):
            if not fname.endswith('.png') or fname.endswith('_GT.png'):
                continue
            img_path = os.path.join(src_dir, fname)
            gt_path = os.path.join(src_dir, fname.replace('.png', '_GT.png'))
            label = 'defect' if os.path.isfile(gt_path) and _kolektor_has_defect(gt_path) else 'normal'
            sub = 'train' if split == 'train' else 'val'
            out_path = os.path.join(dest, sub, label, fname)
            save_flat_jpg(Image.open(img_path).convert('RGB'), out_path)
            counts[(sub, label)] += 1
    return dict(counts)


def _is_real_image(path, min_bytes=2048):
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def _spectrum_has_real_images(rel_sub):
    """Defect_Spectrum LFS often leaves empty dirs / pointer stubs."""
    root = os.path.join(SPECTRUM_ROOT, rel_sub)
    if not os.path.isdir(root):
        return False
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if _is_real_image(os.path.join(dirpath, fname)):
                    return True
    return False


def convert_mvtec_ad_binary(out_root, cats, dest_name, crop_policy, seed=42,
                            val_ratio=0.2, max_normal_per_cat=80):
    """MVTec AD categories -> pretrain/{dest_name}/{train,val}/{normal,defect}.

    Defect ROIs from test/*/ + ground_truth masks; normals from train/good (capped).
    """
    random.seed(seed)
    dest = os.path.join(out_root, 'pretrain', dest_name)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    if not os.path.isdir(MVTEC_AD_ROOT):
        return {'error': 'MVTec_AD missing: {}'.format(MVTEC_AD_ROOT)}

    # (label, name, img_path, mask_path|None) — mask_path None => full-image normal
    items = []
    for cat in cats:
        cat_root = os.path.join(MVTEC_AD_ROOT, cat)
        good_dir = os.path.join(cat_root, 'train', 'good')
        test_root = os.path.join(cat_root, 'test')
        gt_root = os.path.join(cat_root, 'ground_truth')
        if not os.path.isdir(test_root):
            continue

        normals = []
        if os.path.isdir(good_dir):
            for fname in sorted(os.listdir(good_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    normals.append(os.path.join(good_dir, fname))
        if max_normal_per_cat and len(normals) > max_normal_per_cat:
            normals = random.sample(normals, max_normal_per_cat)
        for path in normals:
            stem = os.path.splitext(os.path.basename(path))[0]
            items.append(('normal', '{}_good_{}'.format(cat, stem), path, None))

        for defect_type in sorted(os.listdir(test_root)):
            if defect_type == 'good':
                continue
            type_dir = os.path.join(test_root, defect_type)
            if not os.path.isdir(type_dir):
                continue
            for fname in sorted(os.listdir(type_dir)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(type_dir, fname)
                stem = os.path.splitext(fname)[0]
                mask_path = os.path.join(gt_root, defect_type, '{}_mask.png'.format(stem))
                if not os.path.isfile(mask_path):
                    continue
                items.append((
                    'defect', '{}_{}_{}'.format(cat, defect_type, stem),
                    img_path, mask_path,
                ))

    random.shuffle(items)
    by_label = defaultdict(list)
    for item in items:
        by_label[item[0]].append(item)

    counts = defaultdict(int)
    for label, group in by_label.items():
        n_val = max(1, int(round(len(group) * val_ratio)))
        for i, (_lab, name, img_path, mask_path) in enumerate(group):
            sub = 'val' if i < n_val else 'train'
            out_path = os.path.join(dest, sub, label, name + '.jpg')
            img = Image.open(img_path).convert('RGB')
            if mask_path is None:
                save_flat_jpg(img, out_path)
            else:
                crop, _, _ = defect_crop_from_mask(
                    img, Image.open(mask_path), 'MVTec_AD', crop_policy)
                if crop is None:
                    continue
                save_flat_jpg(crop, out_path)
            counts[(sub, label)] += 1
    return dict(counts)


def convert_spectrum_dagm_binary(out_root, crop_policy, seed=42, val_ratio=0.2):
    """DS-DAGM -> pretrain/ds_dagm when LFS images are present (optional third source)."""
    random.seed(seed)
    img_root = os.path.join(SPECTRUM_ROOT, 'DS-DAGM', 'image')
    mask_root = os.path.join(SPECTRUM_ROOT, 'DS-DAGM', 'mask')
    if not _spectrum_has_real_images('DS-DAGM'):
        return {'skipped': 'DS-DAGM has no real images (run: cd data/Defect_Spectrum && git lfs pull)'}

    dest = os.path.join(out_root, 'pretrain', 'ds_dagm')
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    items = []
    for dirpath, _, filenames in os.walk(img_root):
        for fname in sorted(filenames):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(dirpath, fname)
            if not _is_real_image(img_path):
                continue
            rel = os.path.relpath(img_path, img_root)
            mask_path = os.path.join(mask_root, rel)
            stem = os.path.splitext(rel.replace(os.sep, '_'))[0]
            img = Image.open(img_path).convert('RGB')
            if os.path.isfile(mask_path) and _is_real_image(mask_path):
                crop, _, _ = defect_crop_from_mask(
                    img, Image.open(mask_path), 'DS-DAGM', crop_policy)
                if crop is not None:
                    items.append((crop, 'defect', stem))
                    continue
            # no usable mask → treat as normal surface patch
            items.append((img, 'normal', stem))

    if not items:
        return {'error': 'DS-DAGM walk found no usable samples'}
    random.shuffle(items)
    by_label = defaultdict(list)
    for item in items:
        by_label[item[1]].append(item)
    counts = defaultdict(int)
    for label, group in by_label.items():
        n_val = max(1, int(round(len(group) * val_ratio)))
        for i, (src, _lab, name) in enumerate(group):
            sub = 'val' if i < n_val else 'train'
            save_flat_jpg(src if isinstance(src, Image.Image) else Image.open(src).convert('RGB'),
                          os.path.join(dest, sub, label, name + '.jpg'))
            counts[(sub, label)] += 1
    return dict(counts)


def check_layout(out_root):
    lines = []
    for rel in (
        'pretrain/cardd_cls/train/scratch',
        'pretrain/neu_cls/train/crazing',
        'pretrain/asde_binary/train/normal',
        'pretrain/mvtec_tex/train/normal',
        'pretrain/mvtec_tex/train/defect',
        'pretrain/mvtec_obj/train/normal',
        'pretrain/mvtec_obj/train/defect',
        'cmpl/domain_0/0_scratch',
        'cmpl/domain_3/0_scratch',
    ):
        path = os.path.join(out_root, rel)
        n = 0
        if os.path.isdir(path):
            n = len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) or f.endswith('.jpg')])
        lines.append('  {} : {}'.format(rel, n if n else ('OK' if os.path.isdir(path) else 'MISSING')))
    return lines


def main():
    parser = argparse.ArgumentParser(description='Convert automotive raw data to CMPL layout')
    parser.add_argument('--raw_root', default=AUTOMOTIVE_RAW)
    parser.add_argument('--out_root', default=AUTOMOTIVE_PROCESSED)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_neu_only', action='store_true',
                        help='Class-split Phase0a: only convert NEU-DET (no CarDD/ASDE)')
    parser.add_argument('--pretrain_classsplit', action='store_true',
                        help='Class-split Phase0b: NEU + MVTec texture + (DS-DAGM|MVTec object)')
    parser.add_argument('--neu_val_ratio', type=float, default=0.2,
                        help='NEU stratified val ratio (Class-split default 0.2; legacy used 0.1)')
    parser.add_argument('--meta', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expand_ratio', type=float, default=1.8)
    parser.add_argument('--meta_policy', choices=('track2', 'legacy'), default='track2',
                        help='track2: ASDE domain_0 + CarDD test domain_3; '
                             'legacy: CarDD train/val/test -> domain 0/2/3')
    parser.add_argument('--legacy_kolektor', action='store_true',
                        help='Also convert Kolektor binary pretrain (deprecated, imbalanced)')
    args = parser.parse_args()

    global CARDD_ROOT, CARDD_COCO, ASDE_ROOT, NEU_ROOT, NEU_IMAGE_DIR, KOLEKTOR_ROOT, MVTEC_AD_ROOT
    CARDD_ROOT = os.path.join(args.raw_root, 'CarDD', 'CarDD_release')
    CARDD_COCO = os.path.join(CARDD_ROOT, 'CarDD_COCO', 'annotations')
    ASDE_ROOT = os.path.join(args.raw_root, 'ASDE', 'image')
    NEU_ROOT = os.path.join(args.raw_root, 'NEU-DET', 'NEU-DET')
    NEU_IMAGE_DIR = os.path.join(NEU_ROOT, 'IMAGES')
    KOLEKTOR_ROOT = os.path.join(args.raw_root, 'KolektorSDD2')
    MVTEC_AD_ROOT = os.path.join(args.raw_root, 'MVTec_AD')

    crop_policy = merge_crop_policy({'expand_ratio': args.expand_ratio})
    do_pretrain = args.pretrain or args.all
    do_meta = args.meta or args.all

    if args.pretrain_classsplit:
        print('=== Class-split Phase0b: NEU + MVTec_tex + third (DAGM|MVTec_obj) ===')
        print('NEU 6-cls:', convert_neu_pretrain(
            args.out_root, args.seed, val_ratio=args.neu_val_ratio))
        print('MVTec texture binary:', convert_mvtec_ad_binary(
            args.out_root, MVTEC_TEXTURE_CATS, 'mvtec_tex', crop_policy,
            seed=args.seed, val_ratio=args.neu_val_ratio))
        dagm = convert_spectrum_dagm_binary(
            args.out_root, crop_policy, seed=args.seed, val_ratio=args.neu_val_ratio)
        print('DS-DAGM binary:', dagm)
        if 'skipped' in dagm or 'error' in dagm:
            print('MVTec object binary (DAGM stand-in):', convert_mvtec_ad_binary(
                args.out_root, MVTEC_OBJECT_CATS, 'mvtec_obj', crop_policy,
                seed=args.seed, val_ratio=args.neu_val_ratio))
        print('Note: do not stack raw MVTec_tex with DS-MVTec once Spectrum LFS is pulled.')
    elif args.pretrain_neu_only:
        print('=== Class-split Phase0 prep: NEU-DET only (val_ratio={}) ==='.format(
            args.neu_val_ratio))
        print('NEU 6-cls:', convert_neu_pretrain(
            args.out_root, args.seed, val_ratio=args.neu_val_ratio))
    elif do_pretrain:
        print('=== Phase 0 pretrain (CarDD 80% + NEU 15% + ASDE 5%) ===')
        print('CarDD cls:', convert_cardd_pretrain(args.out_root, crop_policy, args.seed))
        print('NEU 6-cls:', convert_neu_pretrain(
            args.out_root, args.seed, val_ratio=args.neu_val_ratio))
        print('ASDE binary:', convert_asde_pretrain(args.out_root, args.seed))
        if args.legacy_kolektor:
            print('Kolektor binary (legacy):', convert_kolektor_pretrain(args.out_root, args.seed))

    if do_meta:
        if args.meta_policy == 'legacy':
            print('=== CMPL meta LEGACY (CarDD train/val/test -> domain 0/2/3) ===')
            print('CarDD meta:', convert_cardd_meta(args.out_root, crop_policy, splits=('train', 'val', 'test')))
        else:
            print('=== CMPL meta Track2 (ASDE domain_0 + CarDD test domain_3) ===')
            print('ASDE meta:', convert_asde_meta(args.out_root))
            print('CarDD meta (test holdout):', convert_cardd_meta(args.out_root, crop_policy, splits=('test',)))

    if args.check or args.all:
        print('=== Check ===')
        for line in check_layout(args.out_root):
            print(line)


if __name__ == '__main__':
    main()
