"""
Unified dataset conversion (real YOLO / AeBAD / bladeSynth).

Tasks (CLI flags):
  --real            YOLO  -> data/aviation/domain_0~3   (MAML)
  --aebad           AeBAD -> data/aviation/domain_4~7   (MAML)
  --bladesynth      bladeSynth -> data/pretrain/bladesynth/ (Phase 0)
  --aebad_pretrain  AeBAD -> data/pretrain/aebad/       (Phase 0)

Shortcuts: --maml  --pretrain  --all  --check

Parallel: multiple tasks run in separate processes by default (--no_parallel to disable).

Usage:
  cd D:\\元强化学习\\代码
  python scripts/convert_datasets.py --all --check
  python scripts/convert_datasets.py --all --no_parallel
  python scripts/convert_datasets.py --all --jobs 2 --progress_interval 100
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.aviation_taxonomy import (
    AEBAD_DEFECT_TO_CLASS,
    AEBAD_DEFECT_TO_DOMAIN,
    BLADESYNTH_CLASSES,
    CLASS_FOLDERS,
    DEFAULT_CROP_POLICY,
    NORMAL_FOLDER,
    UNIFIED_CLASS,
)
from common.crop_utils import (
    defect_crop_from_box,
    defect_crop_from_mask,
    save_aviation_sample,
    save_flat_jpg,
    save_normal_sample,
    yolo_box_to_pixels,
)
from common.knowledge_base import DEFAULT_FUSION_PRIOR, DEFAULT_MATERIAL_PARAMS
from common.domain_split_v2 import (
    build_domain_map_v2,
    count_samples_by_base_from_raw,
    domain_material_summary,
    load_specimen_registry,
    validate_domain_split,
)

_DEFAULT_DATA = os.path.join(_ROOT, 'data')
DEFAULT_TEST_BASES = {'20', '21', '22', '23', '24', '25', '26'}
_BLADESYNTH_ID_RE = re.compile(r'_(\d+)$')


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_image_filename(fname):
    """Skip macOS AppleDouble metadata (._IMG_xxx.png) and non-images."""
    if fname.startswith('._'):
        return False
    return fname.lower().endswith(('.png', '.jpg', '.jpeg'))


def build_crop_policy(args):
    return {
        'expand_ratio': args.expand_ratio,
        'min_defect_ratio': args.min_defect_ratio,
        'min_short_side': args.min_short_side,
        'max_short_side': args.max_short_side,
    }


def add_crop_args(parser):
    g = parser.add_argument_group('ROI crop (shared)')
    g.add_argument('--no_crop', action='store_true',
                   help='Real only: copy full image without ROI crop')
    g.add_argument('--expand_ratio', type=float, default=DEFAULT_CROP_POLICY['expand_ratio'])
    g.add_argument('--min_defect_ratio', type=float,
                   default=DEFAULT_CROP_POLICY['min_defect_ratio'])
    g.add_argument('--min_short_side', type=int, default=DEFAULT_CROP_POLICY['min_short_side'])
    g.add_argument('--max_short_side', type=int, default=DEFAULT_CROP_POLICY['max_short_side'])


class ProgressTracker(object):
    """Console progress: current/total, percent, bar."""

    def __init__(self, tag, total, interval=1, inline=True):
        self.tag = tag
        self.total = max(int(total), 1)
        self.interval = max(1, int(interval))
        self.inline = inline
        self._last_shown = 0

    def update(self, current, label=''):
        current = min(int(current), self.total)
        if current < self.total:
            if current <= self._last_shown:
                return
            if (current - self._last_shown) < self.interval and current != 1:
                return
        self._last_shown = current
        pct = 100.0 * current / self.total
        width = 30
        filled = int(width * current / self.total)
        if filled >= width:
            bar = '=' * width
        else:
            bar = '=' * filled + '>' + ' ' * (width - filled - 1)
        msg = '[{}] [{}] {}/{} ({:.1f}%) {}'.format(
            self.tag, bar, current, self.total, pct, label)
        if self.inline:
            sys.stdout.write('\r' + msg + ' ' * 8)
            sys.stdout.flush()
        else:
            print(msg, flush=True)
        if current >= self.total:
            self.finish()

    def finish(self):
        if self.inline:
            sys.stdout.write('\n')
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# [1] Real YOLO -> aviation/domain_0~3
# ---------------------------------------------------------------------------

def _read_class_map(label_dir):
    path = os.path.join(label_dir, 'classes.txt')
    names = [line.strip() for line in open(path, encoding='utf-8') if line.strip()]
    return {i: name for i, name in enumerate(names)}


def _primary_detection(label_path, local_class_map):
    lines = [line.strip() for line in open(label_path, encoding='utf-8') if line.strip()]
    best = None
    best_area = -1.0
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy = float(parts[1]), float(parts[2])
        w, h = float(parts[3]), float(parts[4])
        area = w * h
        name = local_class_map.get(cid)
        if name not in UNIFIED_CLASS:
            continue
        if area > best_area:
            best_area = area
            best = (UNIFIED_CLASS[name], cx, cy, w, h)
    return best


def _base_id(stem):
    m = re.match(r'^(\d+)', stem)
    return m.group(1) if m else stem


def _base_sort_key(base_id):
    try:
        return (0, int(base_id))
    except ValueError:
        return (1, str(base_id))


def build_domain_map(all_bases, test_bases=None):
    if test_bases is None:
        test_bases = set(DEFAULT_TEST_BASES)
    bases = sorted(set(all_bases), key=_base_sort_key)
    missing_test = test_bases - set(bases)
    if missing_test:
        print('Warning: requested test bases not in dataset:', sorted(missing_test))
    domain_map = {}
    train_bases = [b for b in bases if b not in test_bases]
    for idx, base_id in enumerate(train_bases):
        domain_map[base_id] = idx % 3
    for base_id in test_bases:
        if base_id in bases:
            domain_map[base_id] = 3
    return domain_map, train_bases, sorted(test_bases & set(bases), key=_base_sort_key)


def _collect_real_samples(raw_root):
    samples = []
    skipped = 0
    for split in ('train', 'val'):
        image_dir = os.path.join(raw_root, 'images', split)
        label_dir = os.path.join(raw_root, 'labels', split)
        if not os.path.isdir(image_dir):
            continue
        local_map = _read_class_map(label_dir)
        for img_name in sorted(os.listdir(image_dir)):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            stem = os.path.splitext(img_name)[0]
            label_path = os.path.join(label_dir, stem + '.txt')
            if not os.path.isfile(label_path):
                skipped += 1
                continue
            det = _primary_detection(label_path, local_map)
            if det is None:
                skipped += 1
                continue
            class_id, cx, cy, w, h = det
            samples.append({
                'split': split,
                'stem': stem,
                'base_id': _base_id(stem),
                'class_id': class_id,
                'box': (cx, cy, w, h),
                'src_image': os.path.join(image_dir, img_name),
            })
    return samples, skipped


def _clear_real_domains(out_root, domain_ids=None):
    """Remove only real-YOLO domain folders (default 0-3, or explicit id set)."""
    aviation_root = os.path.join(out_root, 'aviation')
    if not os.path.isdir(aviation_root):
        return
    if domain_ids is None:
        domain_ids = range(4)
    for domain_id in set(domain_ids):
        domain_dir = os.path.join(aviation_root, 'domain_{}'.format(domain_id))
        if os.path.isdir(domain_dir):
            shutil.rmtree(domain_dir)


def _write_real_domain_meta(out_root, domain_id, class_counts, split_info, crop_policy,
                            material_id='AL7075'):
    hist = [0.0] * 4
    total = max(sum(class_counts.values()), 1)
    for k, v in class_counts.items():
        hist[k] = v / total
    meta = {
        'material_id': material_id,
        'material_params': DEFAULT_MATERIAL_PARAMS,
        'process_params': [0.0] * 8,
        'defect_history': hist + [0.0] * 6,
        'fusion_prior': list(DEFAULT_FUSION_PRIOR),
        'source': 'Aero-engine_defect-detect_new',
        'split_policy': split_info,
        'crop_policy': crop_policy,
    }
    domain_dir = os.path.join(out_root, 'aviation', 'domain_{}'.format(domain_id))
    os.makedirs(domain_dir, exist_ok=True)
    with open(os.path.join(domain_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _resolve_domain_mapping(all_bases, test_bases, domain_split_cfg):
    """Return domain_map, train_bases, test_base_list, split_info."""
    cfg = domain_split_cfg or {}
    strategy = cfg.get('strategy', 'legacy_mod3')

    if strategy == 'csv' and cfg.get('domain_map_csv'):
        registry = load_specimen_registry(cfg['domain_map_csv'])
        return build_domain_map_v2(
            all_bases,
            registry=registry,
            test_bases=test_bases,
            test_domain=cfg.get('test_domain', 3),
            group_by=cfg.get('group_by', 'train_domain'),
        )

    if strategy == 'auto':
        base_counts = cfg.get('base_counts')
        if not base_counts:
            raise ValueError('auto domain split requires base_counts in config')
        from common.domain_split_v2 import search_auto_partitions
        suggestions = search_auto_partitions(
            base_counts,
            [b for b in all_bases if b not in test_bases],
            k_min=cfg.get('k_min', 3),
            k_max=cfg.get('k_max', 5),
            ways=cfg.get('ways', 2),
            shots=cfg.get('shots', 5),
        )
        if not suggestions or not suggestions[0]['feasible']:
            print('[real] Warning: auto split found no feasible partition; '
                  'falling back to legacy %3.')
            domain_map, train_bases, test_base_list = build_domain_map(all_bases, test_bases)
            split_info = {
                'strategy': 'specimen_holdout',
                'train_bases': train_bases,
                'test_bases': test_base_list,
                'merged_splits': ['train', 'val'],
                'fallback': 'legacy_mod3',
            }
            return domain_map, train_bases, test_base_list, split_info
        best = suggestions[0]
        domain_map = best['domain_map']
        train_bases = [b for b in all_bases if b not in test_bases and b in domain_map]
        test_base_list = sorted(test_bases & set(all_bases), key=_base_sort_key)
        split_info = {
            'strategy': 'auto_greedy_v2',
            'k_domains': best['k_domains'],
            'groups': best['groups'],
            'domain_map': domain_map,
            'validation': best['validation'],
            'train_bases': train_bases,
            'test_bases': test_base_list,
            'merged_splits': ['train', 'val'],
        }
        return domain_map, train_bases, test_base_list, split_info

    domain_map, train_bases, test_base_list = build_domain_map(all_bases, test_bases)
    split_info = {
        'strategy': 'specimen_holdout',
        'train_bases': train_bases,
        'test_bases': test_base_list,
        'merged_splits': ['train', 'val'],
    }
    return domain_map, train_bases, test_base_list, split_info


def convert_real(raw_root, out_root, test_bases=None, clean=True, crop_policy=None,
                 use_crop=True, progress_interval=10, inline_progress=True,
                 domain_split_cfg=None):
    crop_policy = dict(DEFAULT_CROP_POLICY if crop_policy is None else crop_policy)
    samples, skipped = _collect_real_samples(raw_root)
    if not samples:
        raise RuntimeError('No valid samples found under {}'.format(raw_root))

    all_bases = {s['base_id'] for s in samples}
    if test_bases is None:
        test_bases = set(DEFAULT_TEST_BASES)

    cfg = dict(domain_split_cfg or {})
    if cfg.get('strategy') == 'auto' and 'base_counts' not in cfg:
        cfg['base_counts'] = count_samples_by_base_from_raw(samples)

    domain_map, train_bases, test_base_list, split_info = _resolve_domain_mapping(
        all_bases, test_bases, cfg)

    if clean:
        _clear_real_domains(out_root, domain_ids=set(domain_map.values()))

    registry = {}
    if cfg.get('domain_map_csv'):
        try:
            registry = load_specimen_registry(cfg['domain_map_csv'])
        except FileNotFoundError:
            registry = {}

    stats = defaultdict(lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    converted = 0
    progress = ProgressTracker('real', len(samples), progress_interval, inline_progress)
    for idx, sample in enumerate(samples, start=1):
        domain_id = domain_map.get(sample['base_id'])
        if domain_id is None:
            skipped += 1
            progress.update(idx, sample['stem'])
            continue
        class_id = sample['class_id']
        sample_id = sample['stem'].replace(os.sep, '_')
        dst_dir = os.path.join(
            out_root, 'aviation', 'domain_{}'.format(domain_id),
            CLASS_FOLDERS[class_id], sample_id,
        )
        image = Image.open(sample['src_image']).convert('RGB')
        img_w, img_h = image.size
        row = registry.get(sample['base_id'], {})
        source_meta = {
            'dataset': 'Aero-engine_defect-detect_new',
            'original_path': os.path.relpath(sample['src_image'], raw_root),
            'specimen_id': sample['base_id'],
            'split': sample['split'],
        }
        if row.get('material_id'):
            source_meta['material_id'] = row['material_id']
        if row.get('process_batch'):
            source_meta['process_batch'] = row['process_batch']
        if row.get('camera_batch'):
            source_meta['camera_batch'] = row['camera_batch']
        if use_crop:
            cx, cy, w, h = sample['box']
            defect_box = yolo_box_to_pixels(cx, cy, w, h, img_w, img_h)
            crop_img, full_img, roi_meta = defect_crop_from_box(
                image, defect_box, source_dataset='real_yolo',
                policy=crop_policy, extra={'yolo_box': [cx, cy, w, h]},
            )
            save_aviation_sample(dst_dir, crop_img, source_meta, roi_meta, full_img)
        else:
            save_aviation_sample(dst_dir, image, source_meta, None, None, save_full=False)

        stats[domain_id][class_id] += 1
        converted += 1
        progress.update(idx, sample['stem'])

    split_info.setdefault('merged_splits', ['train', 'val'])
    for domain_id in sorted(stats.keys()):
        material_id = domain_material_summary(registry, domain_map, domain_id)
        _write_real_domain_meta(
            out_root, domain_id, stats[domain_id], split_info, crop_policy,
            material_id=material_id)

    if cfg.get('validate_2w5s', True):
        base_counts = count_samples_by_base_from_raw(samples)
        validation = validate_domain_split(
            base_counts, domain_map,
            ways=cfg.get('ways', 2), shots=cfg.get('shots', 5))
        print('[real] 2w{}s domain validation: ok={} task_ub_hint={}'.format(
            cfg.get('shots', 5), validation['ok'], validation['task_upper_bound_hint']))
        for did, info in sorted(validation['domains'].items(), key=lambda x: int(x[0])):
            print('[real]   domain_{}: scratch={} damage={} feasible={}'.format(
                did, info['scratch'], info['damage'],
                info.get('feasible_2w{}s'.format(cfg.get('shots', 5)))))

    print('[real] converted:', converted, 'skipped:', skipped, 'use_crop:', use_crop)
    print('[real] split strategy:', split_info.get('strategy'))
    print('[real] train bases:', train_bases)
    print('[real] test bases :', test_base_list)
    print('[real] class counts:', dict(stats))
    return converted


# ---------------------------------------------------------------------------
# [2] AeBAD -> aviation/domain_4~7
# ---------------------------------------------------------------------------

def _aeb_ad_variant(rel_path):
    parts = rel_path.replace('\\', '/').split('/')
    return parts[0] if parts else 'unknown'


def _collect_aebad_defect_samples(aebad_root):
    samples = []
    skipped = 0
    test_root = os.path.join(aebad_root, 'AeBAD_S', 'test')
    gt_root = os.path.join(aebad_root, 'AeBAD_S', 'ground_truth')
    for defect_type in AEBAD_DEFECT_TO_DOMAIN:
        defect_test_dir = os.path.join(test_root, defect_type)
        defect_gt_dir = os.path.join(gt_root, defect_type)
        if not os.path.isdir(defect_test_dir):
            continue
        for dirpath, _, filenames in os.walk(defect_test_dir):
            for fname in filenames:
                if not _is_image_filename(fname):
                    continue
                img_path = os.path.join(dirpath, fname)
                rel = os.path.relpath(img_path, defect_test_dir)
                mask_path = os.path.join(defect_gt_dir, rel)
                if not os.path.isfile(mask_path):
                    mask_path = os.path.join(defect_gt_dir, os.path.splitext(rel)[0] + '.png')
                if not os.path.isfile(mask_path):
                    skipped += 1
                    continue
                variant = _aeb_ad_variant(rel)
                stem = os.path.splitext(rel)[0].replace(os.sep, '_').replace('/', '_')
                samples.append({
                    'defect_type': defect_type,
                    'domain_id': AEBAD_DEFECT_TO_DOMAIN[defect_type],
                    'class_id': AEBAD_DEFECT_TO_CLASS[defect_type],
                    'sample_id': '{}_{}'.format(stem, variant),
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'variant': variant,
                })
    return samples, skipped


def _collect_aebad_normal_paths(aebad_root):
    """Collect normal images under train/good/ (flat or variant subdirs)."""
    good_dir = os.path.join(aebad_root, 'AeBAD_S', 'train', 'good')
    if not os.path.isdir(good_dir):
        return []
    paths = []
    for dirpath, _, filenames in os.walk(good_dir):
        for fname in filenames:
            if _is_image_filename(fname):
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def _aebad_normal_stem(src, good_dir):
    rel = os.path.relpath(src, good_dir)
    return os.path.splitext(rel)[0].replace(os.sep, '_').replace('/', '_')


def _aviation_sample_ready(sample_dir):
    return os.path.isfile(os.path.join(sample_dir, 'rgb.jpg'))


def _scan_aebad_maml_stats(out_root):
    """Rescan domain_4~7 folders for accurate meta/stats after incremental runs."""
    folder_to_class = {v: k for k, v in CLASS_FOLDERS.items()}
    stats = {d: {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0} for d in range(4, 8)}
    for domain_id in range(4, 8):
        domain_dir = os.path.join(out_root, 'aviation', 'domain_{}'.format(domain_id))
        if not os.path.isdir(domain_dir):
            continue
        for class_name in os.listdir(domain_dir):
            class_dir = os.path.join(domain_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            count = sum(
                1 for sample_id in os.listdir(class_dir)
                if _aviation_sample_ready(os.path.join(class_dir, sample_id))
            )
            if class_name == NORMAL_FOLDER:
                stats[domain_id][-1] = count
            elif class_name in folder_to_class:
                stats[domain_id][folder_to_class[class_name]] = count
    return stats


def _clear_aebad_domains(out_root):
    for domain_id in range(4, 8):
        domain_dir = os.path.join(out_root, 'aviation', 'domain_{}'.format(domain_id))
        if os.path.isdir(domain_dir):
            shutil.rmtree(domain_dir)


def _write_aebad_domain_meta(out_root, domain_id, class_counts, crop_policy, defect_type):
    hist = [0.0] * 4
    total = max(sum(class_counts.values()), 1)
    for k, v in class_counts.items():
        hist[k] = v / total
    meta = {
        'material_id': 'TBC',
        'material_params': DEFAULT_MATERIAL_PARAMS,
        'process_params': [0.0] * 8,
        'defect_history': hist + [0.0] * 6,
        'fusion_prior': list(DEFAULT_FUSION_PRIOR),
        'source': 'AeBAD_S',
        'aeb_ad_defect_type': defect_type,
        'split_policy': {
            'strategy': 'aeb_ad_defect_domain',
            'defect_type': defect_type,
            'normal_source': 'train/good',
        },
        'crop_policy': crop_policy,
    }
    domain_dir = os.path.join(out_root, 'aviation', 'domain_{}'.format(domain_id))
    os.makedirs(domain_dir, exist_ok=True)
    with open(os.path.join(domain_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def convert_aebad_maml(aebad_root, out_root, clean=True, crop_policy=None,
                       normal_per_domain=50, seed=42,
                       progress_interval=20, inline_progress=True):
    crop_policy = dict(DEFAULT_CROP_POLICY if crop_policy is None else crop_policy)
    samples, skipped = _collect_aebad_defect_samples(aebad_root)
    if not samples:
        raise RuntimeError('No AeBAD defect samples under {}'.format(aebad_root))
    if clean:
        _clear_aebad_domains(out_root)

    rng = random.Random(seed)
    normal_paths = _collect_aebad_normal_paths(aebad_root)
    if len(normal_paths) < normal_per_domain:
        print('[aebad] warning: only {} normal images (want {} per domain)'.format(
            len(normal_paths), normal_per_domain))

    stats = {d: {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0} for d in range(4, 8)}
    converted = skipped = skipped_existing = 0
    good_dir = os.path.join(aebad_root, 'AeBAD_S', 'train', 'good')
    normal_total = 4 * min(normal_per_domain, len(normal_paths)) if normal_paths else 0
    total_steps = normal_total + len(samples)
    progress = ProgressTracker('aebad', total_steps, progress_interval, inline_progress)
    step = 0

    for domain_id in range(4, 8):
        if not normal_paths:
            continue
        picks = rng.sample(normal_paths, min(normal_per_domain, len(normal_paths)))
        for src in picks:
            sample_id = _aebad_normal_stem(src, good_dir)
            dst_dir = os.path.join(
                out_root, 'aviation', 'domain_{}'.format(domain_id),
                NORMAL_FOLDER, sample_id,
            )
            step += 1
            if _aviation_sample_ready(dst_dir):
                stats[domain_id][-1] += 1
                skipped_existing += 1
                progress.update(step, 'normal d{} {}'.format(domain_id, sample_id))
                continue
            save_normal_sample(dst_dir, Image.open(src).convert('RGB'), {
                'dataset': 'AeBAD_S',
                'original_path': os.path.relpath(src, aebad_root),
                'role': 'normal_negative',
            })
            stats[domain_id][-1] += 1
            converted += 1
            progress.update(step, 'normal d{} {}'.format(domain_id, sample_id))

    for sample in samples:
        domain_id = sample['domain_id']
        class_id = sample['class_id']
        dst_dir = os.path.join(
            out_root, 'aviation', 'domain_{}'.format(domain_id),
            CLASS_FOLDERS[class_id], sample['sample_id'],
        )
        step += 1
        if _aviation_sample_ready(dst_dir):
            stats[domain_id][class_id] += 1
            skipped_existing += 1
            progress.update(step, sample['sample_id'])
            continue
        image = Image.open(sample['image_path']).convert('RGB')
        mask = Image.open(sample['mask_path'])
        crop_img, full_img, roi_meta = defect_crop_from_mask(
            image, mask, source_dataset='aeb_ad', policy=crop_policy,
            extra={'aeb_ad_defect_type': sample['defect_type'],
                   'aeb_ad_variant': sample['variant']},
        )
        if crop_img is None:
            skipped += 1
            progress.update(step, sample['sample_id'])
            continue
        save_aviation_sample(dst_dir, crop_img, {
            'dataset': 'AeBAD_S',
            'original_path': os.path.relpath(sample['image_path'], aebad_root),
            'mask_path': os.path.relpath(sample['mask_path'], aebad_root),
            'aeb_ad_defect_type': sample['defect_type'],
            'aeb_ad_variant': sample['variant'],
        }, roi_meta, full_img)
        stats[domain_id][class_id] += 1
        converted += 1
        progress.update(step, sample['sample_id'])

    stats = _scan_aebad_maml_stats(out_root)
    defect_by_domain = {v: k for k, v in AEBAD_DEFECT_TO_DOMAIN.items()}
    for domain_id in range(4, 8):
        _write_aebad_domain_meta(
            out_root, domain_id,
            {k: stats[domain_id][k] for k in range(4)},
            crop_policy, defect_by_domain[domain_id],
        )

    print('[aebad] converted:', converted, 'skipped:', skipped,
          'already_present:', skipped_existing)
    print('[aebad] stats:', stats)
    return converted


# ---------------------------------------------------------------------------
# [3] bladeSynth -> pretrain/bladesynth/
# ---------------------------------------------------------------------------

def _bladesynth_split(stem):
    m = _BLADESYNTH_ID_RE.search(stem)
    sid = int(m.group(1)) if m else 0
    return 'val' if sid % 10 == 0 else 'train'


def _collect_bladesynth_entries(raw_root):
    images_root = os.path.join(raw_root, 'Images')
    entries = []
    for class_name in BLADESYNTH_CLASSES:
        class_dir = os.path.join(images_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if _is_image_filename(fname):
                entries.append((class_name, fname))
    return entries


def convert_bladesynth(raw_root, out_root, clean=True, crop_policy=None,
                       progress_interval=100, inline_progress=True):
    crop_policy = dict(DEFAULT_CROP_POLICY if crop_policy is None else crop_policy)
    images_root = os.path.join(raw_root, 'Images')
    masks_root = os.path.join(raw_root, 'Segmentation_masks')
    if not os.path.isdir(images_root):
        raise FileNotFoundError('Missing Images/ under {}'.format(raw_root))

    target = os.path.join(out_root, 'pretrain', 'bladesynth')
    if clean and os.path.isdir(target):
        shutil.rmtree(target)

    entries = _collect_bladesynth_entries(raw_root)
    stats = {'train': {}, 'val': {}}
    converted = skipped = 0
    progress = ProgressTracker('bladesynth', len(entries), progress_interval, inline_progress)

    for idx, (class_name, fname) in enumerate(entries, start=1):
        stem = os.path.splitext(fname)[0]
        split = _bladesynth_split(stem)
        image = Image.open(os.path.join(images_root, class_name, fname)).convert('RGB')
        out_path = os.path.join(target, split, class_name, '{}.jpg'.format(stem))

        if class_name == 'Normal':
            save_flat_jpg(image, out_path)
        else:
            mask_path = os.path.join(masks_root, class_name, fname)
            if not os.path.isfile(mask_path):
                mask_path = os.path.join(
                    masks_root, class_name, os.path.splitext(fname)[0] + '.png')
            if not os.path.isfile(mask_path):
                skipped += 1
                progress.update(idx, fname)
                continue
            crop_img, _, _ = defect_crop_from_mask(
                image, Image.open(mask_path), source_dataset='bladesynth',
                policy=crop_policy, extra={'class': class_name},
            )
            if crop_img is None:
                skipped += 1
                progress.update(idx, fname)
                continue
            save_flat_jpg(crop_img, out_path)

        stats[split][class_name] = stats[split].get(class_name, 0) + 1
        converted += 1
        progress.update(idx, fname)

    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'source': 'Dataset_bladesynth',
            'split_rule': 'numeric_id % 10 == 0 -> val',
            'crop_policy': crop_policy,
            'stats': stats,
        }, f, indent=2, ensure_ascii=False)

    print('[bladesynth] converted:', converted, 'skipped:', skipped)
    print('[bladesynth] stats:', stats)
    return converted


# ---------------------------------------------------------------------------
# [4] AeBAD -> pretrain/aebad/
# ---------------------------------------------------------------------------

def _hash_split(path, val_mod=10):
    return (hash(path) % val_mod) == 0


def _collect_aebad_pretrain_defect_jobs(aebad_root):
    jobs = []
    test_root = os.path.join(aebad_root, 'AeBAD_S', 'test')
    gt_root = os.path.join(aebad_root, 'AeBAD_S', 'ground_truth')
    for defect_type in AEBAD_DEFECT_TO_DOMAIN.keys():
        defect_test_dir = os.path.join(test_root, defect_type)
        defect_gt_dir = os.path.join(gt_root, defect_type)
        if not os.path.isdir(defect_test_dir):
            continue
        for dirpath, _, filenames in os.walk(defect_test_dir):
            for fname in filenames:
                if _is_image_filename(fname):
                    jobs.append((defect_type, os.path.join(dirpath, fname), defect_test_dir, defect_gt_dir))
    return jobs


def convert_aebad_pretrain(aebad_root, out_root, clean=True, crop_policy=None, val_mod=10,
                            progress_interval=20, inline_progress=True):
    crop_policy = dict(DEFAULT_CROP_POLICY if crop_policy is None else crop_policy)
    good_dir = os.path.join(aebad_root, 'AeBAD_S', 'train', 'good')
    if not os.path.isdir(good_dir):
        raise FileNotFoundError('Missing AeBAD_S/train/good under {}'.format(aebad_root))

    target = os.path.join(out_root, 'pretrain', 'aebad')
    if clean and os.path.isdir(target):
        shutil.rmtree(target)

    good_paths = _collect_aebad_normal_paths(aebad_root)
    defect_jobs = _collect_aebad_pretrain_defect_jobs(aebad_root)
    total_steps = len(good_paths) + len(defect_jobs)
    progress = ProgressTracker('aebad_pretrain', total_steps, progress_interval, inline_progress)

    stats = {
        'normal_train': 0,
        'binary_train_normal': 0, 'binary_train_defect': 0,
        'binary_val_normal': 0, 'binary_val_defect': 0,
    }
    step = skipped = skipped_existing = 0

    for src in good_paths:
        stem = _aebad_normal_stem(src, good_dir)
        normal_out = os.path.join(target, 'normal', 'train', '{}.jpg'.format(stem))
        step += 1
        if os.path.isfile(normal_out):
            stats['normal_train'] += 1
            skipped_existing += 1
        else:
            save_flat_jpg(Image.open(src).convert('RGB'), normal_out)
            stats['normal_train'] += 1
        split = 'val' if _hash_split(src, val_mod) else 'train'
        binary_out = os.path.join(target, 'binary', split, 'normal', '{}.jpg'.format(stem))
        if os.path.isfile(binary_out):
            stats['binary_{}_normal'.format(split)] += 1
            skipped_existing += 1
        else:
            save_flat_jpg(Image.open(src).convert('RGB'), binary_out)
            stats['binary_{}_normal'.format(split)] += 1
        progress.update(step, stem)

    for defect_type, img_path, defect_test_dir, defect_gt_dir in defect_jobs:
        rel = os.path.relpath(img_path, defect_test_dir)
        stem = os.path.splitext(rel)[0].replace(os.sep, '_').replace('/', '_')
        split = 'val' if _hash_split(img_path, val_mod) else 'train'
        out_path = os.path.join(
            target, 'binary', split, 'defect',
            '{}_{}.jpg'.format(defect_type, stem),
        )
        step += 1
        if os.path.isfile(out_path):
            stats['binary_{}_defect'.format(split)] += 1
            skipped_existing += 1
            progress.update(step, rel)
            continue
        mask_path = os.path.join(defect_gt_dir, rel)
        if not os.path.isfile(mask_path):
            mask_path = os.path.join(defect_gt_dir, os.path.splitext(rel)[0] + '.png')
        if not os.path.isfile(mask_path):
            skipped += 1
            progress.update(step, rel)
            continue
        crop_img, _, _ = defect_crop_from_mask(
            Image.open(img_path).convert('RGB'), Image.open(mask_path),
            source_dataset='aeb_ad_pretrain', policy=crop_policy,
            extra={'defect_type': defect_type},
        )
        if crop_img is None:
            skipped += 1
            progress.update(step, rel)
            continue
        save_flat_jpg(crop_img, out_path)
        stats['binary_{}_defect'.format(split)] += 1
        progress.update(step, rel)

    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'source': 'AeBAD_S',
            'excluded': ['test/good'],
            'val_split': 'hash(path) % {} == 0'.format(val_mod),
            'crop_policy': crop_policy,
            'stats': stats,
            'skipped_defect': skipped,
            'already_present': skipped_existing,
        }, f, indent=2, ensure_ascii=False)

    print('[aebad_pretrain] stats:', stats, 'skipped:', skipped,
          'already_present:', skipped_existing)
    return stats


# ---------------------------------------------------------------------------
# CLI + parallel dispatch
# ---------------------------------------------------------------------------

TASK_ORDER = ('real', 'aebad', 'bladesynth', 'aebad_pretrain')


def _task_config(args, inline_progress=True):
    test_bases = {b.strip() for b in args.test_bases.split(',') if b.strip()}
    crop_policy = build_crop_policy(args)
    domain_split_cfg = {
        'strategy': args.domain_split,
        'domain_map_csv': args.domain_map_csv or None,
        'group_by': args.domain_group_by,
        'k_min': args.domain_k_min,
        'k_max': args.domain_k_max,
        'ways': args.domain_ways,
        'shots': args.domain_shots,
        'test_domain': 3,
        'validate_2w5s': True,
    }
    return {
        'out_root': os.path.abspath(args.out_root),
        'real_root': os.path.abspath(args.real_root),
        'aebad_root': os.path.abspath(args.aebad_root),
        'bladesynth_root': os.path.abspath(args.bladesynth_root),
        'crop_policy': crop_policy,
        'clean': not args.no_clean,
        'use_crop': not args.no_crop,
        'test_bases': test_bases,
        'normal_per_domain': args.normal_per_domain,
        'seed': args.seed,
        'val_mod': args.val_mod,
        'progress_interval': args.progress_interval,
        'inline_progress': inline_progress,
        'domain_split_cfg': domain_split_cfg,
    }


def _run_task(task, config):
    """Execute one conversion task (used for sequential and parallel runs)."""
    print('\n========== {} START =========='.format(task))
    if task == 'real':
        convert_real(
            config['real_root'], config['out_root'],
            test_bases=config['test_bases'], clean=config['clean'],
            crop_policy=config['crop_policy'], use_crop=config['use_crop'],
            progress_interval=config['progress_interval'],
            inline_progress=config['inline_progress'],
            domain_split_cfg=config.get('domain_split_cfg'),
        )
    elif task == 'aebad':
        convert_aebad_maml(
            config['aebad_root'], config['out_root'],
            clean=config['clean'], crop_policy=config['crop_policy'],
            normal_per_domain=config['normal_per_domain'], seed=config['seed'],
            progress_interval=config['progress_interval'],
            inline_progress=config['inline_progress'],
        )
    elif task == 'bladesynth':
        convert_bladesynth(
            config['bladesynth_root'], config['out_root'],
            clean=config['clean'], crop_policy=config['crop_policy'],
            progress_interval=config['progress_interval'],
            inline_progress=config['inline_progress'],
        )
    elif task == 'aebad_pretrain':
        convert_aebad_pretrain(
            config['aebad_root'], config['out_root'],
            clean=config['clean'], crop_policy=config['crop_policy'],
            val_mod=config['val_mod'],
            progress_interval=config['progress_interval'],
            inline_progress=config['inline_progress'],
        )
    else:
        raise ValueError('Unknown task: {}'.format(task))
    print('========== {} DONE =========='.format(task))
    return task


def _run_tasks_parallel(tasks, config, jobs):
    task_list = [t for t in TASK_ORDER if t in tasks]
    parallel_config = dict(config)
    parallel_config['inline_progress'] = False
    print('[parallel] running {} tasks with {} workers: {}'.format(
        len(task_list), jobs, ', '.join(task_list)))
    errors = []
    # ThreadPoolExecutor: safe on Windows from __main__, tasks write to separate dirs
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_task, t, parallel_config): t for t in task_list}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                errors.append((name, exc))
                print('[parallel] task {} FAILED: {}'.format(name, exc), flush=True)
    if errors:
        raise RuntimeError('Parallel conversion failed: {}'.format(errors))


def _run_check(out_root):
    print('\n========== Check ==========')
    _scripts = os.path.dirname(os.path.abspath(__file__))
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from check_all_data import main as check_main
    sys.argv = ['check_all_data.py', '--data_dir_path', out_root]
    check_main()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert aviation datasets (single entry point)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    t = parser.add_argument_group('tasks')
    t.add_argument('--real', action='store_true', help='YOLO -> aviation/domain_0~3')
    t.add_argument('--aebad', action='store_true', help='AeBAD -> aviation/domain_4~7')
    t.add_argument('--bladesynth', action='store_true', help='bladeSynth -> pretrain/bladesynth/')
    t.add_argument('--aebad_pretrain', action='store_true', help='AeBAD -> pretrain/aebad/')
    t.add_argument('--maml', action='store_true', help='--real --aebad')
    t.add_argument('--pretrain', action='store_true', help='--bladesynth --aebad_pretrain')
    t.add_argument('--all', action='store_true', help='All four tasks')
    t.add_argument('--check', action='store_true', help='Run check_all_data.py after convert')

    x = parser.add_argument_group('parallel / progress')
    x.add_argument('--no_parallel', action='store_true',
                   help='Disable parallel; run tasks one by one (default: parallel when 2+ tasks)')
    x.add_argument('--jobs', type=int, default=0,
                   help='Max parallel workers (0=auto, up to 4)')
    x.add_argument('--progress_interval', type=int, default=0,
                   help='Print every N images (0=auto, default 50)')

    p = parser.add_argument_group('paths')
    p.add_argument('--out_root', default=_DEFAULT_DATA,
                   help='Output root (contains aviation/ pretrain/)')
    p.add_argument('--real_root',
                   default=os.path.join(_DEFAULT_DATA, 'Aero-engine_defect-detect_new'))
    p.add_argument('--aebad_root', default=os.path.join(_DEFAULT_DATA, 'AeBAD'))
    p.add_argument('--bladesynth_root',
                   default=os.path.join(_DEFAULT_DATA, 'Dataset_bladesynth'))

    r = parser.add_argument_group('real')
    r.add_argument('--test_bases', default='20,21,22,23,24,25,26')
    r.add_argument('--domain_split', default='legacy_mod3',
                   choices=('legacy_mod3', 'csv', 'auto'),
                   help='Real YOLO domain assignment: legacy %%3, CSV registry, or auto greedy')
    r.add_argument('--domain_map_csv', default='',
                   help='CSV registry for --domain_split csv (see data/航天/specimen_domain_map.template.csv)')
    r.add_argument('--domain_group_by', default='train_domain',
                   choices=('train_domain', 'material_id', 'process_batch', 'camera_batch'),
                   help='When train_domain empty in CSV, bucket by this column')
    r.add_argument('--domain_k_min', type=int, default=3,
                   help='Auto split: minimum train domain count')
    r.add_argument('--domain_k_max', type=int, default=5,
                   help='Auto split: maximum train domain count')
    r.add_argument('--domain_ways', type=int, default=2,
                   help='Feasibility check: N-way')
    r.add_argument('--domain_shots', type=int, default=5,
                   help='Feasibility check: K-shot (needs 2*K ROI per class per domain)')

    a = parser.add_argument_group('aebad MAML')
    a.add_argument('--normal_per_domain', type=int, default=50)
    a.add_argument('--seed', type=int, default=42)

    pr = parser.add_argument_group('aebad pretrain')
    pr.add_argument('--val_mod', type=int, default=10)

    c = parser.add_argument_group('clean')
    c.add_argument('--no_clean', action='store_true')

    add_crop_args(parser)
    return parser.parse_args()


def resolve_tasks(args):
    if args.all:
        return {'real', 'aebad', 'bladesynth', 'aebad_pretrain'}
    tasks = set()
    if args.maml:
        tasks.update({'real', 'aebad'})
    if args.pretrain:
        tasks.update({'bladesynth', 'aebad_pretrain'})
    if args.real:
        tasks.add('real')
    if args.aebad:
        tasks.add('aebad')
    if args.bladesynth:
        tasks.add('bladesynth')
    if args.aebad_pretrain:
        tasks.add('aebad_pretrain')
    return tasks


def run(args):
    tasks = resolve_tasks(args)
    if not tasks and not args.check:
        raise SystemExit('No task selected. Use --real / --aebad / --bladesynth / '
                         '--aebad_pretrain / --maml / --pretrain / --all. See --help.')

    out_root = os.path.abspath(args.out_root)

    if args.progress_interval <= 0:
        args.progress_interval = 50

    if tasks:
        use_parallel = not args.no_parallel and len(tasks) > 1
        inline = not use_parallel
        config = _task_config(args, inline_progress=inline)
        jobs = args.jobs if args.jobs > 0 else min(4, len(tasks))

        if use_parallel:
            _run_tasks_parallel(tasks, config, jobs)
        else:
            for task in TASK_ORDER:
                if task in tasks:
                    _run_task(task, config)

    if args.check:
        _run_check(out_root)


def main():
    run(parse_args())


# Backward-compatible aliases used by old script names
convert = convert_real

if __name__ == '__main__':
    main()
