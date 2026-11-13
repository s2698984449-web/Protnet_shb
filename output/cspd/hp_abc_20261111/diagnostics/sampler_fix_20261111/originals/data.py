"""通用类别文件夹 episode。由 scripts/load_data.py、train_cspd.py、infer_cspd.py 调用。

目录只认配置里的绝对路径：

    <split_root>/
      类别A/
        来源1/  图片
        来源2/
      类别B/

cross_source 为 true 时，同一类的支持集和查询集必须来自不同的直接子目录。
holdout_source 有值时，训练和验证去掉该来源；测试的查询只用该来源，支持集用其他来源。
来源名写入 source_slot 指定的工况槽，缺测维仍不进编码。

可选旁路也全部由配置给出，不写死数据集：
    condition_file          槽位 JSON
    illumination_manifest   亮度脚本 CSV
    mask_root + mask_pattern 掩码
    mask_root_train/val/test 按划分覆盖 mask_root
    restored_root           扩散/光照还原后的平行目录
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .annotate import crop_around_mask, load_mask, resolve_mask_path
from .condition_io import DEFAULT_SLOTS, ConditionBuilder


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIX = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SPLIT_KEYS = ('train_dir', 'val_dir', 'test_dir')


def absolute_dir(cfg, key, required=True):
    raw = str(cfg.get(key) or '').strip()
    if not raw:
        if required:
            raise ValueError('配置缺少 {}'.format(key))
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError('{} 必须是绝对路径，当前是 {}'.format(key, raw))
    if not path.is_dir():
        raise FileNotFoundError('目录不存在: {}'.format(path))
    return path


def optional_file(cfg, key):
    raw = str(cfg.get(key) or '').strip()
    if not raw:
        return ''
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError('{} 必须是绝对路径，当前是 {}'.format(key, raw))
    return str(path)


class ClassFolderEpisodes:
    """从一个类别文件夹目录抽 n-way k-shot。类名只在本集内重映射为 0..ways-1。"""

    def __init__(self, root, cfg, seed=0):
        self.root = Path(root)
        self.n_way = int(cfg['n_way'])
        self.k_shot = int(cfg['k_shot'])
        self.q_query = int(cfg['q_query'])
        self.image_size = int(cfg['image_size'])
        self.rng = random.Random(seed)
        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.mask_root = str(cfg.get('mask_root') or '').strip()
        self.mask_pattern = str(cfg.get('mask_pattern') or '{stem}_mask.png')
        self.restored_root = str(cfg.get('restored_root') or '').strip()
        self.cross_source = bool(cfg.get('cross_source', False))
        self.holdout_source = str(cfg.get('holdout_source') or '').strip()
        self.split_key = str(cfg.get('_split_key') or '')
        self.query_source = ''
        self.source_slot = str(cfg.get('source_slot') or '').strip()
        self.source_table = {
            str(name): dict(slots or {})
            for name, slots in (cfg.get('source_table') or {}).items()
        }
        slots = cfg.get('cond_slots') or list(DEFAULT_SLOTS)
        self.cond = ConditionBuilder(
            slots=slots,
            condition_file=optional_file(cfg, 'condition_file'),
            illumination_manifest=optional_file(cfg, 'illumination_manifest'),
            fallback_image_stats=bool(cfg.get('cond_fallback_image_stats', True)),
            illuminance_offset_scale=float(cfg.get('illuminance_offset_scale', 80.0)),
            source_slot=self.source_slot,
        )
        if int(cfg.get('cond_dim', self.cond.dim)) != self.cond.dim:
            raise ValueError(
                'cond_dim={} 与 cond_slots 长度 {} 不一致'.format(
                    cfg.get('cond_dim'), self.cond.dim))
        include = cfg.get('include_classes') or []
        exclude = set(cfg.get('exclude_classes') or [])
        include = set(include) if include else None
        self.by_class = {}
        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            name = class_dir.name
            if include is not None and name not in include:
                continue
            if name in exclude:
                continue
            paths = [
                p for p in class_dir.rglob('*')
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIX
            ]
            if paths:
                self.by_class[name] = self._by_source(class_dir, paths)
        if self.holdout_source and self.split_key in ('train_dir', 'val_dir'):
            self._drop_source(self.holdout_source)
        elif self.holdout_source and self.split_key == 'test_dir':
            self.query_source = self.holdout_source
        self.classes = sorted(self.by_class)
        self.source_names = sorted({
            source
            for groups in self.by_class.values()
            for source in groups
            if source != '_flat'
        })
        self.source_value = {
            name: (index + 1) / float(max(len(self.source_names), 1))
            for index, name in enumerate(self.source_names)
        }
        self.cross_classes = [name for name in self.classes if self._source_pairs(name)]
        self.pool = self._episode_pool()
        if self.cross_source or self.query_source:
            if len(self.pool) < self.n_way:
                detail = {
                    name: {src: len(paths) for src, paths in groups.items()}
                    for name, groups in self.by_class.items()
                }
                raise RuntimeError(
                    '{} 只有 {} 个类能进 episode，n_way={}。各类来源张数 {}'.format(
                        self.root, len(self.pool), self.n_way, detail))
        else:
            need = self.k_shot + self.q_query
            if len(self.classes) < self.n_way:
                raise RuntimeError(
                    '{} 只有 {} 个类别，n_way={}'.format(self.root, len(self.classes), self.n_way))
            for name, groups in self.by_class.items():
                real = [len(paths) for src, paths in groups.items() if src != '_flat']
                if real:
                    if max(real) < need:
                        raise RuntimeError(
                            '{} 的 {} 没有来源同时够 {} 张，最多 {}'.format(
                                self.root, name, need, max(real)))
                elif sum(len(paths) for paths in groups.values()) < need:
                    raise RuntimeError(
                        '{} 的 {} 只有 {} 张图，需要 {}'.format(
                            self.root, name, sum(len(paths) for paths in groups.values()), need))

    def _by_source(self, class_dir, paths):
        groups = {}
        for path in paths:
            rel = path.relative_to(class_dir)
            source = rel.parts[0] if len(rel.parts) > 1 else '_flat'
            groups.setdefault(source, []).append(path)
        return {source: sorted(items) for source, items in groups.items()}

    def _drop_source(self, source):
        kept = {}
        for name, groups in self.by_class.items():
            groups = {
                src: paths for src, paths in groups.items()
                if src != source and paths
            }
            if groups:
                kept[name] = groups
        self.by_class = kept

    def _episode_pool(self):
        if self.query_source:
            return [name for name in self.classes if self._source_pairs(name)]
        if self.cross_source:
            names = list(self.cross_classes)
            if self.holdout_source:
                names.extend(
                    name for name in self.classes
                    if name not in self.cross_classes and self._same_source_ok(name))
            return names
        return list(self.classes)

    def _same_source_ok(self, name):
        groups = self.by_class[name]
        need = self.k_shot + self.q_query
        return any(
            src != '_flat' and len(paths) >= need
            for src, paths in groups.items())

    def _source_pairs(self, name):
        groups = self.by_class[name]
        usable_s = [src for src, paths in groups.items() if src != '_flat' and len(paths) >= self.k_shot]
        if self.query_source:
            query_paths = groups.get(self.query_source) or []
            if len(query_paths) < self.q_query:
                return []
            return [(src_s, self.query_source) for src_s in usable_s if src_s != self.query_source]
        usable_q = [src for src, paths in groups.items() if src != '_flat' and len(paths) >= self.q_query]
        return [(src_s, src_q) for src_s in usable_s for src_q in usable_q if src_s != src_q]

    def _restored_path(self, path):
        if not self.restored_root:
            return None
        rel = path.relative_to(self.root)
        candidate = Path(self.restored_root) / rel
        if candidate.is_file():
            return candidate
        # 亮度脚本常把后缀改成 .png
        png = candidate.with_suffix('.png')
        return png if png.is_file() else None

    def _load_rgb(self, path):
        img = Image.open(path).convert('RGB')
        img = self.resize(img)
        rgb_01 = transforms.ToTensor()(img)
        return self.normalize(rgb_01), rgb_01

    def _load(self, path):
        x, rgb_01 = self._load_rgb(path)
        rel = path.relative_to(self.root)
        source_name = rel.parts[-2] if len(rel.parts) >= 3 else None
        source_value = self.source_value.get(source_name)
        source_slots = self.source_table.get(source_name)
        cond, cond_mask = self.cond.build(
            rel.as_posix(), rgb_01, source_value=source_value, source_slots=source_slots)
        mask_path = resolve_mask_path(path, self.root, self.mask_root, self.mask_pattern)
        mask = load_mask(mask_path, self.image_size)
        has_mask = mask is not None
        if mask is None:
            mask = torch.zeros(1, self.image_size, self.image_size)
        restored = self._restored_path(path)
        if restored is None:
            x_rest = x
            has_restored = False
        else:
            x_rest, _ = self._load_rgb(restored)
            has_restored = True
        return {
            'x': x,
            'x_restored': x_rest,
            'y_cond': cond,
            'y_cond_mask': cond_mask,
            'mask': mask,
            'has_mask': has_mask,
            'has_restored': has_restored,
            'path': str(path),
        }

    def _take(self, name):
        groups = self.by_class[name]
        if self.cross_source and self._source_pairs(name):
            src_s, src_q = self.rng.choice(self._source_pairs(name))
            support = self.rng.sample(self.by_class[name][src_s], self.k_shot)
            query = self.rng.sample(self.by_class[name][src_q], self.q_query)
            return support, query
        if not self.cross_source or self.holdout_source:
            real = {src: paths for src, paths in groups.items() if src != '_flat'}
            if real:
                need = self.k_shot + self.q_query
                usable = [src for src, paths in real.items() if len(paths) >= need]
                src = self.rng.choice(usable)
                picked = self.rng.sample(real[src], need)
                return picked[:self.k_shot], picked[self.k_shot:]
            paths = [path for group in groups.values() for path in group]
            picked = self.rng.sample(paths, self.k_shot + self.q_query)
            return picked[:self.k_shot], picked[self.k_shot:]
        raise RuntimeError('{} 的 {} 抽不出支持集和查询集'.format(self.root, name))

    def sample(self):
        pool = self.pool
        chosen = self.rng.sample(pool, self.n_way)
        buckets = {
            'support_x': [], 'support_x_restored': [], 'support_y': [],
            'support_cond': [], 'support_cond_mask': [],
            'support_mask': [], 'support_has_mask': [],
            'query_x': [], 'query_x_restored': [], 'query_y': [],
            'query_cond': [], 'query_cond_mask': [],
            'query_mask': [], 'query_has_mask': [],
        }
        for local_y, name in enumerate(chosen):
            support_paths, query_paths = self._take(name)
            for path in support_paths:
                item = self._load(path)
                buckets['support_x'].append(item['x'])
                buckets['support_x_restored'].append(item['x_restored'])
                buckets['support_y'].append(local_y)
                buckets['support_cond'].append(item['y_cond'])
                buckets['support_cond_mask'].append(item['y_cond_mask'])
                buckets['support_mask'].append(item['mask'])
                buckets['support_has_mask'].append(item['has_mask'])
            for path in query_paths:
                item = self._load(path)
                buckets['query_x'].append(item['x'])
                buckets['query_x_restored'].append(item['x_restored'])
                buckets['query_y'].append(local_y)
                buckets['query_cond'].append(item['y_cond'])
                buckets['query_cond_mask'].append(item['y_cond_mask'])
                buckets['query_mask'].append(item['mask'])
                buckets['query_has_mask'].append(item['has_mask'])
        return {
            'support_x': torch.stack(buckets['support_x'], 0),
            'support_x_restored': torch.stack(buckets['support_x_restored'], 0),
            'support_y': torch.tensor(buckets['support_y'], dtype=torch.long),
            'support_cond': torch.stack(buckets['support_cond'], 0),
            'support_cond_mask': torch.stack(buckets['support_cond_mask'], 0),
            'support_mask': torch.stack(buckets['support_mask'], 0),
            'support_has_mask': torch.tensor(buckets['support_has_mask'], dtype=torch.bool),
            'query_x': torch.stack(buckets['query_x'], 0),
            'query_x_restored': torch.stack(buckets['query_x_restored'], 0),
            'query_y': torch.tensor(buckets['query_y'], dtype=torch.long),
            'query_cond': torch.stack(buckets['query_cond'], 0),
            'query_cond_mask': torch.stack(buckets['query_cond_mask'], 0),
            'query_mask': torch.stack(buckets['query_mask'], 0),
            'query_has_mask': torch.tensor(buckets['query_has_mask'], dtype=torch.bool),
            'class_names': list(chosen),
        }


def make_episodes(cfg, key, seed):
    local = dict(cfg)
    local['_split_key'] = key
    if key == 'test_dir' and cfg.get('n_way_test') not in (None, ''):
        local['n_way'] = int(cfg['n_way_test'])
    override = str(cfg.get('mask_root_' + key[:-4]) or '').strip()
    if override:
        local['mask_root'] = override
    return ClassFolderEpisodes(absolute_dir(local, key), local, seed=seed)


def describe(sampler):
    state = sampler.rng.getstate()
    episode = sampler.sample()
    sampler.rng.setstate(state)
    return {
        'dir': str(sampler.root),
        'classes': list(sampler.classes),
        'counts': {
            name: {src: len(paths) for src, paths in groups.items()}
            for name, groups in sampler.by_class.items()
        },
        'cross_source': sampler.cross_source,
        'holdout_source': sampler.holdout_source or None,
        'query_source': sampler.query_source or None,
        'pool': list(sampler.pool),
        'cross_classes': list(sampler.cross_classes),
        'cond_slots': list(sampler.cond.slots),
        'mask_root': sampler.mask_root or None,
        'restored_root': sampler.restored_root or None,
        'episode': {
            'class_names': list(episode['class_names']),
            'support_x': list(episode['support_x'].shape),
            'query_x': list(episode['query_x'].shape),
            'support_cond': list(episode['support_cond'].shape),
            'support_cond_mask_mean': [
                round(float(v), 4) for v in episode['support_cond_mask'].mean(0)],
            'support_has_mask_ratio': float(episode['support_has_mask'].float().mean()),
            'query_has_mask_ratio': float(episode['query_has_mask'].float().mean()),
        },
    }


def describe_config(cfg, seed):
    report = {}
    for key in SPLIT_KEYS:
        if not str(cfg.get(key) or '').strip():
            continue
        report[key] = describe(make_episodes(cfg, key, seed))
    if 'train_dir' not in report or 'test_dir' not in report:
        raise ValueError('配置至少要有 train_dir 和 test_dir 的绝对路径')
    return report


def curriculum_rho(progress, schedule=None):
    """progress ∈ [0,1]。默认 0–25%→0.3，25–75% 线性到 0.8，之后 0.8。"""
    schedule = schedule or ((0.0, 0.3), (0.25, 0.3), (0.75, 0.8), (1.0, 0.8))
    points = sorted((float(a), float(b)) for a, b in schedule)
    if progress <= points[0][0]:
        return points[0][1]
    for (p0, r0), (p1, r1) in zip(points, points[1:]):
        if progress <= p1:
            if p1 <= p0:
                return r1
            t = (progress - p0) / (p1 - p0)
            return r0 + t * (r1 - r0)
    return points[-1][1]


class CurriculumNewSplitEpisodes:
    """data/new_split：细类/来源/split。查询来源均匀；跨来源时支持来自其他有掩码来源。"""

    def __init__(self, cfg, split='train', seed=0):
        root = Path(str(cfg.get('data_root') or '').strip())
        if not root.is_absolute() or not root.is_dir():
            raise ValueError('curriculum 需要绝对且存在的 data_root')
        self.root = root
        self.split = str(split)
        self.image_size = int(cfg.get('image_size', 224))
        self.n_way_cap = int(cfg.get('n_way', 5))
        self.q_query = int(cfg.get('q_query', 8))
        self.k_choices = list(cfg.get('k_choices') or [3, 5])
        self.k_probs = list(cfg.get('k_probs') or [0.7, 0.3])
        self.require_mask = bool(cfg.get('require_mask', True))
        self.holdout = str(cfg.get('holdout_source') or '').strip()
        self.query_only = str(cfg.get('query_only') or '').strip()
        exclude_sources = set(cfg.get('exclude_sources') or [])
        # holdout_eval=same 时，普通测试也不放留出来源；留出评估自己把 query_only 设成该来源。
        if self.holdout:
            if self.split in ('train', 'val') or (
                self.holdout != self.query_only and str(cfg.get('holdout_eval') or '') == 'same'
            ):
                exclude_sources.add(self.holdout)
        self.exclude_cross = set(cfg.get('exclude_classes_cross') or [])
        self.source_table = dict(cfg.get('source_table') or {})
        self.cond = ConditionBuilder(
            slots=cfg.get('cond_slots') or DEFAULT_SLOTS,
            condition_file=cfg.get('condition_file') or '',
            illumination_manifest=cfg.get('illumination_manifest') or '',
            fallback_image_stats=bool(cfg.get('cond_fallback_image_stats', False)),
            illuminance_offset_scale=float(cfg.get('illuminance_offset_scale', 80.0)),
            source_slot=cfg.get('source_slot') or '',
        )
        if int(cfg.get('cond_dim', self.cond.dim)) != self.cond.dim:
            raise ValueError('cond_dim 与 cond_slots 不一致')
        self.crop_to_mask = bool(cfg.get('crop_to_mask', True))
        self.crop_margin = float(cfg.get('crop_margin', 0.35))
        self.crop_min_side = int(cfg.get('crop_min_side', 96))
        self.crop_square = bool(cfg.get('crop_square', True))
        self.resize = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.rng = random.Random(seed)
        self.holdout_pool = str(cfg.get('holdout_pool') or '')
        raw_cross = cfg.get('cross_classes')
        self.cross_classes = {str(name) for name in raw_cross} if raw_cross else None
        self.group_of = {}
        self.source_dist = {}
        self.hard_pairs = []  # legacy non-suite hook
        self.same_hard_pairs = {}
        self.hard_pair_enabled = bool(cfg.get('same_hard_pairs', False))
        self.hard_rng = random.Random(seed + 17011)
        self.hard_probability = 0.0
        self.strict_groups = bool(cfg.get('strict_groups', False))
        self.group_failures = {}
        self.source_attempts = {}
        self.source_successes = {}
        self.by_class = {}
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            name = class_dir.name
            groups = {}
            for source_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                source = source_dir.name
                if source in exclude_sources:
                    continue
                split_names = [self.split]
                if (
                    self.query_only
                    and source == self.query_only
                    and self.split == 'test'
                    and self.holdout_pool == 'all'
                ):
                    split_names = ['train', 'val', 'test']
                group_file = source_dir / 'groups.json'
                group_map = {}
                if group_file.is_file():
                    group_map = json.loads(group_file.read_text(encoding='utf-8'))
                items = []
                for split_name in split_names:
                    split_dir = source_dir / split_name
                    if not split_dir.is_dir():
                        continue
                    for path in sorted(split_dir.iterdir()):
                        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIX:
                            continue
                        mask = split_dir / 'mask' / (path.stem + '.png')
                        if self.require_mask and not mask.is_file():
                            continue
                        items.append((path, mask if mask.is_file() else None))
                        if path.stem in group_map:
                            self.group_of[(name, source, path.stem)] = str(group_map[path.stem])
                if items:
                    groups[source] = items
            if groups:
                self.by_class[name] = groups
        self.classes = sorted(self.by_class)
        self.sources = sorted({
            src for groups in self.by_class.values() for src in groups
        })
        self.source_classes = {
            src: [name for name in self.classes if src in self.by_class[name]]
            for src in self.sources
        }
        # 跨来源池：至少还有一个其他来源、且不在 exclude_cross
        self.cross_ok = {
            name for name, groups in self.by_class.items()
            if name not in self.exclude_cross and len(groups) >= 2
        }
        if not self.sources:
            raise RuntimeError('{} 的 {} 没有可用来源'.format(root, self.split))

    @property
    def n_way(self):
        return self.n_way_cap

    def _load(self, path, mask_path, source_name):
        img = Image.open(path).convert('RGB')
        mask_img = None
        if mask_path is not None and Path(mask_path).is_file():
            mask_img = Image.open(mask_path).convert('L')
        if self.crop_to_mask and mask_img is not None:
            img, mask_img = crop_around_mask(
                img, mask_img,
                margin=self.crop_margin,
                min_side=self.crop_min_side,
                square=self.crop_square,
            )
        img = self.resize(img)
        rgb_01 = transforms.ToTensor()(img)
        x = self.normalize(rgb_01)
        slots = self.source_table.get(source_name)
        cond, cond_mask = self.cond.build(
            path.relative_to(self.root).as_posix(), rgb_01,
            source_value=None, source_slots=slots)
        if mask_img is None:
            mask = torch.zeros(1, self.image_size, self.image_size)
            has_mask = False
        else:
            mask_img = self.resize_mask(mask_img)
            mask = transforms.ToTensor()(mask_img)
            if float(mask.max()) > 0:
                mask = (mask > 0.5).float()
            has_mask = True
        return {
            'x': x,
            'y_cond': cond,
            'y_cond_mask': cond_mask,
            'mask': mask,
            'has_mask': has_mask,
            'path': str(path),
        }

    def _pick_k(self):
        return self.rng.choices(self.k_choices, weights=self.k_probs, k=1)[0]

    def _query_candidates(self, cross_source):
        if cross_source:
            allowed = self.cross_classes
            candidates = [
                src for src, names in self.source_classes.items()
                if any(
                    name in self.cross_ok
                    and (allowed is None or name in allowed)
                    and src in self.by_class[name]
                    and any(other != src for other in self.by_class[name])
                    for name in names
                )
            ]
        else:
            candidates = [src for src, names in self.source_classes.items() if names]
        if self.query_only:
            candidates = [src for src in candidates if src == self.query_only]
        return candidates

    def _pick_support_source(self, options, query_source, progress, stage_sources):
        if (not stage_sources) or (not self.source_dist) or len(options) <= 1:
            return self.rng.choice(options)
        ranked = sorted(
            options,
            key=lambda src: self.source_dist.get((query_source, src), 1.0))
        n = len(ranked)
        width = max(1, (n + 2) // 3)
        if progress < 0.25:
            pool = ranked[:width]
        elif progress < 0.75:
            start = n // 3
            pool = ranked[start:max(start + 1, (2 * n) // 3)] or ranked
        else:
            pool = ranked[-width:]
        return self.rng.choice(pool)

    def _split_by_group(self, pool, k, need_q, class_name, source):
        """支持集和查询集尽量来自不同的组。组不够时返回 None，调用方再按图抽。"""
        if not self.group_of:
            return None
        by_group = {}
        for item in pool:
            gid = self.group_of.get((class_name, source, item[0].stem))
            if gid is None:
                return None
            by_group.setdefault(gid, []).append(item)
        groups = list(by_group.values())
        self.rng.shuffle(groups)
        if self.strict_groups:
            # Subset sum reserves whole groups for query; never split a group.
            total = sum(len(g) for g in groups)
            states = {0: []}
            selected = None
            for index, group in enumerate(groups):
                for size, ids in list(states.items()):
                    new_size = size + len(group)
                    if new_size > total - need_q:
                        continue
                    states.setdefault(new_size, ids + [index])
                    if new_size >= k:
                        selected = states[new_size]
                        break
                if selected is not None:
                    break
            if selected is None:
                return None
            support = [item for i in selected for item in groups[i]]
            query = [item for i, g in enumerate(groups) if i not in selected for item in g]
            return self.rng.sample(support, k), self.rng.sample(query, need_q)
        support = []
        query = []
        for group in groups:
            if len(support) < k:
                support.extend(group)
            else:
                query.extend(group)
        if len(support) < k or len(query) < need_q:
            return None
        self.rng.shuffle(support)
        self.rng.shuffle(query)
        return support[:k], query[:need_q]

    def _sample_source(self, query_source, cross_source, progress, stage_sources):
        k = self._pick_k()
        need_q = self.q_query
        usable = []
        support_of = {}
        same_names = []
        for name in self.source_classes[query_source]:
            q_paths = self.by_class[name][query_source]
            if len(q_paths) >= k + need_q:
                same_names.append(name)
        if cross_source:
            allowed = self.cross_classes
            for name in self.source_classes[query_source]:
                if allowed is not None and name not in allowed:
                    continue
                if name not in self.cross_ok:
                    continue
                q_paths = self.by_class[name][query_source]
                support_sources = [
                    src for src, items in self.by_class[name].items()
                    if src != query_source and len(items) >= k
                ]
                if support_sources and len(q_paths) >= need_q:
                    usable.append(name)
                    support_of[name] = support_sources
            if not usable:
                raise RuntimeError('来源 {} 没有可跨来源的类'.format(query_source))
            # 其余类仍用同一来源的支持，避免把未确认的同名类当成跨来源正样本。
            for name in same_names:
                if name not in support_of:
                    usable.append(name)
        else:
            usable = list(same_names)
            if len(usable) < 2:
                raise RuntimeError('来源 {} 可抽类不足'.format(query_source))
        if len(usable) < 2:
            raise RuntimeError('来源 {} 可抽类不足'.format(query_source))
        n_way = min(self.n_way_cap, len(usable))
        chosen = []
        if stage_sources and progress >= 0.75 and not cross_source:
            for left, right in self.hard_pairs:
                if left in usable and right in usable:
                    chosen = [left, right]
                    break
        if cross_source:
            cross_first = [name for name in usable if name in support_of]
            chosen = list(cross_first[:n_way])
        hard_info = None
        if self.hard_pair_enabled and stage_sources and not cross_source:
            from cspd.hard_pairs import choose_classes
            chosen, hard_info = choose_classes(
                usable, n_way, self.rng, self.hard_rng,
                self.same_hard_pairs.get(query_source, []), self.hard_probability)
        else:
            rest = [name for name in usable if name not in chosen]
            need = n_way - len(chosen)
            if need > 0:
                if len(rest) < need:
                    raise RuntimeError('来源 {} 可抽类不足'.format(query_source))
                chosen.extend(self.rng.sample(rest, need))
        buckets = {
            'support_x': [], 'support_y': [], 'support_cond': [], 'support_cond_mask': [],
            'support_mask': [], 'support_has_mask': [],
            'query_x': [], 'query_y': [], 'query_cond': [], 'query_cond_mask': [],
            'query_mask': [], 'query_has_mask': [],
        }
        for local_y, name in enumerate(chosen):
            q_pool = list(self.by_class[name][query_source])
            if name in support_of:
                src_s = self._pick_support_source(
                    support_of[name], query_source, progress, stage_sources)
                support_items = self.rng.sample(self.by_class[name][src_s], k)
                query_items = self.rng.sample(q_pool, need_q)
            else:
                split_pair = self._split_by_group(q_pool, k, need_q, name, query_source)
                if split_pair is None:
                    if self.strict_groups:
                        key = '{}:{}'.format(query_source, name)
                        self.group_failures[key] = self.group_failures.get(key, 0) + 1
                        raise RuntimeError('strict groups unavailable: ' + key)
                    picked = self.rng.sample(q_pool, k + need_q)
                    support_items = picked[:k]
                    query_items = picked[k:]
                else:
                    support_items, query_items = split_pair
            for path, mask in support_items:
                src_name = path.parent.parent.name
                item = self._load(path, mask, src_name)
                buckets['support_x'].append(item['x'])
                buckets['support_y'].append(local_y)
                buckets['support_cond'].append(item['y_cond'])
                buckets['support_cond_mask'].append(item['y_cond_mask'])
                buckets['support_mask'].append(item['mask'])
                buckets['support_has_mask'].append(item['has_mask'])
            for path, mask in query_items:
                item = self._load(path, mask, query_source)
                buckets['query_x'].append(item['x'])
                buckets['query_y'].append(local_y)
                buckets['query_cond'].append(item['y_cond'])
                buckets['query_cond_mask'].append(item['y_cond_mask'])
                buckets['query_mask'].append(item['mask'])
                buckets['query_has_mask'].append(item['has_mask'])
        return {
            'support_x': torch.stack(buckets['support_x'], 0),
            'support_y': torch.tensor(buckets['support_y'], dtype=torch.long),
            'support_cond': torch.stack(buckets['support_cond'], 0),
            'support_cond_mask': torch.stack(buckets['support_cond_mask'], 0),
            'support_mask': torch.stack(buckets['support_mask'], 0),
            'support_has_mask': torch.tensor(buckets['support_has_mask'], dtype=torch.bool),
            'query_x': torch.stack(buckets['query_x'], 0),
            'query_y': torch.tensor(buckets['query_y'], dtype=torch.long),
            'query_cond': torch.stack(buckets['query_cond'], 0),
            'query_cond_mask': torch.stack(buckets['query_cond_mask'], 0),
            'query_mask': torch.stack(buckets['query_mask'], 0),
            'query_has_mask': torch.tensor(buckets['query_has_mask'], dtype=torch.bool),
            'class_names': list(chosen),
            'query_source': query_source,
            'cross_source': bool(cross_source),
            'n_way': n_way,
            'k_shot': k,
            'hard_pair': hard_info,
        }

    def sample(
            self, cross_source=None, progress=0.0, avoid_sources=None,
            stage_sources=False, allow_fallback=True):
        if cross_source is None:
            cross_source = self.rng.random() < curriculum_rho(progress)
        candidates = self._query_candidates(cross_source)
        if avoid_sources:
            preferred = [src for src in candidates if src not in set(avoid_sources)]
            if preferred:
                candidates = preferred
        # 跨来源候选为空时（例如留出 CR7 后冲孔只剩一家），退回同产线。
        if not candidates and cross_source and not self.query_only and allow_fallback:
            return self.sample(
                cross_source=False,
                progress=progress,
                avoid_sources=avoid_sources,
                stage_sources=stage_sources,
                allow_fallback=False,
            )
        if not candidates:
            raise RuntimeError('没有可采样的查询来源 cross={}'.format(cross_source))
        order = list(candidates)
        self.rng.shuffle(order)
        last_error = None
        for query_source in order:
            self.source_attempts[query_source] = self.source_attempts.get(query_source, 0) + 1
            try:
                episode = self._sample_source(query_source, cross_source, progress, stage_sources)
                self.source_successes[query_source] = self.source_successes.get(query_source, 0) + 1
                return episode
            except RuntimeError as exc:
                last_error = exc
        if cross_source and not self.query_only and allow_fallback:
            return self.sample(
                cross_source=False,
                progress=progress,
                avoid_sources=avoid_sources,
                stage_sources=stage_sources,
                allow_fallback=False,
            )
        raise last_error or RuntimeError('来源不足')


def make_curriculum(cfg, split, seed):
    return CurriculumNewSplitEpisodes(cfg, split=split, seed=seed)
