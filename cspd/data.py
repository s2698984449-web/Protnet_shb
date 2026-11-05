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

import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .annotate import load_mask, resolve_mask_path
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
