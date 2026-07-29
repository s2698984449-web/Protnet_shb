"""Phase 0 pretrain datasets for automotive CMPL track (v6: car-only).

Each dataset returns a dict (image, label) for standard task head,
and the raw task_name + raw class name for the auxiliary scratch binary head.
If aux_scratch_head is enabled, we also return is_scratch (0 or 1).

v6 (2026-07): removed neu/asde/kolektor/mvtec_* (工业域, 与车漆特征无关).
Added car_parts_damage (8 类细粒度), minor-dent (单类 dent),
and syndcar (合成 4 类).
"""

import os

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .cmpl_taxonomy import CARDD_PRETRAIN_CLASSES, CARPARTS_PRETRAIN_CLASSES


# ------------------------------------------------------------------
# CMPL v6: 4 task 数据集 + scratch 二分类标签映射
# ------------------------------------------------------------------
# - cardd: scratch / dent / crack / glass_shatter  (CarDD official)
# - carparts: Scratch / Dent / Cracked / Paint chip / Flaking / Broken part / Missing part / Corrosion
# - minor_dent: dent (单类)
# - syndcar: broken_glass / crack / scratch / broken_light
_SCRATCH_LABELS = {
    'cardd':      {'scratch': 1, 'dent': 0, 'crack': 0, 'glass_shatter': 0},
    'carparts':   {'Scratch': 1, 'Dent': 0, 'Cracked': 0, 'Paint chip': 0,
                   'Flaking': 0, 'Broken part': 0, 'Missing part': 0, 'Corrosion': 0},
    'minor_dent': {'dent': 0},     # 单类, 非 scratch
    'syndcar':    {'broken_glass': 0, 'crack': 0, 'scratch': 1, 'broken_light': 0},
}

# ------------------------------------------------------------------
# ImageNet mean/std for RGB (CMPL v6 全 RGB — 已移除 NEU 灰度分支)
# ------------------------------------------------------------------
_IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_RGB_STD  = [0.229, 0.224, 0.225]  # noqa: E501 保留引用避免旧 import 失败


def _open_rgb_for_pretrain(path, imgsz, grayscale=False):
    """Load image (CMPL v6: 全 RGB, grayscale 参数保留兼容旧代码)."""
    with Image.open(path) as handle:
        if grayscale:
            img = handle.convert('L')
            img = ImageOps.invert(img)
            img = Image.merge('RGB', (img, img, img))
        else:
            img = handle.convert('RGB')
        if max(img.size) > imgsz:
            img.thumbnail((imgsz, imgsz), Image.Resampling.BILINEAR)
        return img.copy()


def _list_jpgs(root):
    samples = []
    if not os.path.isdir(root):
        return samples
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and not fname.startswith('._'):
                samples.append(os.path.join(dirpath, fname))
    return sorted(samples)


def _split_pooled_subdirs(root, val_ratio=0.15, seed=42, k=5, fold=0):
    """对一池子图像按 (parent_subdir,) 做 stratified-style 划分, 返回 (train_paths, val_paths).

    适用于 task_b_minor_dent/dent/, task_c_carparts_damage/<Class>/, task_e_syndcar/<Class>/
    这些没有 train/val split 的目录. 同一 parent_subdir 的样本尽量不跨 train/val 分割
    (这里 parent_subdir 即类别名).

    当 k>1 时启用 stratified k-fold: val = fold-th slice, train = 其余, 保证每个类都有
    验证样本 (按比例分配, 不可整除时回退到 ceil/floor 交替).
    """
    import random
    rng = random.Random(seed)
    train, val = [], []
    if not os.path.isdir(root):
        return train, val
    for sub in sorted(os.listdir(root)):
        sub_path = os.path.join(root, sub)
        if not os.path.isdir(sub_path):
            continue
        files = _list_jpgs(sub_path)
        rng.shuffle(files)
        if k and k > 1:
            n = len(files)
            # Distribute so cum counts produce stable per-fold sizes.
            fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
            starts = [0]
            for s in fold_sizes[:-1]:
                starts.append(starts[-1] + s)
            ends = [start + size for start, size in zip(starts, fold_sizes)]
            # Defensive: clamp fold index
            if fold < 0:
                fold = 0
            if fold >= k:
                fold = k - 1
            val.extend(files[starts[fold]:ends[fold]])
            for i in range(k):
                if i == fold:
                    continue
                train.extend(files[starts[i]:ends[i]])
        else:
            n_val = max(1, int(round(len(files) * val_ratio)))
            val.extend(files[:n_val])
            train.extend(files[n_val:])
    return train, val


def _base_transforms(imgsz, augment=False):
    """CMPL v6: 全 RGB 数据, 不再有 grayscale 分支."""
    tfms = [transforms.Resize((imgsz, imgsz))]
    if augment:
        tfms.append(transforms.RandomHorizontalFlip())
        tfms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))
    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_RGB_MEAN, _IMAGENET_RGB_STD),
    ])
    return transforms.Compose(tfms)


class CarDDPretrainDataset(Dataset):
    """data/车/processed/pretrain/cardd_cls/{train,val}/{class}/*.jpg

    samples: (path, label, raw_class, is_scratch)  — is_scratch via _SCRATCH_LABELS lookup.
    """

    def __init__(self, pretrain_root, split='train', imgsz=640, augment=False,
                 val_split_strategy='random', val_split_k=5, val_split_fold=0,
                 **kwargs):
        # CarDD has official train/val split; ignore kfold kwargs but accept
        # them silently for API symmetry with the pooled-split datasets.
        root = os.path.join(pretrain_root, 'cardd_cls', split)
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CARDD_PRETRAIN_CLASSES)}
        for class_name in CARDD_PRETRAIN_CLASSES:
            for path in _list_jpgs(os.path.join(root, class_name)):
                is_sc = _SCRATCH_LABELS['cardd'].get(class_name, 0)
                self.samples.append((path, self.class_to_idx[class_name], class_name, is_sc))
        if not self.samples:
            raise FileNotFoundError('No CarDD pretrain data under {}'.format(root))
        self.imgsz = imgsz
        self.transform = _base_transforms(imgsz, augment=augment)
        self._task_name = 'cardd'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, raw_class, is_scratch = self.samples[index]
        img = self.transform(_open_rgb_for_pretrain(path, self.imgsz))
        return {'img': img, 'label': label, 'is_scratch': is_scratch,
                'task_name': self._task_name, 'raw_class': raw_class}


class CarPartsPretrainDataset(Dataset):
    """data/车/processed/pretrain/task_c_carparts_damage/<ClassTitle>/*.jpg

    8 个原生类: Scratch, Dent, Cracked, Paint chip, Flaking, Broken part, Missing part, Corrosion.
    没有官方 train/val split — 按子目录 + 15% 比例自动分 (与 ClassTitle 同步).
    """

    CLASSES = CARPARTS_PRETRAIN_CLASSES  # 8 个原生类名

    def __init__(self, pretrain_root, split='train', imgsz=640, augment=False, val_ratio=0.15, seed=42,
                 val_split_strategy='random', val_split_k=5, val_split_fold=0):
        root = os.path.join(pretrain_root, 'task_c_carparts_damage')
        k = val_split_k if val_split_strategy == 'stratified_kfold' else 0
        train_pool, val_pool = _split_pooled_subdirs(
            root, val_ratio=val_ratio, seed=seed, k=k, fold=val_split_fold)
        paths = train_pool if split == 'train' else val_pool
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASSES)}
        for path in paths:
            # 父目录即 class name (例如 pretrain/task_c_carparts_damage/Scratch/x.jpg)
            cls = os.path.basename(os.path.dirname(path))
            if cls not in self.class_to_idx:
                continue
            is_sc = _SCRATCH_LABELS['carparts'].get(cls, 0)
            self.samples.append((path, self.class_to_idx[cls], cls, is_sc))
        if not self.samples:
            raise FileNotFoundError('No carparts pretrain data under {}'.format(root))
        self.imgsz = imgsz
        self.transform = _base_transforms(imgsz, augment=augment)
        self._task_name = 'carparts'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, raw_class, is_scratch = self.samples[index]
        img = self.transform(_open_rgb_for_pretrain(path, self.imgsz))
        return {'img': img, 'label': label, 'is_scratch': is_scratch,
                'task_name': self._task_name, 'raw_class': raw_class}


class MinorDentPretrainDataset(Dataset):
    """data/车/processed/pretrain/task_b_minor_dent/dent/*.jpg

    单类 (dent), 文件名以 split 前缀 (train/valid/test) 区分原始 split.
    在 pretrain 里按 80/10/10 比例随机再分 (train/val).
    """

    CLASSES = ('dent',)

    def __init__(self, pretrain_root, split='train', imgsz=640, augment=False, val_ratio=0.10, seed=42,
                 val_split_strategy='random', val_split_k=5, val_split_fold=0):
        # 单类 task_b_minor_dent/dent/ — split_pooled_subdirs 直接以 dent 为父, 视为唯一子目录 dent
        dent_dir = os.path.join(pretrain_root, 'task_b_minor_dent', 'dent')
        # 直接对 dent 目录下文件分层, 因为我们只有 1 个类别 (dent)
        k = val_split_k if val_split_strategy == 'stratified_kfold' else 0
        train_pool, val_pool = _split_pooled_subdirs(
            os.path.join(pretrain_root, 'task_b_minor_dent'),
            val_ratio=val_ratio, seed=seed, k=k, fold=val_split_fold)
        # 上述 split 返回的是 dent_dir 里的全量 (sub=dent); 重新按 split 过滤:
        # 其实 _split_pooled_subdirs 已经把唯一子目录 dent 全部分好, 直接拿出来即可.
        paths = train_pool if split == 'train' else val_pool
        self.samples = []
        self.class_to_idx = {'dent': 0}
        for path in paths:
            is_sc = _SCRATCH_LABELS['minor_dent'].get('dent', 0)
            self.samples.append((path, 0, 'dent', is_sc))
        if not self.samples:
            raise FileNotFoundError('No minor_dent pretrain data under {}'.format(dent_dir))
        self.imgsz = imgsz
        self.transform = _base_transforms(imgsz, augment=augment)
        self._task_name = 'minor_dent'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, raw_class, is_scratch = self.samples[index]
        img = self.transform(_open_rgb_for_pretrain(path, self.imgsz))
        return {'img': img, 'label': label, 'is_scratch': is_scratch,
                'task_name': self._task_name, 'raw_class': raw_class}


class SyndcarPretrainDataset(Dataset):
    """data/车/processed/pretrain/task_e_syndcar/<class>/*.jpg

    4 类: sd_broken_glass / sd_crack / sd_scratch / sd_broken_light (3D 合成).
    子目录带 sd_ 前缀是因为 convert_syndcar.py 用了命名空间隔离 cp_/md_/sd_.
    按 15% 随机划 val.
    """

    CLASSES = ('sd_broken_glass', 'sd_crack', 'sd_scratch', 'sd_broken_light')

    def __init__(self, pretrain_root, split='train', imgsz=640, augment=False, val_ratio=0.15, seed=42,
                 val_split_strategy='random', val_split_k=5, val_split_fold=0):
        root = os.path.join(pretrain_root, 'task_e_syndcar')
        k = val_split_k if val_split_strategy == 'stratified_kfold' else 0
        train_pool, val_pool = _split_pooled_subdirs(
            root, val_ratio=val_ratio, seed=seed, k=k, fold=val_split_fold)
        paths = train_pool if split == 'train' else val_pool
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASSES)}
        # raw_class 在 _SCRATCH_LABELS 中不带 sd_ 前缀, 去掉前做查询
        _SCRATCH_LOOKUP = {
            'sd_broken_glass': 'broken_glass',
            'sd_crack':        'crack',
            'sd_scratch':      'scratch',
            'sd_broken_light': 'broken_light',
        }
        for path in paths:
            cls = os.path.basename(os.path.dirname(path))
            if cls not in self.class_to_idx:
                continue
            is_sc = _SCRATCH_LABELS['syndcar'].get(_SCRATCH_LOOKUP.get(cls, cls), 0)
            self.samples.append((path, self.class_to_idx[cls], cls, is_sc))
        if not self.samples:
            raise FileNotFoundError('No syndcar pretrain data under {}'.format(root))
        self.imgsz = imgsz
        self.transform = _base_transforms(imgsz, augment=augment)
        self._task_name = 'syndcar'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, raw_class, is_scratch = self.samples[index]
        img = self.transform(_open_rgb_for_pretrain(path, self.imgsz))
        return {'img': img, 'label': label, 'is_scratch': is_scratch,
                'task_name': self._task_name, 'raw_class': raw_class}
