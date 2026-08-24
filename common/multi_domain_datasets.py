"""Multi-source domain pretrain datasets: MVTEC + GC10 + NEU.

Each dataset returns (image, cls_label, cond_vec, product_id) where:
  - image:      (3, H, W) RGB tensor
  - cls_label:  int class label within the dataset's CE head space
  - cond_vec:   (n_conditions,) multi-hot condition vector (auto-computed)
  - product_id: (long) ProductEncoder slot id

Product ID space (ProductEncoder, 17 slots total):
  MVTEC: 0-14   (15 products: bottle/cable/.../zipper)
  GC10:  15     (steel plate, single product)
  NEU:   16     (steel strip, single product)

Classification label space (CE heads):
  MVTEC: 15-way product classification (each product = one class)
  GC10:  8-way  defect classification
  NEU:   6-way  defect classification

Source datasets:
  - MVTEC: data/MVTEC/<product>/{train,test}/<defect_type>/*.png
            → product_id is the canonical 0-14 index
            → cls_label = product_id (product classification)
            → all defect types within one product are merged into one class
  - GC10-DET: data/GC10-DET/cls_8class/<defect_name>/*.jpg
            → product_id = 15 (steel plate)
            → cls_label = defect type (8 classes)
            → ~2300 images
  - NEU-CLS: data/NEU-CLS/NEU-CLS/{Cr,In,Pa,PS,RS,Sc}_*.bmp
            → product_id = 16 (steel strip)
            → cls_label = defect type (6 classes)
            → 1800 images
"""

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _default_transform(img_size=224, augment=False):
    """ImageNet-style transform. ResNet18 expects 224."""
    tfms = [transforms.Resize((img_size, img_size))]
    if augment:
        tfms.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ])
    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return transforms.Compose(tfms)


def _estimate_condition(img_np):
    """从图像统计估算 7 维工况 one-hot (per-dim 0/1).

    img_np: H x W x 3 uint8 RGB.

    工况定义 (config/common.yaml 的 CONDITION_SPEC):
      c0 bright      : mean  > 180              (高亮)
      c1 dim         : mean  < 60               (低亮)
      c2 low_contrast: std   < 25               (低对比度)
      c3 noisy       : laplacian.var() > 500    (高噪声)
      c4 blur        : laplacian.var() < 50     (模糊)
      c5 color_cast  : max_ch_mean - min_ch_mean > 25  (色偏)
      c6 view        : 视角占位; 默认 0, 标注时由 dataset 层覆盖

    光源类型 (LED/荧光/自然光/红外) **不在此函数估算**, 应由 metadata 注入;
    若调用方需要 8 维 (含光源), 请扩展 CONDITION_SPEC.n_conditions 并在
    dataset 的 __getitem__ 里拼 metadata.

    Returns:
      torch.float32 tensor (7,)
    """
    cond = np.zeros(7, dtype=np.float32)
    if img_np.size == 0:
        return cond

    gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np
    mean_val = float(gray.mean())
    std_val = float(gray.std())

    # Brightness
    if mean_val > 180:
        cond[0] = 1.0
    elif mean_val < 60:
        cond[1] = 1.0

    # Contrast
    if std_val < 25:
        cond[2] = 1.0

    # Color cast
    if img_np.ndim == 3:
        ch_mean = img_np.reshape(-1, 3).mean(axis=0)
        if ch_mean.max() - ch_mean.min() > 25:
            cond[5] = 1.0

    # Noise / blur via Laplacian variance (downsample for speed)
    try:
        from scipy import ndimage
        sample = gray[::4, ::4]
        lap = ndimage.laplace(sample.astype(np.float32))
        lap_var = float(lap.var())
        if lap_var > 500:
            cond[3] = 1.0
        elif lap_var < 50:
            cond[4] = 1.0
    except Exception:
        pass

    # c6 view 占位 (默认 0; dataset 层若标注了视角, 可在 __getitem__ 里覆盖 cond[6])
    return cond


# ──────────────────────────────────────────────────────────
# MVTEC dataset
# ──────────────────────────────────────────────────────────

# 15 products, canonical index order matches ProductEncoder slots 0-14
MVTEC_PRODUCTS = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]
MVTEC_PRODUCT_TO_ID = {p: i for i, p in enumerate(MVTEC_PRODUCTS)}


class MVTECDataset(Dataset):
    """MVTEC pretrain dataset — 15-way product classification.

    cls_label = product_id ∈ [0, 14]  (同一产品的所有 defect 合为一类)
    product_id = cls_label             (恒等)

    Returns: (image, cls_label, cond_vec, product_id)
    """

    def __init__(self, root, split='train', img_size=224, augment=False,
                 transform=None, products=None):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.transform = transform or _default_transform(img_size, augment)
        self.samples = []  # (path, product_id)
        self.class_names = list(MVTEC_PRODUCTS)
        self.cls_to_idx = {c: i for i, c in enumerate(self.class_names)}

        products = products or MVTEC_PRODUCTS
        for product in products:
            pid = MVTEC_PRODUCT_TO_ID[product]
            if split == 'train':
                # Stage 0a: only 'good' samples from train/good/ (normal-only pretrain)
                good_dir = os.path.join(root, product, 'train', 'good')
                if not os.path.isdir(good_dir):
                    continue
                for f in sorted(os.listdir(good_dir)):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(good_dir, f), pid))
            elif split == 'train_defect':
                # Stage 0b: train/good + test/{all defect}
                good_dir = os.path.join(root, product, 'train', 'good')
                if os.path.isdir(good_dir):
                    for f in sorted(os.listdir(good_dir)):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(good_dir, f), pid))
                test_dir = os.path.join(root, product, 'test')
                if os.path.isdir(test_dir):
                    for defect in sorted(os.listdir(test_dir)):
                        defect_dir = os.path.join(test_dir, defect)
                        if not os.path.isdir(defect_dir):
                            continue
                        for f in sorted(os.listdir(defect_dir)):
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((os.path.join(defect_dir, f), pid))
            else:
                # Test: all defect types + good from test/
                test_dir = os.path.join(root, product, 'test')
                if not os.path.isdir(test_dir):
                    continue
                for defect in sorted(os.listdir(test_dir)):
                    defect_dir = os.path.join(test_dir, defect)
                    if not os.path.isdir(defect_dir):
                        continue
                    for f in sorted(os.listdir(defect_dir)):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(defect_dir, f), pid))

        if not self.samples:
            raise FileNotFoundError(f'No MVTEC samples under {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, product_id = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        cls_label = product_id  # cls_label = product_id for MVTEC
        return self.transform(img), cls_label, torch.from_numpy(cond), product_id


# ──────────────────────────────────────────────────────────
# GC10-DET dataset
# ──────────────────────────────────────────────────────────

GC10_CLASSES = [
    'chongkong', 'hanfeng', 'yueyawan', 'shuiban',
    'youban', 'siban', 'yiwu', 'yaozhed',
]
GC10_PRODUCT_ID = 15  # 钢板 (单产品)


class GC10Dataset(Dataset):
    """GC10-DET pretrain dataset — 8-way defect classification.

    product_id = 15 (固定: 钢板)
    cls_label  = defect type idx ∈ [0, 7]

    Returns: (image, cls_label, cond_vec, product_id)
    """

    def __init__(self, root, img_size=224, augment=False, transform=None):
        self.root = root
        self.img_size = img_size
        self.transform = transform or _default_transform(img_size, augment)
        self.samples = []  # (path, defect_label)
        self.class_names = list(GC10_CLASSES)
        self.cls_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.product_id = GC10_PRODUCT_ID

        cls_root = os.path.join(root, 'cls_8class')
        if not os.path.isdir(cls_root):
            raise FileNotFoundError(f'GC10 cls_8class not found: {cls_root}')

        # Map canonical name → actual folder name (with prefix index)
        folder_map = {
            'chongkong': '1_chongkong', 'hanfeng': '2_hanfeng',
            'yueyawan': '3_yueyawan', 'shuiban': '4_shuiban',
            'youban': '5_youban', 'siban': '6_siban',
            'yiwu': '7_yiwu', 'yaozhed': '10_yaozhed',
        }
        for cls_name, folder_name in folder_map.items():
            cls_dir = os.path.join(cls_root, folder_name)
            if not os.path.isdir(cls_dir):
                # fallback: prefix match
                for cand in os.listdir(cls_root):
                    if cls_name in cand:
                        cls_dir = os.path.join(cls_root, cand)
                        break
                else:
                    continue
            label = self.cls_to_idx[cls_name]
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(cls_dir, f), label))

        if not self.samples:
            raise FileNotFoundError(f'No GC10 samples under {cls_root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        return self.transform(img), cls_label, torch.from_numpy(cond), self.product_id


# ──────────────────────────────────────────────────────────
# NEU-CLS dataset
# ──────────────────────────────────────────────────────────

NEU_CLASSES = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
NEU_PRODUCT_ID = 16  # 带钢 (单产品)


# ──────────────────────────────────────────────────────────
# 车漆 CMPL 域 (新增: 域泛化实验)
# ──────────────────────────────────────────────────────────
# 3 个车漆数据源, 每个独立 product_id (PID 17/18/19)
# - CarDD 官方:   4 类 (scratch / dent / crack / glass_shatter)
# - Car-Parts Damage: 8 类细粒度
# - SynthCar (合成): 4 类
# 数据路径默认在 <root>/cardd_cls/{train,val}/<class>/*.jpg
#              <root>/task_c_carparts_damage/<ClassTitle>/*.jpg
#              <root>/task_e_syndcar/<class>/*.jpg

CARPAINT_CLASSES = {
    'cardd':     ['scratch', 'dent', 'crack', 'glass_shatter'],
    'carparts':  ['Scratch', 'Dent', 'Cracked', 'Paint chip',
                  'Flaking', 'Broken part', 'Missing part', 'Corrosion'],
    'syndcar':   ['broken_glass', 'crack', 'scratch', 'broken_light'],
}

CARPAINT_PRODUCT_TO_ID = {
    'cardd':    17,
    'carparts': 18,
    'syndcar':  19,
}
# 顺序, 便于 sampler/multi_domain_model 默认 n_products 推断
CARPAINT_PRODUCT_IDS = [17, 18, 19]  # → 续接 GC10(15)/NEU(16) 之后

# ──────────────────────────────────────────────────────────
# 航天真实数据 (新增 PID 20)
# ──────────────────────────────────────────────────────────
# 路径: data\航天\raw\Aero-engine_defect-detect_new
# 结构:
#   images/{train,val}/*.jpg
#   labels/{train,val}/*.txt (YOLO: cls_id x y w h)
# 注: train classes.txt = [scratch, dirty, stain, damage]
#     val classes.txt = [scratch, dot, crease, damage]
#     → 取两域交集 (YOLO id 0=scratch, 3=damage), 共 2 类, adaptive n_way.
AERO_REAL_PRODUCT_ID = 20  # PID 20: 航天真实数据
AERO_REAL_CLASSES = ['scratch', 'damage']

# ──────────────────────────────────────────────────────────
# MVTEC dropout (新增 PID 21) — 被踢出的产品重新入池
# ──────────────────────────────────────────────────────────
# 默认被剔: toothbrush (PID 11, n_classes=2 < n_way=5)
# 复用 MVTEC 原数据, 通过新 PID 21 + 2-way 评价模式进入 episode pool
MVTEC_DROPOUT_PRODUCT_ID = 21  # PID 21: MVTEC dropout (currently = toothbrush)
MVTEC_DROPOUT_PRODUCTS = ['toothbrush']  # 未来扩 zipper 等可追加

# 数据子目录布局 (相对 carpaint_root):
CARPAINT_DIR_LAYOUT = {
    'cardd':    'cardd_cls',                # 内部还有 train/val/<class>/
    'carparts': 'task_c_carparts_damage',   # <ClassTitle>/*.jpg
    'syndcar':  'task_e_syndcar',           # <class>/*.jpg
}

# 车漆 → 工况 (细粒度 view 类由后续 metadata 注入, 此处先固定 7 维零填充)


class CarPaintPretrainDataset(Dataset):
    """车漆 CMPL 域的 pretrain 数据集 (Phase 0 style).

    路径示例:
      <carpaint_root>/cardd_cls/train/scratch/001.jpg
      <carpaint_root>/task_c_carparts_damage/Scratch/x.jpg
      <carpaint_root>/task_e_syndcar/sd_scratch/y.jpg   ← 命名空间前缀保留

    自动识别所有 <carpaint_root>/<task_subdir> 下存在的任务. 任意子任务缺失
    时静默跳过, 不报错.

    product_id ∈ {17, 18, 19}; cls_label = 子任务内的类 idx (从 0 起, 仅在该 task 内一致).
    """

    def __init__(self, root: str, task: str, img_size: int = 224,
                 augment: bool = False, transform=None):
        assert task in CARPAINT_PRODUCT_TO_ID, f'Unknown car-paint task: {task}'
        self.root = root
        self.task = task
        self.img_size = img_size
        self.transform = transform or _default_transform(img_size, augment)
        self.product_id = CARPAINT_PRODUCT_TO_ID[task]
        self.class_names = []
        self.cls_to_idx = {}
        self.samples = []  # (path, cls_idx)

        task_root = os.path.join(root, CARPAINT_DIR_LAYOUT[task])
        if not os.path.isdir(task_root):
            return

        # cardd_cls/{train,val}/<class>/*.jpg; 其它 task/<class>/*.jpg
        if task == 'cardd':
            for split in ('train', 'val'):
                split_root = os.path.join(task_root, split)
                if not os.path.isdir(split_root):
                    continue
                for cls_name in sorted(os.listdir(split_root)):
                    cls_dir = os.path.join(split_root, cls_name)
                    if not os.path.isdir(cls_dir):
                        continue
                    if cls_name not in self.cls_to_idx:
                        self.cls_to_idx[cls_name] = len(self.class_names)
                        self.class_names.append(cls_name)
                    for f in sorted(os.listdir(cls_dir)):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('._'):
                            self.samples.append((os.path.join(cls_dir, f),
                                                 self.cls_to_idx[cls_name]))
        else:
            for cls_name in sorted(os.listdir(task_root)):
                cls_dir = os.path.join(task_root, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                # 还原 syndcar 命名空间前缀 (sd_/cp_/md_)
                lookup = cls_name
                if task == 'syndcar' and lookup.startswith('sd_'):
                    lookup = lookup[3:]
                if lookup not in self.cls_to_idx:
                    self.cls_to_idx[lookup] = len(self.class_names)
                    self.class_names.append(lookup)
                for f in sorted(os.listdir(cls_dir)):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('._'):
                        self.samples.append((os.path.join(cls_dir, f),
                                             self.cls_to_idx[lookup]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        return self.transform(img), cls_label, torch.from_numpy(cond), self.product_id


class NEUDataset(Dataset):
    """NEU-CLS pretrain dataset — 6-way defect classification.

    product_id = 16 (固定: 带钢)
    cls_label  = defect type idx ∈ [0, 5]

    Returns: (image, cls_label, cond_vec, product_id)
    """

    def __init__(self, root, img_size=224, augment=False, transform=None):
        self.root = root
        self.img_size = img_size
        self.transform = transform or _default_transform(img_size, augment)
        self.samples = []  # (path, defect_label)
        self.class_names = list(NEU_CLASSES)
        self.cls_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.product_id = NEU_PRODUCT_ID

        for f in sorted(os.listdir(root)):
            if not f.lower().endswith('.bmp'):
                continue
            prefix = f.split('_')[0]
            if prefix in self.cls_to_idx:
                self.samples.append((os.path.join(root, f), self.cls_to_idx[prefix]))

        if not self.samples:
            raise FileNotFoundError(f'No NEU samples under {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        return self.transform(img), cls_label, torch.from_numpy(cond), self.product_id


# ──────────────────────────────────────────────────────────
# Combined loader helper
# ──────────────────────────────────────────────────────────

import torch


def build_multi_domain_loaders(
    mvtec_root,
    gc10_root,
    neu_root,
    batch_size=8,
    img_size=224,
    augment=True,
    num_workers=0,
    mvtec_split='train',
):
    """Build three dataloaders + per-task specs.

    Args:
        mvtec_split: 'train' (only normal, 3629) — 唯一合法值.
            ⚠️ 必须严格只用 train/good/! 否则 test defect 会 leak 到 pretrain,
            导致 Stage 1 episode 评估虚高 (model 在 pretrain 时已经看过 query 图).
    Returns:
        loaders:    {'mvtec': DataLoader, 'gc10': DataLoader, 'neu': DataLoader}
        task_specs: [('mvtec', 15), ('gc10', 8), ('neu', 6)]
                    (每个数据集的 CE 头类别数)
        cls_counts: {'mvtec': [15 ints], 'gc10': [8 ints], 'neu': [6 ints]}
                    (每个数据集各类的样本数, 用于 LDAM/class weight)
        product_ids: {'mvtec': 0-14 (per-class), 'gc10': 15, 'neu': 16}
                     (每个 CE 类的 product slot)
    """
    if mvtec_split != 'train':
        raise ValueError(
            f"mvtec_split='{mvtec_split}' is FORBIDDEN — would leak test defects "
            "into pretrain and inflate Stage 1 episode accuracy.")
    mvtec_ds = MVTECDataset(mvtec_root, split='train', img_size=img_size, augment=augment)
    gc10_ds = GC10Dataset(gc10_root, img_size=img_size, augment=augment)
    neu_ds = NEUDataset(neu_root, img_size=img_size, augment=augment)

    loaders = {
        'mvtec': torch.utils.data.DataLoader(
            mvtec_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True),
        'gc10': torch.utils.data.DataLoader(
            gc10_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True),
        'neu': torch.utils.data.DataLoader(
            neu_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True),
    }

    def _counts(ds, kind):
        """kind='mvtec' → count by product_id; kind in {gc10,neu} → count by cls_label."""
        if kind == 'mvtec':
            k = 15
            counts = [0] * k
            for _, pid in ds.samples:
                counts[pid] += 1
            return counts
        else:
            k = len(ds.class_names)
            counts = [0] * k
            for _, lbl in ds.samples:
                counts[lbl] += 1
            return counts

    task_specs = [
        ('mvtec', 15),  # 15 products
        ('gc10', 8),
        ('neu', 6),
    ]
    cls_counts = {
        'mvtec': _counts(mvtec_ds, 'mvtec'),
        'gc10': _counts(gc10_ds, 'gc10'),
        'neu': _counts(neu_ds, 'neu'),
    }
    product_ids = {
        'mvtec': list(range(15)),  # cls i → product i
        'gc10': [GC10_PRODUCT_ID] * 8,  # all defects share product 15
        'neu': [NEU_PRODUCT_ID] * 6,
    }
    return loaders, task_specs, cls_counts, product_ids