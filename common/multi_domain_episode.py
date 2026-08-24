"""Multi-source domain episode sampler for N-way K-shot meta-train.

构造逻辑 (per iter):
  1) 随机选一个 source ∈ {mvtec_test, gc10, neu}
     (MVTEC train 因只有 good 不参与 episode, 仅参与 pretrain)
  2) 随机选一个 product (在 MVTEC 内; GC10/NEU 单一 product = 固定 id)
  3) 随机选 N 个 defect type (在该 product/source 内部)
     - 默认 N = n_way (e.g. 5)
     - adaptive: N = min(n_way, n_classes_in_product)  // bottle 自动 4-way
  4) 每个 defect 抽 K shot support + Q query
     - 从 split pool 抽 (默认 meta_train, 评测时切 meta_test)
  5) 同步抽工况 (estimate from image stats)
  6) 返回拼好的 episode tensors + 元信息

Returns:
  support_rgb: (N*K, 3, H, W)
  support_y  : (N*K,)   ∈ [0, N)
  support_pid: (N*K,)   product_id
  support_cond:(N*K, n_conditions)
  query_rgb  : (N*Q, 3, H, W)
  query_y    : (N*Q,)   ∈ [0, N)
  query_pid  : (N*Q,)   product_id
  query_cond : (N*Q, n_conditions)
  source_name: str
  product_id : int (the sampled product)
  n_way      : int (实际采到的, 可能 < config.n_way 若 adaptive)
"""

from __future__ import annotations

import os
import json
import random
from typing import Optional, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .multi_domain_datasets import (
    MVTEC_PRODUCTS, MVTEC_PRODUCT_TO_ID,
    GC10_PRODUCT_ID, NEU_PRODUCT_ID,
    AERO_REAL_PRODUCT_ID, MVTEC_DROPOUT_PRODUCT_ID,
    MVTEC_DROPOUT_PRODUCTS,
    _estimate_condition, _default_transform,
)


# ──────────────────────────────────────────────────────────
# Split files
# ──────────────────────────────────────────────────────────

SPLITS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'splits',
)


def _load_split(name: str) -> Optional[Dict[str, List[str]]]:
    """加载 split 文件; 不存在则返回 None."""
    p = os.path.join(SPLITS_DIR, name)
    if not os.path.isfile(p):
        return None
    with open(p, 'r') as f:
        return json.load(f)


def _build_mvtec_class_split_filter(split_data, root):
    """split_data 是 mvtec_test_split.json 内容.
    返回 set[(product, defect_name, file_basename)] 表示 '在 meta_train pool' 的样本.
    """
    if split_data is None:
        return None
    train_set = set()
    for rel in split_data.get('meta_train', []):
        # rel = 'cable/test/bent_wire/000.png'
        parts = rel.split('/')
        product, defect, fname = parts[0], parts[2], parts[3]
        train_set.add((product, defect, fname))
    return train_set


def _build_gc10_class_split_filter(split_data):
    """split_data 是 gc10_split.json 内容.
    返回 set['class_folder/file_basename'] 表示 '在 meta_train pool'.
    """
    if split_data is None:
        return None
    return set(split_data.get('meta_train', []))


def _build_neu_class_split_filter(split_data):
    """split_data 是 neu_split.json 内容.
    返回 set['file_basename'] 表示 '在 meta_train pool'.
    """
    if split_data is None:
        return None
    return set(split_data.get('meta_train', []))


# ──────────────────────────────────────────────────────────
# Episode datasets
# ──────────────────────────────────────────────────────────


class MVTECTestEpisodeDataset(Dataset):
    """MVTEC test split exposed for episode-level indexing.

    按 (product_id, defect_type) 索引样本, 返回 (img, defect_local_idx, product_id).
    其中 defect_local_idx ∈ [0, n_defect_types_in_product) 本地化标签 (跨产品不通用).

    split: 'meta_train' | 'meta_test' | 'all'
       若 split='meta_train', 仅从 meta_train pool 返回样本路径.
       若 split='meta_test', 仅从 meta_test pool 返回样本路径.
       若 split='all' 或 split 文件不存在, 返回所有样本.
    """

    def __init__(self, root: str, img_size: int = 224, augment: bool = False,
                 transform=None, split: str = 'meta_train',
                 skip_products: Optional[List[str]] = None):
        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.transform = transform or _default_transform(img_size, augment)
        self.split = split

        # 加载 split filter
        split_data = _load_split('mvtec_test_split.json')
        if split_data is not None:
            self.usable_products = set(split_data.get('usable_products', []))
            self.adaptive_products = set(split_data.get('adaptive_products', []))
        else:
            self.usable_products = set(MVTEC_PRODUCTS) - {'toothbrush'}
            self.adaptive_products = set()
        # 用户额外排除
        if skip_products:
            self.usable_products -= set(skip_products)

        train_filter = _build_mvtec_class_split_filter(split_data, root) \
            if split_data else None

        # self.classes[(pid, defect_name)] = [sample_paths] (按 split 过滤后)
        self.classes = {}        # (pid, lid) -> [paths]
        self.classes_full = {}   # (pid, lid) -> [all paths, 不分 split]
        self.samples = []        # [{pid, lid, path}]
        self.product_defects = {}  # pid -> [lid]
        self.defect_idx_to_name = {}
        self.defect_name_to_idx = {}

        for product in MVTEC_PRODUCTS:
            if product not in self.usable_products:
                continue
            pid = MVTEC_PRODUCT_TO_ID[product]
            test_dir = os.path.join(root, product, 'test')
            if not os.path.isdir(test_dir):
                continue
            defect_names = sorted(d for d in os.listdir(test_dir)
                                  if os.path.isdir(os.path.join(test_dir, d)))
            local = {name: i for i, name in enumerate(defect_names)}
            self.product_defects[pid] = []
            for defect_name in defect_names:
                defect_dir = os.path.join(test_dir, defect_name)
                all_paths = []
                train_paths = []
                for f in sorted(os.listdir(defect_dir)):
                    if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    full = os.path.join(defect_dir, f)
                    all_paths.append(full)
                    if train_filter is None or (product, defect_name, f) in train_filter:
                        train_paths.append(full)
                if not all_paths:
                    continue
                lid = local[defect_name]
                # 选 split
                if split == 'all' or train_filter is None:
                    picked = all_paths
                elif split == 'meta_train':
                    picked = train_paths
                elif split == 'meta_test':
                    picked = [p for p in all_paths if p not in train_paths]
                else:
                    raise ValueError(f'Unknown split: {split}')

                self.classes_full[(pid, lid)] = all_paths
                if picked:
                    self.classes[(pid, lid)] = picked
                    self.product_defects[pid].append(lid)
                self.defect_idx_to_name[(pid, lid)] = defect_name
                self.defect_name_to_idx[(product, defect_name)] = lid
                for p in picked:
                    self.samples.append({'pid': pid, 'lid': lid, 'path': p})

    def num_defects(self, pid: int) -> int:
        return len(self.product_defects.get(pid, []))

    def is_adaptive(self, pid: int) -> bool:
        """该产品的 defect 类数 < n_way_max 时, 需要 adaptive n_way."""
        return pid in self.adaptive_products

    def sample_paths(self, pid: int, lid: int):
        return self.classes.get((pid, lid), [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        return self.transform(img), s['lid'], s['pid'], torch.from_numpy(cond)


class GC10EpisodeDataset(Dataset):
    """GC10 8-way 缺陷, product_id 固定 15.

    split: 'meta_train' | 'meta_test' | 'all'
    """

    def __init__(self, root, img_size=224, augment=False, transform=None,
                 split: str = 'meta_train'):
        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.transform = transform or _default_transform(img_size, augment)
        self.split = split

        split_data = _load_split('gc10_split.json')
        train_filter = _build_gc10_class_split_filter(split_data) \
            if split_data else None

        self.samples = []  # [(path, defect_label)]
        self.class_names = []
        from .multi_domain_datasets import GC10_CLASSES
        cls_root = os.path.join(root, 'cls_8class')
        if not os.path.isdir(cls_root):
            raise FileNotFoundError(cls_root)
        folder_map = {
            'chongkong': '1_chongkong', 'hanfeng': '2_hanfeng',
            'yueyawan': '3_yueyawan', 'shuiban': '4_shuiban',
            'youban': '5_youban', 'siban': '6_siban',
            'yiwu': '7_yiwu', 'yaozhed': '10_yaozhed',
        }
        self.cls_to_idx = {}
        for cls_name, folder_name in folder_map.items():
            cls_dir = os.path.join(cls_root, folder_name)
            if not os.path.isdir(cls_dir):
                for cand in os.listdir(cls_root):
                    if cls_name in cand:
                        cls_dir = os.path.join(cls_root, cand); break
                else: continue
            self.cls_to_idx[cls_name] = len(self.class_names)
            self.class_names.append(cls_name)
            for f in sorted(os.listdir(cls_dir)):
                if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                rel = f'{folder_name}/{f}'
                if split == 'all' or train_filter is None \
                        or split == 'meta_train':
                    if train_filter is not None and rel not in train_filter:
                        # skip from train pool
                        if split == 'meta_train':
                            continue
                if split == 'meta_test':
                    if train_filter is not None and rel in train_filter:
                        continue
                self.samples.append((os.path.join(cls_dir, f),
                                     self.cls_to_idx[cls_name]))

    def num_defects(self, pid: int = GC10_PRODUCT_ID) -> int:
        return len(self.class_names)

    def is_adaptive(self, pid: int = GC10_PRODUCT_ID) -> bool:
        return False  # GC10 8 类足够 5-way

    def sample_paths(self, pid: int = GC10_PRODUCT_ID, lid: int = 0):
        return [p for p, l in self.samples if l == lid]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lid = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        return self.transform(img), lid, GC10_PRODUCT_ID, torch.from_numpy(cond)


class NEUEpisodeDataset(Dataset):
    """NEU 6-way 缺陷, product_id 固定 16.

    split: 'meta_train' | 'meta_test' | 'all'
    """

    def __init__(self, root, img_size=224, augment=False, transform=None,
                 split: str = 'meta_train'):
        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.transform = transform or _default_transform(img_size, augment)
        self.split = split

        split_data = _load_split('neu_split.json')
        train_filter = _build_neu_class_split_filter(split_data) \
            if split_data else None

        self.samples = []
        from .multi_domain_datasets import NEU_CLASSES
        self.class_names = list(NEU_CLASSES)
        self.cls_to_idx = {c: i for i, c in enumerate(self.class_names)}
        for f in sorted(os.listdir(root)):
            if not f.lower().endswith('.bmp'): continue
            if split == 'all' or train_filter is None:
                pass
            elif split == 'meta_train':
                if f not in train_filter:
                    continue
            elif split == 'meta_test':
                if f in train_filter:
                    continue
            prefix = f.split('_')[0]
            if prefix in self.cls_to_idx:
                self.samples.append((os.path.join(root, f),
                                     self.cls_to_idx[prefix]))

    def num_defects(self, pid: int = NEU_PRODUCT_ID) -> int:
        return len(self.class_names)

    def is_adaptive(self, pid: int = NEU_PRODUCT_ID) -> bool:
        return False

    def sample_paths(self, pid: int = NEU_PRODUCT_ID, lid: int = 0):
        return [p for p, l in self.samples if l == lid]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lid = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        return self.transform(img), lid, NEU_PRODUCT_ID, torch.from_numpy(cond)


class CarPaintEpisodeDataset(Dataset):
    """车漆 CMPL 域的 episode dataset — 跨 cardd / carparts / syndcar.

    每个 task 独立 product_id (17/18/19), 内部按 (product_id, class_name) 索引.
    一个 ds 实例承载所有 task, 类标签是 task 内 local idx (跨 task 不通用,
    但 episode 采样永远在同一 task 内, 故不影响).

    split: 'meta_train' | 'meta_test' | 'all'
       若 split='meta_train', 只从 carpaint_split.json 的 meta_train pool 返回.
       若 split 文件不存在, 默认按文件名前缀 hash 80/20 切.

    注意: 这是 *新增* dataset,  *完全独立于* MVTEC/GC10/NEU. 通过构造时传
    carpaint_root 选择启用. 路径不存在时 dataset 自动 zero-sample, 不报错.
    """

    def __init__(self, root: str, img_size: int = 224, augment: bool = False,
                 transform=None, split: str = 'meta_train',
                 task_filter: Optional[List[str]] = None):
        from .multi_domain_datasets import (
            CARPAINT_PRODUCT_TO_ID, CARPAINT_DIR_LAYOUT,
        )
        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.transform = transform or _default_transform(img_size, augment)
        self.split = split

        # 加载 split (默认 split 文件名: carpaint_split.json)
        split_data = _load_split('carpaint_split.json')
        train_filter = None
        if split_data is not None:
            # train_filter = set(task, class, fname)
            train_filter = set()
            for rel in split_data.get('meta_train', []):
                # rel = 'cardd/scratch/001.jpg'
                parts = rel.split('/')
                if len(parts) >= 3:
                    train_filter.add((parts[0], parts[1], parts[2]))

        # self.classes[(pid, lid)] = [paths]
        # self.product_defects[pid] = [lid]
        # self.task_name[pid] = str
        self.classes = {}
        self.classes_full = {}
        self.samples = []
        self.product_defects = {}
        self.task_name = {}

        all_tasks = list(CARPAINT_PRODUCT_TO_ID.keys())
        if task_filter is not None:
            all_tasks = [t for t in all_tasks if t in task_filter]

        for task in all_tasks:
            pid = CARPAINT_PRODUCT_TO_ID[task]
            sub = CARPAINT_DIR_LAYOUT[task]
            task_root = os.path.join(root, sub) if root else None
            if not task_root or not os.path.isdir(task_root):
                # 数据未就绪 → 静默 skip
                continue

            self.task_name[pid] = task

            # 枚举该 task 下的类目录
            if task == 'cardd':
                # 嵌套 train/val/<class>
                class_dirs = []
                for split_name in ('train', 'val'):
                    split_root = os.path.join(task_root, split_name)
                    if not os.path.isdir(split_root):
                        continue
                    for cls_name in sorted(os.listdir(split_root)):
                        cls_dir = os.path.join(split_root, cls_name)
                        if os.path.isdir(cls_dir):
                            class_dirs.append((cls_name, cls_dir))
            else:
                class_dirs = []
                for cls_name in sorted(os.listdir(task_root)):
                    cls_dir = os.path.join(task_root, cls_name)
                    if os.path.isdir(cls_dir):
                        # syndcar 命名空间还原 (sd_/cp_/md_ 前缀)
                        lookup = cls_name
                        if task == 'syndcar' and lookup.startswith('sd_'):
                            lookup = lookup[3:]
                        class_dirs.append((lookup, cls_dir))

            if not class_dirs:
                continue

            self.product_defects[pid] = []
            for lid, (cls_name, cls_dir) in enumerate(class_dirs):
                all_paths = []
                train_paths = []
                for f in sorted(os.listdir(cls_dir)):
                    if not f.lower().endswith(('.jpg', '.jpeg', '.png')) or f.startswith('._'):
                        continue
                    full = os.path.join(cls_dir, f)
                    all_paths.append(full)
                    if train_filter is None:
                        train_paths.append(full)
                    elif (task, cls_name, f) in train_filter:
                        train_paths.append(full)

                if not all_paths:
                    continue

                if split == 'all' or train_filter is None:
                    picked = all_paths
                elif split == 'meta_train':
                    picked = train_paths
                elif split == 'meta_test':
                    picked = [p for p in all_paths if p not in set(train_paths)]
                else:
                    raise ValueError(f'Unknown split: {split}')

                self.classes_full[(pid, lid)] = all_paths
                if picked:
                    self.classes[(pid, lid)] = picked
                    self.product_defects[pid].append(lid)
                    for p in picked:
                        self.samples.append({'pid': pid, 'lid': lid, 'path': p})

    def num_defects(self, pid: int) -> int:
        return len(self.product_defects.get(pid, []))

    def is_adaptive(self, pid: int) -> bool:
        """cardd=4 类, syndcar=4 类 < n_way=5, 视为 adaptive."""
        from .multi_domain_datasets import CARPAINT_PRODUCT_TO_ID
        adaptive_pids = {CARPAINT_PRODUCT_TO_ID['cardd'],
                         CARPAINT_PRODUCT_TO_ID['syndcar']}
        return pid in adaptive_pids

    def sample_paths(self, pid: int, lid: int):
        return self.classes.get((pid, lid), [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        return self.transform(img), s['lid'], s['pid'], torch.from_numpy(cond)


# ──────────────────────────────────────────────────────────
# 航天真实数据 (PID 20)
# ──────────────────────────────────────────────────────────

def _parse_voc_xml(xml_path: str) -> Optional[str]:
    """从 VOC .xml 提取第一个 <object><name>. 失败返回 None."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name_el = obj.find('name')
            if name_el is not None and name_el.text:
                return name_el.text.strip()
        return None
    except Exception:
        return None


class AeroRealEpisodeDataset(Dataset):
    """航天真实数据集 (PID 20).

    路径布局:
      <aero_root>/images/{train,val}/*.jpg
      <aero_root>/labels/{train,val}/*.txt (YOLO: 'cls_id x y w h')

    注: 数据集 train 和 val 的 classes.txt 不一致 (train: scratch/dirty/stain/damage,
    val: scratch/dot/crease/damage). 为保证 train/val 类一致, 此处只保留两域都
    有的类 — scratch (YOLO id=0) 和 damage (YOLO id=3).
    共 2 类, 自适应 n_way=2 (也支持 n_way=5 时降级为 2-way).
    类名: scratch, damage.
    """
    AERO_REAL_PRODUCT_ID = 20
    # 限定统一类 (train ∩ val) — scratch, damage
    AERO_REAL_CLASSES = ['scratch', 'damage']
    # YOLO id → cls_to_idx 映射 (id=0 → scratch=0, id=3 → damage=1, 其它跳过)
    AERO_YOLO_ID_TO_NAME = {0: 'scratch', 3: 'damage'}

    def __init__(self, root: str, img_size: int = 224, augment: bool = False,
                 transform=None, split: str = 'meta_train'):
        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.transform = transform or _default_transform(img_size, augment)
        self.split = split
        self.product_id = self.AERO_REAL_PRODUCT_ID
        self.class_names = list(self.AERO_REAL_CLASSES)
        self.cls_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.task_name = {20: 'aero_real'}
        self.classes = {}
        self.samples = []
        self.product_defects = {}

        # split 文件优先
        split_data = _load_split('aero_real_split.json')
        train_filter = None
        if split_data is not None:
            train_filter = set(split_data.get('meta_train', []))

        img_root = os.path.join(root, 'images')
        lbl_root = os.path.join(root, 'labels')
        if not os.path.isdir(img_root):
            return

        local = {}

        for sub_split in ('train', 'val'):
            sub_img = os.path.join(img_root, sub_split)
            sub_lbl = os.path.join(lbl_root, sub_split)
            if not os.path.isdir(sub_img):
                continue

            for img_file in sorted(os.listdir(sub_img)):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')) or img_file.startswith('._'):
                    continue
                img_path = os.path.join(sub_img, img_file)
                base, _ = os.path.splitext(img_file)
                txt_path = os.path.join(sub_lbl, base + '.txt')
                if not os.path.isfile(txt_path):
                    continue

                # 解析 YOLO .txt 第一行的 class id
                yolo_id = None
                try:
                    with open(txt_path, encoding='utf-8') as f:
                        first = f.readline().strip().split()
                        if first and first[0].isdigit():
                            yolo_id = int(first[0])
                except Exception:
                    continue
                if yolo_id is None:
                    continue
                cls_name = self.AERO_YOLO_ID_TO_NAME.get(yolo_id)
                if cls_name is None:
                    continue
                lid = self.cls_to_idx[cls_name]

                rel = f'{sub_split}/{img_file}'
                is_train_in_split = (train_filter is not None) and (rel in train_filter)
                # 缺省规则: 原 train -> meta_train, 原 val -> meta_test
                if split == 'all':
                    picked = True
                elif split == 'meta_train':
                    if train_filter is not None:
                        picked = is_train_in_split
                    else:
                        picked = (sub_split == 'train')
                elif split == 'meta_test':
                    if train_filter is not None:
                        picked = not is_train_in_split
                    else:
                        picked = (sub_split == 'val')
                else:
                    raise ValueError(f'Unknown split: {split}')

                if not picked:
                    continue
                local[lid] = True
                self.classes.setdefault((self.product_id, lid), []).append(img_path)
                self.samples.append({'pid': self.product_id, 'lid': lid, 'path': img_path})

        for lid in sorted(local.keys()):
            self.product_defects.setdefault(self.product_id, []).append(lid)

    def num_defects(self, pid: int = 20) -> int:
        return len(self.product_defects.get(pid, []))

    def is_adaptive(self, pid: int = 20) -> bool:
        # 只 2 类, 始终 adaptive
        return True

    def sample_paths(self, pid: int = 20, lid: int = 0):
        return self.classes.get((pid, lid), [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        return self.transform(img), s['lid'], s['pid'], torch.from_numpy(cond)


# ──────────────────────────────────────────────────────────
# MVTEC dropout (PID 21)
# ──────────────────────────────────────────────────────────

class MVTecDropoutEpisodeDataset(Dataset):
    """MVTEC 被剔除产品重新入池 (PID 21).

    复用 MVTEC 原始 test/ 目录结构, 但重新注册为新 PID 21.
    默认产品列表: ['toothbrush'] (2-way: defective / good).
    """
    MVTEC_DROPOUT_PRODUCT_ID = 21

    def __init__(self, mvtec_root: str, img_size: int = 224, augment: bool = False,
                 transform=None, split: str = 'meta_train',
                 products: Optional[List[str]] = None):
        from .multi_domain_datasets import MVTEC_DROPOUT_PRODUCTS
        products = products or MVTEC_DROPOUT_PRODUCTS
        self.mvtec_root = mvtec_root
        self.img_size = img_size
        self.augment = augment
        self.transform = transform or _default_transform(img_size, augment)
        self.split = split
        self.product_id = self.MVTEC_DROPOUT_PRODUCT_ID
        self.task_name = {21: 'mvtec_dropout'}
        self.classes = {}
        self.samples = []
        self.product_defects = {}
        self.product_names = []

        split_data = _load_split('mvtec_test_split.json')
        train_filter = None
        if split_data is not None:
            # mvtec_test_split entry = 'toothbrush/test/defective/019.png' (4 段)
            # file-level 切: key = (prod, 'test', defect_name, fname)
            train_filter = set()
            for rel in split_data.get('meta_train', []):
                parts = rel.split('/')
                if len(parts) >= 4:
                    train_filter.add(tuple(parts[:4]))

        local_by_product = {}
        for prod in products:
            test_dir = os.path.join(mvtec_root, prod, 'test')
            if not os.path.isdir(test_dir):
                continue
            self.product_names.append(prod)
            defect_dirs = sorted(d for d in os.listdir(test_dir)
                                  if os.path.isdir(os.path.join(test_dir, d)))
            local = {name: i for i, name in enumerate(defect_dirs)}
            local_by_product[prod] = local
            for defect_name in defect_dirs:
                defect_dir = os.path.join(test_dir, defect_name)
                # 收集 (fname, full_path) 列表
                files = sorted([f for f in os.listdir(defect_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if not files:
                    continue
                lid = local[defect_name]
                picked = []
                for f in files:
                    full = os.path.join(defect_dir, f)
                    in_train = (train_filter is None or
                                (prod, 'test', defect_name, f) in train_filter)
                    if split == 'all':
                        picked.append(full)
                    elif split == 'meta_train' and in_train:
                        picked.append(full)
                    elif split == 'meta_test' and not in_train:
                        picked.append(full)

                if picked:
                    self.classes.setdefault((self.product_id, lid), []).extend(picked)
                    for p in picked:
                        self.samples.append({'pid': self.product_id, 'lid': lid, 'path': p})

        all_lids = set()
        for local in local_by_product.values():
            all_lids.update(local.values())
        for lid in sorted(all_lids):
            if (self.product_id, lid) in self.classes:
                self.product_defects.setdefault(self.product_id, []).append(lid)

    def num_defects(self, pid: int = 21) -> int:
        return len(self.product_defects.get(pid, []))

    def is_adaptive(self, pid: int = 21) -> bool:
        return True

    def sample_paths(self, pid: int = 21, lid: int = 0):
        return self.classes.get((pid, lid), [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        img_np = np.array(img)
        cond = _estimate_condition(img_np)
        return self.transform(img), s['lid'], s['pid'], torch.from_numpy(cond)


# ──────────────────────────────────────────────────────────
# Episode sampler (PyTorch Dataset wrapper)
# ──────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────
# Episode sampler (PyTorch Dataset wrapper)
# ──────────────────────────────────────────────────────────


class MultiDomainEpisodeSampler:
    """采样一个 N-way K-shot episode, 跨 (source, product, defect_types).

    支持 per-product adaptive n_way:
      对 adaptive products (e.g. bottle 4 类), 实际 n_way = min(config.n_way, n_classes).

    支持 split 切换:
      sampler.sample(split='meta_train')  - 训练用, 只从 train pool 抽
      sampler.sample(split='meta_test')   - 评测用, 只从 test pool 抽
    """

    def __init__(self,
                 mvtec_ds: MVTECTestEpisodeDataset,
                 gc10_ds: GC10EpisodeDataset,
                 neu_ds: NEUEpisodeDataset,
                 carpaint_ds: Optional['CarPaintEpisodeDataset'] = None,
                 aero_ds: Optional['AeroRealEpisodeDataset'] = None,
                 mvtec_dropout_ds: Optional['MVTecDropoutEpisodeDataset'] = None,
                 n_way: int = 5,
                 k_shot: int = 5,
                 q_query: int = 5,
                 augment: bool = True):
        self.mvtec = mvtec_ds
        self.gc10 = gc10_ds
        self.neu = neu_ds
        self.carpaint = carpaint_ds
        self.aero = aero_ds
        self.mvtec_dropout = mvtec_dropout_ds
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.augment = augment
        self._transform = mvtec_ds.transform

        # product → (source_name, ds)
        self.product_to_source = {}
        self.product_n_classes = {}
        for pid in mvtec_ds.product_defects:
            n = mvtec_ds.num_defects(pid)
            if n >= 2:
                self.product_to_source[pid] = ('mvtec', mvtec_ds)
                self.product_n_classes[pid] = n
        # GC10 / NEU 始终可用
        for pid, ds in [(GC10_PRODUCT_ID, gc10_ds), (NEU_PRODUCT_ID, neu_ds)]:
            n = ds.num_defects(pid)
            if n >= 2:
                self.product_to_source[pid] = (
                    'gc10' if pid == GC10_PRODUCT_ID else 'neu', ds)
                self.product_n_classes[pid] = n
        # 车漆 CMPL
        if carpaint_ds is not None:
            for pid in carpaint_ds.product_defects:
                n = carpaint_ds.num_defects(pid)
                if n >= 2:
                    task = carpaint_ds.task_name.get(pid, 'carpaint')
                    self.product_to_source[pid] = (task, carpaint_ds)
                    self.product_n_classes[pid] = n
        # 航天真实 (PID 20)
        if aero_ds is not None:
            for pid in aero_ds.product_defects:
                n = aero_ds.num_defects(pid)
                if n >= 2:
                    self.product_to_source[pid] = ('aero_real', aero_ds)
                    self.product_n_classes[pid] = n
        # MVTEC dropout (PID 21)
        if mvtec_dropout_ds is not None:
            for pid in mvtec_dropout_ds.product_defects:
                n = mvtec_dropout_ds.num_defects(pid)
                if n >= 2:
                    self.product_to_source[pid] = ('mvtec_dropout', mvtec_dropout_ds)
                    self.product_n_classes[pid] = n

        self.usable_products = sorted(self.product_to_source.keys())
        print(f'[MultiDomainEpisodeSampler] n_way_max={n_way} k_shot={k_shot} '
              f'q_query={q_query}, split="{mvtec_ds.split}"')
        print(f'  usable_products={len(self.usable_products)}')
        for pid in self.usable_products:
            src, ds = self.product_to_source[pid]
            n = self.product_n_classes[pid]
            eff_n = min(n_way, n)
            adapt = ' (adaptive)' if ds.is_adaptive(pid) else ''
            print(f'    product {pid:2d} ({src}): n_classes={n} effective_n_way={eff_n}{adapt}')

    def effective_n_way(self, product_id: int) -> int:
        return min(self.n_way, self.product_n_classes.get(product_id, 2))

    def sample(self, source_filter: Optional[str] = None,
               split: Optional[str] = None) -> dict:
        """随机抽一个 episode.

        source_filter: 若指定 ('mvtec'/'gc10'/'neu'), 限定从某 source 采样.
        split: 若指定, 切换 ds 的 split pool (e.g. 'meta_test' for eval).
               注意: 切换 split 需要 sampler 在 init 时已分别加载 train/test pool,
               或者重新加载 ds. 这里简单做法: 重新加载 split.
        """
        # 为支持 split 切换, 在每次 sample 时根据 split 重新加载 ds (lazy)
        mvtec_ds, gc10_ds, neu_ds, carpaint_ds, aero_ds, mvtec_dropout_ds = \
            self._resolve_ds(split or self.mvtec.split)

        rng = random.Random()
        candidates = list(self.product_to_source.keys())
        # 过滤掉当前 split 下没有样本的 product
        candidates = [p for p in candidates
                     if self._has_any_samples(p, self.product_to_source[p][1])]
        if source_filter is not None:
            candidates = [p for p in candidates
                         if self.product_to_source[p][0] == source_filter]
        if not candidates:
            raise ValueError(
                f'No products match source_filter={source_filter}, split={split}')

        product_id = rng.choice(candidates)
        source_name, ds = self.product_to_source[product_id]

        # 2) 选 N 个 defect 类 (adaptive)
        n_total = ds.num_defects(product_id)
        eff_n = min(self.n_way, n_total)
        if n_total < 2:
            raise ValueError(f'product {product_id} has only {n_total} classes')

        chosen_classes = rng.sample(range(n_total), eff_n)

        # 3) 抽 K shot + Q query per class
        s_imgs, s_y, s_pid, s_cond = [], [], [], []
        q_imgs, q_y, q_pid, q_cond = [], [], [], []

        for ep_label, class_label in enumerate(chosen_classes):
            paths = ds.sample_paths(product_id, class_label)
            if not paths:
                continue
            need = self.k_shot + self.q_query
            if len(paths) < need:
                picks = [rng.choice(paths) for _ in range(need)]
            else:
                picks = rng.sample(paths, need)
            for shot_idx in range(self.k_shot):
                img, cond = _load_image(picks[shot_idx], self._transform, rng)
                s_imgs.append(img); s_y.append(ep_label); s_pid.append(product_id)
                s_cond.append(cond)
            for q_idx in range(self.q_query):
                img, cond = _load_image(picks[self.k_shot + q_idx],
                                        self._transform, rng)
                q_imgs.append(img); q_y.append(ep_label); q_pid.append(product_id)
                q_cond.append(cond)

        return {
            'support_rgb': torch.stack(s_imgs, 0) if s_imgs else torch.empty(0),
            'support_y': torch.tensor(s_y, dtype=torch.long),
            'support_pid': torch.tensor(s_pid, dtype=torch.long),
            'support_cond': torch.stack(s_cond, 0) if s_cond else torch.empty(0),
            'query_rgb': torch.stack(q_imgs, 0) if q_imgs else torch.empty(0),
            'query_y': torch.tensor(q_y, dtype=torch.long),
            'query_pid': torch.tensor(q_pid, dtype=torch.long),
            'query_cond': torch.stack(q_cond, 0) if q_cond else torch.empty(0),
            'source_name': source_name,
            'product_id': product_id,
            'n_way': eff_n,
        }

    def _resolve_ds(self, split: str):
        """根据 split 重新加载 ds (cache 避免重复加载)."""
        if not hasattr(self, '_ds_cache'):
            self._ds_cache = {}
        if split not in self._ds_cache:
            mvtec = MVTECTestEpisodeDataset(
                self.mvtec.root, img_size=self.mvtec.img_size,
                augment=self.mvtec.augment, split=split)
            gc10 = GC10EpisodeDataset(
                self.gc10.root, img_size=self.gc10.img_size,
                augment=self.gc10.augment, split=split)
            neu = NEUEpisodeDataset(
                self.neu.root, img_size=self.neu.img_size,
                augment=self.neu.augment, split=split)
            carpaint = None
            if self.carpaint is not None:
                carpaint = CarPaintEpisodeDataset(
                    self.carpaint.root, img_size=self.carpaint.img_size,
                    augment=self.carpaint.augment, split=split)
            aero = None
            if self.aero is not None:
                aero = AeroRealEpisodeDataset(
                    self.aero.root, img_size=self.aero.img_size,
                    augment=self.aero.augment, split=split)
            mvtec_dropout = None
            if self.mvtec_dropout is not None:
                mvtec_dropout = MVTecDropoutEpisodeDataset(
                    self.mvtec_dropout.mvtec_root,
                    img_size=self.mvtec_dropout.img_size,
                    augment=self.mvtec_dropout.augment, split=split)
            self._ds_cache[split] = (mvtec, gc10, neu, carpaint, aero, mvtec_dropout)
        return self._ds_cache[split]

    def _has_enough_samples(self, pid, ds):
        n = ds.num_defects(pid)
        for lid in range(n):
            if len(ds.sample_paths(pid, lid)) < self.k_shot + self.q_query \
                    and len(ds.sample_paths(pid, lid)) < self.k_shot + 1:
                # 允许重复采样, 只要至少有 1 张
                pass
        return True

    def _has_any_samples(self, pid, ds):
        n = ds.num_defects(pid)
        return any(len(ds.sample_paths(pid, lid)) > 0 for lid in range(n))


def _load_image(path: str, transform, rng: random.Random):
    """加载一张图 + 估算工况 + 应用 transform."""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img)
    cond = _estimate_condition(img_np)
    return transform(img), torch.from_numpy(cond)


# ──────────────────────────────────────────────────────────
# 工厂
# ──────────────────────────────────────────────────────────


def build_episode_samplers(mvtec_root, gc10_root, neu_root,
                           n_way=5, k_shot=5, q_query=5,
                           img_size=224, augment=True,
                           split: str = 'meta_train',
                           carpaint_root: Optional[str] = None,
                           aero_real_root: Optional[str] = None,
                           mvtec_dropout_root: Optional[str] = None):
    """构建 episode pool + sampler 一站式入口.

    split: 'meta_train' (默认, 训练用) | 'meta_test' (评测用) | 'all'
    carpaint_root: 若指定, 加载车漆 CMPL 域 (cardd/carparts/syndcar).
    aero_real_root: 若指定, 加载航天真实数据 (PID 20).
    mvtec_dropout_root: 若显式指定, 加载 MVTEC 被剔除产品 (PID 21, 默认 toothbrush)
        作为 OOD 域. 缺省 = 关闭 (PID 21 不进入 sampler).

    任一路径不存在时, 对应 ds 为 None, sampler 自动跳过该 PID, 不影响其它源.
    """
    mvtec = MVTECTestEpisodeDataset(mvtec_root, img_size=img_size,
                                    augment=augment, split=split)
    gc10 = GC10EpisodeDataset(gc10_root, img_size=img_size,
                              augment=augment, split=split)
    neu = NEUEpisodeDataset(neu_root, img_size=img_size,
                            augment=augment, split=split)
    carpaint = None
    if carpaint_root is not None and os.path.isdir(carpaint_root):
        carpaint = CarPaintEpisodeDataset(carpaint_root, img_size=img_size,
                                          augment=augment, split=split)
    aero = None
    if aero_real_root is not None and os.path.isdir(aero_real_root):
        aero = AeroRealEpisodeDataset(aero_real_root, img_size=img_size,
                                      augment=augment, split=split)
    mvtec_dropout = None
    # 默认关闭: 必须显式传 mvtec_dropout_root 才启用 (PID 21 OOD 域)
    if mvtec_dropout_root is not None and os.path.isdir(mvtec_dropout_root):
        mvtec_dropout = MVTecDropoutEpisodeDataset(mvtec_dropout_root,
                                                   img_size=img_size,
                                                   augment=augment, split=split)
    sampler = MultiDomainEpisodeSampler(mvtec, gc10, neu,
                                        carpaint_ds=carpaint,
                                        aero_ds=aero,
                                        mvtec_dropout_ds=mvtec_dropout,
                                        n_way=n_way, k_shot=k_shot, q_query=q_query,
                                        augment=augment)
    return sampler, mvtec, gc10, neu, carpaint, aero, mvtec_dropout