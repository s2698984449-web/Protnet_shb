"""Multimodal aviation defect dataset for MAML task sampling."""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .aviation_taxonomy import CLASS_FOLDERS
from .cmpl_taxonomy import CMPL_CLASS_FOLDERS
from .knowledge_base import DEFAULT_FUSION_PRIOR, DEFAULT_MATERIAL_PARAMS, load_domain_meta

_VALID_CLASS_NAMES = set(CLASS_FOLDERS.values()) | set(CMPL_CLASS_FOLDERS.values())


def _default_process_params():
    return [0.0] * 8


def _default_defect_history(num_classes=10):
    return [1.0 / max(num_classes, 1)] * num_classes


class AviationDefectDataset(Dataset):
    """
    Expected layout:

        {data_dir}/{dataset_name}/domain_{id}/{class_name}/{sample_id}/
            rgb.png | rgb.jpg
            thermal.png | thermal.jpg
            ultrasonic.npy

    Optional domain metadata:
        {data_dir}/{dataset_name}/domain_{id}/meta.json
    """

    RGB_SIZE = (256, 256)
    THERMAL_SIZE = (320, 240)

    def __init__(self, domain_id, data_dir, train=True, rgb_only=False,
                 exclude_sample_ids=None, dataset_name='aviation'):
        super().__init__()
        self.domain_id = domain_id
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.train = train
        self.rgb_only = rgb_only
        self.exclude_sample_ids = set(exclude_sample_ids or [])
        self.root = os.path.join(data_dir, dataset_name, 'domain_{}'.format(domain_id))
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                'Domain folder not found: {} (dataset_name={}).'.format(
                    self.root, dataset_name))

        self.samples = []
        self.class_to_idx = {}
        for class_name in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name.startswith('_') or class_name not in _VALID_CLASS_NAMES:
                continue
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            for sample_id in sorted(os.listdir(class_dir)):
                if sample_id in self.exclude_sample_ids:
                    continue
                sample_dir = os.path.join(class_dir, sample_id)
                if os.path.isdir(sample_dir):
                    self.samples.append((sample_dir, self.class_to_idx[class_name]))

        if not self.samples:
            raise FileNotFoundError('No samples under {}'.format(self.root))

        self.domain_meta = load_domain_meta(data_dir, domain_id, dataset_name=dataset_name)
        if self.domain_meta is None:
            self.domain_meta = {
                'material_id': 'AL7075',
                'material_params': list(DEFAULT_MATERIAL_PARAMS),
                'process_params': _default_process_params(),
                'defect_history': _default_defect_history(len(self.class_to_idx)),
                'fusion_prior': list(DEFAULT_FUSION_PRIOR),
            }

        self.rgb_transform = transforms.Compose([
            transforms.Resize(self.RGB_SIZE),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.thermal_transform = transforms.Compose([
            transforms.Resize(self.THERMAL_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, sample_dir):
        for name in ('rgb.png', 'rgb.jpg', 'RGB.png', 'RGB.jpg'):
            path = os.path.join(sample_dir, name)
            if os.path.isfile(path):
                return self.rgb_transform(Image.open(path).convert('RGB'))
        raise FileNotFoundError('Missing rgb image in {}'.format(sample_dir))

    def _load_thermal(self, sample_dir):
        for name in ('thermal.png', 'thermal.jpg', 'Thermal.png', 'Thermal.jpg'):
            path = os.path.join(sample_dir, name)
            if os.path.isfile(path):
                return self.thermal_transform(Image.open(path).convert('L'))
        return torch.zeros(1, self.THERMAL_SIZE[1], self.THERMAL_SIZE[0])

    def _load_ultrasonic(self, sample_dir):
        path = os.path.join(sample_dir, 'ultrasonic.npy')
        if os.path.isfile(path):
            arr = np.load(path).astype(np.float32).reshape(-1)
            if arr.size >= 512:
                arr = arr[:512]
            else:
                arr = np.pad(arr, (0, 512 - arr.size))
            return torch.from_numpy(arr)
        return torch.zeros(512, dtype=torch.float32)

    def __getitem__(self, index):
        sample_dir, label = self.samples[index]
        rgb = self._load_rgb(sample_dir)
        if self.rgb_only:
            thermal = torch.zeros(1, self.THERMAL_SIZE[1], self.THERMAL_SIZE[0])
            ultrasonic = torch.zeros(512)
        else:
            thermal = self._load_thermal(sample_dir)
            ultrasonic = self._load_ultrasonic(sample_dir)
        return (rgb, thermal, ultrasonic), torch.tensor(label, dtype=torch.int64)

    def get_domain_meta(self):
        return self.domain_meta


def save_domain_meta_template(path, material_id='AL7075'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    template = {
        'material_id': material_id,
        'material_params': DEFAULT_MATERIAL_PARAMS,
        'process_params': _default_process_params(),
        'defect_history': _default_defect_history(),
        'fusion_prior': list(DEFAULT_FUSION_PRIOR),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    return template
