"""Flat-folder datasets for Phase 0 backbone pretraining."""

import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .aviation_taxonomy import BLADESYNTH_CLASSES


def _list_jpgs(root):
    samples = []
    if not os.path.isdir(root):
        return samples
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith('.jpg') and not fname.startswith('._'):
                samples.append(os.path.join(dirpath, fname))
    return sorted(samples)


class BladeSynthPretrainDataset(Dataset):
    """Load data/pretrain/bladesynth/{split}/{class}/*.jpg"""

    def __init__(self, data_dir, split='train', imgsz=640, augment=False):
        root = os.path.join(data_dir, 'pretrain', 'bladesynth', split)
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(BLADESYNTH_CLASSES)}
        for class_name in BLADESYNTH_CLASSES:
            class_dir = os.path.join(root, class_name)
            for path in _list_jpgs(class_dir):
                self.samples.append((path, self.class_to_idx[class_name]))
        if not self.samples:
            raise FileNotFoundError('No bladeSynth pretrain data under {}'.format(root))

        tfms = [transforms.Resize((imgsz, imgsz))]
        if augment:
            tfms.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ])
        tfms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(tfms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert('RGB')), label


class AeBADBinaryPretrainDataset(Dataset):
    """Load data/pretrain/aebad/binary/{split}/{normal,defect}/*.jpg"""

    def __init__(self, data_dir, split='train', imgsz=640, augment=False):
        root = os.path.join(data_dir, 'pretrain', 'aebad', 'binary', split)
        self.samples = []
        for label, folder in ((0, 'normal'), (1, 'defect')):
            for path in _list_jpgs(os.path.join(root, folder)):
                self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError('No AeBAD binary pretrain data under {}'.format(root))

        tfms = [transforms.Resize((imgsz, imgsz))]
        if augment:
            tfms.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ])
        tfms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(tfms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert('RGB')), label
