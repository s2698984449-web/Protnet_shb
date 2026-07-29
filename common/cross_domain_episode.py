"""Cross-domain N-way K-shot episode sampling (real domains only, scratch vs damage)."""

import os
import random

import torch

SCRATCH_CLASS = '0_scratch'
DAMAGE_CLASS = '3_damage'
UNIFIED_CLASS_FOLDERS = (SCRATCH_CLASS, DAMAGE_CLASS)


def build_cross_domain_pool(data_dir, train_domains, rgb_only=True):
    """
    Pool scratch/damage ROI indices across train domains.

    Returns:
        pools: unified_label (0=scratch, 1=damage) -> [(domain_id, dataset_idx), ...]
        datasets: domain_id -> AviationDefectDataset
    """
    from .aviation_datasets import AviationDefectDataset

    pools = {0: [], 1: []}
    datasets = {}
    folder_to_unified = {SCRATCH_CLASS: 0, DAMAGE_CLASS: 1}

    for domain_id in train_domains:
        ds = AviationDefectDataset(domain_id, data_dir, train=True, rgb_only=rgb_only)
        datasets[domain_id] = ds
        for idx, (sample_dir, _label) in enumerate(ds.samples):
            parts = sample_dir.replace('\\', '/').split('/')
            unified = None
            for folder in parts:
                if folder in folder_to_unified:
                    unified = folder_to_unified[folder]
                    break
            if unified is not None:
                pools[unified].append((domain_id, idx))

    for unified, entries in pools.items():
        if len(entries) < 2:
            raise ValueError(
                'Cross-domain pool label {} has only {} samples (need >= 2).'.format(
                    unified, len(entries)))

    return pools, datasets


def sample_cross_domain_episode(pools, datasets, ways, shots):
    """
    Sample one episode in learn2learn layout:
      class0: (S,Q) x shots, then class1: (S,Q) x shots, ...
    """
    need = 2 * shots
    rgb_list, th_list, us_list, label_list = [], [], [], []

    for unified_label in range(ways):
        entries = pools[unified_label]
        if len(entries) < need:
            chosen = [random.choice(entries) for _ in range(need)]
        else:
            chosen = random.sample(entries, need)

        for shot_idx in range(shots):
            sup_dom, sup_idx = chosen[shot_idx]
            qry_dom, qry_idx = chosen[shots + shot_idx]

            (rgb, th, us), _ = datasets[sup_dom][sup_idx]
            rgb_list.append(rgb)
            th_list.append(th)
            us_list.append(us)
            label_list.append(unified_label)

            (rgb, th, us), _ = datasets[qry_dom][qry_idx]
            rgb_list.append(rgb)
            th_list.append(th)
            us_list.append(us)
            label_list.append(unified_label)

    rgb = torch.stack(rgb_list, dim=0)
    thermal = torch.stack(th_list, dim=0)
    ultrasonic = torch.stack(us_list, dim=0)
    labels = torch.tensor(label_list, dtype=torch.int64)
    return (rgb, thermal, ultrasonic), labels


def cross_domain_pool_stats(pools):
    return {label: len(entries) for label, entries in pools.items()}
