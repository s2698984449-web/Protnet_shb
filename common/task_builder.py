"""Build learn2learn Tasksets for aviation multimodal MAML."""

import logging

import learn2learn as l2l
from learn2learn.data.transforms import (
    ConsecutiveLabels,
    FusedNWaysKShots,
    LoadData,
    RemapLabels,
)

from .aviation_datasets import AviationDefectDataset


def _parse_exclude_sample_ids(text):
    if not text:
        return []
    return [s.strip() for s in str(text).split(',') if s.strip()]


def create_tasksets(args):
    logging.info('Training domains: {}.'.format(args.train_domains))
    train_domains = [int(i) for i in args.train_domains.split(',')]
    logging.info('Testing domain: {}.'.format(args.test_domain))

    exclude_ids = _parse_exclude_sample_ids(getattr(args, 'exclude_sample_ids', ''))
    dataset_name = getattr(args, 'dataset_name', 'aviation')

    train_tasks = []
    domain_meta_map = {}
    for domain in train_domains:
        dataset = AviationDefectDataset(
            domain, args.data_dir_path, train=True, rgb_only=args.rgb_only,
            dataset_name=dataset_name)
        domain_meta_map[domain] = dataset.get_domain_meta()
        meta_dataset = l2l.data.MetaDataset(dataset)
        transforms = [
            FusedNWaysKShots(meta_dataset, n=args.ways, k=2 * args.shots),
            LoadData(meta_dataset),
            RemapLabels(meta_dataset),
            ConsecutiveLabels(meta_dataset),
        ]
        train_tasks.append(l2l.data.Taskset(
            meta_dataset,
            task_transforms=transforms,
            num_tasks=args.train_task_num,
        ))

    test_dataset = AviationDefectDataset(
        args.test_domain, args.data_dir_path, train=False, rgb_only=args.rgb_only,
        exclude_sample_ids=exclude_ids or None, dataset_name=dataset_name)
    if exclude_ids:
        logging.info('Excluded {} sample(s) from test domain_{}: {}.'.format(
            len(exclude_ids), args.test_domain, ', '.join(exclude_ids)))
    domain_meta_map[args.test_domain] = test_dataset.get_domain_meta()
    test_meta = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        FusedNWaysKShots(test_meta, n=args.ways, k=2 * args.shots),
        LoadData(test_meta),
        RemapLabels(test_meta),
        ConsecutiveLabels(test_meta),
    ]
    test_tasks = l2l.data.Taskset(
        test_meta,
        task_transforms=test_transforms,
        num_tasks=args.test_task_num,
    )
    return train_tasks, test_tasks, domain_meta_map


# ─── TaskEncoder: material/process/history → 64-d task embedding ───

import torch
import torch.nn as nn


class TaskEncoder(nn.Module):
    """Map domain metadata vectors to 64-d task embedding τ."""

    def __init__(self, material_dim=12, process_dim=8, history_dim=10, out_dim=64):
        super().__init__()
        self.material_net = nn.Sequential(
            nn.Linear(material_dim, 64), nn.ReLU(), nn.Linear(64, 32),
        )
        self.process_net = nn.Sequential(
            nn.Linear(process_dim, 32), nn.ReLU(), nn.Linear(32, 16),
        )
        self.history_net = nn.Sequential(
            nn.Linear(history_dim, 32), nn.ReLU(), nn.Linear(32, 16),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, material_params, process_params, defect_history):
        m = self.material_net(material_params)
        p = self.process_net(process_params)
        h = self.history_net(defect_history)
        return self.fusion(torch.cat([m, p, h], dim=-1))


def meta_vectors_to_tensors(meta, device):
    """Convert domain meta dict to batched tensors on device."""
    material = torch.tensor(meta['material_params'], dtype=torch.float32, device=device)
    process = torch.tensor(meta['process_params'], dtype=torch.float32, device=device)
    history = torch.tensor(meta['defect_history'], dtype=torch.float32, device=device)
    if material.dim() == 1:
        material = material.unsqueeze(0)
    if process.dim() == 1:
        process = process.unsqueeze(0)
    if history.dim() == 1:
        history = history.unsqueeze(0)
    return material, process, history


# ─── Per-task detailed eval for hard-task diagnosis ───

import os
import re
from collections import defaultdict

import numpy as np
import torch

from .protonet_loop import prototypical_logits


def _specimen_from_path(sample_dir):
    norm = sample_dir.replace('\\', '/')
    sample_id = os.path.basename(norm.rstrip('/'))
    class_name = os.path.basename(os.path.dirname(norm))
    base = re.match(r'^(\d+)', sample_id)
    specimen = base.group(1) if base else sample_id
    return specimen, sample_id, class_name


def task_description_indices(taskset):
    desc = taskset.sample_task_description()
    meta = taskset.dataset
    av = meta.dataset
    rows = []
    for dd in desc:
        idx = dd.index
        sample_dir, _ = av.samples[idx]
        label = meta.indices_to_labels[idx]
        specimen, sample_id, class_name = _specimen_from_path(sample_dir)
        rows.append({
            'index': int(idx),
            'label': int(label),
            'class_name': class_name,
            'sample_id': sample_id,
            'specimen_id': specimen,
            'sample_dir': sample_dir,
        })
    return desc, rows


def evaluate_protonet_task_detailed(
        model, batch, shots, ways, device, distance='euclidean'):
    from .maml_loop import split_support_query  # local import to avoid circular dep
    support, query = split_support_query(batch, shots, ways, device)
    s_rgb, s_th, s_us, s_y = support
    q_rgb, q_th, q_us, q_y = query
    domain_meta = batch.get('domain_meta', None)

    s_emb = model.embed(s_rgb, s_th, s_us, domain_meta)
    q_emb = model.embed(q_rgb, q_th, q_us, domain_meta)
    logits = prototypical_logits(q_emb, s_emb, s_y, ways, distance=distance)
    preds = logits.argmax(dim=-1)
    acc = (preds == q_y).float().mean().item()

    return {
        'acc': float(acc),
        'query_labels': q_y.cpu().tolist(),
        'query_preds': preds.cpu().tolist(),
        'query_correct': (preds == q_y).cpu().tolist(),
    }


def evaluate_tasks_detailed_protonet(
        model, test_tasks, shots, ways, n_tasks, device,
        distance='euclidean', hard_threshold=0.7):
    model.eval()
    tasks = []
    query_sample_stats = defaultdict(lambda: {
        'query_count': 0,
        'wrong_count': 0,
        'class_name': '',
        'specimen_id': '',
        'sample_id': '',
        'sample_dir': '',
    })

    for task_id in range(n_tasks):
        desc, layout = task_description_indices(test_tasks)
        batch = test_tasks.get_task(desc)
        detail = evaluate_protonet_task_detailed(
            model, batch, shots, ways, device, distance=distance)

        n = len(layout)
        adapt_mask = np.zeros(n, dtype=bool)
        adapt_mask[np.arange(shots * ways) * 2] = True

        support_rows = []
        query_rows = []
        q_ptr = 0
        for i, row in enumerate(layout):
            if adapt_mask[i]:
                support_rows.append({**row, 'role': 'support'})
            else:
                enriched = {
                    **row,
                    'role': 'query',
                    'true_label': detail['query_labels'][q_ptr],
                    'pred_label': detail['query_preds'][q_ptr],
                    'correct': detail['query_correct'][q_ptr],
                }
                query_rows.append(enriched)
                key = row['sample_dir']
                st = query_sample_stats[key]
                st['query_count'] += 1
                st['wrong_count'] += 0 if enriched['correct'] else 1
                st['class_name'] = row['class_name']
                st['specimen_id'] = row['specimen_id']
                st['sample_id'] = row['sample_id']
                st['sample_dir'] = row['sample_dir']
                q_ptr += 1

        tasks.append({
            'task_id': task_id,
            'acc': detail['acc'],
            'hard': detail['acc'] < hard_threshold,
            'support': support_rows,
            'query': query_rows,
        })

    return tasks, query_sample_stats
