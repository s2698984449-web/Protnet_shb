"""Per-task / per-sample ProtoNet eval with path tracing (hard-task diagnosis)."""

import os
import re
from collections import defaultdict

import numpy as np
import torch

from .maml_loop import split_support_query
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
        model, batch, domain_meta, shots, ways, device, distance='euclidean'):
    support, query = split_support_query(batch, shots, ways, device)
    s_rgb, s_th, s_us, s_y = support
    q_rgb, q_th, q_us, q_y = query

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
        model, test_tasks, domain_meta, shots, ways, n_tasks, device,
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
            model, batch, domain_meta, shots, ways, device, distance=distance)

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
