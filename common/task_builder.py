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
