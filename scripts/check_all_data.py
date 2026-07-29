"""Validate converted aviation + pretrain datasets."""

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.aviation_taxonomy import NORMAL_FOLDER


def _count_samples(domain_dir):
    counts = {}
    total = 0
    for class_name in sorted(os.listdir(domain_dir)):
        if class_name == 'meta.json':
            continue
        class_dir = os.path.join(domain_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        n = 0
        missing_rgb = 0
        with_roi = 0
        for sample_id in os.listdir(class_dir):
            sample_dir = os.path.join(class_dir, sample_id)
            if not os.path.isdir(sample_dir):
                continue
            rgb = os.path.join(sample_dir, 'rgb.jpg')
            if os.path.isfile(rgb):
                n += 1
                if os.path.isfile(os.path.join(sample_dir, 'roi.json')):
                    with_roi += 1
            else:
                missing_rgb += 1
        counts[class_name] = {'samples': n, 'missing_rgb': missing_rgb, 'with_roi': with_roi}
        total += n
    return total, counts


def _check_aviation(data_root, domains):
    aviation_root = os.path.join(data_root, 'aviation')
    print('=== aviation/ ===')
    for domain_id in domains:
        domain_dir = os.path.join(aviation_root, 'domain_{}'.format(domain_id))
        if not os.path.isdir(domain_dir):
            print('domain_{}: MISSING'.format(domain_id))
            continue
        total, counts = _count_samples(domain_dir)
        meta_path = os.path.join(domain_dir, 'meta.json')
        source = 'unknown'
        if os.path.isfile(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                source = json.load(f).get('source', source)
        print('domain_{} [{}]: {} samples'.format(domain_id, source, total))
        for cls, info in counts.items():
            print('  {}: {} (roi: {})'.format(cls, info['samples'], info['with_roi']))


def _count_flat(root):
    stats = {}
    total = 0
    if not os.path.isdir(root):
        return total, stats
    for dirpath, _, filenames in os.walk(root):
        jpgs = [f for f in filenames if f.lower().endswith('.jpg')]
        if jpgs:
            rel = os.path.relpath(dirpath, root)
            stats[rel] = len(jpgs)
            total += len(jpgs)
    return total, stats


def _check_pretrain(data_root):
    print('=== pretrain/ ===')
    for name in ('bladesynth', 'aebad'):
        root = os.path.join(data_root, 'pretrain', name)
        total, stats = _count_flat(root)
        print('{}: {} images'.format(name, total))
        for rel, n in sorted(stats.items()):
            print('  {}: {}'.format(rel, n))
        meta_path = os.path.join(root, 'meta.json')
        if os.path.isfile(meta_path):
            print('  meta.json: OK')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_path', default='../data')
    parser.add_argument('--domains', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--skip_pretrain', action='store_true')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_dir_path)
    domains = [int(x) for x in args.domains.split(',') if x.strip()]
    _check_aviation(data_root, domains)
    if not args.skip_pretrain:
        _check_pretrain(data_root)
    print('Note: {} samples use full image (no roi.json).'.format(NORMAL_FOLDER))


if __name__ == '__main__':
    main()
