#!/usr/bin/env python
"""
Plan aviation meta-train domain splits (v2).

Usage (from repo root):
  python 航空缺陷检测/scripts/plan_domain_split.py
  python 航空缺陷检测/scripts/plan_domain_split.py --export_csv data/航天/specimen_domain_map.proposed.csv
  python 航空缺陷检测/scripts/plan_domain_split.py --from_raw --raw_root data/航天/raw/Aero-engine_defect-detect_new

Outputs:
  reports/domain_split/plan_report.json
  optional CSV for convert_datasets.py --domain_map_csv
"""

import argparse
import json
import os
import sys

import re

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.domain_split_v2 import (
    DEFAULT_TEST_BASES,
    build_domain_map_v2,
    count_samples_by_base_from_processed,
    count_samples_by_base_from_raw,
    export_registry_template,
    load_specimen_registry,
    registry_rows_to_csv,
    search_auto_partitions,
    validate_domain_split,
)
from common.project_paths import AEROSPACE_PROCESSED, AEROSPACE_RAW


def _is_real_specimen_base(base_id):
    return bool(re.match(r'^\d+$', str(base_id)))


def _filter_real_specimen_counts(base_counts):
    return {b: dict(v) for b, v in base_counts.items() if _is_real_specimen_base(b)}


def _collect_raw_samples(raw_root):
    from scripts.convert_datasets import _collect_real_samples
    samples, _ = _collect_real_samples(raw_root)
    return samples


def _print_validation(title, validation):
    print('\n=== {} ==='.format(title))
    print('ok:', validation['ok'],
          '| task_upper_bound_hint:', validation['task_upper_bound_hint'])
    for domain_id, info in sorted(validation['domains'].items(), key=lambda x: int(x[0])):
        print('  domain_{}: scratch={} damage={} min_pair={} feasible={}'.format(
            domain_id, info['scratch'], info['damage'], info['min_pair'],
            info.get('feasible_2w{}s'.format(validation['shots']), info.get('feasible'))))


def main():
    parser = argparse.ArgumentParser(description='Plan aviation domain split v2')
    parser.add_argument('--aviation_root', default=os.path.join(AEROSPACE_PROCESSED, 'aviation'))
    parser.add_argument('--raw_root',
                        default=os.path.join(AEROSPACE_RAW, 'Aero-engine_defect-detect_new'))
    parser.add_argument('--from_raw', action='store_true', default=True,
                        help='Count ROI from raw YOLO (default True; use --no_from_raw for processed)')
    parser.add_argument('--no_from_raw', action='store_false', dest='from_raw',
                        help='Use processed aviation/domain_0~2 instead of raw YOLO')
    parser.add_argument('--ways', type=int, default=2)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--k_min', type=int, default=3)
    parser.add_argument('--k_max', type=int, default=5)
    parser.add_argument('--domain_map_csv', default='',
                        help='Validate an existing registry CSV')
    parser.add_argument('--export_csv', default='',
                        help='Write proposed greedy-best CSV here')
    parser.add_argument('--export_template', default='',
                        help='Write legacy-seed editable template CSV')
    parser.add_argument('--report_path',
                        default=os.path.join(_ROOT, 'reports', 'domain_split', 'plan_report.json'))
    args = parser.parse_args()

    if args.from_raw:
        samples = _collect_raw_samples(args.raw_root)
        base_counts = _filter_real_specimen_counts(count_samples_by_base_from_raw(samples))
        all_bases = sorted(base_counts.keys())
    else:
        base_counts = _filter_real_specimen_counts(
            count_samples_by_base_from_processed(args.aviation_root, train_domain_ids=[0, 1, 2]))
        all_bases = sorted(base_counts.keys())

    train_bases = [b for b in all_bases if b not in DEFAULT_TEST_BASES]
    os.makedirs(os.path.dirname(os.path.abspath(args.report_path)), exist_ok=True)

    report = {
        'ways': args.ways,
        'shots': args.shots,
        'train_bases': train_bases,
        'per_base_counts': {
            b: {
                'scratch': base_counts[b].get(0, 0),
                'damage': base_counts[b].get(3, 0),
            } for b in sorted(all_bases, key=lambda x: (len(x), x))
        },
    }

    # legacy %3 baseline
    from scripts.convert_datasets import build_domain_map
    legacy_map, _, test_bases = build_domain_map(all_bases, DEFAULT_TEST_BASES)
    legacy_val = validate_domain_split(
        base_counts, legacy_map, ways=args.ways, shots=args.shots)
    report['legacy_mod3'] = legacy_val
    _print_validation('Legacy base_id % 3 (current)', legacy_val)

    # optional CSV validation
    if args.domain_map_csv:
        registry = load_specimen_registry(args.domain_map_csv)
        csv_map, _, _, split_info = build_domain_map_v2(
            all_bases, registry=registry, test_bases=DEFAULT_TEST_BASES)
        csv_val = validate_domain_split(
            base_counts, csv_map, ways=args.ways, shots=args.shots)
        report['csv_registry'] = {'path': args.domain_map_csv, 'validation': csv_val,
                                  'split_info': split_info}
        _print_validation('CSV registry: {}'.format(args.domain_map_csv), csv_val)

    # auto greedy search
    suggestions = search_auto_partitions(
        base_counts, train_bases, k_min=args.k_min, k_max=args.k_max,
        ways=args.ways, shots=args.shots)
    report['auto_suggestions'] = []
    print('\n=== Greedy auto partitions (K={}..{}) ==='.format(args.k_min, args.k_max))
    best = suggestions[0] if suggestions else None
    for sug in suggestions:
        val = sug['validation']
        line = ('K={} feasible={} score_min_pair={} task_ub_hint={} groups={}'.format(
            sug['k_domains'], sug['feasible'], sug['score_min_pair'],
            val['task_upper_bound_hint'], sug['groups']))
        print(' ', line)
        report['auto_suggestions'].append({
            'k_domains': sug['k_domains'],
            'feasible': sug['feasible'],
            'score_min_pair': sug['score_min_pair'],
            'groups': sug['groups'],
            'domain_map': sug['domain_map'],
            'validation': val,
        })

    if args.export_template:
        export_registry_template(args.export_template, train_bases, DEFAULT_TEST_BASES)
        print('\nWrote template:', args.export_template)

    if args.export_csv and best:
        registry = {}
        if args.domain_map_csv and os.path.isfile(args.domain_map_csv):
            registry = load_specimen_registry(args.domain_map_csv)
        registry_rows_to_csv(
            args.export_csv, best['domain_map'], registry, train_bases, DEFAULT_TEST_BASES)
        print('Wrote proposed CSV:', args.export_csv)

    with open(args.report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('\nReport saved:', args.report_path)


if __name__ == '__main__':
    main()
