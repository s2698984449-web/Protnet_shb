"""Metadata-driven aviation domain split (v2) + 2-way K-shot feasibility checks."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict

SCRATCH_CLASS = 0
DAMAGE_CLASS = 3
META_TWO_WAY_FOLDERS = ('0_scratch', '3_damage')

DEFAULT_TEST_BASES = frozenset({'20', '21', '22', '23', '24', '25', '26'})
DEFAULT_TEST_DOMAIN = 3

CSV_COLUMNS = (
    'specimen_id',
    'train_domain',
    'material_id',
    'process_batch',
    'camera_batch',
    'exclude',
    'notes',
)

GROUP_BY_COLUMNS = ('train_domain', 'material_id', 'process_batch', 'camera_batch')


def base_sort_key(base_id):
    try:
        return (0, int(base_id))
    except ValueError:
        return (1, str(base_id))


def _norm_bool(value):
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in ('1', 'true', 'yes', 'y', 't')


def load_specimen_registry(csv_path):
    """Load specimen_id -> metadata row from CSV."""
    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError('Specimen registry not found: {}'.format(csv_path))

    registry = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if 'specimen_id' not in (reader.fieldnames or []):
            raise ValueError('CSV missing required column: specimen_id')
        for row in reader:
            sid = str(row.get('specimen_id', '')).strip()
            if not sid or sid.startswith('#'):
                continue
            clean = {
                k: (str(row.get(k, '')).strip() if row.get(k) is not None else '')
                for k in CSV_COLUMNS
            }
            clean['exclude'] = _norm_bool(clean.get('exclude'))
            registry[sid] = clean
    return registry


def export_registry_template(path, train_bases, test_bases=None, default_material='AL7075'):
    """Write an editable CSV seeded from current train/test base lists."""
    test_bases = set(test_bases or DEFAULT_TEST_BASES)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for idx, base_id in enumerate(sorted(train_bases, key=base_sort_key)):
            if base_id in test_bases:
                continue
            writer.writerow({
                'specimen_id': base_id,
                'train_domain': str(idx % 3),
                'material_id': default_material,
                'process_batch': '',
                'camera_batch': '',
                'exclude': 'false',
                'notes': 'legacy idx%3 seed',
            })
        for base_id in sorted(test_bases, key=base_sort_key):
            writer.writerow({
                'specimen_id': base_id,
                'train_domain': str(DEFAULT_TEST_DOMAIN),
                'material_id': default_material,
                'process_batch': '',
                'camera_batch': '',
                'exclude': 'false',
                'notes': 'holdout meta-test',
            })


def build_domain_map_v2(
        all_bases,
        registry=None,
        test_bases=None,
        test_domain=DEFAULT_TEST_DOMAIN,
        group_by='train_domain',
        legacy_mod=3,
        excluded_bases=None):
    """
    Build specimen_id -> meta domain id.

    Priority per train specimen:
      1) excluded -> skip
      2) registry.train_domain if numeric
      3) registry[group_by] bucket -> dense domain index (0..K-1)
      4) legacy: sorted train index % legacy_mod
    """
    if test_bases is None:
        test_bases = set(DEFAULT_TEST_BASES)
    excluded_bases = set(excluded_bases or [])
    registry = registry or {}

    for sid, row in registry.items():
        if row.get('exclude'):
            excluded_bases.add(sid)

    bases = sorted(set(all_bases), key=base_sort_key)
    domain_map = {}
    train_bases = []

    for base_id in bases:
        if base_id in excluded_bases:
            continue
        if base_id in test_bases:
            domain_map[base_id] = int(test_domain)
            continue
        train_bases.append(base_id)

    bucket_to_domain = {}
    unassigned = []

    for base_id in train_bases:
        row = registry.get(base_id, {})
        td = row.get('train_domain', '')
        if td not in (None, ''):
            domain_map[base_id] = int(td)
            continue

        bucket = row.get(group_by, '') if group_by != 'train_domain' else ''
        if bucket:
            if bucket not in bucket_to_domain:
                bucket_to_domain[bucket] = len(bucket_to_domain)
            domain_map[base_id] = bucket_to_domain[bucket]
        else:
            unassigned.append(base_id)

    for idx, base_id in enumerate(sorted(unassigned, key=base_sort_key)):
        domain_map[base_id] = idx % max(int(legacy_mod), 1)

    test_base_list = sorted(test_bases & set(bases), key=base_sort_key)
    split_info = {
        'strategy': 'specimen_registry_v2',
        'group_by': group_by,
        'test_domain': int(test_domain),
        'train_bases': [b for b in train_bases if b in domain_map],
        'test_bases': test_base_list,
        'excluded_bases': sorted(excluded_bases, key=base_sort_key),
        'domain_map': {k: domain_map[k] for k in sorted(domain_map, key=base_sort_key)},
    }
    return domain_map, split_info['train_bases'], test_base_list, split_info


def count_samples_by_base_from_raw(samples):
    """Aggregate ROI counts from convert_real sample dicts."""
    counts = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        counts[sample['base_id']][sample['class_id']] += 1
    return counts


def count_samples_by_base_from_processed(aviation_root, train_domain_ids=None):
    """Scan processed aviation folders for scratch/damage counts per specimen."""
    counts = defaultdict(lambda: defaultdict(int))
    folder_to_class = {'0_scratch': SCRATCH_CLASS, '3_damage': DAMAGE_CLASS}

    if train_domain_ids is None:
        train_domain_ids = []
        if os.path.isdir(aviation_root):
            for name in os.listdir(aviation_root):
                if name.startswith('domain_'):
                    try:
                        train_domain_ids.append(int(name.split('_', 1)[1]))
                    except ValueError:
                        pass
        train_domain_ids = sorted(train_domain_ids)

    for domain_id in train_domain_ids:
        domain_dir = os.path.join(aviation_root, 'domain_{}'.format(domain_id))
        if not os.path.isdir(domain_dir):
            continue
        for folder, class_id in folder_to_class.items():
            class_dir = os.path.join(domain_dir, folder)
            if not os.path.isdir(class_dir):
                continue
            for sample_id in os.listdir(class_dir):
                sample_dir = os.path.join(class_dir, sample_id)
                if not os.path.isdir(sample_dir):
                    continue
                base_id = None
                src = os.path.join(sample_dir, 'source.json')
                if os.path.isfile(src):
                    with open(src, encoding='utf-8') as f:
                        meta = json.load(f)
                    base_id = str(meta.get('specimen_id') or meta.get('base_id') or '')
                if not base_id:
                    m = re.match(r'^(\d+)', sample_id)
                    base_id = m.group(1) if m else sample_id
                counts[base_id][class_id] += 1
    return counts


def domain_class_totals(base_counts, domain_map, domain_id, class_ids=(SCRATCH_CLASS, DAMAGE_CLASS)):
    totals = {cid: 0 for cid in class_ids}
    for base_id, did in domain_map.items():
        if did != domain_id:
            continue
        for cid in class_ids:
            totals[cid] += base_counts.get(base_id, {}).get(cid, 0)
    return totals


def validate_domain_split(
        base_counts,
        domain_map,
        ways=2,
        shots=5,
        train_domain_ids=None,
        meta_class_ids=(SCRATCH_CLASS, DAMAGE_CLASS)):
    """Check each train domain supports N-way K-shot (k = 2*shots per class)."""
    need_per_class = 2 * int(shots)
    train_domain_ids = train_domain_ids or sorted({
        d for d in set(domain_map.values()) if d != DEFAULT_TEST_DOMAIN
    })

    report = {
        'ways': int(ways),
        'shots': int(shots),
        'need_per_class': need_per_class,
        'domains': {},
        'ok': True,
        'task_upper_bound_hint': 0,
    }

    for domain_id in train_domain_ids:
        totals = domain_class_totals(base_counts, domain_map, domain_id, meta_class_ids)
        scratch_n = totals.get(SCRATCH_CLASS, 0)
        damage_n = totals.get(DAMAGE_CLASS, 0)
        min_pair = min(scratch_n, damage_n)
        feasible = scratch_n >= need_per_class and damage_n >= need_per_class
        key = 'feasible_2w{}s'.format(shots)
        report['domains'][str(domain_id)] = {
            'scratch': scratch_n,
            'damage': damage_n,
            'min_pair': min_pair,
            key: feasible,
        }
        if not feasible:
            report['ok'] = False
        if need_per_class:
            report['task_upper_bound_hint'] += min_pair // need_per_class

    return report


def _partition_score(base_counts, groups, need_per_class):
    domain_totals = {}
    for gi, group in enumerate(groups):
        totals = {SCRATCH_CLASS: 0, DAMAGE_CLASS: 0}
        for base_id in group:
            for cid in (SCRATCH_CLASS, DAMAGE_CLASS):
                totals[cid] += base_counts.get(base_id, {}).get(cid, 0)
        domain_totals[gi] = totals

    score_min_pair = 10 ** 9
    for totals in domain_totals.values():
        scratch = totals[SCRATCH_CLASS]
        damage = totals[DAMAGE_CLASS]
        if scratch < need_per_class or damage < need_per_class:
            return None
        score_min_pair = min(score_min_pair, min(scratch, damage))
    return score_min_pair


def suggest_greedy_partition(base_counts, train_bases, k_domains, need_per_class=10):
    """Greedy balance by damage load (bottleneck class)."""
    train_bases = sorted(
        [b for b in train_bases if b not in DEFAULT_TEST_BASES],
        key=base_sort_key)

    groups = [[] for _ in range(k_domains)]
    damage_load = [0] * k_domains
    scratch_load = [0] * k_domains

    def _specimen_key(b):
        sc = base_counts.get(b, {}).get(SCRATCH_CLASS, 0)
        dm = base_counts.get(b, {}).get(DAMAGE_CLASS, 0)
        return (-max(sc, dm), -dm, -sc, base_sort_key(b))

    for base_id in sorted(train_bases, key=_specimen_key):
        dm = base_counts.get(base_id, {}).get(DAMAGE_CLASS, 0)
        sc = base_counts.get(base_id, {}).get(SCRATCH_CLASS, 0)
        best_i = min(
            range(k_domains),
            key=lambda i: (damage_load[i] + dm, scratch_load[i] + sc, i))
        groups[best_i].append(base_id)
        damage_load[best_i] += dm
        scratch_load[best_i] += sc

    domain_map = {}
    for domain_id, group in enumerate(groups):
        for base_id in group:
            domain_map[base_id] = domain_id
    for base_id in DEFAULT_TEST_BASES:
        if base_id in base_counts or base_id in train_bases:
            domain_map[base_id] = DEFAULT_TEST_DOMAIN

    score = _partition_score(base_counts, groups, need_per_class)
    return {
        'k_domains': k_domains,
        'groups': groups,
        'domain_map': domain_map,
        'score_min_pair': score,
        'feasible': score is not None,
    }


def search_auto_partitions(base_counts, train_bases, k_min=3, k_max=5, ways=2, shots=5):
    need = 2 * int(shots)
    suggestions = []
    for k in range(int(k_min), int(k_max) + 1):
        proposal = suggest_greedy_partition(base_counts, train_bases, k, need_per_class=need)
        validation = validate_domain_split(
            base_counts, proposal['domain_map'], ways=ways, shots=shots)
        proposal['validation'] = validation
        suggestions.append(proposal)
    suggestions.sort(
        key=lambda x: (
            0 if x['feasible'] else 1,
            -(x['score_min_pair'] or -1),
            -x['k_domains'],
        ))
    return suggestions


def domain_material_summary(registry, domain_map, domain_id):
    counts = defaultdict(int)
    for base_id, did in domain_map.items():
        if did != domain_id:
            continue
        mat = (registry.get(base_id, {}) or {}).get('material_id') or 'AL7075'
        counts[mat] += 1
    if not counts:
        return 'AL7075'
    return max(counts.items(), key=lambda x: x[1])[0]


def registry_rows_to_csv(path, domain_map, registry, train_bases, test_bases):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        ids = sorted(set(domain_map) | set(registry), key=base_sort_key)
        for base_id in ids:
            row = dict(registry.get(base_id, {}))
            row['specimen_id'] = base_id
            row['train_domain'] = str(domain_map.get(base_id, ''))
            if base_id in test_bases:
                row['notes'] = row.get('notes') or 'holdout meta-test'
            writer.writerow({k: row.get(k, '') for k in CSV_COLUMNS})
