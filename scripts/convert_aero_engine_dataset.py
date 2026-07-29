"""Deprecated wrapper — use: python scripts/convert_datasets.py --real"""
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from convert_datasets import convert_real, build_crop_policy, DEFAULT_CROP_POLICY
import argparse


def main():
    p = argparse.ArgumentParser(description='Use convert_datasets.py --real instead')
    p.add_argument('--raw_root', default='../data/Aero-engine_defect-detect_new')
    p.add_argument('--out_root', default='../data')
    p.add_argument('--test_bases', default='20,21,22,23,24,25,26')
    p.add_argument('--no_clean', action='store_true')
    p.add_argument('--no_crop', action='store_true')
    p.add_argument('--expand_ratio', type=float, default=DEFAULT_CROP_POLICY['expand_ratio'])
    p.add_argument('--min_defect_ratio', type=float, default=DEFAULT_CROP_POLICY['min_defect_ratio'])
    p.add_argument('--min_short_side', type=int, default=DEFAULT_CROP_POLICY['min_short_side'])
    p.add_argument('--max_short_side', type=int, default=DEFAULT_CROP_POLICY['max_short_side'])
    a = p.parse_args()
    convert_real(
        os.path.abspath(a.raw_root), os.path.abspath(a.out_root),
        test_bases={b.strip() for b in a.test_bases.split(',') if b.strip()},
        clean=not a.no_clean, crop_policy=build_crop_policy(a), use_crop=not a.no_crop,
    )


if __name__ == '__main__':
    main()
