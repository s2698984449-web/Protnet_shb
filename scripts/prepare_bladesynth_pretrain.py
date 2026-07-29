"""Deprecated wrapper — use: python scripts/convert_datasets.py --bladesynth"""
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from convert_datasets import convert_bladesynth, build_crop_policy, DEFAULT_CROP_POLICY
import argparse


def main():
    p = argparse.ArgumentParser(description='Use convert_datasets.py --bladesynth instead')
    p.add_argument('--raw_root', default='../data/Dataset_bladesynth')
    p.add_argument('--out_root', default='../data')
    p.add_argument('--no_clean', action='store_true')
    p.add_argument('--expand_ratio', type=float, default=DEFAULT_CROP_POLICY['expand_ratio'])
    p.add_argument('--min_defect_ratio', type=float, default=DEFAULT_CROP_POLICY['min_defect_ratio'])
    p.add_argument('--min_short_side', type=int, default=DEFAULT_CROP_POLICY['min_short_side'])
    p.add_argument('--max_short_side', type=int, default=DEFAULT_CROP_POLICY['max_short_side'])
    a = p.parse_args()
    convert_bladesynth(
        os.path.abspath(a.raw_root), os.path.abspath(a.out_root),
        clean=not a.no_clean, crop_policy=build_crop_policy(a),
    )


if __name__ == '__main__':
    main()
