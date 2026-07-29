"""Validate aviation domain folders before training."""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.aviation_datasets import AviationDefectDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_path', default='../data')
    parser.add_argument('--domains', default='0,1,2,3')
    args = parser.parse_args()

    for d in [int(x) for x in args.domains.split(',')]:
        ds = AviationDefectDataset(d, os.path.abspath(args.data_dir_path), train=False, rgb_only=True)
        counts = {}
        for _, label in ds.samples:
            counts[label] = counts.get(label, 0) + 1
        print('domain_{}: {} samples, class counts {}'.format(d, len(ds), counts))
        (rgb, th, us), y = ds[0]
        print('  sample0 rgb{} thermal{} us{} label{}'.format(
            tuple(rgb.shape), tuple(th.shape), tuple(us.shape), y.item()))


if __name__ == '__main__':
    main()
