"""运行:
  python scripts/split_class_folder.py --input 类别根目录 --output 输出根目录 --train 0.5 --val 0.25 --test 0.25
  python scripts/split_class_folder.py --input 类别根目录 --output 输出根目录 --train 0.5 --val 0.25 --test 0.25 --link hard

把「类别文件夹」按比例拆成 train/val/test。不绑定数据集。
默认硬链接，不复制像素；Windows 也可用 --link copy。
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


IMAGE_SUFFIX = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def setup_stdio():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


def list_images(class_dir):
    return sorted(
        p for p in class_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIX
    )


def place(src, dest, mode):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if mode == 'hard':
        os.link(src, dest)
    elif mode == 'symlink':
        os.symlink(src, dest)
    else:
        shutil.copy2(src, dest)


def main():
    setup_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='类别文件夹根目录的绝对或相对路径')
    parser.add_argument('--output', required=True, help='输出根目录，将创建 train/val/test')
    parser.add_argument('--train', type=float, default=0.5)
    parser.add_argument('--val', type=float, default=0.25)
    parser.add_argument('--test', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--link', choices=('hard', 'symlink', 'copy'), default='hard')
    parser.add_argument('--min-per-split', type=int, default=1, help='每个划分每类至少几张')
    args = parser.parse_args()
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit('train+val+test 必须为 1')
    root = Path(args.input).resolve()
    out = Path(args.output).resolve()
    if not root.is_dir():
        raise SystemExit('输入不存在: {}'.format(root))
    if out == root or root in out.parents:
        raise SystemExit('输出目录不能落在输入目录内')
    rng = random.Random(args.seed)
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit('输入下没有类别子目录')
    summary = {}
    for class_dir in classes:
        files = list_images(class_dir)
        if not files:
            continue
        files = list(files)
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        n_test = n - n_train - n_val
        # 保证有划分时每边至少 min_per_split
        parts = {'train': n_train, 'val': n_val, 'test': n_test}
        for name, ratio in (('train', args.train), ('val', args.val), ('test', args.test)):
            if ratio > 0 and parts[name] < args.min_per_split and n >= args.min_per_split * sum(
                    1 for r in (args.train, args.val, args.test) if r > 0):
                parts[name] = args.min_per_split
        # 重新按比例切，避免上面修补破坏总数时退回简单切分
        n_train = max(parts['train'], args.min_per_split if args.train > 0 else 0)
        n_val = max(parts['val'], args.min_per_split if args.val > 0 else 0)
        if n_train + n_val >= n:
            raise SystemExit('{} 只有 {} 张，无法按当前比例拆分'.format(class_dir.name, n))
        n_test = n - n_train - n_val
        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:],
        }
        summary[class_dir.name] = {k: len(v) for k, v in splits.items()}
        for split_name, group in splits.items():
            for src in group:
                rel = src.relative_to(class_dir)
                dest = out / split_name / class_dir.name / rel
                place(src, dest, args.link)
    print('saved {}'.format(out))
    for name, counts in summary.items():
        print('{} {}'.format(name, counts))


if __name__ == '__main__':
    main()
