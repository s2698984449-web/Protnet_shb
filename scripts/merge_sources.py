"""按配置把多个来源的原始类名收成语义类，并按来源分子目录。

    python scripts/merge_sources.py --config config/steel_sources.yaml

输出：

    <out_dir>/{train,val,test}/<语义类>/<来源>/图片
    <mask_dir>/{train,val,test}/<语义类>/<来源>/<stem>.png

一张图上的框若映射到不止一个语义类，整张跳过。没有映射的类名不进入目录。
图片用硬链接。框栅格成掩码，训练仍走原来的 mask_root。
"""

import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cspd.annotate import rasterize_boxes


IMAGE_SUFFIX = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def link_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_files(root):
    return [
        path for path in Path(root).rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIX
    ]


def voc_objects(path):
    root = ET.parse(path).getroot()
    size = root.find('size')
    width = int(size.find('width').text) if size is not None and size.find('width') is not None else 0
    height = int(size.find('height').text) if size is not None and size.find('height') is not None else 0
    names = []
    boxes = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        box = obj.find('bndbox')
        if box is None or not name:
            continue
        names.append(name)
        boxes.append((
            float(box.findtext('xmin')),
            float(box.findtext('ymin')),
            float(box.findtext('xmax')),
            float(box.findtext('ymax')),
        ))
    return width, height, names, boxes


def yolo_objects(path, width, height):
    names = []
    boxes = []
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text:
        return names, boxes
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, bw, bh = (float(parts[i]) for i in range(1, 5))
        except ValueError:
            continue
        names.append(parts[0])
        boxes.append((
            (cx - bw / 2.0) * width,
            (cy - bh / 2.0) * height,
            (cx + bw / 2.0) * width,
            (cy + bh / 2.0) * height,
        ))
    return names, boxes


def semantic_of(names, class_map):
    mapped = []
    for name in names:
        if name not in class_map:
            return None
        mapped.append(class_map[name])
    if len(set(mapped)) != 1:
        return None
    return mapped[0]


def collect(spec):
    kind = spec['kind']
    class_map = {str(key): str(value) for key, value in spec['class_map'].items()}
    image_dir = Path(spec['image_dir'])
    box_dir = Path(spec['box_dir']) if spec.get('box_dir') else None
    skip = set(spec.get('skip_dirs') or [])
    rows = []
    skipped = defaultdict(int)

    if kind == 'folder':
        for class_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
            if class_dir.name in skip or class_dir.name not in class_map:
                skipped[class_dir.name] += 1
                continue
            semantic = class_map[class_dir.name]
            for image in image_files(class_dir):
                rows.append((semantic, image, None))
        return rows, skipped

    if kind == 'folder_voc':
        for class_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
            if class_dir.name in skip or class_dir.name not in class_map:
                continue
            semantic = class_map[class_dir.name]
            for image in image_files(class_dir):
                xml_path = box_dir / (image.stem + '.xml') if box_dir else None
                boxes = None
                if xml_path is not None and xml_path.is_file():
                    _, _, _, boxes = voc_objects(xml_path)
                rows.append((semantic, image, boxes))
        return rows, skipped

    if kind == 'voc_flat':
        for image in image_files(image_dir):
            xml_path = box_dir / (image.stem + '.xml')
            if not xml_path.is_file():
                skipped['no_xml'] += 1
                continue
            _, _, names, boxes = voc_objects(xml_path)
            semantic = semantic_of(names, class_map)
            if semantic is None:
                skipped['multi_or_unknown'] += 1
                continue
            rows.append((semantic, image, boxes))
        return rows, skipped

    if kind == 'folder_yolo':
        for class_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
            if class_dir.name in skip or class_dir.name not in class_map:
                continue
            semantic = class_map[class_dir.name]
            for image in image_files(class_dir):
                boxes = None
                label = box_dir / class_dir.name / (image.stem + '.txt') if box_dir else None
                if label is not None and label.is_file():
                    with Image.open(image) as img:
                        width, height = img.size
                    _, boxes = yolo_objects(label, width, height)
                rows.append((semantic, image, boxes))
        return rows, skipped

    if kind == 'yolo_flat':
        for image in image_files(image_dir):
            label = box_dir / (image.stem + '.txt')
            if not label.is_file():
                skipped['no_label'] += 1
                continue
            with Image.open(image) as img:
                width, height = img.size
            names, boxes = yolo_objects(label, width, height)
            semantic = semantic_of(names, class_map)
            if semantic is None:
                skipped['multi_or_unknown'] += 1
                continue
            rows.append((semantic, image, boxes))
        return rows, skipped

    raise ValueError('未知 kind: {}'.format(kind))


def split_rows(rows, ratios, rng):
    order = list(rows)
    rng.shuffle(order)
    n = len(order)
    n_train = int(round(n * float(ratios[0])))
    n_val = int(round(n * float(ratios[1])))
    if n_train + n_val > n:
        n_val = n - n_train
    return {
        'train': order[:n_train],
        'val': order[n_train:n_train + n_val],
        'test': order[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out_dir = Path(cfg['out_dir'])
    mask_dir = Path(cfg['mask_dir'])
    ratios = cfg.get('ratios') or [0.6, 0.2, 0.2]
    rng = random.Random(int(cfg.get('seed', 0)))
    report = {'sources': {}, 'splits': {}}

    grouped = defaultdict(list)
    for spec in cfg['sources']:
        rows, skipped = collect(spec)
        report['sources'][spec['name']] = {
            'kept': len(rows),
            'skipped': dict(skipped),
        }
        for semantic, image, boxes in rows:
            grouped[(semantic, spec['name'])].append((image, boxes))

    for (semantic, source), rows in sorted(grouped.items()):
        parts = split_rows(rows, ratios, rng)
        for split, items in parts.items():
            report['splits'].setdefault(split, {})
            report['splits'][split].setdefault(semantic, {})
            report['splits'][split][semantic][source] = len(items)
            for image, boxes in items:
                stem = '{}_{}'.format(source, image.stem)
                dst = out_dir / split / semantic / source / (stem + image.suffix.lower())
                link_file(image, dst)
                if not boxes:
                    continue
                with Image.open(image) as img:
                    width, height = img.size
                mask = rasterize_boxes(width, height, boxes)
                mask_path = mask_dir / split / semantic / source / (stem + '.png')
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                mask.save(mask_path)

    report_path = out_dir / 'merge_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
