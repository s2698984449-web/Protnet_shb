"""按配置把多个来源的原始类名收成语义类，并按来源分子目录。

    python scripts/merge_sources.py --config config/cspd/steel_sources.yaml

输出：

    layout=split_first：<out_dir>/{train,val,test}/<语义类>/<来源>/图片
    layout=class_source_split：<out_dir>/<语义类>/<来源>/{train,val,test}/图片
        掩码在同一划分的 mask/ 下。课程学习读这种目录。

一张图上的框若映射到不止一个语义类，整张跳过。没有映射的类名不进入目录。
图片用硬链接。划分按组：同一 group 的图进同一个 train/val/test。
组号来自配置里的正则；没有正则时用 32×32 灰度直方图，距离小于阈值的图为一组。
"""

import argparse
import json
import os
import random
import re
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
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIX
        and path.parent.name != 'mask'
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
    class_map = {str(key): str(value) for key, value in (spec.get('class_map') or {}).items()}
    image_dir = Path(spec['image_dir'])
    box_dir = Path(spec['box_dir']) if spec.get('box_dir') else None
    skip = set(spec.get('skip_dirs') or [])
    rows = []
    skipped = defaultdict(int)

    if kind == 'masked_folder':
        semantic = str(spec['semantic'])
        for image in image_files(image_dir):
            mask = image.parent / 'mask' / (image.stem + '.png')
            payload = ('maskfile', mask) if mask.is_file() else None
            if spec.get('require_mask') and payload is None:
                skipped['no_mask'] += 1
                continue
            rows.append((semantic, image, payload))
        return rows, skipped

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


def thumb_hist(path):
    with Image.open(path) as image:
        gray = image.convert('L').resize((32, 32))
    values = list(gray.getdata())
    hist = [0] * 8
    for value in values:
        hist[min(7, value * 8 // 256)] += 1
    total = float(sum(hist) or 1)
    return tuple(count / total for count in hist)


def group_ids_for(rows, spec):
    """同一组必须进同一个划分。正则优先；否则用缩略图直方图。"""
    group = spec.get('group') or {}
    mode = str(group.get('mode') or 'hist')
    if mode == 'regex':
        pattern = re.compile(str(group['pattern']))
        ids = []
        for image, _boxes in rows:
            match = pattern.search(image.stem)
            ids.append(match.group(1) if match else image.stem)
        return ids
    if mode == 'stem':
        return [image.stem for image, _boxes in rows]
    threshold = float(group.get('threshold', 0.3))
    centers = []
    counts = []
    ids = []
    for image, _boxes in rows:
        hist = thumb_hist(image)
        best_i = None
        best_d = threshold
        for index, center in enumerate(centers):
            dist = sum(abs(a - b) for a, b in zip(hist, center))
            if dist < best_d:
                best_i = index
                best_d = dist
        if best_i is None:
            centers.append(hist)
            counts.append(1)
            ids.append('h{}'.format(len(centers) - 1))
        else:
            count = counts[best_i]
            centers[best_i] = tuple(
                (centers[best_i][dim] * count + hist[dim]) / (count + 1)
                for dim in range(8))
            counts[best_i] = count + 1
            ids.append('h{}'.format(best_i))
    return ids


def split_rows(rows, ratios, rng, group_ids=None):
    if not group_ids:
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
        }, {}
    buckets = defaultdict(list)
    for row, gid in zip(rows, group_ids):
        buckets[gid].append(row)
    groups = list(buckets.items())
    rng.shuffle(groups)
    n = len(rows)
    quota = {
        'train': n * float(ratios[0]),
        'val': n * float(ratios[1]),
        'test': n * float(ratios[2]),
    }
    parts = {'train': [], 'val': [], 'test': []}
    assigned = {}
    for gid, group in groups:
        key = max(quota, key=quota.get)
        parts[key].extend(group)
        quota[key] -= len(group)
        for image, _boxes in group:
            assigned[image.stem] = gid
    return parts, assigned


def write_mask(image, boxes, mask_path):
    if not boxes:
        return
    if isinstance(boxes, tuple) and boxes and boxes[0] == 'maskfile':
        if boxes[1].is_file():
            link_file(boxes[1], mask_path)
        return
    with Image.open(image) as img:
        width, height = img.size
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    rasterize_boxes(width, height, boxes).save(mask_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out_dir = Path(cfg['out_dir'])
    layout = str(cfg.get('layout') or 'split_first')
    mask_dir = Path(cfg['mask_dir']) if cfg.get('mask_dir') else None
    ratios = cfg.get('ratios') or [0.6, 0.2, 0.2]
    rng = random.Random(int(cfg.get('seed', 0)))
    report = {'layout': layout, 'sources': {}, 'splits': {}, 'groups': {}}
    default_group = cfg.get('group') or {'mode': 'hist', 'threshold': 0.3}
    specs = {}
    source_specs = []
    for spec in cfg['sources']:
        spec = dict(spec)
        spec['group'] = spec.get('group') or default_group
        specs[spec['name']] = spec
        source_specs.append(spec)

    grouped = defaultdict(list)
    for spec in source_specs:
        rows, skipped = collect(spec)
        label = spec['name']
        if spec.get('semantic'):
            label = '{}:{}'.format(spec['name'], spec['semantic'])
        report['sources'][label] = {
            'kept': len(rows),
            'skipped': dict(skipped),
        }
        for semantic, image, boxes in rows:
            grouped[(semantic, spec['name'])].append((image, boxes))

    for (semantic, source), rows in sorted(grouped.items()):
        gids = group_ids_for(rows, specs[source])
        parts, assigned = split_rows(rows, ratios, rng, gids)
        n_groups = len(set(gids))
        report['groups']['{}/{}'.format(semantic, source)] = {
            'images': len(rows),
            'groups': n_groups,
        }
        saved_groups = {}
        for split, items in parts.items():
            report['splits'].setdefault(split, {})
            report['splits'][split].setdefault(semantic, {})
            report['splits'][split][semantic][source] = len(items)
            for image, boxes in items:
                stem = '{}_{}'.format(source, image.stem)
                if layout == 'class_source_split':
                    dst = out_dir / semantic / source / split / (stem + image.suffix.lower())
                    mask_path = dst.parent / 'mask' / (stem + '.png')
                else:
                    dst = out_dir / split / semantic / source / (stem + image.suffix.lower())
                    mask_path = mask_dir / split / semantic / source / (stem + '.png')
                link_file(image, dst)
                write_mask(image, boxes, mask_path)
                saved_groups[stem] = assigned.get(image.stem, image.stem)
        if layout == 'class_source_split':
            group_path = out_dir / semantic / source / 'groups.json'
            group_path.parent.mkdir(parents=True, exist_ok=True)
            group_path.write_text(
                json.dumps(saved_groups, ensure_ascii=False, indent=2), encoding='utf-8')

    report_path = out_dir / 'merge_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
