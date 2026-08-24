"""生成 MVTEC test / GC10 / NEU 的 meta-train vs meta-test split.

每个 defect 类内部按 80/20 切 (seed=42), 输出 JSON list of relative paths.

注意: 这里 'meta-test' 不是元 test, 是
  - meta-train pool: 用于 Phase 1 episode meta-training (作为 K-shot/Q-query 候选)
  - meta-test pool : 用于 Phase 1 同分布评估 (验证模型在 seen domain 内的泛化)
真正 cross-domain meta-test (汽车/航天/Kolektor) 是 Phase 2.
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = r'd:\元强化学习\代码\data'
SPLITS_DIR = r'd:\元强化学习\代码\data\splits'
os.makedirs(SPLITS_DIR, exist_ok=True)
SEED = 42
TRAIN_RATIO = 0.8


def _split_files(files, seed):
    rng = random.Random(seed)
    rng.shuffle(files)
    n_train = max(1, int(len(files) * TRAIN_RATIO))
    if len(files) >= 5:  # 至少留 1 张给 test
        n_train = min(n_train, len(files) - 1)
    return files[:n_train], files[n_train:]


def split_gc10():
    """GC10 cls_8class/ — 8 个 defect folder, 每个 80/20."""
    root = os.path.join(DATA_ROOT, 'GC10-DET', 'cls_8class')
    if not os.path.isdir(root):
        print(f'[skip] GC10 not found at {root}')
        return False
    folders = sorted(f for f in os.listdir(root)
                     if os.path.isdir(os.path.join(root, f)))
    splits = {'meta_train': [], 'meta_test': []}
    print(f'GC10: {len(folders)} classes')
    for f in folders:
        files = sorted([x for x in os.listdir(os.path.join(root, f))
                        if x.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        n_total = len(files)
        tr, te = _split_files(files, SEED)
        print(f'  {f:<25} total={n_total:<4} train={len(tr):<3} test={len(te):<3}')
        for x in tr: splits['meta_train'].append(f'{f}/{x}')
        for x in te: splits['meta_test'].append(f'{f}/{x}')
    out = os.path.join(SPLITS_DIR, 'gc10_split.json')
    with open(out, 'w') as fp:
        json.dump(splits, fp, indent=2)
    print(f'  -> saved {out} (train={len(splits["meta_train"])} test={len(splits["meta_test"])})')
    return True


def split_neu():
    """NEU NEU-CLS/ — 按 prefix (rolled/scratched/...) 分组, 每个 80/20."""
    root = os.path.join(DATA_ROOT, 'NEU-CLS', 'NEU-CLS')
    if not os.path.isdir(root):
        print(f'[skip] NEU not found at {root}')
        return False
    files = sorted([f for f in os.listdir(root) if f.lower().endswith('.bmp')])
    by_class = {}
    for f in files:
        prefix = f.split('_')[0]
        by_class.setdefault(prefix, []).append(f)
    splits = {'meta_train': [], 'meta_test': []}
    print(f'NEU: {len(by_class)} classes')
    for cls, cls_files in sorted(by_class.items()):
        tr, te = _split_files(cls_files, SEED)
        print(f'  {cls:<15} total={len(cls_files):<4} train={len(tr):<3} test={len(te):<3}')
        for x in tr: splits['meta_train'].append(x)
        for x in te: splits['meta_test'].append(x)
    out = os.path.join(SPLITS_DIR, 'neu_split.json')
    with open(out, 'w') as fp:
        json.dump(splits, fp, indent=2)
    print(f'  -> saved {out} (train={len(splits["meta_train"])} test={len(splits["meta_test"])})')
    return True


def split_mvtec_test():
    """MVTEC test/{defects}/ — 按 (product, defect) 内部 80/20 切."""
    from common.multi_domain_datasets import MVTEC_PRODUCTS
    root = os.path.join(DATA_ROOT, 'MVTEC')
    if not os.path.isdir(root):
        print(f'[skip] MVTEC not found at {root}')
        return False
    splits = {'meta_train': [], 'meta_test': [], 'usable_products': [],
               'adaptive_products': []}  # n_way < 5 的产品, sampler 自动按 min(N, n_classes)
    print('MVTEC test:')
    for product in MVTEC_PRODUCTS:
        test_dir = os.path.join(root, product, 'test')
        if not os.path.isdir(test_dir):
            continue
        defects = sorted(d for d in os.listdir(test_dir)
                         if os.path.isdir(os.path.join(test_dir, d)))
        n_total = len(defects)
        # 至少 4 类才进 episode pool (4-way 5-shot 是下限)
        if n_total < 4:
            tag = '[SKIP]'
        else:
            tag = '[OK]'
            splits['usable_products'].append(product)
            if n_total < 5:
                splits['adaptive_products'].append(product)
                tag = '[ADAPTIVE]'
        print(f'  {product:<15} n_defect_types={n_total:<3} {tag}')
        for d in defects:
            defect_dir = os.path.join(test_dir, d)
            files = sorted([x for x in os.listdir(defect_dir)
                            if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
            n_total_files = len(files)
            if n_total_files < 2:
                # 文件数 < 2 → 全放 train, 不切 (避免空 test)
                for x in files:
                    splits['meta_train'].append(f'{product}/test/{d}/{x}')
                continue
            tr, te = _split_files(files, SEED)
            for x in tr: splits['meta_train'].append(f'{product}/test/{d}/{x}')
            for x in te: splits['meta_test'].append(f'{product}/test/{d}/{x}')
    out = os.path.join(SPLITS_DIR, 'mvtec_test_split.json')
    with open(out, 'w') as fp:
        json.dump(splits, fp, indent=2)
    print(f'  -> saved {out} (train={len(splits["meta_train"])} test={len(splits["meta_test"])})')
    print(f'  -> usable_products ({len(splits["usable_products"])}): {splits["usable_products"]}')
    print(f'  -> adaptive_products ({len(splits["adaptive_products"])}): {splits["adaptive_products"]}')
    return True


def split_carpaint():
    """车漆 CMPL 域 — cardd/carparts/syndcar 各自的 80/20 切.

    路径预期:
      <DATA_ROOT>/carpaint/cardd_cls/{train,val}/<class>/*.jpg
      <DATA_ROOT>/carpaint/task_c_carparts_damage/<ClassTitle>/*.jpg
      <DATA_ROOT>/carpaint/task_e_syndcar/<class>/*.jpg

    若目录整体不存在, 跳过; 单个 task 缺失, 仅跳过该 task.
    输出 JSON 格式: {meta_train: [task/class/file.jpg], meta_test: [...]}
    """
    import os as _os
    root = _os.path.join(DATA_ROOT, '车', 'carpaint')
    if not _os.path.isdir(root):
        print(f'[skip] carpaint root not found at {root}')
        return False

    from common.multi_domain_datasets import CARPAINT_DIR_LAYOUT

    splits = {'meta_train': [], 'meta_test': []}
    for task, sub in CARPAINT_DIR_LAYOUT.items():
        task_root = _os.path.join(root, sub)
        if not _os.path.isdir(task_root):
            print(f'[skip] task {task}: {task_root} not found')
            continue

        if task == 'cardd':
            # 嵌套 train/val/<class>
            classes = []
            for split_name in ('train', 'val'):
                split_root = _os.path.join(task_root, split_name)
                if not _os.path.isdir(split_root):
                    continue
                for cls_name in sorted(_os.listdir(split_root)):
                    cls_dir = _os.path.join(split_root, cls_name)
                    if _os.path.isdir(cls_dir):
                        files = sorted([x for x in _os.listdir(cls_dir)
                                        if x.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        classes.append((cls_name, files))
            n_total = sum(len(f) for _, f in classes)
            print(f'carpaint/{task} total={n_total} classes={len(classes)}')
        else:
            classes = []
            for cls_name in sorted(_os.listdir(task_root)):
                cls_dir = _os.path.join(task_root, cls_name)
                if not _os.path.isdir(cls_dir):
                    continue
                lookup = cls_name
                if task == 'syndcar' and lookup.startswith('sd_'):
                    lookup = lookup[3:]
                files = sorted([x for x in _os.listdir(cls_dir)
                                if x.lower().endswith(('.jpg', '.jpeg', '.png'))])
                classes.append((lookup, files))
            n_total = sum(len(f) for _, f in classes)
            print(f'carpaint/{task} total={n_total} classes={len(classes)}')

        for cls_name, files in classes:
            tr, te = _split_files(files, SEED)
            for x in tr:
                splits['meta_train'].append(f'{task}/{cls_name}/{x}')
            for x in te:
                splits['meta_test'].append(f'{task}/{cls_name}/{x}')

    out = _os.path.join(SPLITS_DIR, 'carpaint_split.json')
    with open(out, 'w') as fp:
        json.dump(splits, fp, indent=2)
    print(f'  -> saved {out} '
          f'(train={len(splits["meta_train"])} test={len(splits["meta_test"])})')
    return True


def split_aero_real():
    """航天真实数据 — YOLO .txt 第一列类 ID 映射 (仅保留 train ∩ val 类).

    路径预期: <DATA_ROOT>/航天/raw/Aero-engine_defect-detect_new/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt  (YOLO: 'cls_id x y w h')

    类交集: YOLO id=0 → scratch, id=3 → damage. 其余 id 跳过.
    meta_train = 原 train/, meta_test = 原 val/ (尊重原划分).
    """
    import os as _os
    root = _os.path.join(DATA_ROOT, '航天', 'raw', 'Aero-engine_defect-detect_new')
    if not _os.path.isdir(root):
        print(f'[skip] aero_real root not found at {root}')
        return False

    img_root = _os.path.join(root, 'images')
    lbl_root = _os.path.join(root, 'labels')
    if not _os.path.isdir(img_root):
        print(f'[skip] aero_real images dir missing')
        return False

    ID_MAP = {0: 'scratch', 3: 'damage'}

    splits = {'meta_train': [], 'meta_test': []}
    class_counter = {}
    for sub in ('train', 'val'):
        sub_img = _os.path.join(img_root, sub)
        sub_lbl = _os.path.join(lbl_root, sub)
        if not _os.path.isdir(sub_img):
            continue
        files = sorted([f for f in _os.listdir(sub_img)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        kept = 0
        for f in files:
            base, _ = _os.path.splitext(f)
            txt_path = _os.path.join(sub_lbl, base + '.txt')
            if not _os.path.isfile(txt_path):
                continue
            yolo_id = None
            try:
                with open(txt_path, encoding='utf-8') as fp:
                    first = fp.readline().strip().split()
                    if first and first[0].isdigit():
                        yolo_id = int(first[0])
            except Exception:
                continue
            if yolo_id is None or yolo_id not in ID_MAP:
                continue
            cls_name = ID_MAP[yolo_id]
            class_counter.setdefault(sub, {})
            class_counter[sub][cls_name] = class_counter[sub].get(cls_name, 0) + 1
            splits['meta_train' if sub == 'train' else 'meta_test'].append(f'{sub}/{f}')
            kept += 1
        print(f'aero_real/{sub} total_kept={kept}/{len(files)} classes={class_counter.get(sub, {})}')

    out = _os.path.join(SPLITS_DIR, 'aero_real_split.json')
    with open(out, 'w') as fp:
        json.dump(splits, fp, indent=2)
    print(f'  -> saved {out} '
          f'(train={len(splits["meta_train"])} test={len(splits["meta_test"])})')
    return True


def split_mvtec_dropout():
    """MVTEC 被剔除产品 (默认 toothbrush) — 复用 mvtec_test_split.

    路径预期: <DATA_ROOT>/MVTEC/<product>/test/{defect}/<file>.*
    这里不写 split 文件, 因为 MVTEC dropout 复用 mvtec_test_split.json 已有的
    (product, defect, file) 切分. 仅打印 summary.

    如果未来需要独立的 split 策略 (e.g. 80/20 重洗), 在这里实现.
    """
    import os as _os
    from common.multi_domain_datasets import MVTEC_DROPOUT_PRODUCTS
    mvtec_root = _os.path.join(DATA_ROOT, 'MVTEC')
    if not _os.path.isdir(mvtec_root):
        print(f'[skip] MVTEC root not found at {mvtec_root}')
        return False

    total = 0
    for prod in MVTEC_DROPOUT_PRODUCTS:
        test_dir = _os.path.join(mvtec_root, prod, 'test')
        if not _os.path.isdir(test_dir):
            print(f'[skip] dropout product {prod}: not found')
            continue
        n = 0
        for d in _os.listdir(test_dir):
            d_path = _os.path.join(test_dir, d)
            if not _os.path.isdir(d_path):
                continue
            for f in _os.listdir(d_path):
                if _os.path.isfile(_os.path.join(d_path, f)) \
                        and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    n += 1
        total += n
        print(f'mvtec_dropout/{prod}: {n} test files (uses mvtec_test_split.json)')
    print(f'  -> total dropout samples: {total}; no separate split written.')
    return True


if __name__ == '__main__':
    print('=' * 70)
    print('Phase 1 episode pool split (meta-train vs meta-test, seed=42, 80/20)')
    print('=' * 70)
    split_gc10()
    print()
    split_neu()
    print()
    split_mvtec_test()
    print()
    split_carpaint()
    print()
    split_aero_real()
    print()
    split_mvtec_dropout()
    print()
    print('All splits written to {}'.format(SPLITS_DIR))
    print(f'Splits saved under {SPLITS_DIR}')