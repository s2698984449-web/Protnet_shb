"""
Feature-space diagnosis before any MAML run.

Run from 航空缺陷检测/:
    python scripts/diagnose_features.py
    python scripts/diagnose_features.py --track automotive
    python scripts/diagnose_features.py --track aerospace

Outputs under reports/diagnose/{track}/:
    tsne_domain3.png, tsne_train_domains.png, diagnose_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ----- paths (script lives in 航空缺陷检测/scripts/) -----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.backbones import YOLO26BackboneEncoder  # noqa: E402
from common.project_paths import resolve_track  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 128
YOLO_IMGSZ = 640
SHOTS = 5
QUERY = SHOTS  # MAML split: 10 support + 10 query per 2-way task

TRACK_CONFIGS = {
    'aerospace': {
        'class_dir_map': {
            '0_scratch': 0,
            '1_contamination': 1,
            '3_damage': 2,
        },
        'class_id_to_name': {0: 'scratch', 1: 'contamination', 2: 'damage'},
        'binary_class_ids': (0, 2),
        'binary_label': 'scratch vs damage',
        'train_domains': (0, 1, 2),
        'test_domain': 3,
        'train_domains_tag': 'domain_0-2',
    },
    'automotive': {
        'class_dir_map': {
            '0_scratch': 0,
            '3_dent': 2,
        },
        'class_id_to_name': {0: 'scratch', 2: 'dent'},
        'binary_class_ids': (0, 2),
        'binary_label': 'scratch vs dent',
        'train_domains': (0,),
        'test_domain': 3,
        'train_domains_tag': 'domain_0 (ASDE)',
    },
}

RGB_NAMES = ('rgb.jpg', 'rgb.png', 'RGB.jpg', 'RGB.png')


def _find_rgb_path(sample_dir):
    for name in RGB_NAMES:
        path = sample_dir / name
        if path.is_file():
            return path
    return None


def _resolve_track_context(track_name):
    track_name = track_name.strip().lower()
    if track_name not in TRACK_CONFIGS:
        raise ValueError('Unknown track: {!r}. Use aerospace or automotive.'.format(track_name))
    info = resolve_track(track_name)
    cfg = TRACK_CONFIGS[track_name]
    meta_root = Path(info['processed']) / info['dataset_name']
    weights_dir = Path(info['weights_dir'])
    weights = {
        'pretrained': weights_dir / info['pretrain_weights_name'],
        'yolo26n': weights_dir / info['yolo_init_name'],
    }
    output_dir = PROJECT_DIR / 'reports' / 'diagnose' / info['track']
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        'track': info['track'],
        'meta_root': meta_root,
        'weights': weights,
        'output_dir': output_dir,
        **cfg,
    }


def count_effective_tasks(ctx, shots=SHOTS, query=QUERY):
    """Count ROI samples and independent 2-way task upper bound per domain."""
    meta_root = ctx['meta_root']
    class_dir_map = ctx['class_dir_map']
    train_domains = ctx['train_domains']
    pos_name = ctx['class_id_to_name'][ctx['binary_class_ids'][1]]

    print('\n' + '=' * 50)
    print('有效 Task 数量统计 ({}, 2-way {}-shot)'.format(ctx['binary_label'], shots))
    print('=' * 50)
    print('  每 task 每类需 support+query = {} 张'.format(shots + query))

    per_class_need = shots + query
    results = {}
    domain_ids = sorted(set(train_domains) | {ctx['test_domain']})

    for domain_id in domain_ids:
        domain_dir = meta_root / 'domain_{}'.format(domain_id)
        if not domain_dir.is_dir():
            continue

        counts = {}
        for dir_name, cls_id in class_dir_map.items():
            cls_dir = domain_dir / dir_name
            if not cls_dir.is_dir():
                continue
            n = sum(1 for d in cls_dir.iterdir() if d.is_dir() and _find_rgb_path(d))
            counts[cls_id] = n

        results[domain_id] = counts
        scratch_n = counts.get(0, 0)
        pos_n = counts.get(2, 0)
        bottleneck = min(scratch_n, pos_n)
        indep_tasks = bottleneck // per_class_need

        extra = ''
        if 1 in counts:
            extra = ', contamination={}'.format(counts.get(1, 0))
        print('  domain_{}: scratch={}, {}={}{}'.format(
            domain_id, scratch_n, pos_name, pos_n, extra))
        print('    → {} 独立 task 上界: {}//{} = {}'.format(
            ctx['binary_label'], bottleneck, per_class_need, indep_tasks))

    scratch_total = sum(results.get(d, {}).get(0, 0) for d in train_domains)
    pos_total = sum(results.get(d, {}).get(2, 0) for d in train_domains)
    bottleneck_combined = min(scratch_total, pos_total)
    indep_combined = bottleneck_combined // per_class_need

    print('\n  {} 合并: scratch={}, {}={}'.format(
        ctx['train_domains_tag'], scratch_total, pos_name, pos_total))
    print('    → 合并后独立 task 上界: {}//{} = {}'.format(
        bottleneck_combined, per_class_need, indep_combined))

    summary = {
        'per_domain': {
            str(k): {
                'scratch': v.get(0, 0),
                'contamination': v.get(1, 0),
                'positive_class': v.get(2, 0),
                'independent_tasks_binary': min(v.get(0, 0), v.get(2, 0)) // per_class_need,
            }
            for k, v in results.items()
        },
        'combined_train_domains': {
            'scratch': scratch_total,
            'positive_class': pos_total,
            'independent_tasks_upper_bound': indep_combined,
        },
        'test_domain': results.get(ctx['test_domain'], {}),
    }

    if indep_combined < 5:
        print('\n  ⚠️ 严重不足：task 多样性几乎为零，MAML 难以学到可迁移适配')
    elif indep_combined < 20:
        print('\n  △ 任务多样性不足：需增数据或改训练范式')

    return summary


def load_domain_rois(domain_ids, meta_root, class_dir_map, class_id_to_name):
    """Load ROI tensors with the same RGB pipeline as AviationDefectDataset (256, no aug)."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    imgs, labels, domains_out = [], [], []

    for domain_id in domain_ids:
        domain_dir = meta_root / 'domain_{}'.format(domain_id)
        if not domain_dir.is_dir():
            print('  ⚠️ domain_{} 不存在，跳过'.format(domain_id))
            continue

        for dir_name, cls_id in class_dir_map.items():
            cls_dir = domain_dir / dir_name
            if not cls_dir.is_dir():
                continue
            for sample_dir in sorted(cls_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                rgb_path = _find_rgb_path(sample_dir)
                if rgb_path is None:
                    continue
                img = Image.open(rgb_path).convert('RGB')
                imgs.append(transform(img))
                labels.append(cls_id)
                domains_out.append(domain_id)

    if not imgs:
        raise RuntimeError('未找到 ROI 图像，请检查: {}'.format(meta_root))

    print('  加载完成：{} 张 ROI (domains {})'.format(len(imgs), domain_ids))
    labels_arr = np.array(labels)
    for cls_id, name in class_id_to_name.items():
        n = int(np.sum(labels_arr == cls_id))
        if n > 0:
            print('    {}: {}'.format(name, n))

    return torch.stack(imgs), labels_arr, np.array(domains_out)


def load_backbone(weight_path):
    return YOLO26BackboneEncoder(
        out_dim=EMBED_DIM,
        imgsz=YOLO_IMGSZ,
        weights_path=str(weight_path),
    ).to(DEVICE)


def extract_features(backbone, imgs, batch_size=8):
    backbone.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i + batch_size].to(DEVICE)
            chunks.append(backbone.encode(batch).cpu().float().numpy())
    return np.concatenate(chunks, axis=0)


def linear_separability(features, labels, binary_class_ids, desc='', verbose=True):
    """LogReg CV on binary classes; other classes ignored."""
    mask = np.isin(labels, binary_class_ids)
    X = features[mask]
    y = labels[mask]
    y_binary = (y == 2).astype(int)

    n_scratch = int(np.sum(y_binary == 0))
    n_damage = int(np.sum(y_binary == 1))

    if n_scratch == 0 or n_damage == 0:
        if verbose:
            print('  {}: 缺少某一类，无法计算'.format(desc))
        return 0.0, 0.0, n_scratch, n_damage

    X_scaled = StandardScaler().fit_transform(X)
    n_splits = min(5, n_scratch, n_damage)
    if n_splits < 2:
        if verbose:
            print('  {}: 样本太少 (scratch={}, damage={})'.format(
                desc, n_scratch, n_damage))
        return 0.0, 0.0, n_scratch, n_damage

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    scores = cross_val_score(clf, X_scaled, y_binary, cv=cv, scoring='accuracy')
    mean, std = float(scores.mean()), float(scores.std())

    if verbose:
        print('  {}: {:.3f} ± {:.3f} (scratch={}, damage={}, {}-fold)'.format(
            desc, mean, std, n_scratch, n_damage, n_splits))

    return mean, std, n_scratch, n_damage


def _run_tsne(X, perplexity):
    """Compatible with sklearn 0.24 (n_iter) and sklearn 1.x (max_iter)."""
    base = dict(n_components=2, random_state=42, perplexity=perplexity, init='pca')
    try:
        return TSNE(n_iter=1000, learning_rate=200, **base).fit_transform(X)
    except TypeError:
        return TSNE(max_iter=1000, learning_rate='auto', **base).fit_transform(X)


def plot_tsne(features_dict, labels, binary_class_ids, pos_name, title, save_path):
    mask = np.isin(labels, binary_class_ids)
    y_binary = (labels[mask] == binary_class_ids[1]).astype(int)

    n_weights = len(features_dict)
    fig, axes = plt.subplots(1, n_weights, figsize=(6 * n_weights, 5))
    if n_weights == 1:
        axes = [axes]

    for ax, (wname, feats) in zip(axes, features_dict.items()):
        X = feats[mask]
        perp = min(30, max(2, len(X) - 1))
        X_2d = _run_tsne(X, perp)

        for cls_bin, color, cname in ((0, 'steelblue', 'scratch'), (1, 'tomato', pos_name)):
            cm = y_binary == cls_bin
            ax.scatter(
                X_2d[cm, 0], X_2d[cm, 1], c=color,
                label='{} (n={})'.format(cname, cm.sum()),
                alpha=0.75, s=55, edgecolors='white', linewidths=0.4)

        sep_mean, sep_std, _, _ = linear_separability(
            feats, labels, binary_class_ids, desc=wname, verbose=False)
        ax.set_title('{}\nLogReg: {:.3f}±{:.3f}'.format(wname, sep_mean, sep_std), fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print('  图保存：{}'.format(save_path))
    plt.close()


def decision(sep_d3, sep_train, train_tag):
    print('\n' + '=' * 50)
    print('决策建议 (基于 pretrained 或首个可用权重)')
    print('=' * 50)
    print('  test domain  可分性: {:.3f}'.format(sep_d3))
    print('  {} 可分性: {:.3f}'.format(train_tag, sep_train))

    thresh_low, thresh_high = 0.60, 0.70
    d3_low = sep_d3 < thresh_low
    d3_high = sep_d3 >= thresh_high
    train_low = sep_train < thresh_low
    train_high = sep_train >= thresh_high

    if train_high and d3_low:
        print('\n  → 情况 A：训练域可分，测试域不可分 → 域偏移为主 → Phase0-v2')
    elif train_low and d3_low:
        print('\n  → 情况 B：两域都不可分 → backbone 无区分度 → Phase0-v2（MAML 暂缓）')
    elif train_high and d3_high:
        print('\n  → 情况 C：两域都可分 → P1-1 解冻 MAML, fast_lr ∈ [0.01, 0.02, 0.03]')
    else:
        print('\n  → 情况 D：弱可分 → Phase0-v2 + P1-1 并行')

    if sep_d3 < 0.55:
        print('\n  ⚠️ test domain 接近随机 (<0.55)，须先解决特征再谈 MAML')


def parse_args():
    parser = argparse.ArgumentParser(description='Feature-space diagnosis (LogReg + t-SNE)')
    parser.add_argument('--track', default='aerospace',
                        help='aerospace|automotive (or 航天|车)')
    return parser.parse_args()


def main():
    args = parse_args()
    ctx = _resolve_track_context(args.track)

    print('设备: {}'.format(DEVICE))
    print('track: {}'.format(ctx['track']))
    print('数据: {}'.format(ctx['meta_root']))
    print('输出: {}'.format(ctx['output_dir']))

    if not ctx['meta_root'].is_dir():
        raise RuntimeError(
            '数据目录不存在: {}\n请先转换数据，例如:\n'
            '  automotive: python scripts/convert_cmpl_datasets.py --pretrain --meta --meta_policy track2 --check\n'
            '  aerospace:  (processed/aviation under data/航天/processed/)'.format(
                ctx['meta_root']))

    task_summary = count_effective_tasks(ctx)
    train_domains = list(ctx['train_domains'])
    test_domain = ctx['test_domain']
    pos_name = ctx['class_id_to_name'][ctx['binary_class_ids'][1]]

    print('\n' + '=' * 50)
    print('加载 ROI 数据')
    print('=' * 50)
    print('  domain_{} (测试域)...'.format(test_domain))
    imgs_d3, labels_d3, _ = load_domain_rois(
        [test_domain], ctx['meta_root'], ctx['class_dir_map'], ctx['class_id_to_name'])
    print('  {} (训练域)...'.format(ctx['train_domains_tag']))
    imgs_train, labels_train, _ = load_domain_rois(
        train_domains, ctx['meta_root'], ctx['class_dir_map'], ctx['class_id_to_name'])

    print('\n' + '=' * 50)
    print('特征提取 + 线性可分性')
    print('=' * 50)

    features_d3 = {}
    features_train = {}
    sep_scores = {}

    for wname, wpath in ctx['weights'].items():
        if not wpath.is_file():
            print('  ⚠️ 权重不存在: {}，跳过'.format(wpath))
            continue

        print('\n[{}] {}'.format(wname, wpath.name))
        backbone = load_backbone(wpath)

        print('  提取 domain_3 特征...')
        feats_d3 = extract_features(backbone, imgs_d3)
        features_d3[wname] = feats_d3
        sep_d3, std_d3, _, _ = linear_separability(
            feats_d3, labels_d3, ctx['binary_class_ids'],
            desc='{} domain_{}'.format(wname, test_domain))

        print('  提取 {} 特征...'.format(ctx['train_domains_tag']))
        feats_train = extract_features(backbone, imgs_train)
        features_train[wname] = feats_train
        sep_train, std_train, _, _ = linear_separability(
            feats_train, labels_train, ctx['binary_class_ids'],
            desc='{} {}'.format(wname, ctx['train_domains_tag']))

        sep_scores[wname] = {
            'test_mean': sep_d3, 'test_std': std_d3,
            'train_mean': sep_train, 'train_std': std_train,
            'feature_dim': int(feats_d3.shape[1]),
        }

        del backbone
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    print('\n' + '=' * 50)
    print('t-SNE 可视化')
    print('=' * 50)
    if features_d3:
        plot_tsne(
            features_d3, labels_d3, ctx['binary_class_ids'], pos_name,
            title='Domain {} (测试域) | {}'.format(test_domain, ctx['binary_label']),
            save_path=ctx['output_dir'] / 'tsne_domain3.png',
        )
    if features_train:
        plot_tsne(
            features_train, labels_train, ctx['binary_class_ids'], pos_name,
            title='{} (训练域) | {}'.format(ctx['train_domains_tag'], ctx['binary_label']),
            save_path=ctx['output_dir'] / 'tsne_train_domains.png',
        )

    print('\n' + '=' * 50)
    print('汇总报告')
    print('=' * 50)
    train_col = ctx['train_domains_tag']
    print('{:<12} {:<22} {:<22}'.format('权重', 'test domain', train_col))
    print('-' * 56)
    for wname, s in sep_scores.items():
        print('{:<12} {:<22} {:<22}'.format(
            wname,
            '{:.3f}±{:.3f}'.format(s['test_mean'], s['test_std']),
            '{:.3f}±{:.3f}'.format(s['train_mean'], s['train_std']),
        ))

    p0_3_note = None
    if 'pretrained' in sep_scores and 'yolo26n' in sep_scores:
        diff_test = sep_scores['pretrained']['test_mean'] - sep_scores['yolo26n']['test_mean']
        diff_train = sep_scores['pretrained']['train_mean'] - sep_scores['yolo26n']['train_mean']
        print('\n  预训练 vs yolo26n 差值 (Phase0 特征层替代):')
        print('    test domain: {:+.3f}'.format(diff_test))
        print('    {}: {:+.3f}'.format(train_col, diff_train))
        p0_3_note = 'no_gain' if abs(diff_test) < 0.03 else 'has_gain'
        print('    → {}'.format(
            '预训练对 test domain 可分性无实质帮助 (<0.03)' if p0_3_note == 'no_gain'
            else '预训练对可分性有明显帮助'))

    main_key = 'pretrained' if 'pretrained' in sep_scores else (
        next(iter(sep_scores)) if sep_scores else None)
    if main_key:
        decision(
            sep_scores[main_key]['test_mean'],
            sep_scores[main_key]['train_mean'],
            train_col,
        )

    report = {
        'device': str(DEVICE),
        'track': ctx['track'],
        'data_dir': str(ctx['meta_root']),
        'weights': {k: str(v) for k, v in ctx['weights'].items()},
        'preprocess': 'Resize(256) + normalize; backbone encode interpolates to 640',
        'binary_task': ctx['binary_label'],
        'train_domains': train_domains,
        'test_domain': test_domain,
        'task_diversity': task_summary,
        'separability': sep_scores,
        'p0_3_feature_diff_note': p0_3_note,
    }
    report_path = ctx['output_dir'] / 'diagnose_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('\n报告保存: {}'.format(report_path))


if __name__ == '__main__':
    main()
