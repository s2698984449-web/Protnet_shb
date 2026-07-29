"""综合诊断: 跨域光照 / 材质 / 多尺度分布分析
   输入: data/aviation/{domain_0..3}/*/*/rgb.jpg
   输出: reports/diagnose/cross_domain_distill.json
   统计维度:
     1. 光照: 平均亮度 (HSV-V) 分布 + 亮度跨度 + 亮度方差 per domain/per class
     2. 材质: HSV 颜色直方图分布 + 灰度均值 + 饱和度均值 (作为材质代理)
     3. 多尺度: 缺陷区域尺寸分布 (目标mask 像素数, 需要 bbox)
     4. 类别内方差: 同类样本的 embedding 类内方差 (受限于无标签, 用颜色直方图做代理)
"""
import os
import json
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

DATA_ROOT = r"D:\元强化学习\代码\data\航天\processed\aviation"
OUT_DIR = r"D:\元强化学习\代码\航空缺陷检测\reports\diagnose"
os.makedirs(OUT_DIR, exist_ok=True)


def list_samples(domain_dir):
    """返回 [(class_name, rgb_path)]"""
    samples = []
    for cls in sorted(os.listdir(domain_dir)):
        cls_dir = os.path.join(domain_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for sample_dir in sorted(os.listdir(cls_dir)):
            rgb = os.path.join(cls_dir, sample_dir, 'rgb.jpg')
            if os.path.isfile(rgb):
                samples.append((cls, rgb))
    return samples


def compute_color_stats(img_np):
    """img_np: HxWx3 uint8, 返回 dict"""
    hsv = np.array(Image.fromarray(img_np).convert('HSV'))
    H, S, V = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    # 灰度 (材质亮度代理)
    gray = (0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]).astype(np.float32)
    return {
        'V_mean': float(V.mean()),
        'V_std': float(V.std()),
        'V_p10': float(np.percentile(V, 10)),
        'V_p50': float(np.percentile(V, 50)),
        'V_p90': float(np.percentile(V, 90)),
        'V_range': float(np.percentile(V, 95) - np.percentile(V, 5)),  # 5-95 跨度
        'S_mean': float(S.mean()),  # 饱和度
        'gray_mean': float(gray.mean()),  # 灰度
        'H_hist': np.histogram(H, bins=16, range=(0, 256))[0].tolist(),  # 色相分布
    }


def compute_target_size(img_np):
    """用 Sobel 边缘检测 + Otsu 阈值估计 ROI 区域占比"""
    try:
        from scipy import ndimage
        gray = (0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]).astype(np.float32)
        gx = ndimage.sobel(gray, axis=0)
        gy = ndimage.sobel(gray, axis=1)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        # 阈值: 用 70% 分位做边缘 mask
        thresh = np.percentile(mag, 70)
        edges = (mag > thresh).astype(np.float32)
        # 连通区域估计 ROI 占比
        labeled, n_components = ndimage.label(edges)
        if n_components == 0:
            return {'edge_density': float(edges.mean()), 'n_components': 0}
        # 取最大 5 个连通区域估计目标尺度
        comp_sizes = ndimage.sum(edges, labeled, range(1, n_components + 1))
        comp_sizes = np.sort(comp_sizes)[::-1][:5]
        # 归一到图像面积
        comp_sizes = comp_sizes / float(gray.size)
        return {
            'edge_density': float(edges.mean()),
            'largest_comp_ratio': float(comp_sizes[0]) if len(comp_sizes) > 0 else 0.0,
            'top5_comp_ratio': float(comp_sizes.sum()) if len(comp_sizes) > 0 else 0.0,
        }
    except Exception as e:
        return {'edge_density': -1, 'error': str(e)}


def main():
    report = {
        'domains': {},
        'meta': {
            'note': 'domain_3 是试件 20-26 holdout, 训练只用 domain_0/1/2',
            'data_root': DATA_ROOT,
        }
    }

    for d in range(4):
        d_dir = os.path.join(DATA_ROOT, f'domain_{d}')
        if not os.path.isdir(d_dir):
            continue
        samples = list_samples(d_dir)
        print(f"\n[domain_{d}] {len(samples)} samples")

        domain_stats = {'total_samples': len(samples), 'per_class': {}, 'all_V': [], 'all_gray': [], 'all_S': []}

        for cls, rgb_path in samples:
            try:
                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img.resize((256, 256)))  # 统一到 256x256
                cs = compute_color_stats(img_np)
                ts = compute_target_size(img_np)

                if cls not in domain_stats['per_class']:
                    domain_stats['per_class'][cls] = []
                domain_stats['per_class'][cls].append({
                    'sample': os.path.basename(os.path.dirname(rgb_path)),
                    'color': cs,
                    'target_size': ts,
                })
                domain_stats['all_V'].append(cs['V_mean'])
                domain_stats['all_gray'].append(cs['gray_mean'])
                domain_stats['all_S'].append(cs['S_mean'])
            except Exception as e:
                print(f"  ! ERR {rgb_path}: {e}")

        # 域级聚合
        V = np.array(domain_stats.pop('all_V'))
        G = np.array(domain_stats.pop('all_gray'))
        S = np.array(domain_stats.pop('all_S'))

        domain_stats['aggregate'] = {
            'V_mean': float(V.mean()),
            'V_std': float(V.std()),
            'V_p10': float(np.percentile(V, 10)),
            'V_p90': float(np.percentile(V, 90)),
            'V_range_pct': float((np.percentile(V, 95) - np.percentile(V, 5)) / 255.0 * 100),
            'gray_mean': float(G.mean()),
            'gray_std': float(G.std()),
            'S_mean': float(S.mean()),
        }

        report['domains'][f'domain_{d}'] = domain_stats
        print(f"  → V_mean={domain_stats['aggregate']['V_mean']:.1f}, "
              f"gray={domain_stats['aggregate']['gray_mean']:.1f}, "
              f"classes={list(domain_stats['per_class'].keys())}")

    # 关键跨域差异计算
    print('\n=== 跨域汇总 ===')
    if 'domain_0' in report['domains'] and 'domain_3' in report['domains']:
        d0 = report['domains']['domain_0']['aggregate']
        d3 = report['domains']['domain_3']['aggregate']
        report['cross_domain_diff'] = {
            'V_mean_diff': d3['V_mean'] - d0['V_mean'],
            'V_std_diff': d3['V_std'] - d0['V_std'],
            'gray_diff': d3['gray_mean'] - d0['gray_mean'],
            'S_diff': d3['S_mean'] - d0['S_mean'],
        }
        print(f"  domain_3 vs domain_0:")
        for k, v in report['cross_domain_diff'].items():
            print(f"    {k}: {v:+.2f}")

    # 按类别统计
    print('\n=== 按类别统计 (domain_3) ===')
    d3 = report['domains'].get('domain_3', {})
    for cls, items in d3.get('per_class', {}).items():
        V = np.array([x['color']['V_mean'] for x in items])
        G = np.array([x['color']['gray_mean'] for x in items])
        S = np.array([x['color']['S_mean'] for x in items])
        E = np.array([x['target_size'].get('edge_density', 0) for x in items])
        print(f"  {cls}: n={len(items)}, V={V.mean():.1f}±{V.std():.1f}, "
              f"gray={G.mean():.1f}±{G.std():.1f}, edge_density={E.mean():.3f}")

    out_path = os.path.join(OUT_DIR, 'cross_domain_distill.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n=== Saved: {out_path} ===')


if __name__ == '__main__':
    main()
