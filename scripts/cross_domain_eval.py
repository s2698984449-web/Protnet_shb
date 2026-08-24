"""Cross-Domain Few-Shot Evaluation: NEU model → GC10 test.

Loads a NEU-trained checkpoint and evaluates on GC10-DET without any retraining.
GC10 images are loaded directly from their original directory
(D:/元强化学习/代码/data/GC10-DET/cls_8class/<class>/<image>.jpg) — no preprocessing
or layout changes required.

Usage:
    cd D:/元强化学习/代码
    python scripts/cross_domain_eval.py \
        --checkpoint outputs/航天/checkpoints/neu_resnet18_protonet_frozen/best.pt \
        --data_dir_path D:/元强化学习/代码/data/GC10-DET/cls_8class \
        --ways 5 --shots 5 --eval_tasks 50
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.multimodal_model import AviationDefectModel
from common.protonet_loop import (
    build_prototypes,
    prototypical_logits,
)


# GC10-DET class folder mapping (index used by --class_subset)
GC10_CLASS_FOLDERS = {
    0: '1_chongkong',   # 冲孔 (punching)
    1: '2_hanfeng',     # 焊缝 (weld line)
    2: '3_yueyawan',    # 月牙弯 (crescent)
    3: '4_shuiban',     # 水斑 (water spot)
    4: '5_youban',      # 油斑 (oil spot)
    5: '6_siban',       # 丝斑 (silk spot)
    6: '7_yiwu',        # 异物 (inclusion)
    7: '10_yaozhed',    # 折皱 (crease/waist)
}


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')


def list_class_images(data_dir, class_name, limit=None):
    """Return up to `limit` image paths from a GC10 class folder, sorted deterministically."""
    class_dir = os.path.join(data_dir, class_name)
    files = sorted(glob(os.path.join(class_dir, '*.jpg')))
    files += sorted(glob(os.path.join(class_dir, '*.png')))
    files += sorted(glob(os.path.join(class_dir, '*.bmp')))
    if limit is not None:
        files = files[:limit]
    return files


def kolektor_index(data_dir):
    """Index Kolektor dataset by defect / ok based on label mask.

    Returns:
        defect_paths: list of image paths where label.bmp has any pixel > 0
        ok_paths:     list of image paths where label.bmp is empty
    """
    defect_paths = []
    ok_paths = []
    kos_folders = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('kos')])
    for k in kos_folders:
        pdir = os.path.join(data_dir, k)
        files = sorted(os.listdir(pdir))
        # pair up PartX.jpg with PartX_label.bmp
        labels = {f.replace('_label.bmp', ''): f for f in files if f.endswith('_label.bmp')}
        for prefix, label_file in labels.items():
            img_file = prefix + '.jpg'
            if not os.path.isfile(os.path.join(pdir, img_file)):
                continue
            label_path = os.path.join(pdir, label_file)
            arr = np.asarray(Image.open(label_path))
            if arr.size == 0:
                continue
            img_full = os.path.join(pdir, img_file)
            if arr.sum() > 0:
                defect_paths.append(img_full)
            else:
                ok_paths.append(img_full)
    return defect_paths, ok_paths


def load_and_preprocess(image_path, transform):
    img = Image.open(image_path).convert('RGB')
    return transform(img)


def sample_episode(class_image_pool, ways, shots, query_per_class, rng):
    """Sample one episode: support + query.

    Returns:
        support_x: (ways*shots, 3, H, W)
        support_y: (ways*shots,)
        query_x:   (ways*query_per_class, 3, H, W)
        query_y:   (ways*query_per_class,)
    """
    available_classes = [c for c, paths in class_image_pool.items() if len(paths) >= shots + query_per_class]
    if len(available_classes) < ways:
        raise RuntimeError(
            f'Not enough classes with >= {shots + query_per_class} images. '
            f'Need {ways}, got {len(available_classes)} ({available_classes}).'
        )

    chosen = rng.choice(available_classes, size=ways, replace=False)
    chosen_sorted = sorted(int(c) for c in chosen)

    support_paths, query_paths = [], []
    support_labels, query_labels = [], []
    for new_label, cls in enumerate(chosen_sorted):
        paths = class_image_pool[int(cls)]
        idx = rng.permutation(len(paths))[: shots + query_per_class]
        support_paths.extend([paths[i] for i in idx[:shots]])
        query_paths.extend([paths[i] for i in idx[shots:]])
        support_labels.extend([new_label] * shots)
        query_labels.extend([new_label] * query_per_class)

    return support_paths, support_labels, query_paths, query_labels, chosen_sorted


def build_transform(imgsz):
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def encode_rgb_batch(model, rgb_batch, device):
    """Encode RGB batch through the model's RGB encoder (matches training inference path)."""
    rgb_tensor = rgb_batch.to(device)
    with torch.no_grad():
        # AviationDefectModel.embed is the RGB-only path used during training/inference
        emb = model.embed(rgb_tensor, None, None)
    return emb


def evaluate_cross_domain(args, model, class_image_pool, transform, device):
    """Run N episodes and return per-episode accuracy list."""
    rng = np.random.RandomState(args.seed)
    accs = []
    chosen_class_history = []

    print(f'Running {args.eval_tasks} episodes: {args.ways}-way {args.shots}-shot '
          f'(query/class={args.query_per_class})')

    for ep in range(args.eval_tasks):
        support_paths, support_y, query_paths, query_y, chosen = sample_episode(
            class_image_pool,
            ways=args.ways,
            shots=args.shots,
            query_per_class=args.query_per_class,
            rng=rng,
        )
        chosen_class_history.append(chosen)

        # Load + preprocess (lazy: on-the-fly, no caching needed for small batches)
        sup_rgb = torch.stack([load_and_preprocess(p, transform) for p in support_paths])
        qry_rgb = torch.stack([load_and_preprocess(p, transform) for p in query_paths])

        # Encode through model (RGB-only embedding path)
        sup_emb = encode_rgb_batch(model, sup_rgb, device)
        qry_emb = encode_rgb_batch(model, qry_rgb, device)

        # Compute logits via prototypical_logits (it builds prototypes internally)
        qry_y = torch.tensor(query_y, dtype=torch.long, device=device)
        logits = prototypical_logits(
            qry_emb, sup_emb, torch.tensor(support_y, dtype=torch.long, device=device),
            args.ways,
            distance=args.protonet_distance,
            prototype_type=args.protonet_prototype_type,
        )
        pred = logits.argmax(dim=1)
        acc = (pred == qry_y).float().mean().item()
        accs.append(acc)

        if (ep + 1) % max(1, args.eval_tasks // 10) == 0:
            arr = np.array(accs)
            print(f'  ep {ep + 1}/{args.eval_tasks}: mean={arr.mean():.4f} '
                  f'std={arr.std():.4f}')

    return accs, chosen_class_history


def load_model(checkpoint_path, args, device):
    """Load AviationDefectModel from a training checkpoint (handles run_config.json).

    Only loads RGB-encoder weights (rgb_encoder, adaptive_proto) plus thermal/us encoder
    if compatible. Ways-dependent layers (fusion, cls_rgb, stop_net, mahal_head, task_encoder)
    are skipped — only the rgb_encoder matters for ProtoNet inference.
    """
    ckpt_dir = os.path.dirname(checkpoint_path)
    cfg_path = os.path.join(ckpt_dir, 'run_config.json')
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)

    backbone_type = cfg.get('backbone_type', args.backbone_type)
    rgb_cbam = bool(cfg.get('rgb_cbam', False))
    embed_dim = int(cfg.get('embed_dim', args.embed_dim))
    yolo_imgsz = int(cfg.get('yolo_imgsz', args.yolo_imgsz))
    yolo_weights = cfg.get('yolo_weights', args.yolo_weights)
    # Number of classes the model was trained with (for fusion head size)
    train_ways = int(cfg.get('ways', args.ways))

    print(f'Loading checkpoint: {checkpoint_path}')
    print(f'  backbone_type={backbone_type}, embed_dim={embed_dim}, '
          f'rgb_cbam={rgb_cbam}, imgsz={yolo_imgsz}, train_ways={train_ways}')

    # Always build model with train_ways to match checkpoint size
    model = AviationDefectModel(
        num_classes=train_ways,
        embed_dim=embed_dim,
        yolo_weights=yolo_weights,
        yolo_imgsz=yolo_imgsz,
        rgb_cbam=rgb_cbam,
        use_multiscale_cff=False,
        backbone_type=backbone_type,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)

    # Keep ONLY way-shape-independent weights (rgb_encoder, adaptive_proto, thermal_encoder,
    # us_encoder). Drop ways-dependent layers to avoid size mismatch.
    keep_keys = []
    for k in state.keys():
        if k.startswith('rgb_encoder.'):
            keep_keys.append(k)
        elif k.startswith('adaptive_proto.'):
            keep_keys.append(k)
        elif k.startswith('thermal_encoder.') or k.startswith('us_encoder.'):
            # encoders are ways-independent but kept for completeness
            keep_keys.append(k)
    filtered_state = {k: state[k] for k in keep_keys}

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing or unexpected:
        print(f'  load_state_dict (rgb-only): missing={len(missing)}, '
              f'unexpected={len(unexpected)} keys')
        if len(missing) > 0:
            # Print only rgb_encoder-related missing
            rgb_missing = [k for k in missing if 'rgb_encoder' in k]
            print(f'    rgb_encoder missing: {len(rgb_missing)} (first 3: {rgb_missing[:3]})')
        if len(unexpected) > 0:
            print(f'    first unexpected: {unexpected[:3]}')

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, embed_dim


def main():
    p = argparse.ArgumentParser(
        description='Cross-domain few-shot evaluation (NEU → GC10-DET or NEU → Kolektor)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to .pt checkpoint (e.g., outputs/.../best.pt)')
    p.add_argument('--dataset', type=str, default='gc10',
                   choices=('gc10', 'kolektor'),
                   help='Target test dataset. gc10=cls_8class, kolektor=2-way defect/ok')
    p.add_argument('--data_dir_path', type=str,
                   default=r'D:\元强化学习\代码\data\GC10-DET\cls_8class',
                   help='For gc10: <class_folder>/*.jpg; for kolektor: root containing kos*/ folders')
    p.add_argument('--ways', type=int, default=5,
                   help='Number of target classes to sample per episode')
    p.add_argument('--shots', type=int, default=5,
                   help='Support examples per class')
    p.add_argument('--query_per_class', type=int, default=15,
                   help='Query examples per class (NEU eval used 15)')
    p.add_argument('--eval_tasks', type=int, default=50,
                   help='Number of episodes to evaluate (NEU eval used 50)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--backbone_type', type=str, default='resnet18',
                   choices=('resnet18', 'yolo26'))
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--yolo_weights', type=str,
                   default='./weights/aviation_yolo26_pretrained.pt')
    p.add_argument('--yolo_imgsz', type=int, default=640)
    p.add_argument('--protonet_distance', type=str, default='euclidean',
                   choices=('euclidean', 'cosine', 'mahalanobis'))
    p.add_argument('--protonet_prototype_type', type=str, default='mean',
                   choices=('mean', 'trim_mean', 'mean_shift'))
    p.add_argument('--class_subset', type=str, default='1,2,3,4,5',
                   help='(gc10 only) Comma-separated GC10 class indices (1..7,10). '
                        'Default: first 5 classes.')
    p.add_argument('--cuda', type=_str2bool, default=True)
    p.add_argument('--out_json', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Target dataset: {args.dataset}')

    # Build image pool
    if args.dataset == 'gc10':
        selected = [int(x.strip()) for x in args.class_subset.split(',') if x.strip()]
        if args.ways > len(selected):
            raise ValueError(f'--ways={args.ways} but only {len(selected)} GC10 classes selected.')

        class_image_pool = {}
        print(f'\nIndexing GC10 from {args.data_dir_path}:')
        for cls in selected:
            folder = GC10_CLASS_FOLDERS.get(cls, str(cls))
            paths = list_class_images(args.data_dir_path, folder)
            if not paths:
                print(f'  WARNING: class {cls} ({folder}): no images found')
                continue
            class_image_pool[cls] = paths
            print(f'  class {cls} ({folder}): {len(paths)} images')

        if len(class_image_pool) < args.ways:
            raise RuntimeError(
                f'Only {len(class_image_pool)} classes have images; '
                f'need at least {args.ways}. Adjust --class_subset.'
            )
        class_image_pool = {i: class_image_pool[selected[i]] for i in range(args.ways)}
        target_name = 'GC10-DET (cls_8class)'
        target_classes_desc = [GC10_CLASS_FOLDERS.get(c, str(c)) for c in selected]
    elif args.dataset == 'kolektor':
        print(f'\nIndexing Kolektor from {args.data_dir_path}:')
        defect_paths, ok_paths = kolektor_index(args.data_dir_path)
        print(f'  defect: {len(defect_paths)} images')
        print(f'  ok:     {len(ok_paths)} images')

        if args.ways != 2:
            raise ValueError(
                f'Kolektor only supports 2-way (defect/ok). Got --ways={args.ways}.'
            )
        if len(defect_paths) < args.shots + args.query_per_class:
            raise ValueError(
                f'defect class only has {len(defect_paths)} images; '
                f'need at least {args.shots + args.query_per_class}. '
                f'Try lower --shots or --query_per_class.'
            )
        # class 0 = defect, class 1 = ok
        class_image_pool = {0: defect_paths, 1: ok_paths}
        target_name = 'Kolektor (binary defect/ok)'
        target_classes_desc = ['defect', 'ok']

    # Load model
    model, embed_dim = load_model(args.checkpoint, args, device)

    # Build transform matching train-time preprocessing
    imgsz = 224 if args.backbone_type == 'resnet18' else args.yolo_imgsz
    transform = build_transform(imgsz)

    # Run evaluation
    print('\n' + '=' * 60)
    accs, chosen_history = evaluate_cross_domain(args, model, class_image_pool, transform, device)

    # Aggregate
    arr = np.array(accs, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci_margin = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    ci_low = mean - ci_margin
    ci_high = mean + ci_margin

    print('\n' + '=' * 60)
    print(f'CROSS-DOMAIN RESULT: NEU → {target_name}')
    print('=' * 60)
    print(f'  Checkpoint:    {os.path.abspath(args.checkpoint)}')
    print(f'  Test dataset:  {target_name}, {args.ways}-way {args.shots}-shot')
    print(f'  Target classes:{target_classes_desc}')
    print(f'  Episodes:      {len(accs)}')
    print(f'  Mean acc:      {mean:.4f}')
    print(f'  Std:           {std:.4f}')
    print(f'  95% CI:        [{ci_low:.4f}, {ci_high:.4f}]')
    print(f'  Min / Max:     {arr.min():.4f} / {arr.max():.4f}')
    print(f'  Random baseline (1/{args.ways}-way): {1.0/args.ways:.4f}')
    print('=' * 60)

    # Save report
    if args.out_json:
        report = {
            'checkpoint': os.path.abspath(args.checkpoint),
            'test_dataset': args.dataset,
            'data_root': args.data_dir_path,
            'target_classes': target_classes_desc,
            'protocol': {
                'ways': args.ways,
                'shots': args.shots,
                'query_per_class': args.query_per_class,
                'n_episodes': len(accs),
                'protonet_distance': args.protonet_distance,
                'protonet_prototype_type': args.protonet_prototype_type,
                'seed': args.seed,
                'random_baseline': 1.0 / args.ways,
            },
            'result': {
                'mean': mean,
                'std': std,
                'ci95_low': ci_low,
                'ci95_high': ci_high,
                'min': float(arr.min()),
                'max': float(arr.max()),
            },
            'per_episode_acc': [float(a) for a in accs],
            'per_episode_classes': [
                [target_classes_desc[c] if 0 <= c < len(target_classes_desc) else str(c)
                 for c in ep] for ep in chosen_history
            ],
        }
        out_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f'\nReport saved: {out_path}')


if __name__ == '__main__':
    main()
