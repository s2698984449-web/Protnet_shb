"""Multi-source domain pretrain smoke test (ResNet18 + FiLM).

Usage:
    python scripts/pretrain_multi_domain.py --epochs 2 --max_steps_per_epoch 5

验证链路通:
  1) 加载 MVTEC / GC10 / NEU 三个数据集
  2) 每个 dataset → (image, cls_label, cond_vec, product_id)
  3) forward: ResNet18 backbone + FiLM 调制 → 128-d feat
  4) per-task CE head 算 logits → CE loss
  5) combined SupCon 跨源域聚类
  6) 反向 + 优化器 step
"""

import argparse
import logging
import os
import sys
import time

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.multi_domain_datasets import build_multi_domain_loaders
from common.multi_domain_model import MultiDomainPretrainModel
from common.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-domain ResNet18 pretrain smoke')
    parser.add_argument('--mvtec_root', type=str,
                        default=os.path.join(_ROOT, 'data', 'MVTEC'))
    parser.add_argument('--gc10_root', type=str,
                        default=os.path.join(_ROOT, 'data', 'GC10-DET'))
    parser.add_argument('--neu_root', type=str,
                        default=os.path.join(_ROOT, 'data', 'NEU-CLS', 'NEU-CLS'))
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max_steps_per_epoch', type=int, default=5)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_backbone', type=float, default=1e-4)
    parser.add_argument('--supcon_weight', type=float, default=0.5)
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--unfreeze_backbone', action='store_true',
                        help='Train backbone too (default: frozen)')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--out_weights', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = os.path.join(_ROOT, 'outputs', 'logs', 'multi_domain')
    os.makedirs(log_path, exist_ok=True)
    setup_logger(log_path, 'Multi-domain ResNet18 pretrain smoke')
    logging.info('Args: %s', args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Device: %s', device)

    # ── Load data ──────────────────────────────────────────
    logging.info('Loading multi-domain datasets...')
    t0 = time.time()
    loaders, task_specs, cls_counts, product_ids = build_multi_domain_loaders(
        mvtec_root=args.mvtec_root,
        gc10_root=args.gc10_root,
        neu_root=args.neu_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=True,
        num_workers=args.num_workers,
    )
    logging.info('Datasets ready in %.1fs. task_specs=%s', time.time() - t0, task_specs)
    for name in ('mvtec', 'gc10', 'neu'):
        ds = loaders[name].dataset
        logging.info('  %s: %d samples, %d classes, cls_counts=%s',
                     name, len(ds), len(ds.class_names), cls_counts[name])

    # ── Build model ────────────────────────────────────────
    model = MultiDomainPretrainModel(
        task_specs=task_specs,
        n_products=17,  # 15 MVTEC + 1 GC10 + 1 NEU
        n_conditions=6,
        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        film_layer='layer4',
        film_channels=512,  # ResNet18 layer4 output channels
        freeze_backbone=not args.unfreeze_backbone,
        pretrained_backbone=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Model: %.2f M total, %.2f M trainable.',
                 n_params / 1e6, n_trainable / 1e6)

    # ── Optimizer: separate LR for head / backbone ─────────
    head_params = model.head_params()
    backbone_params = model.backbone_params() if not args.unfreeze_backbone else []
    param_groups = [{'params': head_params, 'lr': args.lr_head}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # ── Train loop ─────────────────────────────────────────
    iterators = {n: iter(loaders[n]) for n in ('mvtec', 'gc10', 'neu')}
    for epoch in range(args.epochs):
        model.train()
        for step in range(args.max_steps_per_epoch):
            try:
                batches, cond_batches, pid_batches = {}, {}, {}
                for name in ('mvtec', 'gc10', 'neu'):
                    img, lbl, cond, pid = next(iterators[name])
                    batches[name] = (img.to(device), lbl.to(device))
                    cond_batches[name] = cond.to(device)
                    pid_batches[name] = pid.to(device)

                total_loss, total_ce, feats = model.compute_combined_loss(
                    batches, cond_batches,
                    supcon_weight=args.supcon_weight,
                    ce_weight=args.ce_weight,
                    global_supcon=True,
                )

                # Log per-task acc
                with torch.no_grad():
                    accs = {}
                    for name, (img, lbl) in batches.items():
                        logits, _ = model.forward_task(name, img, cond_batches[name])
                        pred = logits.argmax(dim=1)
                        accs[name] = (pred == lbl).float().mean().item()

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                logging.info(
                    'ep %d step %d: loss=%.4f (ce=%.4f)  acc_mvtec=%.3f acc_gc10=%.3f acc_neu=%.3f',
                    epoch, step, total_loss.item(), total_ce.item(),
                    accs['mvtec'], accs['gc10'], accs['neu'],
                )
            except StopIteration:
                logging.warning('Data exhausted at ep %d step %d', epoch, step)
                break

    # ── Save ───────────────────────────────────────────────
    if args.out_weights:
        out = os.path.abspath(args.out_weights)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'task_specs': task_specs,
            'n_products': 17,
            'n_conditions': 6,
            'feat_dim': args.feat_dim,
            'proj_dim': args.proj_dim,
            'film_layer': 'layer3',
        }, out)
        logging.info('Saved weights to %s', out)


if __name__ == '__main__':
    main()