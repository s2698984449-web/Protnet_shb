"""Entry point: aviation multimodal defect detection with MAML / ProtoNet."""

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import common  # noqa: F401
from common import maml_origin
from common.maml_loop import (
    adapt_supervised,
    curriculum_adapt_steps,
    curriculum_w_rl,
    evaluate_query,
    meta_task_loss,
    split_support_query,
)
from common.episode_aug import augment_support_query
from common.protonet_loop import evaluate_protonet_query, protonet_episode_loss
from common.eval_loop import evaluate_tasks
from common.multimodal_model import AviationDefectModel, ValueBaseline
from common.task_builder import create_tasksets
from common.utils import setup_logger, print_logs
from common.project_paths import apply_meta_train_defaults


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('1', 'true', 'yes', 'y', 't'):
        return True
    if value in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value, got {!r}.'.format(value))


def parse_args():
    parser = argparse.ArgumentParser(
        description='MAML/ANIL meta-learning (aerospace multimodal or automotive RGB CMPL)')

    parser.add_argument('--track', type=str, default='aerospace',
                        help='aerospace|automotive (or 航天|车). Sets data + artifact dirs.')
    parser.add_argument('--ways', type=int, default=2,
                        help='N-way; current Aero-engine domains mostly have 2 effective classes')
    parser.add_argument('--shots', type=int, default=3)
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Outer lr when freeze_backbone=True (proj+fusion)')
    parser.add_argument('--backbone_meta_lr', type=float, default=1e-5,
                        help='Outer lr for YOLO26 det_model when freeze_backbone=False')
    parser.add_argument('--head_meta_lr', type=float, default=1e-3,
                        help='Outer lr for proj/fusion/encoders when freeze_backbone=False')
    parser.add_argument('--fast_lr', type=float, default=0.01)
    parser.add_argument('--adapt_steps', type=int, default=5)
    parser.add_argument('--meta_batch_size', type=int, default=2)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--first_order', type=_str2bool, default=True,
                        help='True=FOMAML（4GB 默认）；False=二阶 MAML（需大显存）')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--yolo_weights', type=str, default='./weights/yolo26n.pt',
                        help='Local YOLO26 weights (avoid auto-download if SSL fails). '
                             'Ignored when --backbone_type=resnet18.')
    parser.add_argument('--resnet_weights', type=str, default='',
                        help='Optional local ResNet18 ImageNet weights (.pth). '
                             'Empty=auto-download via torchvision to ~/.cache/torch/hub/checkpoints/. '
                             'Only used when --backbone_type=resnet18.')
    parser.add_argument('--yolo_imgsz', type=int, default=640,
                        help='YOLO26 input size; RGB is resized in encoder')
    parser.add_argument('--rgb_only', type=_str2bool, default=False,
                        help='Phase-A ablation: RGB encoder only')
    parser.add_argument('--freeze_backbone', type=_str2bool, default=True,
                        help='Phase-1: freeze YOLO26, inner loop adapts fusion only')
    parser.add_argument('--eval_tasks', type=int, default=50,
                        help='Multi-task eval on test domain (0=disable)')
    parser.add_argument('--eval_every', type=int, default=50,
                        help='Run multi-task eval every N iters')

    parser.add_argument('--stage1_end', type=int, default=100)
    parser.add_argument('--stage2_end', type=int, default=200)
    parser.add_argument('--w_rl_max', type=float, default=0.5)

    parser.add_argument('--cuda', type=_str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_dir_path', type=str, default=None,
                        help='Processed data root (default: data/{track}/processed)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        choices=('aviation', 'cmpl', 'neu', 'carparts_cls'),
                        help='Meta subfolder (default from --track)')
    parser.add_argument('--anil', type=_str2bool, default=False,
                        help='ANIL mode: alias for freeze_backbone=True (inner loop adapts fusion only)')
    parser.add_argument('--train_domains', type=str, default='0,1,2')
    parser.add_argument('--test_domain', type=int, default=3)
    parser.add_argument('--train_task_num', type=int, default=40)
    parser.add_argument('--test_task_num', type=int, default=50)

    parser.add_argument('--plot', type=_str2bool, default=True)
    parser.add_argument('--plot_path', type=str, default='./images')
    parser.add_argument('--plot_step', type=int, default=50)
    parser.add_argument('--log', type=_str2bool, default=True)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--checkpoint', type=_str2bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_step', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='',
                        help='Experiment subfolder under checkpoints/logs/images')
    parser.add_argument('--episode_aug', type=_str2bool, default=False,
                        help='Episode-level RGB aug on meta-train support/query only')
    parser.add_argument('--aug_illumination', type=_str2bool, default=False,
                        help='Enable illumination augmentation (gamma+brightness+specular) in episode_aug. '
                             'supersedes --episode_aug when set.')
    parser.add_argument('--input_scales', type=str, default='',
                        help='Comma-separated train input scales for multi-scale training, '
                             'e.g. "320,480,640". Empty=fixed 640. Applied in meta-train only.')
    parser.add_argument('--cross_domain_episodes', type=_str2bool, default=False,
                        help='Pool scratch/damage across train domains for meta-train episodes')
    parser.add_argument('--eval_patience', type=int, default=0,
                        help='Stop after N evals without improvement (0=disabled)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='Minimum eval gain to reset early-stop patience')
    parser.add_argument('--meta_algorithm', type=str, default='maml',
                        choices=('maml', 'protonet'),
                        help='Meta-learning algorithm: maml (inner-loop adapt) or protonet')
    parser.add_argument('--protonet_distance', type=str, default='euclidean',
                        choices=('euclidean', 'cosine', 'mahalanobis'),
                        help='Prototype distance for ProtoNet (ignored for MAML)')
    parser.add_argument('--protonet_prototype_type', type=str, default='mean',
                        choices=('mean', 'trim_mean', 'mean_shift'),
                        help='Prototype aggregation for ProtoNet: mean / trim_mean / mean_shift')
    parser.add_argument('--rgb_cbam', type=_str2bool, default=False,
                        help='CBAM attention on YOLO neck features before global pool (exp10)')
    parser.add_argument('--use_multiscale_cff', type=_str2bool, default=False,
                        help='Multi-scale CFF: P2/P3/P4 fusion + CBAM on RGB branch')
    parser.add_argument('--use_adaptive_proto', type=_str2bool, default=True,
                        help='Adaptive Prototype: cross-attention between query and prototypes (default ON)')
    parser.add_argument('--backbone_type', type=str, default='yolo26',
                        choices=['yolo26', 'resnet18'],
                        help='Backbone type: yolo26 (default) or resnet18')

    args = parser.parse_args()
    args._rgb_only_explicit = '--rgb_only' in sys.argv
    args._anil_explicit = '--anil' in sys.argv
    args._freeze_backbone_explicit = '--freeze_backbone' in sys.argv
    args._meta_algorithm_explicit = '--meta_algorithm' in sys.argv
    args._w_rl_max_explicit = '--w_rl_max' in sys.argv
    args._stage_end_explicit = '--stage1_end' in sys.argv or '--stage2_end' in sys.argv
    args._rgb_cbam_explicit = '--rgb_cbam' in sys.argv
    args._use_multiscale_cff_explicit = '--use_multiscale_cff' in sys.argv
    args._use_adaptive_proto_explicit = '--use_adaptive_proto' in sys.argv
    args._backbone_type_explicit = '--backbone_type' in sys.argv
    apply_meta_train_defaults(args)
    return args


def _apply_exp_dirs(args):
    """Isolate each run under exp_name subfolders."""
    if not args.exp_name:
        return
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.exp_name)
    args.log_path = os.path.join(args.log_path, args.exp_name)
    args.plot_path = os.path.join(args.plot_path, args.exp_name)


def _save_run_config(args, experiment_title):
    os.makedirs(args.checkpoint_path, exist_ok=True)
    config_path = os.path.join(args.checkpoint_path, 'run_config.json')
    payload = {
        'experiment_title': experiment_title,
        'exp_name': args.exp_name,
        'ways': args.ways,
        'shots': args.shots,
        'iters': args.iters,
        'meta_lr': args.meta_lr,
        'backbone_meta_lr': args.backbone_meta_lr,
        'head_meta_lr': args.head_meta_lr,
        'fast_lr': args.fast_lr,
        'meta_batch_size': args.meta_batch_size,
        'train_domains': args.train_domains,
        'test_domain': args.test_domain,
        'yolo_weights': (os.path.abspath(args.yolo_weights) if getattr(args, 'backbone_type', 'yolo26') == 'yolo26' else '(unused; backbone_type=resnet18)'),
        'freeze_backbone': args.freeze_backbone,
        'rgb_only': args.rgb_only,
        'w_rl_max': args.w_rl_max,
        'stage1_end': args.stage1_end,
        'stage2_end': args.stage2_end,
        'seed': args.seed,
        'eval_every': args.eval_every,
        'eval_tasks': args.eval_tasks,
        'episode_aug': args.episode_aug,
        'cross_domain_episodes': args.cross_domain_episodes,
        'eval_patience': args.eval_patience,
        'early_stop_min_delta': args.early_stop_min_delta,
        'adapt_steps': args.adapt_steps,
        'meta_algorithm': args.meta_algorithm,
        'protonet_distance': args.protonet_distance,
        'protonet_prototype_type': args.protonet_prototype_type,
        'rgb_cbam': args.rgb_cbam,
        'use_multiscale_cff': getattr(args, 'use_multiscale_cff', False),
        'backbone_type': getattr(args, 'backbone_type', 'yolo26'),
        'resnet_weights': getattr(args, 'resnet_weights', '') or '(auto-download)',
        'dataset_name': args.dataset_name,
        'anil': getattr(args, 'anil', False),
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def create_model(args, device):
    meta_algorithm = getattr(args, 'meta_algorithm', 'maml')
    model = AviationDefectModel(
        num_classes=args.ways,
        embed_dim=args.embed_dim,
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
        rgb_cbam=getattr(args, 'rgb_cbam', False),
        use_multiscale_cff=getattr(args, 'use_multiscale_cff', False),
        backbone_type=getattr(args, 'backbone_type', 'yolo26'),
        resnet_weights=getattr(args, 'resnet_weights', '') or None,
    ).to(device)
    fusion_only = args.freeze_backbone
    if fusion_only:
        model.set_backbone_frozen(True)
        logging.info('Backbone frozen (ANIL / encoder-only ProtoNet).')
    else:
        model.set_backbone_frozen(False)
        logging.info(
            'Backbone unfrozen; outer loop backbone_lr={:.1e}, head_lr={:.1e}.'.format(
                args.backbone_meta_lr, args.head_meta_lr))

    param_groups = model.meta_optimizer_param_groups(
        fusion_only=fusion_only,
        backbone_meta_lr=args.backbone_meta_lr,
        head_meta_lr=args.head_meta_lr,
        meta_lr=args.meta_lr,
    )
    opt = torch.optim.Adam(param_groups)
    for i, group in enumerate(param_groups):
        logging.info(
            'Meta optimizer group {}: {} tensors, lr={:.1e}.'.format(
                i, len(group['params']), group['lr']))

    if meta_algorithm == 'protonet':
        distance = getattr(args, 'protonet_distance', 'euclidean')
        prototype_type = getattr(args, 'protonet_prototype_type', 'mean')
        logging.info(
            'Meta algorithm: ProtoNet (distance={}, prototype={}, adaptive_proto={}).'.format(
                distance, prototype_type, getattr(args, 'use_adaptive_proto', False)))
        return model, None, opt, model.mahal_head

    logging.info(
        'Meta algorithm: MAML (inner loop adapts classifier; '
        'fast_lr={:.4g}).'.format(args.fast_lr))
    maml = maml_origin.MAML(
        model, lr=args.fast_lr, first_order=args.first_order, allow_nograd=True)
    value_net = ValueBaseline(state_dim=model.state_dim()).to(device)
    return model, maml, opt, value_net


def _train_loop(args, experiment_title):
    """Meta-learning training loop (merged from maml_train.py)."""
    logging.info('Experiment: {}'.format(experiment_title))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda')
        logging.info('Training with GPU.')
    else:
        device = torch.device('cpu')
        logging.info('Training with CPU.')

    train_tasks, test_tasks, domain_meta_map = create_tasksets(args)
    model, maml, opt, value_net = create_model(args, device)
    meta_algorithm = getattr(args, 'meta_algorithm', 'maml')
    protonet_distance = getattr(args, 'protonet_distance', 'euclidean')
    protonet_prototype_type = getattr(args, 'protonet_prototype_type', 'mean')
    use_adaptive_proto = getattr(args, 'use_adaptive_proto', False)
    is_protonet = meta_algorithm == 'protonet'

    train_accs, test_accs = [], []
    best_test_acc = 0.0
    best_eval_acc = 0.0
    train_domains = [int(i) for i in args.train_domains.split(',')]
    test_meta = domain_meta_map[args.test_domain]
    fusion_only = args.freeze_backbone
    use_episode_aug = getattr(args, 'episode_aug', False)
    use_illumination_aug = getattr(args, 'aug_illumination', False)
    cross_domain = getattr(args, 'cross_domain_episodes', False)
    eval_patience = getattr(args, 'eval_patience', 0)
    early_stop_min_delta = getattr(args, 'early_stop_min_delta', 0.0)
    evals_without_improve = 0
    best_eval_iter = 0
    stopped_early = False

    if cross_domain:
        from common.cross_domain_episode import (
            build_cross_domain_pool,
            cross_domain_pool_stats,
            sample_cross_domain_episode,
        )
        cd_pools, cd_datasets = build_cross_domain_pool(
            args.data_dir_path, train_domains, rgb_only=args.rgb_only)
        stats = cross_domain_pool_stats(cd_pools)
        logging.info(
            'Cross-domain episodes ON (scratch/damage pooled over domains {}): '
            'scratch={}, damage={}.'.format(
                train_domains, stats.get(0, 0), stats.get(1, 0)))
    else:
        cd_pools, cd_datasets = None, None
    if use_episode_aug:
        logging.info('Episode augmentation ON (meta-train only, 4 policies).')
    if use_illumination_aug:
        logging.info('Illumination augmentation ON (gamma+brightness+specular, meta-train only).')
    if eval_patience > 0:
        logging.info(
            'Early stopping ON: patience={} evals, min_delta={:.4f}.'.format(
                eval_patience, early_stop_min_delta))

    input_scales_str = getattr(args, 'input_scales', '') or ''
    train_scales = [args.yolo_imgsz]
    if input_scales_str.strip():
        try:
            train_scales = sorted(set(int(s.strip()) for s in input_scales_str.split(',') if s.strip()))
            logging.info('Multi-scale train ON: scales={}'.format(train_scales))
        except Exception:
            logging.warning('Failed to parse input_scales="{}", using default 640.'.format(input_scales_str))
            train_scales = [args.yolo_imgsz]

    for iteration in range(1, args.iters + 1):
        w_rl = curriculum_w_rl(
            iteration,
            stage1_end=args.stage1_end,
            stage2_end=args.stage2_end,
            w_rl_max=args.w_rl_max,
        )
        adapt_steps = curriculum_adapt_steps(
            iteration, default_steps=args.adapt_steps, stage3_start=args.stage2_end)

        opt.zero_grad()
        sum_train = {'loss': 0., 'acc': 0., 'depth': 0.}
        sum_test = {'loss': 0., 'acc': 0.}
        if cross_domain:
            train_meta = domain_meta_map[random.choice(train_domains)]
        else:
            train_index = random.randint(0, len(train_domains) - 1)
            train_domain = train_domains[train_index]
            train_meta = domain_meta_map[train_domain]

        for _ in range(args.meta_batch_size):
            train_size = random.choice(train_scales)

            if cross_domain:
                from common.cross_domain_episode import sample_cross_domain_episode
                batch = sample_cross_domain_episode(
                    cd_pools, cd_datasets, args.ways, args.shots)
            else:
                batch = train_tasks[train_index].sample()

            support, query = split_support_query(
                batch, args.shots, args.ways, device)

            # multi-scale resize
            if len(train_scales) > 1:
                if support[0].shape[-1] != train_size or support[0].shape[-2] != train_size:
                    support = (F.interpolate(support[0], size=(train_size, train_size),
                                             mode='bilinear', align_corners=False), *support[1:])
                    query = (F.interpolate(query[0], size=(train_size, train_size),
                                           mode='bilinear', align_corners=False), *query[1:])

            if use_illumination_aug:
                support, query, _ = augment_support_query(support, query, policy_id=4)
            elif use_episode_aug:
                support, query, _ = augment_support_query(support, query)

            if is_protonet:
                meta_loss, train_acc = protonet_episode_loss(
                    model, support, query, train_meta,
                    distance=protonet_distance,
                    prototype_type=protonet_prototype_type,
                    mahal_head=model.mahal_head,
                    adaptive_proto=model.adaptive_proto if use_adaptive_proto else None)
                train_depth = 1.0
            else:
                learner = maml.clone()
                param_list = learner.module.adapt_parameters(fusion_only=fusion_only)
                adapt_supervised(
                    learner, support, train_meta, adapt_steps, param_list=param_list)
                meta_loss, _, train_acc, train_depth = meta_task_loss(
                    learner, query, train_meta, value_net, w_rl)
            meta_loss.backward()
            sum_train['loss'] += meta_loss.item()
            sum_train['acc'] += train_acc
            sum_train['depth'] += train_depth

            test_batch = test_tasks.sample()
            test_support, test_query = split_support_query(
                test_batch, args.shots, args.ways, device)

            if is_protonet:
                with torch.no_grad():
                    test_loss, test_acc = evaluate_protonet_query(
                        model, test_support, test_query, test_meta,
                        distance=protonet_distance,
                        prototype_type=protonet_prototype_type,
                        mahal_head=model.mahal_head,
                        adaptive_proto=model.adaptive_proto if use_adaptive_proto else None)
            else:
                learner = maml.clone()
                adapt_supervised(
                    learner, test_support, test_meta, adapt_steps,
                    param_list=learner.module.adapt_parameters(fusion_only=fusion_only))
                with torch.no_grad():
                    test_loss, test_acc = evaluate_query(learner, test_query, test_meta)
            sum_test['loss'] += test_loss
            sum_test['acc'] += test_acc

        for p in model.meta_train_parameters(fusion_only=fusion_only):
            if p.grad is not None:
                p.grad.data.mul_(1.0 / args.meta_batch_size)
        clip_grad_norm_(
            model.meta_train_parameters(fusion_only=fusion_only),
            max_norm=args.grad_clip)
        opt.step()

        bs = args.meta_batch_size
        train_acc = sum_train['acc'] / bs
        test_acc = sum_test['acc'] / bs
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        eval_mean, eval_std = None, None
        if args.eval_tasks > 0 and (
                iteration % args.eval_every == 0 or iteration == args.iters):
            eval_mean, eval_std, _ = evaluate_tasks(
                model if is_protonet else maml,
                test_tasks, test_meta, args.shots, args.ways,
                adapt_steps, args.eval_tasks, device,
                fusion_only=fusion_only,
                meta_algorithm=meta_algorithm,
                protonet_distance=protonet_distance,
                protonet_prototype_type=protonet_prototype_type,
                mahal_head=model.mahal_head if is_protonet else None)
            logging.info(
                'Eval@{}: domain_{} mean acc {:.4f} +/- {:.4f} ({} tasks).'.format(
                    iteration, args.test_domain, eval_mean, eval_std, args.eval_tasks))

        if args.log:
            print_logs(
                iteration,
                1.0 - train_acc,
                train_acc,
                1.0 - test_acc,
                test_acc,
                mean_depth=sum_train['depth'] / bs,
                w_rl=w_rl,
            )

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if eval_mean is not None:
            if eval_mean > best_eval_acc + early_stop_min_delta:
                best_eval_acc = eval_mean
                best_eval_iter = iteration
                evals_without_improve = 0
                if args.checkpoint:
                    path = os.path.join(args.checkpoint_path, 'best.pt')
                    torch.save(model.state_dict(), path)
                    torch.save(model.state_dict(), os.path.join(
                        args.checkpoint_path, '{}_best.pt'.format(experiment_title)))
                    logging.info(
                        'New best eval acc {:.4f}, saved to {}'.format(best_eval_acc, path))
            else:
                evals_without_improve += 1
                if eval_patience > 0:
                    logging.info(
                        'Eval@{} no improvement ({}/{} patience).'.format(
                            iteration, evals_without_improve, eval_patience))

            if eval_patience > 0 and evals_without_improve >= eval_patience:
                logging.info(
                    'Early stop at iter {} (best eval {:.4f} @ iter {}).'.format(
                        iteration, best_eval_acc, best_eval_iter))
                stopped_early = True
                break

        if args.plot and iteration % args.plot_step == 0:
            _plot(args, iteration, train_accs, test_accs, experiment_title)

        if args.checkpoint and iteration % args.checkpoint_step == 0:
            torch.save(model.state_dict(),
                os.path.join(args.checkpoint_path, 'iter_{:04d}.pt'.format(iteration)))

    summary = {
        'experiment_title': experiment_title,
        'exp_name': getattr(args, 'exp_name', ''),
        'best_single_task_acc': best_test_acc,
        'best_eval_mean_acc': best_eval_acc,
        'best_eval_iter': best_eval_iter,
        'stopped_early': stopped_early,
        'iters': args.iters,
        'actual_iters': iteration,
        'train_domains': args.train_domains,
        'test_domain': args.test_domain,
        'w_rl_max': args.w_rl_max,
        'freeze_backbone': args.freeze_backbone,
        'backbone_meta_lr': getattr(args, 'backbone_meta_lr', None),
        'head_meta_lr': getattr(args, 'head_meta_lr', None),
        'episode_aug': getattr(args, 'episode_aug', False),
        'aug_illumination': getattr(args, 'aug_illumination', False),
        'input_scales': getattr(args, 'input_scales', '') or '',
        'cross_domain_episodes': getattr(args, 'cross_domain_episodes', False),
        'eval_patience': getattr(args, 'eval_patience', 0),
        'early_stop_min_delta': getattr(args, 'early_stop_min_delta', 0.0),
        'adapt_steps': args.adapt_steps,
        'meta_algorithm': meta_algorithm,
        'protonet_distance': protonet_distance if is_protonet else None,
        'protonet_prototype_type': protonet_prototype_type if is_protonet else None,
        'use_adaptive_proto': use_adaptive_proto if is_protonet else None,
    }
    summary_path = os.path.join(args.checkpoint_path, 'train_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(
        'Training done. Best single-task acc: {:.4f}, best eval mean: {:.4f}.'.format(
            best_test_acc, best_eval_acc))
    logging.info('Summary saved to {}.'.format(summary_path))
    return train_accs, test_accs


def _plot(args, iteration, train_accs, test_accs, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(train_accs, label='train acc')
    plt.plot(test_accs, label='test acc')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    os.makedirs(args.plot_path, exist_ok=True)
    plt.savefig(os.path.join(args.plot_path, '{}_{}.png'.format(title, iteration)))
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    if args.anil and not args._freeze_backbone_explicit:
        args.freeze_backbone = True
    _apply_exp_dirs(args)

    if args.track == 'automotive':
        prefix = 'CMPL_ProtoNet' if args.meta_algorithm == 'protonet' else 'CMPL_MAML'
    else:
        prefix = 'AviationMAML'
    experiment_title = '{}_{}w_{}s'.format(prefix, args.ways, args.shots)
    if args.rgb_only:
        experiment_title += '_rgb'
    experiment_title += '_frozen' if args.freeze_backbone else '_unfreeze'
    if args.episode_aug:
        experiment_title += '_epaug'
    if getattr(args, 'aug_illumination', False):
        experiment_title += '_illum'
    scales_str = getattr(args, 'input_scales', '') or ''
    if scales_str.strip():
        scales_abbrev = '_'.join(s.strip() for s in scales_str.split(',') if s.strip())
        experiment_title += '_ms' + scales_abbrev
    if args.cross_domain_episodes:
        experiment_title += '_crossdom'
    if args.meta_algorithm == 'protonet':
        experiment_title += '_protonet'
    if args.rgb_cbam:
        experiment_title += '_cbam'
    experiment_title += '_yolo26'

    for path in (args.plot_path, args.checkpoint_path, args.log_path):
        if path:
            os.makedirs(path, exist_ok=True)

    _save_run_config(args, experiment_title)

    if args.log:
        setup_logger(args.log_path, experiment_title)

    _train_loop(args, experiment_title)
