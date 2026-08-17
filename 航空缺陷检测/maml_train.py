"""Multimodal meta-learning training (MAML + ProtoNet) for aviation defect detection."""

import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from common import maml_origin
from common.maml_loop import (
    adapt_supervised,
    curriculum_adapt_steps,
    curriculum_w_rl,
    evaluate_query,
    meta_task_loss,
    split_support_query,
)
from common.episode_aug import augment_support_query, augment_rgb
from common.cross_domain_episode import (
    build_cross_domain_pool,
    cross_domain_pool_stats,
    sample_cross_domain_episode,
)
from common.protonet_loop import evaluate_protonet_query, protonet_episode_loss
from common.eval_loop import evaluate_tasks
from common.multimodal_model import AviationDefectModel, ValueBaseline
from common.task_builder import create_tasksets
from common.utils import print_logs


def create_model(args, device):
    meta_algorithm = getattr(args, 'meta_algorithm', 'maml')
    model = AviationDefectModel(
        num_classes=args.ways,
        embed_dim=args.embed_dim,
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
        rgb_cbam=getattr(args, 'rgb_cbam', False),
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
            'Meta algorithm: ProtoNet (distance={}, prototype={}).'.format(
                distance, prototype_type))
        return model, None, opt, model.mahal_head

    logging.info(
        'Meta algorithm: MAML (inner loop adapts classifier; '
        'fast_lr={:.4g}).'.format(args.fast_lr))
    maml = maml_origin.MAML(
        model, lr=args.fast_lr, first_order=args.first_order, allow_nograd=True)
    value_net = ValueBaseline(state_dim=model.state_dim()).to(device)
    return model, maml, opt, value_net


def train(args, experiment_title):
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

    def _apply_input_scale(s_rgb, q_rgb, target_size):
        """Resize support/query RGB tensors to target_size before encoding."""
        print(f"[DEBUG _apply_input_scale] IN s={s_rgb.shape}, q={q_rgb.shape}, target={target_size}")
        if s_rgb.shape[-1] != target_size or s_rgb.shape[-2] != target_size:
            s_rgb = F.interpolate(s_rgb, size=(target_size, target_size),
                                  mode='bilinear', align_corners=False)
            q_rgb = F.interpolate(q_rgb, size=(target_size, target_size),
                                  mode='bilinear', align_corners=False)
        print(f"[DEBUG _apply_input_scale] OUT s={s_rgb.shape}, q={q_rgb.shape}")
        return s_rgb, q_rgb

    cd_pools, cd_datasets = None, None
    if cross_domain:
        cd_pools, cd_datasets = build_cross_domain_pool(
            args.data_dir_path, train_domains, rgb_only=args.rgb_only)
        stats = cross_domain_pool_stats(cd_pools)
        logging.info(
            'Cross-domain episodes ON (scratch/damage pooled over domains {}): '
            'scratch={}, damage={}.'.format(
                train_domains, stats.get(0, 0), stats.get(1, 0)))
    if use_episode_aug:
        logging.info('Episode augmentation ON (meta-train only, 4 policies).')
    if use_illumination_aug:
        logging.info('Illumination augmentation ON (gamma+brightness+specular, meta-train only).')
    if eval_patience > 0:
        logging.info(
            'Early stopping ON: patience={} evals, min_delta={:.4f}.'.format(
                eval_patience, early_stop_min_delta))

    # --- multi-scale train input parsing ---
    input_scales_str = getattr(args, 'input_scales', '') or ''
    train_scales = [args.yolo_imgsz]  # default fallback
    if input_scales_str.strip():
        try:
            train_scales = [int(s.strip()) for s in input_scales_str.split(',') if s.strip()]
            train_scales = sorted(set(train_scales))
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
        sum_test = {'loss': 0., 'acc': 0., 'depth': 0.}
        if cross_domain:
            train_meta = domain_meta_map[random.choice(train_domains)]
        else:
            train_index = random.randint(0, len(train_domains) - 1)
            train_domain = train_domains[train_index]
            train_meta = domain_meta_map[train_domain]

        for _ in range(args.meta_batch_size):
            # --- multi-scale: randomly pick a train scale ---
            train_size = random.choice(train_scales)

            if cross_domain:
                batch = sample_cross_domain_episode(
                    cd_pools, cd_datasets, args.ways, args.shots)
            else:
                batch = train_tasks[train_index].sample()
            print(f"[DEBUG] batch type={type(batch)}, is_seq={isinstance(batch, (list,tuple))}")
            if isinstance(batch, (list, tuple)):
                print(f"[DEBUG] batch[0] shapes: rgb={batch[0][0].shape if hasattr(batch[0][0],'shape') else 'N/A'}")
            support, query = split_support_query(
                batch, args.shots, args.ways, device)
            print(f"[DEBUG split] s={support[0].shape}, q={query[0].shape}")

            # --- multi-scale resize (before aug to avoid size artifacts) ---
            if len(train_scales) > 1:
                s_rgb, q_rgb = _apply_input_scale(support[0], query[0], train_size)
                support = (s_rgb, *support[1:])
                query = (q_rgb, *query[1:])

            # --- illumination augmentation (meta-train only, policy_id=4) ---
            if use_illumination_aug:
                support, query, _ = augment_support_query(support, query, policy_id=4)
            elif use_episode_aug:
                support, query, _ = augment_support_query(support, query)

            if is_protonet:
                meta_loss, train_acc = protonet_episode_loss(
                    model, support, query, train_meta,
                    distance=protonet_distance,
                    prototype_type=protonet_prototype_type,
                    mahal_head=model.mahal_head)
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
                        mahal_head=model.mahal_head)
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
                    legacy = os.path.join(
                        args.checkpoint_path, '{}_best.pt'.format(experiment_title))
                    torch.save(model.state_dict(), legacy)
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
            path = os.path.join(args.checkpoint_path, 'iter_{:04d}.pt'.format(iteration))
            torch.save(model.state_dict(), path)

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
