"""Multi-task evaluation for MAML and ProtoNet on held-out domain."""

import numpy as np
import torch
import torch.nn.functional as F

from .maml_loop import adapt_supervised, evaluate_query, split_support_query
from .protonet_loop import (
    evaluate_protonet_query,
    prototypical_logits,
    build_prototypes,
    evaluate_tasks_protonet,
)


def evaluate_tasks_maml(
        maml, test_tasks, domain_meta, shots, ways, adapt_steps, n_tasks, device,
        fusion_only=False):
    accs = []
    for _ in range(n_tasks):
        batch = test_tasks.sample()
        support, query = split_support_query(batch, shots, ways, device)
        learner = maml.clone()
        adapt_supervised(
            learner, support, domain_meta, adapt_steps,
            param_list=learner.module.adapt_parameters(fusion_only=fusion_only))
        with torch.no_grad():
            _, acc = evaluate_query(learner, query, domain_meta)
        accs.append(acc)
    arr = np.array(accs, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), accs


def _tta_protonet_logits(model, support, query, domain_meta, device,
                          distance, prototype_type, mahal_head, tta_scales, base_imgsz):
    """
    Multi-scale TTA: extract logits at multiple input scales and average.
    tta_scales: list of int, e.g. [320, 480, 640]
    base_imgsz: the model's native input size (must be in tta_scales)
    Returns: averaged logits over TTA scales.
    """
    # build prototypes at native scale (support has no augmentation)
    _unused, acc = evaluate_protonet_query(
        model, support, query, domain_meta,
        distance=distance, prototype_type=prototype_type, mahal_head=mahal_head)
    # store prototypes for TTA
    s_rgb = support[0]
    prototypes, _ = build_prototypes(
        s_rgb, domain_meta, distance=distance,
        prototype_type=prototype_type, mahal_head=mahal_head)

    q_rgb = query[0]
    all_logits = []
    for scale in tta_scales:
        # resize query to this scale, encode, resize back if needed
        if scale != base_imgsz:
            q_resized = F.interpolate(
                q_rgb, size=(scale, scale), mode='bilinear', align_corners=False)
        else:
            q_resized = q_rgb
        # encode at this scale (model handles resize internally, but we force it)
        # we need to bypass the model's internal resize to keep original scale intent
        # simplest: just encode as-is and let model interpolate
        with torch.no_grad():
            q_feat = model.encode(q_resized)
            # prototypes are already at base_imgsz; re-extract prototypes at this scale
            # (support is still at base_imgsz — use base prototypes, not ideal but stable)
            if scale == base_imgsz:
                proto_feat = prototypes
            else:
                # re-encode support at this scale too
                s_resized = F.interpolate(
                    s_rgb, size=(scale, scale), mode='bilinear', align_corners=False)
                proto_feat, _ = build_prototypes(
                    s_resized, domain_meta, distance=distance,
                    prototype_type=prototype_type, mahal_head=mahal_head)
            logits = prototypical_logits(
                q_feat, proto_feat, distance=distance, mahal_head=mahal_head)
        all_logits.append(logits)

    avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    pred = avg_logits.argmax(dim=1)
    correct = (pred == query[3].long()).sum().item()
    acc = correct / query[3].size(0)
    return avg_logits, acc


def evaluate_tasks_protonet_tta(
        model, test_tasks, domain_meta, shots, ways, n_tasks, device,
        distance='euclidean', prototype_type='mean', mahal_head=None,
        tta_scales=(320, 480, 640), base_imgsz=640):
    """
    Multi-scale TTA evaluation for ProtoNet.
    At each task, extract logits at each TTA scale and average.
    tta_scales: tuple of input sizes to average over.
    base_imgsz: the model's native training scale (included in tta_scales).
    """
    accs = []
    model.eval()
    for _ in range(n_tasks):
        batch = test_tasks.sample()
        support, query = split_support_query(batch, shots, ways, device)
        with torch.no_grad():
            _, acc = _tta_protonet_logits(
                model, support, query, domain_meta, device,
                distance=distance, prototype_type=prototype_type,
                mahal_head=mahal_head,
                tta_scales=tta_scales, base_imgsz=base_imgsz)
        accs.append(acc)
    arr = np.array(accs, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), accs


def evaluate_tasks(
        learner_or_model, test_tasks, domain_meta, shots, ways, adapt_steps, n_tasks,
        device, fusion_only=False, meta_algorithm='maml',
        protonet_distance='euclidean', protonet_prototype_type='mean',
        mahal_head=None, adaptive_proto=None,
        tta_scales=None,
        tta_base_imgsz=640):
    """Unified eval entry: MAML wrapper or bare model for ProtoNet.

    tta_scales: if set, enables multi-scale TTA for ProtoNet (e.g. (320,480,640)).
    adaptive_proto: optional AdaptivePrototype module (query-aware prototype refinement).
    """
    if meta_algorithm == 'protonet':
        if tta_scales:
            return evaluate_tasks_protonet_tta(
                learner_or_model, test_tasks, domain_meta, shots, ways, n_tasks, device,
                distance=protonet_distance,
                prototype_type=protonet_prototype_type,
                mahal_head=mahal_head,
                tta_scales=tta_scales,
                base_imgsz=tta_base_imgsz)
        return evaluate_tasks_protonet(
            learner_or_model, test_tasks, domain_meta, shots, ways, n_tasks, device,
            distance=protonet_distance,
            prototype_type=protonet_prototype_type,
            mahal_head=mahal_head,
            adaptive_proto=adaptive_proto)
    return evaluate_tasks_maml(
        learner_or_model, test_tasks, domain_meta, shots, ways, adapt_steps, n_tasks,
        device, fusion_only=fusion_only)
