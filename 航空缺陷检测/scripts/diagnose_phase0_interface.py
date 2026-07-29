"""
Gate D1: Phase0 interface diagnosis — LogReg vs frozen ProtoNet 5-shot.

Default: 128-d YOLO backbone features (aligned with diagnose_features B2).

Plan A (--compare-projection): load full Phase0 checkpoint and compare
  - backbone feats (feat_dim, usually 256)
  - SupCon projection feats (proj_dim, L2-normalized, usually 64)

Run from 航空缺陷检测/:
    python scripts/diagnose_phase0_interface.py --track automotive \\
        --weights ../车/weights/cmpl_yolo26_supcon.pt

    # Plan A: backbone vs projection
    python scripts/diagnose_phase0_interface.py --track automotive \\
        --compare-projection --checkpoint ../车/checkpoints/pretrain_v4_best.pt

Output:
    reports/diagnose/{track}/interface_diagnosis.json
    reports/diagnose/{track}/interface_diagnosis_proj.json  (Plan A)
"""

from __future__ import print_function

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.aviation_datasets import AviationDefectDataset  # noqa: E402
from common.backbones import YOLO26BackboneEncoder  # noqa: E402
from common.maml_loop import split_support_query  # noqa: E402
from common.pretrain_model import PretrainMultitaskModel  # noqa: E402
from common.project_paths import apply_meta_train_defaults, resolve_track  # noqa: E402
from common.protonet_loop import prototypical_logits  # noqa: E402
from common.task_builder import create_tasksets  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 128
YOLO_IMGSZ = 640
GAP_NEED_SUPCON = 0.05
GAP_OK = 0.03


def _collate(batch):
    xs, ys = zip(*batch)
    rgbs, thermals, ultrasonics = zip(*xs)
    return (
        torch.stack(rgbs),
        torch.stack(thermals),
        torch.stack(ultrasonics),
    ), torch.stack(ys)


def extract_encoder_features(encode_fn, domain_id, data_dir, dataset_name, batch_size=16):
    """Extract features via encode_fn(rgb) -> [B, D]."""
    ds = AviationDefectDataset(
        domain_id, data_dir, train=False, rgb_only=True, dataset_name=dataset_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    chunks, labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            rgb, _, _ = [t.to(DEVICE) for t in batch_x]
            chunks.append(encode_fn(rgb).cpu().float().numpy())
            labels.append(batch_y.numpy())
    return np.concatenate(chunks, axis=0), np.concatenate(labels, axis=0)


def logreg_cv(features, labels, desc=''):
    y = labels.astype(int)
    n_splits = min(5, int(np.min(np.bincount(y))))
    if n_splits < 2:
        print('  {}: too few samples per class'.format(desc))
        return 0.0, 0.0, int(len(y))
    X = StandardScaler().fit_transform(features)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    mean, std = float(scores.mean()), float(scores.std())
    print('  {}: {:.3f} ± {:.3f} (n={}, {}-fold, dim={})'.format(
        desc, mean, std, len(y), n_splits, features.shape[1]))
    return mean, std, int(len(y))


def frozen_protonet_eval(encode_fn, args, label=''):
    """ProtoNet 5-shot episodic eval on frozen encode_fn features."""
    _, test_tasks, _ = create_tasksets(args)
    accs = []
    for _ in range(args.eval_tasks):
        batch = test_tasks.sample()
        support, query = split_support_query(batch, args.shots, args.ways, DEVICE)
        s_rgb, _, _, s_y = support
        q_rgb, _, _, q_y = query
        with torch.no_grad():
            s_emb = encode_fn(s_rgb)
            q_emb = encode_fn(q_rgb)
            ways = int(max(s_y.max().item(), q_y.max().item())) + 1
            logits = prototypical_logits(
                q_emb, s_emb, s_y, ways, distance=args.protonet_distance)
            acc = (logits.argmax(dim=-1) == q_y).float().mean().item()
        accs.append(acc)
    arr = np.array(accs, dtype=np.float64)
    mean, std = float(arr.mean()), float(arr.std())
    print('  frozen ProtoNet {} [{}]: {:.3f} ± {:.3f} ({} tasks)'.format(
        '{}-way {}-shot'.format(args.ways, args.shots), label, mean, std, args.eval_tasks))
    return mean, std, accs


def decision(logreg_mean, protonet_mean):
    gap = logreg_mean - protonet_mean
    if gap > GAP_NEED_SUPCON:
        verdict = 'need_supcon_v4'
        note = '接口断层明显 (gap>{:.0%})：Phase0 改 SupCon+CE 后再跑 M1'.format(GAP_NEED_SUPCON)
    elif gap <= GAP_OK:
        verdict = 'ok_proceed_m1'
        note = '接口正常 (gap≤{:.0%})：可推进 M1（仍需 Gate D2/D3）'.format(GAP_OK)
    else:
        verdict = 'borderline'
        note = '接口边界 (gap {:.1%})：可并行推进 M1，SupCon 作 P1 消融'.format(gap)
    return gap, verdict, note


def _infer_task_specs(state):
    specs = []
    for key, tensor in state.items():
        if key.startswith('heads.') and key.endswith('.weight'):
            name = key.split('.')[1]
            specs.append((name, int(tensor.shape[0])))
    if not specs:
        raise RuntimeError('No task heads found in checkpoint.')
    return specs


def load_pretrain_for_projection(checkpoint_path, yolo_init):
    """Load PretrainMultitaskModel with projection from Phase0 best.pt."""
    ckpt = torch.load(str(checkpoint_path), map_location='cpu')
    state = ckpt['model_state']
    task_specs = _infer_task_specs(state)
    feat_dim = int(state['heads.{}.weight'.format(task_specs[0][0])].shape[1])
    proj_w = state.get('projection.0.weight')
    if proj_w is None:
        raise RuntimeError('Checkpoint has no projection head (loss_mode != supcon_ce?).')
    proj_dim = int(proj_w.shape[0])
    model = PretrainMultitaskModel(
        yolo_weights=str(yolo_init),
        yolo_imgsz=YOLO_IMGSZ,
        feat_dim=feat_dim,
        task_specs=task_specs,
        loss_mode='supcon_ce',
        proj_dim=proj_dim,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print('  warn missing keys: {}'.format(missing[:8]))
    if unexpected:
        print('  warn unexpected keys: {}'.format(unexpected[:8]))
    model = model.to(DEVICE).eval()
    return model, feat_dim, proj_dim, ckpt


def parse_args():
    p = argparse.ArgumentParser(description='Gate D1: LogReg vs frozen ProtoNet interface diagnosis')
    p.add_argument('--track', default='automotive', help='aerospace|automotive')
    p.add_argument('--weights', default=None, help='Phase0 YOLO backbone .pt (default D1)')
    p.add_argument('--checkpoint', default=None,
                   help='Full Phase0 best.pt (required for --compare-projection)')
    p.add_argument('--compare-projection', action='store_true',
                   help='Plan A: compare backbone feats vs SupCon projection feats')
    p.add_argument('--test-domain', type=int, default=None)
    p.add_argument('--ways', type=int, default=2)
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--eval-tasks', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--protonet-distance', default='euclidean')
    p.add_argument('--batch-size', type=int, default=16)
    return p.parse_args()


def _meta_args(info, args):
    meta_args = Namespace(
        track=info['track'],
        data_dir_path=info['processed'],
        dataset_name=info['dataset_name'],
        train_domains='0',
        test_domain=args.test_domain if args.test_domain is not None else 3,
        ways=args.ways,
        shots=args.shots,
        train_task_num=40,
        test_task_num=40,
        rgb_only=True,
        exclude_sample_ids='',
        eval_tasks=args.eval_tasks,
        protonet_distance=args.protonet_distance,
    )
    apply_meta_train_defaults(meta_args)
    if info['track'] == 'automotive':
        meta_args.train_domains = '0'
    return meta_args


def run_space(name, encode_fn, meta_args, batch_size):
    print('\n--- Space: {} ---'.format(name))
    print('[1] LogReg CV')
    features, labels = extract_encoder_features(
        encode_fn, meta_args.test_domain, meta_args.data_dir_path,
        meta_args.dataset_name, batch_size=batch_size)
    logreg_mean, logreg_std, n_samples = logreg_cv(
        features, labels, desc='LogReg domain_{}'.format(meta_args.test_domain))
    print('[2] Frozen ProtoNet')
    protonet_mean, protonet_std, _ = frozen_protonet_eval(encode_fn, meta_args, label=name)
    gap, verdict, note = decision(logreg_mean, protonet_mean)
    print('  Gap={:+.3f}  Verdict={}'.format(gap, verdict))
    return {
        'space': name,
        'embed_dim': int(features.shape[1]),
        'n_samples': n_samples,
        'logreg_mean': logreg_mean,
        'logreg_std': logreg_std,
        'frozen_protonet_mean': protonet_mean,
        'frozen_protonet_std': protonet_std,
        'gap_logreg_minus_protonet': gap,
        'verdict': verdict,
        'note': note,
    }


def main_compare_projection(args, info, meta_args):
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        Path(info['checkpoints_dir']) / 'pretrain_v4_best.pt')
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            'Full Phase0 checkpoint not found: {}. Pass --checkpoint.'.format(ckpt_path))

    yolo_init = Path(info['weights_dir']) / info['yolo_init_name']
    if not yolo_init.is_file():
        yolo_init = Path(info['weights_dir']) / 'yolo26n.pt'

    print('=' * 60)
    print('Gate D1 Plan A: backbone vs SupCon projection')
    print('  track: {}'.format(info['track']))
    print('  device: {}'.format(DEVICE))
    print('  checkpoint: {}'.format(ckpt_path))
    print('  test domain: {}'.format(meta_args.test_domain))
    print('  protocol: {}-way {}-shot, {} tasks'.format(
        meta_args.ways, meta_args.shots, args.eval_tasks))
    print('=' * 60)

    model, feat_dim, proj_dim, ckpt = load_pretrain_for_projection(ckpt_path, yolo_init)
    print('  loaded epoch={} score={:.3f} loss_mode={} feat_dim={} proj_dim={}'.format(
        ckpt.get('epoch'), float(ckpt.get('score', 0)), ckpt.get('loss_mode'),
        feat_dim, proj_dim))

    def encode_backbone(rgb):
        return model.encode(rgb)

    def encode_projection(rgb):
        return model.project(model.encode(rgb))

    bb = run_space('backbone_{}d'.format(feat_dim), encode_backbone, meta_args, args.batch_size)
    pj = run_space('projection_{}d_l2'.format(proj_dim), encode_projection, meta_args, args.batch_size)

    delta_gap = bb['gap_logreg_minus_protonet'] - pj['gap_logreg_minus_protonet']
    delta_pn = pj['frozen_protonet_mean'] - bb['frozen_protonet_mean']

    if pj['gap_logreg_minus_protonet'] <= GAP_OK and delta_pn >= 0.05:
        plan_a = 'projection_helps'
        plan_note = (
            'Projection gap 达标且 ProtoNet 明显高于 backbone → '
            'SupCon 有效但导出/评测用错了空间；考虑 meta 接入 projection 或把 SupCon 打到 backbone')
    elif delta_pn >= 0.03 or delta_gap >= 0.03:
        plan_a = 'projection_partial'
        plan_note = (
            'Projection 略好于 backbone → SupCon 有弱信号；建议 v4.1 提高 supcon_weight '
            '或去掉独立 proj、直接约束 backbone')
    else:
        plan_a = 'projection_no_help'
        plan_note = (
            'Projection 未明显改善 episodic 几何 → SupCon 训练目标/任务混合可能无效；'
            '改 v4.1（权重/温度/backbone 直连）或 episodic proxy')

    print('\n' + '=' * 60)
    print('Plan A Summary')
    print('  Backbone   LogReg={:.3f}  ProtoNet={:.3f}  Gap={:+.3f}'.format(
        bb['logreg_mean'], bb['frozen_protonet_mean'], bb['gap_logreg_minus_protonet']))
    print('  Projection LogReg={:.3f}  ProtoNet={:.3f}  Gap={:+.3f}'.format(
        pj['logreg_mean'], pj['frozen_protonet_mean'], pj['gap_logreg_minus_protonet']))
    print('  Δ ProtoNet (proj-bb): {:+.3f}'.format(delta_pn))
    print('  Δ Gap shrink (bb-proj): {:+.3f}'.format(delta_gap))
    print('  Verdict: {}'.format(plan_a))
    print('  → {}'.format(plan_note))
    print('=' * 60)

    out_dir = PROJECT_DIR / 'reports' / 'diagnose' / info['track']
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'gate': 'D1_plan_a_backbone_vs_projection',
        'track': info['track'],
        'device': str(DEVICE),
        'checkpoint': str(ckpt_path),
        'test_domain': meta_args.test_domain,
        'ways': meta_args.ways,
        'shots': meta_args.shots,
        'eval_tasks': args.eval_tasks,
        'seed': args.seed,
        'backbone': bb,
        'projection': pj,
        'delta_protonet_proj_minus_backbone': delta_pn,
        'delta_gap_shrink_backbone_minus_proj': delta_gap,
        'plan_a_verdict': plan_a,
        'plan_a_note': plan_note,
        'thresholds': {
            'need_supcon_v4_if_gap_gt': GAP_NEED_SUPCON,
            'ok_proceed_m1_if_gap_le': GAP_OK,
        },
    }
    out_path = out_dir / 'interface_diagnosis_proj.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('\nReport saved: {}'.format(out_path))


def main_default_d1(args, info, meta_args):
    weights_path = Path(args.weights) if args.weights else Path(
        info['weights_dir']) / info['pretrain_weights_name']
    if not weights_path.is_file():
        raise FileNotFoundError('Weights not found: {}'.format(weights_path))

    out_dir = PROJECT_DIR / 'reports' / 'diagnose' / info['track']
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Gate D1: Phase0 interface diagnosis')
    print('  track: {}'.format(info['track']))
    print('  device: {}'.format(DEVICE))
    print('  weights: {}'.format(weights_path))
    print('  feature space: 128-d YOLO backbone (Phase0)')
    print('  test domain: {}'.format(meta_args.test_domain))
    print('  protocol: {}-way {}-shot, {} tasks/seed'.format(
        meta_args.ways, meta_args.shots, args.eval_tasks))
    print('=' * 60)

    backbone = YOLO26BackboneEncoder(
        out_dim=EMBED_DIM, imgsz=YOLO_IMGSZ, weights_path=str(weights_path),
    ).to(DEVICE)
    backbone.eval()

    result = run_space('backbone_128d', backbone.encode, meta_args, args.batch_size)
    gap, verdict, note = result['gap_logreg_minus_protonet'], result['verdict'], result['note']

    print('\n' + '=' * 60)
    print('Summary')
    print('  LogReg CV:        {:.3f} ± {:.3f}'.format(result['logreg_mean'], result['logreg_std']))
    print('  Frozen ProtoNet:  {:.3f} ± {:.3f}'.format(
        result['frozen_protonet_mean'], result['frozen_protonet_std']))
    print('  Gap (LogReg-PN):  {:+.3f}'.format(gap))
    print('  Verdict:          {}'.format(verdict))
    print('  → {}'.format(note))
    print('=' * 60)

    report = {
        'gate': 'D1_phase0_interface',
        'track': info['track'],
        'device': str(DEVICE),
        'weights': str(weights_path),
        'feature_space': '128-d YOLO backbone (aligned with diagnose_features B2)',
        'test_domain': meta_args.test_domain,
        'ways': meta_args.ways,
        'shots': meta_args.shots,
        'eval_tasks': args.eval_tasks,
        'seed': args.seed,
        'n_samples': result['n_samples'],
        'embed_dim': result['embed_dim'],
        'logreg_mean': result['logreg_mean'],
        'logreg_std': result['logreg_std'],
        'frozen_protonet_mean': result['frozen_protonet_mean'],
        'frozen_protonet_std': result['frozen_protonet_std'],
        'gap_logreg_minus_protonet': gap,
        'verdict': verdict,
        'note': note,
        'thresholds': {
            'need_supcon_v4_if_gap_gt': GAP_NEED_SUPCON,
            'ok_proceed_m1_if_gap_le': GAP_OK,
        },
        'reference_meta_trained': {
            'automotive_track2_400iter': 0.625,
            'comment': 'Meta-trained ProtoNet uses full model; this gate tests Phase0 backbone geometry only',
        },
    }
    out_path = out_dir / 'interface_diagnosis.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('\nReport saved: {}'.format(out_path))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    info = resolve_track(args.track)
    meta_args = _meta_args(info, args)

    if args.compare_projection:
        main_compare_projection(args, info, meta_args)
    else:
        main_default_d1(args, info, meta_args)


if __name__ == '__main__':
    main()
