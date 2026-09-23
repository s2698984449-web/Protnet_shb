"""Phase 1: Episode-level Meta-Train 主脚本.

数据流:
  sampler.sample(split='meta_train')  →  train episode
  sampler.sample(split='meta_test')   →  val   episode (每 N iter 评估一次)
# # 2-way 5-shot smoke (30秒,验证链路)
# python "d:\元强化学习\代码\scripts\train_meta.py" --n-way 2 --k-shot 5 --q-query 5 --iters 100 --eval-every 50 --eval-episodes 20 --lr 1e-4 --out-dir "d:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_2w5s_smoke"

# # 5-way 5-shot 标准 (推荐主实验)
# python "d:\元强化学习\代码\scripts\train_meta.py" --n-way 3 --k-shot 5 --q-query 5 --iters 3000 --eval-every 300 --eval-episodes 50 --lr 1e-4 --out-dir "d:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_3w5s"

# # 看 log 最后几行 (稳定后平均)
# Get-Content "d:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_5w5s\train.log" | Select-Object -Last 20
模型:
  - backbone (ResNet18) 来自 Phase 0 预训练, 默认冻结
  - FiLM 用 condition_emb 调制 backbone 输出
  - support/query 都 encode 一次 → 原型 (mean) → 距离 → CE

head (在 ProtoNet 里是 prototype, 无可训 head 参数; ANIL 模式才有).
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.multi_domain_model import MultiDomainPretrainModel
from common.multi_domain_episode import build_episode_samplers
try:
    from common.episode_aug import augment_rgb
    _HAS_AUG = True
except ImportError:
    _HAS_AUG = False


def setup_logger(log_dir: Path, name: str = 'train'):
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f'{name}.log', mode='w', encoding='utf-8')
    sh = logging.StreamHandler()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger = logging.getLogger(name); logger.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# 课程学习：跨域采样调度
# ---------------------------------------------------------------------------
def get_cross_domain_prob(iter_num: int, iters: int = 2000,
                          stages: tuple = (0.6, 0.8, 1.0),
                          probs: tuple = (0.2, 0.4, 0.8)) -> float:
    """根据当前 iter 返回跨域采样概率 (按比例划分阶段).

    默认策略 (20% -> 40% -> 80%):
      阶段 1 (0-60% iters):   跨域概率 20% (先易后难)
      阶段 2 (60-80% iters):  跨域概率 40%
      阶段 3 (80-100% iters): 跨域概率 80% (最难)

    Args:
        iter_num: 当前迭代数
        iters: 总迭代数 (阶段按比例划分)
        stages: 分界点 (总 iters 的比例, 默认 0.6, 0.8, 1.0)
        probs: 对应阶段的跨域概率 (默认 0.2, 0.4, 0.8)
    Returns:
        cross_domain_prob: float ∈ [0, 1]
    """
    # 将比例转换为实际迭代数
    stage1 = int(stages[0] * iters)
    stage2 = int(stages[1] * iters)

    if iter_num < stage1:
        return probs[0]
    elif iter_num < stage2:
        return probs[1]
    else:
        return probs[2]


def is_cross_domain_episode(episode: dict) -> bool:
    """判断当前 episode 是否为跨域样本.

    跨域定义: support 和 query 的 product_id 不同。
    当前 sampler 每个 episode 来自单一 product，所以用 s_pid != q_pid 判断。
    """
    s_pid = episode.get('support_pid', torch.tensor([0]))
    q_pid = episode.get('query_pid', torch.tensor([0]))

    # 提取标量值
    if hasattr(s_pid, 'item'):
        s_pid = s_pid.item() if s_pid.numel() == 1 else s_pid[0].item()
    if hasattr(q_pid, 'item'):
        q_pid = q_pid.item() if q_pid.numel() == 1 else q_pid[0].item()

    return s_pid != q_pid


# ---------------------------------------------------------------------------
# Domain Reliability 加权 Loss
# ---------------------------------------------------------------------------
def weighted_proto_loss(model, mahal_head, episode, device, distance: str = 'mahalanobis',
                        prototype_type: str = 'trim_mean', augment: bool = False,
                        domain_reliability=None, is_cross_domain: bool = False,
                        n_domains: int = 16) -> tuple:
    """ProtoNet loss + Domain Reliability 加权.

    跨域样本: 根据 reliability score 加权，降低不可靠域的贡献
    同域样本: 正常 loss

    Args:
        n_domains: 产品/域的数量 (默认 16)

    Returns:
        loss, acc, n_way, weights (用于加权平均的权重)
    """
    # 先算普通 proto loss
    loss, acc, n_way = proto_loss(model, mahal_head, episode, device,
                                   distance, prototype_type, augment)

    if domain_reliability is None or not is_cross_domain:
        return loss, acc, n_way, torch.ones(1, device=device)

    # 处理 RGB 维度
    q_rgb = episode['query_rgb']
    if q_rgb.dim() == 5:
        q_rgb = q_rgb.squeeze(0)
    q_rgb = q_rgb.to(device)
    q_cond = episode['query_cond'].to(device)
    q_pid = episode['query_pid'].to(device)

    # 提取 query 特征
    with torch.no_grad():
        q_emb = model.encode_with_product(q_rgb, q_cond, q_pid)

    # 用 query 的 product_id 推断 domain_id (每个 product 是一个域)
    domain_ids = _infer_domain_from_product_id(q_pid, n_domains).to(device)

    # 计算 reliability score
    domain_one_hot = F.one_hot(domain_ids, n_domains).float()
    reliability = domain_reliability(q_emb, domain_one_hot)  # (Nq,)

    # 跨域加权: 不可靠的样本权重降低
    weights = reliability / (reliability.mean() + 1e-8)  # 归一化

    # 重新计算 logits 并加权
    q_y = episode['query_y'].to(device)
    logits = _get_logits(model, mahal_head, episode, device, distance, prototype_type, augment)

    # weighted cross entropy
    ce_per_sample = F.cross_entropy(logits, q_y, reduction='none')  # (Nq,)
    loss = (ce_per_sample * weights).mean()

    acc = (logits.argmax(dim=-1) == q_y).float().mean().item()

    return loss, acc, n_way, weights


def _infer_domain_from_product_id(product_ids: torch.Tensor, n_domains: int) -> torch.Tensor:
    """从 product_id 推断 domain id.

    每个 product_id 对应一个域，domain_id = product_id 本身。
    如果 product_id >= n_domains，需要做映射。
    """
    pids = product_ids.clone()
    # 如果 product_id 范围超过 n_domains，做模运算映射
    if pids.max() >= n_domains:
        pids = pids % n_domains
    return pids.long()


def _get_logits(model, mahal_head, episode, device, distance, prototype_type, augment):
    """临时获取 logits 用于加权 loss."""
    from common.protonet_loop import _trim_mean

    def _fix_rgb(t):
        if t.dim() == 5:
            return t.squeeze(0)
        return t

    s_rgb = _fix_rgb(episode['support_rgb'].to(device))
    q_rgb = _fix_rgb(episode['query_rgb'].to(device))
    s_y = episode['support_y'].to(device)
    s_cond = episode['support_cond'].to(device)
    s_pid = episode['support_pid'].to(device)
    q_cond = episode['query_cond'].to(device)
    q_pid = episode['query_pid'].to(device)
    n_way = episode['n_way']

    with torch.set_grad_enabled(model.training):
        s_emb = model.encode_with_product(s_rgb, s_cond, s_pid)
        q_emb = model.encode_with_product(q_rgb, q_cond, q_pid)

    if prototype_type == 'trim_mean':
        protos = []
        for c in range(n_way):
            mask = s_y == c
            if mask.any():
                protos.append(_trim_mean(s_emb[mask], trim_ratio=0.1))
            else:
                protos.append(s_emb.mean(0))
        protos = torch.stack(protos, 0)
    else:
        protos = torch.stack([s_emb[s_y == c].mean(0) for c in range(n_way)], 0)

    if distance == 'cosine':
        logits = torch.mm(F.normalize(q_emb, dim=1), F.normalize(protos, dim=1).t())
    elif distance == 'mahalanobis' and mahal_head is not None:
        logits = mahal_head(q_emb, protos, s_y)
    else:
        q_exp = q_emb.unsqueeze(1)
        p_exp = protos.unsqueeze(0)
        logits = -((q_exp - p_exp) ** 2).sum(-1)

    return logits


def proto_loss(model, mahal_head, episode, device, distance: str = 'mahalanobis',
               prototype_type: str = 'trim_mean', augment: bool = False):
    """Compute ProtoNet episode loss/acc on a single episode.

    Args:
        model: MultiDomainPretrainModel
        mahal_head: MahalanobisHead or None
        episode: dict from sampler.sample()
        device: torch device
        distance: 'euclidean' | 'cosine' | 'mahalanobis'
        prototype_type: 'mean' | 'trim_mean'
        augment: 是否对 sup/q 各自独立采样增强策略 (仅 RGB)
    Returns:
        loss, acc, n_way
    """
    # ── 形状断言 (检测 5D 异常, 自动 squeeze 掉多余 batch 维) ──
    def _fix_rgb(t, name):
        if t.dim() == 5:
            assert t.size(0) == 1, f'{name} 5D with batch>1: {t.shape}'
            return t.squeeze(0)
        return t
    s_rgb = _fix_rgb(episode['support_rgb'].to(device), 's_rgb')
    q_rgb = _fix_rgb(episode['query_rgb'].to(device), 'q_rgb')
    s_y   = episode['support_y'].to(device)
    s_pid = episode['support_pid'].to(device)
    s_cond = episode['support_cond'].to(device)
    q_y   = episode['query_y'].to(device)
    q_pid = episode['query_pid'].to(device)
    q_cond = episode['query_cond'].to(device)
    n_way = episode['n_way']

    # ── 可选: episode 级数据增强 (仅 RGB, sup/q 各自独立采样) ──
    if augment and _HAS_AUG and model.training:
        s_rgb, _ = augment_rgb(s_rgb)
        q_rgb, _ = augment_rgb(q_rgb)

    # Encode support & query (use product injection for cross-domain hint)
    with torch.set_grad_enabled(model.training):
        s_emb = model.encode_with_product(s_rgb, s_cond, s_pid)
        q_emb = model.encode_with_product(q_rgb, q_cond, q_pid)

    # Build prototypes (trim_mean as default)
    if prototype_type == 'trim_mean':
        from common.protonet_loop import _trim_mean
        protos = []
        for c in range(n_way):
            mask = s_y == c
            if mask.any():
                protos.append(_trim_mean(s_emb[mask], trim_ratio=0.1))
            else:
                protos.append(s_emb.mean(0))
        protos = torch.stack(protos, 0)
    else:
        protos = torch.stack([s_emb[s_y == c].mean(0) for c in range(n_way)], 0)

    # Distance & logits (mahalanobis as default)
    if distance == 'cosine':
        logits = torch.mm(F.normalize(q_emb, dim=1), F.normalize(protos, dim=1).t())
    elif distance == 'mahalanobis' and mahal_head is not None:
        logits = mahal_head(q_emb, protos, s_y)
    else:
        # euclidean: -||q - p||^2
        q_exp = q_emb.unsqueeze(1)            # (Nq, 1, D)
        p_exp = protos.unsqueeze(0)           # (1, n_way, D)
        logits = -((q_exp - p_exp) ** 2).sum(-1)  # (Nq, n_way)

    loss = F.cross_entropy(logits, q_y)
    acc = (logits.argmax(-1) == q_y).float().mean().item()
    return loss, acc, n_way


def evaluate(model, mahal_head, sampler, device, n_episodes=100, distance='mahalanobis',
             prototype_type='trim_mean', augment=False):
    model.eval()
    accs, n_way_hist = [], []
    for _ in range(n_episodes):
        try:
            ep = sampler.sample(split='meta_test')
        except ValueError:
            continue
        if ep['support_rgb'].numel() == 0:
            continue
        _, acc, nw = proto_loss(model, mahal_head, ep, device, distance, prototype_type,
                                augment=augment)
        accs.append(acc); n_way_hist.append(nw)
    model.train()
    if not accs:
        return 0.0, 0.0, 0
    mean_acc = sum(accs) / len(accs)
    std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
    return mean_acc, std_acc, len(accs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mvtec-root', default=r'd:\元强化学习\代码\data\MVTEC')
    parser.add_argument('--gc10-root',  default=r'd:\元强化学习\代码\data\GC10-DET')
    parser.add_argument('--neu-root',   default=r'd:\元强化学习\代码\data\NEU-CLS\NEU-CLS')
    parser.add_argument('--carpaint-root', default=None,
                        help='车漆 CMPL 域根目录 (cardd/carparts/syndcar). '
                             '缺省或不存在时, 这 3 个 PID 自动跳过, 不影响其它源.')
    parser.add_argument('--aero-real-root', default=None,
                        help='航天真实数据根目录 (含 images/ + labels/). '
                             '缺省或不存在时, PID 20 自动跳过.')
    parser.add_argument('--mvtec-dropout-products', default=None,
                        help='逗号分隔, 加入 PID 21 的 MVTEC 产品名列表. '
                             '缺省使用 MVTEC_DROPOUT_PRODUCTS = [toothbrush]. '
                             '注: 数据目录与 MVTEC 共享, 自动按 MVTEC_ROOT 解析.')
    parser.add_argument('--use-mvtec-dropout', action='store_true',
                        help='启用 MVTEC dropout (PID 21) 作为 OOD 评估源. '
                             '默认关闭: 常规训练时不应让 PID 21 进入 meta_train pool.')
    parser.add_argument('--skip-products', default='', type=str,
                        help='逗号分隔的 PID 列表, 强制从 episode pool 排除. '
                             '常用于剔除某些 MVTEC 产品 (e.g. "toothbrush,zipper").')
    parser.add_argument('--pretrain-ckpt', default=r'd:\元强化学习\代码\outputs\multi_domain_smoke.pt',
                        help='Phase 0 预训练权重 (backbone/fiLM/cond_enc/prod_enc).')
    parser.add_argument('--out-dir', default=r'd:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_meta_proto')
    parser.add_argument('--n-way', type=int, default=5)
    parser.add_argument('--k-shot', type=int, default=5)
    parser.add_argument('--q-query', type=int, default=5)
    parser.add_argument('--iters', type=int, default=2000,
                        help='总迭代数 (默认 2000, 配合课程学习)')
    parser.add_argument('--curriculum-stages', type=str, default='0.6,0.8,1.0',
                        help='课程学习阶段分界点 (总 iters 的比例, 默认 0.6,0.8,1.0)')
    parser.add_argument('--curriculum-probs', type=str, default='0.2,0.4,0.8',
                        help='对应阶段的跨域采样概率 (逗号分隔, 默认 0.2,0.4,0.8)')
    parser.add_argument('--use-domain-reliability', action='store_true', default=True,
                        help='启用 Domain Reliability 加权 (默认开启, 降低不可靠域样本的 loss 权重)')
    parser.add_argument('--no-domain-reliability', dest='use_domain_reliability', action='store_false',
                        help='关闭 Domain Reliability')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--inner-lr', type=float, default=1e-2,
                        help='ANIL head inner-lr (若 unfreeze head)')
    parser.add_argument('--inner-steps', type=int, default=1)
    parser.add_argument('--distance', choices=['euclidean', 'cosine', 'mahalanobis'], default='mahalanobis',
                        help='距离度量: mahalanobis (默认, 考虑协方差)')
    parser.add_argument('--prototype', choices=['mean', 'trim_mean'], default='trim_mean',
                        help='原型构建: trim_mean (默认, 去异常值)')
    parser.add_argument('--unfreeze-backbone', action='store_true',
                        help='解冻 backbone (默认关闭, 保持预训练质量)')
    parser.add_argument('--unfreeze-prod', action='store_true',
                        help='训练 ProductEncoder 注入项 (cross-domain 微调)')
    parser.add_argument('--unfreeze-film', action='store_true')
    parser.add_argument('--use-anil-head', action='store_true',
                        help='在 prototype 上加可训 linear head 做 ANIL')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='episode 级数据增强 (5 种策略: hflip/hflip_rotate/'
                             'color_jitter/random_erasing/illumination_aug). '
                             'train 阶段 sup/q 各自独立采样策略. (默认开启)')
    parser.add_argument('--augment-eval', action='store_true', default=True,
                        help='评估阶段也启用增强. (默认开启)')
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir, 'train_meta')

    logger.info(f'args: {vars(args)}')
    if args.augment and not _HAS_AUG:
        logger.warning('--augment 启用但 episode_aug 模块不可用, 跳过.')

    # ── Sampler (split=meta_train, 训练时只在 train pool 抽) ──
    sampler, mvtec_ds, gc10_ds, neu_ds, carpaint_ds, aero_ds, mvtec_dropout_ds = build_episode_samplers(
        args.mvtec_root, args.gc10_root, args.neu_root,
        n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query,
        split='meta_train',
        carpaint_root=args.carpaint_root,
        aero_real_root=args.aero_real_root,
        mvtec_dropout_root=(args.mvtec_root if args.use_mvtec_dropout else None))
    logger.info(f'aero_ds: {aero_ds is not None}, mvtec_dropout_ds: {mvtec_dropout_ds is not None}')
    # 应用 --skip-products (强制从 episode pool 剔除)
    if args.skip_products:
        skip_pids = set()
        for tok in args.skip_products.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                skip_pids.add(int(tok))
            except ValueError:
                # 也接受 MVTEC 产品名 → PID
                from common.multi_domain_datasets import MVTEC_PRODUCT_TO_ID
                if tok in MVTEC_PRODUCT_TO_ID:
                    skip_pids.add(MVTEC_PRODUCT_TO_ID[tok])
        before = len(sampler.usable_products)
        sampler.usable_products = [p for p in sampler.usable_products
                                    if p not in skip_pids]
        sampler.product_to_source = {k: v for k, v in sampler.product_to_source.items()
                                      if k not in skip_pids}
        sampler.product_n_classes = {k: v for k, v in sampler.product_n_classes.items()
                                      if k not in skip_pids}
        logger.info(f'[skip-products] removed {before - len(sampler.usable_products)} '
                    f'PIDs {sorted(skip_pids)}; remaining={len(sampler.usable_products)}')
    logger.info(f'usable_products: {sampler.usable_products}')

    # ── Model ──
    model = MultiDomainPretrainModel(
        n_products=20, feat_dim=128, freeze_backbone=not args.unfreeze_backbone)
    if args.pretrain_ckpt and os.path.isfile(args.pretrain_ckpt):
        sd = torch.load(args.pretrain_ckpt, map_location='cpu')
        # 兼容两种格式: 直接 state_dict 或 {'model_state_dict': ...}
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']

        # ── 检测维度兼容性 (cond_enc / film 等可能因版本演进 shape 变化) ──
        size_mismatch = []
        msd = model.state_dict()
        for k in sd:
            if k in msd and hasattr(sd[k], 'shape') and hasattr(msd[k], 'shape') \
                    and sd[k].shape != msd[k].shape:
                size_mismatch.append((k, tuple(sd[k].shape), tuple(msd[k].shape)))

        if size_mismatch:
            logger.warning(f'pretrain ckpt has {len(size_mismatch)} shape mismatches; '
                           f'falling back to random init for those layers.')
            for k, old, new in size_mismatch[:5]:
                logger.warning(f'  {k}: ckpt={old} vs model={new}')
            # 自动去掉不匹配的 key, 让剩余层 (如 backbone) 仍可加载
            sd_filtered = {k: v for k, v in sd.items()
                           if not (k in msd and hasattr(v, 'shape')
                                   and hasattr(msd[k], 'shape')
                                   and v.shape != msd[k].shape)}
            missing, unexpected = model.load_state_dict(sd_filtered, strict=False)
            logger.info(f'loaded (partial) pretrain: '
                        f'loaded={len(sd_filtered)}/{len(sd)} '
                        f'missing={len(missing)} unexpected={len(unexpected)}')
        else:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(f'loaded pretrain: missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            logger.info(f'  missing keys (first 5): {missing[:5]}')
        if unexpected:
            logger.info(f'  unexpected keys (first 5): {unexpected[:5]}')
    else:
        logger.warning(f'pretrain not found at {args.pretrain_ckpt}; random init.')

    # 选择可训参数
    if args.unfreeze_backbone:
        for p in model.backbone_params(): p.requires_grad = True
    else:
        for p in model.backbone_params(): p.requires_grad = False
    for p in model.cond_enc.parameters(): p.requires_grad = True
    for p in model.film.parameters():
        p.requires_grad = args.unfreeze_film
    for p in model.prod_enc.parameters():
        p.requires_grad = args.unfreeze_prod
    for p in model._prod_proj.parameters():
        p.requires_grad = args.unfreeze_prod

    model = model.to(args.device)

    # ── Mahalanobis Head (默认启用) ──
    mahal_head = None
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.distance == 'mahalanobis':
        from common.protonet_loop import MahalanobisHead
        mahal_head = MahalanobisHead(feat_dim=128, ways=args.n_way).to(args.device)
        trainable_params.extend(mahal_head.parameters())
        logger.info(f'[MahalHead] 启用, feat_dim=128, ways={args.n_way}')

    # ── Domain Reliability (默认启用) ──
    # n_domains = 产品数量 (每个 product 是一个域)
    domain_reliability = None
    n_domains = len(sampler.usable_products) if sampler else 16
    if args.use_domain_reliability:
        from common.condition_encoder import DomainReliability
        domain_reliability = DomainReliability(feat_dim=128, n_domains=n_domains).to(args.device)
        trainable_params.extend(domain_reliability.parameters())
        logger.info(f'[DomainReliability] 启用, n_domains={n_domains} (产品数)')

    # ── ANIL head (可选) ──
    head = None
    if args.use_anil_head:
        head = torch.nn.Linear(128, args.n_way).to(args.device)
        torch.nn.init.normal_(head.weight, std=0.01)
        torch.nn.init.zeros_(head.bias)

    params = trainable_params
    if head is not None:
        params += list(head.parameters())
    n_params = sum(p.numel() for p in params if isinstance(p, torch.Tensor))
    logger.info(f'trainable params: {n_params:,}')

    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ── 课程学习参数解析 (stages 是比例, probs 是概率) ──
    curriculum_stages = tuple(float(x) for x in args.curriculum_stages.split(','))
    curriculum_probs = tuple(float(x) for x in args.curriculum_probs.split(','))

    # ── 训练主循环 ──
    best_val_acc = 0.0
    val_history = []   # [(iter, mean_acc, std_acc), ...] 仅保留稳定段
    cross_domain_count = 0
    total_count = 0
    t0 = time.time()
    for it in range(1, args.iters + 1):
        # 1) 抽 train episode
        try:
            ep = sampler.sample(split='meta_train')
        except ValueError as e:
            logger.warning(f'iter {it} skip: {e}')
            continue
        if ep['support_rgb'].numel() == 0:
            continue

        # 2) 课程学习: 判断是否跨域
        is_cross = is_cross_domain_episode(ep)
        cross_prob = get_cross_domain_prob(it, args.iters, curriculum_stages, curriculum_probs)

        # 随机决定是否强制跨域/同域
        r = torch.rand(1).item()
        force_cross = r < cross_prob  # 这轮要跨域采样

        # 如果当前 episode 类型不符合预期，尝试重新采样（最多3次）
        for _ in range(3):
            if (force_cross and is_cross) or (not force_cross and not is_cross):
                break
            try:
                ep = sampler.sample(split='meta_train')
                is_cross = is_cross_domain_episode(ep)
            except ValueError:
                break

        if is_cross:
            cross_domain_count += 1
        total_count += 1

        # 3) ProtoNet episode loss (含 Domain Reliability 加权)
        if domain_reliability is not None and is_cross:
            loss, acc, n_way, weights = weighted_proto_loss(
                model, mahal_head, ep, args.device,
                args.distance, args.prototype, args.augment,
                domain_reliability, is_cross, n_domains)
        else:
            loss, acc, n_way = proto_loss(model, mahal_head, ep, args.device,
                                           args.distance, args.prototype,
                                           augment=args.augment)

        # 3) 可选 ANIL (functional): 内层 adapt head on support, 外层 evaluate on query
        if head is not None and args.inner_steps > 0:
            # 重新算 emb (proto_loss 里的 s_emb/q_emb 没保留, 这里重新算一次)
            s_emb = model.encode_with_product(
                ep['support_rgb'].to(args.device),
                ep['support_cond'].to(args.device),
                ep['support_pid'].to(args.device))
            q_emb = model.encode_with_product(
                ep['query_rgb'].to(args.device),
                ep['query_cond'].to(args.device),
                ep['query_pid'].to(args.device))
            s_y = ep['support_y'].to(args.device)
            q_y = ep['query_y'].to(args.device)

            # ── functional 拷贝 head 参数 (类似 MAML 的 "fast weights") ──
            #    这样 inner_step 只更新临时副本, 不污染 outer head.
            fast_w = head.weight.detach().clone().requires_grad_(True)
            fast_b = head.bias.detach().clone().requires_grad_(True)

            # 一次性把所有 inner_steps 串起来 (PyTorch 自动构建二阶计算图)
            inner_loss = 0.0
            cur_w, cur_b = fast_w, fast_b
            for _ in range(args.inner_steps):
                logits_s = F.linear(s_emb, cur_w, cur_b)  # (N_s, n_way)
                # 构造 n_way-类 logits: 实际任务里每 episode 的 support 标签是 [0, n_way)
                # 但 ANIL head 输出维度固定为 args.n_way; 当 n_way < args.n_way (adaptive),
                # 只取前 n_way 列.
                if cur_w.size(0) > n_way:
                    logits_s_use = logits_s[:, :n_way]
                else:
                    logits_s_use = logits_s
                l_in = F.cross_entropy(logits_s_use, s_y)
                grad_w, grad_b = torch.autograd.grad(l_in, [cur_w, cur_b],
                                                     create_graph=True)
                cur_w = cur_w - args.inner_lr * grad_w
                cur_b = cur_b - args.inner_lr * grad_b
                inner_loss = inner_loss + l_in

            # 外层: 用 fast weights 在 query 上 CE, 反向传播到 model + head (二阶)
            logits_q = F.linear(q_emb, cur_w, cur_b)[:, :n_way]
            loss_q = F.cross_entropy(logits_q, q_y)
            loss = loss + loss_q

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)  # ProtoNet 经验值; 防 FiLM/bias 突然爆炸
        optimizer.step()

        if it % 20 == 0 or it == 1:
            logger.info(f'iter {it:4d} | loss={loss.item():.4f} acc={acc:.3f} '
                        f'n_way={n_way} cross={int(is_cross)} '
                        f'cross_prob={cross_prob:.2f} src={ep["source_name"]} pid={ep["product_id"]}')

        if it % args.eval_every == 0 or it == args.iters:
            mean_acc, std_acc, n_eps = evaluate(
                model, mahal_head, sampler, args.device,
                n_episodes=args.eval_episodes,
                distance=args.distance, prototype_type=args.prototype,
                augment=args.augment_eval)
            logger.info(f'[eval] iter {it} meta_test_acc={mean_acc:.3f}±{std_acc:.3f} '
                        f'({n_eps} eps) elapsed={time.time()-t0:.1f}s')
            val_history.append((it, mean_acc, std_acc))
            # save best
            if mean_acc > best_val_acc:
                best_val_acc = mean_acc
                ckpt = {
                    'iter': it,
                    'model': model.state_dict(),
                    'mahal_head': mahal_head.state_dict() if mahal_head is not None else None,
                    'domain_reliability': domain_reliability.state_dict() if domain_reliability is not None else None,
                    'head': head.state_dict() if head is not None else None,
                    'args': vars(args),
                    'best_val_acc': best_val_acc,
                }
                torch.save(ckpt, out_dir / 'best.pt')
                logger.info(f'  saved best.pt (acc={best_val_acc:.3f})')

    # final save
    torch.save({
        'iter': args.iters,
        'model': model.state_dict(),
        'mahal_head': mahal_head.state_dict() if mahal_head is not None else None,
        'domain_reliability': domain_reliability.state_dict() if domain_reliability is not None else None,
        'head': head.state_dict() if head is not None else None,
        'args': vars(args),
        'best_val_acc': best_val_acc,
        'cross_domain_ratio': cross_domain_count / max(total_count, 1),
    }, out_dir / 'final.pt')
    # 汇总稳定段统计
    val_history_full = [(it, m, s) for it, m, s in val_history]
    last3 = val_history[-3:] if len(val_history) >= 3 else val_history
    last5 = val_history[-5:] if len(val_history) >= 5 else val_history
    stable_last3_mean = sum(m for _, m, _ in last3) / len(last3)
    stable_last5_mean = sum(m for _, m, _ in last5) / len(last5)
    final_acc = val_history[-1][1] if val_history else None
    final_std = val_history[-1][2] if val_history else None

    summary = {
        'iters': args.iters,
        'best_val_acc': best_val_acc,
        'final_acc': final_acc,
        'final_std': final_std,
        'stable_last3_mean_acc': stable_last3_mean,
        'stable_last5_mean_acc': stable_last5_mean,
        'val_curve': [(it, round(m, 4), round(s, 4)) for it, m, s in val_history_full],
        'elapsed_sec': round(time.time() - t0, 1),
        'final_lr': args.lr,
        'n_way_max': args.n_way, 'k_shot': args.k_shot, 'q_query': args.q_query,
        'distance': args.distance, 'prototype': args.prototype,
        'unfreeze_backbone': args.unfreeze_backbone,
        'unfreeze_prod': args.unfreeze_prod,
        'unfreeze_film': args.unfreeze_film,
        'use_anil_head': args.use_anil_head,
        'augment': args.augment,
        'augment_eval': args.augment_eval,
        'use_domain_reliability': args.use_domain_reliability,
        'curriculum_stages': curriculum_stages,
        'curriculum_probs': curriculum_probs,
        'cross_domain_ratio': cross_domain_count / max(total_count, 1),
    }
    with open(out_dir / 'run_config.json', 'w') as fp:
        json.dump(summary, fp, indent=2)
    logger.info(f'DONE. best_val_acc={best_val_acc:.3f} | final={final_acc:.3f}±{final_std:.3f} '
                f'| last3_mean={stable_last3_mean:.3f} | last5_mean={stable_last5_mean:.3f} '
                f'| {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()