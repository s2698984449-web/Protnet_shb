# CRML: 基于现有架构的模块扩展方案

## 一、现有架构回顾

```
现有模块:
├── multi_domain_episode.py
│   └── MultiDomainEpisodeSampler.sample()
│       └── 从单一产品采样 support + query
│
├── multi_domain_model.py
│   └── MultiDomainPretrainModel
│       └── encode_with_product(img, cond, pid) → emb
│
└── train_meta.py
    └── proto_loss() → CE Loss
```

### 现有数据流

```python
episode = sampler.sample()  # 返回 dict
    ├── support_rgb, support_y, support_pid
    ├── query_rgb, query_y, query_pid
    └── source_name, product_id, n_way

loss = proto_loss(model, episode, device)
```

---

## 二、第一阶段改动：跨域 Episode 采样

### 2.1 改动位置

**文件**: `multi_domain_episode.py`

**新增类**: `CrossDomainEpisodeSampler` (继承 `MultiDomainEpisodeSampler`)

### 2.2 采样器扩展

```python
# multi_domain_episode.py 新增

class CrossDomainEpisodeSampler(MultiDomainEpisodeSampler):
    """
    跨域 Episode 采样器
    
    与父类区别:
    - 父类: support 和 query 从同一产品采样
    - 本类: support 从 Domain A, query 从 Domain B (可能不同域)
    
    返回新增字段:
    - is_cross_domain: bool  # 是否跨域
    - support_domain_id: tensor  # support 的域 ID
    - query_domain_id: tensor   # query 的域 ID
    """

    # ── 域 ID 映射 ──────────────────────────
    DOMAIN_ID_MAP = {
        'mvtec': 0,
        'gc10': 1,
        'neu': 2,
        'cardd': 3,
        'carparts': 4,
        'syndcar': 5,
        'aero_real': 6,
        'mvtec_dropout': 7,
    }
    
    # ── 域采样权重 ──────────────────────────
    DOMAIN_WEIGHTS = {
        0: 0.35,  # mvtec (产品多)
        1: 0.25,  # gc10
        2: 0.20,  # neu
        3: 0.05,  # cardd
        4: 0.05,  # carparts
        5: 0.05,  # syndcar
        6: 0.03,  # aero_real
        7: 0.02,  # mvtec_dropout
    }
    
    # ── 跨域时的域对权重 ────────────────────
    # 越难的对偶权重越高
    CROSS_DOMAIN_PAIR_WEIGHTS = {
        # (source_domain, target_domain): weight
        (0, 1): 0.15,  # mvtec ↔ gc10
        (0, 2): 0.15,  # mvtec ↔ neu
        (1, 2): 0.10,  # gc10 ↔ neu
        (0, 3): 0.08,  # mvtec ↔ car paint
        (0, 6): 0.05,  # mvtec ↔ aero
        (1, 6): 0.05,  # gc10 ↔ aero
        (2, 6): 0.05,  # neu ↔ aero
        # ... 可继续扩展
    }

    def __init__(self, base_sampler, 
                 cross_domain_prob: float = 0.5,
                 domain_weights: dict = None,
                 pair_weights: dict = None):
        """
        Args:
            base_sampler: 父类采样器 (MultiDomainEpisodeSampler)
            cross_domain_prob: 跨域采样概率 (0.5 = 50%)
            domain_weights: 各域出现概率
            pair_weights: 跨域时不同域对的采样权重
        """
        # 复用父类所有属性
        self.__dict__.update(base_sampler.__dict__)
        self.cross_domain_prob = cross_domain_prob
        self.domain_weights = domain_weights or self.DOMAIN_WEIGHTS
        self.pair_weights = pair_weights or self.CROSS_DOMAIN_PAIR_WEIGHTS

    def _sample_domain(self) -> int:
        """按权重采样一个域"""
        domains = list(self.domain_weights.keys())
        weights = list(self.domain_weights.values())
        return random.choices(domains, weights=weights, k=1)[0]

    def _sample_domain_excluding(self, exclude: int, 
                                  pair_weights: dict = None) -> int:
        """采样一个与 exclude 不同的域"""
        # 收集所有可用的目标域
        candidates = []
        for (src, tgt), w in (pair_weights or self.pair_weights).items():
            if src == exclude:
                candidates.append((tgt, w))
            elif tgt == exclude:
                candidates.append((src, w))
        
        if not candidates:
            # fallback: 随机选一个不同的域
            all_domains = set(self.domain_weights.keys()) - {exclude}
            return random.choice(list(all_domains))
        
        targets, weights = zip(*candidates)
        return random.choices(targets, weights=weights, k=1)[0]

    def _get_products_in_domain(self, domain_id: int) -> List[int]:
        """获取某域内的所有产品 ID"""
        domain_name = {v: k for k, v in self.DOMAIN_ID_MAP.items()}[domain_id]
        products = []
        for pid, src in self.product_to_source.items():
            if src[0] == domain_name:
                products.append(pid)
        return products if products else [random.choice(self.usable_products)]
```

### 2.3 核心采样方法

```python
def sample_cross_domain(self, split: str = 'meta_train') -> dict:
    """
    采样跨域 Episode
    
    返回值相比父类新增:
    - is_cross_domain: bool
    - support_domain_id: (N*K,) tensor
    - query_domain_id: (N*Q,) tensor
    """
    # ── Step 1: 决定是否跨域 ───────────────
    is_cross = random.random() < self.cross_domain_prob
    
    # ── Step 2: 采样 support 域 ───────────
    support_domain = self._sample_domain()
    
    # ── Step 3: 采样 query 域 ──────────────
    if is_cross:
        query_domain = self._sample_domain_excluding(support_domain)
    else:
        query_domain = support_domain
    
    # ── Step 4: 在各域内选产品 ─────────────
    support_products = self._get_products_in_domain(support_domain)
    query_products = self._get_products_in_domain(query_domain)
    
    support_product = random.choice(support_products)
    query_product = random.choice(query_products)
    
    # ── Step 5: 解析数据集 ─────────────────
    _, support_ds = self.product_to_source[support_product]
    _, query_ds = self.product_to_source[query_product]
    
    # ── Step 6: 选 defect classes ─────────
    # 注意: 跨域时只要求 support/query 使用相同的 n_way
    # 不要求实际 defect 类型相同 (这正是 CRML 的核心!)
    support_n_classes = support_ds.num_defects(support_product)
    query_n_classes = query_ds.num_defects(query_product)
    
    eff_n_way = min(self.n_way, support_n_classes, query_n_classes)
    
    # 从 support 域选 classes
    support_classes = random.sample(range(support_n_classes), eff_n_way)
    # 从 query 域选对应的 classes (按索引映射)
    # query_classes[i] 对应 support_classes[i] 的位置
    # 这样 episode 内 label=0 对应划痕, label=1 对应裂纹, 等等
    query_classes = random.sample(range(query_n_classes), eff_n_way)
    
    # ── Step 7: 采样样本 ───────────────────
    s_imgs, s_y, s_pid, s_cond = [], [], [], []
    q_imgs, q_y, q_pid, q_cond = [], [], [], []
    
    for ep_label, (s_class, q_class) in enumerate(zip(support_classes, query_classes)):
        # Support 采样
        s_paths = support_ds.sample_paths(support_product, s_class)
        self._sample_and_load(s_paths, self.k_shot, 
                             s_imgs, s_y, s_pid, s_cond,
                             support_product, support_domain, ep_label)
        
        # Query 采样
        q_paths = query_ds.sample_paths(query_product, q_class)
        self._sample_and_load(q_paths, self.q_query,
                             q_imgs, q_y, q_pid, q_cond,
                             query_product, query_domain, ep_label)
    
    # ── Step 8: 构建返回 ───────────────────
    result = {
        # 原有字段
        'support_rgb': torch.stack(s_imgs),
        'support_y': torch.tensor(s_y, dtype=torch.long),
        'support_pid': torch.tensor(s_pid, dtype=torch.long),
        'support_cond': torch.stack(s_cond),
        'query_rgb': torch.stack(q_imgs),
        'query_y': torch.tensor(q_y, dtype=torch.long),
        'query_pid': torch.tensor(q_pid, dtype=torch.long),
        'query_cond': torch.stack(q_cond),
        'source_name': f'{support_domain}→{query_domain}',
        'product_id': support_product,  # 主要产品
        'n_way': eff_n_way,
        
        # 新增字段
        'is_cross_domain': is_cross,
        'support_domain_id': torch.tensor(
            [support_domain] * len(s_imgs), dtype=torch.long),
        'query_domain_id': torch.tensor(
            [query_domain] * len(q_imgs), dtype=torch.long),
    }
    
    return result

def _sample_and_load(self, paths, n_samples,
                    imgs_list, y_list, pid_list, cond_list,
                    product_id, domain_id, ep_label):
    """辅助: 从路径列表采样并加载"""
    rng = random.Random()
    need = n_samples
    if len(paths) < need:
        picks = [rng.choice(paths) for _ in range(need)]
    else:
        picks = rng.sample(paths, need)
    
    for p in picks:
        img, cond = _load_image(p, self._transform, rng)
        imgs_list.append(img)
        y_list.append(ep_label)
        pid_list.append(product_id)
        cond_list.append(cond)
```

---

## 三、第一阶段改动：Relation Consistency Loss

### 3.1 改动位置

**文件**: `train_meta.py`

**新增函数**: `relation_consistency_loss()`

### 3.2 Loss 实现

```python
# train_meta.py 新增

def relation_consistency_loss(model, episode, device,
                              distance: str = 'euclidean',
                              mode: str = 'intra_variance',
                              prototype_type: str = 'mean') -> torch.Tensor:
    """
    Relation Consistency Loss
    
    核心思想: 同类 query 的关系分布应该集中 (低方差)
    
    Args:
        model: MultiDomainPretrainModel
        episode: cross-domain episode dict
        device: torch device
        distance: 'euclidean' | 'cosine'
        mode: 
            - 'intra_variance': 同类 query 的关系方差
            - 'cross_label': 不同 label 关系的分离度
        prototype_type: 'mean' | 'trim_mean'
    
    Returns:
        loss_rc: scalar tensor
    """
    s_rgb = episode['support_rgb'].to(device)
    q_rgb = episode['query_rgb'].to(device)
    s_y = episode['support_y'].to(device)
    q_y = episode['query_y'].to(device)
    s_pid = episode['support_pid'].to(device)
    q_pid = episode['query_pid'].to(device)
    s_cond = episode['support_cond'].to(device)
    q_cond = episode['query_cond'].to(device)
    n_way = episode['n_way']
    
    # Encode
    with torch.set_grad_enabled(model.training):
        s_emb = model.encode_with_product(s_rgb, s_cond, s_pid)
        q_emb = model.encode_with_product(q_rgb, q_cond, q_pid)
    
    # Compute prototypes (from support only)
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
        protos = torch.stack([s_emb[s_y == c].mean(0) 
                              for c in range(n_way)], 0)
    
    # Compute relations: (N_q, N_way)
    if distance == 'cosine':
        q_emb_n = F.normalize(q_emb, dim=1)
        protos_n = F.normalize(protos, dim=1)
        relations = torch.mm(q_emb_n, protos_n.t())
    else:
        q_exp = q_emb.unsqueeze(1)
        p_exp = protos.unsqueeze(0)
        relations = -((q_exp - p_exp) ** 2).sum(-1)
    
    # ── Intra-Variance Mode ─────────────────
    if mode == 'intra_variance':
        """
        同类 query 的关系分布方差应该小
        
        例: 所有 label=0 的 query，到 proto[0] 的距离应该集中
        """
        loss = 0.0
        n_labels = 0
        
        for c in range(n_way):
            mask = (q_y == c)
            if mask.sum() < 2:
                continue
            
            # label=c 的 query 对各 proto 的关系
            rel_c = relations[mask]  # (N_c, N_way)
            
            # 每列 (每个 proto) 的方差
            variance_per_proto = rel_c.var(dim=0)  # (N_way,)
            
            # 加权平均方差
            loss += variance_per_proto.mean()
            n_labels += 1
        
        if n_labels == 0:
            return torch.tensor(0.0, device=device)
        
        return loss / n_labels
    
    # ── Cross-Label Separation Mode ──────────
    elif mode == 'cross_label':
        """
        不同 label 的关系应该分离
        
        例: label=0 的 query，对 proto[0] 的关系应该 >> 对 proto[1] 的关系
        """
        loss = 0.0
        n_pairs = 0
        
        for c_i in range(n_way):
            for c_j in range(n_way):
                if c_i == c_j:
                    continue
                
                mask_i = (q_y == c_i)
                if mask_i.sum() < 1:
                    continue
                
                # label=i 的 query，对 proto[i] 的平均关系
                r_ii = relations[mask_i][:, c_i].mean()
                # label=i 的 query，对 proto[j] 的平均关系
                r_ij = relations[mask_i][:, c_j].mean()
                
                # 应该有 margin: r_ii - r_ij > margin
                margin = 1.0
                loss += F.relu(margin - (r_ii - r_ij))
                n_pairs += 1
        
        if n_pairs == 0:
            return torch.tensor(0.0, device=device)
        
        return loss / n_pairs
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

---

## 四、训练脚本改动

### 4.1 命令行参数

```python
# train_meta.py main() 中新增

# ── 跨域采样参数 ──────────────────────────────
parser.add_argument('--cross-domain-prob', type=float, default=0.5,
                   help='跨域 episode 采样概率 (0.5 = 50%%)')
parser.add_argument('--domain-weights', type=str, default=None,
                   help='JSON 格式的域权重, 如 \'{"0":0.35,"1":0.25}\'')
parser.add_argument('--cross-pair-weights', type=str, default=None,
                   help='JSON 格式的跨域对权重')

# ── Relation Consistency Loss 参数 ─────────────
parser.add_argument('--relation-weight', type=float, default=0.5,
                   help='Relation Consistency Loss 权重')
parser.add_argument('--relation-mode', choices=['intra_variance', 'cross_label'],
                   default='intra_variance',
                   help='RC Loss 计算模式')
parser.add_argument('--rc-weight-cross', type=float, default=1.5,
                   help='跨域 episode 的 RC Loss 权重倍数')
parser.add_argument('--rc-weight-same', type=float, default=0.5,
                   help='同域 episode 的 RC Loss 权重倍数')
```

### 4.2 训练循环改动

```python
def main():
    # ... 现有代码省略 ...
    
    # ── 采样器: 根据参数选择 ────────────────────
    if args.cross_domain_prob > 0:
        # 使用跨域采样器
        from common.multi_domain_episode import CrossDomainEpisodeSampler
        
        base_sampler, mvtec_ds, gc10_ds, neu_ds, carpaint_ds, aero_ds, dropout_ds = \
            build_episode_samplers(...)
        
        sampler = CrossDomainEpisodeSampler(
            base_sampler=base_sampler,
            cross_domain_prob=args.cross_domain_prob,
        )
        print(f'[CrossDomainEpisodeSampler] cross_domain_prob={args.cross_domain_prob}')
    else:
        # 使用原有采样器
        sampler, _, _, _, _, _, _ = build_episode_samplers(...)
    
    # ── 训练循环 ────────────────────────────────
    for it in range(1, args.iters + 1):
        # 采样 episode
        try:
            ep = sampler.sample()
        except ValueError as e:
            logger.warning(f'iter {it} skip: {e}')
            continue
        
        # ── ProtoNet Loss ─────────────────────────
        loss_ce, acc, n_way = proto_loss(
            model, ep, args.device,
            args.distance, args.prototype,
            augment=args.augment
        )
        
        # ── Relation Consistency Loss ─────────────
        if args.relation_weight > 0:
            loss_rc = relation_consistency_loss(
                model, ep, args.device,
                distance=args.distance,
                mode=args.relation_mode,
                prototype_type=args.prototype
            )
            
            # 根据跨域/同域调整权重
            is_cross = ep.get('is_cross_domain', False)
            λ = args.relation_weight * (
                args.rc_weight_cross if is_cross else args.rc_weight_same
            )
            
            total_loss = loss_ce + λ * loss_rc
            rc_loss_val = loss_rc.item()
        else:
            total_loss = loss_ce
            rc_loss_val = 0.0
        
        # ── 反向传播 ──────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        # ── 日志 ──────────────────────────────────
        if it % 20 == 0 or it == 1:
            cross_flag = '(cross)' if ep.get('is_cross_domain', False) else '(same)'
            logger.info(
                f'iter {it:4d} | loss={total_loss.item():.4f} '
                f'(ce={loss_ce.item():.4f}, rc={rc_loss_val:.4f}) '
                f'acc={acc:.3f} n_way={n_way} {cross_flag}'
            )
```

---

## 五、架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      train_meta.py                               │
│                                                                  │
│  ┌─────────────┐     ┌──────────────────────┐                  │
│  │ Sampler     │     │ ProtoNet Loss         │                  │
│  │             │     │ L_ce = CE(r, labels)  │                  │
│  │ 同域/跨域    │────▶│                      │                  │
│  │ 采样        │     └──────────────────────┘                  │
│  │             │                 │                              │
│  │             │     ┌──────────────────────┐                  │
│  │ episode = { │────▶│ RC Loss (新增)        │                  │
│  │   is_cross, │     │ L_rc = Var(r[label]) │                  │
│  │   s_domain, │     │ × λ_cross/λ_same    │                  │
│  │   q_domain  │     └──────────────────────┘                  │
│  │ }           │                 │                              │
│  └─────────────┘                 │                              │
│                                  ▼                              │
│                    ┌──────────────────────┐                  │
│                    │ Total Loss            │                  │
│                    │ L = L_ce + λ × L_rc   │                  │
│                    └──────────────────────┘                  │
│                                  │                              │
│                                  ▼                              │
│                    ┌──────────────────────┐                  │
│                    │ Backward + Update     │                  │
│                    │ encoder(θ) ← θ - lr·∇ │                  │
│                    └──────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               CrossDomainEpisodeSampler                          │
│                                                                  │
│  输入:  MultiDomainEpisodeSampler (父类)                         │
│  参数:  cross_domain_prob, domain_weights, pair_weights          │
│                                                                  │
│  sample_cross_domain():                                          │
│      │                                                          │
│      ├─ 1. 按 domain_weights 采样 support_domain                 │
│      │                                                          │
│      ├─ 2. 按 cross_domain_prob 决定是否跨域                      │
│      │                                                          │
│      ├─ 3. 若跨域: 按 pair_weights 采样 query_domain ≠ support   │
│      │      若同域: query_domain = support_domain                │
│      │                                                          │
│      ├─ 4. 在各域内选产品                                        │
│      │                                                          │
│      ├─ 5. 选相同的 n_way (但实际 defect 类型可能不同)           │
│      │                                                          │
│      └─ 6. 返回 episode + is_cross_domain 标志                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、参数配置建议

```bash
# 阶段一实验配置

# Baseline (无跨域)
python scripts/train_meta.py \
    --n-way 4 --k-shot 5 --q-query 5 \
    --iters 2000 --eval-every 200 \
    --out-dir checkpoints/phase1_baseline

# + 跨域采样 (50%)
python scripts/train_meta.py \
    --n-way 4 --k-shot 5 --q-query 5 \
    --iters 2000 --eval-every 200 \
    --cross-domain-prob 0.5 \
    --out-dir checkpoints/phase1_cross_domain

# + RC Loss (λ=0.5)
python scripts/train_meta.py \
    --n-way 4 --k-shot 5 --q-query 5 \
    --iters 2000 --eval-every 200 \
    --cross-domain-prob 0.5 \
    --relation-weight 0.5 \
    --relation-mode intra_variance \
    --rc-weight-cross 1.5 \
    --rc-weight-same 0.5 \
    --out-dir checkpoints/phase1_crml
```

---

## 七、关键设计决策

### 7.1 为什么跨域时不要求 defect 类型相同？

```
MVTEC cable: [断路, 弯曲, 锡球, 扎伤, 错位]
GC10 steel:  [冲孔, 焊缝, 月牙弯, 水斑, 油斑]

如果我们强制 "断路" ↔ "冲孔" 对应:
  - 需要人工定义跨域标签映射 (不可行)
  - 不同人对"语义相似"的定义不同

CRML 的做法:
  - 在 episode 内，标签就是 [0, 1, 2, 3]
  - 模型学习: "如何从 support 建立判别关系"
  - 迁移时: 只要新域也有"划痕-like"的缺陷，就能泛化
```

### 7.2 为什么用方差而不是对比学习？

```
对比学习需要:
  - 正样本对 (同 label 的特征应该接近)
  - 负样本对 (不同 label 的特征应该远离)

问题:
  - 跨域时正样本的特征空间本身不对齐
  - 强制拉近会损害有用的 domain-specific 信息

方差的好处:
  - 不要求特征接近，只要求"关系分布"集中
  - 更宽松，更安全
  - 计算简单
```

### 7.3 权重调整的直觉

```
同域 episode:  关系本来就一致 → λ 小 (0.5)
跨域 episode: 关系不一致 → λ 大 (1.5)

原因:
  - 如果所有 episode 都用同样的 λ，模型会:
    1. 主要优化容易的同域 episode
    2. 忽略难优化的跨域 episode
    
  - 放大跨域权重，强迫模型关注跨域一致性
```
