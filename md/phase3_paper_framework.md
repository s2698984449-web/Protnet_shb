# Phase 3 — 论文级 Framework 设计

> **MetaProto: Industrial Meta-Learning Framework for Few-Shot Defect Detection**
>
> 通用工业小样本缺陷检测元学习框架
>
> 分析日期: 2026-07-29

---

## 1. 框架总览 (Framework Overview)

### 1.1 名称建议

```
MetaProto: A Task-Conditioned Meta-Learning Framework for
           Industrial Few-Shot Cross-Domain Defect Detection
```

或者按你给的格式：
```
XXX Meta Learning Framework for Industrial Few-shot Defect Detection
```

**论文定位**：跨产品/跨材料的少样本工业缺陷检测通用框架

### 1.2 核心方法论

```
┌─────────────────────────────────────────────────────────────────┐
│                      MetaProto Framework                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Module 1        Module 2        Module 3        Module 4      │
│  Universal       Task            Prototype       Domain        │
│  Feature         Generator       Adaptation      Generalizer   │
│  Encoder          ↓               ↓               ↓            │
│  ──────        ──────────      ──────────      ──────────      │
│  输入域迁移     构造变化        K-shot 适配     跨域泛化       │
│                                                                 │
│              ┌──────────────────────────────────┐              │
│              │      Phase 0: Pretraining        │              │
│              │      Phase 1: Meta-Training      │              │
│              │      Phase 2: Few-Shot Deploy    │              │
│              └──────────────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 方法章节骨架 (Methods Section Skeleton)

### 2.1 Problem Formulation

```
Problem Statement:
  Given K labeled samples per defect class from a NEW industrial
  product/material (target domain), classify defects accurately.

Settings:
  · Source domains: D_train = {D_1, ..., D_M}    (M ≥ 2)
  · Target domain:  D_test (held out, never seen)
  · N-way K-shot:   K ∈ {1, 3, 5}
  · Generalization: D_test has NO overlapping specimens with D_train
```

**与现有工作的关键区别**:

| 现有方法 | MetaProto |
|---------|-----------|
| 单数据集训练 | 多域跨产线训练 |
| 固定 backbone | 可插拔 backbone + CBAM |
| 单一适配策略 | 4 种可切换适配策略 |
| 无任务条件化 | Task embedding 条件化融合 |

### 2.2 Industrial Meta Task

**Definition 1 (Industrial Meta Task)**:

```
T_i = (C_i, M_i, L_i, D_i)
  C_i: Component type     (部件)
  M_i: Material           (材料)
  L_i: Lighting condition (光照)
  D_i: Defect class set   (缺陷类别)
```

**每个任务对应 N-way K-shot episode**:

```
Episode_i = {
  Support_i:  {(x_j, y_j)}_{j=1}^{NK},   y_j ∈ {0,...,N-1}
  Query_i:    {(x_j, y_j)}_{j=NK+1}^{NK+NQ}
}
```

### 2.3 Framework: Four Modules

---

## Module 1 — Universal Industrial Feature Encoder (UIFE)

**Goal**: 学习跨产品、跨材料不变的视觉特征

```python
class UIFE(nn.Module):
    """
    Universal Industrial Feature Encoder.
    Architecture: Backbone (frozen/partial/full) + optional CBAM + GAP + Linear
    """
    def __init__(self,
                 backbone='yolo26n',   # 可插拔: yolo26/resnet50/swin_t
                 embed_dim=128,
                 use_cbam=True,
                 freeze_backbone=True):
        self.backbone = build_backbone(backbone)
        self.cbam = CBAM(channels) if use_cbam else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, embed_dim)

    def encode(self, x):
        feat = self.backbone(x)             # (B, C, H, W)
        feat = self.cbam(feat) if self.cbam else feat
        feat = self.pool(feat).flatten(1)   # (B, C)
        return self.proj(feat)              # (B, embed_dim)
```

**设计要点**:
1. **两阶段训练策略**: Phase 0 预训练 → Phase 1 元学习
2. **可插拔 backbone**: YOLO26 (工业实时性) / ResNet (通用) / ConvNeXt (强表征)
3. **可选 CBAM**: 实验证明 +5pp (exp10)
4. **冻结/解冻权衡**: 冻结节省显存 + 抗过拟合；解冻 +30pp 增益

**论文图示建议**:

```
Input Image ──► Backbone ──► [CBAM] ──► GAP ──► Linear ──► Embedding z
                (frozen/        ↓
                partial)      Channel × Spatial Attention
                              (option)
```

---

## Module 2 — Industrial Meta Task Generator (IMTG)

**Goal**: 模拟真实工业变化，构造多样化的元训练任务

```python
class IMTG:
    """
    Industrial Meta Task Generator.
    Constructs N-way K-shot episodes from multiple source domains.
    """
    def __init__(self,
                 domain_partition_strategy='metadata_based',
                 episode_aug_policies=('illumination', 'flip', 'rotate')):
        self.partition = DomainPartitionStrategy(strategy=domain_partition_strategy)
        self.aug_policies = episode_aug_policies

    def sample_episode(self, source_domains, n_way, k_shot, n_query):
        """
        Sample an N-way K-shot episode from source_domains.
        """
        # 1. Randomly pick a source domain
        domain = random.choice(source_domains)

        # 2. Sample n_way classes from domain
        classes = self._sample_classes(domain, n_way)

        # 3. Sample k_shot + n_query per class
        support, query = [], []
        for c in classes:
            samples = self._sample_n(domain, c, k_shot + n_query)
            support.extend(samples[:k_shot])
            query.extend(samples[k_shot:])

        # 4. Apply episode-level augmentation (same to support/query)
        support, query = self._apply_episode_aug(support, query)

        return Episode(support=support, query=query, domain=domain)
```

**域变化模拟**:

```
Domain Shift Dimensions:
  ① Component shift  → 不同 ROI 几何
  ② Material shift   → 不同表面反射/纹理
  ③ Lighting shift   → 不同亮度/色温 (augmentation)
  ④ Defect shift     → 不同类内方差
```

**Episode Augmentation Policies** (支持/查询同策略):
- Flip / Rotate / Crop
- Color Jitter (brightness/contrast/saturation)
- **Illumination Aug** (gamma correction + specular highlight) — 工业特化
- Random Erasing (模拟遮挡)

**论文图示建议**:

```
Source domains: D_1, D_2, D_3, ...
            ↓
   Domain Partition Strategy
   (metadata-based / industrial-stratified)
            ↓
   Episode Sampling (N-way K-shot)
            ↓
   Episode Augmentation (5 policies)
            ↓
   Episode: (Support, Query, DomainMeta)
```

---

## Module 3 — Prototype-based Adaptation Module (PAM)

**Goal**: 利用 K 个 support 样本快速构建类别原型

```python
class PAM(nn.Module):
    """
    Prototype-based Adaptation Module.
    Aggregates support embeddings into class prototypes + distance metrics.
    """
    def __init__(self,
                 prototype_type='trim_mean',   # mean / trim_mean / mean_shift
                 distance='euclidean',         # euclidean / cosine / mahalanobis
                 feat_dim=128,
                 ways=5):
        self.prototype_type = prototype_type
        self.distance = distance
        self.mahal_head = MahalanobisHead(feat_dim, ways) if distance == 'mahalanobis' else None

    def build_prototypes(self, support_emb, support_labels):
        """
        support_emb: (N_s, D)
        support_labels: (N_s,)
        Returns:
            prototypes: (N_way, D) or List[(K_c, D)]
        """
        prototypes = []
        for c in range(self.ways):
            mask = support_labels == c
            emb_c = support_emb[mask]
            if self.prototype_type == 'mean':
                p = emb_c.mean(dim=0)
            elif self.prototype_type == 'trim_mean':
                p = self._trim_mean(emb_c, trim_ratio=0.1)
            elif self.prototype_type == 'mean_shift':
                p = self._mean_shift(emb_c)
            prototypes.append(p)
        return prototypes

    def compute_logits(self, query_emb, prototypes):
        """
        Returns logits for each query embedding.
        """
        if self.distance == 'euclidean':
            return self._euclidean_logits(query_emb, prototypes)
        elif self.distance == 'cosine':
            return self._cosine_logits(query_emb, prototypes)
        elif self.distance == 'mahalanobis':
            return self.mahal_head(query_emb, prototypes)
```

**设计要点**:

1. **3 种原型策略**:
   - **Mean**: 简单平均 — 通用 baseline
   - **Trim Mean**: 去除 10% 极值 — 抗异常值
   - **Mean Shift**: 多聚类中心 — 表征多模态类

2. **3 种距离度量**:
   - **Euclidean**: 基线
   - **Cosine**: 关注方向，不关注幅值
   - **Mahalanobis**: 类内方差加权 — 几何上最优

3. **消融结果** (来自项目实验):
   ```
   euclidean: 85.2%
   + trim_mean: +1.5pp
   + mahalanobis: +2.1pp
   ```

**论文图示建议**:

```
Support Embeddings  z_s ∈ R^{NK × D}
            ↓
   Prototype Aggregation
   (mean / trim_mean / mean_shift)
            ↓
   Class Prototypes  P ∈ R^{N × D}
            ↓                          
   Distance Metric  ──────►  Logits q ∈ R^{NQ × N}
   (euclidean / cosine / mahal)
            ↓
   Class Prediction
```

---

## Module 4 — Domain Generalization Module (DGM)

**Goal**: 提升跨工件泛化能力

```python
class DGM(nn.Module):
    """
    Domain Generalization Module.
    Combines: (1) multi-domain meta-training, (2) task conditioning.
    """
    def __init__(self, descriptor_dim=30, embed_dim=128, out_dim=64):
        # Task embedding: metadata → τ
        self.task_encoder = TaskEncoder(descriptor_dim, out_dim)
        # Feature conditioning
        self.cond_proj = nn.Linear(embed_dim + out_dim, embed_dim)

    def condition_features(self, z, task_metadata):
        """
        z: (B, embed_dim)
        task_metadata: (B, descriptor_dim)
        Returns: task-conditioned feature
        """
        τ = self.task_encoder(task_metadata)
        z_cond = torch.cat([z, τ], dim=-1)
        return self.cond_proj(z_cond)
```

**Generalization 三大机制**:

| 机制 | 实现 | 作用 |
|------|------|------|
| **Multi-domain meta-training** | 多域混合 episode | 防止过拟合单一域 |
| **Task conditioning** | TaskEncoder 注入 τ | 让特征感知当前任务 |
| **Cross-domain aug** | Illumination/Jitter | 鲁棒性提升 |

**关键创新点**: Task Embedding τ

```
τ = TaskEncoder(material ⊕ process ⊕ history ⊕ component_id)

作用:
  · 让 backbone 知道 "当前是新工件/新材料"
  · 可用于融合权重、特征调制、原型修正
  · 形成 跨任务知识 注入通道
```

**论文图示建议**:

```
                  ┌─────────────────────────────┐
                  │      Task Embedding τ       │
                  │   (Material/Process/Hist)   │
                  └──────────┬──────────────────┘
                             ↓
   Feature z ────► Concatenate ──► Linear ──► z_conditioned
                             ↓
                     Prototype / Logits
```

---

## 3. 训练流程 (Training Pipeline)

### Phase 0: Backbone Pretraining

```
输入: 多源工业数据 (ImageNet-pretrained 或自监督)
过程:
  · 多任务头 (per-source) 同时训练
  · SupCon + CE loss
  · 目标: 学到 通用工业视觉特征
输出: 预训练 backbone weights
```

### Phase 1: Meta-Training

```
输入: M 个 source domain 的元数据
过程:
  · 每个 iteration:
    1. IMTG.sample_episode()   → Episode
    2. UIFE.encode(support)    → z_s
    3. UIFE.encode(query)      → z_q
    4. PAM.build_prototypes(z_s, y_s) → P
    5. PAM.compute_logits(z_q, P) → logits
    6. loss = CE(logits, y_q)
    7. 反向传播 + 优化
输出: MetaProto meta-model
```

### Phase 2: Few-Shot Deployment

```
输入: 新产线 K 个标注样本 (per defect class)
过程:
  1. UIFE.encode(support)    → z_s
  2. PAM.build_prototypes(z_s) → P
  3. UIFE.encode(query)      → z_q
  4. PAM.compute_logits(z_q, P) → 预测
输出: 预测类别
```

---

## 4. 算法伪代码 (Algorithm)

```python
# MetaProto Algorithm
def train_metaproto(source_domains, n_way=2, k_shot=5, n_query=15,
                    iters=500, adapt_strategy='protonet'):
    """
    MetaProto Training Algorithm.
    """
    # Initialize four modules
    encoder = UIFE(backbone='yolo26n', embed_dim=128, use_cbam=True)
    task_gen = IMTG(strategy='metadata_based')
    pam = PAM(prototype_type='trim_mean', distance='euclidean')
    dgm = DGM(descriptor_dim=30)

    for iteration in range(iters):
        # Module 2: Sample episode
        episode = task_gen.sample_episode(source_domains, n_way, k_shot, n_query)

        # Module 1: Encode
        z_s = encoder.encode(episode.support_rgb)  # (NK, 128)
        z_q = encoder.encode(episode.query_rgb)    # (NQ, 128)

        # Module 4: Task conditioning (optional)
        if use_task_conditioning:
            z_s = dgm.condition_features(z_s, episode.support_meta)
            z_q = dgm.condition_features(z_q, episode.query_meta)

        # Module 3: Prototype + Logits
        prototypes = pam.build_prototypes(z_s, episode.support_labels)
        logits = pam.compute_logits(z_q, prototypes)

        # Loss
        loss = F.cross_entropy(logits, episode.query_labels)
        loss.backward()
        optimizer.step()
```

---

## 5. 论文 Method 章节结构 (Draft Outline)

```
3. Method

3.1 Problem Formulation
    · Industrial few-shot defect detection definition
    · Source / Target domain notation
    · N-way K-shot protocol

3.2 Industrial Meta Task
    · Task = (Component, Material, Lighting, Defect)
    · Domain shift taxonomy

3.3 MetaProto Framework
    3.3.1 Module 1: Universal Industrial Feature Encoder
          · Backbone, CBAM, Projection
          · Two-stage training (pretrain + meta-train)
    3.3.2 Module 2: Industrial Meta Task Generator
          · Domain partition strategy
          · Episode sampling
          · Episode-level augmentation (5 policies)
    3.3.3 Module 3: Prototype-based Adaptation Module
          · Prototype aggregation (mean / trim_mean / mean_shift)
          · Distance metric (euclidean / cosine / mahalanobis)
    3.3.4 Module 4: Domain Generalization Module
          · Multi-domain meta-training
          · Task embedding conditioning
          · Cross-domain regularization

3.4 Training Procedure
    · Phase 0: pretraining
    · Phase 1: meta-training algorithm
    · Phase 2: few-shot deployment

3.5 Inference
    · Build prototypes from K-shot
    · Match query to nearest prototype
```

---

## 6. 与现有方法的对比 (Comparison Table)

| 方法 | 多域泛化 | 任务条件化 | 适配策略 | Backbone灵活性 |
|------|---------|-----------|----------|---------------|
| ProtoNet [Snell 2017] | ❌ | ❌ | mean | ❌ |
| MAML [Finn 2017] | ❌ | ❌ | inner-loop | ❌ |
| ANIL [Rajendran 2019] | ❌ | ❌ | fusion-only | ❌ |
| FEAT [Ye 2020] | ❌ | ❌ | task-emb | ❌ |
| **MetaProto (Ours)** | ✅ | ✅ | 4种可选 | ✅ |

---

## 7. 创新点总结 (Novelty Claims)

```
1. Industrial Meta Task Formalization
   · 提出工业元任务四元组 (Component, Material, Lighting, Defect)
   · 域变化四维分类 (C/M/L/D shift)

2. Four-Module Meta-Learning Framework
   · 首个面向工业缺陷检测的端到端模块化设计
   · 各模块可独立替换/消融

3. Task-Conditioned Feature Adaptation
   · Task embedding τ 跨任务知识注入
   · 材料/工艺元数据 → 特征调制

4. Versatile Prototype + Distance Module
   · 3 × 3 = 9 种组合
   · 灵活适配不同 K-shot 场景
```

---

## 8. 论文实验章节预告

```
4. Experiments

4.1 Datasets
    · Aerospace blade defect (this work)
    · Car paint (CMPL: CarDD + ASDE)
    · Optional: PCB / Steel surface / Bearing

4.2 Protocol
    · Cross-domain few-shot
    · 3-seed × 100 episodes pooled

4.3 Main Results
    · Aerospace: 85.2% (ours) vs ProtoNet baseline 53.4%

4.4 Ablation Studies
    · Module-by-module ablation
    · Backbone swap (YOLO26 / ResNet / Swin)
    · Distance metric ablation
    · K-shot sensitivity (1/3/5/10)

4.5 Visualization
    · t-SNE of learned embeddings
    · Attention maps (CBAM)
    · Prototype visualization
```

---

## 9. 论文命名建议

| 命名 | 含义 |
|------|------|
| **MetaProto** | Meta + Prototypical Networks |
| **MetaDefect** | Meta + Defect Detection |
| **CrossDefect** | 强调跨域 |
| **TASAD** | Task-Adaptive Shot detector |
| **IFD-Meta** | Industrial Few-shot Defect Meta |

推荐 **MetaProto** — 简洁，已暗示方法核心，且与 ProtoNet 关系明确。

---

**下一步**：进入 **第四阶段：代码重构规划** (`refactor_plan.md`)。我会逐文件分析，标出风险等级，但要等你确认后才会改代码。