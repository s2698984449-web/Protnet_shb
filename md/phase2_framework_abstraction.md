# Phase 2 — 通用方法抽象

> 从工业缺陷小样本项目中抽象出可迁移的 Meta-Learning Framework
>
> 分析日期: 2026-07-29
> 范围: 单模态 RGB (适配汽车漆面 / 轴承 / PCB / 半导体等通用场景)

---

## A. 与具体工业任务绑定的部分 (Constraint-specific)

这些必须从通用框架中抽象剥离：

### 1. 数据层面

| 绑定项 | 示例 | 抽象方式 |
|--------|------|----------|
| 文件夹结构 | `domain_{0,1,2}/0_scratch/` | 抽象为 `ClassFolder` 协议 |
| 缺陷类名 | `scratch`, `damage`, `dent` | 抽象为 `DefectTaxonomy` |
| 图像命名 | `rgb.png`, `ultrasonic.npy` | 抽象为 `ModalityReader` |
| 元数据字段 | `material_id`, `process_params` | 抽象为 `TaskMetadata` |
| 域ID号 | 0, 1, 2, 3 | 抽象为 `DomainId` (任意 str) |
| 路径 | `data/{track}/processed/...` | 抽象为 `DatasetRoot` |

### 2. 任务层面

| 绑定项 | 抽象方式 |
|--------|----------|
| 航空 `2-way 5-shot` | 抽象为 `N-way K-shot` |
| 车漆 `d0 → d3` | 抽象为 `source_domains → target_domain` |
| 试件 holdout | 抽象为 `heldout_specimens` |
| `"scratch" / "damage"` 标签 | 抽象为 `class_idx` (无意义整数) |

### 3. 模型层面

| 绑定项 | 抽象方式 |
|--------|----------|
| YOLO26 backbone | 抽象为 `FeatureEncoder` (任意视觉 backbone) |
| 128-d 嵌入 | 抽象为 `embed_dim` (超参数) |
| AL7075/TC4/CFRP 物理参数 | 抽象为 `material_context` (可选输入) |
| `domain_split_v2` CSV 协议 | 抽象为 `DomainPartitionStrategy` |

### 4. 训练层面

| 绑定项 | 抽象方式 |
|--------|----------|
| `train_domains="0,1,2"` | 抽象为 `source_domain_ids` |
| `test_domain=3` | 抽象为 `target_domain_id` |
| `iters=500`, `shots=5` | 抽象为训练超参数 |

---

## B. 可以形成通用方法的部分 (Framework-specific)

### 1. Industrial Meta Task 构造策略

#### 1.1 抽象任务定义

```python
@dataclass
class IndustrialMetaTask:
    """A unified meta-task definition for industrial defect detection."""

    # 数据标识
    task_id: str                # 任务唯一标识
    domain_id: str              # 域ID (零件/批次/产线组合)
    classes: List[str]          # 该 task 包含的缺陷类别

    # 元数据 (Task Description)
    component: str              # 部件类型 (e.g., "blade", "door_panel")
    material: str               # 材料 (e.g., "AL7075", "steel", "PCB_FR4")
    defect_category: str        # 缺陷大类 (e.g., "surface", "structural")
    illumination: str           # 光照条件 (e.g., "natural", "studio", "UV")

    # Few-shot 配置
    n_way: int                  # 2~5
    k_shot: int                 # 1~10
    n_query: int                # 15~30

    # 采样策略
    sample_policy: str          # "balanced", "hard_mining", "natural"
```

#### 1.2 域划分原则 (Domain Partition Principle)

**核心思想**: 域 ≠ 简单的 K-fold 划分，而是模拟真实工业变化

```
域变化维度 (Domain Shift Dimensions):
  ① Component shift   (零件变更)   → 几何结构、ROI 区域变化
  ② Material shift    (材料变更)   → 表面纹理、反射特性变化
  ③ Lighting shift    (光照变更)   → 亮度、对比度、色温变化
  ④ Defect shift      (缺陷变异)   → 类内方差增加
```

**域划分协议**:

```python
class DomainPartitionStrategy:
    """Generic domain partition for industrial few-shot."""

    def __init__(self, strategy="metadata_based"):
        self.strategy = strategy

    def split(self, specimens, metadata):
        """
        Args:
            specimens: List[Specimen] with (sample_dir, label, meta)
            metadata: Dict[specimen_id] -> {material, batch, lighting, ...}

        Returns:
            domain_map: Dict[specimen_id] -> domain_id
        """
        if self.strategy == "metadata_based":
            # 按 metadata 字段聚类 (推荐)
            return self._cluster_by_metadata(specimens, metadata)
        elif self.strategy == "random_kfold":
            return self._random_partition(specimens, k=4)
        elif self.strategy == "industrial_stratified":
            # 按 试件/批次/时间 划分
            return self._industrial_stratify(specimens, metadata)
```

**可迁移的域划分原则**:
- **训练域**: 包含足够多 component × material × lighting 组合
- **测试域 (target)**: 包含 \*\*全新 component 或 material\*\*，模拟"新产线上线"
- **域冲突避免**: 同一 ROI crop 不能横跨训练/测试域

#### 1.3 Episode 构造抽象

```python
class EpisodeSampler:
    """N-way K-shot episode sampler for any industrial task."""

    def sample_episode(self, domain_id, n_way, k_shot, n_query):
        """
        Returns:
            support: (n_way * k_shot, [rgb, optional_aux])
            query:   (n_way * n_query, [rgb, optional_aux])
            labels:  (n_way * k_shot,) and (n_way * n_query,)
        """
        # 1. 从指定 domain 采样
        # 2. 随机选 n_way 类
        # 3. 每类选 k_shot + n_query 样本
        # 4. 应用 episode-level augmentation
        pass
```

---

### 2. Feature Representation 抽象

#### 2.1 当前模型学习的是什么？

通过项目分析 (尤其 ProtoNet embed 机制)，模型学习的本质特征：

```
f_θ(x) = Encoder(x) → 128-d 嵌入向量

其中 Encoder 学习的特征类型:
  ✅ 跨域缺陷共性特征 (defect edges, texture anomalies)
  ✅ 产品结构不变性 (object position 不变)
  ✅ 域无关的视觉语义 (scratch/dent/crack 的形态学pattern)
  ❌ 而不是: 特定产品名称、特定光照配置
```

#### 2.2 通用特征表示

```python
class UniversalIndustrialFeatureEncoder(nn.Module):
    """
    跨产品/跨材料的视觉特征编码器
    Backbone swap-in: YOLO26 / ResNet / Swin / ConvNeXt
    """

    def __init__(self, backbone='yolo26', embed_dim=128, use_cbam=False):
        self.backbone = build_backbone(backbone)  # 任意 CNN/ViT
        self.attention = CBAM(embed_dim) if use_cbam else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        feat = self.backbone(x)
        if self.attention:
            feat = self.attention(feat)
        return self.proj(self.pool(feat).flatten(1))
```

**关键设计原则**:

| 原则 | 解释 |
|------|------|
| **域无关** | Backbone 应在多种工业数据上预训练 (ImageNet 或自监督) |
| **维度统一** | 无论是 YOLO26 还是 ResNet，输出统一为 `embed_dim` |
| **可插拔** | 支持将整套 backbone 替换为新架构 |
| **两阶段训练** | Phase 0: 预训练 (在多种工业数据上); Phase 1: 元学习 (冻结/解冻) |

#### 2.3 特征 quality 评估 (LogReg 探针)

现有项目已有 `diagnose_features.py` 工具, 用于验证特征可分性:

```
LogReg accuracy > 0.7  → 特征空间可分
LogReg accuracy > 0.85 → 可直接 ProtoNet few-shot
```

这应该作为框架的内置诊断工具。

---

### 3. Meta Knowledge 抽象

#### 3.1 哪些知识应该被共享？

```
Shared Meta Knowledge (跨任务不变):
  ① Defect prior          缺陷的视觉先验 (边缘、纹理、形状)
  ② Visual grammar        工业图像的视觉语法 (对称、周期、纹理)
  ③ Domain-invariant feats 跨域不变的视觉特征
  ④ Backbone weights      预训练特征提取器

Task-specific Knowledge (任务特异):
  ① Class prototypes      每类的原型向量
  ② Distance metric       类内协方差矩阵 (Mahalanobis)
  ③ Material context      任务级元数据嵌入
```

#### 3.2 元知识的形式化

```python
class MetaKnowledgeBase:
    """Meta-learned knowledge shared across industrial tasks."""

    def __init__(self):
        self.backbone = None          # 共享视觉特征提取器
        self.defect_dictionary = {}   # 缺陷类别视觉原型集合
        self.material_embeddings = {} # 材料语义嵌入
        self.task_memory = {}        # 任务级记忆 (可选)

    def get_pretrained_backbone(self):
        """Returns the shared backbone, lazy-loaded."""
        return self.backbone

    def query_prototype(self, defect_class, domain_id):
        """Retrieve cached prototype for a defect class in a domain."""
        return self.defect_dictionary.get((defect_class, domain_id))

    def update_prototype(self, defect_class, domain_id, prototype):
        """Online update of prototype after new samples."""
        self.defect_dictionary[(defect_class, domain_id)] = prototype
```

#### 3.3 任务级元数据 → Task Embedding

当前 `TaskEncoder` 将 `{material_params, process_params, defect_history}` 编码为 64-d。**通用化**：

```python
class TaskEncoder(nn.Module):
    """
    Maps domain metadata to task embedding τ.
    Generalizes to any industrial setting with descriptor vector.
    """

    def __init__(self, descriptor_dim=30, out_dim=64):
        # 任意长度的元数据描述符
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, metadata_descriptor):
        """
        Args:
            metadata_descriptor: (B, descriptor_dim) tensor
        Returns:
            task_embedding: (B, out_dim)
        """
        return self.encoder(metadata_descriptor)
```

**Descriptor 包含哪些字段** (通用化):
- Material physics parameters (12-d)
- Process conditions (8-d)
- Defect frequency history (10-d)
- **新增**: Camera/lighting ID, Part geometry hash, Inspection line ID

---

### 4. Fast Adaptation 抽象

#### 4.1 新产品/新类别上线时, 需要更新什么？

```
                          固定 (frozen)              更新 (adapted)
                       ──────────────────         ──────────────────
Phase 0 预训练:         无                         Backbone weights
Phase 1 元学习:         Backbone                   Fusion, Task Encoder
Few-shot 适配:          Backbone                   Prototypes / fast weights
```

#### 4.2 几种适配策略 (可选择)

| 策略 | 适用场景 | 框架实现 |
|------|----------|----------|
| **ProtoNet** (零训练) | K ≥ 5 / 类 | 直接计算 support set 原型 |
| **ANIL** (冻结 backbone) | K = 3~5 | 内循环仅适应 fusion |
| **MAML** (内循环适应) | K = 1~3 | clone + 5-step adapt |
| **Prototype Refinement** | K = 5~10 | 原型 + 距离度量联合优化 |
| **Test-time Tuning** | 长时间运行 | EMA 更新原型 |

#### 4.3 通用适配流程

```python
class FastAdaptation:
    """Unified fast adaptation interface."""

    def adapt(self, model, support_set, support_labels, **kwargs):
        """
        Adapts model to new task using support set.

        Args:
            model: Pre-trained meta-model
            support_set: (N_s, ...) samples
            support_labels: (N_s,) labels

        Returns:
            adapted_model: Ready for inference
        """
        strategy = kwargs.get('strategy', 'protonet')

        if strategy == 'protonet':
            return self._protonet_adapt(model, support_set, support_labels)
        elif strategy == 'anil':
            return self._anil_adapt(model, support_set, support_labels)
        elif strategy == 'maml':
            return self._maml_adapt(model, support_set, support_labels)
        elif strategy == 'prototype_refine':
            return self._prototype_refine(model, support_set, support_labels)
```

#### 4.4 适配参数选择原则

```
样本数 K    推荐策略            理由
─────     ────────────────    ───────────────────────
K = 1~3   MAML/ANIL           原型不稳, 需梯度支持
K = 5     ProtoNet            原型收敛, 距离度量重要
K = 10+   ProtoNet + Refine   原型充分, 可微调
```

---

## C. 通用框架四大模块设计 (Module Design)

### Module 1: Universal Industrial Feature Encoder

```python
class UniversalIndustrialFeatureEncoder(nn.Module):
    """
    输入: 任意 RGB 工业图像
    输出: embed_dim 维特征向量
    Backbone: 可插拔 (YOLO26 / ResNet50 / Swin-T / ConvNeXt)
    """
```

**职责**:
- 提取跨产品/跨材料的视觉特征
- 支持两阶段训练 (预训练 + 元学习)
- 可选注意力增强 (CBAM)

### Module 2: Industrial Meta Task Generator

```python
class IndustrialMetaTaskGenerator:
    """
    输入: 数据集 + 域划分协议
    输出: 符合 IndustrialMetaTask 规范的 episode 采样器
    """
```

**职责**:
- 模拟真实工业变化 (component/material/lighting shift)
- 构造 N-way K-shot episode
- 应用 episode-level augmentation

### Module 3: Prototype-based Adaptation Module

```python
class PrototypeAdaptationModule(nn.Module):
    """
    输入: support set (rgb, labels)
    输出: 类别原型 + 适配后的距离度量
    """
```

**职责**:
- 支持多种原型构建 (mean/trim_mean/mean_shift)
- 支持多种距离度量 (euclidean/cosine/mahalanobis)
- 提供 prototype 缓存/更新机制

### Module 4: Domain Generalization Module

```python
class DomainGeneralizationModule(nn.Module):
    """
    输入: 多个 source domain 的样本
    输出: 域不变的泛化特征
    """
```

**职责**:
- 通过元学习训练获得域不变特征
- 支持 task embedding 条件化
- 跨域一致性正则化 (可选)

---

## 总结: 抽象前后对比

| 维度 | 当前代码 (Project-specific) | 通用框架 (Framework) |
|------|----------------------------|----------------------|
| 数据 | `AviationDefectDataset` | `IndustrialDefectDataset` + `ClassFolder` |
| 任务 | `2-way 5-shot on blade/damage` | `N-way K-shot on any defect` |
| 模块 | YOLO26 + Thermal + US + Fusion | YOLO26 + (optional aux) |
| 域划分 | `domain_split_v2.py` CSV | `DomainPartitionStrategy` |
| Episode | `task_builder.py` | `EpisodeSampler` |
| 适配 | ProtoNet / MAML | `FastAdaptation` (4 strategies) |
| 元数据 | `AL7075`/`CFRP` | `MaterialDescriptor` (any) |
| 评估 | 单一域 holdout | `MultiProtocolEvaluator` |

---

**下一步**: 当你确认第二阶段 OK 后，进入 **第三阶段：论文级 Framework 设计**，给出四大模块的完整架构图。
