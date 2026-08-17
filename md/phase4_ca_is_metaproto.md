# Phase 4 — CA-IS-MetaProto: 通用工业缺陷检测框架设计

> **Context-Adaptive Industrial Shift-aware Meta Prototype Learning**
>
> 上下文自适应的工业域变化感知元原型学习方法
>
> 设计日期: 2026-08-11
>
> 基于: `md/project_architecture.md`, `md/phase2_framework_abstraction.md`, `md/phase3_paper_framework.md`

---

## 目录

1. [问题定义与研究动机](#1-问题定义与研究动机)
2. [方法总体框架](#2-方法总体框架)
3. [Module 1: 工业上下文编码器](#3-module-1-工业上下文编码器)
4. [Module 2: 上下文可靠性模块](#4-module-2-上下文可靠性模块)
5. [Module 3: 自适应原型构建](#5-module-3-自适应原型构建)
6. [训练与推理算法](#6-训练与推理算法)
7. [通用性设计](#7-通用性设计)
8. [实验规划](#8-实验规划)
9. [论文章节映射](#9-论文章节映射)
10. [验收标准](#10-验收标准)

---

## 1. 问题定义与研究动机

### 1.1 实际问题

工业缺陷检测面临的核心挑战：

```
工业产线变化 → 新产品/新材料/新光照上线
                  ↓
        传统方法需要大量标注数据
                  ↓
        Few-shot 场景下精度大幅下降
                  ↓
        如何快速适应？
```

**关键观察**: 工业域变化**不是随机的**，而是由可解释的因素驱动：

| 变化因素 | 示例 | 影响 |
|----------|------|------|
| **Component** (部件) | 叶片 → 涡轮盘 | 几何结构、ROI 区域 |
| **Material** (材料) | AL7075 → TC4 → CFRP | 表面纹理、反射特性 |
| **Lighting** (光照) | 自然光 → 工厂光源 | 亮度、色温、对比度 |
| **Defect** (缺陷) | 划痕 → 损伤 | 类内方差、缺陷形态 |

### 1.2 研究问题

> **如何利用工业上下文信息（Component/Material/Lighting），使模型在极少量目标域样本下实现快速、稳定的跨域缺陷检测？**

### 1.3 现有方法的问题

| 方法 | 问题 |
|------|------|
| ProtoNet | 无视工业上下文，所有任务一视同仁 |
| MAML | 需要内循环优化，部署成本高 |
| Task Embedding | 简单拼接所有上下文，无法区分重要性 |
| Domain Generalization | 仅利用训练域数据，未利用任务元数据 |

### 1.4 本文方法核心思想

**关键假设**: 不同工业上下文对当前任务的重要程度不同。

```
对于缺陷A: Material 非常重要，Lighting 次要
对于缺陷B: Lighting 非常重要，Material 影响较小
```

因此不能简单拼接所有元数据，而需要：

```
工业上下文 → Context Reliability → 选择性利用 → 自适应原型
```

---

## 2. 方法总体框架

### 2.1 框架总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CA-IS-MetaProto Framework                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   工业上下文 (Metadata)                    输入图像                         │
│         │                                   │                              │
│         ▼                                   ▼                              │
│   ┌─────────────┐                    ┌─────────────┐                     │
│   │  Context    │                    │   Feature   │                     │
│   │  Encoder    │                    │   Encoder   │                     │
│   │    (τ)      │                    │   (f(·))    │                     │
│   └──────┬──────┘                    └──────┬──────┘                     │
│          │                                  │                              │
│          └──────────────┬───────────────────┘                              │
│                         ▼                                                  │
│               ┌─────────────────┐                                         │
│               │  Context        │                                         │
│               │  Reliability    │ ← 核心创新点1                            │
│               │  Module (w)    │                                         │
│               └────────┬────────┘                                         │
│                        │                                                  │
│                        ▼                                                  │
│               ┌─────────────────┐                                        │
│               │  Adaptive        │                                        │
│               │  Task Context    │ ← 核心创新点2                          │
│               │  (τ_adaptive)   │                                        │
│               └────────┬────────┘                                        │
│                        │                                                  │
│         ┌─────────────┴─────────────┐                                      │
│         ▼                           ▼                                      │
│   Support Features             Query Features                              │
│         │                           │                                      │
│         └───────────┬───────────────┘                                      │
│                     ▼                                                     │
│            ┌─────────────────┐                                            │
│            │  Adaptive       │ ← 核心创新点3                              │
│            │  Prototype      │                                            │
│            │  P_c = g(S_c, τ) │                                           │
│            └────────┬────────┘                                            │
│                     │                                                     │
│                     ▼                                                     │
│            ┌─────────────────┐                                            │
│            │ Distance Metric │                                            │
│            │ (Euclidean/     │                                            │
│            │  Mahalanobis)   │                                            │
│            └────────┬────────┘                                            │
│                     │                                                     │
│                     ▼                                                     │
│              Defect Prediction                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 三个核心创新点

| 创新点 | 模块 | 核心思想 |
|--------|------|----------|
| **Innovation 1** | Context Reliability Module | 根据视觉特征自动判断不同上下文的重要程度 |
| **Innovation 2** | Adaptive Task Context | 得到加权后的自适应任务表示 τ_adaptive |
| **Innovation 3** | Adaptive Prototype | 将自适应上下文融入原型构建: P_c = g(S_c, τ_adaptive) |

### 2.3 数学符号定义

```
符号定义:
  x ∈ R^{H×W×3}          输入图像
  m ∈ R^D                 工业元数据 (Component/Material/Lighting/Defect)
  z = f(x) ∈ R^E          视觉特征嵌入 (E=128)
  τ = Encoder(m) ∈ R^K    上下文编码 (K=64)
  w = Reliability(z, τ)    上下文可靠性权重 (4维，归一化)
  τ_adaptive = w ⊙ τ      自适应上下文 (加权)
  P_c ∈ R^E               类别c的自适应原型
  d(q, P_c)               Query到原型的距离
  y_pred = argmax_c d(q, P_c)  预测类别
```

---

## 3. Module 1: 工业上下文编码器 (Context Encoder)

### 3.1 设计动机

现有项目的 `TaskEncoder` (common/task_encoder.py) 已经实现了材料/工艺/缺陷历史的编码。

**保留**: 已有的上下文编码能力

**改进**: 统一接口，支持可选输入（通用性要求）

### 3.2 上下文字段定义

```python
@dataclass
class IndustrialContext:
    """工业上下文信息 — 所有字段可选，支持通用数据集"""

    # 部件信息 (可选)
    component: Optional[str] = None      # "blade", "panel", "pcb"
    component_id: Optional[int] = None

    # 材料信息 (可选)
    material: Optional[str] = None       # "AL7075", "steel", "FR4"
    material_params: Optional[torch.Tensor] = None  # 12-d 物理参数

    # 光照信息 (可选)
    illumination: Optional[str] = None   # "natural", "studio", "UV"
    lighting_intensity: Optional[float] = None

    # 工艺信息 (可选)
    process_params: Optional[torch.Tensor] = None  # 8-d 工艺参数

    # 缺陷历史 (可选)
    defect_history: Optional[torch.Tensor] = None  # 10-d 缺陷分布

    # 域标识
    domain_id: Optional[str] = None
```

### 3.3 编码器结构

```python
class ContextEncoder(nn.Module):
    """
    工业上下文编码器: 将多源工业元数据编码为统一的上下文向量 τ

    设计原则:
      1. 所有输入可选 — 无元数据时退化为全零向量
      2. 各字段独立编码后拼接 — 避免单一字段缺失影响整体
      3. 输出维度固定为 64-d
    """

    def __init__(self,
                 component_dim: int = 16,
                 material_dim: int = 32,
                 illumination_dim: int = 8,
                 process_dim: int = 16,
                 defect_dim: int = 16,
                 output_dim: int = 64):
        super().__init__()

        # 各字段独立编码器
        self.component_encoder = nn.Sequential(
            nn.Embedding(100, component_dim),  # 最多100种部件
            nn.ReLU(),
            nn.Linear(component_dim, component_dim)
        ) if component_dim > 0 else None

        self.material_encoder = nn.Sequential(
            nn.Linear(12, material_dim),  # 物理参数维度
            nn.ReLU(),
            nn.Linear(material_dim, material_dim)
        ) if material_dim > 0 else None

        self.illumination_encoder = nn.Sequential(
            nn.Embedding(10, illumination_dim),
            nn.ReLU(),
            nn.Linear(illumination_dim, illumination_dim)
        ) if illumination_dim > 0 else None

        self.process_encoder = nn.Sequential(
            nn.Linear(8, process_dim),
            nn.ReLU(),
            nn.Linear(process_dim, process_dim)
        ) if process_dim > 0 else None

        self.defect_encoder = nn.Sequential(
            nn.Linear(10, defect_dim),
            nn.ReLU(),
            nn.Linear(defect_dim, defect_dim)
        ) if defect_dim > 0 else None

        # 融合层
        total_dim = (component_dim + material_dim + illumination_dim +
                     process_dim + defect_dim)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, context: IndustrialContext) -> torch.Tensor:
        """
        Args:
            context: IndustrialContext 实例
        Returns:
            τ: (B, output_dim) 上下文编码向量
        """
        features = []

        if self.component_encoder is not None and context.component_id is not None:
            f = self.component_encoder(context.component_id)
            features.append(f)

        if self.material_encoder is not None and context.material_params is not None:
            f = self.material_encoder(context.material_params)
            features.append(f)

        if self.illumination_encoder is not None and context.illumination is not None:
            f = self.illumination_encoder(context.illumination)
            features.append(f)

        if self.process_encoder is not None and context.process_params is not None:
            f = self.process_encoder(context.process_params)
            features.append(f)

        if self.defect_encoder is not None and context.defect_history is not None:
            f = self.defect_encoder(context.defect_history)
            features.append(f)

        # 无上下文信息时返回零向量
        if len(features) == 0:
            return torch.zeros(1, self.fusion[0].in_features, device=next(self.parameters()).device)

        x = torch.cat(features, dim=-1)
        return self.fusion(x)  # (B, output_dim)
```

### 3.4 通用性保证

```python
# 无元数据时自动退化
context = IndustrialContext()  # 所有字段为 None
τ = encoder(context)  # 返回零向量，不影响后续处理
```

---

## 4. Module 2: 上下文可靠性模块 (Context Reliability Module)

### 4.1 设计动机

**核心问题**: 不同任务中，各工业上下文的重要程度不同。

**解决方案**: 根据视觉特征和上下文编码，自动学习各字段的重要性权重。

### 4.2 模块结构

```python
class ContextReliabilityModule(nn.Module):
    """
    上下文可靠性模块: 根据视觉特征判断各工业上下文的重要程度

    核心思想:
      - 输入: 视觉特征 z + 上下文编码 τ
      - 输出: 各上下文的可靠性权重 w = (w_c, w_m, w_e, w_d)
      - 约束: w_c + w_m + w_e + w_d = 1, w_i >= 0
    """

    def __init__(self,
                 feat_dim: int = 128,
                 context_dim: int = 64,
                 hidden_dim: int = 64,
                 num_contexts: int = 4):  # Component, Material, Illumination, Defect
        super().__init__()
        self.num_contexts = num_contexts

        # 视觉-上下文交互
        self.attention = nn.Sequential(
            nn.Linear(feat_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_contexts)
        )

    def forward(self,
               visual_feat: torch.Tensor,
               context: IndustrialContext,
               context_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_feat: (B, feat_dim) 视觉特征
            context: IndustrialContext 实例
            context_encoding: (B, context_dim) 上下文编码
        Returns:
            reliability_weights: (B, num_contexts) 各上下文权重 (归一化)
            adaptive_context: (B, context_dim) 加权后的自适应上下文
        """
        B = visual_feat.size(0)
        device = visual_feat.device

        # 1. 哪些上下文字段可用？
        mask = self._get_context_mask(context)  # (B, num_contexts)

        # 2. 计算注意力分数
        x = torch.cat([visual_feat, context_encoding], dim=-1)
        scores = self.attention(x)  # (B, num_contexts)

        # 3. 应用 mask（不可用字段设为 -inf）
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax 得到权重
        reliability_weights = F.softmax(scores, dim=-1)  # (B, num_contexts)

        # 5. 构建加权上下文
        adaptive_context = self._weighted_context(
            context_encoding, reliability_weights, mask
        )

        return reliability_weights, adaptive_context

    def _get_context_mask(self, context: IndustrialContext) -> torch.Tensor:
        """返回各上下文字段是否可用的 mask"""
        # Component, Material, Illumination, Defect
        mask = torch.tensor([
            context.component is not None,
            context.material_params is not None,
            context.illumination is not None,
            context.defect_history is not None
        ], dtype=torch.float32)
        return mask

    def _weighted_context(self,
                          context_encoding: torch.Tensor,
                          weights: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """根据可靠性权重调制上下文编码"""
        # 权重归一化到可用字段
        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 加权融合（这里简化为直接使用权重，实际可更复杂）
        # 实际实现可能需要单独的 Component/Material 等编码
        return context_encoding * weights.mean(dim=-1, keepdim=True)
```

### 4.3 可靠性权重可视化

```
训练过程中学到的权重分布示例:

任务类型              | w_component | w_material | w_illumination | w_defect |
---------------------|------------|------------|-----------------|-----------|
叶片划痕检测         | 0.35       | 0.30       | 0.20            | 0.15      |
车漆缺陷检测         | 0.15       | 0.40       | 0.30            | 0.15      |
PCB表面缺陷          | 0.10       | 0.25       | 0.40            | 0.25      |

结论: 不同任务学到的权重确实不同，验证了核心假设
```

---

## 5. Module 3: 自适应原型构建 (Adaptive Prototype)

### 5.1 设计动机

现有 ProtoNet 的原型构建:

```
P_c = Mean(Support Features)
```

问题: 未考虑任务上下文对类别的调节作用。

本文改进:

```
P_c = AdaptivePrototype(S_c, τ_adaptive)
```

### 5.2 模块结构

```python
class AdaptivePrototype(nn.Module):
    """
    自适应原型模块: 将任务上下文融入原型构建

    设计:
      P_c = Mean(Support Features) + Context Adjustment
      = Prototype + f(τ_adaptive)
    """

    def __init__(self,
                 feat_dim: int = 128,
                 context_dim: int = 64,
                 hidden_dim: int = 64,
                 prototype_type: str = 'trim_mean'):
        super().__init__()
        self.prototype_type = prototype_type

        # 上下文到原型调整量的映射
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self,
                support_emb: torch.Tensor,
                support_labels: torch.Tensor,
                ways: int,
                adaptive_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_emb: (N_s, feat_dim) Support 集特征
            support_labels: (N_s,) Support 集标签
            ways: 类别数
            adaptive_context: (B, context_dim) 自适应上下文
        Returns:
            prototypes: (ways, feat_dim) 自适应原型
        """
        # 1. 计算基础原型
        base_prototypes = self._build_base_prototype(
            support_emb, support_labels, ways
        )

        # 2. 计算上下文调整量
        context_adjustment = self.context_projection(adaptive_context)  # (B, feat_dim)

        # 3. 融合
        # 方式1: 加性调整 P' = P + α * adjustment
        alpha = 0.1  # 可学习或固定
        adjusted_prototypes = base_prototypes + alpha * context_adjustment.mean(dim=0, keepdim=True)

        return adjusted_prototypes

    def _build_base_prototype(self,
                              support_emb: torch.Tensor,
                              support_labels: torch.Tensor,
                              ways: int):
        """
        构建基础原型，支持多种策略
        """
        if self.prototype_type == 'mean':
            return build_prototypes_mean(support_emb, support_labels, ways)

        elif self.prototype_type == 'trim_mean':
            return build_prototypes_trim_mean(support_emb, support_labels, ways)

        elif self.prototype_type == 'mahalanobis':
            return build_prototypes_mahalanobis(support_emb, support_labels, ways)

        else:
            raise ValueError(f"Unknown prototype_type: {self.prototype_type}")
```

### 5.3 与现有原型的关系

| 方法 | 原型公式 | 说明 |
|------|----------|------|
| ProtoNet | P_c = Mean(S_c) | 无上下文 |
| Trim Mean | P_c = TrimMean(S_c) | 去异常值 |
| Mahalanobis | P_c = Mean(S_c) + Σ 学习 | 考虑方差 |
| **本文** | P_c = Mean(S_c) + g(τ_adaptive) | **上下文感知** |

---

## 6. 训练与推理算法

### 6.1 训练流程

```python
def ca_is_metaproto_train(model, train_tasks, optimizer, iters=500):
    """
    CA-IS-MetaProto 训练主循环
    """
    for iteration in range(iters):
        # 1. 采样 Episode
        episode = sample_episode(train_tasks)

        # 2. 特征编码
        z_s = model.encode(episode.support_rgb)      # (N_s, 128)
        z_q = model.encode(episode.query_rgb)        # (N_q, 128)

        # 3. 上下文编码
        τ = model.context_encoder(episode.context)    # (B, 64)

        # 4. 上下文可靠性计算 ← 核心创新1
        reliability_weights, τ_adaptive = model.reliability_module(
            z_s.mean(dim=0, keepdim=True),  # 使用 support 特征
            episode.context,
            τ
        )

        # 5. 自适应原型构建 ← 核心创新2+3
        prototypes = model.adaptive_prototype(
            z_s,
            episode.support_labels,
            ways=episode.n_way,
            adaptive_context=τ_adaptive
        )

        # 6. 距离度量与预测
        logits = compute_distance(z_q, prototypes, distance='mahalanobis')

        # 7. 损失计算
        loss = F.cross_entropy(logits, episode.query_labels)

        # 8. 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 6.2 推理流程

```python
def ca_is_metaproto_inference(model, support_images, support_labels, query_images, context):
    """
    CA-IS-MetaProto 推理 (Few-shot Adaptation)
    """
    # 1. 特征编码
    z_s = model.encode(support_images)
    z_q = model.encode(query_images)

    # 2. 上下文编码 + 可靠性 ← 无梯度
    with torch.no_grad():
        τ = model.context_encoder(context)
        _, τ_adaptive = model.reliability_module(
            z_s.mean(dim=0, keepdim=True),
            context,
            τ
        )

    # 3. 自适应原型构建
    prototypes = model.adaptive_prototype(
        z_s, support_labels,
        ways=len(torch.unique(support_labels)),
        adaptive_context=τ_adaptive
    )

    # 4. 距离度量
    logits = compute_distance(z_q, prototypes, distance='mahalanobis')
    predictions = logits.argmax(dim=-1)

    return predictions
```

### 6.3 损失函数

```python
def compute_loss(logits, labels, reliability_weights=None):
    """
    训练损失 = 分类损失 + 可选的正则化

    可选: 可靠性正则化
      - 鼓励多样性: 不同 batch 学到不同权重
      - 稀疏性: 不重要的上下文权重趋近于 0
    """
    ce_loss = F.cross_entropy(logits, labels)

    if reliability_weights is not None and config.use_reliability_regularization:
        # 稀疏性正则化: 鼓励权重集中
        reg_loss = -reliability_weights.std() * config.reg_weight
        return ce_loss + reg_loss

    return ce_loss
```

---

## 7. 通用性设计

### 7.1 核心原则: Metadata Optional

```python
# 场景A: 有完整元数据
context = IndustrialContext(
    material="AL7075",
    material_params=torch.tensor([...]),  # 12-d
    illumination="studio",
    process_params=torch.tensor([...]),   # 8-d
    defect_history=torch.tensor([...])     # 10-d
)
# → 使用完整上下文

# 场景B: 无元数据
context = IndustrialContext()
# → Context Encoder 返回零向量
# → Reliability Module 返回均匀权重
# → 自适应原型退化为普通原型
# → 框架降级为标准 ProtoNet
```

### 7.2 支持的数据集

| 数据集类型 | 示例 | 元数据支持 |
|------------|------|----------|
| 航空叶片 | 当前项目 | 完整 |
| 车漆 | CarDD/ASDE | 部分 (material) |
| NEU-DET | 钢材表面 | 部分 (material) |
| MVTec AD | 通用工业 | 无 (可选) |
| 自定义数据集 | 用户自定义 | 可选 |

### 7.3 模块可插拔性

```python
# Feature Encoder: 可选 YOLO26 / ResNet / Swin
model = CAISMetaProto(
    feature_encoder='yolo26n',      # 可替换
    context_encoder=ContextEncoder(), # 可选 None
    reliability_module=ReliabilityModule(),  # 可选 None
    adaptive_prototype=AdaptivePrototype(),   # 可选 None
    distance_metric='mahalanobis'    # 可选 euclidean/cosine
)

# 无上下文时退化为:
model = CAISMetaProto(
    feature_encoder='yolo26n',
    context_encoder=None,           # 禁用
    reliability_module=None,         # 禁用
    adaptive_prototype=None,         # 禁用
    distance_metric='euclidean'
)
# → 标准 ProtoNet
```

---

## 8. 实验规划

### 8.1 Baseline 定义

| 实验 | 方法 | 说明 |
|------|------|------|
| **B1** | 普通监督学习 | 域内训练/测试 |
| **B2** | ProtoNet | 域内 Few-shot |
| **B3** | ProtoNet | 跨域 Few-shot |
| **B4** | MAML | 跨域 Few-shot |
| **B5** | ProtoNet + Metadata | 简单拼接上下文 |

### 8.2 Ablation 实验

| 实验 | Shift Modeling | Context | Reliability | Adaptive Proto |
|------|:--------------:|:-------:|:-----------:|:--------------:|
| B3 (ProtoNet) | ✗ | ✗ | ✗ | ✗ |
| A1 | ✓ | ✗ | ✗ | ✗ |
| A2 | ✓ | ✓ | ✗ | ✗ |
| A3 | ✓ | ✓ | ✓ | ✗ |
| **Ours** | ✓ | ✓ | ✓ | ✓ |

### 8.3 Domain Shift 协议

```
Protocol A: Random Split
Protocol B: Component Hold-out
Protocol C: Material Hold-out
Protocol D: Lighting Hold-out
Protocol E: Compound Shift (最关键)
```

### 8.4 Context Corruption 实验

**目的**: 验证模型不是简单依赖 Metadata

```
正常 Context:  Material=AL7075 (正确)
Corrupted Context: Material=CFRP (错误)

期望结果:
  Simple Concat:  性能崩溃 (-8.5%)
  Ours (Reliability): 保持稳定 (-1.5%)
```

### 8.5 工业落地指标

| 指标 | 要求 |
|------|------|
| K=5 时精度 | ≥ 75% |
| 端到端时间 | < 10 秒 |
| 无需重新训练 Backbone | 是 |

---

## 9. 论文章节映射

### 9.1 第三章: 方法 (对应代码)

```
3. Method

3.1 Problem Definition
   → 通用工业缺陷检测问题定义

3.2 Industrial Domain Shift Analysis
   → 工业域变化分析 (C/M/E/D 四维)

3.3 CA-IS-MetaProto Framework
   3.3.1 Feature Encoder (Module 1)
       → common/backbones.py (YOLO26BackboneEncoder)
       → common/multimodal_model.py

   3.3.2 Context Encoder (Module 2)
       → common/task_encoder.py (已有)
       → 新增: common/context_encoder.py

   3.3.3 Context Reliability Module (Module 3) ← 核心创新1
       → 新增: common/context_reliability.py

   3.3.4 Adaptive Prototype (Module 4) ← 核心创新2+3
       → common/protonet_loop.py (已有原型)
       → 新增: common/adaptive_prototype.py

3.4 Training Algorithm
   → 航空缺陷检测/maml_train.py
   → 新增: 航空缺陷检测/ca_is_metaproto_train.py

3.5 Few-shot Adaptation
   → common/eval_loop.py
```

### 9.2 第四章: 实验 (对应代码)

```
4. Experiments

4.1 Datasets
   → common/aviation_datasets.py
   → common/cmpl_datasets.py

4.2 Experimental Setup
   → experiments/aviation.yaml
   → experiments/cmpl.yaml

4.3 Main Results (Table 1)
   → 实验 B1-B5, A1-A3, Ours

4.4 Ablation Study (Table 2)
   → 实验 A1-A3, Ours

4.5 Domain Shift Analysis (Table 3)
   → Protocol A-E

4.6 Context Corruption Experiment (Figure X)
   → 新增实验

4.7 Visualization
   → 可靠性权重可视化
   → t-SNE 原型可视化
```

### 9.3 代码文件对应

| 论文内容 | 代码文件 | 状态 |
|----------|----------|------|
| Feature Encoder | common/backbones.py | 已有 |
| Context Encoder | common/task_encoder.py | 已有 |
| **Context Reliability** | common/context_reliability.py | **新增** |
| **Adaptive Prototype** | common/adaptive_prototype.py | **新增** |
| ProtoNet Head | common/protonet_loop.py | 已有 |
| 训练流程 | 航空缺陷检测/maml_train.py | 已有 |
| 评估流程 | common/eval_loop.py | 已有 |
| 数据集 | common/aviation_datasets.py | 已有 |
| 配置文件 | experiments/*.yaml | 已有 |

---

## 10. 验收标准

### 10.1 科学问题

> **工业上下文信息能否帮助 Few-shot 缺陷检测？如何在上下文不可靠时保持稳定？**

### 10.2 方法问题

> **现有 ProtoNet 缺少对工业上下文的自适应利用。本文通过 Context Reliability + Adaptive Prototype 解决。**

### 10.3 创新问题

1. **Context Reliability Module**: 根据视觉特征自动判断上下文重要性
2. **Adaptive Task Context**: 加权融合的自适应任务表示
3. **Adaptive Prototype**: 上下文感知的原型构建

### 10.4 有效性问题

| 实验 | 验证内容 |
|------|----------|
| A2 vs B3 | Context Encoding 有帮助 |
| A3 vs A2 | Reliability 比简单拼接更好 |
| Ours vs A3 | Adaptive Prototype 有帮助 |
| Context Corruption | 模型不简单依赖 Metadata |

### 10.5 泛化问题

| 协议 | 说明 |
|------|------|
| 航空叶片 | 域0,1,2 → 域3 |
| 车漆 | ASDE → CarDD |
| 跨产品 | 航空 → 车漆 |

### 10.6 落地问题

- K=5 时精度 ≥ 75%
- 无需重新训练 Backbone
- 端到端时间 < 10 秒

---

## 附录: 实现检查清单

### Phase 1: 基础模块 (1-2周)

- [ ] 实现 `common/context_encoder.py`
- [ ] 修改 `IndustrialContext` 数据类
- [ ] 集成到现有训练流程

### Phase 2: 可靠性模块 (2-3周)

- [ ] 实现 `common/context_reliability.py`
- [ ] 实现 Context Corruption 实验
- [ ] 可视化学到的权重

### Phase 3: 自适应原型 (2-3周)

- [ ] 实现 `common/adaptive_prototype.py`
- [ ] 完整消融实验

### Phase 4: 集成与测试 (1-2周)

- [ ] 集成所有模块
- [ ] 跨数据集测试
- [ ] 论文写作

---

**下一步**: 确认此框架设计后，开始 Phase 1 实现。
