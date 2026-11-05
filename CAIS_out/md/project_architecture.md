# Project Architecture Analysis

> Meta-Learning for Industrial Few-Shot Defect Detection
>
> 分析日期: 2026-07-29
>
> 项目路径: `D:\元强化学习\代码\`

---

## 1. 数据输入流程 (Data Input Pipeline)

### 1.1 数据格式

#### 航空域 (Aerospace)

```
data/航天/processed/aviation/
├── domain_0/
│   ├── 0_scratch/          # 划痕类 (class_id = 0)
│   │   ├── specimen_01/
│   │   │   ├── rgb.png     # RGB 图像
│   │   │   ├── thermal.png # 红外热成像
│   │   │   └── ultrasonic.npy  # 超声波信号 (512点)
│   │   └── ...
│   └── 3_damage/            # 损伤类 (class_id = 3)
│       └── ...
├── domain_1/
├── domain_2/
└── domain_3/               # Holdout 测试域 (试件 20-26)
```

#### 车漆域 (Automotive)

```
data/车/processed/cmpl/
├── domain_0/               # 训练域 (ASDE)
└── domain_3/               # 测试域 (CarDD holdout)
```

### 1.2 Dataset 定义

**核心类**: `AviationDefectDataset` (`common/aviation_datasets.py`)

```python
class AviationDefectDataset(Dataset):
    RGB_SIZE = (256, 256)
    THERMAL_SIZE = (320, 240)
    US_LENGTH = 512
```

**关键特性**:
- 支持多模态: RGB + Thermal + Ultrasonic
- 可选 `rgb_only` 模式（仅用 RGB）
- 域元数据从 `meta.json` 加载

**返回格式**:
```python
# __getitem__ 返回
((rgb_tensor, thermal_tensor, ultrasonic_tensor), label_tensor)
# rgb_tensor: (3, 256, 256), normalized with ImageNet stats
# thermal_tensor: (1, 320, 240), normalized [-1, 1]
# ultrasonic_tensor: (512,), raw float32 waveform
# label: torch.int64
```

### 1.3 Annotation 读取

**标签来源**: 文件夹结构隐式标注
- `0_scratch/` → label 0
- `3_damage/` → label 3
- 额外支持 `CMPL_CLASS_FOLDERS` (车漆数据集)

**域元数据** (`meta.json`):
```json
{
  "material_id": "AL7075",
  "material_params": [ρ, λ, E, Cp, T_melt, hardness, roughness, thickness, porosity, fiber_vol, resin_vol, damping],
  "process_params": [8维工艺参数],
  "defect_history": [10维缺陷历史分布],
  "fusion_prior": [0.45, 0.30, 0.25]  // RGB/Thermal/US 权重先验
}
```

### 1.4 Image Preprocessing

```python
# RGB 预处理
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip() if train else Lambda(x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Thermal 预处理
transforms.Compose([
    transforms.Resize((320, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Ultrasonic: 读取 .npy，不做归一化，padding/truncate 到 512 点
```

### 1.5 Feature Extraction

**Backbone 架构** (`common/backbones.py`):

| 模态 | Encoder | 结构 | 输出维度 |
|------|---------|------|----------|
| RGB | `YOLO26BackboneEncoder` | YOLO26 backbone + (可选CBAM) + GAP + Linear | 128 |
| Thermal | `ThermalEncoder` | Conv3×3 + BN + ReLU × 3 + GAP + Linear | 128 |
| Ultrasonic | `UltrasonicEncoder` | Conv1D × 3 + GAP + Linear | 128 |

**YOLO26 Backbone**:
```python
class YOLO26BackboneEncoder(nn.Module):
    det_model: ultralytics.YOLO.model  # YOLO26n 预训练权重
    cbam: CBAM (可选, exp10)
    pool: AdaptiveAvgPool2d(1)
    proj: Linear(in_ch, 128)
```

**CBAM 注意力** (可选):
```python
class CBAM(nn.Module):
    # Channel Attention: AvgPool + MaxPool → MLP → Sigmoid
    # Spatial Attention: Concat[Avg, Max] → Conv7×7 → Sigmoid
```

---

## 2. 模型结构 (Model Architecture)

### 2.1 整体架构图

```
                    ┌─────────────────────────────────────────────────┐
                    │           AviationDefectModel                   │
                    │                                                 │
  RGB Image  ──────►│  ┌──────────────┐                              │
                    │  │ YOLO26Encoder │────┐                        │
                    │  └──────────────┘     │                        │
                    │                        ▼                        │
  Thermal  ───────►│  ┌──────────────┐  ┌─────────────┐             │
                    │  │ThermalEncoder │─►│             │             │
                    │  └──────────────┘  │             │             │
                    │                     │  Asymmetric│──► Logits  │
  Ultrasonic ─────►│  ┌──────────────┐  │  Fusion     │             │
                    │  │USEncoder     │─►│             │             │
                    │  └──────────────┘  └─────────────┘             │
                    │                        ▲                        │
                    │  ┌──────────────────────┘                        │
                    │  │           TaskEncoder (τ)                    │
                    │  │  material_params → 64-d task embedding       │
                    │  └─────────────────────────────────────────────────┘
```

### 2.2 Backbone 模块

**YOLO26BackboneEncoder** (`backbones.py:78-152`):
- 输入: `(B, 3, 640, 640)` RGB 图像
- 过程: 移除 YOLO detection head → 提取 neck 特征 → CBAM(可选) → GAP → Linear
- 输出: `(B, 128)` 嵌入向量

**冻结策略**:
```python
def freeze_backbone(self):
    for p in self.det_model.parameters():
        p.requires_grad = False
    # proj layer 仍可训练（用于 outer loop）
```

### 2.3 Feature Extractor

三路独立编码器，输出维度统一为 128-d:

```
RGB Encoder:     YOLO26 backbone + (CBAM) + GAP + Linear(128)
Thermal Encoder: Conv(1,32) → BN → ReLU → Pool
                 Conv(32,64) → BN → ReLU → Pool
                 Conv(64,64) → BN → ReLU → GAP → Linear(128)
US Encoder:      Conv1D(1,32,stride=2) → BN → ReLU
                 Conv1D(32,64,stride=2) → BN → ReLU
                 Conv1D(64,64,stride=2) → BN → ReLU → GAP → Linear(128)
```

### 2.4 Detection Head

#### ProtoNet Head

```python
# common/protonet_loop.py
class MahalanobisHead(nn.Module):
    """Mahalanobis 距离头: d(x,y) = sqrt((x-y)^T Σ^{-1} (x-y))"""
    log_vars: nn.Parameter(ways, feat_dim)  # 每类对数方差
```

**原型构建策略**:
1. **Mean**: 简单平均 `p_c = (1/|S_c|) Σ_{s∈S_c} f(s)`
2. **Trim Mean**: 去除 10% 最高/最低值后平均（去异常值）
3. **Mean Shift**: 每类多个聚类中心（MeanShift）

**距离度量**:
- Euclidean: `d = ||q - p||²`
- Cosine: `d = 1 - cos(q, p)`
- Mahalanobis: `d = Σ_d (q_d - p_d)² / σ²_{c,d}`

#### MAML Head

内循环自适应分类器（冻结 backbone 时仅适应 fusion 层）:

```python
# MAML inner loop
for _ in range(adapt_steps):  # 默认 5 步
    loss = cross_entropy(model(support), support_labels)
    grads = grad(loss, adapt_params)
    fast_weights = theta - alpha * grads  # 快速权重更新
```

### 2.5 Meta-Learning 模块

#### Fusion 模块 (`fusion.py`)

```python
class AsymmetricFusion(nn.Module):
    """动态多模态融合"""
    cls_rgb = Linear(128, num_classes)
    cls_thermal = Linear(128, num_classes)
    cls_us = Linear(128, num_classes)
    task_to_weights = Linear(64, 3)  # 任务嵌入 → 模态权重

    def forward(self, f_rgb, f_thermal, f_us, kb_prior, task_emb):
        # 1. 各模态独立分类 logits
        logits = [cls_i(f_i) for i in (rgb, th, us)]

        # 2. 置信度权重
        conf = softmax(max(logits))

        # 3. 动态融合
        w = softmax(β * conf + α * kb_prior + γ * task_emb)
        return Σ_i w_i * logits_i
```

#### Task Encoder (`task_encoder.py`)

```python
class TaskEncoder(nn.Module):
    """域元数据 → 64维任务嵌入 τ"""
    material_net: Linear(12, 32)
    process_net: Linear(8, 16)
    history_net: Linear(10, 16)
    fusion: Linear(64, 64)
```

### 2.6 Prototype / Memory / Attention 模块

| 模块 | 实现 | 作用 |
|------|------|------|
| Prototype | `build_prototypes()` | 从 support 集构建类原型 |
| Mahalanobis 协方差 | `MahalanobisHead` | 学习类内方差，加权距离度量 |
| Task Embedding | `TaskEncoder` | 编码材料/工艺/缺陷历史为 64-d |
| Knowledge Base | `knowledge_base.py` | 材料参数先验 (AL7075, TC4, CFRP) |
| CBAM Attention | `backbones.py:46-55` | 可选通道+空间注意力 |

---

## 3. Training Pipeline

### 3.1 Episode 构造

**Task Builder** (`common/task_builder.py`):

```python
transforms = [
    FusedNWaysKShots(meta_dataset, n=ways, k=2*shots),
    LoadData(meta_dataset),
    RemapLabels(meta_dataset),      # 映射为 0...(n-1)
    ConsecutiveLabels(meta_dataset),
]
train_tasks = [Taskset(transforms) for domain in train_domains]
```

**Episode 结构**:
```
Episode:
  ├── Support Set:  N_ways × K_shots = 2 × 3 = 6 样本
  │   └── 每类: 3 个标注样本
  └── Query Set:    N_ways × K_shots = 2 × 3 = 6 样本
      └── 每类: 3 个待预测样本
```

### 3.2 Support / Query Split

```python
# common/maml_loop.py: split_support_query()
n = labels.size(0)  # 2 * ways * shots = 12
adapt_mask = np.zeros(n, dtype=bool)
adapt_mask[np.arange(shots * ways) * 2] = True  # 每类取前 shots 个
```

### 3.3 Loss 设计

#### ProtoNet Loss

```python
# common/protonet_loop.py
def protonet_episode_loss(model, support, query, domain_meta):
    s_emb = model.embed(s_rgb, s_th, s_us, domain_meta)  # (N_s, 384)
    q_emb = model.embed(q_rgb, q_th, q_us, domain_meta)  # (N_q, 384)

    prototypes = build_prototypes(s_emb, s_labels, ways)  # (ways, 384)
    logits = prototypical_logits(q_emb, prototypes)       # (N_q, ways)
    return cross_entropy(logits, q_labels)
```

#### MAML Loss

```python
# common/maml_loop.py
def meta_task_loss(learner, query, domain_meta, value_net, w_rl):
    supervised_loss = cross_entropy(learner(support), support_labels)
    rl_loss = compute_stop_rl_loss(...)  # 停止头 RL (可选)
    return supervised_loss + w_rl * rl_loss
```

**Curriculum Schedule**:
```python
# w_rl (RL 权重)
if iter <= 100:  w_rl = 0.0      # Stage 1: 仅监督
if iter <= 200:  w_rl = 0.3 * progress  # Stage 2: 渐增 RL
else:            w_rl = 0.5      # Stage 3: 满 RL

# adapt_steps (内循环步数)
if iter > 200:  adapt_steps = max(default - 2, 1)  # 后期减少步数
```

### 3.4 Optimization

**优化器配置** (`train.py:54-64`):

```python
param_groups = [
    {'params': backbone_params, 'lr': 1e-5},    # Backbone
    {'params': head_params, 'lr': 1e-3},         # Fusion + Encoders
]
optimizer = Adam(param_groups)

# ProtoNet: 无需内循环，直接前向
# MAML: clone 模型 → 内循环 → 外循环梯度更新
```

**关键超参数** (从 `cmpl.yaml` 默认值):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `meta_lr` | 0.001 | Outer loop 学习率 (冻结 backbone) |
| `backbone_meta_lr` | 1e-5 | Backbone outer loop lr |
| `head_meta_lr` | 1e-3 | Fusion/Encoder outer loop lr |
| `fast_lr` | 0.01 | Inner loop 快速学习率 (MAML) |
| `adapt_steps` | 5 | 内循环步数 |
| `meta_batch_size` | 2 | 每 iteration 采样的 task 数 |
| `iters` | 500 | 总迭代次数 |

### 3.5 Episode 增强

**光照增强** (`episode_aug.py:74-110`):
```python
def _policy_illumination_aug(rgb):
    # 1. Gamma 校正: 0.7 ~ 1.3
    # 2. 亮度偏移: ±20%
    # 3. 通道增益: 0.92 ~ 1.08 (色温变化)
    # 4. 镜面高光: 5% 概率局部加亮
```

**多尺度训练**:
```python
input_scales = [320, 480, 640]  # 随机选择输入尺度
```

---

## 4. Evaluation Pipeline

### 4.1 Few-Shot 测试方式

```python
# common/eval_loop.py
def evaluate_tasks(learner_or_model, test_tasks, ...):
    for _ in range(n_tasks):  # 默认 100 tasks
        batch = test_tasks.sample()
        support, query = split_support_query(batch, shots, ways, device)

        if MAML:
            learner = maml.clone()
            adapt_supervised(learner, support, steps=adapt_steps)
            _, acc = evaluate_query(learner, query)
        else:  # ProtoNet
            _, acc = evaluate_protonet_query(model, support, query)

        accs.append(acc)
    return mean(accs), std(accs), accs
```

### 4.2 Cross-Domain 测试方式

**域划分策略** (`domain_split_v2.py`):

```
航空:
  训练域: domain_0, domain_1, domain_2 (试件 1-19)
  测试域: domain_3 (试件 20-26 holdout)

车漆:
  训练域: domain_0 (ASDE)
  测试域: domain_3 (CarDD holdout)
```

**跨域 Episode 采样** (可选):
```python
# common/cross_domain_episode.py
build_cross_domain_pool(train_domains, ways, shots)
# 从多个训练域合并 scratch/damage 样本，增加域多样性
```

### 4.3 泛化评价指标

| 指标 | 说明 | 当前结果 |
|------|------|----------|
| 准确率 | 多 task 平均 | 航空 85.2%, 车漆 72.1% |
| 95% CI 下界 | Bootstrap | 84.2% (航空) |
| 多 seed 池化 | 3-seed × 100 pooled | 稳定性评估 |
| Per-Class AP | CarDD hard class | 30-40% (dent/scratch 标注噪声大) |

---

## 5. 关键实验结果汇总

### 5.1 主结果 (2026-07-14)

| 轨道 | 协议 | 训练→测试 | 指标 | 数值 |
|------|------|----------|------|------|
| 航空 | ProtoNet+解冻 | d0,1,2→d3 | 3-seed×100 | **85.2%** |
| 航空 | ProtoNet+贪心分域 | 同上 | 3-seed×100 | 84.7% |
| 航空 | MAML+ANIL 冻 | 同上 | 训练中 best | 53.4% |
| 车漆 | ProtoNet+解冻 | ASDE d0→CarDD d3 | 训练中 best | **72.1%** |
| 车漆 | ProtoNet 冻 | 同上 | 训练中 best | 64.4% |
| 车漆 | ANIL | CarDD d0,2→d3 | 训练中 best | 56.5% |

### 5.2 核心发现

```
核心瓶颈: ProtoNet 切换 + 解冻 backbone (非数据/非增广)
  - ANIL 冻 → 53%
  - ProtoNet 解冻 → 85%
  - 差距: 32 pp

数据扩域无效:
  - AeBAD 扩 Meta 训域 (exp05) → 无效
  - 跨域 episode 采样 (exp07) → 无效
  - 结论: 真实 ROI 域内充分 > 加同语义异常检测数据

车漆天花板 ≠ 航天天花板:
  - CarDD test 类内方差/标注噪声更大
  - 主指标 60-75% 区间合理
```

---

## 6. 代码文件映射

| 功能 | 文件路径 | 关键类/函数 |
|------|----------|-------------|
| 入口 | `航空缺陷检测/train.py` | `parse_args()`, `train()` |
| 训练 | `航空缺陷检测/maml_train.py` | `train()`, `create_model()` |
| 预训练 | `航空缺陷检测/pretrain_backbone.py` | Phase 0 多任务预训练 |
| 模型 | `common/multimodal_model.py` | `AviationDefectModel` |
| Backbone | `common/backbones.py` | `YOLO26BackboneEncoder`, `CBAM` |
| 融合 | `common/fusion.py` | `AsymmetricFusion` |
| ProtoNet | `common/protonet_loop.py` | `protonet_episode_loss`, `build_prototypes` |
| MAML | `common/maml_loop.py` | `adapt_supervised`, `meta_task_loss` |
| 评测 | `common/eval_loop.py` | `evaluate_tasks` |
| Task构建 | `common/task_builder.py` | `create_tasksets` |
| 数据集 | `common/aviation_datasets.py` | `AviationDefectDataset` |
| 域划分 | `common/domain_split_v2.py` | `build_domain_map_v2` |
| 知识库 | `common/knowledge_base.py` | `KNOWLEDGE_BASE`, `fusion_prior_tensor` |
| 任务编码 | `common/task_encoder.py` | `TaskEncoder` |
| 增强 | `common/episode_aug.py` | `augment_support_query` |
| 实验配置 | `航空缺陷检测/experiments/cmpl.yaml` | YAML 超参数配方 |

---

## 7. 流程总览图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          端到端处理流程                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Phase 0: 预训练 (可选)                                                     │
│   ┌─────────────────────────────────────┐                                  │
│   │ common/pretrain_model.py            │                                  │
│   │ · YOLO26 backbone                   │                                  │
│   │ · 多任务头 (blade/aebad/cardd/...)  │                                  │
│   │ · SupCon + CE loss                  │                                  │
│   └─────────────────────────────────────┘                                  │
│                     │                                                        │
│                     ▼                                                        │
│   ┌─────────────────────────────────────┐                                  │
│   │ scripts/convert_*.py                │                                  │
│   │ · ROI crop + 域划分                 │                                  │
│   │ · 格式统一                          │                                  │
│   └─────────────────────────────────────┘                                  │
│                     │                                                        │
│   ┌─────────────────────────────────────┐                                  │
│   │ Meta-Training Data                  │                                  │
│   │ data/{track}/processed/aviation/    │                                  │
│   │ └── domain_{0,1,2,3}/               │                                  │
│   │     ├── 0_scratch/                 │                                  │
│   │     ├── 3_damage/                   │                                  │
│   │     └── meta.json                  │                                  │
│   └─────────────────────────────────────┘                                  │
│                     │                                                        │
│   ┌─────────────────────────────────────┐                                  │
│   │ Phase 1: 元学习训练                  │                                  │
│   │ common/task_builder.py              │                                  │
│   │ · learn2learn Taskset               │                                  │
│   │ · FusedNWaysKShots                  │                                  │
│   └─────────────────────────────────────┘                                  │
│                     │                                                        │
│   ┌─────────────────────────────────────┐                                  │
│   │ AviationDefectModel                 │                                  │
│   │ ┌───────────┐ ┌───────────┐        │                                  │
│   │ │YOLO26     │ │Thermal    │        │                                  │
│   │ │Encoder    │ │Encoder    │        │                                  │
│   │ └─────┬─────┘ └─────┬─────┘        │                                  │
│   │       └──────┬──────┘              │                                  │
│   │              ▼                     │                                  │
│   │       ┌───────────────┐            │                                  │
│   │       │Asymmetric     │            │                                  │
│   │       │Fusion        │            │                                  │
│   │       └───────┬───────┘            │                                  │
│   │               ▼                    │                                  │
│   │       ┌───────────────┐            │                                  │
│   │       │Head          │            │                                  │
│   │       │(ProtoNet/    │            │                                  │
│   │       │ MAML adapt)  │            │                                  │
│   │       └───────────────┘            │                                  │
│   └─────────────────────────────────────┘                                  │
│                     │                                                        │
│   ┌─────────────────────────────────────┐                                  │
│   │ Training Loop                       │                                  │
│   │ for iteration:                      │                                  │
│   │   sample episode from domain_d      │                                  │
│   │   split support/query              │                                  │
│   │   if ProtoNet:                     │                                  │
│   │     build prototypes               │                                  │
│   │     compute euclidean/cosine dist  │                                  │
│   │     cross_entropy loss             │                                  │
│   │   if MAML:                         │                                  │
│   │     inner loop adapt (5 steps)     │                                  │
│   │     outer loop gradient update     │                                  │
│   │   evaluate on domain_3             │                                  │
│   └─────────────────────────────────────┘                                  │
│                     │                                                        │
│                     ▼                                                        │
│   ┌─────────────────────────────────────┐                                  │
│   │ Phase 2: Few-Shot Evaluation        │                                  │
│   │ evaluate_tasks(n_tasks=100)        │                                  │
│   │ · sample from held-out domain      │                                  │
│   │ · ProtoNet: prototype matching     │                                  │
│   │ · MAML: clone + adapt              │                                  │
│   │ · report mean ± std accuracy       │                                  │
│   └─────────────────────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录: 配置示例 (cmpl.yaml)

```yaml
defaults:
  track: automotive
  yolo_weights: ../车/weights/cmpl_yolo26_pretrained.pt
  ways: 2
  shots: 5
  rgb_only: true
  train_domains: "0"
  test_domain: 3
  meta_batch_size: 2
  first_order: true
  meta_algorithm: protonet
  protonet_distance: euclidean
  freeze_backbone: false
  backbone_meta_lr: 1e-5
  head_meta_lr: 1e-3
  iters: 500
  eval_tasks: 100
  eval_every: 50
```
