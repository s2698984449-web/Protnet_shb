# Phase 4 — 实验设计

> **CA-IS-MetaProto 实验规划**
>
> 基于当前项目已有实验 (exp08-exp18) 和 Baseline，设计 CA-IS-MetaProto 的完整实验体系

---

## 1. 当前项目已有实验结果（Baseline 基础）

### 1.1 航空叶片数据集实验

| Exp | 方法 | 配置 | 关键结果 | 状态 |
|-----|------|------|---------|------|
| exp08 | ProtoNet | 旧 pretrain, unfreeze | **85.2%** (3-seed) | 已有 |
| exp11 | ProtoNet | v5 supcon backbone | 98.3% eval (单seed) | 已有 |
| exp12 | ProtoNet | + trim_mean | 待 3-seed | 待复现 |
| exp13 | ProtoNet | + mahalanobis | 待 3-seed | 待复现 |
| exp14 | ProtoNet | + trim_mean + mahal | 待 3-seed | 待复现 |
| exp16 | ProtoNet | + illumination aug | 训练中 | 2026-08-11 |
| exp17 | ProtoNet | + multi-scale | 待跑 | backlog |
| exp18 | ProtoNet | + illum + multi-scale | 待跑 | backlog |

### 1.2 车漆数据集实验

| Exp | 方法 | 配置 | 结果 |
|-----|------|------|------|
| cmpl_protonet_track2 | ProtoNet | ASDE→CarDD | 待复现 |
| cmpl_protonet_classsplit3 | ProtoNet | NEU预训练 | 待复现 |

### 1.3 当前模型能力

```
已有能力:
  ✓ Feature Encoder (YOLO26 + 可选CBAM)
  ✓ Task Encoder (材料/工艺 → 64-d embedding)
  ✓ ProtoNet / MAML 算法
  ✓ 域划分策略 (random / greedy / metadata-based)
  ✓ 原型构建: mean / trim_mean / mean_shift
  ✓ 距离度量: euclidean / cosine / mahalanobis
  ✓ Episode Augmentation

缺失能力 (本文新增):
  ✗ Context Reliability Module (核心创新)
  ✗ Adaptive Prototype (核心创新)
  ✗ Context Corruption 实验
```

---

## 2. CA-IS-MetaProto 实验总表

### 2.1 实验编号映射

```
# ===== Baseline 系列 (验证现有方法) =====
B1: 普通监督学习 (域内训练/测试)
B2: ProtoNet (域内)
B3: ProtoNet (跨域)
B4: MAML (跨域)
B5: ProtoNet + 域泛化

# ===== Ablation 系列 (验证本文创新) =====
A1: ProtoNet + Shift Modeling (工业域变化建模)
A2: A1 + Context Encoding
A3: A2 + Context Reliability (固定权重)
A4: A3 + Context Reliability (可学习)
A5: A4 + Adaptive Prototype

# ===== Ours =====
O1: CA-IS-MetaProto (完整方法)
```

### 2.2 消融实验矩阵

| 实验 | Shift Modeling | Context Encoding | Reliability (固定) | Reliability (学习) | Adaptive Proto |
|------|:--------------:|:---------------:|:-----------------:|:-----------------:|:-------------:|
| B3 (ProtoNet baseline) | ✗ | ✗ | ✗ | ✗ | ✗ |
| A1 | ✓ | ✗ | ✗ | ✗ | ✗ |
| A2 | ✓ | ✓ | ✗ | ✗ | ✗ |
| A3 | ✓ | ✓ | ✓ | ✗ | ✗ |
| A4 | ✓ | ✓ | ✗ | ✓ | ✗ |
| A5 | ✓ | ✓ | ✗ | ✓ | ✓ |
| **O1 (Ours)** | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 3. RQ1：工业域变化影响验证

**研究问题**: 工业域变化是否真的影响 Few-shot Defect Detection？

### 3.1 Domain Shift 协议设计

```
Protocol A: Random Split (随机划分)
  训练: 70% 样本 (所有域混合)
  测试: 30% 样本 (所有域混合)
  → 基准性能

Protocol B: Component Hold-out
  训练: domain 0, 1
  测试: domain 2 (新 Component)
  → 验证 Component Shift 影响

Protocol C: Material Hold-out
  训练: AL7075
  测试: TC4
  → 验证 Material Shift 影响 (需数据集支持)

Protocol D: Environment/Lighting Hold-out
  训练: 正常光照
  测试: 异常光照 / 不同相机
  → 验证 Lighting Shift 影响

Protocol E: Compound Shift (最关键)
  训练: domain 0, 1 (Component A, Material A)
  测试: domain 2 (Component B, Material B, 新光照)
  → 综合泛化能力
```

### 3.2 实验配置

```yaml
# 实验 RQ1-D: Compound Shift 核心协议
domain_shift_type: compound
train_domains: "0,1"
test_domain: 2
n_way: 2
k_shot: 5
n_query: 15
eval_tasks: 100
seeds: "42,43,44"
```

### 3.3 预期结果

| Protocol | ProtoNet Baseline | CA-IS-MetaProto | Gap |
|----------|------------------|-----------------|-----|
| A (Random) | ~95% | ~96% | +1% |
| B (Component) | ~85% | ~88% | +3% |
| C (Material) | ~80% | ~85% | +5% |
| D (Lighting) | ~82% | ~87% | +5% |
| E (Compound) | ~70% | ~78% | **+8%** |

---

## 4. RQ2：Industrial Context 有效性

**研究问题**: Industrial Context 是否能够帮助 Few-shot Adaptation？

### 4.1 实验设计

```
比较:
  B3: ProtoNet (无 Context)
  A2: ProtoNet + Context Encoding (简单拼接)

Context 输入:
  - Component ID (one-hot)
  - Material descriptor (12-d 物理参数)
  - Process parameters (8-d)
  - Lighting condition (categorical)
  - Defect history (10-d)
  → 总计 ~30-d descriptor
```

### 4.2 预期结果

```
RQ2: Context Encoding 有效性

实验: 航空叶片 Compound Shift (domain 0,1 → 2

  ProtoNet (B3):      70.0% ± 8.0%
  ProtoNet + Context (A2):  73.5% ± 7.5%
  
  Δ = +3.5% → Context 确实有帮助，但需要进一步优化
```

### 4.3 论文呈现

```
图 X: ProtoNet with/without Industrial Context on Compound Shift
  - x-axis: K-shot (1, 3, 5, 10)
  - y-axis: Accuracy
  - Two lines: ProtoNet vs ProtoNet+Context
```

---

## 5. RQ3：Context 重要性差异

**研究问题**: 不同 Context 是否具有不同的重要性？

### 5.1 实验设计

```
单独注入不同 Context，验证权重差异:

A2-1: Only Component
A2-2: Only Material
A2-3: Only Lighting
A2-4: Only Defect History
A2-5: All Contexts (equal weight)
A2-6: All Contexts (learned weight)
```

### 5.2 预期观察

```
RQ3: 不同工业上下文的贡献差异

实验设置: 航空叶片 5-shot, Compound Shift

Context Type       |  Importance Weight (learned) |
-------------------|--------------------------------|
Component (部件)   |  0.35                          |
Material (材料)    |  0.30                          |
Lighting (光照)    |  0.20                          |
Defect History     |  0.15                          |

结论: 不同工业上下文的重要性确实不同
```

### 5.3 论文呈现

```
图 Y: Learned Context Importance Weights
  - Stacked bar chart 或 radar chart
  - 展示不同数据集上学到的权重差异
```

---

## 6. RQ4：Context Reliability 有效性

**研究问题**: Context Reliability 是否有效？

### 6.1 实验设计

```
比较三种 Context 融合方式:

A2: Simple Concat (直接拼接所有 Context)
A3: Fixed Weight (固定权重 0.3)
A4: Learned Reliability (可学习权重)
```

### 6.2 Context Corruption 实验（关键！）

```
目的: 验证模型不是简单依赖 Metadata

实验设置:
  正常: 输入正确的 Material=AL7075
  损坏: 输入错误的 Material=CFRP (对航空叶片来说不合理)

比较:
  方法1: Simple Concat → 性能崩溃 (错误 Context 干扰)
  方法2: Fixed Weight → 性能下降但可接受
  方法3: Learned Reliability → 学会降权错误 Context
```

### 6.3 预期结果

```
RQ4: Context Reliability 实验

                          | Clean Context | Corrupted Context | Δ |
--------------------------|---------------|-------------------|-----|
ProtoNet (B3)             | 70.0%         | 70.0%             | 0%  |
Simple Concat (A2)        | 73.5%         | 65.0%             | -8.5% |
Fixed Weight (A3)         | 75.0%         | 70.0%             | -5.0% |
Learned Reliability (A4)  | 76.0%         | 74.5%             | -1.5% |

结论: Learned Reliability 对错误 Context 更鲁棒
```

### 6.4 论文呈现

```
图 Z: Context Corruption Experiment
  - x-axis: Corruption level (0%, 25%, 50%, 75%, 100%)
  - y-axis: Accuracy
  - Three lines: Simple Concat, Fixed Weight, Learned Reliability
  
核心信息: 当 Context 错误时，Learned Reliability 保持稳定
```

---

## 7. RQ5：Adaptive Prototype 有效性

**研究问题**: Context-conditioned Adaptive Prototype 是否有效？

### 7.1 实验设计

```
比较三种原型构建方式:

ProtoNet: P = Mean(Support Features)
A5: P_adaptive = AdaptivePrototype(Support Features, τ_adaptive)
```

### 7.2 预期结果

```
RQ5: Adaptive Prototype 实验

实验: 航空叶片 Compound Shift (domain 0,1 → 2)

K-shot | Mean Prototype | Adaptive Prototype | Δ |
--------|---------------|-------------------|-----|
1-shot  | 55.0%         | 62.0%             | +7.0% |
3-shot  | 68.0%         | 73.0%             | +5.0% |
5-shot  | 70.0%         | 76.0%             | +6.0% |
10-shot | 78.0%         | 81.0%             | +3.0% |

结论: Adaptive Prototype 在低样本时收益更大
```

### 7.3 论文呈现

```
图 W: K-shot Performance Comparison
  - x-axis: K-shot (1, 3, 5, 10)
  - y-axis: Accuracy
  - Two lines: Mean Prototype vs Adaptive Prototype
  - Shaded region: improvement
```

---

## 8. RQ6：跨域泛化能力

**研究问题**: 面对未知工业域是否具有更好的泛化能力？

### 8.1 跨数据集实验设计

```
数据集 A → 数据集 B 泛化:

1. 航空叶片 → 车漆 (跨产品)
   训练: 航空叶片 domain 0,1
   测试: 车漆 CarDD domain 3
   
2. 车漆 ASDE → CarDD (同产品不同域)
   训练: ASDE domain 0
   测试: CarDD domain 3
   
3. 多域混合 → 新域
   训练: 航空叶片 + 车漆 ASDE + NEU-DET
   测试: 任意新工业数据集
```

### 8.2 预期结果

```
RQ6: 跨域泛化实验

Train → Test              | ProtoNet | CA-IS-MetaProto | Δ |
--------------------------|----------|-----------------|-----|
航空 0,1 → 2 (seen)       | 85.0%    | 88.0%           | +3% |
车漆 ASDE → CarDD         | 72.0%    | 76.0%           | +4% |
航空 → 车漆 (cross-prod)  | 55.0%    | 62.0%           | +7% |
混合 → 新工业数据集        | 50.0%    | 58.0%           | +8% |
```

---

## 9. 完整实验执行顺序

### Phase 0: 复现现有 Baseline (已有结果验证)

```
优先级 P0 (立即执行):

P0-1: 复现 exp14 (trim_mean + mahalanobis) 3-seed 结果
      目标: 确认现有最佳方法在 3-seed 下的性能
      命令: exp14_eval

P0-2: 复现车漆 cmpl_protonet_track2 结果
      目标: 建立第二个数据集的 Baseline
      命令: cmpl_protonet_track2 + eval
```

### Phase 1: Ablation 核心实验

```
优先级 P1 (在 P0 完成后):

P1-1: 实现 Context Encoding (A2)
      修改: common/context_encoder.py (新增)
      实验: A2 vs B3 (ProtoNet baseline)
      预期: +3.5% (RQ2)
      
P1-2: 实现 Context Reliability (A4)
      修改: common/context_reliability.py (新增)
      实验: A4 vs A2
      预期: +2.5% (RQ4)
      
P1-3: 实现 Adaptive Prototype (A5)
      修改: common/adaptive_prototype.py (新增)
      实验: A5 vs A4
      预期: +3.0% (RQ5)
```

### Phase 2: Context Corruption 实验

```
优先级 P2 (在 P1 完成后):

P2-1: Context Corruption 消融
      实验: A2 vs A4 在 corrupted context 下
      目标: 验证 Reliability 模块对错误 Context 的鲁棒性
      
P2-2: 分析 Learned Reliability 权重
      可视化: 不同任务下学到的权重分布
```

### Phase 3: Domain Shift 协议实验

```
优先级 P3 (在 P1-2 完成后):

P3-1: Component Hold-out (Protocol B)
P3-2: Material Hold-out (Protocol C)
P3-3: Lighting Hold-out (Protocol D)
P3-4: Compound Shift (Protocol E) ← 最关键
```

### Phase 4: 跨数据集泛化

```
优先级 P4 (最后阶段):

P4-1: 航空 → 车漆 跨产品泛化
P4-2: 多域混合预训练 → 新数据集
```

---

## 10. 实验配置模板

### 10.1 Baseline 配置 (B3: ProtoNet)

```yaml
# configs/baselines/protonet_compound_shift.yaml
experiment_name: protonet_compound_shift_baseline

# 模型
model:
  meta_algorithm: protonet
  backbone: yolo26n
  embed_dim: 128
  freeze_backbone: false
  use_cbam: false

# 任务
task:
  domain_shift_type: compound
  train_domains: "0,1"
  test_domain: 2
  n_way: 2
  k_shot: 5
  n_query: 15

# 上下文 (不使用)
context:
  use_context: false

# 评估
evaluation:
  seeds: "42,43,44"
  eval_tasks: 100
  eval_every: 50
```

### 10.2 CA-IS-MetaProto 配置 (O1)

```yaml
# configs/methods/ca_is_metaproto.yaml
experiment_name: ca_is_metaproto_official

# 模型
model:
  meta_algorithm: protonet
  backbone: yolo26n
  embed_dim: 128
  freeze_backbone: false
  use_cbam: false

# 上下文模块 (新增)
context:
  use_context: true
  context_dim: 30
  context_hidden: 64
  use_reliability: true
  reliability_hidden: 128
  reliability_temperature: 0.1

# 原型模块 (新增)
prototype:
  use_adaptive: true
  adaptive_hidden: 128
  prototype_type: trim_mean
  distance: mahalanobis

# 任务
task:
  domain_shift_type: compound
  train_domains: "0,1"
  test_domain: 2
  n_way: 2
  k_shot: 5
  n_query: 15

# 评估
evaluation:
  seeds: "42,43,44"
  eval_tasks: 100
  eval_every: 50
```

---

## 11. 关键实验时间线

```
Week 1-2: P0 Baseline 复现
  ├─ 复现 exp14 3-seed 结果
  └─ 复现车漆 Baseline

Week 3-4: P1 Context 模块
  ├─ 实现 Context Encoding
  ├─ 实验 A2 vs B3
  └─ 实现 Context Reliability

Week 5-6: P1 Adaptive Prototype
  ├─ 实现 Adaptive Prototype
  ├─ 实验 A5 vs A4
  └─ RQ2-RQ5 主要结果

Week 7-8: P2 Context Corruption
  ├─ 实现 Context Corruption 实验
  ├─ 验证 Reliability 有效性
  └─ 分析 Learned Weights

Week 9-10: P3 Domain Shift 协议
  ├─ Protocol B-E 实验
  └─ RQ1 结果

Week 11-12: P4 跨数据集泛化
  ├─ 航空 → 车漆
  └─ 准备论文实验章节

Week 13-14: 论文写作
  ├─ Methods 章节
  ├─ Experiments 章节
  └─ 完善图表
```

---

## 12. 工业落地性实验

### 12.1 必须记录指标

```
部署指标:
  - Support Shot 数量: 1, 3, 5, 10
  - Adaptation Time: 秒 (few-shot 构建原型)
  - Inference Time: 毫秒/样本
  - GPU Memory: MB
  - Model Size: MB
  - 是否需要重新训练: Yes/No
  - 新产品上线成本: 时间估计
```

### 12.2 新产品上线模拟

```
场景: 新产线上线, 只给 K 个标注样本

流程:
  1. 采集 K-shot 支持样本
  2. 提取特征 (Feature Encoder)
  3. 构建原型 (Adaptive Prototype)
  4. 在线推理 (Distance Matching)
  5. 评估精度

目标:
  K=5 时, 精度 ≥ 75%
  端到端时间 < 10 秒
  无需重新训练 Backbone
```

---

## 13. 论文实验表格模板

### 13.1 主结果表

```
表 X: Few-shot Defect Detection on [Dataset Name]

                1-shot    3-shot    5-shot    10-shot
--------------------------------------------------------
Supervised      -         -         -         XX.X%
ProtoNet       XX.X%     XX.X%     XX.X%     XX.X%
MAML           XX.X%     XX.X%     XX.X%     XX.X%
ProtoNet+Meta  XX.X%     XX.X%     XX.X%     XX.X%
CA-IS-MetaProto(ours)  XX.X%     XX.X%     XX.X%     XX.X%

最佳结果加粗, std 在下标
```

### 13.2 消融表

```
表 Y: Ablation Study on Compound Shift

Model                    | 5-shot Acc | Δ vs Baseline
--------------------------------------------------------
ProtoNet (Baseline)      | 70.0%      | -
+ Shift Modeling (A1)    | 72.0%      | +2.0%
+ Context Encoding (A2)  | 73.5%      | +3.5%
+ Reliability (A4)       | 76.0%      | +6.0%
+ Adaptive Prototype (A5)| 78.0%      | +8.0%
CA-IS-MetaProto (Ours)  | 78.5%      | +8.5%
```

### 13.3 Domain Shift 泛化表

```
表 Z: Cross-Domain Generalization

Protocol         | ProtoNet | MAML | CA-IS-MetaProto
-----------------------------------------------------
Random Split     | XX.X%    | XX.X% | XX.X%
Component        | XX.X%    | XX.X% | XX.X%
Material         | XX.X%    | XX.X% | XX.X%
Lighting         | XX.X%    | XX.X% | XX.X%
Compound Shift   | XX.X%    | XX.X% | XX.X%
```

---

## 14. 下一步行动

1. **立即执行**: 复现 exp14 3-seed 结果 (P0-1)
2. **下一步**: 实现 Context Encoding 模块 (P1-1)
3. **确认**: 是否有其他需要调整的实验设计？

---

**附录: 已有实验 vs 论文实验对照**

| 已有 Exp | 论文中的作用 | 需要新增 |
|---------|-------------|---------|
| exp08 | ProtoNet Baseline (B3) | - |
| exp12 | Trim Mean 消融 | - |
| exp13 | Mahalanobis 消融 | - |
| exp14 | 最佳 ProtoNet | 3-seed 复现 |
| exp16 | Illumination Aug | 独立实验 |
| - | Context Encoding | **新增 A2** |
| - | Context Reliability | **新增 A4** |
| - | Adaptive Prototype | **新增 A5** |
| - | Context Corruption | **新增 RQ4** |
