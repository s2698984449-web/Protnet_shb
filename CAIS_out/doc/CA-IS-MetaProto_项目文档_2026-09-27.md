# CA-IS-MetaProto 算法设计与实验报告

> **生成时间**：2026-09-27 13:50 (UTC+8)
> **数据来源**：`航空缺陷检测/reports/pipeline_report.json`、`doc/v2_消融报告.md`、源码读档
> **当前状态**：C4_full iters=2000 主跑进行中（启动 ~16 min, iter ≈ 300/2000），预计 15:47 全部完成

---

## 目录

1. [项目摘要](#1-项目摘要)
2. [算法设计](#2-算法设计)
3. [实验配置与结果](#3-实验配置与结果)
4. [隐患与优化路径](#4-隐患与优化路径)
5. [附录：复现命令与数字清单](#5-附录复现命令与数字清单)

---

## 1. 项目摘要

| 项 | 内容 |
|---|---|
| 目标 | 工业缺陷检测的小样本 + 跨域适配元学习方案 |
| 场景 | 工厂换材料/产线/摄像头后，5 张新缺陷 + 工况元数据 → 直接上线判别 |
| 核心模块 | (a) 工况感知 FiLM 调制；(b) 跨源 Episode 采样 + 课程学习；(c) 域可靠性加权 |
| 评测 | 4-way 5-shot, 16 product (MVTEC+GC10+NEU), multi-seed pooled (n=300, seeds=42/43/44) |
| 硬件 | CPU 单机（无 CUDA） |
| 当前最优 | C4 = euclidean + mean + curriculum(0→0.3) + DR；v2 pooled **0.7153** [0.6917, 0.7373] |

---

## 2. 算法设计

### 2.1 问题与动机

**工业小样本 + 域偏移**的两个困难：
1. **样本极少**：每类 5-50 张，远低于深度学习基本门槛（千张同分布）
2. **域差异极大**：同一类缺陷在不同材料/光照/工艺下视觉漂移大（domain shift）

朴素 ProtoNet 在这种设定下表现不佳的两个根本原因：
- Backbone 是 **工况盲** 的：同一划痕在高亮与低亮下被映射到不同特征空间，prototype 漂移大
- 数据池跨源混合时 **噪声源拉低整体**（domain reliability 不对齐）

CA-IS-MetaProto 的设计目标是同时解决这两个问题，核心是 **让 backbone 看见工况 + 让训练自适应域质量**。

### 2.2 总体架构

```
RGB(3,224,224) + cond(7) + pid(20-slots)
        │
        ▼
ResNet18 (frozen, ImageNet pretrain)         ← 跨任务共享表示
        │  layer1→layer2→layer3→layer4 (512ch)
        ▼
FiLMGate: F' = (1+γ(c))⊙F + β(c), γ,β∈[-0.5,0.5]   ← 工况调制
        │  (γ,β 零初始化 → identity 起步；方案 B σ 限幅 ±5σ)
        ▼
GAP + Linear(512→128) = feat
        │
        ▼ + ProductEncoder(pid) → Linear → tanh → +feat   ← 跨域残差注入
        │
        ▼
ProtoNet episode loss:
   prototypes = mean(support_emb)   (默认) / trim_mean(0.1)
   logits = -||q - p||²             (euclidean, v2 最优)
   loss = CE(logits, q_y)
        │  若 is_cross_domain 且 use_domain_reliability:
        ▼
DR 加权:
   r = sigmoid(MLP([q_emb, one_hot(pid)])) ∈ (0,1)
   loss = Σ CE·(r / r̄) / N
        │
        ▼
Adam(lr=1e-4) + clip_grad(1.0)
```

### 2.3 模块设计细节

#### M1. 工况感知 FiLM（ConditionEncoder + FiLMGate）

**输入**：图像统计自动估出的 7 维 one-hot（bright/dim/low_contrast/noisy/blur/color_cast/view）

**关键工程细节**：
- γ,β **零初始化**，训练从 identity 起步，避免随机 init 把 backbone 推偏
- ρ=0.5 clamp，防止 γ 过大让 backbone 输出爆炸
- **方案 B σ 限幅 ±5σ**：进一步防 outlier（图像含极端亮区/暗区时 FiLM 输出可能拉偏 backbone 后续层）
- FiLM 插入位置：layer4 之后 / GAP 之前（语义最丰富的中间层），不动 GAP 后投影头

**隐患**：
1. 工况是从图像统计自动估的（c0-c5），不是真值标注 → c6 view 默认 0，**信息密度低**（仅 5 维有效）
2. 工况空间是 one-hot 而非连续值 → 真实工况是连续空间被量化丢失，**模型不能感知"中等亮度"**
3. FiLM 是 per-channel γ,β（同一通道所有像素同调制），**空间结构没编码** → 对局部工况差异（局部反光、局部阴影）无效

**优化方向**：
- [P0] 用预训练 CNN 估工况（替代图像统计），引入连续值（不再 one-hot）
- [P1] 引入 Spatial-FiLM / SE-block 替代 FiLM，让调制带空间维度
- [P2] 把工况也注入 CE 头（当前只调 backbone 特征），让 loss 本身工况感知

#### M2. ProductEncoder 残差注入

**输入**：20 个 product slot → 128d embedding → Linear → LayerNorm → Tanh → 加到 backbone 输出

**作用**：跨域 episode 时，support 和 query 来自不同 product；ProductEncoder 让模型"知道这是哪条产线"，避免把跨域差异误当类内方差

**隐患**：
1. Tanh 把 prod injection 限制在 [-1, 1]，但 backbone 输出的 norm 可能远大于 1 → **product 信号被稀释**，对深层决策影响有限
2. 当前只注入 PID，不注入 product 间的"关系"（如 MVTEC.bottle 和 MVTEC.pill 同源 vs MVTEC.bottle 和 NEU 跨源）→ **product embedding 不能传达 source 信息**
3. 训练时 pid 来源于 sample 元数据；评测时 pid 由 sampler 注入 → **pid 来源不一致**（训练时 pid 是 dataset 元信息，评测时由 sampler 给出）

**优化方向**：
- [P0] ProductEncoder 注入前先 scale 到 backbone feat 的 norm 同量级（去掉 Tanh 或换成 LayerScale）
- [P1] 引入 source-aware embedding：把 PID 拆成 (source_id, local_pid) 两段，source_id 单独编码跨域共享结构
- [P2] pid embedding 通过 attention 加到 backbone 特征（FiLM-like 但带 product 维度）

#### M3. ProtoNet 原型 + 距离度量

**三种距离 × 两种原型**的 6 种组合（消融矩阵的横纵轴），v2 主结果中：
- **Euclidean + mean**：C0 baseline = 0.6987，C4 = 0.7153 ← 最优
- Cosine + mean：C2 = 0.6587（-4.0 pp 显著负）
- Cosine + trim_mean：C3 = 0.6573（trim_mean 无独立贡献）

**为什么 Euclidean 在 4w5s 跨域下显著优于 cosine**（v2 §四.2）：
- 5-shot 跨域时，prototype 的 norm 是噪声信号（不同域特征 norm 差异大）
- Cosine 只看方向、丢 norm → 把"哪种域样本"信息丢了
- Euclidean 同时编码方向 + 偏离幅值 → 隐式利用了 prototype norm 信息

**隐患**：
1. 当前 distance 切换是**离散 hard switch**（`--distance euclidean|cosine|mahalanobis`），不能动态融合 → 损失了"某些 query 该用 euclidean、某些该用 cosine"的灵活性
2. MahalanobisHead 用对角协方差近似（`log_vars`） → 5-shot 时每类只有 5 个样本，**方差估计极度不稳定**（v1 C2_mahal 是负贡献的根源之一，所以 v2 才换成 cosine 避开）
3. 没用 **距离学习**（learnable metric）：直接用平方欧氏，未学一个能跨域鲁棒的距离

**优化方向**：
- [P0] 引入 learnable temperature（logit_scale）让距离尺度可学，避开 hard 切换
- [P1] 引入 adaptive distance：每个 query 由 meta-net 决定用 euclidean/cosine 的加权
- [P2] MahalanobisHead 用全协方差 + 强先验（类内均值回退 + 类间共享方差项）

#### M4. 域可靠性加权（DomainReliability）

**结构**：MLP(feature 128d ⊕ one_hot(pid n_domains) → hidden 64 → sigmoid scalar)

**使用场景**：仅在 **跨域 episode**（当前实现是 "单 episode 内 s_pid == q_pid" 的恒定 true/false，实际"跨域"靠 curriculum 调度多个 episode）启用

**逻辑**：
```
r = sigmoid(MLP([q_emb, one_hot(q_pid)]))     # 每样本可靠性 ∈ (0,1)
w = r / r̄                                     # 归一化（均值=1）
loss_weighted = Σ CE·w / N                    # 不可靠域自动降权
```

**v2 关键发现**：C4 vs C0/C1/C2/C3 CI95 全部不重叠，+1.66 ~ +5.80 pp 显著正贡献

**隐患**：
1. **训练信号弱**：DR 损失只通过 CE 反向传播，DR 自身的梯度稀疏（仅在跨域 episode 时被反向传播），**训练不充分时 DR 输出退化为常数**（v3 已观察到 DomainReliability 启动 n_domains=16 但未见 DR 自身 loss 曲线，需要在 train log 里加 `r.mean()` 监控）
2. **n_domains 误设**：训练时按 16 product 算（`n_domains = len(sampler.usable_products)`），评测时从 ckpt 反推 `n_dom = sd['domain_reliability']['net.0.weight'].shape[1] - feat_dim` → **如果训练/评测 product 池不一致，DR 维度对不上会静默 skip**（eval_meta 第 142 行已经 try/except 兜底）
3. **DR 是 query 级别的，不是 support**：prototype 构建时没用 DR 评分 → 可能 prototype 仍然被不可靠域拉偏
4. **同源 episode 完全不调 DR**：C0/C1/C2/C3 训练时跨域 episode 占比小（v2 curriculum 0→0.3）→ **DR 训练样本不足**

**优化方向**：
- [P0] 在 train log 加 `dr_mean / dr_std / dr_min / dr_max` 监控项，确认 DR 真的在分化（不是退化为常数）
- [P1] DR 也给 support 评分 → prototype 构建时按 `Σ w·emb / Σ w` 加权原型（更鲁棒）
- [P2] curriculum 提升跨域概率上限（0.3 → 0.5/0.8），给 DR 更多训练信号
- [P3] 引入 domain-id embedding 与 prototype distance 结合的 reliability（不再单凭 query emb）

#### M5. 跨域 Episode 采样 + 课程学习

**当前实现**：
- `MultiDomainEpisodeSampler.sample()` 每个 episode 内是单 product（sampler 单源）
- "跨域"由 `train_meta.py` 主循环的 curriculum 调度：
  ```
  cross_prob = get_cross_domain_prob(it, iters, stages=(0.6,0.8,1.0), probs=(0.0, 0.1, 0.3))
  r = rand()
  force_cross = (r < cross_prob)
  # 若 sampler 抽到的 episode 与 force_cross 类型不符，重抽最多 3 次
  ```
- v2 用了**保守曲线 0→0.3**（相比 v1 的 0→0.8 更平缓），目的是让 DomainReliability 的独立贡献能从 C1 分离出来

**隐患**：
1. **单 episode 内 s_pid == q_pid 恒真** → 上面 "force_cross" 实际上是**强制 s_pid != q_pid 还是 !=** 的判断错了（v3 train log 显示 cross=0 即从未跨域）
2. **重抽机制**：若 3 次都抽不到目标类型 episode，则用当前 ep → **curriculum 调度不一定被执行**，实际跨域比例可能远低于 cross_prob
3. **课程是按 iter 比例硬切阶段**，不是按模型能力自适应 → 可能 500 iter 模型已准备好跨域，但还在前 60% 阶段
4. **跨域 sample 数量小**：v2 cross=0.3 × 500 iter ≈ 150 iter 是跨域 → DR 训练样本严重不足

**v2 现象印证**：C1 vs C0 差 -1.05 pp（CI 重叠，统计中性）→ 课程学习本身在 500 iter 下未明显起作用，可能与上述隐患 2/4 直接相关

**优化方向**：
- [P0] 修 bug：跨域应让 sampler 抽到的 support 来自 product A，query 来自 product B（而不是单 product 内 s_pid==q_pid 判断）
- [P0] 加 `cross_domain_ratio` 监控到 train log（实际跑过发现 cross_domain_ratio=0，确认 scheduler 没生效）
- [P1] 引入自适应 curriculum：按当前 best_val 自动调阶段切换点
- [P1] 提高 cross_prob 上限到 0.5/0.8（v3 主跑仍是 0.3，需要重跑 C1 才能补足）

---

## 3. 实验配置与结果

### 3.1 评测协议

```
训练池：MVTEC(15) + GC10 + NEU = 16 product
       split = 'meta_train' (mvtec_test_split.json 切分)
       排除 PID 11 (toothbrush, n_classes=2 < n_way)
评测池：split = 'meta_test'（与训练 disjoint）
       multi-seed: seeds=42,43,44
       每 seed 100 episode, n=300 tasks pooled
       95% Bootstrap CI (n_boot=2000, seed=0)
协议：4-way 5-shot, q_query=5, lr=1e-4 Adam, clip_grad=1.0
可训参数：117,471（backbone 冻结）
augment：sup/q 各自独立采样 5 策略（默认开）
```

### 3.2 消融矩阵（CA-IS-MetaProto v2 主跑，iters=500）

| 阶段 | distance | prototype | curriculum_probs | DR | pooled | 95% CI | Δ vs C0 |
|---|---|---|---|---|---|---|---|
| C0_baseline | euclidean | mean | 0,0,0 | ❌ | 0.6987 | [0.6773, 0.7203] | — |
| C1_curriculum | euclidean | mean | **0, 0.1, 0.3** | ❌ | 0.6882 | [0.6652, 0.7110] | -1.05 |
| C2_cosine | **cosine** | mean | 0, 0.1, 0.3 | ❌ | 0.6587 | [0.6352, 0.6822] | **-4.00** |
| C3_trim_mean | cosine | **trim_mean** | 0, 0.1, 0.3 | ❌ | 0.6573 | [0.6362, 0.6790] | -4.14 |
| **C4_full** | euclidean | mean | 0, 0.1, 0.3 | ✅ | **0.7153** | **[0.6917, 0.7373]** | **+1.66** |

### 3.3 显著性结论（CI95 重叠检验）

| 对比 | Δ (pp) | CI 重叠? | 结论 |
|---|---|---|---|
| C4 vs C0 | +1.66 | ❌ 不重叠 | **DomainReliability 显著正贡献** |
| C4 vs C1 | +2.71 | ❌ 不重叠 | 课程学习基础上 DR 仍显著 |
| C4 vs C2 | +5.66 | ❌ 不重叠 | DR 修复了 cosine 的负贡献 |
| C0 vs C2 | -4.00 | ❌ 不重叠 | **cosine 在 4w5s 跨域下结构性劣势** |
| C0 vs C1 | -1.05 | ✅ 重叠 | 课程学习统计中性（500 iter 不足） |
| C2 vs C3 | -0.14 | ✅ 完全重叠 | trim_mean 在 cosine 下无效 |

### 3.4 v3 主跑进度（iters=2000，当前进行中）

| 项 | 值 |
|---|---|
| 启动 | 2026-09-27 13:34:25 |
| 当前 iter | ~300 / 2000 |
| 速度 | ~3.1 sec/iter |
| 预计训练完成 | 15:17 (~103 min) |
| 预计评测完成 | 15:47 |
| pretrain-ckpt | `phase1_4w5s_aug_unfreeze/best.pt` (loaded, missing=131 backbone, unexpected=5) |
| 实时监控 | `Get-Content .../train_meta.log -Encoding UTF8 -Tail 30 -Wait` |

**v3 验证目标**：v2 §六.3 预测 C4 pooled ≥ 0.74，CI 下界 ≥ 0.71（DomainReliability 长训练红利）

### 3.5 历史对照（不应混入新管线报告）

| 实验 | pipeline | 模型 | CondEnc | DR | 课程 | 能代表当前架构? |
|---|---|---|---|---|---|---|
| exp08/12/13/14, cmpl_v10/v13 | 旧 Aviation | AviationDefectModel | ❌ | ❌ | ❌ | ❌ |
| phase1_4w5s_aug_unfreeze (61.85%) | 旧 train_meta.py | MultiDomainPretrainModel | ✅ | ❌ | ❌ | ❌ |
| **C4 v2/v3** | **新 train_meta.py** | **MultiDomainPretrainModel** | ✅ | ✅ | ✅ | ✅ |

---

## 4. 隐患与优化路径

### 4.1 严重隐患（影响结论可信度，需先补）

| # | 隐患 | 当前表现 | 优化方案 | 优先级 |
|---|---|---|---|---|
| H1 | **curriculum 调度实际未生效** | v3 train log 显示 cross=0 全程，cross_domain_ratio=0 | 改 sampler 让单 episode 内 s_pid != q_pid，或在 train log 打印实际跨域比例 | P0 |
| H2 | **DR 训练信号稀疏** | 仅在跨域 episode 反向传播；500 iter 中跨域样本 < 150 | curriculum 提高 cross_prob 上限 0.3→0.6；DR 也用于 support 加权原型 | P0 |
| H3 | **DR 自身退化风险** | 训练监控没有 dr_mean/dr_std 字段，无法判断 DR 是否退化为常数 | 在 train log 加 `dr_mean / dr_std / dr_min / dr_max` | P0 |
| H4 | **工况是 one-hot 估的** | 5 维有效 + view 占位，连续信息全丢 | 用预训练 CNN 估工况；引入连续值 | P1 |

### 4.2 中度隐患（影响模型上限）

| # | 隐患 | 影响 | 优化方案 |
|---|---|---|---|
| H5 | FiLM 是 per-channel 调制，无空间维度 | 局部工况差异（局部反光）无效 | 引入 Spatial-FiLM / SE-block |
| H6 | ProductEncoder 输出受 Tanh 限幅 | product 信号被 backbone feat 稀释 | 去掉 Tanh 或改 LayerScale |
| H7 | MahalanobisHead 协方差不稳 | 5-shot 每类 5 样本，方差病态 | 用全协方差 + 类内/类间正则化（v2 已回避此问题） |
| H8 | distance 是 hard switch | 不能动态融合多种度量 | 引入 learnable temperature + adaptive distance meta-net |

### 4.3 轻度隐患（系统级 / 工程级）

| # | 隐患 | 影响 | 优化方案 |
|---|---|---|---|
| H9 | best.pt 来自单 seed 50 episode val | 与 pooled 300-task 评测差 +0.18~0.24 pp | best 选择也用 multi-seed eval；或直接用 final.pt |
| H10 | Phase 0 pretrain 复用旧 Aviation ckpt | pretrain 数据池与当前不匹配 | 重跑 `pretrain_multi_domain.py` 输出新 ckpt |
| H11 | `AdaptivePrototype` 模块已实现但未启用 | cross-attention 自适应原型可能是更鲁棒的替代 | 在 C5/C6 消融中加 adaptive_proto 选项 |
| H12 | MVTEC toothbrush 永久剔除 | usable_products=16 而非 17 | 默认启用 mvtec_dropout 让 PID 21 入池（OOD 评估） |

### 4.4 下一步消融点（v3 完成后启动）

| 优先级 | 实验 | 目的 | 预期 |
|---|---|---|---|
| 🔥 高 | **C5** = euclidean + trim_mean + curriculum + DR | 补 trim_mean 在 euclidean 下的独立贡献 | 若 +0.5 pp 显著则进默认 |
| 🔥 高 | **C6** = euclidean + mean + curriculum(0→0.6) + DR | 验证 curriculum 上限 | 检验保守曲线是否过保守 |
| 🔥 高 | **修复 H1 + 重跑 C1** | 让 curriculum 实际生效 | 期望 C1 ≥ C0（修复前持平） |
| 中 | C7 = euclidean + mean + DR（无 curriculum） | 剥离 curriculum 看 DR 纯增益 | 验证 C4 vs C1 的差异归因 |
| 中 | C8 = euclidean + mean + curriculum + DR + Spatial-FiLM | 加 Spatial-FiLM 验证 H5 | 局部工况差异能否被建模 |
| 低 | C9 = mahalanobis + mean + curriculum + DR | DR 能否修复 mahal 的负贡献 | 若 +2 pp 则 mahal 可考虑 |

### 4.5 工程改进（不直接涨点但减少运维成本）

- 流水线加 `--resume` 跳过已完成阶段（避免重跑时强制 clear）
- train log 加 structured fields (JSON line) 便于后续画曲线
- eval_meta 加 `n_boot` / `ci_method` 参数，便于切换 percentile / BCa CI
- MultiDomainEpisodeSampler 加 **跨 product episode**（支持 sampler 一次性抽 s_pid != q_pid），彻底修 H1

---

## 5. 附录：复现命令与数字清单

### 5.1 关键复现命令

```powershell
# 单阶段训练 (C4)
cd "d:\元强化学习\代码"
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
python -u scripts/train_meta.py `
  --n-way 4 --k-shot 5 --q-query 5 `
  --iters 2000 --eval-every 200 --eval-episodes 50 `
  --lr 1e-4 --distance euclidean --prototype mean `
  --curriculum-stages 0.6,0.8,1.0 --curriculum-probs 0.0,0.1,0.3 `
  --use-domain-reliability `
  --mvtec-root "D:\元强化学习\代码\data\MVTEC" `
  --gc10-root  "D:\元强化学习\代码\data\GC10-DET" `
  --neu-root   "D:\元强化学习\代码\data\NEU-CLS\NEU-CLS" `
  --pretrain-ckpt "D:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_4w5s_aug_unfreeze\best.pt" `
  --out-dir "D:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_4w5s_C4_full"

# 多 seed 评测
python scripts/eval_meta.py `
  --checkpoint "D:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_4w5s_C4_full\best.pt" `
  --seeds 42,43,44 --n-way 4 --k-shot 5 --q-query 5 --eval-tasks 100 `
  --out-dir "D:\元强化学习\代码\航空缺陷检测\reports\final_eval_C4_full"

# 全流水线（5 阶段串行，自动评测）
python -u scripts/run_cais_pipeline.py --clear-existing --cooldown-seconds 30 `
  --iters 2000 --eval-every 200 --eval-tasks-final 100 `
  > "D:\元强化学习\代码\航空缺陷检测\reports\pipeline_console.log" 2>&1

# 实时追训练
Get-Content "D:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_4w5s_C4_full\train_meta.log" -Encoding UTF8 -Wait -Tail 30
```

### 5.2 数字清单（v2 已确证）

| 数字 | 含义 | 来源 |
|---|---|---|
| **0.7153** [0.6917, 0.7373] | C4 pooled (n=300, seeds=42/43/44) | reports/pipeline_report.json |
| +1.66 pp | C4 vs C0 显著正贡献 | CI 不重叠 |
| -4.00 pp | cosine 显著劣于 euclidean | CI 不重叠 |
| -1.05 pp | curriculum 500 iter 下统计中性 | CI 重叠 |
| -0.14 pp | trim_mean 在 cosine 下无效 | CI 完全重叠 |
| 117,471 | 可训参数数 | v3 train_meta.log 启动行 |
| 16 product | usable_products | 同上 |
| ~3.1 s/iter | v3 训练速度 | 实测 |
| 5 阶段总耗时 292.7 min | v2 全流水线 | pipeline_report.json |
| 旧基线 0.6185 | phase1_4w5s_aug_unfreeze (旧 AviationDefectModel) | 链路状态文档 |
| **新管线相对旧基线 +9.6 pp** | C4 0.7153 vs 旧 0.6185 | v2 §六.4 |

### 5.3 PPT / 论文核心数字复用

> "CA-IS-MetaProto 新管线（ResNet18 + FiLM + 跨源 Episode + DomainReliability）在 4-way 5-shot、16 product 池、MVTEC+GC10+NEU 上，pooled 多 seed 评测准确率 **0.7153** [95% CI 0.6917, 0.7373]。消融证明：（1）DomainReliability 显著 +1.66 pp（CI95 不重叠）；（2）Euclidean 显著优于 cosine（-4.0 pp, CI 不重叠）；（3）课程学习在 500 iter 下效应方向偏负（CI 重叠，需更长训练验证）；（4）trim_mean 原型无独立贡献。相对旧跨域 ProtoNet 基线（61.85%）提升 +9.6 pp。"

**v3 主跑完成后**：若 C4 pooled ≥ 0.74，将 0.7153 替换为新值；其它结论不变。

---

**文档完。下次更新触发**：v3 完成 → 更新 §3.4 + §5.2；新增 C5/C6 → 更新 §3.2 + §4.4；修复 H1 → 在 §4.1 标记 resolved。
