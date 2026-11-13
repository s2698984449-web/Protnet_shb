# 架构图（bot）

本目录是 bot 根据当前实现新绘制的架构与原理图，与 `doc/架构图/` 下既有的两套旧图分开存放。旧图未改动。这里讲架构与原理，以及已实现但本版关闭的接口；不写入实验精度、种子、折数、步数预算或结果数字，也不声称某条路径已经带来精度收益。

## 图的顺序

1. `00_模型架构与前向原理` — 主图。从左到右给出当前启用的前向路径：给定图像与掩码 → 共享冻结 ResNet18 至 layer3 → 可训练 proj3（14×14×128）→ core / ring 池化 → 支持集按类做原型、查询不聚合 → 固定 β 的余弦分类。输出是类别 logits，不是定位。
2. `01_训练与困难对反馈` — 训练闭环。查询交叉熵只更新投影；验证在训练来源上统计同源混淆，按来源保留 top 3，连续两次确认后写入缓冲，只改变后续 episode 的类别组成，不改变损失；测试 / 留出来源隔离，不回流。验证不做优化梯度。
3. `02_双区域表征细节` — 当 00 不宜写满空间处理时的必要补充。说明给定掩码如何划出 core / ring，以及「先平均、再 L2」的原型顺序。
4. `03_工况编码与条件输入` — 工况表的来源、编码、缺测掩码，以及 cond 的真实消费位置。数据侧始终组装槽位；当前课程检测器默认不把工况送入特征或 β。
5. `04_beta策略网络详解` — `EpisodeBetaPolicy`：整集状态的每一类统计量、输入维度、MLP、β 映射、梯度与启用条件。并单独标出另一套 `PolicyNet`（不属于本版课程模型）。
6. `05_完整方法模块关系` — 把双区域主路径、可选 β 策略、工况真实接点、采样与验证反馈放在一张关系图里。实线为当前启用，虚线为已实现但本版关闭或只属于其他训练模式的路径。

每张图都提供可编辑 SVG 和 PNG。页脚为 `bot · cspd`。00 仅在脚注中增加了指向 03–05 的索引，主路径框与箭头未改。

## 当前启用与关闭

启用（主路径实线）：

- 冻结 ImageNet ResNet18，取 layer3；只训练 1×1 `proj3`（256→128，14×14）。
- 支持集与查询集都必须提供**数据集给出的掩码**，掩码不是模型预测。掩码用于按区域裁切，并在特征图上划分 core / ring。
- core 为掩码落到 14×14 上的内部格子；ring 在**输入分辨率**按 16 像素膨胀后再缩放到 14×14。core 或 ring 为空时用膨胀区顶替，而不是首先退回整图均值。
- 支持集：同类向量**先平均，再 L2 归一化**得到 `p_core`、`p_ring`。查询只做逐样本 L2，不在查询集上聚合。
- 打分：`logit = cos(q_core, p_core) + 0.25 · cos(q_ring, p_ring)`。β 本版固定为 0.25。
- 损失仅为查询分类交叉熵，只回传到 `proj3`。
- 困难对：来自训练来源验证集的同源混淆；按来源 top 3；连续两次确认才入缓冲；只改训练 episode 的类别组成。采样为预热后渐增，并设上限。
- support / query 必须来自不同 `groups.json` 组；不足则放弃该次抽样，不按图随机拆分。
- 留出 / 测试来源与训练隔离，只评估，不更新缓冲。
- 工况数据接口仍然组装：默认 6 槽位 `material, surface, illuminance, light_mode, exposure, camera`，当前课程配置主要来自 `source_table`（按来源目录名填写的 0–1 码本）。缺测维 `mask=0`。

本版关闭或仅作接口（图中虚线 / 预留，不画成正在工作的数据流）：

- `EpisodeBetaPolicy`：`use_policy=false`，`freeze_policy=true`，不参与前向，也不更新。实现仍在 `DualRegionBetaDetector` 中构建。
- 跨来源课程（ρ 采样）关闭，训练为同源 episode。
- 工况在本版 DualRegion 前向中**不被消费**：`query_cond` 在 `forward` 中丢弃；`support_cond` 只在 `use_policy=true` 时以整集均值进入策略状态，即使用上也不做 FiLM 或通道调制。
- `ConditionStrategyDetector` 中的 `cond_encoder`、`FiLM`、`PolicyNet`、零件编码属于**其他训练模式**（非 curriculum），不接入当前 B 的实线前向。

设计动机是：工况描述材料、表面、光照等，策略可按支持集统计调节 ring 项权重（或在其他模式里调制特征）。图中只陈述实现与开关，不陈述精度收益，也不把旧实验步数画成结构固有条件。

## 与代码的对应

| 内容 | 位置 |
|---|---|
| layer3 / `proj3` / `_maps` | `cspd/model.py` 中 `DualRegionBetaDetector` |
| core / ring、空核用膨胀区、输入分辨率膨胀 | `cspd/model.py` 的 `_core_ring`（`dilate_px=16`，核为 `2×16+1`） |
| 掩码均值；膨胀区仍空时才退回全局均值 | `cspd/model.py` 的 `_masked_mean` |
| 原型：先 `mean` 再 `_unit_rows`；查询逐样本归一化；余弦点积 | `cspd/model.py` 的 `forward` |
| 支持 / 查询掩码均为必填，缺则报错 | `DualRegionBetaDetector.forward` |
| 掩码来自 `…/mask/<stem>.png`，并可按掩码裁切 | `cspd/data.py` 的 `_load`；`require_mask` |
| 查询标签不进入打分（forward 丢弃 `query_y`） | `cspd/model.py` 的 `forward` |
| 训练损失为查询交叉熵，只更新可训练参数 | `scripts/train_cspd.py` 的 `train_curriculum`（`F.cross_entropy`） |
| 骨干冻结、策略冻结 | `cspd/runtime.py` 的 `build_model`（`freeze_backbone`、`freeze_policy`） |
| 困难对缓冲、top 3、连续两次、只改组成 | `cspd/hard_pairs.py` 的 `HardPairBuffer`、`choose_classes`、`hard_probability` |
| 验证同源混淆写入缓冲；测试不回流 | `scripts/train_cspd.py`（`buffer.update(same_m['by_source'])`） |
| 严格分组、不同 `groups.json` 组 | `cspd/data.py`（`strict_groups`、`_split_by_group`） |
| 当前默认协议（layer3、固定 β、同源困难对） | `config/cspd/hardpair_abc.yaml` |
| 工况槽位、优先级、缺测掩码 | `cspd/condition_io.py` 的 `ConditionBuilder`、`apply_cond_mask`、`DEFAULT_SLOTS` |
| 课程采样写入 `support_cond` / `query_cond` | `cspd/data.py` 的 `CurriculumNewSplitEpisodes._load` |
| 整集 β 策略、状态维 `6+2×cond_dim`、β 映射 | `cspd/policy.py` 的 `EpisodeBetaPolicy`；`DualRegionBetaDetector.support_stats` |
| 策略是否调用、固定 β | `DualRegionBetaDetector.forward`（`use_policy` / `fixed_beta`） |
| 策略解冻与单独学习率（仅当 `use_policy`） | `scripts/train_cspd.py` 的 `train_curriculum`（`policy_unfreeze_frac`、`policy_lr_ratio`） |
| FiLM、工况编码器、零件编码 | `cspd/condition.py`；由 `ConditionStrategyDetector` 使用 |
| 每类旋钮策略 `PolicyNet` | `cspd/policy.py`；只属于 `ConditionStrategyDetector`，不是本版 β 网络 |

## 重新生成

在本目录运行（不要依赖仓库根目录的临时文件）：

```powershell
Set-Location -LiteralPath 'D:\元强化学习\代码\doc\架构图\bot'
& 'D:\miniconda3\envs\MAML\python.exe' -B generate_figures.py
```

脚本同时写出 SVG 和 PNG。Windows 优先使用微软雅黑，Linux 使用 Noto Sans CJK SC。需要 Pillow。
