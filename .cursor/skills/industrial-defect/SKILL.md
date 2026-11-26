---
name: industrial-defect
description: >-
  通用工业缺陷检测的目录、配置和改代码约定。在本仓库写代码、改训练/评价逻辑、做数据预处理、
  改配置或看产物前必须先阅读本 SKILL.md。
  Use when working in this repo on industrial defect detection, coding, training,
  data audit, configs, data_proc, scripts, doc, or output. Always read this skill
  before writing or changing code.
---

# 工业缺陷检测

做这个项目时遵守下面的目录和改代码约定。**以后写代码都要先阅读本 SKILL.md。**

## 写代码优先级

以后写代码的时候，减少冗余逻辑。优先考虑可复用逻辑和可复用的知识，减少内存占用，以效果和运行速度优先，不做冗余计算。

落点要求：

- 先复用已有函数、通路和结论，禁止平行再写一套同义逻辑。
- 同一前向/反传能合并就合并；禁止为对称接口做双倍扫图或重复打包。
- 常驻内存只留当前步必要张量；大块中间结果算完即释，磁盘产物按轮次覆盖或只留 best/当前。
- 效果与速度冲突时，先保证协议正确，再删冗余计算；不为「看起来完整」保留无用分支。

## 目录

- `config`：配置文件。路径、预算、阈值、折数都写在这里。
- `data_proc`：数据预处理。
- `scripts`：命令入口，只负责调用，不写业务逻辑。
- `common`：可复用代码。读写配置、路径、表格放在这里。
- `doc`：文档整理。结论写在这里。实验完成后追加到 `doc/实验记录.md`。
- `output`：数据产物。
- `learn2learn-0.2.1`：元学习库。FOMAML 用 `MAML(first_order=True)`，不要另找一套实现。

数据准备入口是 `python scripts/prepare.py`。子命令 `audit`、`annotations`、`split`、`episodes`。有限图像核验用 `python scripts/prepare.py audit --verify-images`。配置是 `config/data.yaml`。表在 `output/audit`，结论在 `doc/audit_report.md` 和 `doc/physical_group_audit.md`。

## 改代码

改任何代码前先读本文件，并遵守上面的「写代码优先级」。代码保持规范化和轻量化，这是最高优先级。路径、预算、阈值、折数放进 `config`。业务逻辑放进已有的 `common`、`data_proc` 或 `training`，命令放进 `scripts`。禁止硬编码，禁止为一次查看、一次汇总或一次调试新增脚本。

需要重复使用的逻辑写成通用入口，接收配置和参数，能够服务后续实验。读取已有 `metrics.json` 或做临时计算时，在当次命令中执行，不落盘。若确实写出了一次性文件，同一轮工作结束前立即删除，不留在 `output`、仓库根目录或 `scripts`。

测试或实验开始前读取 `doc/test.md`，登记测试入口、配置、输入依赖和预计产物的相对路径。已有条目时补记本次运行。结束后核对实际产物，更新状态和处理方式。标记待统一清理的临时文件或失败产物，保留复用脚本、正式权重、冻结划分和作为后续输入的文件。

统一清理前核对 `doc/test.md` 中已获删除授权的条目、当前依赖和实际路径。清理后把对应条目标记为「已清理」，记录日期，实验结论留在 `doc/实验记录.md`。

每次修改完代码需验证对应调用链和通路，不要顾此失彼。审计这条链是 `scripts/prepare.py audit` → `data_proc/audit.py` → `common/io.py`，并核对 `config/data.yaml` 里的输出位置。同一条流程放在同一个入口里，不要为其中一步再加一个只转发调用的脚本。

## 实验记录

每次实验做完后必须追加到 `doc/实验记录.md`，不能只留终端输出或 `output` 产物。每条至少写：日期与实验名、目的、配置摘要、命令、产物路径、主要结果、结论。配置按对应 yaml 做简短描述即可，不要整文件粘贴。冒烟、正式训练、短跑对照、诊断脚本都算实验。

## Real-IAD 物理组冻结规则

v1 使用 `category + Sxxxx` 作为 Real-IAD 物理分组主键。同一产品下，OK、全部 NG 缺陷子目录、五个相机视角和 Mask 必须随组管理；不跨产品合并相同 Sxxxx。

官方基准采用 `preserve_official_test`：跨官方 train/test 的冲突组归入测试组，训练侧正常图从训练输入剔除。冲突组清单见 `output/audit/cross_split_conflicts.csv`，冻结边界见 `output/audit/group_split.json`。

所有模型训练入口必须先调用 `common.io.load_verified_group_split()`。硬性条件是 `len(train_group_ids & test_group_ids) == 0`；断言失败时禁止训练。

`group_split.json` 当前只冻结物理组边界和官方基准冲突策略，不等于外层产品留出或内层 OOF。不能按图像直接做 Group K-Fold。

## 产品族留出与 OOF

划分入口是 `python scripts/prepare.py split`，配置是 `config/split.yaml`。产物在 `output/splits`，协议结论在 `doc/split_protocol.md`。

层级固定为 Dataset → Product Family → Product/Task → Physical Group → Image/View。一个 Product/Category 是一个元学习任务；外层留出、Meta-Validation 和内层 OOF 都按 Product Family；R、Support、Query 按 Physical Group。

外层测试产品族完全封存，不参与 θ₀、Δ、源候选、选择器、描述子统计或超参数选择。Outer Train 再拆成 Meta-Train 与独立 Meta-Validation；Meta-Validation 只做模型选择、超参数和 Null 门限校准，不参与参数训练。5折 OOF 只在 Meta-Train 内运行。

所有模型训练入口还必须调用 `common.io.load_verified_research_splits()` 和 `common.io.load_verified_episode_groups()`。每折的 `source_available_mask` 必须随 Episode 和 U 标签保存，缺失 U 不得补成0；Null 始终可用。当前主候选策略为 `fold_excluded`，Meta-Validation 使用完整 Meta-Train 源库。

每个产品内的 `R`、`SUPPORT`、`QUERY_NORMAL`、`QUERY_DEFECT` 物理组池必须两两不相交。混合正常/缺陷的 Real-IAD 物理组只能整体进入 Support 或 Query-Defect，不能把其中 OK 图放进 R。

## 固定 Episode

Episode 构造入口是 `python scripts/prepare.py episodes`，配置是 `config/episode.yaml`。v1 主协议是统一二分类定位用的 1-way 5-shot：每类5个缺陷Query物理组、5个正常参照组、5个正常Query组；K按物理组计，组内全部视角一起移动。

固定 Bank 在 `output/episodes/v1`，结论在 `doc/episode_protocol.md`。Meta-Validation Pilot与OOF Training Bank必须分开；Outer Test不进入这两个Bank。Episode只能从冻结的 `episode_group_assignment.csv` 抽样，不能回原始目录随机取图。

Episode 外缺陷使用 Ignore，不当背景负样本。每条OOF Episode必须带对应折的 `source_available_mask`，Meta-Validation使用完整Meta-Train源库。

`output/splits/split_artifact_hashes.json` 和 `output/episodes/v1/episode_artifact_hashes.json` 已冻结。若协议、配置或名单需要变化，提升版本并使用新的输出目录，禁止覆盖v1依据。

## 统一检测适配与基础检测器

标注有限审计入口是 `python scripts/prepare.py annotations`，配置是 `config/baseline.yaml`，产物是 `output/audit/annotation_audit_report.csv`，结论在 `doc/annotation_audit.md`。统一检测适配写在 `data_proc/detection.py`：类别 `0` 为背景、`1` 为缺陷；Mask 按连通区域转框，正常图保留为空标注负样本，Episode 外缺陷进入 Ignore。

当前正式 B0 对照是 `output/baseline/runs/b0-v3`。专用冒烟入口已退役，结论保留在 `doc/实验记录.md`。正式训练只能使用 Meta-Train，不能对 Meta-Validation 或 Outer Test 做梯度更新。

任何检测器训练入口都必须依次校验 `load_verified_group_split()`、`load_verified_research_splits()`、`load_verified_episode_groups()` 和 `verify_frozen_episode_artifacts()`。任一物理组、产品族、Episode 边界或哈希断言失败，立即中止训练。

正式 B0 入口是 `conda run -n MAML --no-capture-output python -u scripts/train.py b0 --run-name <名称>`。训练、评价和诊断都走 `scripts/train.py` 的子命令：`b0`、`theta0`、`fomaml`、`delta`、`holdout`、`innovation1`、`diagnose-rpn`、`diagnose-roi`、`rpn-budget`。运行目录只含空锁、没有 `last.pt` 时可以重新开始；已有断点时必须换运行名或使用 `--resume`。训练按 `grad_clip_norm` 裁剪梯度；损失变成非有限值时带上图像编号并中止。

数据在配置的 `data_root`。审计只读已知索引，不递归扫盘，不读像素。要扩大读取范围时先说明再做。

## 模块贡献评判

这是后续所有流程的绝对约束。一条记录只表示「哪个旧产品的哪一块，对哪个新产品有没有贡献」。

- 目的：很多产品各自拿出一块经验。新产品可以借甲的某一块，再借乙的另一块。借的是一块，不是把某一个旧产品的整套参数搬过去。
- 有贡献就记下。没有贡献只说明这次不该选这个旧产品的这一块，不能据此宣布这一块本身失败，也不能要求它必须产生通用经验。
- 相近的产品可以更好，差得远的可以不变或变差。禁止用「是不是在全部留出产品上都变好」评价模块好坏，也禁止把这种全产品平均当成暂停或放行后续流程的门槛。
- 要回答的是：训练时没见过的产品，能不能挑到有贡献的那一块。挑选之前先把各块对各个新产品的贡献记全。缺的贡献保持缺失，不能补成 0。
- 创新点一的完整任务同时包含 Feature、Where、Box、Class。禁止拿一个产品、一块经验、贴给所有人，当作模块好坏的结论。

## 留出评价口径

单个「旧产品的某一块 → 某个新产品」仍按下面的读数记贡献，读数本身不上升为全产品门槛。

Feature 的迁移效用只在同一批 Null proposal 上计算：\(U_{\text{特征}} = L_{\text{Null}}(P_{\text{Null}}) - L_{\text{Feature}}(P_{\text{Null}})\)。原生候选下的 AP50 只作为部署效果，不与这条效用混用。报告同时给出固定候选适配前 AP50、固定候选适配后 AP50、固定候选损失效用、原生候选 AP50、候选框 Recall@50，以及一步适配造成的增益。

Where 看候选生成：Recall@50、GT 最佳 IoU、每图候选数、正常图高分候选数、FROC 和最终 AP50。不用检测损失效用判断它是否正向迁移。

底座必须是 FOMAML，入口是 `scripts/train.py fomaml`。按域打乱任务，一条任务是 1-way 5-shot。Feature、Where、Box、Class 四块都参与内层 Support 和外层 Query，不能把骨干或 RPN 冻住。每条任务开始时，用支持图按块看其他产品已经留下的参数：Feature、Box、Class 看同一批候选上的检测损失，Where 看候选覆盖。收益不大于 0 就不用；收益为正时只加「小于 1 的权重 × 这一块」，不能整段复用。查询图不参与挑选。然后内层只看 Support，外层只用 Query 损失更新共享权重，并把借来的初始化从共享权重里减回去。这条任务自己学到的四块参数差写进 `module_memory.pt`，供后面的任务按同样的加权方式继续用。日志每 10 步一条。查询损失不再下降时早停，并恢复最好一轮的权重。同一张图的掩码框只解析一次。不并行解码整条任务的原图，避免和主进程抢内存。检测器本身的 FPN 就是多尺度特征，元训练更新这些层，不再另加一套融合结构。
