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

**以后写代码都要先读本 SKILL.md。**

## 每次改完代码的硬性收尾（两步，缺一不可）

1. **代码审计（轻量化）**  
   - 删死代码、失效开关、平行同义逻辑、无用测试/产物。  
   - 复用已有函数与入口；同一前向/反传能合并就合并；不为「看起来完整」留空跑分支。  
   - 常驻内存只留当前步必要张量；磁盘产物按轮次覆盖或只留 best/当前。  
   - 核对调用链仍通；相关单测通过。

2. **推送一个 Git 版本**  
   - 改完并审计后必须 `commit` + `push` 到当前跟踪分支（本仓库默认 `industrial-v1`）。  
   - 提交说明写清「为什么」；不提交 `output/`、密钥、一次性调试文件。  
   - 未推送不得视为本轮工作结束。

## 写代码优先级

减少冗余，优先可复用；效果与速度优先，不做冗余计算。路径/预算/阈值/折数进 `config`；业务进 `common` / `data_proc` / `training`；命令只放 `scripts`。禁止硬编码；禁止为一次查看新增脚本。临时计算当次执行不落盘；一次性文件当轮删掉。

## 目录与入口

| 路径 | 用途 |
| --- | --- |
| `config/` | 配置 |
| `data_proc/` | 预处理 |
| `scripts/` | 入口（无业务） |
| `common/` | 读写配置、路径、表 |
| `training/` | 训练/评价/检索迁移 |
| `doc/` | 结论；实验追加 `doc/实验记录.md` |
| `output/` | 产物（默认不入 git） |
| `learn2learn-0.2.1/` | FOMAML 用 `MAML(first_order=True)` |

- 数据准备：`python scripts/prepare.py`（`audit` / `annotations` / `split` / `episodes`）
- 训练评价：`python scripts/train.py`（`b0` / `theta0` / `fomaml` / `holdout` / `experience-bank` 等）
- 测试登记：`doc/test.md`；清理前核对该清单中已授权删除项

## 协议硬约束（摘要）

- 训练前依次校验：`load_verified_group_split()` → `load_verified_research_splits()` → `load_verified_episode_groups()` →（检测器）`verify_frozen_episode_artifacts()`；失败即停。
- Real-IAD 物理组主键 `category + Sxxxx`；`train_group_ids ∩ test_group_ids = ∅`。
- 外层留出 / Meta-Val / OOF 按 Product Family；R、Support、Query 按 Physical Group，池两两不相交。
- Episode 只从冻结 `episode_group_assignment.csv` 抽；Outer Test 不进 Meta-Val/OOF Bank；缺 U 不补 0。
- 正式训练只用 Meta-Train；Meta-Val 只做选模/超参/门限，不反传。
- 条件化迁移：相似度只召回；ψ 决策 `(z, c, ρ)`；硬门控；模块 LOO/Add-One；禁止靠扫 `null_eval_threshold` 当主改进；不重建经验库除非协议升级。
- 模块贡献按「旧产品某一块 → 新产品」记账；禁止用全留出产品平均当放行门槛；借的是块不是整套。

## 落盘

- 检测器：只保留 `last.pt` / `query_best.pt`（及必要 `metrics.json`）。
- 检索式 FOMAML：只保留当前 `experience_epoch_*`；结束后丢弃 `module_memory_parts` / 过期 epoch。
- 实验结论写 `doc/实验记录.md`，不能只留终端或 `output`。
