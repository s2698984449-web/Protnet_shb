# 元强化学习 — 工业缺陷小样本检测（航空叶片 / 车漆）

> **双轨项目，统一底座：**  
> **航空叶片** 域 0–2 训 / 域 3 试件 20–26 holdout → ProtoNet+解冻 **85.2%** (95%CI 下界 84.2%) · exp08  
> **车漆 CMPL** ASDE 域 0 训 / CarDD 域 3 holdout → ProtoNet+解冻 **72.1%** · cmpl_protonet_unfreeze_v2  
> **一个脚本 + 一个 yaml 配方**，跨两条产线复用。

本仓库用 **MAML/ProtoNet + YOLO26 视觉骨干 + 跨域小样本评测**，回答工业 QC 的核心问题：

> 「新产线/新试件只有每类 5 张标注，**多快能上线**？」

围绕这个问题，过去两个月走了从「冻结 MAML 50% 随机」到「ProtoNet 85%」的完整路径。下面这份 README 优先讲**处理流程、思路主线、创新点与最新结果**；命令与参数表放到对应 Track 的子 README。

---

## 0. 最新主结果速查（2026-07-14）

| 轨道 | 主协议 | 训练域 → 测试域 | 主指标 | 数值 | 论文/汇报口径 |
|------|--------|-----------------|--------|------|----------------|
| 航空 | exp08 ProtoNet+解冻 | domain_0,1,2 → domain_3 | 3-seed×100 pooled | **85.2%** (CI 下界 84.2%) | 主结果 |
| 航空 | exp09 ProtoNet+贪心分域 | processed_greedy 同上 | 3-seed×100 pooled | 84.7% (CI 下界 83.5%) | partition ablation |
| 航空 | exp02 MAML+ANIL 冻 backbone | domain_0,1,2 → domain_3 | 训练中 best | 53.4% | 算法下界（已证伪） |
| 车漆 | cmpl_protonet_unfreeze_v2 | ASDE d0 → CarDD d3 | 训练中 best | **72.1%** @200 | 跨域主结果 |
| 车漆 | cmpl_protonet_v1 (冻) | ASDE d0 → CarDD d3 | 训练中 best | 64.4% | freeze 对照 |
| 车漆 | cmpl_anil_v2 (MAML 关 RL) | CarDD d0,d2 → CarDD d3 | 训练中 best | 56.5% | ANIL 路径上界 |
| 车漆 | cmpl_protonet_classsplit3 | ASDE d0 → CarDD d3 (class-split backbone) | 训练中 best | 61.2% | 替代 backbone 评估 |

**核心观察：**

- **算法层主瓶颈在 ProtoNet 切换 + 解冻 backbone**（非数据/非增广）。同一份 YOLO26 预训练 + 同一份 2w5s 评测，ANIL 冻 → 53%，ProtoNet 解冻 → 85%，差距 32 pp。
- **车漆天花板 ≠ 航天天花板**。CarDD test 上 scratch/dent 类内方差/标注噪声更大（hard class AP 30–40%），主指标按经验定在 60–75% 区间合理。
- **数据扩域证伪**：AeBAD 扩 Meta 训域（exp05）和跨域 episode 采样（exp07）均**无效**。这反过来说明"真实 ROI 域内充分"比"加同语义异常检测数据"更管用。

---

## 1. 项目目录

```text
D:\元强化学习\代码\
├── README.md                    # 本文件（总览 + 流程 + 思路 + 创新）
├── common/                      # 共享核心（双轨同用）
│   ├── backbones.py             # YOLO26 + CBAM + 热像/超声编码器
│   ├── fusion.py                # 多模态非对称融合
│   ├── multimodal_model.py      # 完整模型 + Stop 头 + RL
│   ├── pretrain_model.py        # Phase 0 多任务头 + SupCon
│   ├── supcon_loss.py           # Supervised Contrastive (Khosla 2020)
│   ├── protonet_loop.py         # ProtoNet episode 损失 / 推理
│   ├── eval_loop.py             # 多 task 评测统一入口
│   ├── task_builder.py          # learn2learn Taskset 构建
│   ├── task_encoder.py          # 域元数据 → 任务嵌入 τ
│   ├── knowledge_base.py        # 材料/缺陷先验 KB
│   ├── maml_loop.py / maml_origin.py
│   ├── domain_split_v2.py       # CSV 驱动的分域规划
│   └── project_paths.py         # 双轨路径解析 + 默认值
├── scripts/
│   ├── convert_datasets.py / convert_cmpl_datasets.py / convert_aebad_dataset.py
│   ├── prepare_aebad_pretrain.py / prepare_bladesynth_pretrain.py
│   ├── run_experiment.py        # ★ yaml 配方统一入口
│   ├── diagnose_features.py     # LogReg 探针（特征可分性）
│   ├── diagnose_hard_tasks.py   # 试件 22 hard case 定位
│   ├── plan_domain_split.py     # 贪心/CSV 分域
│   ├── check_all_data.py / check_aviation_data.py
│   └── organize_maml_checkpoints.py
├── data/
│   ├── 航天/{raw,processed,processed_greedy}/   航空（试件 holdout）
│   ├── 车/{raw,processed}/                       车漆（Track2 / Class-split 双管线）
│   └── Defect_Spectrum/                          工业泛缺陷开源集（LFS 待拉）
├── 航空缺陷检测/                 # ★ 训练入口（双轨同用）
│   ├── train.py                 # 全部训练超参数 (--track aerospace|automotive)
│   ├── pretrain_backbone.py     # Phase 0 多任务预训练
│   ├── maml_train.py / eval.py / final_eval.py
│   ├── experiments/
│   │   ├── aviation.yaml        # exp01–exp10 全配方
│   │   └── cmpl.yaml            # cmpl_p0_* / cmpl_protonet_* 全配方
│   ├── weights/ · logs/ · checkpoints/ · images/ · reports/
└── learn2learn-0.2.1/           # 源码备份
```



## 2. 端到端处理流程（一条主线 + 双轨分支）

```text
                            ┌────────────────────────────┐
                            │  原始数据 (只读)            │
                            │  · 航空: 真实YOLO 试件 1-26 │
                            │  · 车漆: CarDD/ASDE/NEU…   │
                            └─────────────┬──────────────┘
                                          ▼
                       scripts/convert_*.py (ROI crop + 域划分)
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                            ▼
   Phase 0: 多任务预训练                                  MAML 元学习数据 (域 0–3)
   common/pretrain_model.py                                common/task_builder.py
   · YOLO26 backbone (frozen/partial/full/indices)        · learn2learn Taskset
   · task-specific heads (cardd/neu/asde/blade/aebad)      · FusedNWaysKShots +
   · loss: CE / SupCon+CE                                  ·   RemapLabels
            │                                                            │
            ▼                                                            ▼
   产出 backbone 权重:                                ┌──────────────────────────┐
   · 航空: aviation_yolo26_pretrained.pt                │   Meta 训练协议选择       │
   · 车漆: cmpl_yolo26_pretrained.pt                    │   · MAML/ANIL (冻)       │
            │                                           │   · ProtoNet (解冻)      │
            └──────────────────────┬────────────────────┘   · 距离 eu/cos            │
                                   ▼                         └──────────┬───────────┘
                          common/eval_loop.py                           │
                          · 训练域 0..N 上 episode 采样                  │
                          · 测试域 3 上 100-task 评测                    ▼
                          · best.pt 落盘 + 曲线                  common/protonet_loop.py
                                   │                          或 common/maml_loop.py
                                   ▼                                    │
                          final_eval.py (3-seed)                         ▼
                          · seed 42/43/44 × 100 tasks              best.pt → 最终报告
                          · pooled acc + 95% CI 下界
```

**Phase 0 关键设计（双轨共用）**

| 决策 | 选项 | 实测最优 | 备注 |
|------|------|----------|------|
| backbone 训练模式 | frozen / partial / full / indices | **frozen**(航/车) / indices(车 classsplit) | 解冻后 CarDD 4-cls 49–57% 反复崩 |
| Loss | CE / SupCon+CE | **CE**（v4 SupCon 接口 borderline）| SupCon 改善探针但 256d→128d gap>4% |
| Mix 策略 | alternate / combined | combined（NEU 6-cls 稳）, alternate（小数据集够用）| v2 NEU 0.6 weight 防塌 |
| 任务权重 | cardd 0.8 + neu 0.15 + asde 0.05 | classsplit3_v2: neu 0.6 + tex 0.25 + obj 0.15 | NEU 权太低 6-cls head 会塌 |

**Meta 关键设计**

| 决策 | 选项 | 实测最优 | 备注 |
|------|------|----------|------|
| 算法 | MAML/ANIL / ProtoNet | **ProtoNet + 解冻** | 53.4% → 85.2% 单一最大变量 |
| Backbone 状态 | 冻 / 解冻 | **解冻** | 冻 = 算法下界；解冻 backbone_meta_lr 1e-5 |
| 内环 (MAML) | adapt_steps / fast_lr | 5 / 0.02 | grid exp04 验证 |
| 距离 (ProtoNet) | euclidean / cosine | euclidean | cosine 对极暗/极亮 ROI 略降 |
| 课程 RL | w_rl=0 / 三阶段 | **w_rl=0** | RL 在车漆无意义（无停步语义） |
| 域数 (车漆 Track2) | domain_0 (ASDE) 单域 | 同上 | 跨域任务多样性低, 难扩 |

---

## 3. 思路主线 — 我们怎么从 50% 走到 85%

> 一句话：**算法上换 ProtoNet + 解冻 backbone** 是 32 pp 的主要来源，**数据/增广/扩域** 全部证伪。

### 3.1 Phase 1 路径对比（航空 2w5s, domain_3）

```
exp02 冻 MAML/ANIL          53.4%   ← 算法下界 (特征有但不会用)
  ↓ LogReg 探针: 87.4% → 特征是好的
exp04 解冻 MAML             68.2%   ← 算法能用, 但路径次优
  ↓ 换距离度量 / 换 head → 无显著提升
exp08 ProtoNet+解冻         85.2%   ← 当前主结果, 主路径
```

**根因解释**：

- **ANIL 冻 backbone**：内环只能微调 fusion+proj，但 2w5s 信息量不足以从 128d 头选对；外环学到的方向基本是 noise。
- **MAML 解冻**：能学，但二阶近似在 4GB 上 FOMAML 已偏，更关键的是 *任务级支持集对 query 的"原型"信号没显式建模*，梯度被内环多步稀释。
- **ProtoNet 解冻**：把 "support 集 → 类原型 → 距离" 写成前向算子，无内环梯度，**支持集的几何中心直接驱动 query 决策**；外环只需学"什么样的特征适合做距离匹配"。

### 3.2 车漆路径（同一主线，跨域更难）

```
Phase 0 v1/v2 (含 CarDD)  4-cls 57.7%        ← Phase0 自身天花板
cmpl_anil_ep24 (MAML)     best 53.2% @250    ← 算法下界 (与航空同病)
cmpl_anil_v2 (关 RL)      56.5% @100         ← 路径上界
cmpl_protonet_v1 (冻)     64.4% @350         ← ProtoNet 已 +8pp
cmpl_protonet_unfreeze_v2 72.1% @200         ← 跨域主结果 v1 基础
cmpl_protonet_classsplit3 61.2% @250         ← NEU+MVTec pretrain 无帮助（2026-07-27 复盘: 钢铁/工业纹路背离车漆, 撤）
cmpl_protonet_track2      62.3%              ← CardDD-only pretrain 早期版本
cmpl_protonet_v7_meta     83.4% @100         ← cardd+carparts car-only SupCon v7（2026-07-25）
cmpl_protonet_v9_meta     84.8% @100         ← v9 SupCon + LDAM（2026-07-27 当前最优, best eval 0.8480 @ iter 100, 早停 iter 200）
```

**为什么车漆 ≤ 航空**：
- CarDD test 的 scratch/dent 类内方差大、标注含边界框误差（hard class AP 30–40%）。
- ASDE 域 0 → CarDD 域 3 域偏移大于航空域 0–2→3（同为不同产线/相机/光照，但车漆跨域更明显）。
- 训练域单一（只有 ASDE），多样性低；这是 *故意* 保留的 Track2 协议（确保和 Phase0 解耦）。

**🕒 2026-07-27 复盘 — NEU+MVTec 撤销**：
- 历史证据：NEU+MVTec pretrain → 0.612 vs cardd+carparts pretrain → 0.834，**相差 22 个百分点**
- 根因：NEU 是钢材表面、MVTec 是工业物件/纹理，与车漆 ROI 视觉分布不一致——骨干学到工业纹路偏置，反而不利于车漆细粒度分类
- 状态：✅ 已删除（v7+ active_tasks 仅含 cardd/carparts/minor_dent/syndcar/asde）

**🕒 2026-07-27 复盘 — ASDE 视觉分布已对齐**：
- `RGB_SIZE=(256, 256)` 强制 resize 解决了尺度问题
- v7_meta 0.834 已证明（ASDE domain_0 → CarDD domain_3 跨域有效）
- 不再当作"分布不匹配"理由

### 3.3 已证伪的尝试（不要重做）

| 尝试 | 实验 | 结论 |
|------|------|------|
| AeBAD 扩 Meta 训域 | exp05 | 0.681 vs 0.682，**无提升**（语义不匹配） |
| 跨域 episode 采样 | exp07 | 61.9%，**作废**（标签语义已固定） |
| 训练内 episode aug | exp06 | 训练 0.704 vs 终评 64.5%，**虚高**（eval 与 train 不一致） |
| SupCon 投影头 (v4) | cmpl_p0_v4 | LogReg 67.4% / frozen PN 53.4% gap=14% → 投影无帮助 |
| 解冻 backbone (Phase0) | cmpl_p0_v1/v2 | 4-cls 57→49%，**禁止解冻** |
| Class-split Phase0b（NEU+MVTec） | cmpl_protonet_classsplit3 | 61.2% < Track2 72.1% (2026-07-14)；61.2% < v7 83.4% (2026-07-25 复盘，**彻底撤，差 22pp**) |
| CarDD-only pretrain | cmpl_protonet_track2 | 62.3% < v7 83.4% (2026-07-25 复盘) |

### 3.4 🕒 2026-07-27 新发现的未利用资源

| 数据 | 样本数 | 位置 | 浪费原因 |
|------|--------|------|---------|
| `meta_train/roidb` (carparts + minor_dent + syndcar 合并) | **13,520** | `data/车/processed/meta_train/roidb/` | 完全闲置，未进任何阶段 |
| carparts 8 类 N-way episodes | 9084 | `data/车/processed/pretrain/carparts/` | 仅做 P0 pretrain 分类，未做 N-way 验证 |
| CarDD val ROI 作为 meta-val 池 | 2074 | `data/车/processed/pretrain/cardd_cls/val/` | 仅做 P0 pretrain val，未进 meta |

**Specimen 重叠检查（已确认无泄漏）**：
- pretrain/cardd_cls 独立 specimen：3318 (CarDD train+val 去重)
- cmpl/domain_3 独立 specimen：272 (CarDD test)
- 重叠：**0**

**待落地方向（按 ROI 排序）**：

| 方向 | 思路 | 预期收益 | 状态 |
|------|------|----------|------|
| A. meta_train/roidb 进 P1 | 把 13,520 张接入 cmpl/domain_2（与 domain_3 同分布）替换/补充 ASDE | +5~10% | 📝 想法未做 |
| B. carparts N-way scaling 验证 | 2-way scratch vs dent → 5-way → 8-way scaling 实验 | 方法论 | 📝 想法未做 |
| C. CLIP text-anchor 修正 ASDE proxy | YOLO-LA 方案：text embedding 修正 intact→dent | +2~5% | 📝 想法未做 |
| D. 跨域 episode sampling | support/query 各从不同域采样，learn domain-invariant | +2~3% | 📝 想法未做 |

详见 [`车/README.md`](车/README.md) · [`data/车/README.md`](data/车/README.md) · [`data/car_parts_damage/README.md`](data/car_parts_damage/README.md)。

---

## 4. 关键创新与设计选择

按"对最终指标的边际贡献"排序：

1. **算法协议切到 ProtoNet+解冻 backbone**（航 +32 pp / 车 +8 pp）
   - 不依赖 MAML 内环梯度；支持集直接形成类原型
   - 双 lr：backbone_meta_lr=1e-5（YOLO 微调）, head_meta_lr=1e-3（proj 跟随机）
   - 1.5 行代码改动 + 关 RL 即可，比加新结构更稳

2. **Phase 0 backbone 训练模式参数化**（`indices` 模式）
   - YOLO26n 没有 stem/stage 命名, 旧 `partial` 按 ResNet 思路取最后 2 块失效
   - 新增 `freeze_indices="0,1,2,3,4,5"` 直接按 Ultralytics `model[i]` 索引
   - classsplit Phase0 即基于此冻结前 6 块, 训 neck 后期块

3. **三轨数据切换机制**（同代码/同 yaml 配方）
   - `apply_meta_train_defaults` 在 `--track` 切换时填入 5× 域、shot、freeze、rl 默认
   - 双轨（航空/车漆）共用一个 `train.py` + 一个 `pretrain_backbone.py` + 一个 `final_eval.py`
   - 删除 2026-07-06 已合并：旧 `车漆缺陷检测/` 第二套入口不再维护

4. **可复现的实验配方（yaml + run_experiment.py）**
   - `experiments/aviation.yaml` (exp01–exp10) / `cmpl.yaml` (p0 / protonet / classsplit 全配方)
   - 一个 yaml 配方 = 一次完整可执行单元（脚本/超参/输出路径都齐）
   - 历史 `scripts/run_exp*.ps1` 归档到 `scripts/archive/`

5. **诊断基础设施**
   - `diagnose_features.py`: LogReg 探针 → 量化"特征可分性 vs Meta 实际利用率"
   - `diagnose_hard_tasks.py`: 试件 22 damage ROI 错 77% → QC 失败可剔除
   - `plan_domain_split.py` + `domain_split_v2.py`: CSV 驱动 / 贪心 / 多种子 3 种域划分
   - 这些是后续一切决策的事实基础（"53% 不是 Phase0 失败，是 Meta 没把特征用上"）

6. **Phase 0 v3 vs v4 vs classsplit：协议级 Ablation**
   - v3 (含 CarDD 80%): 主协议, 4-cls 57.5%
   - v4 (SupCon+CE, CarDD 90%): 接口 borderline (gap 14%)
   - classsplit (无车漆, NEU+MVTec): 通用工业先验, 不押 novel-class
   - 三套独立成体系, 不是迭代覆盖

7. **CBAM 注意力**（已合入, 待训评）
   - `common/backbones.py` + `train.py --rgb_cbam`
   - 8.3k 额外参数, meta-train 从头学
   - P0 优先级: 与 exp08 同协议跑, 若 +2~3pp 则方向 ② 注意力路径走通

---

## 5. 快速复现主结果

### 5.1 航空 85.2%

```powershell
cd D:\元强化学习\代码\航空缺陷检测

# 训练 (exp08 ProtoNet+解冻)
.\scripts\run_experiment.ps1 -Exp exp08_protonet -Stage train

# 3-seed 终评
.\scripts\run_experiment.ps1 -Exp exp08_eval
# → reports/final_eval_exp08_protonet_unfreeze.json
#    pooled 85.2%, CI 下界 84.2%
```

### 5.2 车漆 72.1%

```powershell
cd D:\元强化学习\代码

# 数据准备 (一次性)
python scripts\convert_cmpl_datasets.py --pretrain --meta --meta_policy track2 --check

# 训练
cd 航空缺陷检测
python scripts\run_experiment.py --config experiments\cmpl.yaml --exp cmpl_protonet_unfreeze --stage train

# 3-seed 终评
python scripts\run_experiment.py --config experiments\cmpl.yaml --exp cmpl_eval_track2
```

### 5.3 关键诊断

```powershell
# 航空 特征可分性 vs Meta 利用率
python scripts\diagnose_features.py --data_dir_path ..\data\航天\processed --out_dir ..\航空缺陷检测\reports\diagnose

# 航空 试件 22 hard case
python scripts\run_experiment.py --exp exp08_diagnose

# 车漆 3-seed final eval
python scripts\run_experiment.py --config experiments\cmpl.yaml --exp cmpl_eval_track2
```

---

## 6. 待执行 / 排队中（按对指标的预期影响排序）

| 优先级 | 实验 | 轨道 | 预期影响 | 状态 |
|--------|------|------|----------|------|
| **P0** | exp10 CBAM 训练+终评 | 航空 | +2~3pp (估 88~90%) | 代码已合入, 待训评 |
| P0 | exp08_diagnose → 试件 22 QC 流程固化 | 航空 | 附录 99.9% (剔除后) | 诊断脚本就绪 |
| P1 | 鲁棒原型 (加权均值 / Mean Shift) | 航空 | +2~5pp | `protonet_loop.py` 待改 |
| P1 | support-query cross-attention (Steel-Plate MRA) | 航空 | +1~3pp | 未启动 |
| P1 | 多种子 partition 稳定性 | 航空 | ablation, 不涨 | 3~5 组 CSV 待跑 |
| P1 | 车漆 3-seed final eval 终评 (cmpl_eval_track2) | 车漆 | 复现 72.1% | 训练已完成, 终评排队 |
| P1 | 车漆 diagnose_features.py LogReg | 车漆 | 验证 70% 阈值 | 套用航空脚本 |
| P2 | 转导式原型校准 (DAPRN 思路) | 航空 | +2~4pp | 轻量, 未设计 |
| P2 | Stratified greedy / embed 聚类 (Tier2/3) | 航空 | ablation | 多种子波动>3% 才优先 |
| P3 | 像素级原型扩展 (FSOD/FS-RSDD) | 航空 | 改任务定义 | 长期, 建议附录 |
| P3 | Class-split Base/Novel (CarDD 实例清洗) | 车漆 | 另开一条线 | 与 Track2 并行, 不替代 |

---

## 7. 不再继续的尝试（写进 README 防重做）

- **重做 AeBAD 扩域 Meta 训练** — 0.681 vs 0.682, 与 exp04 完全无差。
- **重做跨域 episode 采样** — 61.9% 远低于 baseline。
- **Phase 0 解冻 backbone** — 4-cls 49% 反复崩, 永远 frozen-only。
- **重做 episode aug** — 训练 70% 终评 64.5%, 训练-评测不一致导致虚高。
- **SupCon 投影 + frozen PN** — gap 14% borderline, 投影无帮助。

---

## 8. 总体流程（三阶段，旧版速览）

```text
原始三数据集（只读）
    │
    ▼  scripts/convert_datasets.py（ROI crop + 域划分）
    │
    ├── data/pretrain/          Phase 0：backbone 领域预训练（待实现 pretrain_backbone.py）
    └── data/aviation/          Phase 1/2：MAML 元学习
            │
            ▼  航空缺陷检测/train.py
            │
            Phase 1  冻结 YOLO26，训 fusion + proj
            Phase 2  解冻 backbone 小 lr 微调
```

| 阶段 | 数据 | 能否进 MAML 训练 | 说明 |
|------|------|------------------|------|
| Phase 0 预训练 | bladeSynth + AeBAD normal | ❌ | 只更新 backbone 权重 |
| Phase 1 MAML | 真实 domain_0~2 + AeBAD domain_4~7 | ✅ 训练 | `train_domains=0,1,2,4,5,6,7` |
| 金标准测试 | 真实 domain_3（试件 20–26） | ✅ 仅测试 | 永不参与预训练/训练 |
| bladeSynth | pretrain/  only | ❌ | 合成域 gap，不进 MAML |

---

## 9. 模型架构（细节）

```text
rgb.jpg ──► YOLO26 backbone（Detect 头已移除）
              │
              ▼ AdaptiveAvgPool + Linear
           128-d 嵌入
              │
              ▼ fusion + task_encoder（可选 τ）
           分类 logits ──► MAML 内/外环
```

| 模块 | 文件 | 说明 |
|------|------|------|
| YOLO26 编码器 | `common/backbones.py` | 加载 `yolo26n.pt`，`imgsz=640` |
| 融合 + 分类 | `common/multimodal_model.py` | RGB-only 时热像/超声填零 |
| MAML | `common/maml_loop.py` | FOMAML 默认，三阶段课程 w_rl |

**为何继续用 YOLO（无检测框架也可行）：**

- 训练阶段是 **分类**，不是检测；YOLO 仅作 **特征提取 backbone**
- 检测框 / mask 已在 **数据转换** 时用于 ROI crop，训练只读 `rgb.jpg`
- bladeSynth、AeBAD 无 YOLO 标注不影响训练

**Loader 尺寸：** Dataset 内 `Resize(256)` → backbone 内再 `Resize(640)`。

---

## 10. 统一缺陷 taxonomy

| 统一类 | 目录名 | 真实 YOLO | AeBAD | bladeSynth |
|--------|--------|-----------|-------|------------|
| 正常 | `0_normal` | — | train/good | Normal |
| 划痕 | `0_scratch` | scratch | groove | Scratch |
| 污染 | `1_contamination` | dirty/stain/dot | — | Corrosion |
| 折痕 | `2_crease` | crease | — | — |
| 损伤 | `3_damage` | damage | ablation/breakdown/fracture | Nick/Dent |

**MAML 建议：** 先 `--ways 2`（scratch vs damage）；domain_1 可试 3-way。

---

## 11. 域划分（MAML）

| domain | 来源 | 角色 |
|--------|------|------|
| 0, 1, 2 | 真实试件 1–19、2023 | meta-train |
| **3** | 真实试件 **20–26** | **meta-test（金标准）** |
| 4 | AeBAD ablation → damage | meta-train |
| 5 | AeBAD breakdown → damage | meta-train |
| 6 | AeBAD fracture → damage | meta-train |
| 7 | AeBAD groove → scratch | meta-train |

---

## 12. 环境准备

```powershell
conda activate MAML
cd D:\元强化学习\代码\航空缺陷检测
pip install -r requirements.txt
pip install learn2learn
pip install "Pillow>=10.3.0,<11"
```

**YOLO26 权重**（推荐手动下载）：

```text
航空缺陷检测/weights/yolo26n.pt
```

下载：https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

---

## 13. 数据转换（Step 1）

> **必须在项目根目录 `D:\元强化学习\代码` 下执行**（不要在 `data\` 子目录里跑）。

### 7.1 一键转换（推荐）

```powershell
conda activate MAML
cd D:\元强化学习\代码

# 四任务并行 + 进度条 + 转换后校验（约 20~40 分钟，bladeSynth 最大）
python scripts\convert_datasets.py --all --check
```

### 7.2 分步转换

```powershell
# MAML 数据：真实 + AeBAD
python scripts\convert_datasets.py --maml --check

# 预训练数据：bladeSynth + AeBAD
python scripts\convert_datasets.py --pretrain --check
```

### 7.3 常用参数

```powershell
# 顺序执行（单任务进度条在同一行刷新）
python scripts\convert_datasets.py --all --no_parallel --check

# 限制并行 worker 数
python scripts\convert_datasets.py --all --jobs 2 --check

# 降低进度打印频率（bladeSynth 12500 张）
python scripts\convert_datasets.py --all --progress_interval 200 --check

# 真实数据不 crop（legacy）
python scripts\convert_datasets.py --real --no_crop

# 仅校验、不转换
python scripts\convert_datasets.py --check
```

### 7.4 任务开关

| 开关 | 输出 |
|------|------|
| `--real` | `data/aviation/domain_0~3`（YOLO → ROI crop） |
| `--aebad` | `data/aviation/domain_4~7` |
| `--bladesynth` | `data/pretrain/bladesynth/` |
| `--aebad_pretrain` | `data/pretrain/aebad/` |
| `--maml` | `--real` + `--aebad` |
| `--pretrain` | `--bladesynth` + `--aebad_pretrain` |

### 7.5 转换后样本格式

**MAML（`data/aviation/`）：**

```text
domain_N/0_scratch/sample_id/
    rgb.jpg          # 主输入（缺陷=ROI crop，正常=整图）
    rgb_full.jpg     # 可选，整图
    roi.json         # 可选，crop 元数据
    source.json      # 可选，溯源
```

**预训练（`data/pretrain/`）：**

```text
bladesynth/train/Scratch/Scratch_001.jpg
aebad/normal/train/IMG_xxx.jpg
aebad/binary/train/{normal,defect}/*.jpg
```

### 7.6 ROI crop 默认参数

| 参数 | 默认值 |
|------|--------|
| expand_ratio | 1.8 |
| min_defect_ratio | 0.15 |
| min_short_side | 384 px |
| max_short_side | 640 px |

### 7.7 进度输出示例

```text
[real] [===============>              ] 151/291 (51.9%) 23_flip
[bladesynth] [=====>                        ] 2100/12505 (16.8%) Corrosion_2100.jpg
```

---

## 14. Phase 0 — Backbone 预训练（旧版命令）

**目标：** 共享 YOLO26 backbone，多任务联合训练（bladeSynth 5 类 + AeBAD 二分类），输出 `weights/aviation_yolo26_pretrained.pt`。

**损失：** `loss = loss_bladesynth + 0.3 × loss_aebad_binary`（`--mix_strategy combined` 每步同时算；`alternate` 交替 batch）

**单 GPU / DataParallel（多卡加速）：**

```powershell
cd D:\元强化学习\代码\航空缺陷检测
python pretrain_backbone.py --data_dir D:\元强化学习\代码\data --epochs 40 --batch_size 16 --multi_gpu auto
```

**DistributedDataParallel（推荐多卡）：**

```powershell
cd D:\元强化学习\代码\航空缺陷检测
torchrun --nproc_per_node=2 pretrain_backbone.py --data_dir D:\元强化学习\代码\data --epochs 40 --multi_gpu ddp
```

**验收：** bladeSynth val acc > 85%；AeBAD binary val acc > 90%；domain_3 KNN 优于 ImageNet 初值。

**Phase 1 加载：**

```powershell
python train.py --yolo_weights ./weights/aviation_yolo26_pretrained.pt ...
```

---

## 15. Phase 1 — MAML 训练（Step 3，旧版命令）

**冻结 YOLO26**，在扩展域上元学习：


```powershell
conda activate MAML
cd D:\元强化学习\代码\航空缺陷检测

python train.py ^
  --iters 500 ^
  --rgb_only True ^
  --data_dir_path D:\元强化学习\代码\data ^
  --freeze_backbone True ^
  --train_domains 0,1,2,4,5,6,7 ^
  --test_domain 3 ^
  --ways 2 ^
  --shots 5 ^
  --yolo_weights ./weights/yolo26n.pt
```

**Phase 0 完成后**（有领域预训练权重时）：

```powershell
python train.py ... --yolo_weights ./weights/aviation_yolo26_pretrained.pt
```

**独立评估（100 task）：**

```powershell
python eval.py ^
  --checkpoint checkpoints/AviationMAML_2w_3s_rgb_frozen_yolo26_best.pt ^
  --data_dir_path D:\元强化学习\代码\data ^
  --eval_tasks 100
```

**主指标：** `domain_3`（真实试件 20–26）100-task 平均准确率。

---

## 16. Phase 2 — 解冻微调（Step 4，旧版命令）

```powershell
python train.py ^
  --iters 150 ^
  --rgb_only True ^
  --data_dir_path D:\元强化学习\代码\data ^
  --freeze_backbone False ^
  --train_domains 0,1,2,4,5,6,7 ^
  --test_domain 3 ^
  --meta_lr 0.00001 ^
  --fast_lr 0.001 ^
  --ways 2 --shots 5
```

---

## 17. 架构调整路线（旧版）

按优先级，**先数据与预训练，再改模型**：

| 优先级 | 改动 | 状态 |
|--------|------|------|
| ★★★★★ | ROI crop 转换 | ✅ `convert_datasets.py` |
| ★★★★★ | bladeSynth 领域预训练 | ⏳ 待 `pretrain_backbone.py` |
| ★★★★ | AeBAD 扩 domain_4~7 | ✅ 转换脚本 |
| ★★★★ | FPN 多尺度池化替代 GAP | ⏳ 改 `backbones.py` |
| ★★★ | 双分支 crop + 整图融合 | ⏳ 改 `multimodal_model.py` |
| ★★ | 换 EfficientNet / DINOv2 | 最后考虑 |

**暂不建议：** 因「无检测框架」而换掉 YOLO — 转换后已是分类 + crop，与当前架构匹配。

---

## 18. 代码阅读路线

```text
scripts/convert_datasets.py   数据生产
    └── common/crop_utils.py

航空缺陷检测/train.py         入口 / 超参数
    └── maml_train.py
            ├── task_builder.py → aviation_datasets.py
            ├── multimodal_model.py → backbones.py / fusion.py
            └── maml_loop.py → maml_origin.py
```

---

## 19. 相关文档

| 文件 | 内容 |
|------|------|
| [航空缺陷检测/train.txt](航空缺陷检测/train.txt) | 训练方案对齐、调参、数据分布 |
| [航空缺陷检测/README.md](航空缺陷检测/README.md) | 全部超参数说明、FAQ |
| [data/aviation/README.md](data/aviation/README.md) | MAML 数据格式 |

---

## 20. 常见问题

**Q：`can't open file ... data\scripts\convert_datasets.py`？**  
A：当前目录在 `data\`，应 `cd D:\元强化学习\代码` 再运行。

**Q：RAR 解压 bladeSynth？**  
A：用 7-Zip：`& "C:\Program Files\7-Zip\7z.exe" x data\Dataset_bladesynth.rar -odata\`

**Q：`ways=5` 报错？**  
A：多数 domain 有效 2 类，用 `--ways 2`。

**Q：Pillow DLL 错误？**  
A：`pip install "Pillow>=10.3.0,<11"`

**Q：learn2learn 导入失败？**  
A：`pip install learn2learn`（勿直接 import 未编译的 `learn2learn-0.2.1/`）。

---

## 21. 综述 PDF 配套分析（2026-07-15）

> 对应文档：`D:\元强化学习\代码\工业缺陷小样本检测综述.pdf`  
> 目的：把"综述结论 ↔ 项目实验 ↔ 后续创新方向" 三者串起来；后续有新思路直接在本节追加，**实验结果回写到对应 train log**（`training_logs.txt\车\training.log` / `航天\trail.txt`）。

### 21.1 PDF 内容核验（事实修正）

| 位置 | PDF 原文 | 修正 | 来源 |
|------|----------|------|------|
| §5.2 | "MAML 换到 ProtoNet 提升 17.3 pp" | **31.8 pp**（53.4% → 85.2%）| exp02 vs exp08 |
| §6.3 | "Track2 62.5%" | **72.1%**（cmpl_protonet_unfreeze_v2 @200）| 车/training.log 2026-07-09 |
| §4.4 | CarDD SOTA scratch AP=34% / dent AP=40% | 偏低，CarDD 论文 SOTA ~50–55% | 需复核 |
| §2.5.1 | ProtoNet NEU 5w5s ~82-86% | 视方法 80-90% | NEU 论文 |
| §2.5.1 | MVTec-FS 5w5s ~70-78% | 偏低，主流 85-90% | MVTec-FS benchmark |
| §9 | "SimCLR+ProtoNet IJSRM 2025 89.4%" | ⚠ 待复核（这条文献无法在公开检索中直接确认）| 待查 |

**必须改的两条**（影响主结论）：
- §5.2 17.3 → 31.8 pp（你主结果最大论据）
- §6.3 62.5% → 72.1%（避免自降车漆档位）

### 21.2 框架级小建议

- §6 "你的研究在这个脉络中的位置"——综述论文通常不放自指段落，建议挪到 Introduction
- §7 方向 1 伪代码 DANN+ProtoNet 已过时（2023 后被对比解耦 / 因果解耦替代），建议换成对比解耦
- §7 方向 4 "多模态非对称融合"（你自己提的方案）挪到论文方法章节更合适
- 建议新增"瓶颈 5：仿真→真实域差"，对接 exp05 实证

### 21.3 当前瓶颈（结合项目现状）

1. **跨域泛化（PDF §5.1，最痛）**
   - 航空：domain_3 pooled 85.2% 但试件 22 错 77%（域内异质）
   - 车漆：ASDE→CarDD 72.1%，距离工业上线 >90% 还差
   - 算力约束：DANN / DAPRN 训一轮 ~8h，单 ablation 跑不完

2. **任务多样性（PDF §5.2）** — 航空已绕过，车漆卡 72% 的隐藏原因
   - 航空：5 个独立 task，ProtoNet+解冻绕开
   - 车漆 Track2：domain_0 (ASDE) **单域训练**，任务多样性比航空还差

3. **细粒度区分（PDF §5.3，车漆根因）**
   - CarDD scratch/dent AP 都低，本质是类内方差大 + 标注框含边界误差
   - LogReg 64.5% 印证"特征空间本身没分得开"，是**数据问题**不是算法问题

4. **效率 vs 精度（PDF §5.4，部署绕不开）**
   - 当前 ProtoNet+ResNet 推理 ~10ms → 可上线
   - 扩到 5-way 5-shot → 30-50ms → 边缘勉强

5. **仿真→真实域差（PDF 漏）**
   - BladeSynth 训 Phase0 OK，但 AeBAD 进 Meta 训域（exp05）无提升
   - 建议在综述加"仿真域差"小节作为第 5 个瓶颈

### 21.4 可落地创新方向（按 ROI 排序）

#### P0 · 单点 ROI 最高（4GB / 1 周内可出结果）

**创新 1：对比解耦 ProtoNet**（替换 PDF §7 方向 1 的 DANN 方案）
- 思路：ProtoNet 原型加一层「任务条件化对比投影」——在 head 投影后用 SupConLoss（temperature=0.07），内环也走它
- 接入点：`common/protonet_loop.py` +20 行；`common/supcon_loss.py` 已写好
- 预期：跨域 +3~5 pp，训成本 +15%
- 数据：零；算力：4GB 可跑

**创新 2：车漆伪域构造 DRA**（PDF 完全没提，但车漆卡 72% 的隐藏解）
- 思路：ASDE 域内做 Domain Randomized Augmentation 构造 K=4 个伪域（亮度/对比度/HSV 抖动组合），episode 跨伪域采样
- 接入点：`common/task_builder.py` 加 `pseudo_domains` 参数；`scripts/convert_cmpl_datasets.py` 加 `--pseudo_domain_k`
- 预期：车漆 +5~8 pp（接近 80%）
- 数据：零（只用现有 ASDE）；算力：与基线持平

#### P1 · 中 ROI（1-2 周）

**创新 3：轻量 CBAM + 多尺度原型融合**（合并 PDF §7 方向 2、方向 3）
- 思路：CBAM 在 YOLO neck 已实现；升级为 ProtoNet 内部对 support 做 3 尺度池化 → 3 个原型 → 距离加权融合
- 接入点：`common/protonet_loop.py` 重写 `prototypical_logits`
- 预期：航空 +2~3 pp，车漆 +1~2 pp
- 工业 FSD 文献基础：MPFnet 2023 / DCAP 2024

**创新 4：知识蒸馏小模型**（PDF §5.4 部署解）
- 思路：ProtoNet 85.2% 当 teacher，蒸馏一个 YOLOv8n 检测头（5ms 推理 + 5 类缺陷）
- 接入点：新建 `common/distill.py`；`experiments/aviation.yaml` 加 `exp11_distill`
- 预期：保留 80~85% 精度，部署到边缘
- 算力：蒸馏 4GB 训 2h，推理 2GB

#### P2 · 长期（不建议现在做）

**创新 5：基础模型适配**（MVREC / NexViTAD 思路）
- 不建议先做：AlphaCLIP 推理 ~200ms，4GB 装不下
- 若必须：用 AlphaCLIP 的图像编码器（冻结）替换 YOLO
- 预期 +5~8 pp，工程 1 个月起步

**创新 6：跨域域适应**（DAPRN / TADAM）
- 不建议先做：先证 P0 / P1 再说

### 21.5 4 周落地路径（4GB / 单卡预算）

| 周次 | 实验 | 目标 | 创新点 |
|------|------|------|--------|
| W1 | `exp10_protonet_cbam` 训练+终评 | 验证 CBAM backbone 端 | 创新 3 backbone 部分 |
| W1 | `cmpl_diagnose_features.py` | 套用，测车漆 LogReg 上界 | 量化诊断 |
| W2 | `exp11_protonet_multiscale` 实现+训练 | 多尺度原型 | 创新 3 prototype 部分 |
| W2 | `cmpl_pseudo_domain_v1` 实现+训练 | 车漆伪域 | 创新 2 |
| W3 | `exp12_contrastive_protonet` | 对比解耦 ProtoNet | 创新 1 |
| W3 | `cmpl_eval_track2` 3-seed | 车漆主结果复现 | - |
| W4 | 综合 ablation + 论文图表 | 整合 5 个新实验 | - |

**关键判断**：P0 创新 1 + 创新 2 是单点 ROI 最高的两个，**W2 末尾**就能拿到车漆 80% / 航空 88% 的初步数据。

### 21.6 思路速记（随时追加）

> 在这里记零散想法，新条目按时间倒序追加：

- (2026-07-15) **车漆 LogReg 优先测**：先跑 diagnose_features 知道特征上界，再决定 Phase0 是否还要改。当前已知 Phase0 v3 CarDD 4-cls 57.5%，LogReg 探针可能在 60-70% 之间 —— 若 <70%，优先改 Phase0；若 ≥70%，优先改 Meta。
- (2026-07-15) **CBAM 训不动的话**：exp10 单次训 2-3h，若 5-shot eval < exp08，说明 CBAM 随机初始化在少样本下没足够梯度，**改用 SupCon 预热 CBAM**（先训 5 epoch SupCon+CE，再接 ProtoNet）。
- (2026-07-15) **多尺度原型失败迹象**：若航空 exp11 < exp08，怀疑 YOLO 三个尺度特征图分辨率差距过大（20×20 vs 80×80），prototype 距离不可比，**改用归一化每尺度再 concat**。
- (2026-07-15) **DRA 伪域 K 值**：先用 K=4，若车漆 best 仍在 200 iter 前早停，K 过大导致任务难，**退回 K=2**。
- (2026-07-15) **论文叙事角度**：先讲"53% 不是 Phase0 失败，是 Meta 没用上特征"（PDF §6.1 自己贡献），再讲"ProtoNet+解冻是解"（32 pp 单一最大变量），最后讲"未来瓶颈在跨域与细粒度"。这条主线比直接讲算法创新更打动人。

### 21.7 PDF 修正 TODO（更新完后回写到本节）

- [ ] §5.2 17.3 → 31.8 pp
- [ ] §6.3 62.5% → 72.1%
- [ ] §7 方向 1 伪代码换对比解耦
- [ ] 新增"瓶颈 5：仿真→真实域差"
- [ ] §6 自指段落挪出综述
- [ ] §7 方向 4（非对称融合）挪到论文 method
- [ ] §9 "SimCLR+ProtoNet 89.4%" 文献全网复核（已加待查标记）

---
