# 数据分组与采样覆盖（2026-11-12，收窄审计）

生成时间：2026-11-12 11:35:55 Asia/Shanghai

本轮**未打开、未解码任何图片**。未做全量 source×class×split 文件/mask 遍历。来源图像数来自 `data/new_split_v3/merge_report.json`；划分规则来自 `config/cspd/steel_sources_v3.yaml` 与 `scripts/merge_sources.py`；可行性引用已有 `output/cspd/hp_abc_20261111/diagnostics/diagnosis.json`；训练/验证覆盖来自各 run 的 `metrics.json` / `data_summary.json` / 已有 review。收窄前曾对 XSDD 做文件名与 mask 路径存在性清点（1360 图 / 0 个 mask 文件），未读像素。按 split 的独立组数除已有 diagnosis 行外**未细数**。

## 先澄清：不是用同域错例确认跨域几何

- 当前 GC10 paired holdout 是 **holdout_eval=same**：每条 episode 的 support 与 query **都来自 GC10**，不是跨来源 support-query。跨域轴是「训练来源 vs 留出来源」。
- 先前油污 query 远离正确原型、同类错成 punch，是 **同来源 holdout** 上的原型几何观察，**不能**当作「跨域同类距离大于异类」的证据，本轮也未重新算跨域距离、未加载模型。
- `rho=0`（课程 `[[0,0],[1,0]]`）只表示 **训练梯度是同源 episode**；`test_cross` 是另路评估，不进入梯度。
- 分组（GC10 文件名批次 / 其余 hist 簇）是为了防止近似图泄漏，**不等于**各来源域相同。

## 1. 来源层汇总（merge_report 图像数；组数为各类合计，跨 split 总计）

划分：`ratios: [0.6, 0.2, 0.2]`，`layout: class_source_split`，`seed: 0`；`split_rows` **按整个 group** 分配，同组不跨 split。目标比例按组，实际张数不必刚好 60/20/20（例如 S3D 168/61/100，CR7 2594/663/814）。

| 来源 | train图 | val图 | test图 | 合计图 | 组数合计(按 class 命名空间相加) | 语义类数 | 角色 A与B_GC10 / B_NEU / B_CR7 |
|---|---:|---:|---:|---:|---:|---:|---|
| CAST | 468 | 149 | 163 | 780 | 192 | 6 | 训练+同域验证 / 训练+同域验证 / 训练+同域验证 |
| CR7 | 2594 | 663 | 814 | 4071 | 175 | 7 | 训练+同域验证 / 训练+同域验证 / 留出(holdout_pool=all) |
| CSU | 185 | 64 | 57 | 306 | 31 | 3 | 训练可采; 同域val实际成功0 / 训练可采; 同域val实际成功0 / 训练可采; 同域val实际成功0 |
| GC10 | 1382 | 439 | 491 | 2312 | 798 | 9 | 留出(holdout_pool=all, same-source) / 训练+同域验证 / 训练+同域验证 |
| NEU | 1005 | 345 | 327 | 1677 | 197 | 6 | 训练+同域验证 / 留出(holdout_pool=all) / 训练+同域验证 |
| PIPE | 327 | 115 | 106 | 548 | 85 | 3 | 训练+同域验证 / 训练+同域验证 / 训练+同域验证 |
| PLATE | 266 | 108 | 106 | 480 | 59 | 3 | 训练+同域验证 / 训练+同域验证 / 训练+同域验证 |
| S3D | 168 | 61 | 100 | 329 | 50 | 2 | 训练+同域验证(饱和2-way) / 训练+同域验证(饱和2-way) / 训练+同域验证(饱和2-way) |
| XSDD | 758 | 412 | 190 | 1360 | 72 | 7 | 排除(exclude_sources+无mask) / 排除 / 排除 |

数据索引中的来源即上表 9 个。XSDD 在索引中（7 类 1360 图），但四折配置均为 `exclude_sources: [XSDD]` 且 `require_mask: true`，不进入训练/验证/留出。CAST/PIPE/PLATE/S3D 均在训练池（视留出轮换而定）。`data_summary` 与此一致：A/B_GC10 训练来源 CAST,CR7,CSU,NEU,PIPE,PLATE,S3D；B_NEU 用 GC10 换 NEU；B_CR7 用 GC10 换 CR7。饱和来源（类数<5）：CSU/PIPE/PLATE=3，S3D=2。

组定义：GC10 为文件名正则批次；其余为 hist 相似簇，**不要把目录名或簇 id 读成人名/工况**。组 id 命名空间为 source/class/id。未匹配正则的 stem 回退不是经核实的真实批次；本轮未逐条清点回退比例。

## 2. 重点类（完整逐行可行性未全表重算）

### CR7 验证集全部 7 类（已有 diagnosis，k=3/5、q=8、strict_groups）

| 类 | val图 | val组数 | k3q8 | k5q8 | 原因 |
|---|---:|---:|---|---|---|
| dent | 100 | 2 | 否 | 否 | sizes=[98,2]，无法切出合法 support/query |
| linear | 83 | 1 | 否 | 否 | 单组不能拆 |
| macular | 91 | 1 | 否 | 否 | 单组不能拆 |
| inclusion | 58 | 7 | 是 | 是 | |
| oil | 114 | 12 | 是 | 是 | |
| pit_local | 93 | 6 | 是 | 是 | |
| punch | 124 | 6 | 是 | 是 | |

因此 CR7 同域 val 只能 4-way（inclusion/oil/pit_local/punch）。修复后 B_GC10 / A 实际 CR7 val 也只出现这 4 类；dent/linear/macular 不会被采到。CR7 **train** 7 类在已有 diagnosis 中 k3/k5 均可行。B_CR7 折 CR7 是留出，不进训练/同域 val。

### CSU val（<2 个可行类）

- inclusion train/val/test 图 148/53/44，总组 16。
- oxide_scale_csu 8/2/2（总 12 图/6 组）：val 仅 2 图，k3/k5 均不可行。
- scratch 29/9/11：val 9 图 < 11，不可行。
- 故 CSU 同域 val 来源级可行类不足 2，**四面同域验证成功 episode 均为 0**。训练 split 仍采到了 CSU（资格≠验证采到）。

### GC10 留出（图像数来自 merge_report；val rolled_pit 可行性来自 diagnosis）

| 类 | train | val | test | 总组(跨split合计) | 已知可行性 |
|---|---:|---:|---:|---:|---|
| crescent | 137 | 45 | 44 | 82 | 本轮未按 split 重算组可行性 |
| fold_wrinkle | 119 | 40 | 44 | 60 | 本轮未按 split 重算组可行性 |
| inclusion | 128 | 46 | 42 | 69 | 本轮未按 split 重算组可行性 |
| oil | 118 | 37 | 49 | 52 | 本轮未按 split 重算组可行性 |
| punch | 131 | 44 | 44 | 151 | 本轮未按 split 重算组可行性 |
| rolled_pit | 21 | 5 | 5 | 26 | val split 5 图不可行(k+q)；pool_all=31 图/26 组，本轮未重算 |
| silk_spot | 396 | 103 | 152 | 73 | 本轮未按 split 重算组可行性 |
| water | 169 | 63 | 57 | 106 | 本轮未按 split 重算组可行性 |
| weld_line | 163 | 56 | 54 | 179 | 本轮未按 split 重算组可行性 |

`holdout_pool=all`：评估时把 GC10 自己的 train+val+test 合池（paired manifest 实际用了 train 706 / val 242 / test 242 条目），组仍隔离。这不是「标准只用 test split」。support 标签不训练网络。专有困难对 oil-water、punch-weld_line 在 GC10 留出时为 `unavailable_same_source_pair`（训练来源不同时共存）。

### NEU / PLATE

图像数见附表 CSV。已有 `other_val_group_infeasible` **未列入** NEU/PLATE；本轮未重扫，是否存在 k 依赖不可行 **未细数**。prefix 诊断中 NEU inclusion-pit_texture 在 GC10 留出设定下 val 两端曾记为组可行，不能外推到全部 NEU/PLATE 类。

## 3. 真实采样覆盖（日志，不是资格）

训练：每 run 100 iter × meta_batch 4 = **400** 个成功 episode；`attempts=successes`，`group_failures={}`。逐类训练覆盖：**NA**（日志无完整 class_names）。

| 来源 | A_GC10训练 | B_GC10训练 | B_NEU训练 | B_CR7训练 | A/B_GC10同域val成功 | B_NEU同域val成功 | B_CR7同域val成功 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAST | 54 | 54 | 58 | 55 | 62 | 75 | 65 |
| CR7 | 61 | 54 | 54 | 留出 | 58 | 76 | 0 |
| CSU | 56 | 62 | 53 | 57 | 0 | 0 | 0 |
| GC10 | 留出 | 留出 | 68 | 55 | 0 | 50 | 77 |
| NEU | 53 | 54 | 留出 | 66 | 66 | 0 | 71 |
| PIPE | 62 | 60 | 54 | 60 | 75 | 66 | 67 |
| PLATE | 50 | 55 | 56 | 50 | 68 | 66 | 69 |
| S3D | 64 | 61 | 57 | 57 | 71 | 67 | 51 |
| XSDD | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

同域 val：20 次评估 × 20 task，skipped 合计 0；上表是各次 `by_source.episodes` 之和（A 与 B_GC10 的 val 路径一致，成功数相同）。**不要**使用 metrics 末次 `source_successes`（A/B_GC10 里 CSU 显示 76），那是 val_sampler 未重置且穿插了 `test_cross` 的累计。CSU 同域 val 真实成功是 **0**。

未采到：XSDD（排除+无 mask）；各折留出来源不进训练/同域 val；CSU 不进同域 val（可行类不足）；CR7 val 的 dent/linear/macular（组不可行，修复后动态 4-way）。单组 support 只要与 query 组不同，在代码里是合法的。

Paired manifest（复用已有 review，seed 20261111）：20 task 全 5-way，k=3×11 / k=5×9，390 support / 800 query，全部 GC10，组重叠 0。官方 B holdout acc 与 paired 不是同一批 20 task，不可混比。

## 采样原理（按实际代码顺序，`cspd/data.py` + `scripts/train_cspd.py`）

1. 训练：`rho=0` 故 `cross_source=False`。`sample()` 按当前 split 索引取候选来源（已排除 `exclude_sources` 与 holdout），打乱后依次尝试。`avoid_sources` 让同一 meta_batch 尽量换来源。
2. `_sample_source`：先抽 **k**（3 或 5，概率 0.7/0.3），q=8。
3. 在该来源、当前 split 上过滤 **该 k 下组划分可行** 的类（`_same_source_feasible` / 整组 subset-sum）。
4. 可行类 <2 则该来源失败，试下一个来源；否则 `n_way=min(5, 可行类数)`（动态 nway）。
5. 组成类别：协议下旧 hard_pairs 钩子被置空。仅当 `same_hard_pairs=true` 才按缓冲改组成；A 控制关闭注入。验证与 holdout 不看注入。
6. 类内 `_split_by_group`：整组划分，support/query 组不相交。**单组当 support 合法**，只要 query 来自其他组。
7. 预处理（按 mask 裁切等）后前向，训练 CE。
8. 每 5 iter：同域 val（`stage_sources=False`, `allow_fallback=False`，但同一次 `sample()` 仍会在来源列表上重试），再跑 test_cross（另路）。HardPairBuffer **只用同域 val** 的 top3、连续两次确认，回流训练缓冲；**不用 holdout**。
9. 留出评估：`query_only=留出来源`，`holdout_pool=all` 合池，`holdout_eval=same` 仍是同源 episode。

详细数字：`summary.json`、`sampling_coverage.json`、`source_class_split.csv`（图像数完整列出；可行性仅填已有 diagnosis 行，其余标未细数）。未改文档主文，未训练、未加载模型。
