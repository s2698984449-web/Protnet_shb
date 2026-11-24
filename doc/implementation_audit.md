# 检索式经验库 S0+S1 审计

日期：2026-11-24。这一节只记录建库。检索接入 FOMAML 的结果在文末，完整训练仍未启动。`config/fomaml.yaml` 没有改。

## 修改的文件

| 文件 | 作用 |
| --- | --- |
| `training/experience_bank.py` | 新增。特征缓存、任务签名、PCA、球面 K-means、一步梯度、TIES/加权均值、落盘和检索 |
| `config/retrieval_fomaml.yaml` | 补建库用的种子、签名版本、输出目录和已知同物体族。G/M/K、PCA、Top-K、γ 没有改 |
| `scripts/train.py` | 新增子命令 `experience-bank`。`fomaml` 和 `holdout` 的参数没动 |
| `tests/test_experience_bank.py` | 分组、签名、存储和“排除自己经验组”的单测 |

核心入口：`build_experience_bank`、`save_experience_bank`、`load_experience_bank`、`retrieve_module_experience`。

聚类、TIES、跨组加权和动态 Top-K 仍调用 `training/retrieval_memory.py`。检测损失仍调用 `live_detection_loss` 的同一次反向，四块梯度来自同一次反向。内层步长 0.01 读自 `config/fomaml.yaml`。Δ = −0.01 × Support 平均梯度，再按 A_m 做大范数裁剪。这一步没有 `optimizer.step`。

## 实际跑过的测试

MAML 环境，2026-11-24：

- `tests/test_retrieval_memory.py`：12 项通过。
- `tests/test_experience_bank.py`：7 项通过。覆盖同物体同组、Support 组去重、签名维度、目标族拒绝、FP16 恢复，以及排除自身经验组后检索为空。

真实建库命令：

```text
python -B -u scripts/train.py experience-bank --theta0 output/meta_transfer/theta0/fold_0/fold0-detloss-v1/last.pt --run-name fold0-bank-min --max-families 2 --tasks-per-family 2
```

解释器是 `D:\miniconda3\envs\MAML\python.exe`。退出码 0，用时 53.8 秒。产物在 `output/meta_transfer/experience_bank/fold_0/fold0-bank-min/`。

这是最小建库，不是 27 个源族的全库。源族是排序后的前两个：`mvtec:bottle`、`mvtec:cable`。每族冻结 Bank 里有 3 条 Episode，这次按参数各取 2 条，4 条全部成功，没有跳过。

## 建库读数

| 项目 | 结果 |
| --- | --- |
| 签名维度 | Appearance 256，Morphology 262，Difference 256 |
| 请求簇数 / 实际非空簇 | K=8；三种 Key 都是 4 个非空簇 |
| 卡片 | 16 张，全部有效；每张 1 个成员，因此走成员不足 5 的加权均值，没有进入 TIES |
| 存储 | 16 张都是 dense FP16。相对误差最大 2.3e-4。稀疏索引在这个密度下更大，没有选用 |
| 经验组 | bottle=0，cable=1。没有把两者并成同一组 |
| 近重复清单 | 外观余弦 0.959，`merged=0` |
| θ | 建库前后参数和 buffer 的最大绝对差是 0。四条任务梯度写入的 `theta_sha256` 都是 `8f26fbc7b5bd1e586ae05ab63d77cac0bf0f3a075f4c796f6c15eee91dfe7e17` |
| 目标族 | 留出、Meta-Val、Outer Test 都不在 `groups` 和卡片族名单里 |
| 检索 | 重新加载后，Where/Box/Class 与建库时的方向余弦约为 1，来源组只有 1，不含任务自己的组 0。Feature 在排除自身组后没有别的成员，检索为 Null |
| 磁盘 | 991,604,499 字节。其中卡片负载 330,393,288 字节，其余主要是四条任务的 FP32 原始 Δ |
| A_m | Feature 0.282，Where 0.00200，Box 0.00377，Class 0.0365 |
| 下一轮 Trim | 没有 n≥8 的 bin，四个模块的下一轮保留比例都是 1.00 |

`query_seen_by_theta0=true`。这两个源族的 60 个 Query 物理组都在监督 θ₀ 的训练索引里。第一版没有重训 θ₀，不能写成“Query 对 θ₀ 完全没见过”。

## 还没有接到训练循环的部分

下面四条是建库当时的缺口。检索接入已经补上，见文末；这里保留当时的判断。

`training/theta0.py` 当时仍走原来的在线任务记忆，不会调用上面四个入口。留出评价当时也还没有复用这套检索。

已留给下一步、但这次没有执行的部分：

- 把检索结果按 γ·A_m·B_mix 加到 θ，再做 Support 内更新和 Query 外更新。
- 30% 整任务不借用，以及只在真正借用的任务上更新每模块的 g、beta。
- 同一折的后续 epoch 只重算 Δ。现在的 `build_experience_bank` 每次都会重做 PCA 和 K-means；簇中心已经在 `module_memory.pt` 里，下一步要加一个只换卡片的入口。
- 训练任务排除自己的经验组，Meta-Val 和 OOF 目标使用全部源组。检索函数已经按 `excluded_group` 实现，调用方还没有接上。

## 对下一步的影响

可以开始把检索接到 `theta0.py`。这次的库就是这份 θ₀ 的 epoch 0 卡片，参数哈希必须和加载的权重一致。

接线前需要单独处理、但不需要改已定算法的两点：

1. 全折建库会比这次大得多。4 条任务已经占 946MB，主要是 Feature 的 FP32 原始 Δ。27 个源族、每族实际只有 3 条 Episode，全库大约是十几 GB 磁盘；合并时仍按 bin 读取，不能把所有 Feature 差同时放进内存。
2. 冻结 Bank 里每个源族只有 3 条 Episode。配置里的 M=8 不会被改掉，全库也只能记下有效条数 3。这次最小库的卡片都是单成员，真实多成员 TIES 要等更多源族落进同一个簇之后才会发生。

## 检索接入 FOMAML

同一天完成接线，没有启动完整训练，也没有改 `config/fomaml.yaml` 和 `config/retrieval_fomaml.yaml`。最小库目录没有被改写。

`fomaml` 默认使用检索式更新，并自动读取当前折里最新的经验库，启动时不打印配置。现在这份最新库是 `fold0-bank-min`。`_train` 在每轮开始调用 `refresh_experience_cards`，每条任务调用 `retrieval_fomaml_step`。这一轮训练循环本身没有执行。刷新函数单独对更新后的 θ 跑过一次，产物已删除。

单条任务的顺序是：用 θ₀ 缓存描述子检索；来源组掩码和自身经验组过滤；四个模块分别做动态 Top-K 和跨簇加权；γ 限制在 [0, 1]。允许借用时，临时初始化是 θ + γ·A_m·B。Support 只做一步，`module_delta` 相对借用后的初始化计算。Query 梯度算出后，先把共享参数恢复到借用前，再做外层 SGD。随后用 L = Σ γ·A_m·⟨g_Q, B⟩ 更新 g 和 beta。30% 不借用是整条 Episode 一个随机位；不注入，但仍用这条 Query 梯度更新 γ。空的 s0 基线按 0。

留出入口是 `retrieval_holdout_plan`，内部调用同一套 `plan_module_borrow`。它拒绝 Query 记录，也不更新 θ、经验库或 γ。原来的 `run_holdout` 臂循环没有替换，也没有重跑。

接线检查用 `oof:mvtec:bottle:000` 和 `fold0-bank-min`。Where、Box、Class 注入了经验组 1，Feature 在排除自身组后没有卡片。恢复差为 0，借用泄漏最大绝对差 7.7e-9。三处非零 γ 梯度的符号与 ⟨g_Q, B⟩ 一致。刷新后的 θ 哈希是 `d41883cad3ea6675de2a4288462e97ca10045d11b880454b957b8d0372c1465c`，与当时检测器一致，并和原库不同；簇分配仍是原来的 4 条任务。`query_seen_by_theta0` 仍是 true。Meta-Val 和 Outer Test 与 θ₀ 训练集没有交集。

尚未发生的事：完整 FOMAML 训练、留出 AP50、多成员 TIES，以及把 M=8 真正取满。这次最小库请求的是每族 2 条，刷新没有把任务复制成 8 条。

## 模式采样、模块学习率和留出接线

同一天继续完成这三项。没有改检测损失，没有改 `MODULE_PREFIXES`，没有改 `config/fomaml.yaml`、`config/episode.yaml`、冻结划分和 `output/episodes/v1`。最小库 `fold0-bank-min` 的文件哈希没有变。没有启动完整训练，也没有计算留出 AP50。

### 修改的文件

| 文件 | 作用 |
| --- | --- |
| `config/mode_episode.yaml` | 新增。60/40、5-shot 和 MVTec 受控清单的种子、比例与输出目录 |
| `data_proc/mode_episodes.py` | 新增。只从各角色已有物理组池抽样，数量不够就降级 |
| `scripts/prepare.py` | 新增子命令 `mode-episodes`。原来的 `episodes` 仍写固定 Episode |
| `config/retrieval_fomaml.yaml` | 只增加四个模块学习率的初值、上下界、Adam 和元学习率 |
| `training/experience_bank.py` | 模块学习率、Support 适应、α 一阶代理、卡片快照和单条检查 |
| `training/theta0.py` | 训练步把 α 传给适应函数；每轮卡片使用当时的 α 快照；断点和最佳轮次同时保存 γ 与 α |
| `training/meta_transfer.py` | 留出两臂调用同一个适应函数。旧的 `run_holdout` 没有替换 |
| `tests/test_mode_episodes.py` | 60/40、单模式、样本不足、角色隔离和固定种子 |
| `tests/test_experience_bank.py` | α 初值和符号、不借用、卡片快照、留出调用同一函数 |

### 调用链

模式清单：`prepare.py mode-episodes` → `write_mode_manifest` → 核对冻结划分 → `build_mode_episodes`。训练仍由 `load_fomaml_episodes` 读取原来的 OOF Episode，不会自动改读这份清单。

一条训练任务：`retrieval_fomaml_step` 用 R 和 Support 做签名并检索。`retrieval_support_adapt` 按四个 α 做 Support 一步。共享 θ 先恢复，再由原来的外层 SGD 更新。γ 的 g、beta 仍由原来的 Adam 更新。四个 α 只用已经算好的梯度，按 `dL_Q/dα_m = -⟨g_Q, g_S⟩` 更新，不再做检测器前向或反向。

整条任务不借用时，前向增量是空的。Query 梯度仍用来更新 γ，记录为 `unborrowed_query_first_order_estimate`：这是未借用轨迹上的局部一阶估计，不是这条 Episode 已经发生的借用收益。

留出：`retrieval_holdout_pair` 对同一 Support 先做无借用 A，再做检索借用 B，两次都调用 `retrieval_support_adapt`，结束后恢复 θ，并且不更新 α、γ。`retrieval_holdout_query_plan` 在这之后才把 Query 分成同模式、异模式和总体。`load_retrieval_holdout_state` 会读取最佳轮次里的 θ、经验库、γ 和 α；现有旧检查点没有这些字段，因此旧留出命令没有被改道。

卡片：`delta_m = -alpha_m_snapshot * gS_m`。同一版经验库共用一个 θ 哈希和一个 α 快照。PCA、簇中心和簇分配仍不重拟合。固定 0.01 的旧卡没有 `module_snapshot`，不能和新快照声称是同一版本。

### 实际测试

MAML 环境，2026-11-24，退出码都是 0。

```text
D:\miniconda3\envs\MAML\python.exe -B -c "import importlib.util, sys, unittest; ..."
```

载入 `tests/test_retrieval_memory.py`、`tests/test_experience_bank.py`、`tests/test_mode_episodes.py`。35 项通过，失败 0。其中包含原来的检索与接线测试。新增覆盖了 3/2 的 Query 缺陷、单模式不补齐、Support 不足时不复制、R/Support/Query 角色隔离、同一种子重复构造一致、四个 α 的有效初值 0.01、人工梯度的更新符号、旧卡与新快照不能混用，以及留出和训练都调用 `retrieval_support_adapt`。Query 标签只出现在签名的重叠拒绝和评价拆分里，不进入检索键。

```text
D:\miniconda3\envs\MAML\python.exe -B -u scripts/prepare.py mode-episodes
```

MVTec 清单写到 `output/episodes/mode_v1/mode_episode_manifest.jsonl`。分配文件哈希仍是 `20a9cac0b8a1ec2b3118ecb2f66a24401c25e365ec2998e1b351bb0a9dd80932`。15 个产品族、73 种产品内缺陷模式。Outer Test 已排除，实际开放 12 个产品族、57 种模式。生成 103 条 Episode，其中 57 条满足 5-shot 和 60/40，46 条降级，另有 11 次因没有剩余 Support 组而跳过。降级原因计数是 Support 不足 44、同模式 Query 不足 28、其他模式不足 8、只有一种模式 2；一条 Episode 可以同时记下多个原因。84 条来自 inner OOF，可以标记为训练候选；其余是 Meta-Val 清单，不进入当前训练加载器。这是自定义监督式 few-shot，不是 MVTec 官方无监督测试，也不替代 VisA 和 Real-IAD。

```text
D:\miniconda3\envs\MAML\python.exe -B -u -c "from training.experience_bank import check_alpha_gamma_theta; check_alpha_gamma_theta(r'output/meta_transfer/experience_bank/fold_0/fold0-bank-min', r'output/meta_transfer/theta0/fold_0/fold0-detloss-v1/last.pt')"
```

任务 `oof:mvtec:bottle:000`。四个 α 的有效初值都是 0.01（float32 显示为 0.009999996）。Support 与 Query 的点积分别是 Feature 7.889、Where 0.001219、Box 0.002647、Class -0.1774；前三个 α 上升，Class 下降，四个符号都符合。Feature 的 γ 仍是 -1，因为它在排除自身经验组后没有卡片；Where、Box、Class 的 γ 有变化。θ 哈希已变化，恢复差 0，借用泄漏 7.65e-9。四个学习率都没有触及 1e-5 或 0.03。这一步峰值显存 1681.13 MiB，耗时 13.53 秒；含模型加载的进程约 32 秒。最小库文件没有改写。

### 能否启动正式训练

还不能把这次结果当成正式三数据集训练的起点。

- 默认 `fomaml` 仍会读取 `fold0-bank-min`。它只有两个产品族，每族 2 条任务，只能检查接口。
- 新的 60/40 清单没有接进训练循环。直接启动训练用的仍是原来的 OOF Episode。
- 留出两臂和三种 Query 分组已经接上，但没有算出 AP50。旧留出协议没有重跑。
- 学习率上下界和元学习率 0.001 只是初始约束。这条任务没有碰边界，不能当成已经调好的值。
- 真实多成员 TIES 仍没有发生。
