# hp_abc_20261111 诊断结论：CR7 验证缺席与困难对未真正注入

时间：2026-11-11 18:39（北京时间）。脚本与证据在同目录：
output/cspd/hp_abc_20261111/diagnostics/diagnose_val_and_hardpairs.py、diagnosis.json、本文件。
复现命令（项目根目录，无模型、无 GPU 训练/评估，未改生产代码/配置/划分）：

`
D:\miniconda3\envs\MAML\python.exe output/cspd/hp_abc_20261111/diagnostics/diagnose_val_and_hardpairs.py --project D:\元强化学习\代码
`

_load 被替换为占位张量；采样使用官方 cspd.data.CurriculumNewSplitEpisodes 与各折 config.json，验证种子为 seed+2=2。与 metrics.json 比较的是每次 same 验证结束、当次 cross 验证之前的累计计数，两折完全一致。

## 1. 总判定

CR7 验证来源缺席已经定位：不是组 ID 解析错误，也不是重试次数不够；现行 5-way 在 CR7 val 上数学上无解。7 类按张数都够进 usable，但 dent/linear/macular 在严格跨组下无法划分支持/查询，组可行只剩 4 类，任何 5 类组合都必然含有不可行类，故该来源每次尝试都失败、困难对缓冲从未纳入 CR7。留出 GC10 时油污/冲孔对未注入，是「训练来源覆盖结构」与「CR7 缓冲为空」叠加，不是 0.40 调度未用到，也不是 changed 被漏记。

## 2. CR7 val 容量、组可行性与复现

组映射 0 条缺失，独立子集和与官方 _split_by_group 严格分支判定完全一致。图像充足但组结构不够：linear、macular 的 val 整段只有 1 个组；dent 是 98+2，任一端都无法同时满足 k∈{3,5} 与 query=8。这与「同一组不拆到 train/val/test」的划分一致，不是读错文件。

| 类 | 有掩码图像 | 组数 | 组大小 | 缺组ID | 计数可用 k=3/5 | 组可分 k=3 | 组可分 k=5 | 原因 |
|---|---:|---:|---|---:|---|---|---|---|
| dent | 100 | 2 | 98, 2 | 0 | 是 | 否 | 否 | 子集和落不进 [3,92] / [5,92] |
| linear | 83 | 1 | 83 | 0 | 是 | 否 | 否 | 单组不能拆 |
| macular | 91 | 1 | 91 | 0 | 是 | 否 | 否 | 单组不能拆 |
| inclusion | 58 | 7 | 24,22,7,2,1,1,1 | 0 | 是 | 是 | 是 | 可行 |
| oil | 114 | 12 | 63,25,8,4,2×6,1,1 | 0 | 是 | 是 | 是 | 可行 |
| pit_local | 93 | 6 | 22,22,21,19,8,1 | 0 | 是 | 是 | 是 | 可行 |
| punch | 124 | 6 | 37,33,29,16,5,4 | 0 | 是 | 是 | 是 | 可行 |

组合（k=3 与 k=5 相同）：计数可用 7 类，代码 
_way=min(5,7)=5，组可行 5 元组合数 = **0**。组可行 4 类（oil / inclusion / pit_local / punch）可以构成 4-way，算法从未尝试：usable 只按张数过滤，抽满 5 类后任一失败即整源失败，且 sample() 不在同一来源内重抽类别组合。因此这不是「5-way 有解但贪心没找到」，而是「现行 5-way 无解，4-way 有解未走」。

对照：CR7 **train** 七类在 k=3/5 下全部组可分（故训练 episode 能抽到 CR7）；留出评估 holdout_pool=all 合并 train+val+test，组数足够，出现 20/20 成功，与 val 缺席不矛盾。

无模型复现（与日志 after-same 计数逐项相等）：

- B_NEU_s0：CR7 attempts=128、successes=0；group_failures 为 linear 31、macular 49、dent 48。全程 _sample_source 失败 133 次（含最后一次 cross 的 5 次，日志快照不含这 5 次），成功 0。错误全部是 strict groups unavailable: CR7:{linear,macular,dent}，same/cross 都无成功。
- B_GC10_s0：CR7 attempts=120、successes=0；failures 为 dent 37、linear 42、macular 41。全程失败 126 次（含最后 cross 6 次），成功 0，错误同类。
- 先前把 20 个累计快照加总得到约 1318/1161，是重复计数；唯一尝试次数以末条（及本次复现）为准。

附带（不是 CR7 全缺席的同级原因，但解释了部分失败计数）：

- PLATE oxide_scale val 组大小 30,1,1,1：k=3 可行，k=5 不可行。PLATE 只有 3 类，饱和，k=5 时 oxide_scale 必进场，故该来源在 k=5 必失败、k=3 可成功（B_NEU 94/123 成功）。
- NEU scratch val 组大小 41,3：同样 k=3 可行、k=5 不可行，但 NEU 有 6 类，只有抽到 scratch 且 k=5 才失败（B_GC10 有 14 次 NEU:scratch）。
- CSU val 只有 inclusion 张数够（53），scratch 9 张、oxide_scale_csu 2 张，usable 类数 < 2，同来源 val episode 无法成立，故 CSU 缓冲始终为空。

## 3. 困难对实际注入：按来源瓶颈与真因

字段语义与 choose_classes 一致，不是漏记：
equested 是概率抽中要干预；allback 是没有合法缓冲对；pplied 是选中了对（即使两类本来就在 episode 里）；changed 相对**同一次**基础抽样是否改了集合；saturated 是 len(usable)≤n_way，组成无法改变。pplied 且未 changed 在饱和来源上是预期行为（PIPE/PLATE 上大量 applied=requested、changed=0）。

训练来源类数：CAST 6、CR7 7、GC10 9、NEU 6 不饱和；CSU 3、PIPE 3、PLATE 3、S3D 2 **饱和**（约一半以上训练 episode 落在这四源，组成不可能被改）。CAST/CR7/GC10/NEU/PIPE/PLATE/S3D 均无跨 split、无留出泄漏（三折 holdout_in_train_episodes=0）。0.40 在 100 步内按代码就不会出现（最高 0.35），不是本轮注入失败的原因。

按折：

- **B_GC10_s0（留出 GC10）**：changed 只有 CAST 渣系 6 次（slag_skin↔weld_slag 4、slag_skin↔water_slag_mark 2）。CR7 训练 57 episode、请求 11 次全部 fallback；缓冲 20 次快照里 CR7 从未激活。预列对中，oil↔water、punch↔weld_line 在任何训练来源上都不同时存在（只有 GC10 同时有这些类，而 GC10 按协议不得进训练/缓冲）。oil↔inclusion、linear↔pit_local、pit_local↔inclusion 只存在于训练来源 CR7，但 CR7 缓冲为空，故不可能被选。结论：留出 GC10 时「无油污/冲孔注入」是结构限制（GC10 专有对无处可采）加上 CR7 验证缺席（CR7 上的油污/夹杂/线状/局部坑对进不了缓冲），不是候选被抽到却没改组成。
- **B_NEU_s0（留出 NEU）**：changed=15，来自 GC10 8 次（含 punch↔weld_line 2）与 CAST 7 次。CR7 请求 20/20 fallback，缓冲无 CR7。此时 GC10 在训练中，GC10 目标对是有宿主的，也确实出现了少量 punch↔weld_line。
- **B_CR7_s0（留出 CR7）**：changed=14，GC10 5 次（含 punch↔weld_line 2、inclusion↔oil 1）与 CAST 9 次。linear↔pit_local、pit_local↔inclusion 在训练来源上已不存在（只有 CR7 有这两类）。NEU 虽有 inclusion↔pit_texture 且同来源 val 有 episode，但缓冲 20 次从未激活：该对只在个别评估进入 top 且 streak 始终为 1，从未连续两次确认；这是次级剂量问题，不是本次 CR7 缺席的主因。

CAST 主导 changed，是因为它不饱和、val 几乎每次都有 episode、缓冲稳定激活渣系对。不能据此认为「只有 CAST 困难」。

## 4. 最小修复建议（本轮不实施）

必须继续禁止：同组混入 support/query、读取留出来源、跨 train/val 混合。

1. **优先改验证/采样协议（算法层，不改划分）**：usable 先按严格跨组可行性过滤，再 
_way=min(上限, 组可行类数)。对 CR7 val 会变成 4-way（oil/inclusion/pit_local/punch），缓冲才可能出现 oil↔inclusion、pit_local↔inclusion。dent/linear/macular 仍不会进入 val 混淆，因此 linear↔pit_local 单靠这一步仍然进不了缓冲。应用同一过滤后，CSU 同来源 val 仍会因可用类不足而不出 episode，需单独看待。
2. **不要指望多抽几次 5-way 或加长到启用 0.40**：现行 5-way 组合数为 0，重试不会成功；0.40 解决不了空缓冲和饱和来源。
3. **若需要 dent/linear/macular 也进入 val 混淆，才动数据层**：在保持「整组进入同一 split、不拆组」的前提下，检查 val 是否至少有两个组且存在合法 (k,q) 划分；不够则只能调整分组粒度或该组的 split 分配。这会改官方划分，必须单列，不能当成采样热修。
4. **留出 GC10 的 oil↔water、punch↔weld_line**：训练来源上本来就没有这两对，修复 CR7 val 也不会凭空产生它们。只能接受「该折不注入 GC10 专有对」，或另设与留出隔离的、仍只用训练来源类的目标，不得把 GC10 样本引进训练或缓冲。

## 5. 局限

本诊断解释的是验证来源缺席与困难对剂量/方向，**不能解释 GC10 留出 acc 从历史约 0.665 降到 0.63125 的因果**。那一差值仍与 strict_groups、查询集分母变化、单种子噪声等混杂，没有同协议 A 对照，不得把 GC10 下降归因为「困难对伤了模型」或「prototype 不稳」。本轮未改代码、未启动训练或模型评估。
