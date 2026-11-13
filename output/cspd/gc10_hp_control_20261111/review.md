# GC10 A关闭 vs B开启 结果复核（2026-11-11，Asia/Shanghai）

范围仅 output/cspd/gc10_hp_control_20261111/A_GC10_s0 与 paired_eval。未读权重，未改代码/配置，未新训练或新评估。

## 完整性

- A：
un_complete=complete（20:45:13），last.pt 44,936,306 字节，metrics.json 已写。
- 实际配置：hard_pair_protocol=true，same_hard_pairs=false，strict_groups=true，layer3，fixed_beta=0.25，rho=0，seed=0，100 步，GC10 holdout same/all。训练源无 GC10。
- 注入：100 条概率日志全为 0；requested/applied/changed/fallback=0；hard_sampling={}。验证 buffer 仍更新（协议结构保留），但未注入。
- CR7 val 累计成功 110/110（修复保留）。训练各组 attempts=successes，group_failures 空。
- 配对：20:49:21 已齐。aseline=hp_b_samplerfix_20261111_v2/B_GC10_s0/last.pt（B 开启），candidate=gc10_hp_control_20261111/A_GC10_s0/last.pt（A 关闭），标签未反。清单 seed=20261111，20 个全 5-way，k=3×11 + k=5×9，query 800，attempts 20 无 skip。路径全在 GC10，组无交集，无缺失/逃逸。holdout_pool=all 故清单含 GC10 的 train/val/test 图（706/242/242），这是协议不是跨源泄漏。

## 同清单配对（这是唯一严格对比）

| 指标 | A 关闭困难对 | B 开启困难对 | 差值方向（B−A） |
| --- | --- | --- | --- |
| acc | 0.6025（482/800） | 0.6125（490/800） | +0.010，B 更高 |
| loss | 1.3482 | 1.3233 | B 更低（更好） |
| 油污 oil | 31/96 | 34/96 | B 多对 3 |
| 冲孔 punch | 37/80 | 37/80 | 0，完全相同 |
| 夹杂 inclusion | 34/104 | 33/104 | A 多对 1 |
| 轧坑 rolled_pit | 36/96 | 41/96 | B 多对 5（本清单最大类差） |
| 配对计数 | A对B错 11；B对A错 19；两边都对 471；两边都错 299 | 同上 | 净差 8 题 |

油污 62/96 两边都错；冲孔 43/80 两边都错且 0 分歧。主要混淆两边都是 inclusion→silk_spot、punch→weld_line、rolled_pit→fold_wrinkle、oil→punch 等。单 seed、20 任务，不能称显著，也不能推广“困难对普遍有效”。

## 官方评估（不可当严格配对）

| | A 关闭（本 run 官方 holdout） | B 开启（v2 官方，仅描述） |
| --- | --- | --- |
| acc | 0.615（492/800，skip 0，loss 1.3539） | 0.64625（517/800） |
| oil | 32/104 | 35/104 |
| punch | 52/104 | 55/104 |

历史 0.665 口径混杂，不作对比。A 官方 0.615 与同清单 A 0.6025 接近；B 官方 0.64625 高于同清单 B 0.6125，说明 20 任务抽次方差很大，不能把两份官方数字当成困难对效应。

## 结论与建议

已证实：A 确实关闭注入且保留严格跨组/动态 nway 修复；同清单上困难对没有可靠增量证据；油污瓶颈仍在。

1. 不要再为“再调一次 hard-pair”开新训练。本对照已回答开关问题：净差仅 8/800，冲孔无差别，油污仍约三分之一召回。
2. 油污仍是主瓶颈（两边 62 题同错）。下一步若还做单因子，应转到此前已后置的按组聚合原型，并冻结本次 A 或 B 之一为基线，不要同时改 hard-pair。
3. 若资源紧，也可先停用 hard-pairs（A 与 B 几乎持平），把容量留给原型试验。
