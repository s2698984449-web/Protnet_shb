# hp_b_samplerfix_20261111_v2 离线结果（只读）

三折 run_complete=complete，metrics=evaluated，权重 last.pt（日志：no early stop, use last）。配置与旧 B 相同口径：seed=0，iters=100，eval_every=5，eval_tasks=20，layer3，strict_groups，hard_pair_protocol，same_hard_pairs，beta=0.25，rho=0，holdout_pool=all，acceptance_neu_mde=null。日志含新字段 fallback_reason，且 applied 恒等于 changed（三折 applied!=changed 均为 0），确认是修复后采样器，不是旧 B 换目录。

## 留出（metrics.json holdout；分母与旧 B 完全一致，可并列召回；不能仅凭 seed 断言支持/查询图逐张相同，路径未记录）

| 折 | hit/800 | acc | loss | 旧B acc/loss | 约差 | 历史 layer3_p1 | 关键类 v2 vs 旧B（同分母） |
|---|---:|---:|---:|---|---:|---:|---|
| NEU | 771 | 0.96375 | 1.0348 | 0.96375 / 1.0475 | acc 相同，loss -0.013 | 0.951 | 夹杂 122/136 vs 124；麻面 129/136 vs 127 |
| CR7 | 587 | 0.73375 | 1.2452 | 0.72875 / 1.2377 | +4 张 | 0.706 | 油污 50/112 vs 51；线状 99/144 vs 98；局部坑 69/104 vs 64；冲孔仍 112/112 |
| GC10 | 517 | 0.64625 | 1.3309 | 0.63125 / 1.3282 | +12 张 | 0.665 | 油污 35/104 vs 36；冲孔 55/104 vs 50；punch→weld_line 23 vs 25；焊缝 86/96 同 |

相对历史三阈值：NEU、CR7 点估计过，GC10 仍低于 0.665（517 vs 约 532），整套初筛未过。NEU MDE 仍为 null，不可宣称该条已过。test_same 查询数 504/648/616，与留出 800 及旧 B 同源测试都不是同一口径。

## 真实链路（累计用末条，禁止快照相加；changed 分母=100×4=400）

- B_NEU：CR7 val attempts/successes **132/132**（首条 5/5），group_failures={}。20 次 same-val 均有 CR7，均为 4 类 {oil,inclusion,pit_local,punch}，无 linear/macular/dent。训练 400 episode，changed=30，holdout 进入训练=0。CR7 oil-inclusion：缓冲 iter 10 进入，changed=3 首次 iter 24；pit_local-inclusion：缓冲 iter 10，changed=8 首次 iter 20。linear-pit_local：val 不能生成，changed=0，不当失败。GC10 punch-weld_line（训练源 GC10）：缓冲 iter 65，changed=11 首次 iter 67。GC10 oil-water：未进缓冲、changed=0（该折结构上可共存，只是本种子未观察到）。
- B_GC10：CR7 val **110/110**，19/20 次 same-val 有 CR7，均为上述 4 类。changed=17/400，holdout 进训练=0。CR7 oil-inclusion 缓冲 iter 10，changed=5 首次 25；pit_local-inclusion 缓冲 iter 10，changed=4 首次 63。oil-water、punch-weld_line：**unavailable_same_source_pair**，不算失败，changed=0。
- B_CR7：CR7 非训练源。changed=19/400。GC10 punch-weld_line 缓冲 iter 20，changed=1 首次 97。GC10 oil-inclusion 曾进 GC10 缓冲（iter 30）但 changed=0。linear/pit_local 对因留出 CR7 为 unavailable。NEU inclusion-pit_texture 未进缓冲，NEU 请求 23 次全部 fallback 原因为 no_coexist。
- 出现次数：日志仍无 episode 类集合，记 **NA**，不能当 0。请求在 fallback 时 pair 未知。旧 applied 不可与本轮 applied 直接比。

## 准入与下一步

P0（CR7 val 不再 0 成功、4-way 可行类）本轮真实训练**已过**。P1 目标对中，oil-inclusion 与 pit_local-inclusion 在 CR7 为训练源的两折上，真实缓冲且 changed≥1，已观察到。不能因总 changed 升高（30/19/17 vs 旧 B 15/14/6）就判所有目标通过：linear-pit_local 仍不可能；GC10 专有对在留出 GC10 时结构不可注入；oil-water 即使 GC10 在训练中本种子也未注入；NEU 目标对本种子未进缓冲。

留出升降不能确定归因于困难对（无同协议关采样对照；采样修复本身也改了训练组成）。单种子 20 任务不可称显著。**本轮效果验收未全面通过**（GC10 仍低于历史 0.665；MDE 未预注册）。

不在此启动训练。优先：(1) 若要解释 GC10/CR7 留出相对旧 B 的点估计变化，值得做 **同协议、同样本修复、关 hard_pairs 的 A 对照（P2）**，先一种子三折或至少 GC10 seed0；(2) 不要把原型按组聚合提前，等有 P2；(3) 一次只改一个因素。
