# GC10 hard-pair 关闭对照 A（仅 1 次训练）+ 固定清单配对评估

时区：Asia/Shanghai（UTC+8）

B 只读，不重训：`output/cspd/hp_b_samplerfix_20261111_v2/B_GC10_s0`
A 新输出：`output/cspd/gc10_hp_control_20261111/A_GC10_s0`
配对评估：`output/cspd/gc10_hp_control_20261111/paired_eval`

勿改 `config/cspd/hardpair_abc.yaml` 默认三折 B 队列。不要从 B 的 last.pt 续训。
`train_cspd.py` 无 resume；每次从 `build_model` + `seed=0` 初始化。
若 `A_GC10_s0` 目录已存在，训练入口 `output_dir(..., fresh=True)` 会拒绝覆盖。

## 1. 训练 A_GC10_s0

```powershell
cd D:\元强化学习\代码
D:\miniconda3\envs\MAML\python.exe scripts\train_cspd.py --config config\cspd\gc10_hp_control_a.yaml
```

## 2. 固定清单配对评估（manifest 不存在则创建一次，已存在则校验后复用，不随机覆盖）

```powershell
cd D:\元强化学习\代码
D:\miniconda3\envs\MAML\python.exe scripts\eval_paired_holdout.py --config config\cspd\gc10_hp_control_a.yaml --baseline output\cspd\hp_b_samplerfix_20261111_v2\B_GC10_s0\last.pt --candidate output\cspd\gc10_hp_control_20261111\A_GC10_s0\last.pt --manifest output\cspd\gc10_hp_control_20261111\paired_eval\manifest.json --output output\cspd\gc10_hp_control_20261111\paired_eval
```

清单 seed 默认 `20261111`，20 个 holdout episode（与 config `eval_tasks` 一致），协议同 B：GC10 / holdout_eval=same / holdout_pool=all / q=8 / k mixture / strict_groups。这是独立新抽的固定清单，不是 B 官方 20 任务的路径重建。配对数字不可与 B 已公布 holdout acc 直接当成同一批 episode。

可选静态 diff（不加载权重、不采样）：

```powershell
cd D:\元强化学习\代码
D:\miniconda3\envs\MAML\python.exe scripts\eval_paired_holdout.py --check-config --config config\cspd\gc10_hp_control_a.yaml --baseline output\cspd\hp_b_samplerfix_20261111_v2\B_GC10_s0\last.pt
```
