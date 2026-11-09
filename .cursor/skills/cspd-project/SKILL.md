---
name: cspd-project
description: >-
  串起钢板表面缺陷元学习（CSPD）项目的实验和思路。做训练、改配置、看终端、对比指标、判断瓶颈或写下一步时，实验做完写进 doc/实验记录.md，有思路写进 doc/思路历程.md。Use when working in this repo on steel defect meta-learning, CSPD training, configs, train.log, 实验记录, or 思路历程.
---

# CSPD 项目记录

做这个项目时把结论落到文档，不另开记录。

## 现行基线

见 `doc/阶段性基线.md`。工业导入已锁定：冻结 layer4 + 同产线 3～5-shot + `d_core+0.25·d_ring`。

- 入口：`scripts/eval_industrial_proto.py` + `config/industrial_proto.yaml`
- 产物：仅保留 `output/STEEL_industrial_proto/`（不要覆盖）
- 训练入口仍在：`scripts/train_cspd.py` + `config/cspd.yaml`（`cspd/model.py`、`data.py`、`runtime.py`）。不要删训练栈。
- 不做：把一次性探针脚本和中间 `output/` 产物长期留在仓库；主线不默认开 patch / whitening / E。

## 实验做完写进实验 md

路径：`doc/实验记录.md`。只抄日志里有的数。新实验用新的 `exp_name`。

## 有思路写进思路 md

路径：`doc/思路历程.md`。一次判断一节：日期、主题、结论、依据、后续动作。

一次性探针：结论入库后删除脚本、专用配置和对应输出目录。

协议见 `doc/协议卡_目标域少样本.md`。铸坯是开发域（选 β）；分母 CR7/GC10/NEU。

## 读结果时

- 交付主看同源缺陷准确率、Δbg、Δhard、混淆。
- 跨来源支持只附录。
- 不称通用钢板元学习框架，不和检测 mAP、MVTec、全量检测器比。
---
