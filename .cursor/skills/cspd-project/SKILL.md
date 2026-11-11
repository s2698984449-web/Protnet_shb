---
name: cspd-project
description: >-
  串起钢板表面缺陷元学习（CSPD）项目的实验和思路。做训练、改配置、看终端、对比指标、判断瓶颈或写下一步时，实验做完写进 doc/cspd/实验记录.md，有思路写进 doc/cspd/思路历程.md。Use when working in this repo on steel defect meta-learning, CSPD training, configs, train.log, 实验记录, or 思路历程.
---

# CSPD 项目记录

做这个项目时把结论落到文档，不另开记录。

## 现行基线与研究主线

工业交付见 `doc/cspd/阶段性基线.md`：冻结 layer4 + 同产线 3～5-shot + `d_core+0.25·d_ring`。

- 交付入口：`scripts/eval_industrial_proto.py` + `config/cspd/industrial_proto.yaml`
- 产物：勿覆盖 `output/STEEL_industrial_proto/`
- 旧训练栈仍在：`scripts/train_cspd.py` + `config/cspd/cspd.yaml`

研究主线见 `doc/cspd/课程学习主线.md`。全开入口是 `config/cspd/curriculum_full.yaml`，队列是 `scripts/run_p1_queue.py`，数据是 `data/new_split_v3`。三件关闭的对照配置留在 `config/cspd/curriculum_p1.yaml`，九次权重已删，数字在实验记录。主评估见 `doc/cspd/协议卡_P1留出来源.md`。类等价见 `doc/cspd/类等价评判标准.md` 与 `config/cspd/class_manifest.yaml`，等级仍暂定，exact 目前只有冲孔。CSU / S3D 已入过 v2；Rail 未入；XSDD 无掩码，训练排除。工况编码、策略网络、跨来源课程采样在全开配置里一起打开。不默认开 patch / whitening / E / 通道门控。

## 产物目录（硬性禁止）

禁止删除 `output/`，也禁止删除其下任何实验结果目录，包括权重、日志、`last.pt` 和各次 run 目录。即使用户提到轻量化、结论已写入实验记录、或一次性探针已经结束，也不得自行删除 `output/` 里的内容。只有用户明确点名某个目录并当场确认后，才可以删那一个目录。

轻量化只针对一次性探针脚本和专用配置：结论入库后可以删这些脚本和专用配置，但不要动 `output/`。用户需要保留的是核心实验配置（`config/cspd/`）和结论（`doc/cspd/实验记录.md`、`doc/cspd/思路历程.md`）。

新实验的产物必须写在本仓库的 `output/` 下，不得写到项目外。新一节实验记录写到 `doc/cspd/实验记录.md` 文首，便于看见。

## 实验做完写进实验 md

路径：`doc/cspd/实验记录.md`。只抄日志里有的数。新实验用新的 `exp_name`。

## 有思路写进思路 md

路径：`doc/cspd/思路历程.md`。一次判断一节：日期、主题、结论、依据、后续动作。

工业协议见 `doc/cspd/协议卡_目标域少样本.md`。铸坯是开发域（选 β）；分母 CR7/GC10/NEU。研究侧留一评估见 `doc/cspd/协议卡_P1留出来源.md`。

## 读结果时

- 交付主看同源缺陷准确率、Δbg、Δhard、混淆。
- 跨来源支持只附录。
- 不称通用钢板元学习框架，不和检测 mAP、MVTec、全量检测器比。
---
