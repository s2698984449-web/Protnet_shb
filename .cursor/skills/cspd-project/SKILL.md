---
name: cspd-project
description: >-
  串起钢板表面缺陷元学习（CSPD）项目的实验和思路。做训练、改配置、看终端、对比指标、判断瓶颈或写下一步时，实验做完写进 doc/实验记录.md，有思路写进 doc/思路历程.md。Use when working in this repo on steel defect meta-learning, CSPD training, configs, train.log, 实验记录, or 思路历程.
---

# CSPD 项目记录

做这个项目时把结论落到文档，不另开记录。

## 实验做完写进实验 md

路径：`doc/实验记录.md`。

一次训练结束，先读该次 `output/<exp_name>/train.log` 和对应终端。只抄日志里有的数。测试行没打的定位指标不要补。

每批追加一节，包含：

- 日期、数据、协议（换源或同来源）、5-way / shot / query、相对上一批改了什么
- 输出目录的绝对路径
- `best.pt` 与 `last.pt` 的轮次
- 可训练参数
- 验证最好一轮的 acc、loss，以及该行若有的 loc、g、loc@0.2
- 测试行的 acc、loss
- 和上一批对照实验的差值

新实验用新的 `exp_name`。不要覆盖 `output/STEEL_base`、`STEEL_v1`、`STEEL_same_base`、`STEEL_same_v1`、`STEEL_cross_l4`、`STEEL_holdout_CAST`、`STEEL_holdout_CAST_align`。

## 有思路写进思路 md

路径：`doc/思路历程.md`。

一次判断追加一节：日期、主题、结论、依据、后续动作。不写代码细节。一条回答里改变了计划，就写一条。没有新判断就不写。

## 读结果时

- 主指标是查询准确率。5 类随机水平是 0.2。20 个 episode 约 300 张查询图。
- 单次训练 iter 的 acc 不是指标。早停和选权重只看独立 `val_dir`。
- 换源准确率接近同来源，先核对是不是训练时见过全部来源。见过全部来源的换源抽查，不能写成没见过的产线。
- 定位看 `loc@0.2`。分类高、定位低，说明用的是整图纹理。
- 不称通用钢板元学习框架，不和检测 mAP、MVTec、全量检测器比。
---
