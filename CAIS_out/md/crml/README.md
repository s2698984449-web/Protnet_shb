# CRML 实验目录

## 目录结构

```
md/crml/
├── 01_architecture_design.txt      # 架构设计文档
├── 02_cross_domain_sampler.txt    # 跨域采样器设计
├── 03_relation_loss.txt           # Relation Consistency Loss 设计
├── 04_training_integration.txt     # 训练脚本集成
├── 05_experiment_results.txt      # 实验结果记录
└── README.md                       # 本目录说明
```

## 文件说明

### 01_architecture_design.txt
- CRML 核心思想
- 为什么关系跨域不一致
- 如何让关系一致

### 02_cross_domain_sampler.txt
- CrossDomainEpisodeSampler 类设计
- 采样权重配置
- 跨域/同域判断逻辑

### 03_relation_loss.txt
- Relation Consistency Loss 实现
- 梯度流向
- 权重调整策略

### 04_training_integration.txt
- train_meta.py 改动
- 命令行参数
- 训练循环

### 05_experiment_results.txt
- 实验配置
- 结果记录
- 分析总结

---

## 快速参考

### 核心改动文件
1. `common/multi_domain_episode.py` - 新增 CrossDomainEpisodeSampler
2. `train_meta.py` - 新增 relation_consistency_loss()

### 关键参数
```bash
--cross-domain-prob 0.5     # 跨域采样概率
--relation-weight 0.5       # RC Loss 权重
--rc-mode intra_variance     # 同类关系方差模式
--rc-weight-cross 1.5        # 跨域时权重倍数
--rc-weight-same 0.5        # 同域时权重倍数
```
