# CA-IS-MetaProto 当前链路状态 (2026-09-23)

## ⚠️ 重要认知修正

**之前的实验数字 (exp08/12/13/14, cmpl_v10/v13, phase1_4w5s_aug_unfreeze) 与"当前架构水平"不匹配**, 原因如下:

| 历史实验 | Pipeline | 模型 | 含 ConditionEncoder? | 含 DomainReliability? | 课程学习? | 能代表当前架构? |
|----------|----------|------|----------------------|----------------------|----------|---------------|
| exp08 (85.2%) | 旧 Aviation | AviationDefectModel | ❌ | ❌ | ❌ | ❌ 不能 |
| exp12 (97.7%) | 旧 Aviation | AviationDefectModel | ❌ | ❌ | ❌ | ❌ 不能 |
| exp13 (95.8%) | 旧 Aviation | AviationDefectModel | ❌ | ❌ | ❌ | ❌ 不能 |
| **exp14 (98.4%)** | 旧 Aviation | AviationDefectModel | ❌ | ❌ | ❌ | ❌ 不能 |
| cmpl_v10 (84.0%) | 旧 Aviation | AviationDefectModel | ❌ | ❌ | ❌ | ❌ 不能 |
| cmpl_v13_adaptive | 旧 Aviation | AviationDefectModel | ❌ | ❌ | ❌ | ❌ 不能 |
| phase1_4w5s_aug_unfreeze (61.85%) | 旧 train_meta.py | MultiDomainPretrainModel | ✅ | ❌ | ❌ | ❌ 不能 |
| phase1_3w5s | 旧 train_meta.py | MultiDomainPretrainModel | ✅ | ❌ | ❌ | ❌ 不能 |
| **C4 Full (本次新跑)** | **新主链路** | **MultiDomainPretrainModel** | **✅** | **✅** | **✅** | ✅ **唯一代表** |

**结论**: 项目知识包中"78.5% / 70→78.5 消融链"在 CA-IS-MetaProto 新管线 (train_meta.py) 下尚未真实跑出来过, 需要本次 C0→C4 重新建立基线.

---

## 🚀 本次新增脚本

### 1. `scripts/eval_meta.py` — 新管线多 seed 95% CI 评测

**解决的问题**: `scripts/eval.py` 走的是 `AviationDefectModel`, 与新版 `train_meta.py` (MultiDomainPretrainModel) **架构不兼容**, 无法直接 eval 新管线产出的 checkpoint.

**做法**: 复用 `train_meta.py` 的 `evaluate()` 函数 (新管线), 加载 train_meta.py 产出的 `best.pt / final.pt`, 重建 `MultiDomainPretrainModel + MahalanobisHead + DomainReliability`, 跑多 seed 评测, 输出 pooled 95% Bootstrap CI + JSON 报告.

**用法**:
```powershell
python scripts/eval_meta.py `
  --checkpoint "D:/元强化学习/代码/航空缺陷检测/checkpoints/phase1_4w5s_C4_full/best.pt" `
  --seeds 42,43,44 `
  --n-way 4 --k-shot 5 --q-query 5 `
  --eval-tasks 100 `
  --out-dir "D:/元强化学习/代码/航空缺陷检测/reports/final_eval_C4_full"
```

### 2. `scripts/run_ablation_cais.py` — 新管线消融矩阵

**5 个消融点 (C0→C4)**:

| 实验 | distance | prototype | curriculum | domain_reliability | 说明 |
|------|----------|-----------|------------|-------------------|------|
| **C0** | euclidean | mean | ❌ | ❌ | 新管线基线 (无任何创新) |
| **C1** | euclidean | mean | ✅ | ❌ | + 课程学习 |
| **C2** | mahalanobis | mean | ✅ | ❌ | + Mahalanobis |
| **C3** | mahalanobis | trim_mean | ✅ | ❌ | + Trim-Mean |
| **C4** | mahalanobis | trim_mean | ✅ | ✅ | **Full = CA-IS-MetaProto 当前架构** |

**用法**:
```powershell
# 跑全部 5 个消融 (顺序, 串行)
python scripts/run_ablation_cais.py

# 只跑 C4 Full
python scripts/run_ablation_cais.py --only C4_full

# 自定义 iters / eval-tasks
python scripts/run_ablation_cais.py --iters 2000 --eval-tasks 100
```

---

## 🔥 流水线启动情况

### `scripts/run_cais_pipeline.py` 编排

- **Shell ID**: 739773, PID 29112
- **启动时间**: 2026-09-23 11:16 (UTC+8)
- **当前阶段**: stage 1/5 C0_baseline (iter 20/200, 正常收敛)
- **5 阶段总耗时估算**: 5 × (训练 + 多 seed 评测 + 冷却)
  - `--iters 200` 快速验证跑: 训练 ~11 min/阶段, 评测 ~5 min/阶段, 冷却 60s → **~80 分钟**
  - `--iters 2000` 主跑: 训练 ~115 min/阶段, 评测 ~5 min/阶段, 冷却 60s → **~10 小时**
- **总报告路径**: `航空缺陷检测/reports/pipeline_report.json`
- **阶段日志**: 每个阶段单独写到 `航空缺陷检测/checkpoints/phase1_4w5s_<name>/pipeline.log`
- **多 seed 评测**: 每个阶段在 `航空缺陷检测/reports/final_eval_<name>/final_eval_phase1_4w5s_<name>.json`

**实时监控**:
```powershell
# 父进程状态
Get-Process -Name python | Select-Object Id, StartTime, CPU, WorkingSet | Format-Table -AutoSize

# pipeline 总输出
Get-Content "D:\元强化学习\代码\航空缺陷检测\reports\pipeline_console.log" -Encoding UTF8 -Tail 30 -Wait

# 当前阶段训练日志
Get-Content "D:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_4w5s_C0_baseline\train_meta.log" -Tail 30 -Wait

# 报告 (阶段完成后会写入)
Get-Content "D:\元强化学习\代码\航空缺陷检测\reports\pipeline_report.json"
```

**当前流水线启动命令** (--iters 200 快速验证跑):
```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
python -u scripts/run_cais_pipeline.py --clear-existing --cooldown-seconds 60 --iters 200 > `
  "D:\元强化学习\代码\航空缺陷检测\reports\pipeline_console.log" 2>&1
```

**主跑模式** (--iters 2000 全量训练, ~10 小时):
```powershell
python -u scripts/run_cais_pipeline.py --clear-existing --cooldown-seconds 60 --iters 2000 > `
  "D:\元强化学习\代码\航空缺陷检测\reports\pipeline_console.log" 2>&1
```

**单跑某个阶段** (e.g. 只跑 C4_full, 适合补跑):
```powershell
python -u scripts/run_cais_pipeline.py --only C4_full --iters 2000 --force
```

---

## 📋 待办 (按优先级)

1. **🔥 [进行中] C0 → C4 流水线快速验证跑** (--iters 200, ~80 分钟, 后台运行)
   - Shell ID 739773, PID 29112
   - C0_baseline iter 20/200, 正常收敛 (acc=0.700)
2. **⏳ 验证跑完成后**:
   - 看 `pipeline_report.json` 的 5 阶段简表
   - 若数字合理, 用 --iters 2000 主跑 (10 小时)
3. **⏳ 更新项目知识包** — 用新管线下的真实数字替换旧假设数字

---

## ⚠️ 当前限制

- **设备**: CPU (无 CUDA) — 每个实验 ~115 分钟
- **pretrain-ckpt**: 默认 `outputs/multi_domain_smoke.pt` 不存在, 走 ImageNet ResNet18 fallback
  - 与历史实验 (phase1_4w5s_aug_unfreeze 等) 条件一致
  - 如要 Phase 0 真正多任务联合预训练, 需要先跑 `pretrain_multi_domain.py` 产出 pretrain
- **数据源**: MVTEC + GC10 + NEU (16 个 product), 无车漆 / 航空真实数据

---

**文档维护**: 本文档由 Claude-Fable-5-1 在 2026-09-23 11:17 UTC+8 更新, 反映 5 阶段流水线 (run_cais_pipeline.py) 启动后状态.
