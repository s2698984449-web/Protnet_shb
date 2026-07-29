# 车漆缺陷检测 — CMPL 元监督学习

基于 CMPL（Car Maintenance Pretrain + Learning）的少样本车漆缺陷元学习链路，主 backbone=YOLO26n，meta 算法=ProtoNet。

---

## 1. 当前主线：cmpl_protonet_v9_meta

| 项目 | 值 |
|------|----|
| 配置 | `experiments/cmpl.yaml` → `cmpl_protonet_v9_meta` |
| Backbone | YOLO26n + car-only SupCon v9 (`cmpl_yolo26_supcon_v9.pt`) |
| P0 预训练数据 | cardd_cls(train 6882 + val 2074) + carparts(9084) + minor-dent(1500) + syndcar(5500) + asde binary |
| P1 meta-train | cmpl/domain_0（ASDE 二分类全图） |
| P2 meta-test | cmpl/domain_3（CarDD test, 543 specimen） |
| Meta-Test 2-way 5-shot | 100 tasks, 当前 best **0.848 acc** (best eval 0.8480 @ iter 100, 早停 iter 200) |

### 1.1 启动

```powershell
cd D:\元强化学习\代码
python scripts/run_experiment.py --config experiments/cmpl.yaml --exp cmpl_protonet_v9_meta
```

---

## 2. P0 Pretrain 决策与历史对比

### 2.1 全部 P0 backbone 历史 meta-test 结果（domain_3 / CarDD test, 100 tasks 2-way 5-shot）

| 日期 | 实验 | P0 数据组成 | best acc |
|------|------|-------------|----------|
| 2026-07-14 | cmpl_protonet_classsplit3 | **NEU + MVTec_tex + MVTec_obj**（class-split 链路） | **0.612** |
| 2026-07-10 | cmpl_protonet_track2 | CardDD only（track2 早期链路） | 0.623 |
| 2026-07-25 | cmpl_protonet_v7_meta | **cardd + carparts + minor_dent + syndcar + asde_binary**（v7 car-only SupCon） | **0.834** |
| 2026-07-27 | cmpl_protonet_v9_meta | cardd + carparts + v9 SupCon（per-task + LDAM） | 0.848（当前最优, 终端 1.txt 2026-07-27 21:11:59） |

### 2.2 NEU + MVTec 结论

> **🕒 2026-07-27 已确认**：NEU + MVTec 预训练 backbone 对车漆元测试**有害**。
> 历史证据：NEU+MVTec pretrain → 0.612 vs cardd+carparts pretrain → 0.834，**相差 22 个百分点**。
> **根因**：NEU 是钢材表面、MVTec 是工业物件/纹理，与车漆 ROI 视觉分布不一致——骨干学到工业纹路偏置，反而不利于车漆细粒度分类。
> **状态**：✅ 已删除（v7+ 不再使用 NEU/MVTec，active_tasks 仅含 cardd/carparts/minor_dent/syndcar/asde）。

### 2.3 v9 vs v7 关键改动

| 改动 | v7 | v9 |
|------|----|----|
| SupCon 损失 | 单一全局 SupCon | per-task SupCon（每个 car 类单独对比） |
| 类别均衡 | 无 | LDAM Loss（长尾均衡） |
| Pretrain epochs | 30 | 30（一致） |
| 早停 patience | 5 | 5 |

**提升**：+0.014 acc（0.834 → 0.848）。提升小，说明 SupCon 已有充分能力。

---

## 3. 数据资源盘查与未利用资源

### 3.1 当前数据组成

| 数据 | 样本数 | 用途 | 状态 |
|------|--------|------|------|
| CarDD train (specimen) | 3287 | P0 pretrain train | ✅ 使用 |
| CarDD val (specimen) | 1228 | P0 pretrain val | ✅ 使用 |
| CarDD test (specimen) | 272 | cmpl/domain_3 meta-test | ✅ 使用（不泄漏） |
| cardd_cls (ROI crop) | 8956 | P0 pretrain train+val | ✅ 使用 |
| carparts (8 类 ROI) | 9084 | P0 pretrain train | ✅ 使用 |
| minor_dent | ~1500 | P0 pretrain train | ✅ 使用 |
| syndcar | ~5500 | P0 pretrain train | ✅ 使用 |
| ASDE binary (全图) | 6350 | cmpl/domain_0 meta-train | ✅ 使用 |

### 3.2 未利用资源 ⚠️

| 数据 | 样本数 | 位置 | 浪费原因 |
|------|--------|------|---------|
| **meta_train/roidb（carparts + minor_dent + syndcar 合并）** | **13,520** | `data/车/processed/meta_train/roidb/` | 完全闲置，从未被任何阶段使用 |
| **carparts 8 类作为 N-way episodes** | 9084 | `data/车/processed/pretrain/carparts/` | 仅做 P0 预训练分类，未做 N-way 验证 |
| **CarDD val ROI 作为 meta-train pool** | 2074 | `data/车/processed/pretrain/cardd_cls/val/` | 仅做 P0 pretrain val，未进 meta-train |

**🕒 2026-07-27 思路**：
1. `meta_train/roidb` 13,520 张应接入 cmpl/domain_2（与 domain_3 同分布），成为 meta-train 主池（替换或补充 ASDE）
2. carparts 8 类应做 N-way scaling 验证（2-way scratch vs dent → 5-way → 8-way）
3. CarDD val 是同分布扩展集，可作 meta-val（不是 meta-train，避免与 test 信息泄露争议）

### 3.3 Specimen 重叠检查

```
pretrain/cardd_cls 总独立 specimen : 3318  (CarDD train 3287 + val 1228, 去重)
cmpl/domain_3 独立 specimen      : 272   (CarDD test)
重叠                          : 0    ← 无泄漏
```

**🕒 2026-07-27 已确认**：CarDD test（domain_3）从未出现在 P0 pretrain 数据中。**无数据泄漏**。

---

## 4. 设计原则与未来方向

### 4.1 第一性原理

```
研究问题:  少样本 + 跨域缺陷检测能否仅靠多任务自监督 pretrain + 元学习达到工业可用门槛?
        ↓
ProtoNet 假设:  train episodes 和 test episodes 必须来自同一分布
              (Snell et al. 2017 论文核心结论)
        ↓
车漆链路已对齐:  RGB_SIZE=(256,256) 强制 resize 解决了尺度问题
              (v7_meta 0.834 已证明)
        ↓
待解方向:       原型计算精度（CLIP text-anchor 修正）
             N-way scaling（carparts 8 类天然支持）
             跨域 episode sampling（多 domain 共同采样）
```

### 4.2 待落地方向（按 ROI 排序）

| 方向 | 思路 | 预期收益 | 状态 |
|------|------|----------|------|
| A. meta_train/roidb 进 P1 | 把 13,520 张接入 cmpl/domain_2 替换 ASDE 主训 | +5~10% acc | 📝 想法未做 |
| B. carparts N-way 验证 | 2/5/8-way episode 评测 ProtoNet scaling | 方法论验证 | 📝 想法未做 |
| C. CLIP text-anchor 修正 | YOLO-LA 方案：用文本嵌入修正 ASDE intact→dent proxy | +2~5% acc | 📝 想法未做 |
| D. ASDE 数据增强角色 | 不做 P1 主训，做 support set 数据增强源 | +1~2% acc | 📝 想法未做 |
| E. 跨域 episode sampling | support/query 各从不同域采样，learn domain-invariant | +2~3% acc | 📝 想法未做 |
| F. Uniform episode difficulty | 按难度均匀采样 | +1~2% acc | 📝 想法未做 |

### 4.3 不再做的方向

| 方向 | 撤回原因 |
|------|---------|
| NEU + MVTec 进车漆 backbone | 历史证据：meta-test 掉 22 个百分点 |
| ASDE 视觉分布不匹配（拒绝使用） | resize 已解决（v7_meta 0.834） |

---

## 5. 关键代码入口

```
D:\元强化学习\代码\
├── common/
│   ├── aviation_datasets.py    # 多模态 Dataset（RGB_SIZE=256×256 强制 resize）
│   ├── backbones.py            # YOLO26 backbone
│   ├── multimodal_model.py     # 完整模型（含 ProtoNet head）
│   └── maml_loop.py            # meta 训练循环
├── scripts/
│   ├── run_experiment.py       # ★ 统一实验入口
│   └── convert_*.py            # 数据转换脚本
├── data/车/
│   ├── processed/
│   │   ├── pretrain/
│   │   │   ├── cardd_cls/      # CarDD train+val ROI（3318 specimen）
│   │   │   ├── carparts/       # 9084 张 ROI
│   │   │   ├── minor_dent/
│   │   │   └── syndcar/
│   │   ├── meta_train/roidb/   # ⚠️ 13,520 张未利用
│   │   └── cmpl/
│   │       ├── domain_0/       # ASDE 二分类（meta-train）
│   │       └── domain_3/       # CarDD test 272 specimen（meta-test）
├── 车/
│   ├── weights/                # 放置 yolo26n.pt, cmpl_yolo26_supcon_v7.pt, v9.pt
│   ├── checkpoints/            # 训练产物 best.pt
│   ├── logs/                   # 训练日志 *.log
│   └── train.py                # 训练入口
└── 航空缺陷检测/
    └── experiments/cmpl.yaml   # ★ v9 主实验配置
```

---

## 6. 时间线与决策记录

- **2026-07-05**：cardd_cls 数据集转换完成（pretrain/train + pretrain/val）
- **2026-07-06**：cmpl_anil_ep24 (MAML frozen, ASDE) → 0.532 acc（建立 baseline）
- **2026-07-06**：cmpl_anil_v2 (MAML frozen + episode aug) → 0.565 acc
- **2026-07-07**：cmpl_protonet_v1 (ProtoNet frozen, ASDE) → 0.644 acc（ProtoNet 显著优于 MAML）
- **2026-07-07**：cmpl_protonet_unfreeze_v1 (ProtoNet unfreeze) → 0.721 acc（+8%）
- **2026-07-10**：cmpl_protonet_track2 (CardDD pretrain backbone) → 0.623 acc（早卡 P0）
- **2026-07-14**：cmpl_protonet_classsplit3 (NEU+MVTec pretrain) → **0.612 acc**（确认 NEU+MVTec 无效）
- **2026-07-25**：cmpl_protonet_v7_meta (car-only SupCon v7) → 0.834 acc（+20% 突破）
- **2026-07-27**：cmpl_protonet_v9_meta (v9 SupCon + LDAM) → **0.848 acc** (best eval @ iter 100, 早停 iter 200)
- **2026-07-27**：第一性原理复盘：定位未利用资源（meta_train/roidb 13,520）+ carparts N-way 验证方向
