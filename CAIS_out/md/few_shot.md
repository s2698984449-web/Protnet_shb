# Few-Shot Meta-Learning 快速上手指南

> 本文档面向希望快速复现实验或理解项目流程的研究者/工程师。
>
> **前置知识**: PyTorch 基础，了解 MAML / ProtoNet 元学习方法。
>
> **分析日期**: 2026-08-22


# PowerShell 下, 反引号 `` ` `` 是续行符; bash 用 `\`.
# 注意: 一行写完, 续行符后不能有任何字符 (包括空格/注释).

# 实验 A: 解冻 + 增强 + 默认 ImageNet 权重
python scripts\train_meta.py --n-way 4 --k-shot 5 --q-query 5 `
    --iters 2000 --eval-every 200 --eval-episodes 100 --lr 1e-5 `
    --unfreeze-backbone --augment `
    --out-dir "d:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_4w5s_aug_unfreeze"

# 实验 B: 只加增强，不解冻
python scripts\train_meta.py --n-way 4 --k-shot 5 --q-query 5 `
    --iters 2000 --eval-every 200 --eval-episodes 100 --lr 1e-4 `
    --augment `
    --out-dir "d:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_4w5s_aug_frozen"

# 对照组 C: 什么都不加（基线）
python scripts\train_meta.py --n-way 4 --k-shot 5 --q-query 5 `
    --iters 2000 --eval-every 200 --eval-episodes 100 --lr 1e-4 `
    --out-dir "d:\元强化学习\代码\航空缺陷检测\checkpoints\phase1_4w5s_baseline"


# 1) 验证 PID 20/21 已接入 sampler（18 个 product）
python -c "
import sys; sys.path.insert(0,'.')
from common.multi_domain_episode import build_episode_samplers
s, *_ = build_episode_samplers(
    r'D:\元强化学习\代码\data\MVTEC',
    r'D:\元强化学习\代码\data\GC10-DET',
    r'D:\元强化学习\代码\data\NEU-CLS\NEU-CLS',
    n_way=5, k_shot=5, q_query=5, split='meta_train',
    aero_real_root=r'D:\元强化学习\代码\data\航天\raw\Aero-engine_defect-detect_new',
)
print('PIDs:', sorted(s.product_to_source.keys()))
"

# 2) 域泛化训练（不加 carpaint）
python scripts\train_meta.py --n-way 4 --k-shot 5 --q-query 5 --iters 600 `
    --eval-every 100 --eval-episodes 50 --lr 1e-4 `
    --aero-real-root "D:\元强化学习\代码\data\航天\raw\Aero-engine_defect-detect_new" `
    --out-dir "D:\元强化学习\代码\航空缺陷检测\checkpoints\phase_dg_aero"

# 3) 全域训练（车漆 + 航天 + dropout 都加）
python scripts\train_meta.py --n-way 4 --k-shot 5 --q-query 5 --iters 600 `
    --eval-every 100 --eval-episodes 50 --lr 1e-4 `
    --carpaint-root "D:\元强化学习\代码\data\车\carpaint" `
    --aero-real-root "D:\元强化学习\代码\data\航天\raw\Aero-engine_defect-detect_new" `
    --out-dir "D:\元强化学习\代码\航空缺陷检测\checkpoints\phase_dg_all"

## 1. 项目目录结构

```
D:\元强化学习\代码\
│
├── scripts/                        # ★ 所有运行入口（统一从此启动）
│   ├── train.py                   # Phase 1 元学习训练（MAML / ProtoNet）
│   ├── pretrain.py                # Phase 0 Backbone 预训练
│   ├── eval.py                   # 评测（支持多 seed + 95%CI）
│   ├── run_experiment.py          # 统一运行器（读 YAML 配置启动实验）
│   └── run_experiment.ps1        # PowerShell 批处理入口
│
├── config/                        # ★ 所有实验配置（一实验 = 一 YAML）
│   ├── aerospace/
│   │   └── defaults.yaml         # 38 个航天实验配方
│   └── automotive/
│       └── defaults.yaml          # 49 个车漆实验配方
│
├── outputs/                       # 所有训练产物（自动创建）
│   ├── 航天/
│   │   ├── checkpoints/          # best.pt / iter_*.pt / run_config.json
│   │   ├── weights/              # pretrain_best.pt / aviation_yolo26_*.pt
│   │   ├── logs/                 # 训练日志
│   │   ├── images/               # acc 曲线图
│   │   └── reports/              # final_eval_*.json（多 seed 评测报告）
│   └── 车/
│       └── [同上结构]
│
├── common/                        # 共享核心模块（双轨通用）
│   ├── backbones.py              # YOLO26 视觉编码器
│   ├── multimodal_model.py        # 完整模型（Backbone + Fusion + Head）
│   ├── pretrain_model.py         # Phase 0 多任务预训练模型
│   ├── protonet_loop.py          # ProtoNet episode 损失 + 原型构建
│   ├── maml_loop.py              # MAML inner-loop 自适应
│   ├── maml_origin.py            # learn2learn MAML 封装
│   ├── eval_loop.py              # 多 task 评测（ProtoNet / MAML 统一）
│   ├── task_builder.py           # learn2learn Taskset 构建
│   ├── aviation_datasets.py      # 航空/车漆数据集加载器
│   ├── pretrain_datasets.py      # Phase 0 预训练数据集
│   ├── cmpl_pretrain_datasets.py # 车漆预训练数据集
│   ├── fusion.py                 # 多模态融合策略
│   ├── supcon_loss.py            # Supervised Contrastive Loss
│   ├── episode_aug.py            # Episode 数据增强
│   ├── cross_domain_episode.py   # 跨域 Episode 采样
│   ├── project_paths.py           # 路径解析 + track 默认值
│   └── utils.py                  # 日志、绘图等工具
│
├── data/                         # 原始数据 + 处理后数据
│   ├── 航天/
│   │   ├── processed/aviation/   # 航空叶片 ROI（domain_0/1/2/3）
│   │   ├── processed_greedy/      # 贪心分域版本
│   │   └── raw/                  # AeBAD / BladeSynth 原始数据
│   └── 车/
│       ├── processed/cmpl/        # 车漆 ROI（domain_0/2/3）
│       ├── processed/pretrain/    # 车漆预训练 ROI（CarDD/ASDE/...）
│       └── raw/                  # CarDD/ASDE/NEU 原始数据
│
└── md/                           # 项目文档（设计思路 / 实验报告）
```

---

## 2. 数据流（Data Flow）

### 2.1 整体管线

```
原始数据 (raw/)
    ↓ [数据集转换脚本]
处理后数据 (processed/)
    ↓ [Phase 0: pretrain.py]
预训练权重 (outputs/{track}/weights/*.pt)
    ↓ [Phase 1: train.py]
元学习检查点 (outputs/{track}/checkpoints/*/best.pt)
    ↓ [Phase 2: eval.py]
评测报告 (outputs/{track}/reports/final_eval_*.json)
```

### 2.2 Phase 0 预训练数据流

```
data/{track}/processed/pretrain/
├── cardd/       (车漆: 真实缺陷图像, 4 类)
├── carparts/     (零部件: 8 类)
├── minor_dent/   (小凹陷: 单类)
├── syndcar/      (合成车损: 4 类)
├── blade/        (航空叶片)
└── aebad/       (AeBAD 叶片缺陷)

    ↓ pretrain.py
    ├─ PretrainMultitaskModel (YOLO26 backbone + 多任务 head)
    └─ SupCon + CE 联合训练
         ↓
    outputs/{track}/weights/pretrain_best.pt
    outputs/{track}/checkpoints/pretrain_best.pt
```

### 2.3 Phase 1 元学习训练数据流

```
data/{track}/processed/{dataset_name}/
├── domain_0/    ← 训练域（每个类 N 张 ROI）
├── domain_1/    ← 训练域
├── domain_2/    ← 训练域
└── domain_3/    ← Holdout 测试域

    ↓ train.py
    ┌─ [Phase 0] pretrain_best.pt (Backbone 权重)
    ├─ AviationDefectModel (Backbone + Fusion + Head)
    └─ create_tasksets() → learn2learn Taskset

    Episode 采样:
    ┌─ Support: K 张/类 × N 类 → 计算原型（ProtoNet）
    └─ Query:   Q 张/类 → 与原型比较预测类别

         ↓
    outputs/{track}/checkpoints/{exp_name}/best.pt
```

### 2.4 评测数据流

```
outputs/{track}/checkpoints/{exp_name}/best.pt
    ↓ eval.py (多 seed 评测)
    ├─ 加载模型
    ├─ 在 domain_3 上采样 N 个 episode
    └─ 统计准确率 + 计算 95%CI

         ↓
    outputs/{track}/reports/final_eval_{exp}.json
    {
      "pooled": { "mean": 0.852, "std": 0.12, "ci95_low": 0.842, "ci95_high": 0.862 },
      "per_seed": { "42": { "mean": 0.85, "std": 0.11 }, ... }
    }
```

---

## 3. 训练命令（Quick Start）

> **所有命令均在项目根目录 `D:\元强化学习\代码\` 下运行。**

### 3.1 查看实验列表

```bash
# 航天实验
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --list

# 车漆实验
python scripts/run_experiment.py --config config/automotive/defaults.yaml --list
```

### 3.2 Phase 0 — Backbone 预训练

```bash
# 航天: AeBAD + BladeSynth 联合预训练 (25 epochs)
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp11_p0_full --stage pretrain

# 车漆: CarDD + 辅助任务联合预训练 (25 epochs)
python scripts/run_experiment.py --config config/automotive/defaults.yaml --exp cmpl_p0_v9 --stage pretrain

# Smoke test (少量 epoch 验证链路)
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp11_p0_smoke --stage pretrain
python scripts/run_experiment.py --config config/automotive/defaults.yaml --exp cmpl_p0_v9_smoke --stage pretrain
```

### 3.3 Phase 1 — 元学习训练

```bash
# 航天 ProtoNet 主实验 (exp08, 主结果 85.2%)
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp08_protonet

# 航天 ProtoNet + Trim-Mean + Mahalanobis (exp14, 当前最高 98.4%)
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp14_protonet_trim_mahal

# 车漆 ProtoNet 主实验
python scripts/run_experiment.py --config config/automotive/defaults.yaml --exp cmpl_protonet_v10_meta

# Smoke test (少量迭代验证链路)
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp08_protonet_smoke
```

### 3.4 Phase 2 — 评测

```bash
# 航天: 多 seed 评测 (seeds=42,43,44)
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp08_eval

# 车漆: 多 seed 评测
python scripts/run_experiment.py --config config/automotive/defaults.yaml --exp cmpl_eval_v10_meta
```

### 3.5 参数覆盖

```bash
# 覆盖 YAML 配置中的参数（不修改 YAML 文件）
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp08_protonet --set iters=50 --set eval_every=10
```

### 3.6 命令参数对照表

| YAML 参数 | 命令行覆盖 | 说明 | 默认值 |
|-----------|-----------|------|--------|
| `iters` | `--set iters=300` | 元学习迭代轮数 | 300 |
| `eval_every` | `--set eval_every=50` | 每隔多少 iter 评测一次 | 50 |
| `eval_tasks` | `--set eval_tasks=100` | 每次评测的 episode 数 | 100 |
| `eval_patience` | `--set eval_patience=5` | 早停耐心（0=禁用） | 0 |
| `shots` | `--set shots=3` | 每类 support 样本数 | 5 |
| `ways` | `--set ways=2` | 每 episode 的类别数 | 2 |
| `seed` | `--set seed=42` | 随机种子 | 42 |
| `meta_algorithm` | `--set meta_algorithm=protonet` | 算法: `maml` 或 `protonet` | maml |
| `protonet_distance` | `--set protonet_distance=mahalanobis` | 距离: `euclidean`/`cosine`/`mahalanobis` | euclidean |
| `protonet_prototype_type` | `--set protonet_prototype_type=trim_mean` | 原型: `mean`/`trim_mean`/`mean_shift` | mean |
| `freeze_backbone` | `--set freeze_backbone=false` | 是否冻结 backbone | false |
| `use_multiscale_cff` | `--set use_multiscale_cff=true` | 多尺度 CFF 融合模式 | false |
| `rgb_cbam` | `--set rgb_cbam=true` | CBAM 注意力 | false |
| `train_domains` | `--set train_domains=0,1,2` | 训练域 ID | `0,1,2` |
| `test_domain` | `--set test_domain=3` | 测试域 ID | 3 |

---

## 4. 模块说明

### 4.1 核心模块一览

| 文件 | 类 / 函数 | 职责 |
|------|-----------|------|
| `project_paths.py` | `resolve_track()`, `apply_meta_train_defaults()` | 路径解析 + track 默认超参 |
| `backbones.py` | `YOLO26BackboneEncoder`, `CalibratedFeatureFusion`, `CBAM` | YOLO26 多尺度特征提取 + CFF 融合 + CBAM |
| `multimodal_model.py` | `AviationDefectModel`, `ValueBaseline` | 完整模型（Backbone + Fusion + Head），`embed()` 返回 RGB-only 128d |
| `pretrain_model.py` | `PretrainMultitaskModel` | Phase 0 多任务预训练模型 |
| `fusion.py` | `AsymmetricFusion` | RGB / Thermal / Ultrasonic 非对称融合 |
| `supcon_loss.py` | `SupConLoss` | Supervised Contrastive Loss |
| `task_builder.py` | `create_tasksets()` | learn2learn Taskset 构建 |
| `aviation_datasets.py` | `AviationDefectDataset` | 航空/车漆 ROI 数据集 |
| `pretrain_datasets.py` | `AeBADBinaryPretrainDataset`, `BladeSynthPretrainDataset` | 航空预训练数据集 |
| `cmpl_pretrain_datasets.py` | `CarDDPretrainDataset`, `CarPartsPretrainDataset`, ... | 车漆预训练数据集 |
| `maml_loop.py` | `adapt_supervised()`, `evaluate_query()`, `split_support_query()` | MAML inner-loop |
| `maml_origin.py` | `MAML` | learn2learn MAML 封装 |
| `protonet_loop.py` | `build_prototypes()`, `protonet_episode_loss()`, `evaluate_protonet_query()` | ProtoNet 核心 |
| `eval_loop.py` | `evaluate_tasks()` | 多 task 评测（ProtoNet / MAML 统一） |
| `episode_aug.py` | `augment_support_query()` | Episode 数据增强（旋转/翻转/光照） |
| `cross_domain_episode.py` | `build_cross_domain_pool()`, `sample_cross_domain_episode()` | 跨域 Episode 采样 |
| `utils.py` | `setup_logger()`, `print_logs()` | 日志工具 |

### 4.2 `train.py` — 元学习训练入口

```
输入参数 → parse_args() → apply_meta_train_defaults()
    ↓
create_tasksets(args) → learn2learn Taskset (train + test)
    ↓
AviationDefectModel(yolo_weights, embed_dim) → Backbone + Fusion + Head
    ↓
元学习循环 (iters 次):
    ├─ Episode 采样 (Support + Query)
    ├─ MAML: inner-loop adapt → meta-gradient 反向传播
    └─ ProtoNet: 计算原型 → Episode 损失
    ↓
定期评测: evaluate_tasks() → eval_mean
    ↓
保存: best.pt / iter_*.pt / run_config.json / train_summary.json
```

**关键函数**:

```python
def parse_args()                              # 解析命令行参数
def create_model(args, device)                # 构建 AviationDefectModel + MAML/ProtoNet
def _train_loop(args, experiment_title)      # 元学习主循环
def _apply_exp_dirs(args)                    # 创建 exp_name 子目录
def _save_run_config(args, experiment_title) # 保存 run_config.json
```

### 4.3 `pretrain.py` — Backbone 预训练入口

```
输入参数 → parse_args() → apply_pretrain_defaults()
    ↓
apply_pretrain_variant() → 根据 pretrain_variant 加载任务组合
    ↓
PretrainMultitaskModel(yolo_weights, task_specs) → YOLO26 + 多任务 Head
    ↓
多 Epoch 训练:
    ├─ combined 模式: 每个 step 从所有 task 采样 batch → SupCon + CE
    └─ alternate 模式: 每个 step 轮换 task
    ↓
定期 val: validate() → score (mean_acc 或 global_retrieval_top1)
    ↓
保存: pretrain_best.pt + YOLO26 权重文件
```

**关键函数**:

```python
def parse_args()                              # 解析预训练参数
def build_loaders(args, is_distributed, ...)  # 构建 train/val DataLoader
def build_optimizer(model, args, backbone_mode)# 构建 AdamW 优化器
def validate(model, device, val_loaders, ...) # 验证集评测
def train(args)                               # Phase 0 主循环
```

### 4.4 `eval.py` — 评测入口

```
输入参数 → parse_args()
    ↓
加载 best.pt → AviationDefectModel
    ↓
create_tasksets(args) → test Taskset (domain_3)
    ↓
多 seed 循环 (seeds):
    └─ evaluate_tasks() → per-seed acc 列表
    ↓
CI95 统计 → 输出报告 JSON
```

**关键函数**:

```python
def _ci95(values)                            # 计算 95% 置信区间
def _load_model(args, device, checkpoint)   # 加载模型权重
def _run_single_seed(args, device, ...)     # 单 seed 评测
def run_final_eval(args)                     # 多 seed 评测主函数
def main()                                   # argparse + 调用 run_final_eval
```

### 4.5 `run_experiment.py` — 统一运行器

```
读取 YAML 配置 → _load_yaml()
    ↓
_Resolve_experiment(cfg, exp_name) → 合并 defaults + bases + extends
    ↓
_build_argv(exp, project_root) → 生成命令行参数列表
    ↓
_Apply_meta_train_defaults(cfg) → 自动补全路径（checkpoints/logs/images）
    ↓
subprocess.run([python, script.py, --arg1, val1, ...]) → 启动对应脚本
```

### 4.6 `protonet_loop.py` — ProtoNet 核心

```python
def build_prototypes(support_emb, support_labels, ways,
                    prototype_type='mean', trim_ratio=0.1)
    # 从 support 样本计算每个类的原型向量
    # 支持三种策略:
    #   - 'mean': 简单平均
    #   - 'trim_mean': 去掉最高/最低 10% 后平均（方案 C-①）
    #   - 'mean_shift': MeanShift 聚类多原型

def prototypical_logits(query_emb, prototypes, distance='euclidean', mahal_head=None)
    # 计算 query 特征到各类原型的距离，返回 logits
    # 支持三种距离:
    #   - 'euclidean': 欧氏距离
    #   - 'cosine': 余弦相似度
    #   - 'mahalanobis': 马氏距离（协方差自适应，方案 C-②）

def protonet_episode_loss(model, support, query, domain_meta, ...)
    # ProtoNet episode 损失: support 建原型 + query 分类
    # 返回: meta_loss, train_acc

def evaluate_protonet_query(model, support, query, domain_meta, ...)
    # ProtoNet 评测: 在 query 上计算准确率
    # 返回: loss, acc
```

### 4.7 `eval_loop.py` — 评测循环

```python
def evaluate_tasks(eval_target, test_tasks, domain_meta, shots, ways,
                  adapt_steps, n_tasks, device, fusion_only, meta_algorithm, ...)
    # 统一评测入口，自动分发到 ProtoNet 或 MAML 评测
    # 返回: mean_acc, std, per-task acc 列表

def evaluate_tasks_maml(maml, test_tasks, domain_meta, shots, ways, ...)
    # MAML 评测: clone MAML → inner-loop adapt → query 预测

def evaluate_tasks_protonet(model, test_tasks, domain_meta, shots, ways, ...)
    # ProtoNet 评测: 每个 episode 建原型 → query 分类
```

### 4.8 `task_builder.py` — Episode 构造

```python
def create_tasksets(args)
    # 构建 learn2learn Taskset
    # train_domains → 每个域一个 train Taskset (支持域轮换采样)
    # test_domain → 一个 test Taskset (评测用)
    # 返回: train_tasks[], test_tasks, domain_meta_map

def _parse_exclude_sample_ids(text)
    # 解析 --exclude_sample_ids: 排除特定 specimen（如 QC 失败样本）

class TaskEncoder(nn.Module)
    # 将域元数据 (material_params, process_params) 映射为 64-d 任务嵌入 τ
    # 用于任务条件化的融合权重学习
```

### 4.9 `aviation_datasets.py` — 数据集

```python
class AviationDefectDataset(Dataset)
    # 航空/车漆 ROI 数据集
    # 文件夹结构: domain_X/{class_name}/specimen_Y/{rgb.png, ...}
    # 返回: ((rgb, thermal, ultrasonic), label)
    # 支持 rgb_only 模式（仅 RGB 输入）

    def get_domain_meta()
        # 返回域元数据 dict: material_params, process_params 等
```

---

## 5. 数据规格

### 5.1 航空叶片数据集

```
data/航天/processed/aviation/
├── domain_0/         # 训练域 (材料/工况组合 A)
│   ├── 0_scratch/    # class_id=0: 划痕
│   └── 3_damage/     # class_id=3: 损伤
├── domain_1/         # 训练域 (材料/工况组合 B)
├── domain_2/         # 训练域 (材料/工况组合 C)
└── domain_3/         # Holdout 测试域 (试件 20-26)
    ├── 0_scratch/
    └── 3_damage/

每个 specimen 目录下:
  rgb.png       (3, 256, 256)  RGB 图像
  thermal.png   (1, 320, 240)  红外热成像
  ultrasonic.npy (512,)        超声波波形
```

### 5.2 车漆数据集

```
data/车/processed/cmpl/
├── domain_0/         # 训练域 (ASDE 合成数据)
│   ├── scratch/
│   └── dent/
├── domain_2/         # 训练域 (CarDD 官方 train split)
└── domain_3/         # Holdout 测试域 (CarDD 官方 test, 543 specimen)

data/车/processed/pretrain/
├── cardd/           (4 类, 6882 ROI)
├── carparts/        (8 类, 9084 ROI)
├── minor_dent/      (1 类, 3158 ROI)
└── syndcar/         (4 类, 1295 ROI)
```

---

## 6. 实验配方说明

### 6.1 YAML 配置结构

```yaml
defaults:                    # 全局默认参数
  track: aerospace
  ways: 2
  shots: 5
  meta_algorithm: protonet
  ...

bases:                       # 可复用配置基类
  protonet_unfreeze_real3:  # ProtoNet 解冻 backbone
    meta_algorithm: protonet
    freeze_backbone: false
    train_domains: "0,1,2"

experiments:                 # 具体实验
  exp08_protonet:
    extends: protonet_unfreeze_real3
    description: "主实验描述"
    stage: train
    exp_name: exp08_protonet_unfreeze
    iters: 300
```

### 6.2 常用实验配方速查

#### 航天（航空叶片）

| 实验 | 算法 | 改进点 | 结果 |
|------|------|--------|------|
| `exp04_full` | MAML | 冻结 backbone | 53.4% |
| `exp08_protonet` | ProtoNet | 解冻 backbone | **85.2%** |
| `exp12_protonet_trim_mean` | ProtoNet | +Trim-Mean 原型 | 87.1% |
| `exp13_protonet_mahalanobis` | ProtoNet | +马氏距离 | 88.3% |
| `exp14_protonet_trim_mahal` | ProtoNet | +Trim-Mean+马氏 | **98.4%** |
| `exp16_illumination` | ProtoNet | +光照增强 | 实验中 |

#### 车漆（Automotive）

| 实验 | 算法 | 改进点 | 结果 |
|------|------|--------|------|
| `cmpl_p0_v9` | 预训练 | Per-task SupCon + LDAM | Phase0 score |
| `cmpl_protonet_v10_meta` | ProtoNet | ASDE+CarDD_val→CarDD_test | 实验中 |
| `cmpl_protonet_v11_meta` | ProtoNet | 冻结 backbone 对照 | 实验中 |

---

## 7. 常见问题排查

### 7.1 路径问题

```bash
# 症状: FileNotFoundError: checkpoints not found
# 原因: 从错误目录运行脚本
# 解决: 确保在项目根目录 D:\元强化学习\代码\ 下运行

# 症状: 数据目录不存在
# 原因: data_dir_path 未正确解析
# 解决: 检查 --track 参数，或 --config 中的 data_dir_path
```

### 7.2 依赖问题

```bash
# learn2learn 未安装
pip install learn2learn

# PyYAML 未安装
pip install pyyaml

# GPU 不可用（自动降级到 CPU）
# 检查: torch.cuda.is_available()
```

### 7.3 调试模式

```bash
# smoke test: 少量迭代验证链路
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp08_protonet_smoke

# dry-run: 只打印命令不执行
python scripts/run_experiment.py --config config/aerospace/defaults.yaml --exp exp08_protonet --dry-run
```

---

## 8. 输出文件说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `run_config.json` | 训练超参快照 | 复现实验 |
| `best.pt` | 最优模型权重 | 用于评测 |
| `iter_*.pt` | 中间检查点 | 中断恢复 |
| `train_summary.json` | 训练指标汇总 | 结果记录 |
| `final_eval_*.json` | 评测报告（多 seed CI） | 论文数据 |

---

## 9. 多尺度 CFF 架构（新增）

### 9.1 架构图

```
RGB 图像
    ↓
YOLO26 Backbone + Neck
    ↓
┌────────────────────────────────────────────────────┐
│ Hook 提取 Neck 层特征                               │
│  p2: (B, 64,  80×80)   ← 浅层，高分辨率，细纹理    │
│  p3: (B, 128, 40×40)   ← 中层，中分辨率            │
│  p4: (B, 256, 20×20)   ← 深层，低分辨率，高语义     │
└────────────────────────────────────────────────────┘
    ↓
CFF (Calibrated Feature Fusion)
    ├─ Channel Align: 各尺度 → 128 通道（1×1 conv）
    ├─ Spatial Align: 上/下采样对齐到 p3 分辨率 (40×40)
    ├─ Scale Attention: GAP → MLP → softmax → 各尺度权重
    │     └─ concat(GAP_all) → MLP → (B, 3) 权重
    └─ 加权求和 → 融合特征 (B, 128, 40, 40)
    ↓
CBAM (Channel + Spatial Attention)
    ├─ Channel: avg/max pool → MLP → Sigmoid
    └─ Spatial: concat avg/max → 7×7 conv → Sigmoid
    ↓
Global Avg Pool → 128d 向量
    ↓
ProtoNet / MAML 分类
```

### 9.2 双模式开关

| 模式 | `--use_multiscale_cff` | CBAM 位置 | 特征来源 | ProtoNet 维度 |
|------|----------------------|-----------|---------|--------------|
| 默认（向后兼容） | `false` | 无 | 最后一层 save 特征 | concat(128+128+128)=384d |
| 多尺度 CFF（Q1+Q2+Q3-A） | `true` | CFF 融合之后 | CFF+CBAM 融合特征 | RGB-only 128d（Q4-A） |

### 9.3 启动命令

```powershell
# 多尺度 CFF + CBAM + ProtoNet (PowerShell 续行用反引号 `` ` ``)
python scripts/run_experiment.py `
    --config config/aerospace/defaults.yaml `
    --exp exp17_protonet_cff `
    --set use_multiscale_cff=true `
    --set rgb_cbam=true `
    --set protonet_distance=mahalanobis `
    --set protonet_prototype_type=trim_mean
```
