# 航空构件多模态缺陷检测 — MAML 元监督学习

基于 MAML 的跨域小样本航空构件缺陷检测（RGB + 热像 + 超声，当前阶段以 RGB 为主）。

---

## 1. 项目目录结构

```
D:\元强化学习\代码\
├── common/                              # 共享核心代码
│   ├── __init__.py                      # 优先加载 pip 版 learn2learn
│   ├── aviation_datasets.py             # 多模态 Dataset（缺模态自动填零）
│   ├── backbones.py                     # YOLO26 / 热像 / 超声编码器
│   ├── fusion.py                        # 非对称多模态融合
│   ├── task_encoder.py                  # 域元数据 → 任务嵌入 τ
│   ├── knowledge_base.py                # 材料/缺陷先验知识库
│   ├── multimodal_model.py              # 完整模型 + Stop 头 + RL
│   ├── task_builder.py                  # learn2learn Taskset 构建
│   ├── maml_origin.py                   # MAML 算法实现
│   ├── maml_loop.py                     # 内/外环、三阶段课程 w_rl
│   └── utils.py                         # 日志 setup_logger / print_logs
│
├── data/
│   ├── aviation/                        # MAML 格式训练数据
│   │   ├── domain_0/                    # 元训练域（试件 1–19、2023 之一部分）
│   │   ├── domain_1/
│   │   ├── domain_2/
│   │   └── domain_3/                    # 元测试域（试件 20–26，完全留出）
│   │       └── meta.json                # 含 split_policy 划分说明
│   └── Aero-engine_defect-detect_new/   # 原始 YOLO 标注数据
│
├── scripts/
│   ├── convert_aero_engine_dataset.py   # 原始 YOLO → aviation 格式 + 域划分
│   └── check_aviation_data.py           # 训练前数据校验
│
├── learn2learn-0.2.1/                   # learn2learn 源码备份（需 pip 安装后使用）
│
└── 航空缺陷检测/                         # ★ 训练入口（本目录）
    ├── train.py                         # ★ 全部超参数定义处（无独立 config 文件）
    ├── maml_train.py                    # 训练主循环
    ├── requirements.txt                 # Python 依赖
    ├── weights/                         # 放置 yolo26n.pt（见 weights/README.md）
    ├── logs/                            # 训练日志 *.log
    ├── checkpoints/                     # 模型权重 *.pt
    └── images/                          # 准确率曲线 *.png
```

---

## 2. 参数配置在哪里？

**本项目没有独立的 `config.yaml` / `config.json` 文件。**

所有训练超参数集中在：

| 文件 | 作用 |
|------|------|
| **`train.py`** → `parse_args()` | **全部训练超参数**（MAML、数据域、模型、课程学习） |
| `maml_train.py` | 读取 `args`，不含默认值 |
| `scripts/convert_aero_engine_dataset.py` | 数据转换参数（`--test_bases` 等） |
| `data/aviation/domain_*/meta.json` | 域级先验（材料参数、缺陷历史、融合先验） |
| `common/knowledge_base.py` | 知识库默认常量（`DEFAULT_FUSION_PRIOR` 等） |
| `common/fusion.py` | 融合层结构超参（`alpha/beta/gamma=1.0`，写死在代码里） |

修改训练参数：**直接改 `train.py` 默认值，或在命令行传参覆盖。**

---

## 3. 环境准备

```powershell
conda activate MAML
cd D:\元强化学习\代码\航空缺陷检测
pip install -r requirements.txt
```

**learn2learn：**

- 优先使用 `MAML` 环境中 `pip install learn2learn` 的版本
- 项目内 `learn2learn-0.2.1/` 为源码备份，**不能**直接 `sys.path` 导入（Cython 未编译会报错）
- 若需从源码安装：

```powershell
pip install cython
pip install -e D:\元强化学习\代码\learn2learn-0.2.1
```

---

## 4. 数据准备与域划分

### 4.1 数据格式

每个样本目录：

```
data/aviation/domain_{0~3}/{class_name}/{sample_id}/
    rgb.jpg           # 必须
    thermal.jpg       # 可选，缺省填零
    ultrasonic.npy    # 可选，缺省填零
```

缺陷类别（统一 4 类，当前数据有效 2–3 类）：

| 文件夹 | 含义 |
|--------|------|
| `0_scratch` | 划痕 |
| `1_contamination` | 污染（dirty/stain/dot） |
| `2_crease` | 折痕（当前极少） |
| `3_damage` | 损伤 |

### 4.2 当前域划分策略（specimen_holdout）

| domain | 用途 | 试件 base ID | 说明 |
|--------|------|-------------|------|
| **0, 1, 2** | 元训练 | 1–19、2023 | 轮询分配，同一试件所有增强图只在一个域 |
| **3** | 元测试 | **20–26** | 与训练域 **0 试件重叠** |

转换命令（在项目根目录执行）：

```powershell
cd D:\元强化学习\代码
python scripts\convert_aero_engine_dataset.py --raw_root data\Aero-engine_defect-detect_new --out_root data
```

验证数据：

```powershell
python scripts\check_aviation_data.py --data_dir_path data
```

确认新划分：打开 `data\aviation\domain_3\meta.json`，应含：

```json
"split_policy": {
  "strategy": "specimen_holdout",
  "test_bases": ["20", "21", "22", "23", "24", "25", "26"]
}
```

### 4.3 日志里的 0/1/2/3 是什么？

终端打印：

```
Training domains: 0,1,2.
Testing domain: 3.
```

指的是 **文件夹编号** `domain_0` ~ `domain_3`，不是旧的 `%3` 哈希或 YOLO val 划分。  
**编号不变，变的是每个 domain 里放了哪些试件。**

### 4.4 当前各域样本量（转换后）

| domain | 样本数 | 有效类别 |
|--------|--------|----------|
| 0 | 69 | scratch + damage |
| 1 | 99 | scratch + contamination + damage |
| 2 | 60 | scratch + damage |
| 3 | 63 | scratch + damage |

---

## 5. YOLO26 权重（可选但推荐）

SSL 无法自动下载时，手动放置：

```
航空缺陷检测/weights/yolo26n.pt
```

下载地址：https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

未放置时：从 `yolo26n.yaml` 随机初始化（不联网，效果较差）。

---

## 6. 训练指令（Phase 1 推荐）

### 6.0 统一实验入口（推荐）

历史 `scripts/run_exp*.ps1` 已归档至 `scripts/archive/`，请改用：

```powershell
cd D:\元强化学习\代码\航空缺陷检测

# 列出全部实验 / 网格
.\scripts\run_experiment.ps1 -List

# 训练（当前主线 ProtoNet）
.\scripts\run_experiment.ps1 -Exp exp08_protonet -Stage train

# 3-seed 终评
.\scripts\run_experiment.ps1 -Exp exp08_eval

# 超参网格（如 exp04 fast_lr）
.\scripts\run_experiment.ps1 -Grid exp04_fastlr

# 临时改参 / 试跑
.\scripts\run_experiment.ps1 -Exp exp08_protonet -Set iters=20 -DryRun
```

实验配方定义在 **`experiments/aviation.yaml`**（exp01–exp10 训练/终评/诊断/网格）。  
跨平台可直接用 Python：

```powershell
python scripts/run_experiment.py --exp exp08_protonet --stage train
python scripts/run_experiment.py --exp exp08_eval

# P0 注意力 ablation (CBAM, 同 exp08 协议)
python scripts/run_experiment.py --exp exp10_protonet_cbam --stage train
python scripts/run_experiment.py --exp exp10_eval
```

---

**Phase 1：** 冻结 YOLO26 骨干，内环只适应 fusion 分类头；外环 meta-train fusion + 128 维投影层。  
特征由 **yolo26n.pt 预训练权重** 提取（通用视觉表征）；泛化靠 **试件 hold-out + 多 task 评估**。

```powershell
conda activate MAML
cd D:\元强化学习\代码\航空缺陷检测
python train.py --iters 300 --rgb_only True --data_dir_path D:\元强化学习\代码\data --freeze_backbone True
```

训练每 50 iter 在 domain 3 上跑 **50 个 task** 平均准确率（稳定指标），并据此存 `*_best.pt`。

**训练后独立评估（100 task）：**

```powershell
python eval.py --checkpoint checkpoints/AviationMAML_2w_3s_rgb_frozen_yolo26_best.pt --data_dir_path D:\元强化学习\代码\data --eval_tasks 100
```

取消冻结（Phase 2）：`--freeze_backbone False`

---

## 7. 全部训练参数说明（`train.py` 默认值）

### 7.1 MAML / 元学习

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ways` | `2` | N-way：每个 task 抽 N 个缺陷类 |
| `--shots` | `3` | K-shot：每类 K 张 support + K 张 query |
| `--meta_lr` | `0.001` | 外环学习率（Adam，更新初始参数 θ） |
| `--fast_lr` | `0.01` | 内环适应步长 α |
| `--adapt_steps` | `5` | 内环梯度更新步数 |
| `--meta_batch_size` | `2` | 每 iter 聚合的 task 数量（4GB 默认） |
| `--iters` | `300` | 元训练总迭代次数 |
| `--first_order` | `True` | `True`=FOMAML（4GB 默认）；`False`=二阶 MAML |
| `--grad_clip` | `1.0` | 外环梯度裁剪 |

**二阶 vs 一阶：**

- `first_order=True`（默认）：FOMAML，4GB 可跑，仍是元学习
- `first_order=False`：二阶 MAML，需大显存，配合 `--meta_batch_size 4`

### 7.2 数据域

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir_path` | `../data` | 数据根目录（含 `aviation/`） |
| `--train_domains` | `0,1,2` | 元训练域编号 |
| `--test_domain` | `3` | 元测试域编号 |
| `--train_task_num` | `40` | 每训练域 Taskset 大小 |
| `--test_task_num` | `20` | 测试 Taskset 大小 |

### 7.3 模型

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--embed_dim` | `128` | 三模态编码输出维度 |
| `--yolo_weights` | `./weights/yolo26n.pt` | YOLO26 backbone 本地权重 |
| `--yolo_imgsz` | `640` | YOLO26 输入尺寸（encoder 内 resize） |
| `--rgb_only` | `False` | `True` 时仅 RGB，热像/超声强制为零 |

### 7.4 三阶段元监督课程

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage1_end` | `100` | 阶段 1 结束 iter（纯监督，w_rl=0） |
| `--stage2_end` | `200` | 阶段 2 结束 iter（w_rl 线性增至 0.3） |
| `--w_rl_max` | `0.5` | 阶段 3 最大 RL 权重 |

| 阶段 | iter 范围 | w_rl | 行为 |
|------|-----------|------|------|
| 1 纯监督 | 0 – stage1_end | 0.0 | 只优化分类 CE |
| 2 混合 | stage1_end – stage2_end | 0 → 0.3 | 加入 Stop 头 RL |
| 3 效率 | stage2_end – iters | 0.5 | 内环步数减至 adapt_steps-2 |

### 7.5 运行 / 输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cuda` | `True` | 使用 GPU（不可用则自动 CPU） |
| `--seed` | `42` | 随机种子 |
| `--log` | `True` | 终端 + 文件日志 |
| `--log_path` | `./logs` | 日志目录 |
| `--plot` | `True` | 保存准确率曲线 |
| `--plot_path` | `./images` | 曲线目录 |
| `--plot_step` | `50` | 每隔 N iter 存图 |
| `--checkpoint` | `True` | 保存模型 |
| `--checkpoint_path` | `./checkpoints` | 权重目录 |
| `--checkpoint_step` | `50` | 每隔 N iter 存 checkpoint |

---

## 8. 终端输出说明

训练开始后示例：

```
[2026-06-23 14:30:01] Experiment: AviationMAML_2w_3s_rgb_yolo26
[2026-06-23 14:30:02] Training domains: 0,1,2.
[2026-06-23 14:30:02] Testing domain: 3.
[2026-06-23 14:30:05] Training with GPU.

[2026-06-23 14:30:45] Iteration 1:
[2026-06-23 14:30:45] Meta Train Error: 0.4521.
[2026-06-23 14:30:45] Meta Train Accuracy: 0.5479.
[2026-06-23 14:30:45] Meta Test Error: 0.5123.
[2026-06-23 14:30:45] Meta Test Accuracy: 0.4877.
[2026-06-23 14:30:45] Mean Support Depth: 1.0.
[2026-06-23 14:30:45] w_rl: 0.0.
```

| 字段 | 含义 |
|------|------|
| `Meta Train Error/Accuracy` | 训练域（0/1/2）query 集表现 |
| `Meta Test Error/Accuracy` | 测试域（3，试件 20–26）跨域表现 |
| `Mean Support Depth` | Stop 头 RL 相关 |
| `w_rl` | 当前 RL 权重（前 100 iter 为 0） |

输出文件：

| 类型 | 路径 |
|------|------|
| 日志 | `logs/AviationMAML_2w_3s_rgb_yolo26.log` |
| 最佳模型 | `checkpoints/AviationMAML_2w_3s_rgb_yolo26_best.pt` |
| 曲线图 | `images/AviationMAML_2w_3s_rgb_yolo26_{iter}.png` |

---

## 9. 代码阅读路线（熟悉代码）

```
train.py                  入口，解析全部参数
    └── maml_train.py     训练循环
            ├── task_builder.py      构建 learn2learn Taskset
            │       └── aviation_datasets.py   加载 RGB/热像/超声
            ├── multimodal_model.py  完整模型
            │       ├── backbones.py           YOLO26 等编码器
            │       ├── fusion.py              多模态融合
            │       ├── task_encoder.py        任务嵌入
            │       └── knowledge_base.py      先验知识
            ├── maml_loop.py         内环适应 + 外环 meta loss + 课程 w_rl
            └── maml_origin.py       MAML clone / update
```

**一次 iter 的流程：**

1. 随机选一个训练域（0/1/2）
2. 采样 `meta_batch_size` 个 N-way K-shot task
3. 内环：在 support 集上 `adapt_steps` 步监督适应
4. 外环：在 query 集上计算 meta loss，反传更新 θ
5. 同时在测试域 3 采样 task 评估跨域准确率
6. 按课程更新 `w_rl` 和 adapt_steps

---

## 10. 多模态扩展（目前只有 RGB）

| 阶段 | 做法 |
|------|------|
| Phase A（当前） | `--rgb_only True`，只有 RGB 有效 |
| Phase B | 去掉 `--rgb_only`，热像/超声缺省填零，融合网络仍可跑 |
| Phase C | 同目录补 `thermal.jpg` + `ultrasonic.npy` 即为真多模态 |

缺模态时 `aviation_datasets.py` 自动填零，无需改代码。

---

## 11. 数据转换参数（`convert_aero_engine_dataset.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--raw_root` | `../data/Aero-engine_defect-detect_new` | 原始 YOLO 数据 |
| `--out_root` | `../data` | 输出根目录 |
| `--test_bases` | `20,21,22,23,24,25,26` | 留作 domain_3 的试件 ID |
| `--no_clean` | — | 不删除旧 domain 目录（默认会先清空） |

---

## 12. 常见问题

**Q：报错 `DLL load failed while importing _imaging`？**  
A：MAML 环境中 **Pillow 11.x 与 torchvision 0.16 在 Windows 上不兼容**。执行：

```powershell
conda activate MAML
pip install "Pillow>=10.3.0,<11"
```

然后重新训练。已在 `requirements.txt` 中锁定 `Pillow<11`。

**Q：报错 `No module named learn2learn`？**  
A：`conda activate MAML`，确认 `pip show learn2learn`。

**Q：YOLO 权重 SSL 下载失败？**  
A：手动下载放到 `weights/yolo26n.pt`，或用 yaml 随机初始化。

**Q：`ways=5` 报错或效果差？**  
A：当前数据每域有效 2 类为主，用 `--ways 2`。

**Q：报错 `CUDA out of memory`？**  
A：默认已是 4GB 配置。若仍 OOM，试 `--meta_batch_size 1`；大显存二阶用 `--first_order False --meta_batch_size 4`。

**Q：domain 3 和训练域是什么关系？**  
A：domain_3 = 试件 20–26，与 domain_0/1/2 试件完全不重叠。

---

更多数据格式说明见 [../data/aviation/README.md](../data/aviation/README.md)。


#  Blender 是一个独立安装的桌面软件，而不是像 Python 库那样通过 pip install 就能直接安装的普通依赖。

它是一款开源免费的全功能 3D 创作套件。虽然它内部自带了 Python 解释器供脚本调用（这也是为什么生成脚本需要用到它），但你必须先在操作系统中安装 Blender 软件本体。

为什么需要单独安装？
因为生成 BladeSynth 数据集的脚本，本质上是调用 Blender 的后台命令行接口（CLI）来进行物理渲染。如果你的电脑上没有安装这个软件，脚本在调用 blender 命令时就会报错。

如何安装与配置？
你可以根据自己的操作系统选择以下方式：

1. Windows 用户（推荐）
前往 Blender 官网（https://www.blender.org/）下载对应版本的 .msi 安装包。
双击安装，建议勾选 Python Scripting（脚本开发支持）和 System Integration（右键菜单集成）等关键组件。
安装完成后，确保在命令行（CMD/PowerShell）中输入 blender --version 能正常返回版本号，说明环境变量已配置好。

2. Linux 用户
可以通过包管理器快速安装，例如执行 sudo snap install blender --classic。
或者从官网下载 .tar.xz 压缩包，解压后配置软链接到系统路径。

进阶提示：如何在 Blender 环境中安装其他依赖？
如果你拉取的生成脚本除了需要 Blender，还需要用到其他的 Python 第三方库（比如 numpy、requests 等），千万不要用你电脑自带的 Python 环境去安装。

你必须使用 Blender 软件自带的 Python 解释器。例如，在 Linux 下可以通过以下命令安装：
找到 Blender 内置的 Python 并安装第三方库
./blender-3.6.0-linux-x64/3.6/python/bin/python3.10 -m pip install numpy requests

总结： 请先把 Blender 作为一个常规软件安装到你的电脑上，确认命令行可以识别 blender 命令后，再去拉取和运行 BladeSynth 的生成脚本。