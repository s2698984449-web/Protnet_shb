# B0 训练与验证

B0 使用 Faster R-CNN + FPN 进行二分类缺陷定位。训练仅使用冻结的 Meta-Train 产品族。固定 Meta-Val Episode 的 Query 用于无梯度评价，按产品宏平均 AP50 保存最佳权重。Outer Test 继续封存。

## B0-V3 检测协议

MVTec AD、Real-IAD 和 VisA 共用检测器及几何预处理。输入保持长宽比，以短边 1024、长边 1536 为缩放上限，只缩小超限图像。按模型步长在右侧和底部填充，保留原尺寸、缩放后的有效尺寸和坐标映射。填充张量进入检测器，预测框按有效尺寸裁剪并映射回原图。后续特征统计和空间描述使用有效区域。

`config/b0_v3.yaml` 定义共用的输入几何和 RPN 匹配。所有域的 RPN 训练使用 `max(IoU, NWD)` 作为锚框与真值框的相似度。NWD 使用缩放后坐标，归一化常数为 12.8 像素。前景、背景相似度阈值分别为 0.7、0.3。AP50 和 VisA Top-50 真值覆盖按标准 IoU 计算。 [NWD 论文](https://arxiv.org/abs/2110.13389)

B0、各折初始化模型和后续元学习 Episode 复用同一几何预处理。FOMAML 中凡执行含 RPN 损失的检测更新，直接调用检测器的匹配配置。无梯度的 Query 评估使用相同的几何预处理。比较 B0 和元学习模块增益时，使用同一检测协议、评估 Episode 和初始化来源。

## 运行

在项目根目录运行。训练使用 `MAML` Conda 环境和 CUDA。`config/baseline.yaml` 定义数据、训练预算和评价协议。正式训练默认叠加 `config/b0_pretrained.yaml` 的 ResNet-50 ImageNet-1K V2 骨干初始化。FPN、RPN 和二分类 RoI 检测头随机初始化，完整骨干参与微调。

```powershell
conda activate MAML
python -u scripts/train_b0.py --run-name b0-v3 --dry-run
python -u scripts/train_b0.py --run-name b0-v3
python -u scripts/train_b0.py --run-name b0-v3 --resume
```

`--dry-run` 校验配置及数据边界。输入几何和匹配协议记录在训练配置与 checkpoint 中，续训要求当前协议和原运行一致。各入口统一从检测器配置读取协议。

B0 的训练真值框随图像缩放，验证输出映射回原图。FOMAML 可从同一检测器入口复用配置，并用 `data_proc/detection_geometry.py` 的 `image_geometry()` 获取原尺寸、有效尺寸、单图补齐尺寸和框坐标映射。

首次训练前将官方 `resnet50-11ad3fa6.pth` 放在 `.cache/torch/hub/checkpoints/`。训练入口验证权重 SHA-256 并把完整哈希与来源写入 checkpoint。已训练的检测权重可独立用于诊断和续训。使用 `--init-profile scratch` 可新建从零训练对照。

```powershell
New-Item -ItemType Directory -Force .cache/torch/hub/checkpoints
Invoke-WebRequest -Uri https://download.pytorch.org/models/resnet50-11ad3fa6.pth -OutFile .cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
```

`--pilot` 选择三个数据集各一个 Meta-Train 产品族，每轮各取一个缺陷视角和一个正常视角。`--stop-after 1` 在第一轮保存后退出，用同名 `--pilot --resume` 可复现续训验证。断点按轮保存，中途退出时从最近已保存的一轮继续。

正式配置训练 12 轮，每轮每个数据集采样 512 个物理组，每组选一个视角。正负配额和训练预算均由配置决定。训练索引有 95,957 张候选图像，其中排除了 6,595 张 Real-IAD 官方划分冲突的训练侧图像。缺陷 Mask 经面积过滤后若无有效框，该样本跳过，并记录图像 ID。

预训练配置针对 RTX 3050 Ti 的 4 GB 显存使用训练和验证批大小 1。训练开启 2 个常驻数据进程，每进程预取 2 批。GPU 输入使用锁页内存和非阻塞传输，并在日志步和轮末读取损失。验证在训练开始时准备一次固定标注，之后每轮用一个 CPU 线程提前解码下一批。每轮分别记录训练和验证耗时。

输出目录为 `output/baseline/runs/<run-name>`。`last.pt` 包含模型、优化器、调度器、训练历史和随机状态，`best.pt` 保存产品宏平均 AP50 最佳的权重，`metrics.json` 记录每轮训练、采样与评价结果。续训校验配置、来源代码、训练索引及冻结划分哈希。评价分别输出产品、数据集和产品宏平均的 AP50、AP@[.50:.95]、Precision、Recall 和每张正常 Query 图的平均误报框数。

## 锚框短跑对照

单组短跑与多组顺序调度共用 `config/b0_short.yaml`。运行名需唯一。默认按 `default`、`small` 的顺序各跑一次，可用队列的 `--variants` 指定顺序。

```powershell
conda run -n MAML --no-capture-output python -u scripts/test_b0_short.py --run-name anchor-small-check --variant small
conda run -n MAML --no-capture-output python -u scripts/run_b0_short_queue.py --queue-name anchor-ab-check --workers 4
```

每组使用相同的随机初始化、V3 检测协议以及固定的 48 张训练图和 36 张留出图。样本取自 Meta-Train，正负样本均衡，按物理组隔离。短跑默认使用 4 个解码进程缓存图像和标注，训练与验证均为每批 1 张。每两轮报告训练和留出侧的 RPN Recall@50、产品宏平均 AP50 和用时。

每组结果写入 `output/baseline/short_runs/<run-name>/metrics.json`。队列结果写入 `output/baseline/short_runs/queues/<queue-name>/comparison.json`，组间顺序占用同一张 GPU。某组失败时保留当前队列记录，退出后续任务。

## RPN 分段推理诊断

从项目根目录运行。诊断加载已有权重，复用短跑指标文件中的样本清单，在原始锚框、有效解码框、前景分数筛选后、NMS 后不截断及 NMS 后前 50、100、300 个候选框上分别统计真值覆盖率。各阶段的命中阈值为 IoU ≥ 0.5。

```powershell
conda activate MAML
python -u scripts/diagnose_b0_rpn.py --checkpoint output/baseline/runs/b0-v3/best.pt --samples output/baseline/short_runs/anchor-small-check/metrics.json --output output/baseline/runs/b0-v3/rpn-check.json
```

指定一个新的结果文件名。报告里的 `short_holdout` 指短跑留出集。这些图属于正式 B0 的 Meta-Train 产品族，适用于定位 RPN 的候选损失。

要检验扩大候选预算能否改善最终检测，在固定 Meta-Val 上运行以下对照。三组沿用 checkpoint 的评价配置，分别保存产品宏平均 AP50、正常图误报和模型前向耗时。前向耗时包含模型内部的变换、RPN、RoI 和后处理，读图、输入张量传输及评价指标计算单独计入整组评价耗时。

```powershell
python -u scripts/sweep_b0_rpn_budget.py --checkpoint output/baseline/runs/b0-v3/best.pt --output output/baseline/runs/b0-v3/rpn-budget-check.json
```

## 历史验证结果

此前的输入尺度对照及旧 B0 性能记录见 `doc/实验记录.md`。源代码与配置快照见 `doc/archive/b0_scale_comparison/`。

| 检查 | 结果 |
| --- | --- |
| 正式抽样只读验证 | 12 轮各抽取 1,536 张图像，覆盖全部 34 个 Meta-Train 产品族。每轮每域抽取 256 张缺陷图和 256 张正常图 |
| 真实数据 GPU 短跑 | `MAML` 上完成两轮、共 12 个优化步。每轮评价 18 条 Meta-Val Episode 的 837 张唯一 Query 图像 |
| 第 1 轮退出再续训 | 与连续两轮训练的采样顺序、学习率、训练与评价指标、模型张量、优化器、调度器和随机状态完全一致。最佳权重重加载后的预测一致 |
| 测试 | 9 项专项测试通过。另确认基础环境被拒绝、划分文件哈希不符时中止 |

初次训练验证的第 1、2 轮平均损失分别为 0.654481 和 0.636011，产品宏平均 AP50 分别为 0.000012 和 0。随机初始化对照 `b0-v1` 完成 12 轮，第 6 轮产品宏平均 AP50 最佳，约为 0.0001414。锚框短跑用于诊断候选召回和缺陷定位能力。
