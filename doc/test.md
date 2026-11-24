# 测试文件与产物清单

路径以项目根目录为起点。实验开始前登记脚本、配置、输入依赖和预计写入路径。每次完成、中止或复测后，在本次任务结束前同步实际产物、状态和保留理由。统一清理前核对依赖和真实路径，只有明确列为「待统一清理」且已获删除授权的条目才进入清理范围。

## 2026-11-21 手工删除清单

B0-V3 已完成正式训练并作为后续模块对照，三域性能差距记录在 `doc/实验记录.md`，θ₀ 折内训练与四模块 Δ 按现有冻结协议推进。2026-11-21 已执行「待统一清理」21 项删除；「停用对应复跑后可人工删除」的 9 项未动。

### 待统一清理

| 对象 | 路径 | 删除理由 |
| --- | --- | --- |
| 三组中止短跑目录 | `output/baseline/short_runs/anchor-ab-v1-default/`、`output/baseline/short_runs/anchor-ab-offline-default/`、`output/baseline/short_runs/anchor-ab-maml-default/` | 已清理（2026-11-21） |
| 三组中止队列目录 | `output/baseline/short_runs/queues/anchor-ab-v1/`、`output/baseline/short_runs/queues/anchor-ab-offline/`、`output/baseline/short_runs/queues/anchor-ab-maml/` | 已清理（2026-11-21） |
| 首次尺度对照失败目录 | `output/baseline/short_runs/b0-scale-comparison-20261120/` | 已清理（2026-11-21）；成功复跑仍在 `b0-scale-comparison-20261120-r1` |
| V3 试跑权重 | `output/baseline/runs/b0-v3-pilot-20261120/best.pt`、`output/baseline/runs/b0-v3-pilot-20261120/last.pt` | 已清理（2026-11-21）；保留同目录 `metrics.json` |
| 两份独立冒烟小结 | `doc/baseline_smoke.md`、`doc/baseline_smoke_v3.md` | 已清理（2026-11-21） |
| 一次性冒烟及初始化预检结果 | `output/baseline/smoke_metrics.json`、`output/baseline/smoke_v3_metrics.json`、`output/baseline/short_runs/b0-imagenet-v2-preflight.json` | 已清理（2026-11-21） |
| 专用冒烟脚本 | `scripts/smoke_detector.py`、`training/baseline.py` | 已清理（2026-11-21） |
| Python 字节码缓存目录 | `common/__pycache__/`、`data_proc/__pycache__/`、`models/__pycache__/`、`scripts/__pycache__/`、`training/__pycache__/` | 已清理（2026-11-21） |
| 首次 fold 0 留出配对评估目录 | `output/meta_transfer/holdout/fold_0/fold0-paired-01/` | 已清理（2026-11-23）。成功复跑仍在 `fold0-paired-02/` |

`config/baseline.yaml` 中的冒烟输出字段属于正式 V3 的配置快照。保留该文件，避免改变检查点记录的配置哈希。旧冒烟文档和历史实验记录中的入口路径用于追溯。

### 停用对应复跑后可人工删除

| 对象 | 路径 | 删除条件与影响 |
| --- | --- | --- |
| 锚框短跑命令入口 | `scripts/test_b0_short.py`、`scripts/run_b0_short_queue.py` | 已清理（2026-11-21）；保留 `training/b0_short.py` 与 `config/b0_short.yaml` |
| 已退役尺度对照的脚本及配置快照 | `doc/archive/b0_scale_comparison/*.txt` | 已清理（2026-11-21）；保留尺度对照 `metrics.json` |
| 旧正式版续训断点与最佳权重 | `output/baseline/runs/b0-v1/{best,last}.pt`、`output/baseline/runs/b0-imagenet-v2/{best,last}.pt` | 已清理（2026-11-21）；保留各自 `metrics.json` 与诊断报告 |
| FOMAML 独立验收权重 | `output/meta_transfer/**/verify-fomaml-20261121/*.pt` | 新运行名复测通过后可删。三份 `metrics.json` 和 `parameter_manifest.json` 留作验收证据 |

### 继续使用的文件

- 正式 V3 对照的 `output/baseline/runs/b0-v3/best.pt`、`output/baseline/runs/b0-v3/last.pt`、`output/baseline/runs/b0-v3/metrics.json`。
- `scripts/train_theta0.py`、`training/theta0.py`、`config/theta0.yaml` 与 `scripts/train_transfer_delta.py`、`training/transfer_delta.py`、`config/transfer_delta.yaml`，以及它们复用的 B0、评估与检查点代码。
- `scripts/train_theta0_fomaml.py`、`training/theta0_fomaml.py`、`training/fomaml_roi.py`、`data_proc/fomaml_episodes.py` 和 `config/fomaml.yaml`。折内元训练直接复用 B0-V3 检测器。
- `scripts/eval_meta_holdout.py`、`training/holdout_evaluation.py`、`training/holdout_inputs.py`、`training/holdout_features.py` 和 `config/holdout_eval.yaml`，以及 `output/meta_transfer/holdout/fold_0/fold0-paired-02/metrics.json`。留作同折配对复跑和结果溯源。
- 冻结的 `output/splits/`、`output/episodes/v1/`、`output/audit/` 和 `.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`。需清理的缓存仅限上表指定的 `__pycache__` 目录。
- `training/b0_short.py` 和 `config/b0_short.yaml`。RPN 分段诊断依赖短跑采样和样本解码，RoI 诊断和 RPN 预算对照也依赖短跑配置。保留相应诊断脚本和已完成的锚框对照样本 `output/baseline/short_runs/anchor-ab-local-default/metrics.json`、`output/baseline/short_runs/anchor-ab-local-small/metrics.json`。
- `tests/test_b0_*.py` 的回归测试、已完成的历史指标与诊断报告、`doc/test.md` 和 `doc/实验记录.md`。

## 文件与运行产物

| 测试 | 文件、目录或输出 | 状态与处理 |
| --- | --- | --- |
| B0 冒烟 | `scripts/smoke_detector.py`、`training/baseline.py` | 已清理（2026-11-21） |
| B0 冒烟 | `output/baseline/smoke_metrics.json` | 已清理（2026-11-21） |
| 有限标注审计 | `scripts/audit_annotations.py`、`output/audit/annotation_audit_report.csv`、`doc/annotation_audit.md` | 已完成，保留复用和审计证据 |
| 随机初始化 B0 正式运行 | `output/baseline/runs/b0-v1/best.pt`、`output/baseline/runs/b0-v1/last.pt`、`output/baseline/runs/b0-v1/metrics.json` | 权重已清理（2026-11-21）；`metrics.json` 保留作历史对照 |
| 锚框短跑与队列 | `scripts/test_b0_short.py`、`scripts/run_b0_short_queue.py`、`training/b0_short.py`、`config/b0_short.yaml` | 两个命令入口已清理（2026-11-21）；模块与配置仍供诊断复用 |
| 已完成的锚框对照 | `output/baseline/short_runs/anchor-ab-local-default/metrics.json`、`output/baseline/short_runs/anchor-ab-local-small/metrics.json`、`output/baseline/short_runs/queues/anchor-ab-local/comparison.json` | 保留，样本清单和对照指标用于后续诊断 |
| 中止的短跑队列 | `output/baseline/short_runs/anchor-ab-v1-default/`、`output/baseline/short_runs/anchor-ab-offline-default/`、`output/baseline/short_runs/anchor-ab-maml-default/` | 已清理（2026-11-21） |
| 中止的队列状态文件 | `output/baseline/short_runs/queues/anchor-ab-v1/`、`output/baseline/short_runs/queues/anchor-ab-offline/`、`output/baseline/short_runs/queues/anchor-ab-maml/` | 已清理（2026-11-21） |
| RPN 分段筛选 | `scripts/diagnose_b0_rpn.py`、`training/b0_rpn_diagnostic.py` | 已完成，保留复用 |
| RPN 分段结果 | `output/baseline/short_runs/anchor-ab-local-default/rpn-b0-v1-best.json`、`output/baseline/short_runs/anchor-ab-local-default/rpn-b0-v1-best-screening.json` | 已完成，保留对照证据 |
| RPN 候选预算对照 | `scripts/sweep_b0_rpn_budget.py`、`training/b0_rpn_budget.py`、`output/baseline/short_runs/anchor-ab-local-default/rpn-budget-local.json` | 已完成，保留复用和对照证据 |
| B0 回归测试 | `tests/test_b0_checkpoints.py`、`tests/test_b0_data.py`、`tests/test_b0_evaluation.py` | 保留，覆盖断点、划分和评价流程 |
| RoI 真值框诊断 | `scripts/diagnose_b0_roi.py`、`training/b0_roi_diagnostic.py` | 已生成，72 张真实样本验证通过，保留复用 |
| RoI 真值框诊断结果 | `output/baseline/short_runs/anchor-ab-local-default/roi-b0-v1-best.json` | 已完成，含逐图分数和 IoU，保留诊断证据 |
| 预训练 B0 接入验证 | `config/b0_pretrained.yaml`、`models/detector.py`、`scripts/train_b0.py`、`training/train_b0.py` | 已验证来源、主干加载和检测头反向更新，保留复用 |
| 预训练 B0 初始化权重 | `.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth` | 已下载并校验，保留为新训练的输入 |
| 预训练 B0 真实样本验证 | `output/baseline/short_runs/b0-imagenet-v2-preflight.json` | 已清理（2026-11-21）；结论保留在实验记录 |
| 预训练 B0 正式运行 | `output/baseline/runs/b0-imagenet-v2/` | `best.pt/last.pt` 已清理（2026-11-21）；保留指标和诊断报告 |
| 预训练 B0 RoI 对照诊断 | `scripts/diagnose_b0_roi.py`、`output/baseline/runs/b0-imagenet-v2/roi-meta-val.json` | 已完成，逐图确认与旧 RoI 诊断为同一批 72 张样本，保留 |
| VisA 分辨率与 RoI 对照 | `scripts/diagnose_b0_roi.py`、`training/b0_roi_diagnostic.py`、`output/baseline/runs/b0-imagenet-v2/roi-visa-resolution.json` | 已完成，保留固定 B0 权重和 Top-50 下 `256/512` 与 `320/640` 的对照结果 |
| VisA 分辨率反序复测 | `output/baseline/runs/b0-imagenet-v2/roi-visa-resolution-reverse.json` | 已完成，检测及 RoI 汇总指标重复一致，保留两次计时 |
| B0 共享几何预处理回归 | `data_proc/detection_geometry.py`、`models/detector.py`、`tests/test_b0_geometry.py` | 13 项 B0 回归通过，其中 4 项覆盖等比缩小、右下填充、坐标还原及旧配置兼容性。保留复用 |
| 历史 B0 共享预处理入口检查 | `scripts/train_b0.py`、`training/train_b0.py` | 旧版只读验证通过，核对了预处理配置、34 个 Meta-Train 产品族、18 条 Meta-Val Episode 及首轮抽样 |
| B0 输入尺度 GPU 实测与配对短训 | `doc/archive/b0_scale_comparison/`、`output/baseline/short_runs/b0-scale-comparison-20261120-r1/metrics.json` | 已完成。`1024/1536` 通过真实 VisA 图预检，两组各训练 192 步，均评价同一批 837 张 Query。保留结果，旧入口和配置快照按历史复现需求处理 |
| B0 输入尺度实测首次启动 | `output/baseline/short_runs/b0-scale-comparison-20261120/metrics.json` | 已清理（2026-11-21） |
| B0-V3 协议与 RPN 匹配回归 | `config/b0_v3.yaml`、`models/detector.py`、`tests/test_b0_geometry.py`、`tests/test_b0_matching.py` | 18 项通过，覆盖几何、混合相似度、实际 RPN 标签和旧配置拒绝。保留复用 |
| B0-V3 正式入口和 GPU 试跑 | `scripts/train_b0.py`、`training/train_b0.py`、`output/baseline/runs/b0-v3-pilot-20261120/` | 试跑权重已清理（2026-11-21）；保留 `metrics.json` 与正式入口代码 |
| B0-V3 冒烟入口 | `scripts/smoke_detector.py`、`training/baseline.py`、`output/baseline/smoke_v3_metrics.json`、`doc/baseline_smoke_v3.md` | 已清理（2026-11-21）；结论保留在实验记录 |
| B0-V3 正式训练与结果评审 | `output/baseline/runs/b0-v3/best.pt`、`output/baseline/runs/b0-v3/last.pt`、`output/baseline/runs/b0-v3/metrics.json` | 已完成 12 轮、18,429 步，第 12 轮的六产品宏平均 AP50 为 20.36%。保留正式对照权重、续训断点和评估记录 |
| θ₀ 折内入口检查 | `scripts/train_theta0.py`、`training/theta0.py`、`config/theta0.yaml` | 五折及 fold 0 小跑的只读检查通过。各折训练族数为 27、27、27、27、28，与 OOF 目标、Meta-Val、Outer Test 分离。后续小跑预计写入 `output/meta_transfer/theta0/fold_0/pilot-20261121/` |
| 四模块 Δ 隔离检查 | `scripts/train_transfer_delta.py`、`training/transfer_delta.py`、`config/transfer_delta.yaml` | CPU 校验通过，83 个参数张量均有唯一归属，差分可重放。单源 GPU 小跑依赖 fold 0 的 θ₀，预计写入 `output/meta_transfer/delta/fold_0/mvtec-bottle/pilot-20261121/` |
| FOMAML RoI 与 fold 0 联通试验 | `scripts/train_theta0_fomaml.py`、`training/theta0_fomaml.py`、`training/fomaml_roi.py`、`data_proc/fomaml_episodes.py`、`config/fomaml.yaml` | 已完成真实 GPU 验收。监督 θ₀ 6 步，FOMAML 1 步内更新和 1 步外更新，四个 Δ 各 2 步。确认 FOMAML 权重只改动 RoI 预测头的四个张量，83 个张量在 Δ 四模块中互斥。结果在 `output/meta_transfer/theta0/fold_0/verify-fomaml-20261121/`、`output/meta_transfer/theta0_fomaml/fold_0/verify-fomaml-20261121/`、`output/meta_transfer/delta/fold_0/mvtec-bottle/verify-fomaml-20261121/` |
| fold 0 用户复跑与正式训练结果核查 | `output/meta_transfer/theta0/fold_0/fold0-pilot-01/`、`output/meta_transfer/theta0_fomaml/fold_0/fold0-pilot-01/`、`output/meta_transfer/delta/fold_0/mvtec-bottle/fold0-pilot-01/` | 用户小跑完成。6 步监督、1 次元更新及四模块各 2 步，可复现前次联通验收 |
| fold 0 正式监督 θ₀、FOMAML 与单源 Δ | `output/meta_transfer/theta0/fold_0/fold0-1w5s-01/`、`output/meta_transfer/theta0_fomaml/fold_0/fold0-1w5s-01/`、`output/meta_transfer/delta/fold_0/mvtec-bottle/fold0-1w5s-01/` | 已核对两份初始化与四份 Δ 的权重、冻结折和文件哈希。监督 12 轮 18,432 步，FOMAML 3 轮 243 次外更新，单源验证每模块 2 步。保留作为留出评价输入 |
| fold 0 留出配对评估首次运行 | `output/meta_transfer/holdout/fold_0/fold0-paired-01/metrics.json` | 已清理（2026-11-23）。成功复跑在 `fold0-paired-02/` |
| fold 0 留出配对评估 | `scripts/eval_meta_holdout.py`、`training/holdout_evaluation.py`、`training/holdout_inputs.py`、`training/holdout_features.py`、`config/holdout_eval.yaml` | 2026-11-21 GPU 评估完成。7 个留出产品族、21 条固定 `1w5s` Episode、6 个权重方案均完成适配前后评分，四模块各有 21 条有效 (U) 配对。已核对源代码与权重哈希、Query 统计量和缓存推理一致性。保留 `output/meta_transfer/holdout/fold_0/fold0-paired-02/metrics.json` |
| fold 0 `mvtec:bottle` Δ 步数对照 | `scripts/train_transfer_delta.py --steps`、`scripts/eval_meta_holdout.py`、`fold0-1w5s-01` 的监督与 FOMAML `last.pt` | 已完成。2 步沿用 `fold0-paired-02`；5 步结果在 `output/meta_transfer/holdout/fold_0/fold0-steps-5b/metrics.json`；10 步结果在 `output/meta_transfer/holdout/fold_0/fold0-steps-10/metrics.json`。空目录 `fold0-steps-5/` 已清理（2026-11-23） |
| 检测对齐损失多框定位 | `tests/test_fomaml_loss.py`、`training/fomaml_roi.py` | 已覆盖两个真实框的 GIoU。失败空目录 `output/meta_transfer/theta0_fomaml/fold_0/fold0-detloss-v1/` 已清理（2026-11-23）。成功权重在 `fold0-detloss-v1b/` |
| fold 0 固定 Null proposal 评价 | `scripts/eval_meta_holdout.py`、`fold0-detloss-v1` 监督权重、`fold0-detloss-v1b` 的 FOMAML 与 `realiad:audiojack`、`visa:candle` 两步 Δ | 已完成。audiojack 固定候选在 `output/meta_transfer/holdout/fold_0/fold0-detloss-v1b-audiojack-fixed/metrics.json`，candle 在 `fold0-detloss-v1b-candle-fixed-r1/metrics.json`。空目录 `fold0-detloss-v1b-candle-fixed/` 已清理（2026-11-23） |
| 缺陷分数口径与源产品采样 | `tests/test_fomaml_loss.py`、`tests/test_b0_evaluation.py`、`config/delta_sampling.yaml`、`training/fomaml_roi.py`、`training/transfer_delta.py`、`training/holdout_evaluation.py` | 已完成，未写权重。9 项损失与抽样测试、4 项评价测试通过。`realiad:audiojack` 只读核对：成对配置 2 个实物组、2 张图；`source_groups` 为 504 个实物组、2904 张图、0 次重复 |
| 效用协议低秩适配与指定模块 Δ | `config/utility.yaml`、`training/lora_adapt.py`、`training/holdout_evaluation.py`、`training/transfer_delta.py`、`tests/test_fomaml_loss.py` | 单产品两模块试跑已作废。耳机孔与蜡烛的 `fold0-u-*` 权重和评价在 2026-11-23 删除。完整任务改为四模块 |
| 创新点一完整任务入口 | `scripts/train.py innovation1`、`scripts/prepare.py`、`training/meta_transfer.py`、`training/theta0.py`、`training/b0_diagnostic.py` | 2026-11-23 收拢入口。第 0 折训练仍用运行名 `fold0-utility-m4`，结果写到 `output/meta_transfer/innovation1/fold_0/fold0-utility-m4/` |

## 主要输入依赖

| 测试 | 配置与输入 |
| --- | --- |
| B0 冒烟、有限标注审计 | `config/baseline.yaml`、`output/episodes/v1/episode_bank_meta_val.jsonl` |
| 锚框短跑 | `config/baseline.yaml`、`config/b0_short.yaml`、`output/splits/episode_group_assignment.csv` |
| 历史 V1 RPN 分段筛选 | 原运行使用 `output/baseline/runs/b0-v1/best.pt`、`output/baseline/short_runs/anchor-ab-local-default/metrics.json` |
| 历史 V1 候选预算对照、RoI 真值框诊断 | 原运行使用旧版 `config/baseline.yaml`、`output/baseline/runs/b0-v1/best.pt`、`output/episodes/v1/episode_bank_meta_val.jsonl` |
| 预训练 B0 | `config/baseline.yaml`、`config/b0_pretrained.yaml`、现有冻结 Meta-Train 和 Meta-Val |
| 预训练 B0 RoI 对照诊断 | `output/baseline/runs/b0-imagenet-v2/best.pt`、`output/episodes/v1/episode_bank_meta_val.jsonl`、旧 RoI 诊断 JSON |
| VisA 分辨率与 RoI 对照及反序复测 | `output/baseline/runs/b0-imagenet-v2/best.pt`、`output/episodes/v1/episode_bank_meta_val.jsonl`。使用与 `roi-meta-val.json` 中一致的 VisA 诊断样本 |
| B0 正式训练入口检查 | `config/baseline.yaml`、`config/b0_pretrained.yaml`、本地 ImageNet 初始化权重、冻结 Meta-Train 和 Meta-Val |
| 历史 B0 输入尺度 GPU 实测与配对短训 | `doc/archive/b0_scale_comparison/b0_scale_comparison.yaml.txt`、固定权重和数据划分哈希见历史报告 |
| B0-V3 正式入口和 GPU 试跑 | `config/baseline.yaml`、`config/b0_v3.yaml`、`config/b0_pretrained.yaml`、`.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`、冻结 Meta-Train 索引和 `output/episodes/v1/episode_bank_meta_val.jsonl` |
| B0-V3 正式训练与结果评审 | 与 V3 试跑共用检测协议及冻结划分，正式训练覆盖 34 个 Meta-Train 产品族。比较使用 `output/baseline/runs/b0-imagenet-v2/metrics.json` |
| θ₀ 折内入口检查 | `output/splits/inner_oof_folds.json`、`config/b0_v3.yaml`、`config/b0_pretrained.yaml`、本地 ImageNet 初始化权重 |
| 四模块 Δ 隔离检查 | 同折 θ₀ 权重、冻结 Δ 源任务名单和物理组索引 |
| FOMAML RoI 与 fold 0 联通试验 | 同折监督 θ₀、本地初始化权重、`output/splits/inner_oof_folds.json`、`output/episodes/v1/episode_bank_oof.jsonl` 中允许训练族的子集、`config/b0_v3.yaml` |
| fold 0 正式监督 θ₀、FOMAML 与单源 Δ | `config/theta0.yaml`、`config/fomaml.yaml`、`config/transfer_delta.yaml`、`output/splits/inner_oof_folds.json`、`output/episodes/v1/episode_bank_oof.jsonl`。本折 7 个留出产品族的 21 条 Episode 用于后续独立评价 |
| fold 0 留出配对评估 | 使用 `fold0-1w5s-01` 的两份 `last.pt`、四份 `*.pt` 差分、其参数清单与指标文件，以及冻结的 fold 0 OOF Episode Bank |
| fold 0 `mvtec:bottle` Δ 步数对照 | `config/transfer_delta.yaml`、`fold0-1w5s-01` 的两份 `last.pt` 与已完成的 2 步 Δ、冻结 fold 0 OOF Episode Bank。新运行只增加 5 步和 10 步，样本仍是该配置抽出的一张正常图和一张缺陷图 |

分辨率更换前后的指标可用于描述整套基线的变化。检验分辨率或匹配方法的影响时，应固定权重做推理对照，或固定训练样本、初始化和步数做配对训练。模块增益使用同一检测协议下的新 B0 作对照。`doc/实验记录.md` 维护实验结论。清理完成后在本清单把对应条目标记为「已清理」，并登记日期。


| Feature 共同候选评价 | training/holdout_features.py、training/holdout_evaluation.py、scripts/eval_meta_holdout.py | 已完成 audiojack 和 candle。结果写入各自 holdout metrics.json，Feature 选择使用 common_proposal_delta_vs_null，Where 使用 proposal_summary 和最终 AP50 |
| 四模块加权 FOMAML | `config/fomaml.yaml`、`training/theta0.py`、`training/fomaml_roi.py`、`tests/test_fomaml_loss.py`、监督权重 `fold0-detloss-v1/last.pt` | 2026-11-23 已接通并验证。15 项测试通过，干运行 81 条任务、243 步。正式权重尚未训练，运行名 `fold0-fomaml-modules` |
| 在线条件化 FOMAML 评分复用 | `training/fomaml_roi.py`、`training/theta0.py`、`tests/test_fomaml_loss.py` | 2026-11-24 完成。MAML 环境 17 项测试通过，含分层缓存与原评分对照。未启动训练，未写 `output/meta_transfer` |
| 梯度草图预筛选与四模块留出加载 | `training/fomaml_roi.py`、`training/theta0.py`、`training/meta_transfer.py`、`scripts/train.py`、`tests/test_fomaml_loss.py` | 2026-11-24 完成。MAML 环境 20 项测试通过。`--prescreen-k` 默认 0，Feature/Where 才预筛。留出加载按四个模块的参数名核对。未启动训练，没有测速数字 |
| Meta-Val AP50 选模、预筛抽样审计和留出加载演练 | `training/theta0.py`、`training/fomaml_roi.py`、`scripts/train.py`、`tests/test_fomaml_loss.py` | 2026-11-24 完成。MAML 环境 26 项测试通过。新开关默认关闭或等于原行为。未启动训练，未做 GPU 评估 |
| FOMAML 阶段日志 | `training/theta0.py`、`tests/test_fomaml_loss.py` | 2026-11-24 完成。MAML 环境 27 项测试通过。每条任务打印进入和阶段。未启动训练 |
| FOMAML 外更新断点 | `training/theta0.py`、`scripts/train.py`、`tests/test_fomaml_loss.py` | 2026-11-24。`--resume` 从 `resume.pt` 的下一条任务继续。未启动训练 |

