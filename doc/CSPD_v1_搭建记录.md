# CSPD v1 搭建记录

分支：`diffusion_maml_detect`。这一版只保留已经对齐的检测结构，不保留旧接口。

旧代码仍在提交 `354f9e9`。本地旧输出仍在 `CAIS_out/`。

这份文档用来定位问题：术语、参数、激活函数、损失写在哪个文件、参数由谁更新。公式与代码一致，对应 `cspd/model.py`、`cspd/condition.py`、`cspd/policy.py`、`cspd/data.py`、`scripts/train_cspd.py`。

## 1. 术语

| 术语 | 在本项目里的意思 |
| --- | --- |
| episode / 任务 | 一次小样本抽样。抽出 `n_way` 个类，每类 `k_shot` 张支持图、`q_query` 张查询图 |
| 支持集 support | 带类别标签的少量参考图。用来建原型，也用来做那一步快速适应 |
| 查询集 query | 要判的图。训练时它的标签只进入外循环损失，不进原型 |
| 原型 prototype | 某一类支持特征的均值。不是网络参数，每集重算 |
| ProtoNet | 用查询特征到各类原型的距离做分类，距离越近越像该类 |
| 内循环 | 每个 episode 内部，用支持集损失对投影层走一步梯度。步子用完即丢 |
| 外循环 | 用查询集损失更新模型里真正保存的参数。这是跨 episode 学到的检测规则 |
| MAML 式适应 | 内循环的那一步。这里只适应两个 1×1 投影，不是整个网络 |
| 留一法 LOO | 判某一张支持图时，它自己不进入原型，避免和自己比距离 |
| FiLM | 用工况向量生成逐通道的缩放和偏移，去调制特征图 |
| 策略网络 | 按每一类参考图的样子，给出检测先验：信原图还是校正图、看哪个尺度、哪些通道更敏感、距离放多大。一类一套，不输出类别分数。代码在 `cspd/policy.py`。参数只在外循环被查询损失更新 |
| z_part | 零件码。只由支持集特征得到，查询图不参与 |
| z_domain | 工况码。工况向量经过编码网络后的 128 维表示 |
| 分数图 score map | 每个空间位置对每个类别的分数，形状 `(Nq, ways, 14, 14)` |
| logits | 把分数图做 top-k 均值后的图像级分类分数，形状 `(Nq, ways)` |
| 工况向量 | 送进工况编码的原始输入。部署时应是产线记录；当前代码临时用图像统计代替 |

## 2. 这一版做什么

1. 空间原型检测头。保留 ResNet18 的 layer3、layer4。支持集按类别聚成原型，查询图每个位置计算到各类原型的负欧氏距离。
2. 特征级工况调制。每张图用自己的工况向量做 FiLM。输出的是调制后的特征，不是一张新图。扩散的用法见第 2.1 节，代码尚未接入。
3. 策略网络。按每一类参考图给出调制先验，交给距离计算。划痕和麻点不共用一套敏感程度。不输出类别分数。实现见 `cspd/policy.py`。

类别来自 logits 的 argmax。位置是预测类分数图上的峰值，坐标归一化到 0–1。NEU-CLS 没有位置标签，所以没有定位损失。

初始化：FiLM 的 gamma、beta 为 0，因此起始时 FiLM 是恒等映射。策略四个头的权重和偏置为 0，因此每一类都是：混合系数 \(g=0.5\)，两个尺度权重各 0.5，通道门控为 1，距离倍数 \(\tau=1\)。未训练时等价于原图空间原型，各类不偏袒。

## 2.1 扩散模型目前的用法

扩散不负责分类，也不负责把整张图变亮或变暗。整体亮暗由 `data_proc/compute_illumination.py` 做：同一种表面共用一个正常亮度，每张图只做有上限的整体加减，原图保留。

扩散只处理这种校正做不到的光照：一块亮一块暗、反光、暗角。参考图和待检图都变到该表面的正常光照下，再交给元学习建原型、比距离。只改待检图、不改参考图时，两边仍然不是同一种样子。

使用顺序：

1. 先用亮度校正。这一步已经能处理整体偏亮、偏暗。
2. 扩散单独训练，学的是“不均匀光照 → 该表面的正常光照”。训练时必须约束缺陷区域还在，不能把划痕涂成正常表面。
3. 扩散训好后冻住，再训练元学习。两边不要从头一起训。
4. 原图支路留着。大块油斑、水斑若被扩散抹掉，策略改走原图。

没有“同一处缺陷、两种灯光”的实拍之前，扩散不接进当前训练。当前检测仍走原图加 FiLM。

## 3. 目录

```text
cspd/backbone.py      ResNet18 空间特征
cspd/condition.py     工况编码、零件编码、FiLM
cspd/policy.py        策略网络
cspd/model.py         混合、原型、距离、内循环、查询分数
cspd/data.py          通用类别文件夹抽样；工况/掩码/还原图旁路
cspd/condition_io.py  六槽工况 + 缺测掩码 + 亮度 manifest
cspd/annotate.py      通用掩码、高斯权重、质心
cspd/restore.py       还原图支路说明（目录由配置接入）
cspd/runtime.py       训练和推理共用的设备、前向、损失、评估、输出目录
scripts/load_data.py  按配置检查数据
scripts/split_class_folder.py  通用 train/val/test 拆分
scripts/train_cspd.py 外循环训练；早停只用 val_dir
scripts/infer_cspd.py 用 test_dir 评估
data_proc/compute_illumination.py  亮度校正；CSV 可由 illumination_manifest 接入
config/cspd.yaml      训练配置。换数据只改路径和旁路
config/cspd_test.yaml 冻结对照配置
```

`config/common.yaml`、`config/automotive/`、`config/aerospace/` 新训练不读取。

## 4. 激活函数

激活函数按模块列出。Linear、Conv、BatchNorm 本身没有激活。

| 模块 | 文件 | 激活 |
| --- | --- | --- |
| ResNet18 骨干 | `cspd/backbone.py`，来自 torchvision | ReLU。每个残差块是卷积 → BatchNorm → ReLU → 卷积 → BatchNorm → 相加 → ReLU。没有 GELU、Sigmoid |
| 工况编码 `cond_encoder` | `cspd/condition.py` `MLPEncoder` | Linear → LayerNorm → GELU → Linear。最后一层没有激活 |
| 零件编码 `part_encoder` | 同上 | 与工况编码相同：中间 GELU，输出层无激活 |
| FiLM | `cspd/condition.py` | 无激活。gamma、beta 只做截断 `clamp(-rho, rho)`，`rho=0.5` |
| 策略主干 `policy.trunk` | `cspd/policy.py` | Linear → GELU → Linear → GELU。两层隐藏宽度都是 128。最后没有再接激活 |
| 混合系数 \(g\) | `policy.py` `g_head` | Sigmoid。一类一个数，落在 (0, 1)。0.5 表示原图特征和工况特征各半 |
| 尺度权重 \(\alpha\) | `policy.py` `alpha_head` | Softmax，对细尺度、粗尺度两维。一类两个数，相加为 1 |
| 通道门控 gate | `policy.py` `gate_head` | `1 + policy_rho * tanh(.)`。`policy_rho=0.5`，每个通道落在 [0.5, 1.5]。1 表示该通道不放大也不缩小 |
| 距离倍数 \(\tau\) | `policy.py` `tau_head` | `exp(clamp(·, -2, 2))`。一类一个数，落在约 [0.135, 7.39]。1 表示距离不缩放。截断是为了避免倍数爆炸 |
| 距离到 logits | `cspd/model.py` | 无额外激活。分数是负欧氏距离再除以该类的 \(\tau\)，logits 是 top-k 均值 |

LayerNorm 不是激活，它把一个样本的特征减均值、除标准差，再乘可学习缩放、加可学习偏移。

## 5. 前向公式

记支持图投影后的特征为 \(F^{raw}\)，FiLM 之后为 \(F^{film}\)。两个尺度分别是 layer3（14×14）和 layer4（7×7），通道都已投到 128。

工况编码和零件编码：

\[
z = W_2\,\mathrm{GELU}(\mathrm{LayerNorm}(W_1 x))
\]

FiLM，对每个通道 \(c\)：

\[
F^{film}_c = (1 + \mathrm{clamp}(\gamma_c, -\rho, \rho))\, F^{raw}_c + \mathrm{clamp}(\beta_c, -\rho, \rho)
\]

\(\gamma,\beta\) 由该图自己的工况码线性变换得到。\(\rho =\) `film_rho`。

策略按类计算。类别 \(c\) 的输入只来自该类支持图，拼成一行，查询图和类名都不进这一行。`_class_state` 在 `cspd/model.py`：

| 块 | 维数 | 怎么来 |
| --- | --- | --- |
| 零件码 | 128 | 全部支持图 layer4 池化后再平均，过 `part_encoder`。同一集里各类共用这一段 |
| 该类外观 | 128 | 该类支持图 layer4 池化向量的均值 |
| 该类工况 | 7 | 该类支持图工况向量的均值。现在仍是第 9 节的图像统计 |
| 离散度 | 1 | 该类几张池化向量的标准差再取均值。只有 1 张时为 0 |
| 特征差 | 1 | 该类原图特征与 FiLM 特征之差的绝对值，layer3 和 layer4 各算一次再平均 |

四个头都从这 265 维出发，主干共享，输出按类：

\[
g_c = \sigma(a_g),\quad
\alpha_c = \mathrm{softmax}(a_\alpha),\quad
\mathrm{gate}_c = 1 + \rho_p \tanh(a_{gate}),\quad
\tau_c = \exp(\mathrm{clamp}(a_\tau,-2,2))
\]

和类别 \(c\) 比距离时，支持图和查询图都用这一类的 \(g_c\)、\(\mathrm{gate}_c\) 混合。查询图的 FiLM 仍用查询图自己的工况，只是混合比例听该类参考图的：

\[
F^{(c)} = \mathrm{gate}_c \odot \big(g_c F^{film} + (1-g_c) F^{raw}\big)
\]

类别 \(c\) 的原型 \(p_c\) 是该类支持图在 \(F^{(c)}\) 上全部空间位置的均值。位置 \((h,w)\) 的分数：

\[
S_c(h,w) = -\| f^{(c)}(h,w) - p_c \|^2
\]

layer4 的分数图双线性插值到 14×14 后，按该类自己的尺度权重相加，再除以该类的距离倍数：

\[
S_c = \big(\alpha_{c,3} S_c^{(3)} + \alpha_{c,4} S_c^{(4)}\big) / \tau_c
\]

图像级 logits 是每个类别在空间上最大的 \(k\) 个分数的平均。\(k = \max(1, \mathrm{round}(\texttt{topk\_ratio}\cdot 14\cdot 14))\)，默认 `topk_ratio=0.1`，\(k=20\)。

\[
\ell_c = \frac{1}{k}\sum_{j=1}^{k} S_{c,(j)}
\]

预测类 \(\hat y = \arg\max_c \ell_c\)。位置是 \(\hat y\) 那张 14×14 图上的最大值坐标，除以 13，得到 0–1。

内循环的支持集 logits 不用这张空间图。它先把特征做全局平均，再用留一法原型做同样的负欧氏距离。和某一类比时，仍用该类自己的 \(g\)、gate、\(\alpha\)、\(\tau\)。代码在 `loo_pooled_logits`。

## 6. 损失函数的位置和公式

两处都是 `torch.nn.functional.cross_entropy`，也就是对 logits 做 log-softmax 再取真实类的负对数：

\[
\mathcal{L}(\ell, y) = -\ell_y + \log\sum_c \exp(\ell_c)
\]

| 损失 | 代码位置 | 用谁的标签 | 是否写入优化器 |
| --- | --- | --- | --- |
| 内循环支持损失 \(\mathcal{L}_s\) | `cspd/model.py` 的 `forward`，`F.cross_entropy(s_logits, support_y)` | 支持集 | 不直接 `optimizer.step`。只用来对投影层求一步梯度 |
| 外循环查询损失 \(\mathcal{L}_q\) | `scripts/train_cspd.py` 训练循环，`F.cross_entropy(pred['logits'], query_y)` | 查询集 | 是。AdamW 最小化的就是它 |

没有定位损失、没有对比损失、没有策略自己的奖励。策略没有单独的损失项。它要学的调制是否有用，只看待检图被分对没有。

训练时内循环打开 `create_graph=True`，所以 \(\mathcal{L}_q\) 能穿过那一步适应，反传到投影的原始参数。策略在内循环里也被调用了一次（用来算支持集留一法 logits），这一步不改策略参数，但查询损失能穿过适应步，间接看到“策略怎么影响了投影的那一步”。查询打分时策略再被调用一次，\(\mathcal{L}_q\) 直接对这一次的 \(g,\alpha,\mathrm{gate},\tau\) 反传。两条都进同一套策略权重。

测试时 `model.eval()`，内循环仍会走，但 `create_graph=False`，并且输出被 `detach`，没有外循环更新。

## 7. 参数更新方式

### 7.1 内循环：每个 episode 的临时适应

只更新 4 个张量：`proj3.weight`、`proj3.bias`、`proj4.weight`、`proj4.bias`。

\[
\theta' = \theta - \texttt{adapt\_lr}\,\nabla_\theta \mathcal{L}_s
\]

`adapt_lr` 默认 0.1，`adapt_steps` 默认 1。\(\theta'\) 存在局部变量 `fast` 里，用来算本集的查询分数。下一个 episode 仍从模型里的 \(\theta\) 开始。骨干、FiLM、策略、两个编码不在这一步里改。策略在这一步里只被读取，用来给留一法距离提供 \(g,\alpha,\mathrm{gate},\tau\)。

### 7.2 外循环：跨 episode 保存下来的更新

`trainable_parameters` 取所有 `requires_grad=True` 的参数。骨干在 `freeze_backbone=true` 时 `requires_grad=False`，不更新。

被 AdamW 更新的是：

| 参数 | 模块 | 作用 |
| --- | --- | --- |
| `proj3`、`proj4` | 1×1 卷积 | 两个尺度投到 128 维。内循环改的是副本，外循环改的是初始值 |
| `cond_encoder` | MLPEncoder | 工况向量 → 128 维工况码 |
| `part_encoder` | MLPEncoder | 支持集池化特征 → 零件码 |
| `film.gamma`、`film.beta` | Linear | 工况码 → 通道缩放和偏移 |
| `policy.trunk`、`g_head`、`alpha_head`、`gate_head`、`tau_head` | 策略 | 按类的混合、尺度、通道门控、距离倍数。只由外循环更新，见 7.3 |

更新顺序在 `scripts/train_cspd.py`：

1. 前向得到查询 logits。
2. 计算 \(\mathcal{L}_q\)。
3. `zero_grad`，再 `loss.backward()`。
4. 全部可训练梯度做全局范数裁剪，阈值 `grad_clip=1.0`。超过则整组梯度同比缩小。
5. AdamW 一步。学习率 `lr=0.001`，权重衰减 `weight_decay=0.0001`。权重衰减加在参数上，不经过梯度缩放。

AdamW 对每个参数维护一阶矩 \(m\) 和二阶矩 \(v\)，用 \(\mathcal{L}_q\) 的梯度更新 \(m,v\)，再用它们更新参数。默认动量系数是 PyTorch 的 0.9 和 0.999。

不更新的量：ResNet18 全部权重、7 维工况向量、每集算出来的原型、内循环的 \(\theta'\)、策略当次吐出的 \(g,\alpha,\mathrm{gate},\tau\)。这四个数每集重算，存下来的是算出它们的网络权重。

### 7.3 策略怎么更新

策略权重和投影、FiLM、两个编码走同一次 AdamW，学习率、权重衰减、梯度裁剪都一样。没有单独的优化器，也没有按类分开的学习率。

一次 episode 里的顺序：

1. 用当前投影抽出支持图的原特征和 FiLM 特征。
2. 按类拼输入，策略吐出该类的 \(g,\alpha,\mathrm{gate},\tau\)。此时若在训练且 `adapt_steps>0`，这些数只参与支持集留一法损失，用来推动投影的临时副本。策略权重不动。
3. 投影走完内循环后，用适应后的投影再抽一次特征，策略再按类吐一套数。
4. 待检图用第 3 步的数和对应类的原型算 logits，得到 \(\mathcal{L}_q\)。
5. `loss.backward()` 把 \(\mathcal{L}_q\) 送回 `trunk` 和四个头。某一类的参考图长得像细线、而待检图用粗尺度会分错时，该类的细尺度权重会被推高。换一批参考图就重算，不记住“划痕”这个名字。
6. 梯度裁剪后 AdamW 改权重。下一集从新权重出发。

四个头从 0 开始，所以第 1 个 iteration 的调制几乎是“各半、均分、不偏袒、不放大”。后面 iteration 才慢慢离开这个点。日志里的 `g` 是本集各类 \(g\) 的平均，`alpha` 是各类尺度权重再对类取平均。平均离开 0.5，说明尺度开始被用起来；它看不出“划痕和麻点是否已经分开”。

`adapt_steps=0` 时没有第 2 步，策略只从查询损失直接学习。这时更容易看清调制本身有没有用。

## 8. 配置参数含义

都在 `config/cspd.yaml`。换数据集只改 `exp_name` 和三个绝对路径，不另写配置或脚本。

| 键 | 当前值 | 含义 |
| --- | --- | --- |
| exp_name | `NEU-CLS_v1` | 输出目录名。文件写到 `output/<exp_name>/` |
| train_dir | NEU `domain_0` 的绝对路径 | 训练类别文件夹。每个子目录是一类 |
| val_dir | NEU `domain_1` 的绝对路径 | 只在数据检查时读取，训练循环不用 |
| test_dir | NEU `domain_2` 的绝对路径 | 评估用的类别文件夹 |
| image_size | 224 | 送入骨干前的边长 |
| n_way | 5 | 每个 episode 的类别数。NEU 一共 6 类 |
| k_shot | 5 | 每类支持图张数。留一法要求至少 2 |
| q_query | 5 | 每类查询图张数 |
| feat_dim | 128 | 投影、编码、FiLM、策略的通道宽度 |
| cond_dim | 7 | 工况向量长度，必须和 `condition_vector` 一致 |
| film_rho | 0.5 | FiLM 的 gamma、beta 截断半径 |
| policy_rho | 0.5 | 通道门控偏离 1 的最大幅度 |
| topk_ratio | 0.1 | 参与图像 logits 的空间位置比例 |
| adapt_steps | 1 | 内循环步数。0 表示只做原型比较、不做快速适应 |
| adapt_lr | 0.1 | 内循环步长 |
| pretrained_backbone | true | 骨干用 ImageNet 权重 |
| freeze_backbone | true | 骨干不进优化器 |
| lr | 0.001 | 外循环 AdamW 学习率 |
| weight_decay | 0.0001 | AdamW 权重衰减 |
| grad_clip | 1.0 | 梯度全局范数上限 |
| iters | 200 | 外循环 iteration 数，一集一次 |
| eval_every | 50 | 每隔多少次在测试划分上评估 |
| eval_tasks | 20 | 每次评估抽多少个 episode |
| seed | 0 | 随机种子。测试采样器用 seed+1 |
| device | cuda | 没有 CUDA 时脚本改用 CPU |

当前代码把小样本比较和快速适应放在同一次前向里同时学。要分开看时，先把 `adapt_steps` 设为 0 训练一版，确认原型距离本身能分类，再设回 1。否则查询精度里分不清是原型有效，还是那一步支持集梯度在起作用。

## 9. 工况编码和真实生产工况

会有偏差。当前 7 个数不是产线工况，部署时不能把这套数当成原始输入。

产线上应直接输入现场记录，至少包括两类，而且两类不要混成同一个含义：

- 工艺工况：材料、表面处理、工序、批次这一类生产条件。
- 成像工况：灯组、曝光、增益、相机、视角这一类采集条件。

这些才是部署时的原始工况输入。有记录就用记录，缺哪一维就用掩码标出缺测，而不是改成另一套从像素里猜出来的定义。

当前 `cspd/data.py` 的 `condition_vector` 是临时替代，因为 NEU-CLS 没有这些字段。它从像素算出：

| 维 | 现在的计算 | 它实际表示的东西 |
| --- | --- | --- |
| 0 | 灰度均值 / 255 | 这张图有多亮 |
| 1 | 灰度标准差 / 64 | 这张图对比度 |
| 2 | 拉普拉斯方差 / 500 | 这张图有多“锐”。划痕和噪声也会让它变大 |
| 3 | 通道均值差 / 50 | 色偏。NEU 基本是灰度，这一维几乎总是 0 |
| 4 | 均值 > 180 | 第 0 维的硬阈值重复 |
| 5 | 均值 < 60 | 第 0 维的硬阈值重复 |
| 6 | 拉普拉斯方差 < 50 | 第 2 维的硬阈值重复 |

偏差有三处。

1. 输入含义不同。训练时编码网看见的是图像外观统计，部署时要喂的是工艺和成像记录。同一套 `cond_encoder` 权重对不上。
2. 第 2、6 维会把缺陷纹理算进“工况”。划痕越明显，网络越可能把它当成工况变化，而不是缺陷。
3. 第 4、5、6 维没有新增信息，只是把连续量再阈值化一次。

因此这 7 维只用于现在没有工况记录的 NEU 基线。接到真实数据时，工况向量的槽位按产线字段定义；图像统计只作为“记录缺失时的补位”，并且和工艺字段分开，部署有完整记录时就关掉补位。不要把产线的工序号塞进现在的第 0 到第 6 维。

## 10. 现在用哪份数据验证

读取不认数据集名字。`config/cspd.yaml` 里三个绝对路径目前指向 NEU-CLS 的类别文件夹：`domain_0` 训练、`domain_1` 只做检查、`domain_2` 测试。三者是同一 6 类的不同图片，标签是整图类别，没有框。

这份配置能验证的是：网络能不能训起来，查询准确率会不会明显高于 0.2，\(g\) 和尺度权重会不会离开初始的 0.5。不能把它写成换了灯光、换了缺陷类、或者位置找准了。

换数据时，把图片整理成「绝对路径 / 类别名 / 图片」，改 `exp_name`、`train_dir`、`test_dir`。MVTec 原始目录不是这个形状，要先整理成类别文件夹再填路径。GC10 当前不在数据目录里。亮度校正的偏移还没送进策略，校正图不要拿来当这一轮的训练输入。

先检查读取，再短跑，再按配置训练 200 次。输出都在 `output/NEU-CLS_v1/`。

```text
python scripts/load_data.py --config config/cspd.yaml
python scripts/train_cspd.py --config config/cspd.yaml --iters 20 --eval-tasks 5
python scripts/train_cspd.py --config config/cspd.yaml
python scripts/infer_cspd.py --config config/cspd.yaml --ckpt output/NEU-CLS_v1/last.pt
```

一次训练会写下 `config.yaml`、`config.json`、`structure.txt`、`data_summary.json`、`train.log`、`last.pt`。日志开头是这次实际用的配置和各模块参数量，完整网络文本在 `structure.txt`。

短跑看三件事：`loss` 是否往下，测试 `acc` 是否高于 0.2，`g` 是否还在 0.5 附近。200 次之后若测试准确率升了、但 `g` 和 `alpha` 仍贴着 0.5，说明这批图上策略几乎没改检测方式，涨的是投影和 FiLM。若要先把策略和内循环拆开看，把 `adapt_steps` 改成 0 再跑同一条命令。

## 11. 仍然存在的隐患

- 同分布 6 类上的精度不能写成跨工况或新类泛化。
- 整图标签被广播到全图位置再取原型。峰值可以画，不能报定位误差。
- 第 9 节的工况偏差还在代码里：训练输入仍是图像统计。
- 策略已经按类输出调制，并且只由查询交叉熵更新。输入里的工况仍是图像统计，不是产线记录，也还没有亮度校正脚本里的偏移量。5 张参考图若长得很散，按类调制会跟着这几张图乱晃。日志里的尺度权重是各类平均，看不出类和类是否已经分开。
- 扩散尚未接入。整体亮暗用亮度校正；扩散只预留给不均匀光照，且必须先有配对灯光图和缺陷保留约束。
- `k_shot=1` 时留一法会抽空某一类的原型，不要这样设。
