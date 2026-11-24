# v1 外层产品族留出与内层 OOF

日期：2026-11-20
随机种子：20261120

## 协议

外层按数据集分层，以稳定哈希选择约20%的产品族作为完全封存测试组。选择过程不读取模型结果、缺陷难度或测试指标。

VisA 保守合并 `pcb1–4` 为一个产品族，`macaroni1/2` 为一个产品族。其他对象各自成族。Real-IAD 整个产品类别随外层产品族移动，类别内 `category + Sxxxx` 物理组绝不拆分。

外层训练产品族：42；外层测试产品族：11。
Meta-Train：34 个产品族；Meta-Validation：8 个产品族。
内层：5 折；每个 Meta-Train 产品族恰好作为一次OOF目标折。
候选源池主方案：`fold_excluded`。

## 外层封存产品族

- `mvtec:hazelnut`：hazelnut
- `mvtec:pill`：pill
- `mvtec:screw`：screw
- `realiad:phone_battery`：phone_battery
- `realiad:rolled_strip_base`：rolled_strip_base
- `realiad:terminalblock`：terminalblock
- `realiad:usb`：usb
- `realiad:usb_adaptor`：usb_adaptor
- `realiad:zipper`：zipper
- `visa:capsules`：capsules
- `visa:visa_pcb`：pcb1, pcb2, pcb3, pcb4

## Meta-Validation 产品族

- `mvtec:leather`：leather
- `mvtec:wood`：wood
- `realiad:button_battery`：button_battery
- `realiad:sim_card_set`：sim_card_set
- `realiad:switch`：switch
- `realiad:u_block`：u_block
- `realiad:wooden_beads`：wooden_beads
- `visa:visa_macaroni`：macaroni1, macaroni2

## 内层验证折

- fold 0：7 个验证产品族；27 个训练产品族。
- fold 1：7 个验证产品族；27 个训练产品族。
- fold 2：7 个验证产品族；27 个训练产品族。
- fold 3：7 个验证产品族；27 个训练产品族。
- fold 4：6 个验证产品族；28 个训练产品族。

## 硬性校验

- 外层训练/测试产品族交集为0。
- Meta-Train/Meta-Validation 产品族交集为0，二者完整覆盖 Outer Train。
- Meta-Validation 不参与 θ₀、Δ、选择器梯度和内层OOF。
- 外层测试产品族未进入任何内层折。
- 每个 Meta-Train 产品族恰好作为一次内层OOF目标。
- 每个内层折的训练/验证产品族交集为0。
- 每折 `source_available_mask` 与真实源任务池一致；缺失U标签不补0。
- Real-IAD 外层训练/测试物理组交集为0。
- 每个产品内 R、Support、Query-Normal、Query-Defect 四个物理组池两两不相交并完整覆盖。

已为 57 个产品生成物理组级Episode池。当前文件冻结池边界，不等于具体K-shot Episode。

下一步从这些池中生成小规模固定Episode，并把对应折的 `source_available_mask` 原样写入每条Episode。
