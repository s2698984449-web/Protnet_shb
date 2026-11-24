# 固定 Episode 协议 v1

日期：2026-11-20

## 主协议

- N-way：1
- K-shot：5，按独立物理组计数
- 每类缺陷 Query：5 个物理组
- 正常参照 R：5 个纯正常物理组
- 正常 Query：5 个纯正常物理组
- 视角：`all_views_in_physical_group`
- Episode 外缺陷：`episode_external_defects_are_ignore`

选择 1-way 作为首轮统一主协议，是因为 VisA 当前索引只有通用 anomaly 标签；先验证二分类缺陷定位。K=5、Q=5 在三个数据域均有合格任务，未合格产品不会强行补齐。

该协议合格产品：57/57；MVTec 15，VisA 12，Real-IAD 30。
类别级统计覆盖 196 行，合格产品 57 个。

## Episode Bank

- Meta-Validation Pilot：18 条 Episode。
- OOF Training：102 条 Episode。
- Meta-Validation Bank 不参与选择器梯度训练。
- 每条 OOF Episode 保存对应折的完整 `source_available_mask`；缺失 U 按 mask 缺失，不补0。
- 同一物理组的全部视角随组进入同一角色。混合物理组不会进入 R 或正常 Query。

## 验收

- R、Support、Query-Normal、Query-Defect 物理组两两不相交。
- 每类 Support 数量等于 K，每类缺陷 Query 数量等于 Q。
- Episode 只从冻结的角色池抽样，Outer Test 未进入当前两个 Bank。
- Episode ID、种子、物理组、图像路径、Mask、Ignore 类和源候选掩码均已保存。

下一步使用这两套固定 Bank 开发统一数据适配器和 Faster R-CNN 二分类定位基线；仍不启动 Δ、U 或选择器。
