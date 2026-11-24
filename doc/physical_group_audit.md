# Real-IAD 物理样本组审计

日期：2026-11-20

## 冻结规则

同一产品类别下的 `category + Sxxxx` 是 v1 物理分组主键。OK、所有 NG 缺陷子目录、五个相机视角及 Mask 必须随组管理；不跨产品合并相同 Sxxxx。

官方基准采用 `preserve_official_test`：跨官方 train/test 的完整物理组归入测试组，训练侧正常图从官方训练输入中剔除。`group_split.json` 固化组边界；它不是 Group K-Fold。

## 统计

- 物理组：15181
- 冻结训练组：5217
- 冻结测试组：9964
- 跨官方划分冲突组：2076
- 从官方训练输入剔除的冲突图像：10380

训练/测试物理组零交集断言已经通过。断言失败时训练入口必须停止。

## 固定抽样核验

固定抽取 45 个冲突组，覆盖全部产品后再按固定种子补样。
- 文件名编号与目录 Sxxxx 一致：45/45
- 时间戳可解析：45/45
- 首个 OK 时间早于首个 NG：45/45
- 单视角刚体特征对照已运行：45 组；同组内点多于同产品对照组：24/45
- 刚体特征内点只作辅助诊断。视角、照明、缺陷制造和物体重摆会降低该数值，不能据此拆分物理组。

元数据与抽样图像支持保守合并，但官方没有明确声明 Sxxxx 的跨状态全局唯一性。因此结论仍表述为 v1 防泄漏规则，不宣称官方逐件对应关系。

## 产物

- `output/audit/physical_group_manifest.csv`：逐图像物理组清单，含 Mask、官方划分和有效成员资格。
- `output/audit/cross_split_conflicts.csv`：跨官方划分冲突组。
- `output/audit/group_statistics.csv`：产品级物理组统计。
- `output/audit/group_split.json`：冻结物理组边界和零交集划分。
- `output/audit/realiad_pair_verification.csv`：固定抽样核验结果。

下一步才是基于冻结组边界设计外层产品留出和内层 OOF，不允许直接按图像执行 Group K-Fold。
