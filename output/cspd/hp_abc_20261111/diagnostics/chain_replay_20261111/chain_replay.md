# 离线真实链路回放（hp_abc_20261111，修复前日志）

**仍未训练。训练准入：否。** P0 已过，P1 机械注入已过，真实混淆→缓冲→组成改变链路未在修复后验证。
旧 B 日志是修复前真实模型产生，CR7 val 成功为 0，不能凭回放得到修复后的 CR7 真实混淆。
未重采采样器。HardPairBuffer 只在历史混淆上重放以核对 logged active。可行性过滤为反事实筛选。
出现次数因日志无基础类集合记为 NA，不能当 0。旧 applied 不等于新 applied。

## B_NEU_s0（holdout NEU）

- HardPairBuffer 重放与日志 active 一致：True；本日志 CR7 val successes=None
- 训练合计：{'PIPE': {'episodes': 58, 'saturated': 58, 'requested': 19, 'applied': 19}, 'GC10': {'episodes': 55, 'requested': 16, 'applied': 13, 'changed': 8, 'fallback': 3}, 'CSU': {'episodes': 60, 'saturated': 60, 'requested': 15, 'fallback': 15}, 'CR7': {'episodes': 61, 'requested': 20, 'fallback': 20}, 'PLATE': {'episodes': 55, 'saturated': 55, 'requested': 19, 'applied': 19}, 'S3D': {'episodes': 57, 'saturated': 57, 'requested': 13, 'fallback': 9, 'applied': 4}, 'CAST': {'episodes': 54, 'requested': 16, 'applied': 16, 'changed': 7}}
- 请求有 logged pair / fallback 导致 pair 未知：71 / 47
- 历史 changed 对：{'GC10': {'punch | weld_line': 2, 'inclusion | silk_spot': 2, 'silk_spot | water': 1, 'fold_wrinkle | silk_spot': 1, 'fold_wrinkle | weld_line': 1, 'inclusion | water': 1}, 'CAST': {'slag_skin | weld_slag': 4, 'slag_skin | water_slag_mark': 2, 'water_slag_mark | weld_slag': 1}, 'CR7': {}}
- 曾进入缓冲：['CAST:crack_long-weld_slag', 'CAST:slag_skin-water_slag_mark', 'CAST:slag_skin-weld_slag', 'CAST:water_slag_mark-weld_slag', 'GC10:fold_wrinkle-silk_spot', 'GC10:fold_wrinkle-weld_line', 'GC10:inclusion-silk_spot', 'GC10:inclusion-water', 'GC10:oil-punch', 'GC10:punch-weld_line', 'GC10:silk_spot-water', 'PIPE:fold_wrinkle-short_gouge', 'PIPE:fold_wrinkle-warp', 'PIPE:short_gouge-warp', 'PLATE:foreign_embed-oxide_scale', 'PLATE:inclusion-oxide_scale', 'S3D:oxide_scale-scratch']

| 目标对 | 状态 | 训练宿主 | 曾进缓冲 | changed(注入新增) | 旧applied | 出现次数 |
|---|---|---|---|---:|---:|---|
| CR7:oil-inclusion | structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix | CR7,GC10 | False | 0 | 0 | NA |
| CR7:linear-pit_local | structurally_limited | CR7 | False | 0 | 0 | NA |
| CR7:pit_local-inclusion | structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix | CR7 | False | 0 | 0 | NA |
| GC10:oil-inclusion | structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix | CR7,GC10 | False | 0 | 0 | NA |
| GC10:oil-water | structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix | GC10 | False | 0 | 0 | NA |
| GC10:punch-weld_line | historically_observed_injectable | GC10 | True | 2 | 2 | NA |
| NEU:inclusion-pit_texture | unavailable_same_source_pair | - | False | 0 | 0 | NA |

## B_CR7_s0（holdout CR7）

- HardPairBuffer 重放与日志 active 一致：True；本日志 CR7 val successes=None
- 训练合计：{'PIPE': {'episodes': 64, 'saturated': 64, 'requested': 22, 'applied': 22}, 'NEU': {'episodes': 56, 'requested': 15, 'fallback': 15}, 'CSU': {'episodes': 55, 'saturated': 55, 'requested': 18, 'fallback': 18}, 'S3D': {'episodes': 58, 'saturated': 58, 'requested': 14, 'fallback': 12, 'applied': 2}, 'CAST': {'episodes': 62, 'requested': 27, 'fallback': 5, 'applied': 22, 'changed': 9}, 'PLATE': {'episodes': 55, 'saturated': 55, 'requested': 15, 'applied': 13, 'fallback': 2}, 'GC10': {'episodes': 50, 'requested': 11, 'fallback': 4, 'applied': 7, 'changed': 5}}
- 请求有 logged pair / fallback 导致 pair 未知：66 / 56
- 历史 changed 对：{'GC10': {'punch | weld_line': 2, 'inclusion | silk_spot': 2, 'inclusion | oil': 1}, 'CAST': {'slag_skin | water_slag_mark': 6, 'crack_long | weld_slag': 2, 'slag_skin | weld_slag': 1}, 'NEU': {}}
- 曾进入缓冲：['CAST:crack_long-scratch', 'CAST:crack_long-weld_slag', 'CAST:slag_skin-water_slag_mark', 'CAST:slag_skin-weld_slag', 'CAST:water_slag_mark-weld_slag', 'GC10:fold_wrinkle-silk_spot', 'GC10:inclusion-oil', 'GC10:inclusion-silk_spot', 'GC10:inclusion-water', 'GC10:oil-punch', 'GC10:punch-weld_line', 'GC10:silk_spot-water', 'PIPE:fold_wrinkle-short_gouge', 'PIPE:fold_wrinkle-warp', 'PIPE:short_gouge-warp', 'PLATE:foreign_embed-oxide_scale', 'PLATE:inclusion-oxide_scale', 'S3D:oxide_scale-scratch']

| 目标对 | 状态 | 训练宿主 | 曾进缓冲 | changed(注入新增) | 旧applied | 出现次数 |
|---|---|---|---|---:|---:|---|
| CR7:oil-inclusion | historically_observed_injectable | GC10 | True | 1 | 2 | NA |
| CR7:linear-pit_local | unavailable_same_source_pair | - | False | 0 | 0 | NA |
| CR7:pit_local-inclusion | unavailable_same_source_pair | - | False | 0 | 0 | NA |
| GC10:oil-inclusion | historically_observed_injectable | GC10 | True | 1 | 2 | NA |
| GC10:oil-water | structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix | GC10 | False | 0 | 0 | NA |
| GC10:punch-weld_line | historically_observed_injectable | GC10 | True | 2 | 2 | NA |
| NEU:inclusion-pit_texture | structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix | NEU | False | 0 | 0 | NA |

## B_GC10_s0（holdout GC10）

- HardPairBuffer 重放与日志 active 一致：True；本日志 CR7 val successes=None
- 训练合计：{'PIPE': {'episodes': 57, 'saturated': 57, 'requested': 20, 'applied': 17, 'fallback': 3}, 'NEU': {'episodes': 59, 'requested': 16, 'fallback': 14, 'applied': 2}, 'CR7': {'episodes': 57, 'requested': 11, 'fallback': 11}, 'S3D': {'episodes': 64, 'saturated': 64, 'requested': 26, 'fallback': 19, 'applied': 7}, 'CSU': {'episodes': 58, 'saturated': 58, 'requested': 18, 'fallback': 18}, 'PLATE': {'episodes': 51, 'saturated': 51, 'requested': 14, 'applied': 14}, 'CAST': {'episodes': 54, 'requested': 15, 'applied': 15, 'changed': 6}}
- 请求有 logged pair / fallback 导致 pair 未知：55 / 65
- 历史 changed 对：{'CAST': {'slag_skin | weld_slag': 4, 'slag_skin | water_slag_mark': 2}, 'CR7': {}, 'NEU': {}}
- 曾进入缓冲：['CAST:crack_long-slag_skin', 'CAST:crack_long-weld_slag', 'CAST:slag_skin-water_slag_mark', 'CAST:slag_skin-weld_slag', 'CAST:water_slag_mark-weld_slag', 'NEU:inclusion-pit_texture', 'NEU:oxide_scale-pit_texture', 'PIPE:fold_wrinkle-short_gouge', 'PIPE:fold_wrinkle-warp', 'PIPE:short_gouge-warp', 'PLATE:foreign_embed-inclusion', 'PLATE:foreign_embed-oxide_scale', 'PLATE:inclusion-oxide_scale', 'S3D:oxide_scale-scratch']

| 目标对 | 状态 | 训练宿主 | 曾进缓冲 | changed(注入新增) | 旧applied | 出现次数 |
|---|---|---|---|---:|---:|---|
| CR7:oil-inclusion | structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix | CR7 | False | 0 | 0 | NA |
| CR7:linear-pit_local | structurally_limited | CR7 | False | 0 | 0 | NA |
| CR7:pit_local-inclusion | structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix | CR7 | False | 0 | 0 | NA |
| GC10:oil-inclusion | structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix | CR7 | False | 0 | 0 | NA |
| GC10:oil-water | unavailable_same_source_pair | - | False | 0 | 0 | NA |
| GC10:punch-weld_line | unavailable_same_source_pair | - | False | 0 | 0 | NA |
| NEU:inclusion-pit_texture | historically_observed_injectable | NEU | True | 0 | 1 | NA |

链路断裂：修复前 CR7 val 无成功混淆 → CR7 缓冲为空 → CR7 请求全部 fallback（pair 未知）→ 无 CR7 changed。该缺口无法用旧日志离线补全。
