# -*- coding: utf-8 -*-
import json, csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

root = Path(r"D:\元强化学习\代码")
out = root / "output/cspd/gc10_hp_control_20261111/diagnostics/group_coverage_20261112"
out.mkdir(parents=True, exist_ok=True)

generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " Asia/Shanghai"

rep = json.loads((root / "data/new_split_v3/merge_report.json").read_text(encoding="utf-8"))

src_split = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
src_class = defaultdict(set)
for split in ["train", "val", "test"]:
    for cls, srcs in rep["splits"][split].items():
        for src, n in srcs.items():
            src_split[src][split] += int(n)
            src_class[src].add(cls)

src_groups = defaultdict(int)
for key, v in rep["groups"].items():
    cls, src = key.split("/", 1)
    src_groups[src] += int(v["groups"])

roles = {
    "CAST":  {"A_GC10": "训练+同域验证", "B_GC10": "训练+同域验证", "B_NEU": "训练+同域验证", "B_CR7": "训练+同域验证"},
    "CR7":   {"A_GC10": "训练+同域验证", "B_GC10": "训练+同域验证", "B_NEU": "训练+同域验证", "B_CR7": "留出(holdout_pool=all)"},
    "CSU":   {"A_GC10": "训练可采; 同域val实际成功0", "B_GC10": "训练可采; 同域val实际成功0", "B_NEU": "训练可采; 同域val实际成功0", "B_CR7": "训练可采; 同域val实际成功0"},
    "GC10":  {"A_GC10": "留出(holdout_pool=all, same-source)", "B_GC10": "留出(holdout_pool=all, same-source)", "B_NEU": "训练+同域验证", "B_CR7": "训练+同域验证"},
    "NEU":   {"A_GC10": "训练+同域验证", "B_GC10": "训练+同域验证", "B_NEU": "留出(holdout_pool=all)", "B_CR7": "训练+同域验证"},
    "PIPE":  {"A_GC10": "训练+同域验证", "B_GC10": "训练+同域验证", "B_NEU": "训练+同域验证", "B_CR7": "训练+同域验证"},
    "PLATE": {"A_GC10": "训练+同域验证", "B_GC10": "训练+同域验证", "B_NEU": "训练+同域验证", "B_CR7": "训练+同域验证"},
    "S3D":   {"A_GC10": "训练+同域验证(饱和2-way)", "B_GC10": "训练+同域验证(饱和2-way)", "B_NEU": "训练+同域验证(饱和2-way)", "B_CR7": "训练+同域验证(饱和2-way)"},
    "XSDD":  {"A_GC10": "排除(exclude_sources+无mask)", "B_GC10": "排除(exclude_sources+无mask)", "B_NEU": "排除", "B_CR7": "排除"},
}

mask_note = {
    "XSDD": "数据索引中有图; 收窄前目录清点1360图/0个mask文件(只查路径存在,未解码像素); 配置 exclude_sources 且 require_mask=true",
    "CSU": "merge 条目 require_mask=true; 本轮未再逐文件复核",
    "S3D": "merge 条目 require_mask=true; 本轮未再逐文件复核",
}
default_mask = "训练配置 require_mask=true; 本轮未逐文件复核mask。已有 diagnosis 对 CR7 全类及 CSU/GC10 若干 val 行记 missing_mask=0"

train_cov = {
    "A_GC10": {"PIPE": 62, "NEU": 53, "CR7": 61, "S3D": 64, "CSU": 56, "PLATE": 50, "CAST": 54},
    "B_GC10": {"PIPE": 60, "NEU": 54, "CR7": 54, "S3D": 61, "CSU": 62, "PLATE": 55, "CAST": 54},
    "B_NEU":  {"PIPE": 54, "GC10": 68, "CSU": 53, "CR7": 54, "PLATE": 56, "S3D": 57, "CAST": 58},
    "B_CR7":  {"PIPE": 60, "NEU": 66, "CSU": 57, "S3D": 57, "CAST": 55, "PLATE": 50, "GC10": 55},
}
val_succ = {
    "A_GC10": {"NEU": 66, "CR7": 58, "CAST": 62, "S3D": 71, "PIPE": 75, "PLATE": 68, "CSU": 0, "GC10": 0, "XSDD": 0},
    "B_GC10": {"NEU": 66, "CR7": 58, "CAST": 62, "S3D": 71, "PIPE": 75, "PLATE": 68, "CSU": 0, "GC10": 0, "XSDD": 0},
    "B_NEU":  {"GC10": 50, "CAST": 75, "PLATE": 66, "S3D": 67, "CR7": 76, "PIPE": 66, "CSU": 0, "NEU": 0, "XSDD": 0},
    "B_CR7":  {"GC10": 77, "PIPE": 67, "PLATE": 69, "CAST": 65, "NEU": 71, "S3D": 51, "CSU": 0, "CR7": 0, "XSDD": 0},
}
val_cum_last = {
    "A_GC10": {
        "attempts": {"CSU": 137, "NEU": 121, "CR7": 110, "CAST": 126, "S3D": 137, "PIPE": 75, "PLATE": 135},
        "successes": {"NEU": 121, "CR7": 110, "CAST": 126, "S3D": 137, "PIPE": 75, "CSU": 76, "PLATE": 135},
    },
    "B_GC10": {
        "attempts": {"CSU": 137, "NEU": 121, "CR7": 110, "CAST": 126, "S3D": 137, "PIPE": 75, "PLATE": 135},
        "successes": {"NEU": 121, "CR7": 110, "CAST": 126, "S3D": 137, "PIPE": 75, "CSU": 76, "PLATE": 135},
    },
    "B_NEU": {
        "attempts": {"CSU": 93, "GC10": 111, "CAST": 131, "PLATE": 113, "S3D": 119, "CR7": 132, "PIPE": 130},
        "successes": {"GC10": 111, "CAST": 131, "PLATE": 113, "S3D": 119, "CR7": 132, "PIPE": 130, "CSU": 44},
    },
    "B_CR7": {
        "attempts": {"GC10": 148, "PIPE": 130, "PLATE": 117, "CAST": 117, "NEU": 119, "S3D": 105, "CSU": 90},
        "successes": {"GC10": 148, "PIPE": 130, "PLATE": 117, "CAST": 117, "NEU": 119, "S3D": 105, "CSU": 44},
    },
}

diag_feas = [
    {"source": "CR7", "class": "dent", "split": "val", "images": 100, "n_groups": 2, "feas_k3q8": False, "feas_k5q8": False, "reason": "两组 sizes=[98,2]，无法分出 support k 且留下 q=8"},
    {"source": "CR7", "class": "linear", "split": "val", "images": 83, "n_groups": 1, "feas_k3q8": False, "feas_k5q8": False, "reason": "单组 n=83，strict_groups 不能拆组"},
    {"source": "CR7", "class": "macular", "split": "val", "images": 91, "n_groups": 1, "feas_k3q8": False, "feas_k5q8": False, "reason": "单组 n=91，strict_groups 不能拆组"},
    {"source": "CR7", "class": "inclusion", "split": "val", "images": 58, "n_groups": 7, "feas_k3q8": True, "feas_k5q8": True, "reason": ""},
    {"source": "CR7", "class": "oil", "split": "val", "images": 114, "n_groups": 12, "feas_k3q8": True, "feas_k5q8": True, "reason": ""},
    {"source": "CR7", "class": "pit_local", "split": "val", "images": 93, "n_groups": 6, "feas_k3q8": True, "feas_k5q8": True, "reason": ""},
    {"source": "CR7", "class": "punch", "split": "val", "images": 124, "n_groups": 6, "feas_k3q8": True, "feas_k5q8": True, "reason": ""},
    {"source": "CSU", "class": "oxide_scale_csu", "split": "val", "images": 2, "n_groups": 1, "feas_k3q8": False, "feas_k5q8": False, "reason": "too_few_images (need>=11, have 2)"},
    {"source": "CSU", "class": "scratch", "split": "val", "images": 9, "n_groups": 3, "feas_k3q8": False, "feas_k5q8": False, "reason": "too_few_images (need>=11, have 9)"},
    {"source": "GC10", "class": "rolled_pit", "split": "val", "images": 5, "n_groups": 5, "feas_k3q8": False, "feas_k5q8": False, "reason": "too_few_images (need>=11, have 5); 此为 val split。holdout_pool=all 合计31图/26组，本轮未重算 pool 可行性"},
]

summary = {
    "generated_at": generated_at,
    "scope": "narrow audit: merge_report + configs + existing diagnosis/review/metrics; no image decode; no full file/mask walk after steering",
    "opened_images": False,
    "xsd_mask_path_check_before_narrowing": {"XSDD_images_listed": 1360, "XSDD_mask_files_found": 0, "pixels_decoded": False},
    "split_rule": {
        "yaml": "config/cspd/steel_sources_v3.yaml",
        "layout": "class_source_split",
        "seed": 0,
        "ratios": [0.6, 0.2, 0.2],
        "method": "scripts/merge_sources.py split_rows: 按整个 group 划入 train/val/test，同组不跨 split。目标比例按组而非按张，实际图像比不必等于 60/20/20。",
        "group_modes": {"GC10": "regex img_\\d+_(\\d+)_\\d+ 文件名批次号", "others": "hist 32x32 threshold 0.3，是直方图近邻簇，不是真实工况/卷号"},
        "group_namespace": "source/class/id，不能全局裸 id 去重",
    },
    "sources_in_index": sorted(src_split.keys()),
    "source_level": {},
    "train_episodes": {
        "meta_batch": 4,
        "iters": 100,
        "total_per_run": 400,
        "by_run": train_cov,
        "attempts_equal_successes": True,
        "train_group_failures": {},
        "class_coverage": "NA: 训练日志无完整 class_names，不能把资格当成采到",
    },
    "same_val_success_by_source_sum_of_per_eval_by_source": val_succ,
    "same_val_note": "应使用各次 same_validation.by_source.episodes 求和。metrics 末次 source_attempts/successes 是 val_sampler 累计且中间穿插了 test_cross，不能当作纯同域验证覆盖，更不能把 20 个 snapshot 再相加。",
    "val_sampler_last_cumulative_contaminated_by_cross": val_cum_last,
    "known_infeasible_from_existing_diagnosis": diag_feas,
    "neu_plate_k_dependent": "已有 diagnosis.other_val_group_infeasible 未列入 NEU/PLATE；本轮未重扫，是否另有 k 依赖不可行未细数。",
    "exclusive_pairs_when_GC10_holdout": [
        {"pair": ["oil", "water"], "reason": "unavailable_same_source_pair: 训练来源上不同时共存"},
        {"pair": ["punch", "weld_line"], "reason": "unavailable_same_source_pair: 训练来源上不同时共存"},
    ],
    "csu_val": "CSU val 三类中 oxide_scale_csu 与 scratch 均不足 k+q，仅 inclusion 可能可行，来源级可行类 <2，四面同域 val 成功均为 0。训练 split 仍采到了 CSU。",
    "holdout_pool": "holdout_pool=all 把该留出来源自己的 train+val+test 合池，组仍隔离；不是标准只用 test split。support 类标签不用于训练网络。paired 为同来源 same 评估。",
    "paired_manifest_reused": {
        "seed": 20261111,
        "n_tasks": 20,
        "all_5way": True,
        "k3": 11,
        "k5": 9,
        "support_items": 390,
        "query_items": 800,
        "unique_source": "GC10 1190 items in manifest",
        "splits_under_pool_all": {"train": 706, "val": 242, "test": 242},
        "group_overlap": 0,
        "cross_source": False,
    },
}

for src in sorted(src_split):
    d = src_split[src]
    summary["source_level"][src] = {
        "images_train": d["train"],
        "images_val": d["val"],
        "images_test": d["test"],
        "images_total": d["train"] + d["val"] + d["test"],
        "groups_total_all_splits_namespaced_sum": src_groups[src],
        "groups_per_split": "未细数(merge_report 无按 split 组数; 本轮未读全部 groups.json 拆分)",
        "n_semantic_classes": len(src_class[src]),
        "classes": sorted(src_class[src]),
        "roles": roles[src],
        "mask_note": mask_note.get(src, default_mask),
    }

(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

feas_map = {(r["source"], r["class"], r["split"]): r for r in diag_feas}
csv_path = out / "source_class_split.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "class", "split", "images_from_merge_report", "groups_total_all_splits", "feas_k3q8", "feas_k5q8", "feas_source", "note"])
    highlight_sources = ["CR7", "CSU", "GC10", "NEU", "PLATE"]
    ordered = highlight_sources + [s for s in sorted(src_class) if s not in highlight_sources]
    for src in ordered:
        for cls in sorted(src_class[src]):
            g = rep["groups"].get(f"{cls}/{src}", {})
            for split in ["train", "val", "test"]:
                n = int(rep["splits"][split].get(cls, {}).get(src, 0) or 0)
                rec = feas_map.get((src, cls, split))
                if rec:
                    fk3, fk5, note, fsrc = rec["feas_k3q8"], rec["feas_k5q8"], rec["reason"], "diagnosis.json"
                elif src == "CR7" and split == "train":
                    fk3, fk5, note, fsrc = True, True, "diagnosis.json: CR7 train 7 类 k3/k5 均可行", "diagnosis.json"
                else:
                    fk3, fk5, note, fsrc = "未细数", "未细数", "仅图像数; 本轮未全量重算可行性", "merge_report_images_only"
                w.writerow([src, cls, split, n, g.get("groups", ""), fk3, fk5, fsrc, note])

cov = {
    "generated_at": generated_at,
    "train_episode_counts": train_cov,
    "train_note": "100 iter x meta_batch=4 = 400 episodes/run; attempts=successes, group_failures={}. 训练逐类覆盖 NA。",
    "same_val_success_sum_by_source": val_succ,
    "same_val_skipped_sum": 0,
    "not_successfully_sampled_in_same_val": {
        "all_four_runs": ["CSU", "XSDD", "the holdout source of that run"],
        "reason_CSU": "val 可行类 <2 (oxide_scale_csu、scratch 图不足; 只剩 inclusion)",
        "reason_XSDD": "exclude_sources 且无 mask，未进入 sampler 索引",
        "reason_holdout": "holdout_eval=same 时 train/val/普通 test 排除留出来源; 留出评估另路 query_only",
    },
    "do_not_use_last_cumulative_val_counters_as_same_val": val_cum_last,
    "class_train_coverage": "NA",
}
(out / "sampling_coverage.json").write_text(json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")


def train_cell(run, src):
    hold = {"A_GC10": "GC10", "B_GC10": "GC10", "B_NEU": "NEU", "B_CR7": "CR7"}[run]
    if src == "XSDD":
        return "0"
    if src == hold:
        return "留出"
    return str(train_cov[run].get(src, 0))


lines = []
a = lines.append
a("# 数据分组与采样覆盖（2026-11-12，收窄审计）")
a("")
a("生成时间：" + generated_at)
a("")
a("本轮**未打开、未解码任何图片**。未做全量 source×class×split 文件/mask 遍历。来源图像数来自 `data/new_split_v3/merge_report.json`；划分规则来自 `config/cspd/steel_sources_v3.yaml` 与 `scripts/merge_sources.py`；可行性引用已有 `output/cspd/hp_abc_20261111/diagnostics/diagnosis.json`；训练/验证覆盖来自各 run 的 `metrics.json` / `data_summary.json` / 已有 review。收窄前曾对 XSDD 做文件名与 mask 路径存在性清点（1360 图 / 0 个 mask 文件），未读像素。按 split 的独立组数除已有 diagnosis 行外**未细数**。")
a("")
a("## 先澄清：不是用同域错例确认跨域几何")
a("")
a("- 当前 GC10 paired holdout 是 **holdout_eval=same**：每条 episode 的 support 与 query **都来自 GC10**，不是跨来源 support-query。跨域轴是「训练来源 vs 留出来源」。")
a("- 先前油污 query 远离正确原型、同类错成 punch，是 **同来源 holdout** 上的原型几何观察，**不能**当作「跨域同类距离大于异类」的证据，本轮也未重新算跨域距离、未加载模型。")
a("- `rho=0`（课程 `[[0,0],[1,0]]`）只表示 **训练梯度是同源 episode**；`test_cross` 是另路评估，不进入梯度。")
a("- 分组（GC10 文件名批次 / 其余 hist 簇）是为了防止近似图泄漏，**不等于**各来源域相同。")
a("")
a("## 1. 来源层汇总（merge_report 图像数；组数为各类合计，跨 split 总计）")
a("")
a("划分：`ratios: [0.6, 0.2, 0.2]`，`layout: class_source_split`，`seed: 0`；`split_rows` **按整个 group** 分配，同组不跨 split。目标比例按组，实际张数不必刚好 60/20/20（例如 S3D 168/61/100，CR7 2594/663/814）。")
a("")
a("| 来源 | train图 | val图 | test图 | 合计图 | 组数合计(按 class 命名空间相加) | 语义类数 | 角色 A与B_GC10 / B_NEU / B_CR7 |")
a("|---|---:|---:|---:|---:|---:|---:|---|")
for src in ["CAST", "CR7", "CSU", "GC10", "NEU", "PIPE", "PLATE", "S3D", "XSDD"]:
    d = src_split[src]
    r = roles[src]
    a("| {} | {} | {} | {} | {} | {} | {} | {} / {} / {} |".format(
        src, d["train"], d["val"], d["test"], d["train"] + d["val"] + d["test"],
        src_groups[src], len(src_class[src]), r["A_GC10"], r["B_NEU"], r["B_CR7"]))
a("")
a("数据索引中的来源即上表 9 个。XSDD 在索引中（7 类 1360 图），但四折配置均为 `exclude_sources: [XSDD]` 且 `require_mask: true`，不进入训练/验证/留出。CAST/PIPE/PLATE/S3D 均在训练池（视留出轮换而定）。`data_summary` 与此一致：A/B_GC10 训练来源 CAST,CR7,CSU,NEU,PIPE,PLATE,S3D；B_NEU 用 GC10 换 NEU；B_CR7 用 GC10 换 CR7。饱和来源（类数<5）：CSU/PIPE/PLATE=3，S3D=2。")
a("")
a("组定义：GC10 为文件名正则批次；其余为 hist 相似簇，**不要把目录名或簇 id 读成人名/工况**。组 id 命名空间为 source/class/id。未匹配正则的 stem 回退不是经核实的真实批次；本轮未逐条清点回退比例。")
a("")
a("## 2. 重点类（完整逐行可行性未全表重算）")
a("")
a("### CR7 验证集全部 7 类（已有 diagnosis，k=3/5、q=8、strict_groups）")
a("")
a("| 类 | val图 | val组数 | k3q8 | k5q8 | 原因 |")
a("|---|---:|---:|---|---|---|")
a("| dent | 100 | 2 | 否 | 否 | sizes=[98,2]，无法切出合法 support/query |")
a("| linear | 83 | 1 | 否 | 否 | 单组不能拆 |")
a("| macular | 91 | 1 | 否 | 否 | 单组不能拆 |")
a("| inclusion | 58 | 7 | 是 | 是 | |")
a("| oil | 114 | 12 | 是 | 是 | |")
a("| pit_local | 93 | 6 | 是 | 是 | |")
a("| punch | 124 | 6 | 是 | 是 | |")
a("")
a("因此 CR7 同域 val 只能 4-way（inclusion/oil/pit_local/punch）。修复后 B_GC10 / A 实际 CR7 val 也只出现这 4 类；dent/linear/macular 不会被采到。CR7 **train** 7 类在已有 diagnosis 中 k3/k5 均可行。B_CR7 折 CR7 是留出，不进训练/同域 val。")
a("")
a("### CSU val（<2 个可行类）")
a("")
a("- inclusion train/val/test 图 148/53/44，总组 16。")
a("- oxide_scale_csu 8/2/2（总 12 图/6 组）：val 仅 2 图，k3/k5 均不可行。")
a("- scratch 29/9/11：val 9 图 < 11，不可行。")
a("- 故 CSU 同域 val 来源级可行类不足 2，**四面同域验证成功 episode 均为 0**。训练 split 仍采到了 CSU（资格≠验证采到）。")
a("")
a("### GC10 留出（图像数来自 merge_report；val rolled_pit 可行性来自 diagnosis）")
a("")
a("| 类 | train | val | test | 总组(跨split合计) | 已知可行性 |")
a("|---|---:|---:|---:|---:|---|")
for cls in ["crescent", "fold_wrinkle", "inclusion", "oil", "punch", "rolled_pit", "silk_spot", "water", "weld_line"]:
    nums = [int(rep["splits"][s].get(cls, {}).get("GC10", 0) or 0) for s in ["train", "val", "test"]]
    g = rep["groups"].get(f"{cls}/GC10", {})
    note = "val split 5 图不可行(k+q)；pool_all=31 图/26 组，本轮未重算" if cls == "rolled_pit" else "本轮未按 split 重算组可行性"
    a("| {} | {} | {} | {} | {} | {} |".format(cls, nums[0], nums[1], nums[2], g.get("groups", ""), note))
a("")
a("`holdout_pool=all`：评估时把 GC10 自己的 train+val+test 合池（paired manifest 实际用了 train 706 / val 242 / test 242 条目），组仍隔离。这不是「标准只用 test split」。support 标签不训练网络。专有困难对 oil-water、punch-weld_line 在 GC10 留出时为 `unavailable_same_source_pair`（训练来源不同时共存）。")
a("")
a("### NEU / PLATE")
a("")
a("图像数见附表 CSV。已有 `other_val_group_infeasible` **未列入** NEU/PLATE；本轮未重扫，是否存在 k 依赖不可行 **未细数**。prefix 诊断中 NEU inclusion-pit_texture 在 GC10 留出设定下 val 两端曾记为组可行，不能外推到全部 NEU/PLATE 类。")
a("")
a("## 3. 真实采样覆盖（日志，不是资格）")
a("")
a("训练：每 run 100 iter × meta_batch 4 = **400** 个成功 episode；`attempts=successes`，`group_failures={}`。逐类训练覆盖：**NA**（日志无完整 class_names）。")
a("")
a("| 来源 | A_GC10训练 | B_GC10训练 | B_NEU训练 | B_CR7训练 | A/B_GC10同域val成功 | B_NEU同域val成功 | B_CR7同域val成功 |")
a("|---|---:|---:|---:|---:|---:|---:|---:|")
for src in ["CAST", "CR7", "CSU", "GC10", "NEU", "PIPE", "PLATE", "S3D", "XSDD"]:
    a("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
        src,
        train_cell("A_GC10", src),
        train_cell("B_GC10", src),
        train_cell("B_NEU", src),
        train_cell("B_CR7", src),
        val_succ["A_GC10"].get(src, 0),
        val_succ["B_NEU"].get(src, 0),
        val_succ["B_CR7"].get(src, 0),
    ))
a("")
a("同域 val：20 次评估 × 20 task，skipped 合计 0；上表是各次 `by_source.episodes` 之和（A 与 B_GC10 的 val 路径一致，成功数相同）。**不要**使用 metrics 末次 `source_successes`（A/B_GC10 里 CSU 显示 76），那是 val_sampler 未重置且穿插了 `test_cross` 的累计。CSU 同域 val 真实成功是 **0**。")
a("")
a("未采到：XSDD（排除+无 mask）；各折留出来源不进训练/同域 val；CSU 不进同域 val（可行类不足）；CR7 val 的 dent/linear/macular（组不可行，修复后动态 4-way）。单组 support 只要与 query 组不同，在代码里是合法的。")
a("")
a("Paired manifest（复用已有 review，seed 20261111）：20 task 全 5-way，k=3×11 / k=5×9，390 support / 800 query，全部 GC10，组重叠 0。官方 B holdout acc 与 paired 不是同一批 20 task，不可混比。")
a("")
a("## 采样原理（按实际代码顺序，`cspd/data.py` + `scripts/train_cspd.py`）")
a("")
a("1. 训练：`rho=0` 故 `cross_source=False`。`sample()` 按当前 split 索引取候选来源（已排除 `exclude_sources` 与 holdout），打乱后依次尝试。`avoid_sources` 让同一 meta_batch 尽量换来源。")
a("2. `_sample_source`：先抽 **k**（3 或 5，概率 0.7/0.3），q=8。")
a("3. 在该来源、当前 split 上过滤 **该 k 下组划分可行** 的类（`_same_source_feasible` / 整组 subset-sum）。")
a("4. 可行类 <2 则该来源失败，试下一个来源；否则 `n_way=min(5, 可行类数)`（动态 nway）。")
a("5. 组成类别：协议下旧 hard_pairs 钩子被置空。仅当 `same_hard_pairs=true` 才按缓冲改组成；A 控制关闭注入。验证与 holdout 不看注入。")
a("6. 类内 `_split_by_group`：整组划分，support/query 组不相交。**单组当 support 合法**，只要 query 来自其他组。")
a("7. 预处理（按 mask 裁切等）后前向，训练 CE。")
a("8. 每 5 iter：同域 val（`stage_sources=False`, `allow_fallback=False`，但同一次 `sample()` 仍会在来源列表上重试），再跑 test_cross（另路）。HardPairBuffer **只用同域 val** 的 top3、连续两次确认，回流训练缓冲；**不用 holdout**。")
a("9. 留出评估：`query_only=留出来源`，`holdout_pool=all` 合池，`holdout_eval=same` 仍是同源 episode。")
a("")
a("详细数字：`summary.json`、`sampling_coverage.json`、`source_class_split.csv`（图像数完整列出；可行性仅填已有 diagnosis 行，其余标未细数）。未改文档主文，未训练、未加载模型。")
a("")

(out / "group_coverage.md").write_text("\n".join(lines), encoding="utf-8")
print("WROTE", out)
for p in sorted(out.iterdir()):
    print(p.name, p.stat().st_size)
