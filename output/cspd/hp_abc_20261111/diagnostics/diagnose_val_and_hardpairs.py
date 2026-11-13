#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline diagnosis for CR7 val group-feasibility and hard-pair injection.

No model load, no GPU training/eval, no writes to training code or official
configs, no changes to data splits. Reads only the project data root and
hp_abc_20261111 logs.

Run from the project root (or with --project):

  D:\\miniconda3\\envs\\MAML\\python.exe output/cspd/hp_abc_20261111/diagnostics/diagnose_val_and_hardpairs.py

Outputs JSON and a short markdown report in this same directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
K_CHOICES = (3, 5)
NEED_Q = 8
N_WAY_CAP = 5
TARGET_PAIRS = {
    "CR7": [("oil", "inclusion"), ("linear", "pit_local"), ("pit_local", "inclusion")],
    "GC10": [("oil", "inclusion"), ("oil", "water"), ("punch", "weld_line")],
    "NEU": [("inclusion", "pit_texture")],
}


def exists_partition(sizes, k, need_q):
    """Independent complete subset-sum: some group subset sums to [k, total-need_q]."""
    total = sum(sizes)
    if total < k + need_q:
        return False
    reach = {0}
    for s in sizes:
        reach |= {x + s for x in reach}
    cap = total - need_q
    return any(k <= x <= cap for x in reach)


def official_dp_finds(sizes, k, need_q):
    """Mirror cspd.data.CurriculumNewSplitEpisodes._split_by_group strict branch.

    Order-independent as to WHETHER a subset exists (DP). Returns bool.
    """
    total = sum(sizes)
    if total < k + need_q or not sizes:
        return False
    if len(sizes) < 2:
        return False
    states = {0: []}
    selected = None
    for index, size in enumerate(sizes):
        for prev, ids in list(states.items()):
            new_size = prev + size
            if new_size > total - need_q:
                continue
            states.setdefault(new_size, ids + [index])
            if new_size >= k:
                selected = states[new_size]
                break
        if selected is not None:
            break
    return selected is not None


def list_images(split_dir, require_mask=True):
    if not split_dir.is_dir():
        return []
    items = []
    for path in sorted(split_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIX:
            continue
        mask = split_dir / "mask" / (path.stem + ".png")
        if require_mask and not mask.is_file():
            items.append((path.stem, False, "missing_mask"))
        else:
            items.append((path.stem, True, "ok"))
    return items


def load_groups(source_dir):
    gf = source_dir / "groups.json"
    if not gf.is_file():
        return {}
    return {str(k): str(v) for k, v in json.loads(gf.read_text(encoding="utf-8")).items()}


def class_split_record(root, class_name, source, split, require_mask=True):
    source_dir = root / class_name / source
    split_dir = source_dir / split
    gmap = load_groups(source_dir)
    raw = list_images(split_dir, require_mask=False)
    used, missing_mask, missing_gid = [], 0, 0
    by = {}
    for stem, _, _ in raw:
        mask_ok = (split_dir / "mask" / (stem + ".png")).is_file()
        if require_mask and not mask_ok:
            missing_mask += 1
            continue
        used.append(stem)
        if stem not in gmap:
            missing_gid += 1
            by.setdefault("MISSING_GID", []).append(stem)
        else:
            by.setdefault(gmap[stem], []).append(stem)
    sizes = sorted((len(v) for k, v in by.items() if k != "MISSING_GID"), reverse=True)
    n = len(used)
    rec = {
        "class": class_name,
        "source": source,
        "split": split,
        "images": n,
        "raw_files": len(raw),
        "missing_mask": missing_mask,
        "missing_gid": missing_gid,
        "n_groups": len(sizes),
        "group_sizes": sizes,
        "has_groups_json": (source_dir / "groups.json").is_file(),
        "count_usable": {},
        "group_feasible": {},
        "official_dp_agrees": {},
        "infeasible_reason": {},
    }
    for k in K_CHOICES:
        count_ok = n >= k + NEED_Q and missing_gid == 0
        math_ok = bool(count_ok and exists_partition(sizes, k, NEED_Q))
        dp_ok = bool(count_ok and official_dp_finds(sizes, k, NEED_Q))
        rec["count_usable"][str(k)] = count_ok
        rec["group_feasible"][str(k)] = math_ok
        rec["official_dp_agrees"][str(k)] = dp_ok == math_ok
        if not count_ok:
            if missing_gid:
                reason = "missing_group_id"
            elif n < k + NEED_Q:
                reason = "too_few_images (need >= %d, have %d)" % (k + NEED_Q, n)
            else:
                reason = "unknown_count"
        elif math_ok:
            reason = None
        elif len(sizes) < 2:
            reason = "single_group_cannot_split (n_groups=%d, n=%d)" % (len(sizes), n)
        else:
            reason = "no_group_subset_with_sum_in_[%d,%d] sizes=%s" % (
                k, n - NEED_Q, sizes)
        rec["infeasible_reason"][str(k)] = reason
        if math_ok != dp_ok:
            rec.setdefault("dp_mismatch", {})[str(k)] = {"math": math_ok, "official_dp": dp_ok}
    return rec


def inventory(root, require_mask=True):
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    sources = set()
    rows = []
    for cname in classes:
        for sdir in sorted(p for p in (root / cname).iterdir() if p.is_dir()):
            sources.add(sdir.name)
            for split in ("train", "val", "test"):
                if (sdir / split).is_dir():
                    rows.append(class_split_record(root, cname, sdir.name, split, require_mask))
    return sorted(sources), rows


def source_class_index(rows, split):
    out = defaultdict(list)
    for r in rows:
        if r["split"] == split:
            out[r["source"]].append(r)
    return out


def pair_coverage(rows, split="train"):
    """Which diagnostic target pairs exist as same-source classes with count-usable images."""
    by = defaultdict(dict)
    for r in rows:
        if r["split"] == split:
            by[r["source"]][r["class"]] = r
    report = {}
    for src, classes in sorted(by.items()):
        present = sorted(classes)
        rec = {"classes": present, "n_classes": len(present),
               "saturated_at_nway5": len(present) <= N_WAY_CAP,
               "target_pairs_present": {},
               "oil_punch_present": {"oil": "oil" in classes, "punch": "punch" in classes}}
        for origin, pairs in TARGET_PAIRS.items():
            for a, b in pairs:
                key = "%s:%s-%s" % (origin, a, b)
                have = a in classes and b in classes
                rec["target_pairs_present"][key] = {
                    "both_present": have,
                    "count_usable_k3": bool(have and classes[a]["count_usable"]["3"] and classes[b]["count_usable"]["3"]),
                    "count_usable_k5": bool(have and classes[a]["count_usable"]["5"] and classes[b]["count_usable"]["5"]),
                    "group_feasible_val_if_same_split": None,
                }
        report[src] = rec
    return report


def cr7_combo_math(cr7_val_rows):
    by = {r["class"]: r for r in cr7_val_rows}
    out = {}
    for k in K_CHOICES:
        count_usable = sorted(c for c, r in by.items() if r["count_usable"][str(k)])
        group_ok = sorted(c for c, r in by.items() if r["group_feasible"][str(k)])
        bad = sorted(c for c in count_usable if c not in group_ok)
        n_way = min(N_WAY_CAP, len(count_usable))
        feasible_subsets = []
        if n_way >= 2:
            for combo in combinations(count_usable, n_way):
                if all(c in group_ok for c in combo):
                    feasible_subsets.append(list(combo))
        smaller = []
        alt_n = min(N_WAY_CAP, len(group_ok))
        if 2 <= alt_n < n_way or (alt_n >= 2 and not feasible_subsets):
            for combo in combinations(group_ok, alt_n):
                smaller.append(list(combo))
        out[str(k)] = {
            "count_usable": count_usable,
            "group_feasible": group_ok,
            "count_usable_but_group_infeasible": bad,
            "n_way_as_code": n_way,
            "n_count_usable": len(count_usable),
            "n_group_feasible": len(group_ok),
            "n_five_way_subsets_all_group_feasible": len(list(combinations(group_ok, 5))) if len(group_ok) >= 5 else 0,
            "n_code_nway_subsets_all_group_feasible": len(feasible_subsets),
            "any_code_nway_solution": len(feasible_subsets) > 0,
            "any_smaller_nway_from_group_feasible": len(group_ok) >= 2,
            "smaller_nway": min(N_WAY_CAP, len(group_ok)) if len(group_ok) >= 2 else 0,
            "verdict": (
                "code_nway_has_solution" if feasible_subsets else
                ("no_code_nway_solution_but_smaller_nway_exists" if len(group_ok) >= 2 else
                 "insufficient_group_feasible_classes")
            ),
        }
    return out


def dummy_load(sampler):
    import torch

    dim = int(getattr(sampler.cond, "dim", 8))

    def _load(path, mask_path, source_name):
        return {
            "x": torch.zeros(3, 1, 1),
            "y_cond": torch.zeros(dim),
            "y_cond_mask": torch.zeros(dim),
            "mask": torch.zeros(1, 1, 1),
            "has_mask": True,
            "path": str(path),
        }

    sampler._load = _load
    return sampler


def hook_source_tries(sampler, sink):
    """Record every _sample_source attempt, including those swallowed when another source later succeeds."""
    orig = sampler._sample_source

    def wrapped(query_source, cross_source, progress, stage_sources):
        try:
            ep = orig(query_source, cross_source, progress, stage_sources)
            sink.append({
                "query_source": query_source,
                "cross_source": bool(cross_source),
                "ok": True,
                "class_names": list(ep.get("class_names") or []),
                "k_shot": int(ep.get("k_shot") or 0),
                "n_way": int(ep.get("n_way") or 0),
            })
            return ep
        except RuntimeError as exc:
            sink.append({
                "query_source": query_source,
                "cross_source": bool(cross_source),
                "ok": False,
                "error": str(exc),
            })
            raise

    sampler._sample_source = wrapped
    return sampler


def replay_val(project, cfg, fold, n_evals=20, tasks=20):
    """Replay val sampling RNG path of train_curriculum without model forward."""
    sys.path.insert(0, str(project))
    from cspd.data import make_curriculum

    seed = int(cfg.get("seed", 0))
    val_sampler = make_curriculum(cfg, "val", seed + 2)
    dummy_load(val_sampler)
    tries = []
    hook_source_tries(val_sampler, tries)
    events = []
    snapshots_after_same = []
    # Match scripts/train_cspd.py: each eval does same then cross, same sampler.
    # same_validation counters are copied AFTER the same loop and BEFORE that eval's cross loop.
    for ev in range(n_evals):
        for cross in (False, True):
            start_try = len(tries)
            for _ in range(tasks):
                try:
                    ep = val_sampler.sample(
                        cross_source=cross, stage_sources=False, allow_fallback=False)
                    events.append({
                        "eval_index": ev,
                        "cross_source": cross,
                        "ok": True,
                        "query_source": ep["query_source"],
                        "class_names": list(ep["class_names"]),
                        "k_shot": int(ep["k_shot"]),
                        "n_way": int(ep["n_way"]),
                    })
                except RuntimeError as exc:
                    events.append({
                        "eval_index": ev,
                        "cross_source": cross,
                        "ok": False,
                        "error": str(exc),
                    })
            if not cross:
                snapshots_after_same.append({
                    "eval_index": ev,
                    "source_attempts": dict(val_sampler.source_attempts),
                    "source_successes": dict(val_sampler.source_successes),
                    "group_failures": dict(val_sampler.group_failures),
                })
            batch = tries[start_try:]
            for t in batch:
                t["eval_index"] = ev
                t["eval_cross"] = bool(cross)
    cr7_tries = [t for t in tries if t.get("query_source") == "CR7"]
    last_same = snapshots_after_same[-1] if snapshots_after_same else {}
    return {
        "fold": fold,
        "val_seed": seed + 2,
        "n_evals": n_evals,
        "tasks_per_mode": tasks,
        "final_source_attempts": dict(val_sampler.source_attempts),
        "final_source_successes": dict(val_sampler.source_successes),
        "final_group_failures": dict(val_sampler.group_failures),
        "after_last_same_source_attempts": last_same.get("source_attempts"),
        "after_last_same_source_successes": last_same.get("source_successes"),
        "after_last_same_group_failures": last_same.get("group_failures"),
        "cr7_tries": cr7_tries,
        "cr7_try_ok": sum(1 for t in cr7_tries if t.get("ok")),
        "cr7_try_fail": sum(1 for t in cr7_tries if not t.get("ok")),
        "cr7_fail_errors": dict(Counter(t.get("error") for t in cr7_tries if not t.get("ok"))),
        "n_events": len(events),
        "n_ok": sum(1 for e in events if e.get("ok")),
        "n_fail_all_sources": sum(1 for e in events if not e.get("ok")),
        "cr7_successes": [e for e in events if e.get("ok") and e.get("query_source") == "CR7"],
    }


def analyze_hard_jsonl(path, holdout):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    train_rows = [r for r in rows if r.get("probability") is not None]
    val_rows = [r for r in rows if r.get("probability") is None]
    totals = defaultdict(lambda: Counter())
    changed_pairs = defaultdict(Counter)
    fallback_by_source = Counter()
    requests_by_source = Counter()
    applied_unchanged = defaultdict(Counter)
    holdout_in_train = 0
    probs = Counter()
    first_request = first_changed = None
    for r in train_rows:
        probs[float(r.get("probability") or 0)] += 1
        it = r.get("iter")
        for ep in r.get("episodes") or []:
            src = ep.get("source")
            if src == holdout:
                holdout_in_train += 1
            slot = totals[src]
            slot["episodes"] += 1
            for key in ("requested", "applied", "changed", "fallback", "saturated"):
                if ep.get(key):
                    slot[key] += 1
            if ep.get("requested"):
                requests_by_source[src] += 1
            if ep.get("fallback"):
                fallback_by_source[src] += 1
            if ep.get("requested") and first_request is None:
                first_request = it
            if ep.get("changed"):
                if first_changed is None:
                    first_changed = it
                pair = ep.get("pair") or []
                key = tuple(sorted(pair)) if pair else ("?",)
                changed_pairs[src][key] += 1
            if ep.get("applied") and not ep.get("changed"):
                pair = ep.get("pair") or []
                applied_unchanged[src][tuple(sorted(pair)) if pair else ("?",)] += 1
    # buffer snapshots from val rows
    buffer_active = []
    for r in val_rows:
        buf = r.get("buffer") or {}
        active = {s: [tuple(p) if not isinstance(p, tuple) else p
                      for p in (v.get("active") if isinstance(v, dict) else v or [])]
                  for s, v in buf.items()} if isinstance(buf, dict) else {}
        # buffer structure: report[source] = {active, top, ...}
        parsed = {}
        if isinstance(buf, dict):
            for s, v in buf.items():
                if isinstance(v, dict) and "active" in v:
                    parsed[s] = [tuple(p) for p in v.get("active") or []]
                elif isinstance(v, list):
                    parsed[s] = [tuple(p) for p in v]
        buffer_active.append({"iter": r.get("iter"), "active": {s: [list(p) for p in ps]
                                                                for s, ps in parsed.items()}})
    sources_ever_active = sorted({s for snap in buffer_active for s, ps in snap["active"].items() if ps})
    sources_never_active = None
    return {
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "probability_levels": {str(k): v for k, v in sorted(probs.items())},
        "max_probability": max(probs) if probs else None,
        "used_0_40": any(p >= 0.40 - 1e-9 for p in probs),
        "first_request_iter": first_request,
        "first_changed_iter": first_changed,
        "holdout_in_train_episodes": holdout_in_train,
        "per_source": {s: dict(c) for s, c in totals.items()},
        "changed_pairs": {s: {" | ".join(p): n for p, n in c.most_common()}
                          for s, c in changed_pairs.items()},
        "applied_unchanged_pairs": {s: {" | ".join(p): n for p, n in c.most_common()}
                                    for s, c in applied_unchanged.items()},
        "requests_by_source": dict(requests_by_source),
        "fallback_by_source": dict(fallback_by_source),
        "buffer_sources_ever_active": sources_ever_active,
        "buffer_last_active": buffer_active[-1] if buffer_active else None,
        "n_buffer_snapshots": len(buffer_active),
    }


def load_run_cfg(job_dir):
    cfg_path = job_dir / "config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8")), cfg_path


def extract_logged_val(metrics_path):
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    last = m["validation"][-1]["same_validation"]
    first = m["validation"][0]["same_validation"]
    return {
        "n_snapshots": len(m["validation"]),
        "first_iter": m["validation"][0]["iter"],
        "last_iter": m["validation"][-1]["iter"],
        "last_source_attempts": last.get("source_attempts") or {},
        "last_source_successes": last.get("source_successes") or {},
        "last_group_failures": last.get("group_failures") or {},
        "first_source_attempts": first.get("source_attempts") or {},
        "counts_are_cumulative": (
            (last.get("source_attempts") or {}).get("CR7", 0)
            > (first.get("source_attempts") or {}).get("CR7", 0)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path,
                        default=Path(r"D:\元强化学习\代码"))
    parser.add_argument("--run-id", default="hp_abc_20261111")
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    data_root = project / "data" / "new_split_v3"
    run_dir = project / "output" / "cspd" / args.run_id
    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources, rows = inventory(data_root, require_mask=True)
    cr7_val = [r for r in rows if r["source"] == "CR7" and r["split"] == "val"]
    cr7_train = [r for r in rows if r["source"] == "CR7" and r["split"] == "train"]
    combo = cr7_combo_math(cr7_val)
    coverage_train = pair_coverage(rows, "train")
    coverage_val = pair_coverage(rows, "val")

    jobs = {}
    for job in ("B_NEU_s0", "B_CR7_s0", "B_GC10_s0"):
        job_dir = run_dir / job
        if not (job_dir / "run_complete.json").is_file():
            continue
        cfg, cfg_path = load_run_cfg(job_dir)
        holdout = str(cfg.get("holdout_source") or "")
        logged = extract_logged_val(job_dir / "metrics.json")
        hard = analyze_hard_jsonl(job_dir / "hard_pairs.jsonl", holdout)
        train_sources = sorted(s for s in sources if s != holdout and s != "XSDD")
        # XSDD is typically excluded; confirm from cfg / data_summary
        replay = None
        if not args.skip_replay and holdout in ("NEU", "GC10"):
            replay = replay_val(project, cfg, job)
        # structural target-pair availability on training sources only
        struct = {}
        for origin, pairs in TARGET_PAIRS.items():
            struct[origin] = {}
            for a, b in pairs:
                hosts = []
                for src in train_sources:
                    info = coverage_train.get(src, {}).get("target_pairs_present", {}).get(
                        "%s:%s-%s" % (origin, a, b))
                    if info and info.get("both_present"):
                        hosts.append(src)
                struct[origin]["%s-%s" % (a, b)] = {
                    "training_sources_with_both_classes": hosts,
                    "available_on_any_training_source": bool(hosts),
                }
        jobs[job] = {
            "holdout": holdout,
            "config_path": str(cfg_path),
            "logged_val": logged,
            "hard_injection": hard,
            "training_sources": train_sources,
            "target_pair_hosts_among_training_sources": struct,
            "replay": replay,
            "replay_matches_log_attempts": (
                None if replay is None else
                (replay.get("after_last_same_source_attempts") == logged["last_source_attempts"]
                 and replay.get("after_last_same_source_successes") == logged["last_source_successes"]
                 and replay.get("after_last_same_group_failures") == logged["last_group_failures"])
            ),
        }

    # other frequently failing val classes
    extra_fail = [r for r in rows if r["split"] == "val"
                  and r["source"] in ("PLATE", "NEU", "CSU", "PIPE", "S3D", "CAST", "GC10")
                  and not r["group_feasible"]["3"]]

    summary = {
        "run_id": args.run_id,
        "data_root": str(data_root),
        "notes": [
            "Usable in code only checks image count >= k+q, then n_way=min(5,len(usable)).",
            "strict_groups then fail-closes the whole source attempt if any chosen class cannot be group-split.",
            "sample() does not retry another class combination of the same source.",
            "Val attempt counters on the sampler are cumulative across 20 evals x (same+cross).",
            "Holdout isolation was not relaxed in this diagnosis.",
        ],
        "cr7_val_classes": cr7_val,
        "cr7_train_classes": cr7_train,
        "cr7_combination_math": combo,
        "pair_coverage_train": coverage_train,
        "pair_coverage_val": {s: coverage_val[s] for s in coverage_val},
        "other_val_group_infeasible": extra_fail,
        "jobs": jobs,
    }
    json_path = out_dir / "diagnosis.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render_md(summary)
    md_path = out_dir / "diagnosis.md"
    md_path.write_text(md, encoding="utf-8")
    print("Wrote", json_path)
    print("Wrote", md_path)
    print("")
    print("===== CR7 val feasibility =====")
    for r in cr7_val:
        print("{class:10} n={images:4} groups={n_groups} sizes={group_sizes} k3={k3} k5={k5} reason3={r3}".format(
            k3=r["group_feasible"]["3"], k5=r["group_feasible"]["5"],
            r3=r["infeasible_reason"]["3"], **r))
    print("combo", json.dumps(combo, ensure_ascii=False, indent=2))
    for job, rec in jobs.items():
        print("")
        print("===== %s holdout=%s =====" % (job, rec["holdout"]))
        logv = rec["logged_val"]
        print("logged CR7 attempts/successes", logv["last_source_attempts"].get("CR7"),
              logv["last_source_successes"].get("CR7"),
              "group_failures", logv["last_group_failures"])
        if rec["replay"]:
            rp = rec["replay"]
            print("replay CR7 attempts/successes", rp["final_source_attempts"].get("CR7"),
                  rp["final_source_successes"].get("CR7"),
                  "match_log", rec["replay_matches_log_attempts"],
                  "cr7_try_ok/fail", rp.get("cr7_try_ok"), rp.get("cr7_try_fail"),
                  "match_after_same", rec["replay_matches_log_attempts"])
        print("target pair hosts", json.dumps(rec["target_pair_hosts_among_training_sources"], ensure_ascii=False))
        hi = rec["hard_injection"]
        print("hard per_source", hi["per_source"])
        print("changed_pairs", hi["changed_pairs"])
        print("buffer ever active", hi["buffer_sources_ever_active"])
        print("max p", hi["max_probability"], "used 0.40", hi["used_0_40"])


def render_md(summary):
    lines = []
    a = lines.append
    a("# hp_abc_20261111 诊断：CR7 验证缺席与困难对注入")
    a("")
    a("脚本：`output/cspd/hp_abc_20261111/diagnostics/diagnose_val_and_hardpairs.py`。")
    a("证据：`diagnosis.json`。无模型加载，未改生产代码/配置/划分。")
    a("")
    a("## CR7 val 各类容量与组可行性")
    a("")
    a("| 类 | 图像(有掩码) | 组数 | 组大小 | 缺组ID | k=3 计数可用 | k=3 组可分 | k=5 组可分 | 不可行原因 |")
    a("|---|---:|---:|---|---:|---|---|---|---|")
    for r in summary["cr7_val_classes"]:
        a("| {class} | {images} | {n_groups} | {group_sizes} | {missing_gid} | {cu3} | {g3} | {g5} | {r3} |".format(
            cu3=r["count_usable"]["3"], g3=r["group_feasible"]["3"],
            g5=r["group_feasible"]["5"], r3=r["infeasible_reason"]["3"] or "可行", **r))
    a("")
    combo = summary["cr7_combination_math"]
    a("组合结论（k=3 与 k=5 相同口径：计数可用 7 类，组可行仅 4 类 oil/inclusion/pit_local/punch）：")
    a("")
    for k, rec in combo.items():
        a("- k=%s：代码 n_way=%s，组可行 5 元组合数=%s，判定 **%s**；组可行 4 类可以构成 %s-way。" % (
            k, rec["n_way_as_code"], rec["n_code_nway_subsets_all_group_feasible"],
            rec["verdict"], rec["smaller_nway"]))
    a("")
    for job, rec in summary["jobs"].items():
        a("## %s（holdout %s）" % (job, rec["holdout"]))
        a("")
        logv = rec["logged_val"]
        a("- 日志累计（metrics.json validation 末条，含 same+cross 共享计数器）：attempts=%s successes=%s group_failures=%s" % (
            logv["last_source_attempts"], logv["last_source_successes"], logv["last_group_failures"]))
        if rec["replay"]:
            rp = rec["replay"]
            a("- 无模型复现（val seed=seed+2，20×(20 same+20 cross)；与日志比较的是每次 same 循环结束后、当次 cross 之前的计数）：attempts=%s successes=%s failures=%s 与日志完全一致=%s；CR7 _sample_source 成功/失败=%s/%s，错误=%s" % (
                rp.get("after_last_same_source_attempts"), rp.get("after_last_same_source_successes"),
                rp.get("after_last_same_group_failures"), rec["replay_matches_log_attempts"],
                rp.get("cr7_try_ok"), rp.get("cr7_try_fail"), rp.get("cr7_fail_errors")))
        a("- 训练来源上预列目标对的宿主：")
        a("```")
        a(json.dumps(rec["target_pair_hosts_among_training_sources"], ensure_ascii=False, indent=2))
        a("```")
        hi = rec["hard_injection"]
        a("- 注入：max p=%s，用过 0.40=%s；holdout 进训练=%s；各源 counts=%s" % (
            hi["max_probability"], hi["used_0_40"], hi["holdout_in_train_episodes"], hi["per_source"]))
        a("- changed 对：%s" % hi["changed_pairs"])
        a("- 缓冲曾激活的来源：%s" % hi["buffer_sources_ever_active"])
        a("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
