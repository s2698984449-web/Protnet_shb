# -*- coding: utf-8 -*-
"""Offline replay of EXISTING hp_abc_20261111 real logs. No weights, no resample, no training.

Command (project root):
  D:\\miniconda3\\envs\\MAML\\python.exe output/cspd/hp_abc_20261111/diagnostics/chain_replay_20261111/replay.py
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cspd.hard_pairs import HardPairBuffer

RUN = ROOT / "output" / "cspd" / "hp_abc_20261111"
OUT = Path(__file__).resolve().parent
PREFIX = RUN / "diagnostics" / "sampler_fix_20261111" / "prefix_stats.json"
JOBS = {"B_NEU_s0": "NEU", "B_CR7_s0": "CR7", "B_GC10_s0": "GC10"}
TARGETS = {
    "CR7": [("oil", "inclusion"), ("linear", "pit_local"), ("pit_local", "inclusion")],
    "GC10": [("oil", "inclusion"), ("oil", "water"), ("punch", "weld_line")],
    "NEU": [("inclusion", "pit_texture")],
}


def norm_pair(a, b=None):
    if b is None:
        a, b = a
    if isinstance(a, (list, tuple)) and b is None or (isinstance(a, (list, tuple)) and len(a) == 2 and b is None):
        a, b = a
    return tuple(sorted((str(a), str(b))))


def parse_arrow(key):
    if " -> " in key:
        left, right = key.split(" -> ", 1)
        return left, right
    return None


def bucket_from_serial(d):
    conf, opp = {}, {}
    for k, n in (d.get("confusion") or {}).items():
        parsed = parse_arrow(k)
        if parsed:
            conf[parsed] = int(n)
    for k, n in (d.get("opportunities") or {}).items():
        parsed = parse_arrow(k)
        if parsed:
            opp[parsed] = int(n)
    return {
        "per_class": d.get("per_class") or {},
        "confusion": conf,
        "opportunities": opp,
        "episodes": int(d.get("episodes") or 0),
    }


def raw_confusion_pairs(bucket):
    pairs = set()
    for (a, b), n in (bucket.get("confusion") or {}).items():
        if n:
            pairs.add(tuple(sorted((a, b))))
    return pairs


def rankable_pairs(bucket):
    errors, opp = bucket["confusion"], bucket["opportunities"]
    out = []
    seen = {tuple(sorted(p)) for p, n in errors.items() if n > 0}
    for a, b in seen:
        ab, ba = opp.get((a, b), 0), opp.get((b, a), 0)
        count = errors.get((a, b), 0) + errors.get((b, a), 0)
        if ab and ba and count >= 2:
            rate = 0.5 * (errors.get((a, b), 0) / ab + errors.get((b, a), 0) / ba)
            out.append({"pair": tuple(sorted((a, b))), "rate": rate, "errors": count})
    out.sort(key=lambda r: (-r["rate"], r["pair"]))
    return out


def load_feas(prefix):
    out = {}
    for src, rec in (prefix.get("sources") or {}).items():
        out[src] = {}
        for split, sp in (rec.get("splits") or {}).items():
            byk = sp.get("by_k") or {}
            feas = set()
            for k in ("3", "5"):
                feas.update((byk.get(k) or {}).get("group_feasible") or [])
            out[src][split] = {
                "feasible_any_k": feas,
                "train_can_change": any(
                    ((byk.get(k) or {}).get("official_n_way") is not None)
                    and ((byk.get(k) or {}).get("n_feasible") or 0)
                    > (byk.get(k) or {}).get("official_n_way")
                    for k in ("3", "5")
                ) if split == "train" else None,
            }
    return out


def pair_both_feasible(feas, source, split, pair):
    slot = (feas.get(source) or {}).get(split) or {}
    fs = slot.get("feasible_any_k") or set()
    return pair[0] in fs and pair[1] in fs


def coexist_train(prefix, source, pair, holdout):
    if source == holdout:
        return False
    rec = ((prefix.get("sources") or {}).get(source) or {}).get("splits", {}).get("train")
    if not rec:
        return False
    classes = set(rec.get("classes") or [])
    return pair[0] in classes and pair[1] in classes


def as_pair(p):
    if isinstance(p, dict):
        p = p.get("pair")
    return tuple(sorted((str(p[0]), str(p[1]))))


def replay_job(job, holdout, prefix, feas):
    metrics = json.loads((RUN / job / "metrics.json").read_text(encoding="utf-8"))
    jsonl = []
    with (RUN / job / "hard_pairs.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                jsonl.append(json.loads(line))
    train_rows = [r for r in jsonl if r.get("probability") is not None]
    val_snaps = metrics["validation"]
    train_sources = sorted({e.get("source") for r in train_rows for e in r.get("episodes") or []})

    buffer = HardPairBuffer(train_sources, holdout)
    snapshots = []
    hist_top, hist_active = set(), set()
    cf_pass, cf_fail = set(), set()
    buffer_match = True
    mismatch = []

    for snap in val_snaps:
        it = snap["iter"]
        sv = snap["same_validation"]
        by = sv.get("by_source") or {}
        logged_buf = snap.get("buffer") or {}
        if not logged_buf:
            for r in jsonl:
                if r.get("probability") is None and r.get("iter") == it:
                    logged_buf = r.get("buffer") or {}
                    break
        buckets = {s: bucket_from_serial(by[s]) for s in by}
        for s in train_sources:
            buckets.setdefault(s, {"per_class": {}, "confusion": {}, "opportunities": {}, "episodes": 0})
        replayed = buffer.update(buckets)
        row = {"iter": it, "sources": {}}
        for s in train_sources:
            bkt = buckets[s]
            raw = sorted(raw_confusion_pairs(bkt))
            rankable = [p["pair"] for p in rankable_pairs(bkt)]
            lb = logged_buf.get(s) or {}
            logged_top = [as_pair(p) for p in (lb.get("top") or [])]
            logged_active = [as_pair(p) for p in (lb.get("active") or [])]
            rp = replayed.get(s) or {}
            rep_active = [as_pair(p) for p in (rp.get("active") or [])]
            if set(logged_active) != set(rep_active):
                buffer_match = False
                mismatch.append({"iter": it, "source": s, "logged": logged_active, "replayed": rep_active})
            cf_ok, cf_no = [], []
            for p in rankable:
                if pair_both_feasible(feas, s, "val", p):
                    cf_ok.append(p)
                    cf_pass.add((s,) + p)
                else:
                    cf_no.append(p)
                    cf_fail.add((s,) + p)
            hist_top.update((s,) + p for p in logged_top)
            hist_active.update((s,) + p for p in logged_active)
            row["sources"][s] = {
                "val_episodes": bkt["episodes"],
                "confusion_pairs": [list(p) for p in raw],
                "rankable_pairs": [list(p) for p in rankable],
                "logged_top": [list(p) for p in logged_top],
                "logged_active": [list(p) for p in logged_active],
                "replayed_active": [list(p) for p in rep_active],
                "counterfactual_feas_pass": [list(p) for p in cf_ok],
                "counterfactual_feas_fail": [list(p) for p in cf_no],
            }
        snapshots.append(row)

    per_src = defaultdict(lambda: Counter())
    changed_pairs = defaultdict(Counter)
    applied_pairs = defaultdict(Counter)
    request_with_pair = 0
    request_pair_unknown = 0
    for r in train_rows:
        for e in r.get("episodes") or []:
            s = e.get("source")
            per_src[s]["episodes"] += 1
            for k in ("requested", "applied", "changed", "fallback", "saturated"):
                if e.get(k):
                    per_src[s][k] += 1
            if e.get("requested") and e.get("pair"):
                request_with_pair += 1
            elif e.get("requested") and not e.get("pair"):
                request_pair_unknown += 1
            if e.get("applied") and e.get("pair"):
                applied_pairs[s][tuple(sorted(e["pair"]))] += 1
            if e.get("changed") and e.get("pair"):
                changed_pairs[s][tuple(sorted(e["pair"]))] += 1

    targets = {}
    for origin, pairs in TARGETS.items():
        for a, b in pairs:
            key = "%s:%s-%s" % (origin, a, b)
            pair = tuple(sorted((a, b)))
            hosts = [s for s in train_sources if coexist_train(prefix, s, pair, holdout)]
            ever_buf = any((s,) + pair in hist_active for s in hosts) if hosts else False
            ever_chg = sum(changed_pairs[s][pair] for s in hosts) if hosts else 0
            ever_app = sum(applied_pairs[s][pair] for s in hosts) if hosts else 0
            val_feas = [s for s in hosts if pair_both_feasible(feas, s, "val", pair)]
            train_change = [s for s in hosts if ((feas.get(s) or {}).get("train") or {}).get("train_can_change")]
            if not hosts:
                status = "unavailable_same_source_pair"
                note = "训练来源上不同时共存；不是采样失败，不强造"
            elif ever_buf or ever_chg:
                status = "historically_observed_injectable"
                note = "ever_buffer=%s ever_changed=%s ever_applied_old=%s" % (ever_buf, ever_chg, ever_app)
            elif val_feas and train_change:
                status = "structurally_injectable_but_REAL_CHAIN_UNVERIFIED_post_fix"
                note = "val_feas=%s train_can_change=%s; 修复后真实混淆未观测" % (val_feas, train_change)
            else:
                status = "structurally_limited"
                note = "val_feas=%s train_can_change=%s" % (val_feas, train_change)
            targets[key] = {
                "pair": list(pair),
                "holdout": holdout,
                "status": status,
                "training_hosts": hosts,
                "ever_in_buffer": ever_buf,
                "changed_count": ever_chg,
                "applied_old_semantics_count": ever_app,
                "appearance_count": None,
                "appearance_count_note": "NA: jsonl episodes have no class_names/base set; cannot treat as 0",
                "injected_new_count": ever_chg,
                "note": note + "; appearance=NA",
            }

    return {
        "job": job,
        "holdout": holdout,
        "log_is_pre_fix": True,
        "cr7_val_successes_in_this_log": (val_snaps[-1]["same_validation"].get("source_successes") or {}).get("CR7"),
        "hardpairbuffer_replay_matches_logged_active": buffer_match,
        "buffer_mismatch": mismatch[:8],
        "field_limits": {
            "has_val_confusion": True,
            "has_buffer_top_active": True,
            "has_request_flag": True,
            "requested_pair_id_only_when_applied": True,
            "has_changed_pair": True,
            "has_episode_class_set": False,
            "appearance_vs_injected": "injected_new=changed with pair; appearance=NA",
            "old_applied_includes_already_in_or_saturated": True,
            "cannot_invent_post_fix_CR7_confusion": True,
        },
        "val_snapshots": snapshots,
        "train_totals": {s: dict(c) for s, c in per_src.items()},
        "changed_pairs": {s: {" | ".join(p): n for p, n in c.most_common()} for s, c in changed_pairs.items()},
        "applied_pairs_old_semantics": {s: {" | ".join(p): n for p, n in c.most_common()} for s, c in applied_pairs.items()},
        "request_events_with_logged_pair": request_with_pair,
        "request_events_pair_unknown_fallback": request_pair_unknown,
        "unique_logged_top_pairs": sorted(["%s:%s-%s" % (s, a, b) for s, a, b in hist_top]),
        "unique_logged_active_pairs": sorted(["%s:%s-%s" % (s, a, b) for s, a, b in hist_active]),
        "counterfactual_feas_pass_on_historical_rankable": sorted(["%s:%s-%s" % x for x in cf_pass]),
        "counterfactual_feas_fail_on_historical_rankable": sorted(["%s:%s-%s" % x for x in cf_fail]),
        "target_pairs": targets,
    }


def write_md(report):
    lines = []
    a = lines.append
    a("# 离线真实链路回放（hp_abc_20261111，修复前日志）")
    a("")
    a("**仍未训练。训练准入：否。** P0 已过，P1 机械注入已过，真实混淆→缓冲→组成改变链路未在修复后验证。")
    a("旧 B 日志是修复前真实模型产生，CR7 val 成功为 0，不能凭回放得到修复后的 CR7 真实混淆。")
    a("未重采采样器。HardPairBuffer 只在历史混淆上重放以核对 logged active。可行性过滤为反事实筛选。")
    a("出现次数因日志无基础类集合记为 NA，不能当 0。旧 applied 不等于新 applied。")
    a("")
    for job, rec in report["jobs"].items():
        a("## %s（holdout %s）" % (job, rec["holdout"]))
        a("")
        a("- HardPairBuffer 重放与日志 active 一致：%s；本日志 CR7 val successes=%s" % (
            rec["hardpairbuffer_replay_matches_logged_active"], rec["cr7_val_successes_in_this_log"]))
        a("- 训练合计：%s" % rec["train_totals"])
        a("- 请求有 logged pair / fallback 导致 pair 未知：%s / %s" % (
            rec["request_events_with_logged_pair"], rec["request_events_pair_unknown_fallback"]))
        a("- 历史 changed 对：%s" % rec["changed_pairs"])
        a("- 曾进入缓冲：%s" % rec["unique_logged_active_pairs"])
        a("")
        a("| 目标对 | 状态 | 训练宿主 | 曾进缓冲 | changed(注入新增) | 旧applied | 出现次数 |")
        a("|---|---|---|---|---:|---:|---|")
        for k, t in rec["target_pairs"].items():
            a("| %s | %s | %s | %s | %s | %s | NA |" % (
                k, t["status"], ",".join(t["training_hosts"]) or "-",
                t["ever_in_buffer"], t["changed_count"], t["applied_old_semantics_count"]))
        a("")
    a("链路断裂：修复前 CR7 val 无成功混淆 → CR7 缓冲为空 → CR7 请求全部 fallback（pair 未知）→ 无 CR7 changed。该缺口无法用旧日志离线补全。")
    a("")
    (OUT / "chain_replay.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    prefix = json.loads(PREFIX.read_text(encoding="utf-8"))
    feas = load_feas(prefix)
    jobs = {}
    for job, holdout in JOBS.items():
        jobs[job] = replay_job(job, holdout, prefix, feas)
    report = {
        "run_id": "hp_abc_20261111",
        "still_not_trained": True,
        "training_admission": False,
        "P0": "passed",
        "P1_mechanical": "passed",
        "real_chain_post_fix": "NOT verified; pre-fix logs cannot supply post-fix CR7 val confusion",
        "jobs": jobs,
    }
    (OUT / "chain_replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "target_pairs.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "job", "holdout", "target", "status", "training_hosts",
            "ever_in_buffer", "changed_count", "applied_old_semantics_count",
            "appearance_count", "injected_new_count", "note"])
        w.writeheader()
        for job, rec in jobs.items():
            for key, t in rec["target_pairs"].items():
                w.writerow({
                    "job": job, "holdout": rec["holdout"], "target": key,
                    "status": t["status"], "training_hosts": "|".join(t["training_hosts"]),
                    "ever_in_buffer": t["ever_in_buffer"],
                    "changed_count": t["changed_count"],
                    "applied_old_semantics_count": t["applied_old_semantics_count"],
                    "appearance_count": "NA",
                    "injected_new_count": t["injected_new_count"],
                    "note": t["note"],
                })
    with (OUT / "val_snapshots.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "job", "iter", "source", "val_episodes", "n_confusion_pairs",
            "n_rankable", "n_logged_top", "n_logged_active", "logged_active",
            "n_cf_feas_pass", "n_cf_feas_fail"])
        w.writeheader()
        for job, rec in jobs.items():
            for snap in rec["val_snapshots"]:
                for s, v in snap["sources"].items():
                    w.writerow({
                        "job": job, "iter": snap["iter"], "source": s,
                        "val_episodes": v["val_episodes"],
                        "n_confusion_pairs": len(v["confusion_pairs"]),
                        "n_rankable": len(v["rankable_pairs"]),
                        "n_logged_top": len(v["logged_top"]),
                        "n_logged_active": len(v["logged_active"]),
                        "logged_active": ";".join("-".join(p) for p in v["logged_active"]),
                        "n_cf_feas_pass": len(v["counterfactual_feas_pass"]),
                        "n_cf_feas_fail": len(v["counterfactual_feas_fail"]),
                    })
    write_md(report)
    print("WROTE", OUT / "chain_replay.json")
    print("admission", report["training_admission"], "buffer_replay_ok",
          {j: rec["hardpairbuffer_replay_matches_logged_active"] for j, rec in jobs.items()})
    for job, rec in jobs.items():
        print("====", job, "holdout", rec["holdout"], "CR7_suc", rec["cr7_val_successes_in_this_log"])
        print(" train", rec["train_totals"])
        print(" changed", rec["changed_pairs"])
        print(" req_pair_known/unknown", rec["request_events_with_logged_pair"], rec["request_events_pair_unknown_fallback"])
        print(" active", rec["unique_logged_active_pairs"])
        for k, t in rec["target_pairs"].items():
            print(" ", k, t["status"], "buf", t["ever_in_buffer"], "chg", t["changed_count"])


if __name__ == "__main__":
    main()
