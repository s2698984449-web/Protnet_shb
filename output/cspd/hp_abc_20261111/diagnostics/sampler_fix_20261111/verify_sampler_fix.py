#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-fix capacity + post-fix no-model sampler checks.

Does not load a trained model, does not run training, does not write grouping
files or official configs. Data groups.json is read-only.

  D:\\miniconda3\\envs\\MAML\\python.exe output/cspd/hp_abc_20261111/diagnostics/sampler_fix_20261111/verify_sampler_fix.py --project D:\\元强化学习\\代码
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

K_CHOICES = (3, 5)
NEED_Q = 8
N_WAY_CAP = 5
TARGET_PAIRS = {
    "CR7": [("oil", "inclusion"), ("linear", "pit_local"), ("pit_local", "inclusion")],
    "GC10": [("oil", "inclusion"), ("oil", "water"), ("punch", "weld_line")],
    "NEU": [("inclusion", "pit_texture")],
}
# Reuse helpers from the prior diagnostic without overwriting it.
PRIOR = Path(__file__).resolve().parents[1] / "diagnose_val_and_hardpairs.py"


def load_prior():
    import importlib.util
    spec = importlib.util.spec_from_file_location("hp_prior_diag", PRIOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def official_n_way(n_feasible):
    if n_feasible < 2:
        return None
    return min(N_WAY_CAP, n_feasible)


def combo_counts(n_feasible):
    out = {}
    for w in (2, 3, 4, 5):
        out[str(w)] = 0 if n_feasible < w else len(list(combinations(range(n_feasible), w)))
    return out


def prefix_stats(root, prior):
    sources, rows = prior.inventory(root, require_mask=True)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["source"]][r["split"]].append(r)
    per = {}
    for src, splits in sorted(by.items()):
        if src == "XSDD":
            continue
        rec = {"splits": {}}
        for split, items in splits.items():
            ktab = {}
            for k in K_CHOICES:
                feas = sorted(r["class"] for r in items if r["group_feasible"][str(k)])
                count_u = sorted(r["class"] for r in items if r["count_usable"][str(k)])
                infeas = [c for c in count_u if c not in feas]
                n = len(feas)
                nway = official_n_way(n)
                ktab[str(k)] = {
                    "count_usable": count_u,
                    "group_feasible": feas,
                    "count_usable_but_group_infeasible": infeas,
                    "n_feasible": n,
                    "official_n_way": nway,
                    "can_form_official_episode": nway is not None,
                    "official_episode_saturated": (nway is not None and n <= nway),
                    "offline_combo_counts_2_3_4_5": combo_counts(n),
                    "infeasible_detail": {
                        r["class"]: r["infeasible_reason"][str(k)]
                        for r in items if r["class"] in infeas or not r["count_usable"][str(k)]
                    },
                }
            rec["splits"][split] = {
                "classes": sorted({r["class"] for r in items}),
                "by_k": ktab,
            }
        per[src] = rec

    def pair_report(holdout):
        train_sources = [s for s in per if s != holdout]
        out = {}
        for origin, pairs in TARGET_PAIRS.items():
            for a, b in pairs:
                key = "%s:%s-%s" % (origin, a, b)
                hosts_train = []
                hosts_val_both_feasible = []
                reasons = []
                for src in train_sources:
                    tr = per[src]["splits"].get("train", {})
                    va = per[src]["splits"].get("val", {})
                    tr_cls = set(tr.get("classes") or [])
                    if a in tr_cls and b in tr_cls:
                        kinfo = {}
                        for k in K_CHOICES:
                            tk = tr.get("by_k", {}).get(str(k), {})
                            vk = va.get("by_k", {}).get(str(k), {})
                            both_tr = a in tk.get("group_feasible", []) and b in tk.get("group_feasible", [])
                            both_va = a in vk.get("group_feasible", []) and b in vk.get("group_feasible", [])
                            n_fe = tk.get("n_feasible", 0)
                            nway = official_n_way(n_fe)
                            kinfo[str(k)] = {
                                "both_train_group_feasible": both_tr,
                                "both_val_group_feasible": both_va,
                                "train_can_form_episode": nway is not None,
                                "train_can_change_composition": bool(nway is not None and n_fe > nway),
                            }
                        hosts_train.append({"source": src, "by_k": kinfo})
                        if any(v["both_val_group_feasible"] for v in kinfo.values()):
                            hosts_val_both_feasible.append(src)
                if not hosts_train:
                    status = "不可注入"
                    reasons.append("训练来源上不同时共存")
                else:
                    can_change = any(
                        v["train_can_change_composition"]
                        for h in hosts_train for v in h["by_k"].values()
                    )
                    can_val = bool(hosts_val_both_feasible)
                    if can_change and can_val:
                        status = "训练池机械可注入；真实缓冲仍取决于后续真实验证混淆（本次无新模型混淆）"
                    elif can_change and not can_val:
                        status = "给定pair在训练池机械可注入；但当前val组可行类不含该对的两端，真实缓冲不能从现行严格val混淆产生（无新模型混淆）"
                    elif not can_change and can_val:
                        status = "val两端组可行，但训练类池饱和或无法改变组成，机械changed不可用"
                    else:
                        status = "不可注入（组不足或无法改变组成）"
                    if not can_val:
                        reasons.append("val两端不能同时组可行，真实top3/缓冲不能在本次声称")
                    if not can_change:
                        reasons.append("训练类池无法在官方n_way下改变组成（饱和或组不足）")
                out[key] = {
                    "holdout": holdout,
                    "status": status,
                    "reasons": reasons,
                    "training_hosts": hosts_train,
                    "val_hosts_both_feasible": hosts_val_both_feasible,
                }
        return out

    return {
        "sources": per,
        "pair_injectability_by_holdout": {h: pair_report(h) for h in ("NEU", "CR7", "GC10")},
        "notes": [
            "official n_way = min(5, n_feasible) if n_feasible>=2 else cannot form; 2/3-way counts are offline only when official way is higher.",
            "When n_feasible<=5, official episodes are saturated: hard-pair changed cannot occur on that split.",
            "CR7 val 4-way is saturated by construction; val episodes do not need changed.",
        ],
    }


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


def group_ids(sampler, class_name, source, items):
    gids = []
    for item in items:
        path = item[0] if isinstance(item, tuple) else item
        stem = Path(getattr(path, "stem", Path(str(path)).stem)).stem if not hasattr(path, "stem") else path.stem
        # items after split are (path, mask)
        if isinstance(item, tuple):
            stem = item[0].stem
        gids.append(sampler.group_of.get((class_name, source, stem)))
    return set(gids)


def verify_cr7_val(project, cfg, n_direct=128, n_evals=20, tasks=20):
    sys.path.insert(0, str(project))
    from cspd.data import make_curriculum, CurriculumNewSplitEpisodes

    seed = int(cfg.get("seed", 0))
    # A: 128 direct same-source CR7 attempts, k as official (3/5).
    s = dummy_load(make_curriculum(cfg, "val", seed + 2))
    direct = []
    for i in range(n_direct):
        try:
            ep = s._sample_source("CR7", False, 0.0, False)
            # recover group intersection from stored paths? dummy load stores path.
            # support/query tensors don't keep groups; re-check via class names + we hook below.
            direct.append({
                "ok": True, "k_shot": int(ep["k_shot"]), "n_way": int(ep["n_way"]),
                "class_names": list(ep["class_names"]),
            })
        except RuntimeError as exc:
            direct.append({"ok": False, "error": str(exc)})

    # Intercept _split_by_group to record intersections for a second 128-run? Already consumed rng.
    # B: fresh sampler, wrap split to assert disjoint, 128 more? Better wrap on a third fresh sampler
    # and redo 128 with hooks. Use one fresh sampler for hooked 128 so numbers are reproducible
    # from seed+2 with 128 calls (same as A). I'll redo A with a hook from a fresh object.
    s2 = dummy_load(make_curriculum(cfg, "val", seed + 2))
    intersections = []
    orig_split = s2._split_by_group

    def wrapped(pool, k, need_q, class_name, source):
        pair = orig_split(pool, k, need_q, class_name, source)
        if pair is not None:
            g_s = {s2.group_of.get((class_name, source, p.stem)) for p, _ in pair[0]}
            g_q = {s2.group_of.get((class_name, source, p.stem)) for p, _ in pair[1]}
            inter = g_s & g_q
            intersections.append({
                "class": class_name, "source": source, "k": k,
                "intersection": [x for x in inter if x is not None],
                "empty": len(inter) == 0,
            })
        return pair
    s2._split_by_group = wrapped
    hooked = []
    for i in range(n_direct):
        try:
            ep = s2._sample_source("CR7", False, 0.0, False)
            hooked.append({
                "ok": True, "k_shot": int(ep["k_shot"]), "n_way": int(ep["n_way"]),
                "class_names": list(ep["class_names"]),
            })
        except RuntimeError as exc:
            hooked.append({"ok": False, "error": str(exc), "k_shot": None})

    # C: protocol-like mixed-source same+cross, snapshot after last same (like training logs)
    s3 = dummy_load(make_curriculum(cfg, "val", seed + 2))
    after_same = None
    for ev in range(n_evals):
        for cross in (False, True):
            for _ in range(tasks):
                try:
                    s3.sample(cross_source=cross, stage_sources=False, allow_fallback=False)
                except RuntimeError:
                    pass
            if not cross:
                after_same = {
                    "eval": ev,
                    "source_attempts": dict(s3.source_attempts),
                    "source_successes": dict(s3.source_successes),
                    "group_failures": dict(s3.group_failures),
                }

    def summarize(rows):
        ok = [r for r in rows if r.get("ok")]
        fail = [r for r in rows if not r.get("ok")]
        by_k = {}
        for r in rows:
            k = r.get("k_shot")
            # failures before return may not have k; skip
            slot = by_k.setdefault(str(k), {"ok": 0, "fail": 0, "n_way": Counter(), "classes": Counter()})
            if r.get("ok"):
                slot["ok"] += 1
                slot["n_way"][str(r["n_way"])] += 1
                for c in r["class_names"]:
                    slot["classes"][c] += 1
            else:
                slot["fail"] += 1
        return {
            "n": len(rows), "ok": len(ok), "fail": len(fail),
            "fail_errors": dict(Counter(r.get("error") for r in fail)),
            "by_k": {k: {"ok": v["ok"], "fail": v["fail"], "n_way": dict(v["n_way"]),
                         "class_coverage": dict(v["classes"])} for k, v in by_k.items()},
            "ways_seen": sorted({r["n_way"] for r in ok}),
            "classes_seen": sorted({c for r in ok for c in r["class_names"]}),
        }

    return {
        "val_seed": seed + 2,
        "direct_128": summarize(hooked),
        "group_intersections_checked": len(intersections),
        "group_intersections_nonempty": sum(1 for x in intersections if not x["empty"]),
        "protocol_after_last_same": after_same,
    }


def verify_mechanical_injection(project, cfg, given_pairs_by_source):
    """Given pairs + real train class pool. SYNTHETIC / given, not a real buffer."""
    sys.path.insert(0, str(project))
    from cspd.data import make_curriculum
    from cspd.hard_pairs import choose_classes

    seed = int(cfg.get("seed", 0))
    sampler = dummy_load(make_curriculum(cfg, "train", seed))
    sampler.hard_pair_enabled = True
    sampler.hard_probability = 1.0  # force request, mechanical
    out = {}
    for source, pairs in given_pairs_by_source.items():
        if source not in sampler.sources:
            out[source] = {"skipped": "source not in this fold's training pool"}
            continue
        sampler.same_hard_pairs = {source: [list(p) for p in pairs]}
        rows = []
        for i in range(80):
            try:
                ep = sampler._sample_source(source, False, 0.0, True)
                info = ep.get("hard_pair") or {}
                rows.append({
                    "ok": True, "k_shot": int(ep["k_shot"]), "n_way": int(ep["n_way"]),
                    "class_names": list(ep["class_names"]),
                    "usable": info.get("usable"), "saturated": info.get("saturated"),
                    "requested": info.get("requested"), "applied": info.get("applied"),
                    "changed": info.get("changed"), "fallback": info.get("fallback"),
                    "fallback_reason": info.get("fallback_reason"),
                    "pair": info.get("pair"),
                })
            except RuntimeError as exc:
                rows.append({"ok": False, "error": str(exc)})
        ok = [r for r in rows if r.get("ok")]
        pair_changed = Counter()
        for r in ok:
            if r.get("changed") and r.get("pair"):
                pair_changed[tuple(sorted(r["pair"]))] += 1
        out[source] = {
            "n": len(rows), "ok": len(ok), "fail": len(rows) - len(ok),
            "requested": sum(1 for r in ok if r.get("requested")),
            "applied": sum(1 for r in ok if r.get("applied")),
            "changed": sum(1 for r in ok if r.get("changed")),
            "fallback": sum(1 for r in ok if r.get("fallback")),
            "fallback_reasons": dict(Counter(r.get("fallback_reason") for r in ok if r.get("fallback"))),
            "saturated_ok": sum(1 for r in ok if r.get("saturated")),
            "changed_pairs": {" | ".join(p): n for p, n in pair_changed.most_common()},
            "applied_implies_changed": all((not r.get("applied")) or r.get("changed") for r in ok),
            "note": "GIVEN pairs, mechanical only. Not a real-model buffer.",
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(r"D:\元强化学习\代码"))
    parser.add_argument("--run-id", default="hp_abc_20261111")
    args = parser.parse_args()
    project = args.project.resolve()
    data_root = project / "data" / "new_split_v3"
    out_dir = project / "output" / "cspd" / args.run_id / "diagnostics" / "sampler_fix_20261111"
    out_dir.mkdir(parents=True, exist_ok=True)
    prior = load_prior()

    prefix = prefix_stats(data_root, prior)
    (out_dir / "prefix_stats.json").write_text(
        json.dumps(prefix, ensure_ascii=False, indent=2), encoding="utf-8")

    jobs = {}
    for job, holdout in (("B_NEU_s0", "NEU"), ("B_CR7_s0", "CR7"), ("B_GC10_s0", "GC10")):
        cfg = json.loads((project / "output" / "cspd" / args.run_id / job / "config.json").read_text(encoding="utf-8"))
        rec = {"holdout": holdout, "pair_injectability": prefix["pair_injectability_by_holdout"][holdout]}
        if holdout in ("NEU", "GC10"):
            rec["cr7_val"] = verify_cr7_val(project, cfg)
        # Mechanical given pairs: official target pairs hosted on training sources of this fold
        given = {}
        rec["mechanical_injection"] = None
        # CR7 given pairs only if CR7 is a training source
        if holdout != "CR7":
            given["CR7"] = list(TARGET_PAIRS["CR7"])
        if holdout != "GC10":
            given["GC10"] = list(TARGET_PAIRS["GC10"])
        if holdout != "NEU":
            given["NEU"] = list(TARGET_PAIRS["NEU"])
        rec["mechanical_injection_GIVEN_PAIRS_NOT_REAL_BUFFER"] = verify_mechanical_injection(project, cfg, given)
        jobs[job] = rec

    result = {
        "run_id": args.run_id,
        "still_not_trained": True,
        "no_new_model_val_confusion": True,
        "jobs": jobs,
    }
    (out_dir / "postfix_verify.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # compact stdout
    print("WROTE", out_dir / "prefix_stats.json")
    print("WROTE", out_dir / "postfix_verify.json")
    print("\n===== CR7 val prefix (k=3) =====")
    cr7 = prefix["sources"]["CR7"]["splits"]["val"]["by_k"]["3"]
    print("feasible", cr7["group_feasible"], "n_way", cr7["official_n_way"],
          "saturated", cr7["official_episode_saturated"], "combos", cr7["offline_combo_counts_2_3_4_5"])
    print("infeasible", cr7["count_usable_but_group_infeasible"], cr7["infeasible_detail"])
    print("\n===== pair injectability holdout=GC10 =====")
    for k, v in prefix["pair_injectability_by_holdout"]["GC10"].items():
        print(k, v["status"], v["reasons"])
    for job, rec in jobs.items():
        print("\n===== POSTFIX", job, "=====")
        if rec.get("cr7_val"):
            cv = rec["cr7_val"]
            print("CR7 direct128", cv["direct_128"])
            print("intersections nonempty", cv["group_intersections_nonempty"], "/", cv["group_intersections_checked"])
            print("protocol CR7 att/suc",
                  (cv["protocol_after_last_same"]["source_attempts"] or {}).get("CR7"),
                  (cv["protocol_after_last_same"]["source_successes"] or {}).get("CR7"))
        mech = rec["mechanical_injection_GIVEN_PAIRS_NOT_REAL_BUFFER"]
        for src, m in mech.items():
            print("MECH", src, {k: m[k] for k in m if k != "note"})


if __name__ == "__main__":
    main()
