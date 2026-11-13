"""Source-local, validation-only hard pairs; no tensors, data access or loss changes."""


def hard_probability(step):
    if step <= 10:
        return 0.0
    if step <= 50:
        return 0.25
    if step <= 100:
        return 0.35
    return 0.40


def metric_bucket():
    return {"per_class": {}, "confusion": {}, "opportunities": {}, "episodes": 0}


def record_predictions(bucket, names, true_labels, pred_labels):
    bucket["episodes"] += 1
    for y, pred in zip(true_labels, pred_labels):
        true, guessed = names[y], names[pred]
        slot = bucket["per_class"].setdefault(true, [0, 0])
        slot[0] += int(y == pred)
        slot[1] += 1
        for other in names:
            if other != true:
                key = (true, other)
                bucket["opportunities"][key] = bucket["opportunities"].get(key, 0) + 1
        if y != pred:
            key = (true, guessed)
            bucket["confusion"][key] = bucket["confusion"].get(key, 0) + 1


def serial_metrics(value):
    """JSON retains directed errors and opportunity denominators without tuple keys."""
    if isinstance(value, dict):
        return {(" -> ".join(k) if isinstance(k, tuple) else k): serial_metrics(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial_metrics(v) for v in value]
    return value


class HardPairBuffer:
    """Top 3 PER SOURCE, rates averaged over both directions when co-present.

    Enter after two consecutive validations in top3; exit immediately on absence
    (including a source with zero validation episodes). No stale-pair persistence.
    """
    def __init__(self, sources, holdout="", limit=3):
        self.sources = set(sources)
        if holdout and holdout in self.sources:
            raise ValueError("holdout must not occur in training sources")
        self.holdout = holdout
        self.limit = limit
        self.streak = {}
        self.active = {}

    def update(self, by_source):
        if set(by_source) - self.sources:
            raise ValueError("non-training source in hard-pair validation")
        report, active, streak = {}, {}, {}
        for source in sorted(self.sources):
            m = by_source.get(source, metric_bucket())
            errors, opp = m["confusion"], m["opportunities"]
            pairs = {tuple(sorted(pair)) for pair, n in errors.items() if n > 0}
            ranked = []
            for a, b in pairs:
                ab, ba = opp.get((a, b), 0), opp.get((b, a), 0)
                count = errors.get((a, b), 0) + errors.get((b, a), 0)
                if ab and ba and count >= 2:
                    rate = 0.5 * (errors.get((a, b), 0) / ab + errors.get((b, a), 0) / ba)
                    ranked.append({"pair": [a, b], "rate": rate, "errors": count,
                                   "opportunities": [ab, ba]})
            ranked.sort(key=lambda r: (-r["rate"], r["pair"]))
            ranked = ranked[:self.limit]
            active[source] = []
            for row in ranked:
                pair = tuple(row["pair"])
                key = (source, pair)
                streak[key] = self.streak.get(key, 0) + 1
                row["streak"] = streak[key]
                if streak[key] >= 2:
                    active[source].append(pair)
            report[source] = {"episodes": m.get("episodes", 0), "top": ranked,
                              "active": active[source],
                              "removed": [p for p in self.active.get(source, []) if p not in active[source]]}
        self.streak, self.active = streak, active
        return report


def choose_classes(usable, n_way, rng, hard_rng, pairs, probability):
    """Uniform base episode. Replace only missing pair members, retaining n_way.

    Changed is measured against the SAME base draw, not a separate random draw.
    Applied includes already-present pairs; saturated episodes cannot change.
    """
    chosen = rng.sample(usable, n_way)
    info = {"requested": False, "applied": False, "changed": False,
            "fallback": False, "saturated": len(usable) <= n_way,
            "pair": None, "usable": len(usable), "n_way": n_way}
    if probability <= 0 or hard_rng.random() >= probability:
        return chosen, info
    info["requested"] = True
    eligible = [tuple(p) for p in pairs if len(set(p)) == 2 and all(x in usable for x in p)]
    if not eligible or n_way < 2:
        info["fallback"] = True
        return chosen, info
    pair = hard_rng.choice(eligible)
    old = set(chosen)
    for name in pair:
        if name not in chosen:
            replace = hard_rng.choice([i for i, c in enumerate(chosen) if c not in pair])
            chosen[replace] = name
    info.update(applied=True, changed=set(chosen) != old, pair=list(pair))
    return chosen, info
