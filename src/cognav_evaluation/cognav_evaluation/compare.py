"""Paired comparison of two arms over the same episode manifest.

    cognav_compare <arm A>/<scene>.json <arm B>/<scene>.json

Input files are `cognav_metrics --json` outputs. Episodes are paired by log
name; unpaired ones are reported and dropped. Continuous metrics use the
two-sided Wilcoxon signed-rank test on the paired differences, and whether an
episode had a contact uses the exact McNemar test. No p-value is given below
six non-zero differences.
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import product

import numpy as np

#: (json key, printed name, scale, decimals, direction)
#: direction: +1 means larger is better, -1 means smaller is better, 0 neutral.
_METRICS = [
    ("coverage_frac", "coverage %", 100.0, 1, +1),
    ("coverage_m2", "coverage m2", 1.0, 1, +1),
    ("coverage_auc_m2", "coverage m2 time-avg", 1.0, 1, +1),
    ("coverage_per_m", "coverage m2 per m", 1.0, 3, +1),
    ("distance_m", "distance m", 1.0, 1, 0),
    ("motion_duty", "motion duty %", 100.0, 0, +1),
    ("revisit_ratio", "revisit %", 100.0, 1, -1),
    ("collisions", "collisions", 1.0, 1, -1),
    ("collisions_per_100m", "collisions per 100 m", 1.0, 1, -1),
    # Clearance from the platform, which is the same instrument on both arms.
    ("true_min_clearance_m", "min clearance m (true)", 1.0, 2, +1),
    ("true_clearance_p05_m", "clearance p05 m (true)", 1.0, 2, +1),
    ("tick_rate_hz", "tick rate Hz", 1.0, 1, 0),
]


def episode_id(entry: dict) -> str:
    return os.path.splitext(entry.get("log", ""))[0]


def load(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    episodes = data["episodes"] if isinstance(data, dict) else data
    return {episode_id(e): e for e in episodes}


def wilcoxon(deltas):
    """Two-sided Wilcoxon signed-rank p-value, or None when it cannot be run."""
    d = np.asarray([x for x in deltas if x is not None and x != 0.0], dtype=float)
    if d.size < 6:
        return None
    try:
        from scipy.stats import wilcoxon as _w
        return float(_w(d).pvalue)
    except Exception:
        pass
    # Without scipy: exact enumeration of the sign assignments.
    if d.size > 20:
        return None
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    total = float(ranks.sum())
    observed = float(np.sum(ranks[d > 0]))
    stat = min(observed, total - observed)
    extreme = 0
    for signs in product([0.0, 1.0], repeat=d.size):
        w = float(np.dot(ranks, signs))
        if min(w, total - w) <= stat:
            extreme += 1
    return extreme / (2.0 ** d.size)


def mcnemar(a_flags, b_flags):
    """Exact McNemar for paired binary outcomes. Returns (b, c, p)."""
    b = sum(1 for x, y in zip(a_flags, b_flags) if x and not y)
    c = sum(1 for x, y in zip(a_flags, b_flags) if y and not x)
    n = b + c
    if n == 0:
        return b, c, None
    from math import comb
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return b, c, min(1.0, p)


def compare(a_path: str, b_path: str, a_label: str = "", b_label: str = "") -> dict:
    a, b = load(a_path), load(b_path)
    shared = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    a_label = a_label or os.path.basename(os.path.dirname(a_path))
    b_label = b_label or os.path.basename(os.path.dirname(b_path))

    rows = []
    for key, name, scale, nd, direction in _METRICS:
        av = [a[e].get(key) for e in shared]
        bv = [b[e].get(key) for e in shared]
        pairs = [(x, y) for x, y in zip(av, bv) if x is not None and y is not None]
        if not pairs:
            continue
        ax = np.asarray([x for x, _ in pairs], dtype=float)
        bx = np.asarray([y for _, y in pairs], dtype=float)
        delta = bx - ax
        rows.append({
            "metric": name, "key": key, "scale": scale, "decimals": nd,
            "direction": direction, "n": len(pairs),
            "a_median": float(np.median(ax)), "b_median": float(np.median(bx)),
            "delta_median": float(np.median(delta)),
            "delta_iqr": [float(np.percentile(delta, 25)),
                          float(np.percentile(delta, 75))],
            "p": wilcoxon(delta.tolist()),
        })

    a_hit = [bool(a[e].get("collisions", 0)) for e in shared]
    b_hit = [bool(b[e].get("collisions", 0)) for e in shared]
    disc_b, disc_c, p_mcnemar = mcnemar(a_hit, b_hit)
    return {
        "a": a_label, "b": b_label, "paired": len(shared),
        "only_in_a": only_a, "only_in_b": only_b,
        "rows": rows,
        "contact": {
            "a_episodes": sum(a_hit), "b_episodes": sum(b_hit),
            "a_total": sum(a[e].get("collisions", 0) for e in shared),
            "b_total": sum(b[e].get("collisions", 0) for e in shared),
            "a_blind": sum(a[e].get("blind_collisions", 0) for e in shared),
            "b_blind": sum(b[e].get("blind_collisions", 0) for e in shared),
            "discordant_a_only": disc_b, "discordant_b_only": disc_c,
            "p": p_mcnemar,
        },
    }


def render(r: dict) -> str:
    if not r["paired"]:
        return ("no episodes pair between these two runs; the ids do not match, "
                "so they are not the same manifest")
    out = [f"\n=== {r['a']}  vs  {r['b']}   {r['paired']} paired episodes ==="]
    if r["only_in_a"] or r["only_in_b"]:
        out.append(f"  dropped, present in only one arm: "
                   f"{len(r['only_in_a'])} + {len(r['only_in_b'])} "
                   f"{(r['only_in_a'] + r['only_in_b'])[:6]}")
    out.append(f"\n  {'metric':22}{'A':>10}{'B':>10}{'B - A':>12}{'p':>9}")
    for row in r["rows"]:
        s, nd = row["scale"], row["decimals"]
        p = row["p"]
        ptxt = "  n<6" if p is None else f"{p:9.3f}"
        mark = ""
        if p is not None and p < 0.05 and row["direction"]:
            better = (row["delta_median"] > 0) == (row["direction"] > 0)
            mark = "  better" if better else "  worse"
        out.append(f"  {row['metric']:22}"
                   f"{row['a_median'] * s:10.{nd}f}"
                   f"{row['b_median'] * s:10.{nd}f}"
                   f"{row['delta_median'] * s:+12.{nd}f}"
                   f"{ptxt}{mark}")
    ct = r["contact"]
    out.append(f"\n  episodes with contact   A {ct['a_episodes']}/{r['paired']}   "
               f"B {ct['b_episodes']}/{r['paired']}")
    out.append(f"  total contacts          A {ct['a_total']}   B {ct['b_total']}")
    out.append(f"  blind contacts          A {ct['a_blind']}   B {ct['b_blind']}")
    if ct["p"] is None:
        out.append("  McNemar: no discordant pairs, so the arms never differed "
                   "on whether an episode collided")
    else:
        out.append(f"  McNemar exact p={ct['p']:.3f} on {ct['discordant_a_only']} + "
                   f"{ct['discordant_b_only']} discordant pairs")
    out.append("\n  Wilcoxon signed-rank on paired differences, two-sided; zero "
               "differences are dropped\n  and fewer than six non-zero pairs "
               "give no p-value.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="baseline arm, cognav_metrics JSON")
    ap.add_argument("b", help="comparison arm, cognav_metrics JSON")
    ap.add_argument("--label-a", default="")
    ap.add_argument("--label-b", default="")
    ap.add_argument("--json")
    args = ap.parse_args()
    r = compare(args.a, args.b, args.label_a, args.label_b)
    print(render(r))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
