#!/usr/bin/env python3
"""Coverage of the reachable floor for a batch from run_experiments.sh.

    conda run -n habitat python scripts/score_batch.py --batch results/batch_<tag>

Walks <batch>/<arm>/<episode_id>.log, scores each scene with
seen_area.score_scene, and prints one table per scene; scenes are never pooled.
Per-arm JSON is cached in <batch>/coverage unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seen_area import DEFAULT_DATA_ROOT, score_scene  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True,
                    help="a results/batch_* directory holding <arm>/<episode>.log")
    ap.add_argument("--episodes-dir", default="episodes")
    ap.add_argument("--out", default=None,
                    help="where the per-arm JSON goes (default: <batch>/coverage)")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--cell-m", type=float, default=0.10)
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    batch = os.path.normpath(args.batch)
    out_root = args.out or os.path.join(batch, "coverage")
    os.makedirs(out_root, exist_ok=True)

    arms = sorted(d for d in os.listdir(batch)
                  if os.path.isdir(os.path.join(batch, d)) and d != "coverage")
    if not arms:
        print(f"no arm directories under {batch}", file=sys.stderr)
        return 1

    # scene -> arm -> {episode id: coverage}
    table: dict[str, dict[str, dict[str, float]]] = {}
    for arm in arms:
        log_dir = os.path.join(batch, arm)
        scenes = sorted({os.path.basename(f).rsplit("_", 1)[0]
                         for f in os.listdir(log_dir) if f.endswith(".log")})
        for scene in scenes:
            dest = os.path.join(out_root, f"{arm}_{scene}.json")
            if os.path.exists(dest) and not args.force:
                with open(dest, encoding="utf-8") as fh:
                    res = json.load(fh)
            else:
                manifest = os.path.join(args.episodes_dir, f"{scene}.json")
                if not os.path.exists(manifest):
                    print(f"  no manifest for {scene}, skipping", file=sys.stderr)
                    continue
                res = score_scene(manifest, log_dir, scene, args.cell_m,
                                  args.max_range, args.data_root)
                if res is None:
                    continue
                with open(dest, "w", encoding="utf-8") as fh:
                    json.dump(res, fh, indent=2)
                print(f"  wrote {dest}")
            table.setdefault(scene, {})[arm] = {
                r["episode"]: r["coverage"] for r in res["episodes"]
                if r.get("coverage") is not None}
            table[scene].setdefault("_denominator", res["denominator_m2"])

    print()
    for scene in sorted(table):
        denom = table[scene].pop("_denominator", None)
        print(f"{scene}   reachable {denom:.1f} m2" if denom else scene)
        print(f"  {'arm':16}{'n':>4}{'median':>10}{'IQR':>18}{'mean':>9}")
        for arm in arms:
            cov = list(table[scene].get(arm, {}).values())
            if not cov:
                continue
            q = st.quantiles(cov, n=4) if len(cov) >= 4 else None
            iqr = f"{q[0]:.3f}-{q[2]:.3f}" if q else "-"
            print(f"  {arm:16}{len(cov):>4}{st.median(cov):>10.3f}{iqr:>18}"
                  f"{sum(cov) / len(cov):>9.3f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
