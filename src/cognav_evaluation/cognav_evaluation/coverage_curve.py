#!/usr/bin/env python3
"""Observed area against time for one scene, median and IQR band per arm.

    python -m cognav_evaluation.coverage_curve \\
        --arm tree=<dir> --arm navigate_only=<dir> \\
        --scene Adrian --out figures/coverage_Adrian.pdf

Each <dir> holds the `cognav_metrics --json` output <scene>.json of one arm.
Episodes that end early hold their final value. The y axis is in m2 and the
navigable area of the manifest is drawn as a reference line.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _curves(summary_path: str):
    """Every episode's coverage curve from one arm's summary JSON."""
    with open(summary_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for ep in data.get("episodes", []):
        curve = ep.get("coverage_curve") or []
        if len(curve) >= 2:
            out.append(np.asarray(curve, dtype=float))
    return out


def _resample(curves, grid_s):
    """Episodes on a common time grid, each held at its final value past its end."""
    rows = []
    for c in curves:
        t, a = c[:, 0], c[:, 1]
        rows.append(np.interp(grid_s, t, a, left=0.0, right=a[-1]))
    return np.vstack(rows) if rows else None


def render(arms: dict, out_path: str, scene: str = "",
           navigable_m2: float | None = None, duration_s: float | None = None):
    grids = []
    series = {}
    for label, path in arms.items():
        curves = _curves(path)
        if not curves:
            continue
        series[label] = curves
        grids.append(max(c[-1, 0] for c in curves))
    if not series:
        raise SystemExit("no coverage curves found; were the logs scored with "
                         "tree_log_ranges:=full?")

    end = duration_s or max(grids)
    grid_s = np.linspace(0.0, end, 200)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for label, curves in series.items():
        m = _resample(curves, grid_s)
        med = np.median(m, axis=0)
        q1, q3 = np.percentile(m, 25, axis=0), np.percentile(m, 75, axis=0)
        line, = ax.plot(grid_s, med, label=f"{label}  (n={m.shape[0]})", lw=1.8)
        ax.fill_between(grid_s, q1, q3, alpha=0.18, color=line.get_color(),
                        linewidth=0)

    if navigable_m2:
        ax.axhline(navigable_m2, ls="--", lw=0.9, color="0.45")
        ax.annotate(f"navigable {navigable_m2:.0f} m$^2$",
                    xy=(end, navigable_m2), xytext=(-4, 3),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=7, color="0.35")

    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel("observed area (m$^2$)")
    ax.set_title(scene or "", fontsize=10)
    ax.set_xlim(0, end)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="LABEL=DIR",
                   help="arm label and the directory holding its <scene>.json")
    p.add_argument("--scene", required=True,
                   help="scene name, used to find <scene>.json and to title the plot")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default="",
                   help="episodes/<scene>.json, for the navigable-area reference line")
    p.add_argument("--duration", type=float, default=None,
                   help="x-axis end in seconds; defaults to the longest episode")
    args = p.parse_args()

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            p.error(f"--arm expects LABEL=DIR, got {spec!r}")
        label, d = spec.split("=", 1)
        path = os.path.join(d, f"{args.scene}.json")
        if not os.path.exists(path):
            p.error(f"no summary for {args.scene} in {d}")
        arms[label] = path

    navigable = None
    manifest = args.manifest or os.path.join("episodes", f"{args.scene}.json")
    if os.path.exists(manifest):
        with open(manifest, "r", encoding="utf-8") as fh:
            navigable = json.load(fh).get("navigable_area_m2")

    out = render(arms, args.out, scene=args.scene, navigable_m2=navigable,
                 duration_s=args.duration)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
