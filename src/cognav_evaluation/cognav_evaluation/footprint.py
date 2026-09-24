"""Render one episode as a PNG: trail, traversed cells, contacts, observed points.

    cognav_footprint results/<batch>/<arm>/*.log --out figures/footprints
    cognav_footprint <log> --map <lab map yaml>

The traversed cells are the ones `metrics.coverage_gridless` counts. Observed
points need `tree_log_ranges:=full`. With `--map` the episode is drawn over the
occupancy map.
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cognav_evaluation.metrics import summarise
from cognav_evaluation.occupancy import OccupancyMap
from cognav_evaluation.tick_log import parse_log, parse_preamble


def _coverage_cells(trail, robot_radius: float, cell_m: float):
    """The same cells metrics.coverage_gridless counts."""
    cells = set()
    rad = max(1, int(round(robot_radius / cell_m)))
    for x, y, _ in trail:
        c0, r0 = int(math.floor(x / cell_m)), int(math.floor(y / cell_m))
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr * dr + dc * dc <= rad * rad:
                    cells.add((r0 + dr, c0 + dc))
    return cells


def _observed_points(ticks, fov_deg: float):
    """Back-project every finite bin into world coordinates."""
    xs, ys = [], []
    half = math.radians(fov_deg) / 2.0
    for t in ticks:
        if t.geometry_pose is None or not t.raw_ranges:
            continue
        x, y, yaw = t.geometry_pose
        n = len(t.raw_ranges)
        for i, d in enumerate(t.raw_ranges):
            if not math.isfinite(d):
                continue
            bearing = yaw + half - (2 * half) * (i / max(n - 1, 1))
            xs.append(x + d * math.cos(bearing))
            ys.append(y + d * math.sin(bearing))
    return np.asarray(xs), np.asarray(ys)


def render(log_path: str, out_path: str, occ: OccupancyMap | None = None,
           robot_radius: float = 0.18, cell_m: float = 0.05,
           area_m2: float | None = None) -> str:
    ticks = parse_log(log_path)
    if not ticks:
        raise ValueError(f"{log_path}: no tick blocks, nothing to draw")
    run = parse_preamble(log_path)
    m = summarise(ticks, log_path, occ, robot_radius,
                  area_m2=area_m2 or run.get("navigable_area_m2"))
    trail = [t.geometry_pose for t in ticks if t.geometry_pose is not None]
    if not trail:
        raise ValueError(f"{log_path}: no poses, nothing to draw")

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=130)

    if occ is not None:
        h, w = occ.shape
        x0, y0 = occ.origin
        ax.imshow(occ.grid, cmap="gray", origin="upper",
                  extent=[x0, x0 + w * occ.resolution,
                          y0, y0 + h * occ.resolution], alpha=0.55, zorder=0)

    ox, oy = _observed_points(ticks, float(run.get("fov_deg", 89.0)))
    if ox.size:
        ax.scatter(ox, oy, s=0.4, c="#8a8a8a", alpha=0.25, linewidths=0,
                   zorder=1, label=f"observed bins ({ox.size:,})")

    cells = _coverage_cells(trail, robot_radius, cell_m)
    if cells:
        cx = np.asarray([c * cell_m + cell_m / 2 for _, c in cells])
        cy = np.asarray([r * cell_m + cell_m / 2 for r, _ in cells])
        ax.scatter(cx, cy, s=1.6, c="#4c9be8", alpha=0.35, linewidths=0,
                   zorder=2, label=f"covered ({m.coverage_m2:.1f} m2)"
                   if m.coverage_m2 else "covered")

    px = np.asarray([p[0] for p in trail])
    py = np.asarray([p[1] for p in trail])
    # Coloured by time, so doubling back is visible.
    ax.scatter(px, py, s=2.2, c=np.linspace(0, 1, px.size), cmap="viridis",
               zorder=3, linewidths=0)

    sx, sy, syaw = trail[0]
    ax.plot(sx, sy, "o", ms=9, mfc="none", mec="#1a7f37", mew=2, zorder=5)
    ax.arrow(sx, sy, 0.45 * math.cos(syaw), 0.45 * math.sin(syaw),
             head_width=0.14, color="#1a7f37", zorder=5, length_includes_head=True)
    ax.plot(trail[-1][0], trail[-1][1], "s", ms=7, mfc="none", mec="#333",
            mew=1.6, zorder=5, label="end")

    hits = [(t.geometry_pose[0], t.geometry_pose[1]) for t in ticks if t.contacts and t.geometry_pose]
    if hits:
        hx = [h[0] for h in hits]
        hy = [h[1] for h in hits]
        ax.scatter(hx, hy, s=52, marker="x", c="#d1242f", linewidths=1.6,
                   zorder=6, label=f"contact ({m.collisions} in {len(hits)} ticks)")

    cov = "n/a" if m.coverage_frac is None else f"{100 * m.coverage_frac:.1f}%"
    bits = [k for k in ("scene", "num_bins", "seed") if k in run]
    subtitle = "  ".join(f"{k}={os.path.basename(str(run[k]))}" for k in bits)
    ax.set_title(f"{os.path.basename(log_path)}\n"
                 f"coverage {cov}   contacts {m.collisions}   "
                 f"distance {m.distance_m:.1f} m\n"
                 f"{subtitle}", fontsize=9)
    ax.set_xlabel("x (m, odom)")
    ax.set_ylabel("y (m, odom)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85, markerscale=3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--map", help="ROS map_server YAML to draw underneath")
    ap.add_argument("--out", help="output directory (default: beside each log)")
    ap.add_argument("--robot-radius", type=float, default=0.18)
    ap.add_argument("--area", type=float, default=None,
                    help="coverage denominator in m2, overriding the log preamble")
    args = ap.parse_args()
    occ = OccupancyMap(args.map) if args.map else None
    for path in args.logs:
        out_dir = args.out or os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        if args.out:
            # Episode ids repeat across arms; prefix the arm directory.
            parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
            stem = f"{parent}__{stem}" if parent else stem
        out = os.path.join(out_dir, stem + ".png")
        try:
            print(f"  {render(path, out, occ, args.robot_radius, area_m2=args.area)}")
        except ValueError as exc:
            print(f"  skipped {os.path.basename(path)}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
