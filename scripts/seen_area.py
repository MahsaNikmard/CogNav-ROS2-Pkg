#!/usr/bin/env python3
"""Fraction of the reachable floor each episode observed, from its tick log.

The seen cells of every ray are intersected with the navigable cells of the
scene navmesh, and divided by the navigable cells of the starting island
counted on the same lattice. Needs habitat-sim.

    conda run -n habitat python scripts/seen_area.py \\
        --manifest episodes/Adrian.json results/<batch>/<arm>/Adrian_*.log
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "cognav_evaluation"))

from cognav_evaluation.tick_log import parse_log, parse_preamble  # noqa: E402

DEFAULT_DATA_ROOT = os.environ.get(
    "COGNAV_HABITAT_DATA",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "habitat_data"))


def build_navmesh(scene, radius, height, climb):
    """A simulator with the robot's navmesh and no sensors."""
    import habitat_sim
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene
    ag = habitat_sim.agent.AgentConfiguration()
    ag.radius, ag.height = radius, height
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius, ns.agent_height, ns.agent_max_climb = radius, height, climb
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def scene_path(manifest, data_root):
    """The manifest's scene on this machine: `scene_rel` under `data_root`, else `scene`."""
    rel = manifest.get("scene_rel")
    if rel and data_root:
        candidate = os.path.join(data_root, rel)
        if os.path.exists(candidate):
            return candidate
    absolute = manifest.get("scene")
    if absolute and os.path.exists(absolute):
        return absolute
    raise SystemExit(
        f"scene not found: {os.path.join(data_root or '', rel or '')}. Pass --data-root.")


def episode_frame_to_habitat(start_hab):
    """Map an episode-frame point (x, y) back to Habitat coordinates.

    Inverts `habitat_sim_server._pose_ros`, which reports

        x_w, y_w = -(pz - sz), -(px - sx)
        (x, y)   = R(-syaw) (x_w, y_w)
    """
    sx, sy, sz, syaw = start_hab
    c, s = math.cos(syaw), math.sin(syaw)

    def to_habitat(x, y):
        x_w = c * x - s * y
        y_w = s * x + c * y
        return np.array([sx - y_w, sy, sz - x_w], dtype=np.float32)

    return to_habitat


def _centre(cell, cell_m, y):
    """Habitat-frame centre of a lattice cell at floor height y."""
    return np.array([(cell[0] + 0.5) * cell_m, y, (cell[1] + 0.5) * cell_m],
                    dtype=np.float32)


def seen_navigable(ticks, to_habitat, pathfinder, fov_deg, cell_m, max_range):
    """(seen navigable m2, seen total m2, ticks used).

    Each finite bin marks cells along its ray up to its range. Cells are
    quantised in the Habitat frame, shared by all episodes of a scene.
    """
    half = math.radians(fov_deg) / 2.0
    seen, navigable, used = set(), set(), 0
    for t in ticks:
        if t.geometry_pose is None or not t.raw_ranges:
            continue
        used += 1
        x, y, yaw = t.geometry_pose
        n = len(t.raw_ranges)
        for i, d in enumerate(t.raw_ranges):
            if not math.isfinite(d) or d <= 0.0:
                continue
            d = min(float(d), max_range)
            bearing = yaw + half - (2.0 * half) * (i + 0.5) / n
            ca, sa = math.cos(bearing), math.sin(bearing)
            for k in range(int(d / cell_m) + 1):
                r = k * cell_m
                p = to_habitat(x + r * ca, y + r * sa)
                cell = (int(math.floor(float(p[0]) / cell_m)),
                        int(math.floor(float(p[2]) / cell_m)))
                if cell in seen:
                    continue
                seen.add(cell)
                if pathfinder.is_navigable(_centre(cell, cell_m, float(p[1]))):
                    navigable.add(cell)
    a = cell_m * cell_m
    return len(navigable) * a, len(seen) * a, used


def floor_cells_m2(pathfinder, start_hab, cell_m):
    """Navigable area of the starting island, counted on the numerator's lattice."""
    lo, hi = pathfinder.get_bounds()
    start = np.asarray(start_hab[:3], dtype=np.float32)
    island = pathfinder.get_island(start)
    y = float(start[1])
    i0, i1 = int(math.floor(lo[0] / cell_m)), int(math.ceil(hi[0] / cell_m))
    j0, j1 = int(math.floor(lo[2] / cell_m)), int(math.ceil(hi[2] / cell_m))
    count = 0
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            p = _centre((i, j), cell_m, y)
            if not pathfinder.is_navigable(p):
                continue
            if island >= 0 and pathfinder.get_island(p) != island:
                continue
            count += 1
    return count * cell_m * cell_m


def score_logs(manifest_path, logs, cell_m=0.10, max_range=5.0,
               data_root=DEFAULT_DATA_ROOT):
    """Score tick logs against one manifest; logs are matched to episodes by file name."""
    with open(manifest_path, encoding="utf-8") as fh:
        man = json.load(fh)
    by_id = {e["id"]: e for e in man["episodes"]}
    sim = build_navmesh(scene_path(man, data_root), man["agent_radius"],
                        man["agent_height"], man["agent_max_climb"])
    # All episodes of a manifest start on the same island.
    denom = floor_cells_m2(sim.pathfinder, man["episodes"][0]["start_hab"], cell_m)

    rows = []
    for path in sorted(logs):
        ep = os.path.basename(path)[:-4]
        entry = by_id.get(ep)
        if entry is None:
            continue
        ticks = parse_log(path)
        if not ticks:
            continue
        run = parse_preamble(path)
        nav, tot, used = seen_navigable(
            ticks, episode_frame_to_habitat(entry["start_hab"]), sim.pathfinder,
            float(run.get("fov_deg", 89.0)), cell_m, max_range)
        rows.append({"episode": ep, "seen_navigable_m2": nav,
                     "seen_total_m2": tot,
                     "coverage": nav / denom if denom else None,
                     "ticks": used})
    sim.close()
    return {"scene": man.get("alias"), "manifest": manifest_path,
            "denominator_m2": denom,
            "denominator_exact_m2": man.get("navigable_area_m2"),
            "cell_m": cell_m, "episodes": rows}


def score_scene(manifest_path, log_dir, scene_name, cell_m=0.10, max_range=5.0,
                data_root=DEFAULT_DATA_ROOT):
    """Score every <scene_name>_*.log in `log_dir`, or None when there are none."""
    logs = glob.glob(os.path.join(log_dir, f"{scene_name}_*.log"))
    if not logs:
        return None
    result = score_logs(manifest_path, logs, cell_m, max_range, data_root)
    result["scene"] = scene_name
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="directory that the manifest's scene_rel is relative to")
    ap.add_argument("--cell-m", type=float, default=0.10)
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--json")
    args = ap.parse_args()

    res = score_logs(args.manifest, args.logs, args.cell_m, args.max_range,
                     args.data_root)
    print(f"\n  denominator: {res['denominator_m2']:.1f} m2 of starting floor on the "
          f"{args.cell_m:.2f} m lattice ({res['denominator_exact_m2']:.1f} m2 exact)\n")
    print(f"  {'episode':22}{'seen nav':>10}{'of floor':>10}{'seen total':>12}{'ticks':>7}")
    for r in res["episodes"]:
        frac = r["coverage"] if r["coverage"] is not None else float("nan")
        print(f"  {r['episode']:22}{r['seen_navigable_m2']:>9.1f}m{100 * frac:>9.1f}%"
              f"{r['seen_total_m2']:>11.1f}m{r['ticks']:>7}")
    fr = sorted(r["coverage"] for r in res["episodes"] if r["coverage"] is not None)
    if fr:
        print(f"\n  median {100 * fr[len(fr) // 2]:.1f}% of the starting floor, "
              f"range {100 * fr[0]:.1f} to {100 * fr[-1]:.1f}%")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
