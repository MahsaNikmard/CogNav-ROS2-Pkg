"""Scene descriptors: free area, occupied fraction and narrow fraction.

    cognav_arena --maps <lab map yaml>
    conda run -n habitat python -m cognav_evaluation.arena --scenes <glb> [<glb> ...]

    Area     free cells x resolution^2, in m2
    Occ. %   occupied cells over all cells of the mapped extent
    Narrow % free cells whose clearance (Euclidean distance to the nearest
             occupied cell) is below --narrow

Habitat navmeshes are rasterised with `get_topdown_view` and cropped to the
bounding box of the free space. `chord_width_median_m`, the narrowest free
chord through sampled points, is reported as a supplement.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def _stats(free: np.ndarray, resolution: float, narrow_m: float) -> dict:
    """Area, occupied fraction and narrow fraction of a boolean free-space grid."""
    occupied = ~free
    total = free.size
    # The same distance transform lab_bridge uses, with a zero body radius.
    from cognav_evaluation.occupancy import clearance_field

    class _G:
        pass

    g = _G()
    g.free, g.resolution = free, resolution
    clearance = clearance_field(g, 0.0)
    free_clear = clearance[free]
    return {
        "resolution_m": resolution,
        "cells": int(total),
        "area_m2": float(free.sum()) * resolution ** 2,
        "occupied_pct": 100.0 * float(occupied.sum()) / total,
        "narrow_pct": 100.0 * float(np.mean(free_clear < narrow_m)) if free_clear.size else None,
        "clearance_median_m": float(np.median(free_clear)) if free_clear.size else None,
        "clearance_p05_m": float(np.percentile(free_clear, 5)) if free_clear.size else None,
        "narrow_threshold_m": narrow_m,
    }


def _crop_to_free(free: np.ndarray):
    """Bounding box of the free space, the analogue of a mapped extent."""
    rows = np.flatnonzero(free.any(axis=1))
    cols = np.flatnonzero(free.any(axis=0))
    if not rows.size or not cols.size:
        return free
    return free[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def from_map(yaml_path: str, narrow_m=0.6) -> dict:
    """Characterise a ROS map_server occupancy grid."""
    from cognav_evaluation.occupancy import OccupancyMap
    occ = OccupancyMap(yaml_path)
    row = _stats(occ.free, occ.resolution, narrow_m)
    row.update(name=os.path.splitext(os.path.basename(yaml_path))[0],
               source="map", scene=yaml_path,
               extent_m=[occ.shape[1] * occ.resolution,
                         occ.shape[0] * occ.resolution])
    return row


def from_scene(scene: str, agent_radius=0.18, agent_height=0.40,
               resolution=0.05, narrow_m=0.6, seed=0, chord_samples=200,
               agent_max_climb=0.05, island=None) -> dict:
    """Characterise a Habitat scene by rasterising its navmesh."""
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.load_semantic_mesh = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.radius, agent.height = agent_radius, agent_height
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_max_climb = agent_max_climb
    settings.agent_radius = agent_radius
    settings.agent_height = agent_height
    sim.recompute_navmesh(sim.pathfinder, settings)
    pf = sim.pathfinder
    if not pf.is_loaded:
        sim.close()
        raise SystemExit(f"no navmesh for {scene}")

    # One navmesh island, the largest by default, is one arena.
    if island is None:
        island = max(range(pf.num_islands), key=pf.island_area) if pf.num_islands else -1
    island_area = float(pf.island_area(island)) if island >= 0 else float(pf.navigable_area)

    pf.seed(seed)
    pts = []
    for _ in range(chord_samples * 8):
        if len(pts) >= chord_samples:
            break
        p = (pf.get_random_navigable_point(island_index=island) if island >= 0
             else pf.get_random_navigable_point())
        if np.all(np.isfinite(p)):
            pts.append(p)
    height = float(np.median([p[1] for p in pts])) if pts else 0.0

    free = _crop_to_free(np.asarray(pf.get_topdown_view(resolution, height), dtype=bool))
    row = _stats(free, resolution, narrow_m)
    row.update(
        agent_radius=agent_radius, agent_max_climb=agent_max_climb,
        name=os.path.splitext(os.path.basename(scene))[0], source="habitat",
        scene=scene,
        extent_m=[free.shape[1] * resolution, free.shape[0] * resolution],
        navmesh_area_m2=float(pf.navigable_area),
        island_area_m2=island_area, islands=int(pf.num_islands),
        chord_width_median_m=(float(np.median([_chord(pf, p, agent_radius)
                                               for p in pts])) if pts else None),
    )
    sim.close()
    return row


def _chord(pf, p, agent_radius, step=0.05, cap=3.0, axes=6):
    """Narrowest free chord through p over `axes` directions, plus the body width."""
    best = float("inf")
    for k in range(axes):
        theta = math.pi * k / axes
        total = 0.0
        for sign in (1.0, -1.0):
            reach = 0.0
            while reach < cap:
                q = np.array([p[0] + sign * (reach + step) * math.cos(theta), p[1],
                              p[2] + sign * (reach + step) * math.sin(theta)],
                             dtype=np.float32)
                if not pf.is_navigable(q):
                    break
                reach += step
            total += reach
        best = min(best, total + 2.0 * agent_radius)
    return best


def render_table(rows, narrow_m) -> str:
    out = [f"\n  {'arena':22}{'extent':>12}{'Area':>9}{'Occ.':>8}"
           f"{'Narrow':>9}{'clr med':>9}{'clr p05':>9}{'chord med':>11}",
           "  " + "-" * 89]
    for r in rows:
        ex = f"{r['extent_m'][0]:.0f}x{r['extent_m'][1]:.0f}"
        chord = r.get("chord_width_median_m")
        out.append(
            f"  {r['name'][:20]:20}{ex:>12}{r['area_m2']:>9.0f}"
            f"{r['occupied_pct']:>7.0f}%{r['narrow_pct']:>8.0f}%"
            f"{r['clearance_median_m']:>9.2f}{r['clearance_p05_m']:>9.2f}"
            f"{(f'{chord:.2f}' if chord else '-'):>11}")
    out.append(f"\n  Area m2 = free cells x res^2.  Occ. % = occupied / all cells of the "
               f"mapped extent.\n  Narrow % = free cells with clearance < {narrow_m} m. "
               f"'chord med' = median narrowest free chord.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=[], help="habitat .glb scenes")
    ap.add_argument("--maps", nargs="*", default=[], help="ROS map_server YAMLs")
    ap.add_argument("--agent-radius", type=float, default=0.18)
    ap.add_argument("--agent-max-climb", type=float, default=0.05,
                    help="navmesh step allowance; 0.05 m keeps storeys apart")
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--narrow", type=float, default=0.6)
    ap.add_argument("--out")
    args = ap.parse_args()
    if not args.scenes and not args.maps:
        ap.error("give at least one --scenes or --maps")

    rows = []
    for path in args.maps:
        rows.append(from_map(path, narrow_m=args.narrow))
        print(f"  measured {rows[-1]['name']}", flush=True)
    for path in args.scenes:
        try:
            rows.append(from_scene(path, agent_radius=args.agent_radius,
                                   resolution=args.resolution, narrow_m=args.narrow,
                                   agent_max_climb=args.agent_max_climb))
            print(f"  measured {rows[-1]['name']}", flush=True)
        except SystemExit as exc:
            print(f"  skipped {os.path.basename(path)}: {exc}", flush=True)
    print(render_table(rows, args.narrow))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
