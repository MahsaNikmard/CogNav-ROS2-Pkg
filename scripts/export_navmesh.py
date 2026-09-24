#!/usr/bin/env python3
"""Rasterise the navmesh of each manifest's scene for the mesh arm.

    conda run -n habitat python scripts/export_navmesh.py \
        --episodes episodes/Bowlus.json --out navmesh/

Writes one .npz per manifest, read by cognav_representation.navmesh.NavMesh.
The agent radius, climb limit and island come from the manifest, so the
raster describes the surface the episodes were sampled on. Every episode start
must land on a navigable cell, or the export fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def export(scene_path: str, agent_radius: float, agent_height: float,
           agent_max_climb: float, resolution: float, island: int | None,
           seed: int = 0, samples: int = 200):
    """Return a NavMesh-shaped dict for one scene."""
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_path
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
        raise SystemExit(f"no navmesh for {scene_path}")

    # Only the island the episodes start on, at its median height.
    if island is None:
        island = max(range(pf.num_islands), key=pf.island_area) if pf.num_islands else -1
    pf.seed(seed)
    heights = []
    for _ in range(samples * 8):
        if len(heights) >= samples:
            break
        p = (pf.get_random_navigable_point(island_index=island) if island >= 0
             else pf.get_random_navigable_point())
        if np.all(np.isfinite(p)):
            heights.append(float(p[1]))
    height = float(np.median(heights)) if heights else 0.0

    free = np.asarray(pf.get_topdown_view(resolution, height), dtype=bool)
    lower, _upper = pf.get_bounds()
    # get_topdown_view spans the pathfinder bounds and is indexed [z][x].
    origin = (float(lower[0]), float(lower[2]))

    result = dict(
        free=free, resolution=float(resolution), origin=np.asarray(origin),
        height=height,
        scene=os.path.basename(scene_path), agent_radius=float(agent_radius),
        agent_height=float(agent_height), agent_max_climb=float(agent_max_climb),
        island=int(island), islands=int(pf.num_islands),
        island_area_m2=float(pf.island_area(island)) if island >= 0 else float(pf.navigable_area),
        navmesh_area_m2=float(pf.navigable_area),
    )
    sim.close()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True, nargs="+",
                    help="episode manifests; the scene and its navmesh settings "
                         "are read from each")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--data-root", default=os.environ.get(
        "COGNAV_HABITAT_DATA",
        "/opt/ws/install/cognav_platform_adapter/share/cognav_platform_adapter"
        "/simulation/habitat_data"),
        help="where scene_rel resolves against")
    ap.add_argument("--resolution", type=float, default=0.05,
                    help="metres per cell")
    ap.add_argument("--agent-radius", type=float, default=None,
                    help="override the manifest")
    ap.add_argument("--agent-height", type=float, default=0.40)
    ap.add_argument("--force", action="store_true",
                    help="re-export a scene whose .npz already exists")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src", "cognav_representation"))
    from cognav_representation.navmesh import NavMesh

    for manifest_path in args.episodes:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        alias = os.path.splitext(os.path.basename(manifest_path))[0]
        dest = os.path.join(args.out, f"{alias}.npz")
        if os.path.exists(dest) and not args.force:
            print(f"  {alias:14} exists, skipping (--force to redo)")
            continue

        scene_rel = manifest.get("scene_rel")
        if not scene_rel:
            print(f"!! {alias}: manifest has no scene_rel", file=sys.stderr)
            return 1
        scene_path = os.path.join(args.data_root, scene_rel)
        if not os.path.exists(scene_path):
            print(f"!! {alias}: scene not found at {scene_path}", file=sys.stderr)
            return 1

        radius = args.agent_radius if args.agent_radius is not None else \
            float(manifest.get("agent_radius", 0.18))
        climb = float(manifest.get("agent_max_climb", 0.05))
        island = manifest.get("island")

        print(f"  {alias:14} radius={radius} climb={climb} res={args.resolution}")
        data = export(scene_path, radius, args.agent_height, climb,
                      args.resolution, island if island is None or island >= 0 else None)
        free = data.pop("free")
        nav = NavMesh(free, data.pop("resolution"), tuple(data.pop("origin")),
                      data.pop("height"), meta=data)

        # Episode starts were sampled on the navmesh, so each must fall on a
        # free cell; otherwise the origin or axis order is wrong.
        off = []
        for episode in manifest["episodes"]:
            x, _y, z, _yaw = episode["start_hab"][:4]
            row, col = nav.to_cell(x, z)
            if not (nav.inside(row, col) and nav.free[row, col]):
                off.append(episode["id"])
        if off:
            print(f"!! {alias}: {len(off)} of {len(manifest['episodes'])} episode "
                  f"starts do not land on navigable cells: {off[:5]}", file=sys.stderr)
            print("   The raster's origin or axis order is wrong; not writing it.",
                  file=sys.stderr)
            return 1

        nav.save(dest)
        print(f"  {'':14} -> {dest}  {free.shape[1]}x{free.shape[0]} cells, "
              f"{nav.area_m2:.1f} m2 navigable, "
              f"all {len(manifest['episodes'])} starts on the mesh")
        # The raster and the manifest area are measured differently; flag large gaps.
        recorded = manifest.get("navigable_area_m2")
        if recorded:
            ratio = nav.area_m2 / float(recorded)
            flag = "  <-- check this" if not 0.5 <= ratio <= 2.0 else ""
            print(f"  {'':14}    manifest records {float(recorded):.1f} m2, "
                  f"ratio {ratio:.2f}{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
