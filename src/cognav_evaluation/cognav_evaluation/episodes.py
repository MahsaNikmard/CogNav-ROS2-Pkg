"""Episode manifests: (scene, start pose, seed) triples replayed by every arm.

Start poses are navigable points at least `--margin` from the navmesh boundary
on one island, which is also the coverage denominator. Generation needs
habitat-sim; reading a manifest does not.

    conda run -n habitat python -m cognav_evaluation.episodes generate \
        --scene <glb> --alias hm3d_small --count 10 --out episodes/hm3d_small.json

    python -m cognav_evaluation.episodes launch-args episodes/hm3d_small.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

#: Where habitat_data is mounted in the container. Manifests store scene paths
#: relative to it, so they resolve on the host and in the container alike.
CONTAINER_HABITAT_DATA = ("/opt/ws/install/cognav_platform_adapter/share"
                          "/cognav_platform_adapter/simulation/habitat_data")
_ROOT_MARKER = "/habitat_data/"


def _relative_scene(scene: str):
    """Scene path relative to the habitat_data root, or None if it is outside."""
    idx = scene.find(_ROOT_MARKER)
    return scene[idx + len(_ROOT_MARKER):] if idx >= 0 else None


def generate(scene: str, alias: str, count: int, out: str, margin: float,
             agent_radius: float, agent_height: float, seed: int,
             min_separation: float = 1.0,
             depth_scale: float | None = None,
             agent_max_climb: float = 0.05,
             island: int | None = None) -> int:
    import habitat_sim
    import numpy as np

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.load_semantic_mesh = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.radius = agent_radius
    agent_cfg.height = agent_height
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))

    # Rebuild the navmesh for this robot; the climb limit keeps storeys apart.
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_max_climb = agent_max_climb
    settings.agent_radius = agent_radius
    settings.agent_height = agent_height
    sim.recompute_navmesh(sim.pathfinder, settings)
    if not sim.pathfinder.is_loaded:
        raise SystemExit(f"no navmesh could be built for {scene}")

    pf = sim.pathfinder
    pf.seed(seed)
    rng = np.random.default_rng(seed)

    # All episodes start on one island, the largest by default, whose area is
    # the coverage denominator.
    if island is None:
        island = max(range(pf.num_islands), key=pf.island_area) if pf.num_islands else -1
    if island >= pf.num_islands:
        raise SystemExit(f"island {island} does not exist; the scene has {pf.num_islands}")
    island_area = float(pf.island_area(island)) if island >= 0 else float(pf.navigable_area)

    episodes, tries = [], 0
    while len(episodes) < count and tries < count * 400:
        tries += 1
        p = (pf.get_random_navigable_point(island_index=island) if island >= 0
             else pf.get_random_navigable_point())
        if not np.all(np.isfinite(p)):
            continue
        # The navmesh boundary is already inset by agent_radius.
        if pf.distance_to_closest_obstacle(p, 2.0) < margin:
            continue
        if any(float(np.linalg.norm(np.array(e["start_hab"][:3]) - p)) < min_separation
               for e in episodes):
            continue
        episodes.append({
            "id": f"{alias}_{len(episodes):02d}",
            "start_hab": [float(p[0]), float(p[1]), float(p[2]),
                          float(rng.uniform(-math.pi, math.pi))],
            "seed": int(rng.integers(0, 10_000)),
            "clearance_m": float(pf.distance_to_closest_obstacle(p, 2.0)),
            "island": int(island),
        })

    if len(episodes) < count:
        raise SystemExit(
            f"only {len(episodes)} of {count} starts met margin {margin} m in "
            f"{tries} tries on island {island} ({island_area:.1f} m2). Lower "
            f"--margin or --min-separation; remember the margin is measured to "
            f"the navmesh boundary, already inset by agent_radius {agent_radius} m")

    manifest = {
        "scene_rel": _relative_scene(scene),
        "alias": alias,
        "agent_radius": agent_radius,
        "agent_height": agent_height,
        "agent_max_climb": agent_max_climb,
        "margin_m": margin,
        "navigable_area_m2": island_area,
        "navmesh_total_area_m2": float(pf.navigable_area),
        "island": int(island),
        "islands": int(pf.num_islands),
        "seed": seed,
        "depth_scale": depth_scale,
        "episodes": episodes,
    }
    if manifest["scene_rel"] is None:
        manifest["scene"] = scene
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(episodes)} episodes to {out}\n"
          f"  island {island} of {pf.num_islands}: {island_area:.1f} m2 "
          f"(navmesh total {pf.navigable_area:.1f} m2)\n"
          f"  margin {margin} m to the navmesh boundary, "
          f"~{margin + agent_radius:.2f} m from geometry")
    sim.close()
    return 0


def launch_args(path: str, depth_scale: float | None = None,
                scene_root: str = CONTAINER_HABITAT_DATA) -> int:
    """One line of `ros2 launch` arguments per episode, scene resolved under `scene_root`.

    The launch file looks depth_scale up by alias only, so the manifest's (or
    --depth-scale) value is emitted explicitly.
    """
    with open(path, "r", encoding="utf-8") as fh:
        man = json.load(fh)
    rel = man.get("scene_rel")
    if rel:
        scene = os.path.join(scene_root, rel)
    else:
        scene = man["scene"]
        print("# WARNING: the scene is outside habitat_data and may not resolve "
              "inside the container.")
    scale = depth_scale if depth_scale is not None else man.get("depth_scale")
    if scale is None:
        print("# WARNING: no depth_scale for this scene; the launch falls back "
              "to the platform value.")
        print("# Measure it first:  ./scripts/calibrate_depth_scale.sh <scene>")
    for ep in man["episodes"]:
        start = ",".join(f"{v:.4f}" for v in ep["start_hab"])
        extra = f" depth_scale:={scale}" if scale is not None else ""
        print(f"{ep['id']}\tscene:={scene} start_hab:={start} "
              f"seed:={ep['seed']}{extra}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--scene", required=True, help="absolute .glb path")
    g.add_argument("--alias", required=True, help="scene alias used by the launch file")
    g.add_argument("--count", type=int, default=10)
    g.add_argument("--out", required=True)
    g.add_argument("--min-separation", type=float, default=1.0,
                   help="minimum spacing between start poses, so the set spans the scene")
    g.add_argument("--margin", type=float, default=0.40,
                   help="minimum clearance of a start to the navmesh boundary, "
                        "which is already inset by agent_radius")
    g.add_argument("--agent-radius", type=float, default=0.18)
    g.add_argument("--agent-height", type=float, default=0.40)
    g.add_argument("--island", type=int, default=None,
                   help="navmesh island to place episodes on; default the largest")
    g.add_argument("--agent-max-climb", type=float, default=0.05,
                   help="navmesh step allowance; 0.05 m keeps storeys apart")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--depth-scale", type=float, default=None,
                   help="measured depth_scale for this scene, stored in the manifest")

    l = sub.add_parser("launch-args")
    l.add_argument("manifest")
    l.add_argument("--depth-scale", type=float, default=None)
    l.add_argument("--scene-root", default=CONTAINER_HABITAT_DATA,
                   help="habitat_data root; defaults to the container mount")

    args = ap.parse_args()
    if args.cmd == "generate":
        return generate(scene=args.scene, alias=args.alias, count=args.count,
                        out=args.out, margin=args.margin,
                        agent_radius=args.agent_radius,
                        agent_height=args.agent_height, seed=args.seed,
                        min_separation=args.min_separation,
                        depth_scale=args.depth_scale,
                        agent_max_climb=args.agent_max_climb,
                        island=args.island)
    return launch_args(args.manifest, args.depth_scale, args.scene_root)


if __name__ == "__main__":
    raise SystemExit(main())
