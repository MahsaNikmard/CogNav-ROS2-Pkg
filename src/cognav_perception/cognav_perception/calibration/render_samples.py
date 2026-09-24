#!/usr/bin/env python3
"""Render paired RGB and ground-truth depth from a Habitat scene.

    conda run -n habitat python -m cognav_perception.calibration.render_samples \
        --scene <path.glb> --out <sample dir> --samples 24

Poses are navigable points of the robot-sized navmesh with a random yaw.
"""
import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--width", type=int, default=250)
    ap.add_argument("--height", type=int, default=250)
    ap.add_argument("--hfov-deg", type=float, default=89.0)
    ap.add_argument("--sensor-height", type=float, default=0.32)
    ap.add_argument("--agent-radius", type=float, default=0.18)
    ap.add_argument("--agent-height", type=float, default=0.40)
    ap.add_argument("--agent-max-climb", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = args.scene
    backend.enable_physics = False

    def sensor(uuid, sensor_type):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [args.height, args.width]
        spec.hfov = args.hfov_deg
        spec.position = [0.0, args.sensor_height, 0.0]
        return spec

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.radius = args.agent_radius
    agent_cfg.height = args.agent_height
    agent_cfg.sensor_specifications = [
        sensor("color", habitat_sim.SensorType.COLOR),
        sensor("depth", habitat_sim.SensorType.DEPTH),
    ]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))

    # Rebuild the navmesh for this robot, as habitat_sim_server does. A climb
    # limit below one stair riser keeps storeys separate.
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_max_climb = args.agent_max_climb
    navmesh_settings.agent_radius = args.agent_radius
    navmesh_settings.agent_height = args.agent_height
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    if not sim.pathfinder.is_loaded:
        raise SystemExit(f"no navmesh could be built for {args.scene}")
    sim.pathfinder.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out, exist_ok=True)
    agent = sim.get_agent(0)
    written = 0
    for _ in range(args.samples):
        state = agent.get_state()
        point = sim.pathfinder.get_random_navigable_point()
        if not np.all(np.isfinite(point)):
            continue
        state.position = point
        yaw = float(rng.uniform(-np.pi, np.pi))
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
            yaw, np.array([0.0, 1.0, 0.0])
        )
        state.sensor_states = {}
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        depth = np.asarray(obs["depth"], dtype=np.float32)
        # Habitat returns 0 where the ray left the scene; there is no ground truth.
        depth[depth <= 0.0] = np.nan
        np.save(os.path.join(args.out, f"rgb_{written:03d}.npy"),
                np.ascontiguousarray(obs["color"][..., :3]))
        np.save(os.path.join(args.out, f"depth_{written:03d}.npy"), depth)
        written += 1

    # Pinhole intrinsics of the rendering camera.
    fx = (args.width / 2.0) / np.tan(np.radians(args.hfov_deg) / 2.0)
    manifest = {
        "scene": args.scene,
        "samples": written,
        "width": args.width,
        "height": args.height,
        "hfov_deg": args.hfov_deg,
        "sensor_height": args.sensor_height,
        "intrinsics": {"fx": float(fx), "fy": float(fx),
                       "cx": (args.width - 1) / 2.0, "cy": (args.height - 1) / 2.0},
        "seed": args.seed,
    }
    if not written:
        raise SystemExit(
            f"rendered nothing from {args.scene}: the navmesh produced no navigable points"
        )
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {written} samples to {args.out}")
    sim.close()


if __name__ == "__main__":
    main()
