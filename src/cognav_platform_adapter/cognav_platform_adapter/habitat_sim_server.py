#!/usr/bin/env python3
"""Habitat-sim side of the Habitat bridge. No ROS in this process.

Runs in the `habitat` conda environment (Python 3.9; habitat-sim has no 3.10
build) and answers habitat_bridge over ZMQ REQ/REP, multipart [json, rgb]:

    {"cmd": "info"}                     -> {"width", "height", "hfov_deg",
                                            "sensor_height", "scene",
                                            "start_hab", "navigable_area_m2"}
    {"cmd": "step", "v", "w", "dt"}     -> {"pose": [x, y, yaw], "collided",
                                            "true_clearance_m"} + rgb bytes

Motion is a kinematic unicycle (VelocityControl) filtered through the navmesh
with `pathfinder.try_step`; a step collides when the filter removed motion.
The reported pose is in a z-up frame anchored at the start pose, x forward and
y left, so it behaves like odometry that starts at the identity.

Normally started by platform_adapter_launch.py:

    conda run -n habitat python habitat_sim_server.py --scene <path.glb>
"""

import argparse
import json
import math
import os
import sys

import numpy as np

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat_sim  # noqa: E402
import magnum as mn  # noqa: E402
from habitat_sim.physics import VelocityControl  # noqa: E402
from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum  # noqa: E402

import zmq  # noqa: E402


def _candidate_scene_paths(scene_path: str) -> list[str]:
    """Equivalent locations of a test scene under both Habitat dataset layouts."""
    candidates = [scene_path]
    translated = scene_path.replace(
        "/scene_datasets/habitat-test-scenes/",
        "/versioned_data/habitat_test_scenes/",
    )
    if translated not in candidates:
        candidates.append(translated)
    translated = scene_path.replace(
        "/versioned_data/habitat_test_scenes/",
        "/scene_datasets/habitat-test-scenes/",
    )
    if translated not in candidates:
        candidates.append(translated)

    basename = os.path.basename(scene_path)
    data_root = None
    marker = "/simulation/habitat_data/"
    if marker in scene_path:
        prefix = scene_path.split(marker, 1)[0]
        data_root = os.path.join(prefix, marker.strip("/"))

    if data_root:
        for candidate in (
            os.path.join(data_root, "versioned_data", "habitat_test_scenes", basename),
            os.path.join(data_root, "scene_datasets", "habitat-test-scenes", basename),
        ):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _resolve_scene_path(scene_path: str) -> str:
    for candidate in _candidate_scene_paths(scene_path):
        if os.path.exists(candidate):
            return candidate
    return scene_path


def _make_sim(args) -> habitat_sim.Simulator:
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = args.scene
    backend.enable_physics = False
    backend.load_semantic_mesh = False
    backend.use_semantic_textures = False
    if args.gpu_id >= 0:
        backend.gpu_device_id = args.gpu_id

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "color"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [args.height, args.width]
    rgb.position = [0.0, args.sensor_height, 0.0]
    rgb.hfov = args.hfov_deg

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    agent_cfg.radius = args.agent_radius
    agent_cfg.height = args.agent_height

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))

    # The navmesh is the collision model, so it is rebuilt for this robot's
    # radius. A climb limit below one stair riser keeps storeys separate.
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_max_climb = args.agent_max_climb
    navmesh_settings.agent_radius = args.agent_radius
    navmesh_settings.agent_height = args.agent_height
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    return sim


class HabitatServer:
    def __init__(self, args) -> None:
        self.args = args
        self.sim = _make_sim(args)
        self.agent = self.sim.get_agent(0)

        # Start at --start-hab, or at a navigable point drawn from the seed.
        state = self.agent.get_state()
        if args.start_hab:
            vals = [float(v) for v in args.start_hab.split(",")]
            state.position = np.array(vals[:3], dtype=np.float32)
            yaw_h = vals[3] if len(vals) > 3 else 0.0
            state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
                yaw_h, np.array([0.0, 1.0, 0.0])
            )
        else:
            self.sim.pathfinder.seed(args.seed)
            state.position = self.sim.pathfinder.get_random_navigable_point()
        state.sensor_states = {}
        self.agent.set_state(state)

        self._start_pos = np.array(state.position, dtype=np.float64)
        self._start_yaw = self._hab_yaw(state.rotation)

        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        print(f"[habitat_server] scene         : {args.scene}", flush=True)
        print(f"[habitat_server] camera        : {args.width}x{args.height}  "
              f"hfov={args.hfov_deg} deg  h={args.sensor_height} m", flush=True)
        print(f"[habitat_server] agent         : r={args.agent_radius} m  "
              f"h={args.agent_height} m", flush=True)
        print(f"[habitat_server] start (hab)   : pos={np.round(self._start_pos, 3)}  "
              f"yaw={math.degrees(self._start_yaw):.1f} deg", flush=True)

    @staticmethod
    def _hab_yaw(rotation) -> float:
        """Yaw (rad) about habitat +Y from an agent rotation quaternion."""
        q = quat_to_magnum(rotation) if not isinstance(rotation, mn.Quaternion) else rotation
        # Rotate habitat-forward (-Z) and read the heading in the XZ plane.
        fwd = q.transform_vector(mn.Vector3(0.0, 0.0, -1.0))
        return math.atan2(-fwd.x, -fwd.z)

    def _pose_ros(self) -> tuple:
        """Agent pose in the z-up frame anchored at the start pose."""
        st = self.agent.get_state()
        pos = np.array(st.position, dtype=np.float64)
        yaw_h = self._hab_yaw(st.rotation)
        # Habitat displacement into a z-up world frame (x = -z_h, y = -x_h),
        # then rotated into the start frame.
        dx_h = pos[0] - self._start_pos[0]
        dz_h = pos[2] - self._start_pos[2]
        x_w = -dz_h
        y_w = -dx_h
        c, s = math.cos(-self._start_yaw), math.sin(-self._start_yaw)
        x = c * x_w - s * y_w
        y = s * x_w + c * y_w
        yaw = (yaw_h - self._start_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return (x, y, yaw)

    def step(self, v: float, w: float, dt: float) -> tuple:
        """Integrate (v, w) for dt, slide along the navmesh, report collision."""
        self.vel_control.linear_velocity = mn.Vector3(0.0, 0.0, -float(v))
        self.vel_control.angular_velocity = mn.Vector3(0.0, float(w), 0.0)

        state = self.agent.get_state()
        prev = habitat_sim.RigidState(
            quat_to_magnum(state.rotation), state.position
        )
        target = self.vel_control.integrate_transform(float(dt), prev)
        end_pos = self.sim.pathfinder.try_step(  # navmesh sliding filter
            prev.translation, target.translation
        )
        state.position = end_pos
        state.rotation = quat_from_magnum(target.rotation)
        state.sensor_states = {}
        self.agent.set_state(state)

        wanted = np.array(target.translation) - np.array(prev.translation)
        got = np.array(end_pos) - np.array(prev.translation)
        collided = bool(
            np.linalg.norm(got) + 1e-5 < np.linalg.norm(wanted)
        )

        obs = self.sim.get_sensor_observations()
        rgb = np.ascontiguousarray(obs["color"][..., :3])
        return self._pose_ros(), collided, rgb, self._true_clearance(end_pos)

    def _true_clearance(self, position) -> float:
        """Distance to the nearest navmesh boundary, i.e. from the body surface."""
        try:
            return float(self.sim.pathfinder.distance_to_closest_obstacle(
                np.asarray(position, dtype=np.float32)))
        except Exception:
            return float("nan")

    def _floor_area_m2(self) -> float:
        """Area of the navmesh island the robot starts on: the coverage denominator."""
        pf = self.sim.pathfinder
        island = pf.get_island(np.asarray(self._start_pos, dtype=np.float32))
        if island < 0:
            print("[habitat_server] start pose is off the navmesh; using the "
                  "whole navmesh area", flush=True)
            return float(pf.navigable_area)
        return float(pf.island_area(island_index=island))

    def serve(self) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(self.args.endpoint)
        print(f"[habitat_server] listening on {self.args.endpoint}", flush=True)
        try:
            while True:
                parts = sock.recv_multipart()
                req = json.loads(parts[0])
                cmd = req.get("cmd")
                if cmd == "info":
                    meta = {
                        "width": self.args.width,
                        "height": self.args.height,
                        "hfov_deg": self.args.hfov_deg,
                        "sensor_height": self.args.sensor_height,
                        "scene": self.args.scene,
                        "start_hab": list(map(float, self._start_pos))
                                     + [self._start_yaw],
                        "navigable_area_m2": self._floor_area_m2(),
                    }
                    sock.send_multipart([json.dumps(meta).encode(), b""])
                elif cmd == "step":
                    v = float(req.get("v", 0.0))
                    w = float(req.get("w", 0.0))
                    dt = float(req.get("dt", 1.0 / 15.0))
                    pose, collided, rgb, clearance = self.step(v, w, dt)
                    meta = {"pose": list(pose), "collided": collided,
                            "true_clearance_m": clearance}
                    sock.send_multipart([json.dumps(meta).encode(), rgb.tobytes()])
                else:
                    sock.send_multipart(
                        [json.dumps({"error": f"unknown cmd {cmd!r}"}).encode(), b""]
                    )
        except KeyboardInterrupt:
            print("[habitat_server] interrupted; shutting down", flush=True)
        finally:
            sock.close(0)
            ctx.term()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, help="Path to the scene .glb")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5561",
                   help="ZMQ REP endpoint the bridge connects to")
    p.add_argument("--width", type=int, default=250)
    p.add_argument("--height", type=int, default=250)
    p.add_argument("--hfov-deg", type=float, default=89.0,
                   help="Horizontal FOV; must match perception fov_deg")
    p.add_argument("--sensor-height", type=float, default=0.32,
                   help="Camera height above the floor")
    p.add_argument("--agent-radius", type=float, default=0.18,
                   help="Navmesh agent radius; the robot radius")
    p.add_argument("--agent-max-climb", type=float, default=0.05,
                   help="Navmesh step allowance; 0.05 m passes door sills but not stairs")
    p.add_argument("--agent-height", type=float, default=0.40)
    p.add_argument("--start-hab", default="",
                   help="Start pose 'x,y,z[,yaw]' in Habitat coordinates; empty "
                        "draws a navigable point from --seed")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gpu-id", type=int, default=-1)
    args = p.parse_args()
    args.scene = _resolve_scene_path(args.scene)

    if not os.path.exists(args.scene):
        print(f"[habitat_server] scene not found: {args.scene}", file=sys.stderr)
        sys.exit(2)

    HabitatServer(args).serve()


if __name__ == "__main__":
    main()
