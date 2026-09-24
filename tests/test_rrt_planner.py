"""The RRT planner: agreement with the driver, outward bias, and repeatability.

The planner samples from `drivable_bins` and the driver steers through
`find_gaps`, so the planner must never commit where the driver cannot go.
Branches are scored by distance from the visited trail, so new ground must win.
"""
import math

import numpy as np
import rclpy

from cognav_bt_behaviors.base import BehaviorConfig, BehaviorContext
from cognav_bt_behaviors.blackboard import Blackboard
from cognav_representation.geometry import (
    drivable_bins, find_gaps, point_is_free, segment_is_free, to_robot,
)
from cognav_bt_behaviors.planning import RRTExploreWaypoint
from cognav_bt_behaviors.status import Status

from _check import Checker
from _harness import ANGLES, N_BINS, PADDING, REAL_BINS, scene, snapshot

RADIUS, MARGIN, BAND, CLEAR = 0.18, 0.50, 0.40, 0.10
HALF = RADIUS + CLEAR


def main():
    c = Checker()
    rclpy.init()
    node = rclpy.create_node("rrt_test")
    cfg = BehaviorConfig(robot_radius=RADIUS, safety_margin=MARGIN,
                         warning_margin=BAND, maneuver_clearance=CLEAR,
                         max_v=0.5)

    def ctx(tick=1):
        c_ = BehaviorContext(node, None, cfg, enable_cmd_vel=False)
        c_.tick_id = tick
        return c_

    def plan(ranges, odom=(0.0, 0.0, 0.0), seed=1, ticks=1):
        p = RRTExploreWaypoint(ports={"seed": seed})
        bb = Blackboard()
        for t in range(1, ticks + 1):
            p.tick(snapshot(ranges, odom=odom), bb, ctx(t))
        return p, bb.get("reference_path", [])

    print("\n[1] free-space tests")
    open_scene = scene(3.0)
    c("a point inside the robot's own footprint is free",
      point_is_free(open_scene, ANGLES, 0.05, 0.0, HALF, REAL_BINS))
    c("a point in open space is free",
      point_is_free(open_scene, ANGLES, 2.0, 0.0, HALF, REAL_BINS))
    c("a point behind a wall is not",
      not point_is_free(scene(1.0), ANGLES, 2.0, 0.0, HALF, REAL_BINS))
    c("a point outside the FOV is not free (never observed)",
      not point_is_free(open_scene, ANGLES, 0.0, 2.0, HALF, REAL_BINS))
    c("a clear segment passes",
      segment_is_free(open_scene, ANGLES, (0, 0), (2.0, 0.0), HALF, REAL_BINS))
    c("a segment through a wall does not",
      not segment_is_free(scene(1.0), ANGLES, (0, 0), (2.0, 0.0), HALF, REAL_BINS))

    print("\n[2] planner and driver share one feasibility test")
    bar = "." * 12 + ":" * 5 + "+" * 16 + "#" * 4 + "+" + "#" * 34   # clutter
    band = {".": 2.5, ":": 1.5, "+": 0.70, "#": 0.35}
    t140 = np.full(N_BINS + 2 * PADDING, 1.5)
    t140[PADDING:PADDING + N_BINS] = [band[ch] for ch in bar]

    for label, ranges in [("clutter", t140), ("open", scene(3.0)),
                          ("walled", scene(0.32)), ("corridor", scene(1.2))]:
        gaps = find_gaps(ranges, ANGLES, MARGIN + BAND, HALF, REAL_BINS)
        bins = drivable_bins(ranges, ANGLES, MARGIN + BAND, HALF, REAL_BINS)
        _, path = plan(ranges)
        driver_can_go = bool(gaps)
        planner_went = bool(path)
        c(f"{label:16} driver={len(gaps)} gaps, planner={len(path)} waypoints",
          planner_went <= driver_can_go)
        c(f"{label:16} drivable_bins agrees with find_gaps",
          bool(bins) == bool(gaps))

    print("\n[3] every emitted waypoint is drivable")
    _, path = plan(scene(3.0))
    c("a path was produced", bool(path))
    prev = (0.0, 0.0)
    ok = True
    for wp in path:
        here = to_robot((0.0, 0.0, 0.0), *wp)
        if not segment_is_free(scene(3.0), ANGLES, prev, here, HALF, REAL_BINS):
            ok = False
        prev = here
    c("every leg of the emitted path is collision-free", ok)
    c("the path is a branch, not a single waypoint", len(path) > 1,
      f"{len(path)} waypoints")

    print("\n[4] nowhere to go produces an empty path")
    p, path = plan(scene(0.32))
    c("walled in: empty reference_path", path == [], str(path[:2]))
    st = p.tick(snapshot(scene(0.32)), Blackboard(), ctx(9))
    c("still returns SUCCESS, so the pipeline is not cut short",
      st == Status.SUCCESS)

    print("\n[5] growth is biased away from visited ground")
    p = RRTExploreWaypoint(ports={"seed": 3})
    # Pretend the robot already covered everything to its left.
    for k in range(12):
        p._visited.append((0.5 + 0.1 * k, 1.0))
    bb = Blackboard()
    p.tick(snapshot(scene(3.0)), bb, ctx(1))
    path = bb.get("reference_path", [])
    c("a path was produced", bool(path))
    if path:
        end = path[-1]
        near_visited = min(math.hypot(end[0] - vx, end[1] - vy)
                           for vx, vy in p._visited)
        c(f"the branch ends {near_visited:.2f} m from the covered strip",
          near_visited > 0.5)

    print("\n[6] novelty saturates at novelty_radius")
    q = RRTExploreWaypoint()
    q._visited.append((0.0, 0.0))
    cap = 2.0
    c("far away scores the cap, not the raw distance",
      abs(q._novelty((50.0, 0.0), cap) - cap) < 1e-9)
    c("close by scores the true distance",
      abs(q._novelty((0.5, 0.0), cap) - 0.5) < 1e-9)

    print("\n[7] a seeded run repeats")
    _, a = plan(scene(3.0), seed=7)
    _, b = plan(scene(3.0), seed=7)
    _, d = plan(scene(3.0), seed=8)
    c("same seed, same path", a == b)
    c("different seed, different path", a != d)

    print("\n[8] ports are cast to the type of their config field")
    from cognav_bt_behaviors.base import build_behavior

    leaf = build_behavior("RRTExploreWaypoint", None,
                          {"free_space_source": "mesh",
                           "rrt_iterations": "40", "seed": "3"})
    c("str port stays a string", leaf.ports["free_space_source"] == "mesh",
      repr(leaf.ports["free_space_source"]))
    c("int port is cast", leaf.ports["rrt_iterations"] == 40
      and isinstance(leaf.ports["rrt_iterations"], int))
    try:
        build_behavior("RRTExploreWaypoint", None, {"no_such_port": "1"})
        c("an unknown port is rejected", False)
    except ValueError:
        c("an unknown port is rejected", True)

    c.finish("RRT exploration planner")


if __name__ == "__main__":
    main()
