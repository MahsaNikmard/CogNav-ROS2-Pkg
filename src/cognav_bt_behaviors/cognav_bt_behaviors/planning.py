"""Planner leaf: grow an RRT in visible free space and commit to a branch.

The tree is grown in the robot frame from the current snapshot. Each node is
scored by novelty, its distance to a decimated trail of past odometry poses
capped at `novelty_radius`, and the branch to the best node is committed as
`reference_path` in the odom frame. Points outside the field of view are
unknown and refused by the free-space oracle, so the tree stays in view.
"""

import math
from collections import deque

from cognav_bt_behaviors.base import BaseBehavior, BehaviorContext, Role, register
from cognav_representation.geometry import (
    bin_for_bearing,
    drivable_bins,
    to_odom,
    to_robot,
)
from cognav_representation.navmesh import NavMesh
from cognav_representation.oracle import MeshOracle, PolarOracle

from .status import Status


@register("RRTExploreWaypoint", role=Role.PLANNER)
class RRTExploreWaypoint(BaseBehavior):
    """Grow an RRT in visible free space and follow the branch into new ground.

    Writes the whole branch as `reference_path`, nearest waypoint first, or an
    empty path when the tree cannot grow. Always returns SUCCESS so the rest of
    the Navigate sequence runs; `HasWaypoint` turns an empty path into failure.
    """

    WRITES = frozenset({"reference_path", "rejected_bins"})
    #: Cleared on the first attempt of each tick; on a Retry attempt it holds
    #: the bins the safety gate rejected earlier in the same tick.
    READS_CARRIED = frozenset({"rejected_bins"})

    def __init__(self, name=None, ports=None):
        super().__init__(name=name, ports=ports)
        self._path: list[tuple[float, float]] = []
        self._ticks_since_replan = 0
        self._last_tick_id = -1
        self._visited: deque[tuple[float, float]] = deque(maxlen=512)
        self._oracle_cache = None
        self._oracle_tick = -1
        self._navmesh_cache = None

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        if snapshot is None or snapshot.scan_ranges is None or snapshot.odom is None:
            return Status.FAILURE

        tick_id = getattr(context, "tick_id", -1)
        first_attempt = tick_id != self._last_tick_id
        if first_attempt:
            self._last_tick_id = tick_id
            blackboard.set("rejected_bins", [])
        rejected = set(blackboard.get("rejected_bins", []))

        odom = snapshot.odom
        if first_attempt:
            self._record_visit(odom, context)
            self._ticks_since_replan += 1

        self._path = self._advance(self._path, odom, snapshot, context, tick_id,
                                   blackboard)

        # Keep the commitment unless it expired or safety rejected part of it.
        if (self._path and not rejected
                and self._ticks_since_replan < self.setting(context, "replan_period_ticks")):
            blackboard.set("reference_path", list(self._path))
            return Status.SUCCESS

        self._path = self._grow(snapshot, odom, context, rejected, tick_id,
                                blackboard)
        self._ticks_since_replan = 0
        blackboard.set("reference_path", list(self._path))
        return Status.SUCCESS

    def _record_visit(self, odom, context) -> None:
        """Append the pose if it is at least `visit_spacing` from the last one."""
        xy = (odom[0], odom[1])
        spacing = self.setting(context, "visit_spacing")
        if not self._visited or math.hypot(xy[0] - self._visited[-1][0],
                                           xy[1] - self._visited[-1][1]) >= spacing:
            self._visited.append(xy)

    def _oracle(self, snapshot, odom, context, tick_id, blackboard=None):
        """The free-space oracle for this tick, built once and reused by retries."""
        if self._oracle_cache is not None and self._oracle_tick == tick_id:
            return self._oracle_cache
        bin_range = getattr(snapshot, "real_bin_range", None)
        source = self.setting(context, "free_space_source")
        if source == "polar":
            oracle = PolarOracle(snapshot.scan_ranges, snapshot.scan_angles, bin_range)
        elif source == "mesh":
            oracle = MeshOracle(self._navmesh(blackboard), odom,
                                self._start_hab(blackboard),
                                snapshot.scan_angles, bin_range,
                                max_range=self.setting(context, "mesh_max_range"))
        else:
            raise ValueError(
                f"free_space_source must be 'polar' or 'mesh', got {source!r}")
        self._oracle_cache = oracle
        self._oracle_tick = tick_id
        return oracle

    def _navmesh(self, blackboard):
        """The scene's rasterised navmesh, loaded once per mission."""
        if self._navmesh_cache is not None:
            return self._navmesh_cache
        path = blackboard.get("mesh_path", "") if blackboard is not None else ""
        if not path:
            raise ValueError(
                "free_space_source='mesh' needs mesh_path; export the raster "
                "with scripts/export_navmesh.py")
        self._navmesh_cache = NavMesh.load(path)
        return self._navmesh_cache

    @staticmethod
    def _start_hab(blackboard):
        """The Habitat pose (x, y, z, yaw) at which odometry was zeroed."""
        raw = blackboard.get("start_hab", "") if blackboard is not None else ""
        if isinstance(raw, (list, tuple)):
            values = [float(v) for v in raw]
        else:
            values = [float(v) for v in str(raw).replace(",", " ").split()]
        if len(values) < 4:
            raise ValueError(
                "free_space_source='mesh' needs start_hab as x,y,z,yaw in "
                f"Habitat coordinates; got {raw!r}")
        return values[:4]

    def _novelty(self, xy, cap: float) -> float:
        """Distance to the nearest visited pose, capped at `cap`."""
        best = cap
        for vx, vy in self._visited:
            d = math.hypot(xy[0] - vx, xy[1] - vy)
            if d < best:
                best = d
                if best <= 0.0:
                    break
        return best

    def _advance(self, path, odom, snapshot, context, tick_id, blackboard=None):
        """Drop reached waypoints; abandon the path if its next leg is blocked."""
        if not path:
            return []
        tol = self.setting(context, "waypoint_reached_tol")
        while path and math.hypot(path[0][0] - odom[0], path[0][1] - odom[1]) <= tol:
            path.pop(0)
        if not path:
            return []
        radius = self.setting(context, "robot_radius") + self.setting(context, "maneuver_clearance")
        nxt = to_robot(odom, *path[0])
        oracle = self._oracle(snapshot, odom, context, tick_id, blackboard)
        if not oracle.segment_is_free((0.0, 0.0), nxt, radius):
            return []
        return path

    def _grow(self, snapshot, odom, context, rejected, tick_id, blackboard=None):
        """Build the tree in the robot frame and return the best branch in odom.

        Nodes are `(x, y, parent_index)` with the robot at the root.
        """
        ranges, angles = snapshot.scan_ranges, snapshot.scan_angles
        bin_range = getattr(snapshot, "real_bin_range", None)
        radius = self.setting(context, "robot_radius") + self.setting(context, "maneuver_clearance")
        step = self.setting(context, "rrt_step")
        reach = self.setting(context, "rrt_sample_radius")
        cap = self.setting(context, "novelty_radius")
        rng = self.rng(context)
        oracle = self._oracle(snapshot, odom, context, tick_id, blackboard)

        # Sampling is confined to the bearings the driver would accept.
        usable = drivable_bins(ranges, angles,
                               self.setting(context, "safety_margin")
                               + self.setting(context, "warning_margin"),
                               radius, bin_range, rejected)
        if not usable:
            return []
        lo_a, hi_a = min(angles[i] for i in usable), max(angles[i] for i in usable)

        nodes = [(0.0, 0.0, -1)]
        best_i, best_score = -1, -float("inf")

        for _ in range(int(self.setting(context, "rrt_iterations"))):
            if rng.random() < self.setting(context, "rrt_bias"):
                far = max(usable, key=lambda i: ranges[i] if math.isfinite(ranges[i]) else reach)
                theta = angles[far]
            else:
                theta = rng.uniform(lo_a, hi_a)
            rho = reach * math.sqrt(rng.random())
            sx, sy = rho * math.cos(theta), rho * math.sin(theta)

            ni = min(range(len(nodes)),
                     key=lambda k: math.hypot(sx - nodes[k][0], sy - nodes[k][1]))
            nx, ny, _ = nodes[ni]
            d = math.hypot(sx - nx, sy - ny)
            if d < 1e-6:
                continue
            px, py = nx + step * (sx - nx) / d, ny + step * (sy - ny) / d

            if not oracle.point_is_free(px, py, radius):
                continue
            # Out of view, bin_for_bearing returns an edge bin, so the rejected
            # set is consulted only for bearings the vector covers.
            bearing = math.atan2(py, px)
            if rejected and abs(bearing) <= abs(angles[0]) \
                    and bin_for_bearing(angles, bearing) in rejected:
                continue
            if not oracle.segment_is_free((nx, ny), (px, py), radius):
                continue

            nodes.append((px, py, ni))
            score = self._novelty(to_odom(odom, px, py), cap)
            if score > best_score:
                best_i, best_score = len(nodes) - 1, score

        if best_i < 0:
            return []
        return [to_odom(odom, x, y) for x, y in self._branch(nodes, best_i)]

    @staticmethod
    def _branch(nodes, index):
        """Root-to-node path, root excluded, nearest waypoint first."""
        out = []
        while index > 0:
            x, y, parent = nodes[index]
            out.append((x, y))
            index = parent
        out.reverse()
        return out
