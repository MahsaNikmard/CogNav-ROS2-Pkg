"""Driver leaves: turn the reference path into a velocity command."""

import math

from cognav_bt_behaviors.base import BaseBehavior, BehaviorContext, Role, register
from cognav_representation.geometry import (
    bearing_to_point,
    find_gaps,
    steer_through_gaps,
)

from .status import Status


@register("FollowTheGap", role=Role.DRIVER)
class FollowTheGap(BaseBehavior):
    """Follow-the-gap steering in which the next waypoint chooses the gap.

    Obstacles are inflated by the body half-width and the drivable gaps found
    with `find_gaps`, the same test the planner samples from. If the waypoint
    bearing lies inside a gap the robot steers straight at it; otherwise it
    steers at the edge of the gap that needs the least deviation. Forward speed
    falls with the depth of the chosen gap and with the size of the turn.

    Writes `cmd_proposal`; `ExecuteCommand` publishes it once safety passes it.
    """

    READS = frozenset({"reference_path"})
    WRITES = frozenset({"cmd_proposal"})

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        reference_path = blackboard.get("reference_path", [])
        if snapshot is None or snapshot.odom is None or not reference_path:
            blackboard.set("cmd_proposal", (0.0, 0.0))
            return Status.SUCCESS

        x, y, yaw = snapshot.odom
        waypoint_bearing = bearing_to_point((x, y), yaw, reference_path[0])
        half_width = (self.setting(context, "robot_radius")
                      + self.setting(context, "maneuver_clearance"))
        margin = self.setting(context, "safety_margin")
        band = self.setting(context, "warning_margin")

        gaps = find_gaps(snapshot.scan_ranges, snapshot.scan_angles,
                         margin + band, half_width,
                         getattr(snapshot, "real_bin_range", None))
        steer, gap = steer_through_gaps(gaps, snapshot.scan_angles, waypoint_bearing)
        if steer is None:
            blackboard.set("cmd_proposal", (0.0, 0.0))
            return Status.SUCCESS

        depth = gap[2] if math.isfinite(gap[2]) else self.setting(context, "lookahead_dist")
        max_w = self.setting(context, "max_w")
        max_v = self.setting(context, "max_v")
        w = max(-max_w, min(max_w, self.setting(context, "k_w") * steer))
        # A gap is always deeper than margin + band, so speed is scaled against
        # the planning horizon rather than the safety band.
        horizon = max(self.setting(context, "lookahead_dist"), 1e-3)
        reach = max(0.0, min(1.0, (depth - margin) / horizon))
        v = max(0.0, max_v * math.cos(steer) * reach)
        blackboard.set("cmd_proposal", (v, w))
        return Status.SUCCESS


@register("ExecuteCommand", role=Role.DRIVER)
class ExecuteCommand(BaseBehavior):
    """Publish the proposal that the safety gate has just verified."""

    READS = frozenset({"cmd_proposal"})
    WRITES = frozenset({"cmd_vel"})

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        v, w = blackboard.get("cmd_proposal", (0.0, 0.0))
        context.publish_twist(float(v), float(w), blackboard)
        return Status.SUCCESS
