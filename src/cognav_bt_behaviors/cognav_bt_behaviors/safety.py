"""Safety leaves: verify the proposed command, brake, and recover."""

from cognav_bt_behaviors.base import BaseBehavior, BehaviorContext, Role, register
from cognav_representation.geometry import swept_arc_check

from .status import Status


@register("Brake", role=Role.SAFETY)
class Brake(BaseBehavior):
    """Publish a zero command."""

    WRITES = frozenset({"cmd_vel"})

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        context.publish_twist(0.0, 0.0, blackboard)
        return Status.SUCCESS


@register("Recover", role=Role.SAFETY)
class Recover(BaseBehavior):
    """Turn away from the nearest blocked bearing, creeping forward if allowed."""

    #: Navigate can fail before IsCommandSafe runs, so the bins may be from an
    #: earlier tick. Turning in place is safe whatever they say.
    READS_CARRIED = frozenset({"blocked_bins"})
    WRITES = frozenset({"cmd_vel"})

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        blocked = blackboard.get("blocked_bins", [])
        angular = self.setting(context, "recovery_angular_speed")
        if not blocked or snapshot is None:
            context.publish_twist(0.0, angular, blackboard)
            return Status.SUCCESS
        nearest_idx, _ = blocked[0]
        turn_direction = -1.0 if snapshot.scan_angles[nearest_idx] >= 0.0 else 1.0
        context.publish_twist(
            self.setting(context, "recovery_linear_speed"),
            turn_direction * angular,
            blackboard,
        )
        return Status.SUCCESS


@register("IsCommandSafe", role=Role.SAFETY)
class IsCommandSafe(BaseBehavior):
    """Gate: SUCCESS when the proposed command sweeps clear space.

    Simulates the arc the body traces under the proposed (v, w) for
    `arc_horizon_sec` and fails if any observed bin lies inside the swept
    footprint, or if the frame is stale. Turning in place always passes. On
    failure the swept bins are appended to `rejected_bins` so that a Retry
    attempt proposes a different branch.
    """

    #: The planner clears rejected_bins earlier in the same tick, so this read
    #: is always current and the load-time check enforces that ordering.
    READS = frozenset({"cmd_proposal", "rejected_bins"})
    WRITES = frozenset({"blocked_bins", "brake_now", "clearance", "rejected_bins"})

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        blackboard.set("blocked_bins", [])
        blackboard.set("clearance", float("inf"))

        if snapshot is None or snapshot.scan_ranges is None:
            blackboard.set("brake_now", False)
            return Status.FAILURE
        if not snapshot.perception_fresh:
            blackboard.set("brake_now", True)
            return Status.FAILURE

        v, w = blackboard.get("cmd_proposal", (0.0, 0.0))
        swept = swept_arc_check(
            snapshot.scan_ranges,
            snapshot.scan_angles,
            float(v), float(w),
            self.setting(context, "robot_radius") + self.setting(context, "maneuver_clearance"),
            self.setting(context, "arc_horizon_sec"),
            getattr(snapshot, "real_bin_range", None),
        )
        blackboard.set("blocked_bins", swept)
        blackboard.set("clearance", swept[0][1] if swept else float("inf"))

        # Every swept bin is a predicted contact, whatever its range.
        risk = bool(swept)
        blackboard.set("brake_now", risk)
        if risk:
            rejected = list(blackboard.get("rejected_bins", []))
            rejected.extend(idx for idx, _ in swept)
            blackboard.set("rejected_bins", rejected)
        return Status.FAILURE if risk else Status.SUCCESS
