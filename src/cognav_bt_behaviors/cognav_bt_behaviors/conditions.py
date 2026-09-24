"""Condition leaves: decide, and write nothing but their own flags."""

import time

from cognav_bt_behaviors.base import BaseBehavior, BehaviorContext, Role, register

from .status import Status


@register("HasWaypoint", role=Role.CONDITION)
class HasWaypoint(BaseBehavior):
    """SUCCESS when the planner committed a non-empty reference path."""

    READS = frozenset({"reference_path"})

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        return Status.SUCCESS if blackboard.get("reference_path", []) else Status.FAILURE


@register("HasMotion", role=Role.CONDITION)
class HasMotion(BaseBehavior):
    """SUCCESS when the proposed command moves the robot; rotation counts.

    Without it, a zero proposal passes the safety gate and Navigate succeeds
    while the robot stands still, so the Escape rung is never reached.
    """

    READS = frozenset({"cmd_proposal"})

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        v, w = blackboard.get("cmd_proposal", (0.0, 0.0))
        return Status.SUCCESS if abs(v) > 1e-6 or abs(w) > 1e-6 else Status.FAILURE


@register("IsStalled", role=Role.CONDITION)
class IsStalled(BaseBehavior):
    """SUCCESS once the robot has not moved for `stall_ticks` consecutive ticks.

    A tick counts as stalled when the driver proposed no motion or the safety
    gate rejected the proposal. Odometry is not consulted.
    """

    READS_CARRIED = frozenset({"cmd_proposal", "brake_now"})

    def __init__(self, name=None, ports=None):
        super().__init__(name=name, ports=ports)
        self._streak = 0

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        v, w = blackboard.get("cmd_proposal", (0.0, 0.0))
        motionless = abs(v) <= 1e-6 and abs(w) <= 1e-6
        rejected = bool(blackboard.get("brake_now", False))
        self._streak = self._streak + 1 if (motionless or rejected) else 0
        return (Status.SUCCESS
                if self._streak >= self.setting(context, "stall_ticks")
                else Status.FAILURE)


@register("IsMissionComplete", role=Role.CONDITION)
class IsMissionComplete(BaseBehavior):
    """SUCCESS once `trial_timeout_sec` has elapsed since the first tick."""

    WRITES = frozenset({"mission_complete"})

    def __init__(self, name=None, ports=None):
        super().__init__(name=name, ports=ports)
        self._started_at = None

    def tick(self, snapshot, blackboard, context: BehaviorContext) -> Status:
        if self._started_at is None:
            self._started_at = time.monotonic()
        complete = time.monotonic() - self._started_at >= self.setting(context, "trial_timeout_sec")
        blackboard.set("mission_complete", complete)
        return Status.SUCCESS if complete else Status.FAILURE
