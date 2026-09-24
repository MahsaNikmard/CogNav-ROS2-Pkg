"""Leaf base class, role contracts, runner-wide tuning and the tag registry.

Each role is defined by the blackboard keys it reads and writes:

    Role       reads                              writes
    ---------  ---------------------------------  ---------------------------
    CONDITION  anything                           its own flags only
    SAFETY     snapshot, cmd_proposal             blocked_bins, brake_now,
                                                  rejected_bins, cmd_vel
    PLANNER    snapshot, rejected_bins            reference_path
    DRIVER     snapshot.odom, reference_path      cmd_proposal, cmd_vel

A leaf declares its keys as class attributes. Keys that may hold a value from
an earlier tick go in READS_CARRIED, which the load-time check in
`cognav_bt_runner.xml_tree.validate_contracts` does not enforce:

    @register("RRTExploreWaypoint", role=Role.PLANNER)
    class RRTExploreWaypoint(BaseBehavior):
        READS_CARRIED = frozenset({"rejected_bins"})
        WRITES = frozenset({"reference_path", "rejected_bins"})

Any BehaviorConfig field can be overridden per leaf from the mission XML, for
example `<RRTExploreWaypoint seed="7"/>`. Leaves read tuning through
`self.setting(context, name)` so that the override applies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from enum import Enum
from typing import get_type_hints

from geometry_msgs.msg import Twist


class Role(Enum):
    """What a leaf is for."""

    CONDITION = "condition"
    SAFETY = "safety"
    PLANNER = "planner"
    DRIVER = "driver"


#: Keys the runner writes before the first tick. Both are read only by the
#: mesh arm: the rasterised navmesh of the scene and the Habitat pose at which
#: odometry was zeroed.
SEEDED_KEYS = frozenset({"mesh_path", "start_hab"})


@dataclass
class BehaviorConfig:
    """Runner-wide tuning. Every field is also a bt_runner ROS parameter."""

    #: Free-space test used by the planner: "polar" checks the current polar
    #: vector; "mesh" checks the scene's navmesh inside the same field of view
    #: and range, and needs `mesh_path` and `start_hab` on the blackboard.
    free_space_source: str = "polar"
    #: Range beyond which the mesh oracle refuses a point. Must equal the
    #: perception `max_range` so the mesh arm sees no further than the polar arm.
    mesh_max_range: float = 5.0

    robot_radius: float = 0.18
    safety_margin: float = 0.50
    warning_margin: float = 0.40
    #: Consecutive motionless ticks before the Escape rung turns in place.
    stall_ticks: int = 15
    recovery_linear_speed: float = 0.04
    recovery_angular_speed: float = 0.40
    #: Planning horizon; scales forward speed against gap depth in the driver.
    lookahead_dist: float = 1.0
    waypoint_reached_tol: float = 0.36
    #: Ticks a committed branch is kept before the planner replans.
    replan_period_ticks: int = 45

    #: RRT samples per replan.
    rrt_iterations: int = 120
    #: Extension length, which is also the spacing of the emitted path.
    rrt_step: float = 0.35
    #: Radius of the sampling disc in the robot frame.
    rrt_sample_radius: float = 2.5
    #: Fraction of samples aimed at the deepest drivable bearing.
    rrt_bias: float = 0.35
    #: Novelty is capped at this distance from the trail of visited poses.
    novelty_radius: float = 2.0
    #: The visit trail stores one pose per this much travel.
    visit_spacing: float = 0.30

    max_v: float = 0.22
    max_w: float = 0.8
    #: Proportional gain on heading error.
    k_w: float = 1.5
    #: Added to robot_radius when testing whether the body fits along a bearing.
    maneuver_clearance: float = 0.18
    #: Duration of the proposed command that the swept-arc check simulates.
    arc_horizon_sec: float = 1.5
    trial_timeout_sec: float = 500.0
    #: Seed for leaves that sample. Negative draws from system entropy.
    seed: int = -1


#: Field name to type. `get_type_hints` resolves the postponed annotations.
CONFIG_FIELD_TYPES = {
    f.name: get_type_hints(BehaviorConfig)[f.name] for f in fields(BehaviorConfig)
}


def _cast_port(name: str, raw: str):
    """Cast an XML attribute to the type of the config field it overrides."""
    if name not in CONFIG_FIELD_TYPES:
        raise ValueError(
            f"unknown port '{name}'; valid ports are "
            f"{', '.join(sorted(CONFIG_FIELD_TYPES))}"
        )
    declared = CONFIG_FIELD_TYPES[name]
    caster = declared if declared in (int, float, str) else float
    try:
        return caster(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"port '{name}' expects {caster.__name__}, got {raw!r}") from exc


class BehaviorContext:
    """What every leaf of one mission shares: node, publisher, tuning, tick id."""

    def __init__(self, node, cmd_vel_pub, config: BehaviorConfig, enable_cmd_vel: bool):
        self.node = node
        self.cmd_vel_pub = cmd_vel_pub
        self.config = config
        self.enable_cmd_vel = enable_cmd_vel
        self.tick_id = 0

    def publish_twist(self, v: float, w: float, blackboard) -> None:
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        if self.enable_cmd_vel:
            if not self.node.context.ok():
                blackboard.set("cmd_vel", (float(v), float(w)))
                return
            try:
                self.cmd_vel_pub.publish(twist)
            except Exception as exc:
                if self.node.context.ok() and "destruction was requested" not in str(exc):
                    raise
        blackboard.set("cmd_vel", (float(v), float(w)))


class BaseBehavior:
    """Base for every leaf."""

    ROLE: Role = Role.CONDITION
    #: Keys that must hold this tick's value when the leaf ticks.
    READS: frozenset = frozenset()
    #: Keys that may hold an earlier tick's value.
    READS_CARRIED: frozenset = frozenset()
    WRITES: frozenset = frozenset()

    def __init__(self, name: str | None = None, ports: dict | None = None):
        self.name = name or self.__class__.__name__
        self.ports = dict(ports or {})
        self._rng = None

    def setting(self, context: BehaviorContext, key: str):
        """Port value if the mission supplied one, else the runner parameter."""
        if key in self.ports:
            return self.ports[key]
        return getattr(context.config, key)

    def rng(self, context: BehaviorContext) -> random.Random:
        """This leaf's random stream, seeded once from `seed` on first use."""
        if self._rng is None:
            seed = int(self.setting(context, "seed"))
            self._rng = random.Random(None if seed < 0 else seed)
        return self._rng

    def tick(self, snapshot, blackboard, context: BehaviorContext):
        raise NotImplementedError


REGISTRY: dict[str, type] = {}


def register(*names: str, role: Role):
    """Register a leaf class under one or more XML tag names."""

    def decorate(cls):
        cls.ROLE = role
        for name in names:
            if name in REGISTRY and REGISTRY[name] is not cls:
                raise ValueError(f"behavior tag '{name}' is already registered")
            REGISTRY[name] = cls
        return cls

    return decorate


def build_behavior(tag: str, name: str | None, attrib: dict):
    """Instantiate a registered leaf, casting and checking its XML ports."""
    cls = REGISTRY.get(tag)
    if cls is None:
        raise ValueError(f"Unknown behavior node '{tag}'")
    ports = {}
    for key, raw in (attrib or {}).items():
        if key == "name":
            continue
        try:
            ports[key] = _cast_port(key, raw)
        except ValueError as exc:
            raise ValueError(f"<{tag}>: {exc}") from exc
    return cls(name=name, ports=ports)
