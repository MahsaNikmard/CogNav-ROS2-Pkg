"""The swept-arc gate: verify the command that will run, not a proxy for it."""
import rclpy
from cognav_bt_behaviors.base import BehaviorConfig, BehaviorContext
from cognav_bt_behaviors.blackboard import Blackboard
from cognav_representation.geometry import corridor_check, swept_arc_check
from cognav_bt_behaviors.safety import IsCommandSafe
from cognav_bt_behaviors.status import Status
from cognav_bt_runner.xml_tree import load_tree

from _harness import (ANGLES, PADDING, REAL_BINS, Checker, leaf_names,
                      mission_xml, scene, snapshot)

check = Checker()
rclpy.init()
node = rclpy.create_node("arc")
CFG = BehaviorConfig(robot_radius=0.18, maneuver_clearance=0.10,
                     safety_margin=0.50, warning_margin=0.40, arc_horizon_sec=1.5)
HALF = CFG.robot_radius + CFG.maneuver_clearance


def ctx():
    return BehaviorContext(node, None, CFG, enable_cmd_vel=False)


print("\n1. clear ahead, obstacle to the side of a hard turn")
# Off-axis enough that the straight corridor clears it, close enough that a
# hard left arc sweeps through it.
side = scene(5.0)
side[PADDING:PADDING + 3] = 0.70                   # +34.7deg, offset 0.40m
straight = corridor_check(side, ANGLES, 0.0, HALF, 2.0)
arc = swept_arc_check(side, ANGLES, 0.37, 1.00, HALF, CFG.arc_horizon_sec, REAL_BINS)
check("straight-ray check sees nothing", not straight, f"{len(straight)} bins")
check("swept-arc check sees the obstacle", bool(arc),
      f"{len(arc)} bins, nearest {arc[0][1]:.2f}m" if arc else "none")

print("\n2. the same scene with a gentle arc is fine")
gentle = swept_arc_check(side, ANGLES, 0.37, 0.10, HALF, CFG.arc_horizon_sec, REAL_BINS)
check("small w does not sweep into it", not gentle, f"{len(gentle)} bins")
check("so the difference is the turn, not the obstacle",
      bool(swept_arc_check(side, ANGLES, 0.37, 1.00, HALF, CFG.arc_horizon_sec, REAL_BINS)))

print("\n3. turning in place sweeps nothing")
spin = swept_arc_check(scene(0.30), ANGLES, 0.0, 1.0, HALF, CFG.arc_horizon_sec, REAL_BINS)
check("v=0 returns nothing however tight the space", not spin,
      "a disc rotating about its centre occupies the same space")

print("\n4. driving straight at a wall is still caught")
head_on = swept_arc_check(scene(0.60), ANGLES, 0.40, 0.0, HALF, CFG.arc_horizon_sec, REAL_BINS)
check("w=0 straight line still detects", bool(head_on), f"{len(head_on)} bins")

print("\n5. the gate refuses the arc that would collide")
gate = IsCommandSafe()
bb = Blackboard(); bb.set("cmd_proposal", (0.37, 1.00))
result = gate.tick(snapshot(side), bb, ctx())
check("hard arc into the side obstacle -> FAILURE", result == Status.FAILURE,
      f"clearance={bb.get('clearance'):.2f}m")
bb2 = Blackboard(); bb2.set("cmd_proposal", (0.37, 0.10))
check("gentle arc in the same scene -> SUCCESS",
      gate.tick(snapshot(side), bb2, ctx()) == Status.SUCCESS)
bb3 = Blackboard(); bb3.set("cmd_proposal", (0.0, 1.0))
check("turn in place in a tight space -> SUCCESS",
      gate.tick(snapshot(scene(0.30)), bb3, ctx()) == Status.SUCCESS,
      "so a wedged robot can still look around")

print("\n6. a refused command never reaches the wheels")
tree = load_tree(mission_xml())
bb = Blackboard()
c = ctx(); c.tick_id = 1
tree.tick(snapshot(scene(0.30)), bb, c)
check("trailing Brake publishes the stop", bb.get("cmd_vel") == (0.0, 0.0))

print("\n7. the pipeline proposes before it verifies before it publishes")
names = leaf_names([x for x in tree.children if x.name == "Navigate"][0])
check("order is propose, verify, execute",
      names.index("FollowTheGap") < names.index("IsCommandSafe") < names.index("ExecuteCommand"),
      " -> ".join(names))

rclpy.shutdown()
check.finish("swept-arc verification of the commanded trajectory")
