"""Every module imports, a snapshot can be built, and every mission leaf is registered."""
import importlib
import inspect
import os

from _check import Checker

MODULES = [
    "cognav_bt_behaviors",
    "cognav_bt_behaviors.base",
    "cognav_bt_behaviors.blackboard",
    "cognav_bt_behaviors.conditions",
    "cognav_bt_behaviors.driving",
    "cognav_bt_behaviors.planning",
    "cognav_bt_behaviors.safety",
    "cognav_bt_behaviors.status",
    "cognav_bt_behaviors.timing",
    "cognav_bt_runner.bt_runner",
    "cognav_bt_runner.snapshot",
    "cognav_bt_runner.tree_log",
    "cognav_bt_runner.xml_tree",
    "cognav_mission.action",
    "cognav_evaluation.metrics",
    "cognav_evaluation.tick_log",
    "cognav_evaluation.occupancy",
    "cognav_perception.defaults",
    "cognav_perception.encoding",
]

REPRESENTATION = [
    "cognav_representation.geometry",
    "cognav_representation.navmesh",
    "cognav_representation.oracle",
]

#: Every leaf the missions use, with its role.
EXPECTED_LEAVES = {
    "Brake": "SAFETY",
    "ExecuteCommand": "DRIVER",
    "FollowTheGap": "DRIVER",
    "HasMotion": "CONDITION",
    "HasWaypoint": "CONDITION",
    "IsCommandSafe": "SAFETY",
    "IsMissionComplete": "CONDITION",
    "IsStalled": "CONDITION",
    "RRTExploreWaypoint": "PLANNER",
    "Recover": "SAFETY",
}


def main():
    c = Checker()

    print("\n[1] every module imports")
    for name in MODULES + REPRESENTATION:
        try:
            importlib.import_module(name)
            c(name, True)
        except Exception as exc:  # noqa: BLE001
            c(name, False, f"{type(exc).__name__}: {exc}")

    print("\n[2] a snapshot can be constructed")
    try:
        from _harness import ANGLES, REAL_BINS, scene, snapshot

        s = snapshot(scene(2.0), fresh=True)
        c("perception_fresh holds the bool it was given", s.perception_fresh is True)
        c("real_bin_range holds the slice", s.real_bin_range == REAL_BINS)
        c("scan_angles holds the angles", len(s.scan_angles) == len(ANGLES))
    except Exception as exc:  # noqa: BLE001
        c("snapshot() builds a RuntimeSnapshot", False, f"{type(exc).__name__}: {exc}")

    print("\n[3] cognav_representation resolves from one directory")
    roots = set()
    for name in REPRESENTATION:
        try:
            roots.add(os.path.dirname(inspect.getfile(importlib.import_module(name))))
        except Exception as exc:  # noqa: BLE001
            c(f"{name} resolves", False, f"{type(exc).__name__}: {exc}")
    c("one copy of cognav_representation", len(roots) == 1, ", ".join(sorted(roots)))

    print("\n[4] every mission leaf is registered")
    try:
        from cognav_bt_behaviors.base import REGISTRY

        for tag, role in sorted(EXPECTED_LEAVES.items()):
            cls = REGISTRY.get(tag)
            if cls is None:
                c(f"{tag} is registered", False, "tag does not resolve")
            else:
                c(f"{tag} is registered as {role}", cls.ROLE.name == role,
                  f"got {cls.ROLE.name}")
        extra = sorted(set(REGISTRY) - set(EXPECTED_LEAVES))
        c("no unlisted tags in the registry", not extra, ", ".join(extra))
    except Exception as exc:  # noqa: BLE001
        c("the registry is populated", False, f"{type(exc).__name__}: {exc}")

    c.finish("module imports")


if __name__ == "__main__":
    main()
