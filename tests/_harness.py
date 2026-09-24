"""Shared fixtures: synthetic snapshots ticked through the real tree, no ROS graph."""
import math
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory

from cognav_bt_runner.snapshot import RuntimeSnapshot

from _check import Checker  # noqa: F401  (re-exported for the behavior tests)

N_BINS, PADDING, FOV_DEG = 72, 10, 89.0

# The same bin layout as RuntimeBuffer._compute_bin_angles: real bins span the
# field of view and the padding lies outside it.
_HALF = math.radians(FOV_DEG) / 2.0
_BIN_WIDTH = math.radians(FOV_DEG) / N_BINS
ANGLES = _HALF + _BIN_WIDTH * (PADDING - 0.5 - np.arange(N_BINS + 2 * PADDING, dtype=float))
REAL_BINS = (PADDING, PADDING + N_BINS)


def scene(real_range, padding_range=1.5):
    """Ranges with every real bin at `real_range` and padding at the FOV wall."""
    r = np.full(N_BINS + 2 * PADDING, float(padding_range))
    r[PADDING:PADDING + N_BINS] = real_range
    return r


def snapshot(ranges, odom=(0.0, 0.0, 0.0), fresh=True, stamp=1.0, age=0.01):
    return RuntimeSnapshot(
        perception_msg=None,
        odom_msg=None,
        odom=odom,
        scan_ranges=ranges,
        scan_angles=ANGLES,
        perception_fresh=fresh,
        stamp_skew_sec=0.0,
        age_sec=age,
        deadline_sec=0.6,
        period_sec=0.4,
        perception_stamp_sec=stamp,
        real_bin_range=REAL_BINS,
    )


def mission_xml():
    return (Path(get_package_share_directory("cognav_mission"))
            / "mission" / "random_explore.xml").read_text()


def leaf_names(node):
    """Leaf names under `node`, in tick order."""
    if not node.children:
        return [node.name]
    return [name for child in node.children for name in leaf_names(child)]
