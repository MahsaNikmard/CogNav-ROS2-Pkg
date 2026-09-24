"""Thread-safe buffer of the latest perception and odometry, and the snapshot a tick reads."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
import threading

import numpy as np

from cognav_representation.geometry import wrap_angle, yaw_from_quaternion
from cognav_bt_behaviors.timing import ExposureBudget, PeriodEstimator


@dataclass
class RuntimeSnapshot:
    perception_msg: object | None
    odom_msg: object | None
    odom: tuple[float, float, float] | None
    scan_ranges: np.ndarray | None
    scan_angles: np.ndarray | None
    perception_fresh: bool
    stamp_skew_sec: float | None
    #: now - header.stamp, or None when the frame carries no usable stamp.
    age_sec: float | None
    #: Freshness deadline the age was judged against.
    deadline_sec: float
    #: Estimated frame period, or None until two frames have arrived.
    period_sec: float | None
    #: header.stamp of the perception message, which identifies the frame.
    perception_stamp_sec: float | None
    #: Half-open slice of bins with real readings; the rest is FOV padding.
    real_bin_range: tuple[int, int]


#: How far in the future a perception stamp may be before the clocks are
#: considered mismatched.
_CLOCK_SKEW_TOLERANCE_SEC = 0.05


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def odom_pose_tuple(msg) -> tuple[float, float, float]:
    pose = msg.pose.pose
    return (pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation))


class RuntimeBuffer:
    """Holds the latest perception and odometry and turns them into snapshots.

    Subscription callbacks write concurrently with the mission loop that reads,
    so all shared state is accessed under `_lock`.
    """

    def __init__(self, node, sync_model: str, budget: ExposureBudget, max_stamp_skew_sec: float,
                 num_bins: int, fov_deg: float, fov_padding_bins: int,
                 odom_buffer_size: int = 200):
        self.node = node
        self.sync_model = sync_model
        self.budget = budget
        self.max_stamp_skew_sec = max_stamp_skew_sec
        self.num_bins = int(num_bins)
        self.fov_deg = float(fov_deg)
        self.fov_padding_bins = int(fov_padding_bins)
        self.bin_angles = self._compute_bin_angles()
        self.latest_distance_vector = None
        self.latest_odom = None
        self.odom_buffer = deque(maxlen=odom_buffer_size)
        self.period = PeriodEstimator()
        self._lock = threading.Lock()
        self._clock_mismatch_warned = False

    def _warn_clock_mismatch(self, delta: float) -> None:
        """Log a use_sim_time mismatch once."""
        if self._clock_mismatch_warned:
            return
        self._clock_mismatch_warned = True
        self.node.get_logger().error(
            f"Perception stamps are {-delta:.1f}s in the future: bt_runner and "
            f"cognav_perception disagree on use_sim_time. Treating every frame "
            f"as stale."
        )

    @property
    def period_sec(self) -> float | None:
        """Measured frame period, the hold time T_h."""
        return self.period.period_sec

    def deadline_sec(self) -> float:
        """Freshness deadline for the measured hold time; the whole budget before two frames."""
        hold = self.period.period_sec
        return self.budget.deadline_sec(hold if hold is not None else 0.0)

    def set_geometry(self, *, num_bins: int | None = None, fov_deg: float | None = None,
                     fov_padding_bins: int | None = None) -> None:
        with self._lock:
            if num_bins is not None:
                self.num_bins = int(num_bins)
            if fov_deg is not None:
                self.fov_deg = float(fov_deg)
            if fov_padding_bins is not None:
                self.fov_padding_bins = int(fov_padding_bins)
            self.bin_angles = self._compute_bin_angles()

    def set_distance_vector(self, msg) -> None:
        with self._lock:
            self.latest_distance_vector = msg
            stamp_sec = stamp_to_sec(msg.header.stamp)
            if stamp_sec > 0.0:
                self.period.observe(stamp_sec)

    def set_odom(self, msg) -> None:
        with self._lock:
            self.latest_odom = msg
            self.odom_buffer.append(msg)

    def snapshot(self) -> RuntimeSnapshot | None:
        with self._lock:
            if self.latest_distance_vector is None or self.latest_odom is None:
                return None

            perception_msg = self.latest_distance_vector
            odom_msg = self.latest_odom
            if self.sync_model == "interpolation":
                odom_msg = self._odom_at_or_before_scan_stamp() or self.latest_odom
            deadline_sec = self.deadline_sec()
            period_sec = self.period.period_sec
            bin_angles = self.bin_angles
            real_bin_range = (self.fov_padding_bins, self.fov_padding_bins + self.num_bins)

        now_sec = stamp_to_sec(self.node.get_clock().now().to_msg())
        perception_sec = stamp_to_sec(perception_msg.header.stamp)
        odom_sec = stamp_to_sec(odom_msg.header.stamp)

        # A frame without a stamp cannot be aged, so it is never fresh.
        if perception_sec <= 0.0:
            age_sec = None
            fresh = False
        else:
            delta = now_sec - perception_sec
            if delta < -_CLOCK_SKEW_TOLERANCE_SEC:
                # A future stamp means the clocks disagree; fail closed.
                age_sec = None
                fresh = False
                self._warn_clock_mismatch(delta)
            else:
                age_sec = max(0.0, delta)
                fresh = age_sec <= deadline_sec

        skew = abs(perception_sec - odom_sec) if perception_sec > 0.0 and odom_sec > 0.0 else None
        if skew is not None:
            fresh = fresh and skew <= self.max_stamp_skew_sec

        return RuntimeSnapshot(
            perception_msg=perception_msg,
            odom_msg=odom_msg,
            odom=odom_pose_tuple(odom_msg),
            scan_ranges=np.asarray(perception_msg.distance_vector, dtype=np.float64),
            scan_angles=bin_angles,
            perception_fresh=fresh,
            stamp_skew_sec=skew,
            age_sec=age_sec,
            deadline_sec=deadline_sec,
            period_sec=period_sec,
            perception_stamp_sec=perception_sec if perception_sec > 0.0 else None,
            real_bin_range=real_bin_range,
        )

    def _compute_bin_angles(self) -> np.ndarray:
        """Centre bearing of every bin in the padded vector, bin 0 leftmost."""
        fov_rad = math.radians(self.fov_deg)
        bin_width = fov_rad / self.num_bins
        half = fov_rad / 2.0
        n_total = self.num_bins + 2 * self.fov_padding_bins
        idx = np.arange(n_total, dtype=np.float64)
        return half + bin_width * (self.fov_padding_bins - 0.5 - idx)

    def _odom_at_or_before_scan_stamp(self):
        perception_sec = stamp_to_sec(self.latest_distance_vector.header.stamp)
        if perception_sec <= 0.0:
            return None
        before = None
        after = None
        for msg in self.odom_buffer:
            msg_sec = stamp_to_sec(msg.header.stamp)
            if msg_sec <= perception_sec:
                before = msg
            elif msg_sec > perception_sec:
                after = msg
                break
        if before is None:
            return after
        if after is None:
            return before
        before_sec = stamp_to_sec(before.header.stamp)
        after_sec = stamp_to_sec(after.header.stamp)
        if after_sec <= before_sec:
            return before
        alpha = min(1.0, max(0.0, (perception_sec - before_sec) / (after_sec - before_sec)))
        return self._interpolate_odom(before, after, alpha, self.latest_distance_vector.header.stamp)

    def _interpolate_odom(self, before, after, alpha: float, stamp):
        msg = deepcopy(before)
        msg.header.stamp = stamp
        bp = before.pose.pose.position
        ap = after.pose.pose.position
        msg.pose.pose.position.x = bp.x + alpha * (ap.x - bp.x)
        msg.pose.pose.position.y = bp.y + alpha * (ap.y - bp.y)
        msg.pose.pose.position.z = bp.z + alpha * (ap.z - bp.z)

        byaw = yaw_from_quaternion(before.pose.pose.orientation)
        ayaw = yaw_from_quaternion(after.pose.pose.orientation)
        yaw = byaw + alpha * wrap_angle(ayaw - byaw)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        return msg
