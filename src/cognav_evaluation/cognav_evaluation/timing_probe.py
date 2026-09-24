#!/usr/bin/env python3
"""Passive check of the exposure-window bound v_max * (age + T_h) <= M.

Listens to the distance vector and /cmd_vel of a running stack and reports the
frame age (now - header.stamp), the frame period, the command period T_h, and
frames per command, where 1.00 means every frame produced one command.

    ros2 run cognav_evaluation timing_probe --ros-args -p duration_sec:=30.0
"""

from __future__ import annotations

import statistics

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from cognav_msgs.msg import DistanceVector

QOS_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class TimingProbe(Node):
    def __init__(self) -> None:
        super().__init__("cognav_timing_probe")
        self.declare_parameter("perception_topic", "/perception/distance_vector")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("duration_sec", 30.0)
        self.declare_parameter("safety_margin_m", 0.50)
        self.declare_parameter("max_v", 0.22)

        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.safety_margin_m = float(self.get_parameter("safety_margin_m").value)
        self.max_v = float(self.get_parameter("max_v").value)

        self.ages: list[float] = []
        self.frame_gaps: list[float] = []
        self.cmd_gaps: list[float] = []
        self.frames_per_cmd: list[int] = []
        self.unstamped = 0
        self._prev_capture: float | None = None
        self._prev_cmd: float | None = None
        self._frames_since_cmd = 0
        self._prev_was_zero = False
        self._stop_repeats = 0

        self.create_subscription(
            DistanceVector,
            str(self.get_parameter("perception_topic").value),
            self._on_frame,
            QOS_RELIABLE,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            self._on_cmd,
            QOS_RELIABLE,
        )
        self.create_timer(self.duration_sec, self._report)
        self.get_logger().info(f"timing probe listening for {self.duration_sec:.0f}s")

    def _now(self) -> float:
        return _stamp_to_sec(self.get_clock().now().to_msg())

    def _on_frame(self, msg: DistanceVector) -> None:
        capture = _stamp_to_sec(msg.header.stamp)
        if capture <= 0.0:
            self.unstamped += 1
            return
        self.ages.append(self._now() - capture)
        if self._prev_capture is not None and capture > self._prev_capture:
            self.frame_gaps.append(capture - self._prev_capture)
        self._prev_capture = capture
        self._frames_since_cmd += 1

    def _on_cmd(self, msg: Twist) -> None:
        # Repeated zero commands (braking, starvation) count as one event.
        is_zero = abs(msg.linear.x) < 1e-9 and abs(msg.angular.z) < 1e-9
        if is_zero and self._prev_was_zero:
            self._stop_repeats += 1
            return
        self._prev_was_zero = is_zero

        now = self._now()
        if self._prev_cmd is not None:
            self.cmd_gaps.append(now - self._prev_cmd)
        self._prev_cmd = now
        self.frames_per_cmd.append(self._frames_since_cmd)
        self._frames_since_cmd = 0

    @staticmethod
    def _row(label: str, samples: list[float], unit: str = "ms") -> float | None:
        if not samples:
            print(f"  {label:28s} {'no samples':>12s}")
            return None
        scale = 1e3 if unit == "ms" else 1.0
        s = sorted(x * scale for x in samples)
        mean = statistics.mean(s)
        print(f"  {label:28s} {len(s):5d}x  mean {mean:8.2f}  p50 {s[len(s) // 2]:8.2f}  "
              f"p95 {s[int(len(s) * 0.95)]:8.2f}  max {s[-1]:8.2f}  {unit}")
        return mean

    def _report(self) -> None:
        print("\n[measured]")
        age = self._row("age (capture -> consumer)", self.ages)
        self._row("T_p (frame period)", self.frame_gaps)
        hold = self._row("T_h (command period)", self.cmd_gaps)
        if self.unstamped:
            print(f"  {self.unstamped} unstamped frames, which the runner treats as stale")

        if self.frames_per_cmd:
            reuse = sum(1 for n in self.frames_per_cmd if n == 0)
            dropped = sum(max(0, n - 1) for n in self.frames_per_cmd)
            ratio = statistics.mean(self.frames_per_cmd)
            print(f"\n[frames per command] over {len(self.frames_per_cmd)} commands")
            print(f"  frames per command            {ratio:.2f}   (target 1.00)")
            print(f"  commands re-using a frame     {reuse:5d}   "
                  f"({100.0 * reuse / len(self.frames_per_cmd):.1f}%)")
            print(f"  frames never acted on         {dropped:5d}")
            if self._stop_repeats:
                print(f"  ({self._stop_repeats} repeated stop commands collapsed)")

        budget = self.safety_margin_m / max(self.max_v, 1e-6)
        print(f"\n[exposure] budget M/v_max = {budget:.3f}s "
              f"(M={self.safety_margin_m:.2f}m, v_max={self.max_v:.2f}m/s)")
        if age is not None and hold is not None:
            w = (age + hold) / 1e3
            travel = self.max_v * w
            verdict = "OK" if travel <= self.safety_margin_m else "VIOLATED"
            print(f"  W = age + T_h               = {w:.3f}s")
            print(f"  travel at v_max             = {travel:.3f}m of {self.safety_margin_m:.2f}m   {verdict}")
            if travel > self.safety_margin_m:
                print("  Lower max_v or raise safety_margin.")
        else:
            print("  incomplete measurement, cannot evaluate W")

        raise SystemExit(0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TimingProbe()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
