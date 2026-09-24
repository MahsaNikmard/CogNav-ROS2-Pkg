#!/usr/bin/env python3
"""Scoring topics for laboratory runs on the TurtleBot4.

Supplies what Habitat publishes in simulation, from a map, AMCL and the bumpers:

    subscribes                       publishes
    /amcl_pose                       /platform/true_pose
    /odom            (fallback)      /platform/true_clearance_m
    /hazard_detection                /collision
                                     /platform/scene            (latched)
                                     /platform/navigable_area_m2 (latched)

The AMCL pose is used for scoring only; the stack drives on /odom. Clearance
is the distance transform of the map's free space minus the body radius, the
same convention as Habitat's `distance_to_closest_obstacle`. Off the map or on
an occupied cell nothing is published, so a missing value is not read as zero.
"""
from __future__ import annotations

import math
import os

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

from cognav_evaluation.occupancy import OccupancyMap, clearance_field

try:
    from irobot_create_msgs.msg import HazardDetection, HazardDetectionVector

    HAS_IROBOT_MSGS = True
except ImportError:
    HAS_IROBOT_MSGS = False

QOS_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
QOS_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
QOS_LATCHED = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class LabBridge(Node):
    def __init__(self) -> None:
        super().__init__("lab_bridge")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("robot_radius", 0.18)
        self.declare_parameter("pose_source", "amcl")
        self.declare_parameter("odom_fallback_sec", 5.0)
        self.declare_parameter("publish_rate_hz", 15.0)

        g = self.get_parameter
        map_yaml = str(g("map_yaml").value)
        self.robot_radius = float(g("robot_radius").value)
        self.want_amcl = str(g("pose_source").value).lower() != "odom"
        self.fallback_after = float(g("odom_fallback_sec").value)

        if not map_yaml or not os.path.isfile(map_yaml):
            self.get_logger().error(
                f"map_yaml is required and must exist; got {map_yaml!r}")
            raise SystemExit(2)

        self.occ = OccupancyMap(map_yaml)
        self.clear = clearance_field(self.occ, self.robot_radius)
        self.map_yaml = map_yaml

        self.true_pose_pub = self.create_publisher(
            Float64MultiArray, "/platform/true_pose", QOS_RELIABLE)
        self.true_clearance_pub = self.create_publisher(
            Float64, "/platform/true_clearance_m", QOS_RELIABLE)
        self.collision_pub = self.create_publisher(Bool, "/collision", QOS_RELIABLE)
        self.scene_pub = self.create_publisher(String, "/platform/scene", QOS_LATCHED)
        self.area_pub = self.create_publisher(
            Float64, "/platform/navigable_area_m2", QOS_LATCHED)
        self.scene_pub.publish(String(data=map_yaml))

        self._amcl = None
        self._odom = None
        self._amcl_seen = False
        self._warned_fallback = False
        self._area_published = False
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_cb, QOS_RELIABLE)
        self.create_subscription(Odometry, "/odom", self._odom_cb, QOS_BEST_EFFORT)
        if HAS_IROBOT_MSGS:
            self.create_subscription(
                HazardDetectionVector, "/hazard_detection",
                self._hazard_cb, QOS_BEST_EFFORT)
        else:
            self.get_logger().warn(
                "irobot_create_msgs is not importable; bumper contacts will not be counted.")

        self._start = self.get_clock().now()
        self.create_timer(1.0 / max(float(g("publish_rate_hz").value), 1e-3),
                          self._tick)
        self.get_logger().info(
            f"lab_bridge: map={os.path.basename(map_yaml)} "
            f"free={self.occ.free_area_m2():.1f} m2 radius={self.robot_radius} "
            f"pose_source={'amcl' if self.want_amcl else 'odom'}")

    def _amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._amcl = (p.x, p.y, yaw)
        if not self._amcl_seen:
            self._amcl_seen = True
            self.get_logger().info("amcl_pose is live")

    def _odom_cb(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._odom = (p.x, p.y, yaw)

    def _hazard_cb(self, msg) -> None:
        """Publish one contact per bump; cliff, stall and backup-limit events are ignored."""
        if any(d.type == HazardDetection.BUMP for d in msg.detections):
            self.collision_pub.publish(Bool(data=True))

    def _pose(self):
        """(x, y, yaw, source). AMCL when it is live, odometry when it is not."""
        if self.want_amcl and self._amcl is not None:
            return self._amcl[0], self._amcl[1], self._amcl[2], "amcl"
        if self._odom is None:
            return None
        if self.want_amcl and not self._warned_fallback:
            waited = (self.get_clock().now() - self._start).nanoseconds / 1e9
            if waited > self.fallback_after:
                self._warned_fallback = True
                self.get_logger().warn(
                    f"no /amcl_pose after {waited:.0f}s; falling back to wheel "
                    f"odometry, which drifts.")
        return self._odom[0], self._odom[1], self._odom[2], "odom"

    def _tick(self) -> None:
        got = self._pose()
        if got is None:
            return
        x, y, yaw, source = got
        self.true_pose_pub.publish(Float64MultiArray(data=[x, y, yaw]))

        row, col = self.occ.world_to_cell(x, y)
        if self.occ.in_bounds(row, col) and self.occ.free[row, col]:
            self.true_clearance_pub.publish(
                Float64(data=float(self.clear[row, col])))

        if not self._area_published:
            reach = self.occ.reachable_mask((x, y), self.robot_radius)
            if reach is not None:
                area = float(reach.sum()) * self.occ.resolution ** 2
                self.area_pub.publish(Float64(data=area))
                self._area_published = True
                self.get_logger().info(
                    f"reachable from the start pose: {area:.1f} m2 "
                    f"(pose from {source})")
            else:
                self.get_logger().warn(
                    "the start pose is not reachable at the body radius; move the "
                    "robot or check the localisation.")


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = LabBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
