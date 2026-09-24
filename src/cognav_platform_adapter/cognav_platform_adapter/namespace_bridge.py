#!/usr/bin/env python3
"""Relay a namespaced TurtleBot4 to the un-namespaced CogNav topics.

CogNav consumes /oakd/rgb/preview/image_raw, /odom, /tf, /tf_static and
publishes /cmd_vel; the robot publishes the same data under its namespace.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

try:
    from irobot_create_msgs.msg import HazardDetectionVector

    HAS_IROBOT_MSGS = True
except ImportError:
    HAS_IROBOT_MSGS = False


QOS_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

QOS_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

QOS_TF_PUB = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
)

QOS_TF_STATIC = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
)

QOS_TRANSIENT_LOCAL = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

DEFAULT_NAMESPACES = {
    "turtlebot": "/Turtlebot_02493",
}


def _qos_label(qos: QoSProfile) -> str:
    return "BE" if qos.reliability == ReliabilityPolicy.BEST_EFFORT else "REL"


def _normalise_ns(ns: str) -> str:
    ns = str(ns or "").strip()
    if not ns:
        return ""
    if not ns.startswith("/"):
        ns = "/" + ns
    return ns.rstrip("/")


def _resolve_platform_topic(ns: str, topic: str) -> str:
    topic = str(topic or "").strip()
    if not topic:
        return ""
    if topic.startswith("/"):
        return topic.rstrip("/")
    return f"{ns}/{topic.strip('/')}" if ns else f"/{topic.strip('/')}"


def _build_relay_table(ns: str, *, rgb_image_topic: str = ""):
    """(source, destination, type, sub QoS, pub QoS, direction) per relayed topic."""
    tf_sub_qos = QOS_BEST_EFFORT
    camera_info_qos = QOS_BEST_EFFORT
    rgb_source_topic = _resolve_platform_topic(ns, rgb_image_topic or "color/preview/image")

    table = [
        (f"{ns}/tf", "/tf", TFMessage, tf_sub_qos, QOS_TF_PUB, "from_platform"),
        (f"{ns}/tf_static", "/tf_static", TFMessage, QOS_TF_STATIC, QOS_TF_STATIC, "from_platform"),
        (
            rgb_source_topic,
            "/oakd/rgb/preview/image_raw",
            Image,
            QOS_BEST_EFFORT,
            QOS_BEST_EFFORT,
            "from_platform",
        ),
        (
            f"{ns}/oakd/rgb/preview/camera_info",
            "/oakd/rgb/preview/camera_info",
            CameraInfo,
            camera_info_qos,
            camera_info_qos,
            "from_platform",
        ),
        (f"{ns}/stereo/depth", "/stereo/depth", Image, QOS_BEST_EFFORT, QOS_BEST_EFFORT, "from_platform"),
        (f"{ns}/scan", "/scan", LaserScan, QOS_RELIABLE, QOS_RELIABLE, "from_platform"),
        (f"{ns}/odom", "/odom", Odometry, QOS_BEST_EFFORT, QOS_RELIABLE, "from_platform"),
        (f"{ns}/imu", "/imu", Imu, QOS_BEST_EFFORT, QOS_BEST_EFFORT, "from_platform"),
        (
            f"{ns}/robot_description",
            "/robot_description",
            String,
            QOS_TRANSIENT_LOCAL,
            QOS_TRANSIENT_LOCAL,
            "from_platform",
        ),
        (f"{ns}/joint_states", "/joint_states", JointState, QOS_BEST_EFFORT, QOS_BEST_EFFORT, "from_platform"),
        (f"{ns}/cmd_vel", "/cmd_vel", Twist, QOS_RELIABLE, QOS_RELIABLE, "to_platform"),
    ]
    if HAS_IROBOT_MSGS:
        table.append(
            (
                f"{ns}/hazard_detection",
                "/hazard_detection",
                HazardDetectionVector,
                QOS_BEST_EFFORT,
                QOS_BEST_EFFORT,
                "from_platform",
            )
        )
    return table


class NamespaceBridge(Node):
    """Relay the robot's namespaced topics to the canonical CogNav topics."""

    def __init__(self) -> None:
        super().__init__("cognav_namespace_bridge")
        self.declare_parameter("platform", "turtlebot")
        self.declare_parameter("robot_ns", "")
        self.declare_parameter("rgb_image_topic", "")

        platform = str(self.get_parameter("platform").value).strip().lower()
        if platform not in DEFAULT_NAMESPACES:
            raise ValueError(
                f"platform must be one of {sorted(DEFAULT_NAMESPACES)} for namespace_bridge; got {platform!r}"
            )
        requested_ns = str(self.get_parameter("robot_ns").value)
        self.ns = _normalise_ns(requested_ns) or DEFAULT_NAMESPACES[platform]
        self.platform = platform
        rgb_image_topic = str(self.get_parameter("rgb_image_topic").value).strip()

        self.get_logger().info(
            f"Starting {platform} namespace bridge: {self.ns} <-> canonical CogNav topics"
        )

        self._relays = []
        self._msg_counts: dict[str, int] = {}
        for platform_topic, canonical_topic, msg_type, sub_qos, pub_qos, direction in _build_relay_table(
            self.ns,
            rgb_image_topic=rgb_image_topic,
        ):
            if direction == "from_platform":
                sub_topic = platform_topic
                pub_topic = canonical_topic
            else:
                sub_topic = canonical_topic
                pub_topic = platform_topic

            publisher = self.create_publisher(msg_type, pub_topic, pub_qos)
            self._msg_counts[sub_topic] = 0

            def make_callback(pub, src, dst, node_ref):
                def callback(msg):
                    pub.publish(msg)
                    node_ref._msg_counts[src] += 1
                    count = node_ref._msg_counts[src]
                    if count == 1:
                        node_ref.get_logger().info(f"first relay message: {src} -> {dst}")
                    elif count % 500 == 0:
                        node_ref.get_logger().info(f"relay heartbeat: {src} -> {dst} ({count} msgs)")

                return callback

            subscription = self.create_subscription(
                msg_type,
                sub_topic,
                make_callback(publisher, sub_topic, pub_topic, self),
                sub_qos,
            )
            self._relays.append((subscription, publisher))
            self.get_logger().info(
                f"relay: {sub_topic} -> {pub_topic} "
                f"[{msg_type.__name__}, sub={_qos_label(sub_qos)}, pub={_qos_label(pub_qos)}]"
            )

        self.get_logger().info(f"Bridge active with {len(self._relays)} relays")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NamespaceBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
