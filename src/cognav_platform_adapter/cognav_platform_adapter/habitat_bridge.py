#!/usr/bin/env python3
"""ROS 2 side of the Habitat bridge.

Talks to habitat_sim_server.py over ZMQ and exposes the robot topic contract:

    publishes   /oakd/rgb/preview/image_raw    sensor_msgs/Image (rgb8)
                /odom                           nav_msgs/Odometry, identity at start
                /collision                      std_msgs/Bool, navmesh contact
                /clock, TF odom -> base_link
                /platform/scene                 latched
                /platform/navigable_area_m2     latched, the coverage denominator
                /platform/true_clearance_m      navmesh distance, for scoring
                /platform/true_pose             for scoring
    subscribes  /cmd_vel                        geometry_msgs/Twist

With `lockstep` (the default) the simulator advances one step of 1/sim_hz per
new command, so an episode is a deterministic function of its seed. Only the
first step is taken without a command. Without lockstep the bridge free-runs
and a command older than `cmd_stale_sec` is treated as zero.
"""

import json
import math
import time
from typing import Optional

import rclpy
import zmq
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from tf2_ros import TransformBroadcaster

QOS_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
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

QOS_CLOCK = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class HabitatBridge(Node):
    def __init__(self) -> None:
        super().__init__("platform_bridge")

        self.declare_parameter("endpoint", "tcp://127.0.0.1:5561")
        self.declare_parameter("sim_hz", 15.0)
        self.declare_parameter("image_topic", "/oakd/rgb/preview/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("collision_topic", "/collision")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("cmd_stale_sec", 0.5)
        self.declare_parameter("request_timeout_sec", 5.0)
        self.declare_parameter("server_wait_sec", 120.0)
        self.declare_parameter("lockstep", True)
        self.declare_parameter("cmd_starve_warn_sec", 10.0)

        self.endpoint = self.get_parameter("endpoint").value
        self.sim_hz = float(self.get_parameter("sim_hz").value)
        self.dt = 1.0 / max(self.sim_hz, 1e-3)
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.cmd_stale_sec = float(self.get_parameter("cmd_stale_sec").value)
        self.req_timeout_ms = int(
            float(self.get_parameter("request_timeout_sec").value) * 1000
        )

        self._ctx = zmq.Context()
        self._sock: Optional[zmq.Socket] = None
        self._connect()
        # The server takes a while to load the scene and rebuild the navmesh.
        wait_budget = float(self.get_parameter("server_wait_sec").value)
        waited = 0.0
        info = None
        while info is None and waited < wait_budget:
            info = self._request({"cmd": "info"})
            if info is None:
                waited += float(self.get_parameter("request_timeout_sec").value)
                self.get_logger().info(
                    f"waiting for habitat_sim_server on {self.endpoint} "
                    f"({waited:.0f}s / {wait_budget:.0f}s)..."
                )
        if info is None:
            raise RuntimeError(f"habitat_sim_server not reachable on {self.endpoint}")
        meta, _ = info
        self._scene_pub = self.create_publisher(String, "/platform/scene", QOS_LATCHED)
        self._area_pub = self.create_publisher(
            Float64, "/platform/navigable_area_m2", QOS_LATCHED)
        self._scene_pub.publish(String(data=str(meta.get("scene", ""))))
        if meta.get("navigable_area_m2") is not None:
            self._area_pub.publish(Float64(data=float(meta["navigable_area_m2"])))
        self.width = int(meta["width"])
        self.height = int(meta["height"])
        self.get_logger().info(
            f"Habitat server: scene={meta['scene']}  "
            f"{self.width}x{self.height} hfov={meta['hfov_deg']} deg"
        )

        self.image_pub = self.create_publisher(
            Image, self.get_parameter("image_topic").value, QOS_RELIABLE
        )
        self.odom_pub = self.create_publisher(
            Odometry, self.get_parameter("odom_topic").value, QOS_RELIABLE
        )
        self.collision_pub = self.create_publisher(
            Bool, self.get_parameter("collision_topic").value, QOS_RELIABLE
        )
        self.true_clearance_pub = self.create_publisher(
            Float64, "/platform/true_clearance_m", QOS_RELIABLE
        )
        self.true_pose_pub = self.create_publisher(
            Float64MultiArray, "/platform/true_pose", QOS_RELIABLE
        )
        self.clock_pub = self.create_publisher(Clock, "/clock", QOS_CLOCK)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.cmd_sub = self.create_subscription(
            Twist, self.get_parameter("cmd_vel_topic").value,
            self._cmd_cb, QOS_RELIABLE,
        )

        self._cmd_v = 0.0
        self._cmd_w = 0.0
        self._cmd_t = self.get_clock().now()
        # A step is taken only when a command newer than the last stepped one arrived.
        self._cmd_seq = 0
        self._stepped_seq = 0
        self._starved_since = None
        self._tick_n = 0
        self._sim_time_sec = 0.0
        self._server_down_warned = False

        self.lockstep = bool(self.get_parameter("lockstep").value)
        self.cmd_starve_warn_sec = float(
            self.get_parameter("cmd_starve_warn_sec").value)
        # Under lockstep the timer polls faster than dt so a command is acted
        # on promptly; simulated time still advances by exactly dt per step.
        self.timer = self.create_timer(
            self.dt / 4.0 if self.lockstep else self.dt, self._tick)
        self.get_logger().info(
            f"Habitat bridge ready: sim_hz={self.sim_hz}  dt={self.dt:.3f}s  "
            f"cmd_stale={self.cmd_stale_sec}s  "
            f"lockstep={'on' if self.lockstep else 'off'}"
        )

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.endpoint)

    def _request(self, obj: dict) -> Optional[tuple]:
        """One request and its reply, or None on timeout (the socket is recreated)."""
        try:
            self._sock.send_multipart([json.dumps(obj).encode()])
            if not self._sock.poll(self.req_timeout_ms):
                self._connect()
                return None
            parts = self._sock.recv_multipart()
        except zmq.ZMQError as exc:
            self.get_logger().warn(f"ZMQ error: {exc}")
            self._connect()
            return None
        meta = json.loads(parts[0]) if parts[0] else {}
        blob = parts[1] if len(parts) > 1 else b""
        return meta, blob

    def _sim_stamp(self) -> TimeMsg:
        sec = int(self._sim_time_sec)
        return TimeMsg(sec=sec, nanosec=int((self._sim_time_sec - sec) * 1e9))

    def _publish_clock(self, stamp: TimeMsg) -> None:
        msg = Clock()
        msg.clock = stamp
        self.clock_pub.publish(msg)

    def _cmd_cb(self, msg: Twist) -> None:
        self._cmd_v = float(msg.linear.x)
        self._cmd_w = float(msg.angular.z)
        self._cmd_t = self.get_clock().now()
        self._cmd_seq += 1

    def _tick(self) -> None:
        # The first step is free: the policy needs an observation to answer.
        warmup = self._tick_n == 0
        if self.lockstep and not warmup:
            if self._cmd_seq == self._stepped_seq:
                now = time.monotonic()
                if self._starved_since is None:
                    self._starved_since = now
                elif now - self._starved_since > self.cmd_starve_warn_sec:
                    self.get_logger().warn(
                        f"no command for {now - self._starved_since:.0f}s at step "
                        f"{self._tick_n}; the simulator is waiting for the policy")
                    self._starved_since = now
                return
            self._starved_since = None
            self._stepped_seq = self._cmd_seq

        v, w = self._cmd_v, self._cmd_w
        if not self.lockstep:
            age = (self.get_clock().now() - self._cmd_t).nanoseconds / 1e9
            if age > self.cmd_stale_sec:
                v, w = 0.0, 0.0

        rep = self._request({"cmd": "step", "v": v, "w": w, "dt": self.dt})
        if rep is None:
            if not self._server_down_warned:
                self.get_logger().warn(
                    f"habitat_sim_server not answering on {self.endpoint}; retrying."
                )
                self._server_down_warned = True
            return
        self._server_down_warned = False
        meta, blob = rep
        x, y, yaw = meta["pose"]
        collided = bool(meta["collided"])

        self._sim_time_sec += self.dt
        stamp = self._sim_stamp()
        self._publish_clock(stamp)

        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = "oakd_rgb_camera_optical_frame"
        img.height = self.height
        img.width = self.width
        img.encoding = "rgb8"
        img.is_bigendian = 0
        img.step = self.width * 3
        img.data = blob
        self.image_pub.publish(img)

        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = self.odom_frame
        od.child_frame_id = self.base_frame
        od.pose.pose.position.x = x
        od.pose.pose.position.y = y
        od.pose.pose.orientation.z = math.sin(yaw / 2.0)
        od.pose.pose.orientation.w = math.cos(yaw / 2.0)
        od.twist.twist.linear.x = v
        od.twist.twist.angular.z = w
        self.odom_pub.publish(od)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.z = math.sin(yaw / 2.0)
        tf.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_broadcaster.sendTransform(tf)

        self.collision_pub.publish(Bool(data=collided))
        self.true_pose_pub.publish(Float64MultiArray(data=[x, y, yaw]))
        true_clearance = meta.get("true_clearance_m")
        if true_clearance is not None:
            self.true_clearance_pub.publish(Float64(data=float(true_clearance)))
        if collided:
            self.get_logger().warn(
                f"navmesh contact at ({x:+.2f}, {y:+.2f}) with v={v:+.2f} w={w:+.2f}"
            )

        self._tick_n += 1
        if self._tick_n == 1 or self._tick_n % 150 == 0:
            self.get_logger().info(
                f"  step {self._tick_n}: pose=({x:+.2f}, {y:+.2f}, "
                f"{math.degrees(yaw):+.1f} deg)  cmd=({v:+.2f}, {w:+.2f})"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HabitatBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
