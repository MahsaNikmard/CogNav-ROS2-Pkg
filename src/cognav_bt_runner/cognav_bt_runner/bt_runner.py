#!/usr/bin/env python3
"""ROS 2 node that loads a mission tree and ticks it once per perception frame."""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from rcl_interfaces.msg import ParameterEvent, ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

from cognav_bt_behaviors import BehaviorConfig, BehaviorContext
from cognav_bt_behaviors.blackboard import Blackboard
from cognav_bt_behaviors.status import Status
from cognav_bt_behaviors.timing import ExposureBudget
from cognav_mission.action import ExecuteMission
from cognav_msgs.msg import DistanceVector
from cognav_perception.defaults import PERCEPTION_DEFAULTS

from .snapshot import RuntimeBuffer
from .tree_log import TreeLogger
from .xml_tree import load_tree

QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)
QOS_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
QOS_LATCHED = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

_BEHAVIOR_DEFAULTS = BehaviorConfig()

#: Perception parameters the runner mirrors to interpret the distance vector.
PERCEPTION_PARAMETER_NAMES = (
    "num_bins",
    "fov_deg",
    "fov_padding_bins",
    "max_range",
    "padding_fill_depth_m",
)

#: Log directory bind-mounted to the host by the run scripts.
_CONTAINER_LOG_DIR = Path("/opt/ws/logs")


def _default_log_dir() -> Path:
    """The mounted log directory inside the container, else the ROS log directory."""
    if _CONTAINER_LOG_DIR.is_dir():
        return _CONTAINER_LOG_DIR
    return Path(rclpy.logging.get_logging_directory())


def _effective_seeds(root, runner_seed: int) -> str:
    """The runner seed, plus any per-leaf `seed` port overrides in the tree."""
    overrides = []
    stack = [root]
    while stack:
        node = stack.pop()
        behavior = getattr(node, "behavior", None)
        if behavior is not None and "seed" in behavior.ports:
            overrides.append(f"{node.name}={behavior.ports['seed']}")
        stack.extend(node.children)
    if not overrides:
        return str(runner_seed)
    return f"{runner_seed} (overridden: {', '.join(sorted(overrides))})"


def _resolve_share_path(path_value: str) -> Path:
    """Resolve `<package>/<relative path>` through the package share directory."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if path.parts:
        try:
            return Path(get_package_share_directory(path.parts[0])).joinpath(*path.parts[1:])
        except PackageNotFoundError:
            pass
    return Path(path_value)


class _TickStats:
    """Per-mission accounting of the timing contract.

    A frames-per-tick ratio of 1.00 means every frame produced exactly one tick.
    """

    def __init__(self, node, tree_log: TreeLogger) -> None:
        self.node = node
        self.tree_log = tree_log
        self.stale_ticks = 0
        self.reused_frames = 0
        self.zero_commands = 0
        self.starvations = 0
        self.max_age_sec = 0.0
        self.max_acted_age_sec = 0.0
        self.starved = False
        self._last_stamp = None
        self._budget_warned = False

    def observe(self, snapshot) -> None:
        if snapshot is None:
            return
        if snapshot.age_sec is not None:
            self.max_age_sec = max(self.max_age_sec, snapshot.age_sec)
            if snapshot.perception_fresh:
                self.max_acted_age_sec = max(self.max_acted_age_sec, snapshot.age_sec)
        if not snapshot.perception_fresh:
            self.stale_ticks += 1
        if snapshot.perception_stamp_sec is not None:
            if snapshot.perception_stamp_sec == self._last_stamp:
                self.reused_frames += 1
            self._last_stamp = snapshot.perception_stamp_sec

        period = snapshot.period_sec
        if (not self._budget_warned and period is not None
                and self.node.budget.is_exhausted(period)):
            self._budget_warned = True
            budget = self.node.budget
            self.node.get_logger().warn(
                f"Exposure budget exhausted: frame period {period * 1e3:.0f}ms exceeds "
                f"M/v_max={budget.total_sec:.3f}s (M={budget.safety_margin_m:.2f}m, "
                f"v_max={budget.max_v:.2f}m/s); lower max_v or raise safety_margin."
            )

    def deadline_missed(self, deadline_sec: float) -> None:
        self.zero_commands += 1
        if not self.starved:
            self.starved = True
            self.starvations += 1
            self.node.get_logger().warn(
                f"Perception starved: no frame within {deadline_sec * 1e3:.0f}ms, "
                f"commanding zero velocity until it returns"
            )

    def frame_resumed(self) -> None:
        if self.starved:
            self.starved = False
            self.node.get_logger().info("Perception resumed")

    def report(self, frames: int, ticks: int) -> None:
        ratio = (frames / ticks) if ticks else float("nan")
        budget = self.node.budget
        self.node.get_logger().info(
            f"Mission timing: {frames} frames / {ticks} ticks = {ratio:.2f} frames per tick, "
            f"reused {self.reused_frames}, stale ticks {self.stale_ticks}, "
            f"starvations {self.starvations} ({self.zero_commands} zero cmds), "
            f"max age seen {self.max_age_sec * 1e3:.0f}ms, "
            f"max age acted on {self.max_acted_age_sec * 1e3:.0f}ms "
            f"= {budget.travel_m(self.max_acted_age_sec):.3f}m at v_max "
            f"of {budget.safety_margin_m:.2f}m margin"
        )


class BtRunner(Node):
    def __init__(self) -> None:
        super().__init__("bt_runner")
        self._declare_parameters()
        self._load_parameters()

        self.callback_group = ReentrantCallbackGroup()
        self.buffer = RuntimeBuffer(
            self,
            sync_model=self.sync_model,
            budget=self.budget,
            max_stamp_skew_sec=self.max_stamp_skew_sec,
            num_bins=self.num_bins,
            fov_deg=self.fov_deg,
            fov_padding_bins=self.fov_padding_bins,
        )
        # Set by the perception callback and awaited by the mission loop, which
        # run on different executor threads.
        self.frame_event = threading.Event()
        self.frames_received = 0
        self.contacts_total = 0
        self.contacts_since_tick = 0
        self.blackboard = Blackboard()
        self._seed_mesh_keys(self.blackboard)
        self._mission_index = 0

        # Run description and ground truth from the platform. Recorded in the
        # tick log for scoring; never used for control.
        self._scene = ""
        self._navigable_area_m2 = None
        self._true_clearance = None
        self._true_pose = None
        self.create_subscription(
            String, str(self.get_parameter("scene_topic").value),
            lambda m: setattr(self, "_scene", m.data), QOS_LATCHED)
        self.create_subscription(
            Float64, str(self.get_parameter("navigable_area_topic").value),
            lambda m: setattr(self, "_navigable_area_m2", float(m.data)), QOS_LATCHED)
        self.create_subscription(
            Float64, "/platform/true_clearance_m",
            lambda m: setattr(self, "_true_clearance", float(m.data)), QOS_RELIABLE)
        self.create_subscription(
            Float64MultiArray, "/platform/true_pose",
            lambda m: setattr(self, "_true_pose",
                              tuple(float(v) for v in m.data[:3])
                              if len(m.data) >= 3 else None), QOS_RELIABLE)

        self.create_subscription(DistanceVector, self.perception_topic, self._distance_vector_cb, QOS_RELIABLE)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, QOS_SENSOR)
        self.create_subscription(Bool, self.collision_topic, self._collision_cb, QOS_RELIABLE)

        self._initial_parameter_sync_started = False
        if self.sync_perception_parameters:
            self.create_subscription(ParameterEvent, "/parameter_events", self._parameter_event_cb, QOS_RELIABLE)
            self.perception_param_client = self.create_client(
                GetParameters,
                f"{self.perception_node_name}/get_parameters",
                callback_group=self.callback_group,
            )
            self._initial_parameter_sync_timer = self.create_timer(
                1.0,
                self._try_initial_perception_parameter_sync,
                callback_group=self.callback_group,
            )

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, QOS_RELIABLE)
        self.reference_path_pub = (
            self.create_publisher(PathMsg, self.reference_path_pub_topic, QOS_RELIABLE)
            if self.publish_reference_path else None
        )

        self.action_server = ActionServer(
            self,
            ExecuteMission,
            self.runner_action_name,
            execute_callback=self._execute_tree,
            cancel_callback=self._cancel_cb,
            callback_group=self.callback_group,
        )
        self.get_logger().info(
            f"bt_runner ready: perception={self.perception_topic} odom={self.odom_topic} "
            f"sync_model={self.sync_model} internal_action={self.runner_action_name}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("perception_topic", "/perception/distance_vector")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("collision_topic", "/collision")
        self.declare_parameter("scene_topic", "/platform/scene")
        self.declare_parameter("navigable_area_topic", "/platform/navigable_area_m2")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        # The committed waypoints, for viewers and recordings only. The planner
        # hands the path to the driver on the blackboard within one tick.
        self.declare_parameter("publish_reference_path", True)
        self.declare_parameter("reference_path_pub_topic", "/planner/reference_path")
        self.declare_parameter("reference_path_frame_id", "odom")
        # Drawn above the perception markers, which sit at z 0.10.
        self.declare_parameter("reference_path_z", 0.25)
        self.declare_parameter("mission_xml_path", "cognav_mission/mission/random_explore.xml")
        # Mesh arm inputs, seeded on the blackboard: the navmesh raster of the
        # scene and the Habitat pose at which odometry was zeroed.
        self.declare_parameter("mesh_path", "")
        self.declare_parameter("start_hab", "")
        self.declare_parameter("runner_action_name", "/bt_runner/execute_tree")
        # Mirrors of perception parameters, refreshed from the perception node
        # when sync_perception_parameters is true.
        self.declare_parameter("perception_node_name", "/cognav_perception")
        self.declare_parameter("sync_perception_parameters", True)
        for name in PERCEPTION_PARAMETER_NAMES:
            self.declare_parameter(name, getattr(PERCEPTION_DEFAULTS, name))
        # Which odometry sample pairs with a frame: "interpolation" to the frame
        # stamp, or "latest".
        self.declare_parameter("sync_model", "interpolation")
        self.declare_parameter("enable_cmd_vel", False)
        # Lower bound on the freshness deadline derived in timing.ExposureBudget.
        self.declare_parameter("deadline_floor_sec", 0.10)
        # Period of the zero command repeated while perception is starved.
        self.declare_parameter("zero_command_period_sec", 0.05)
        self.declare_parameter("tree_log_enabled", True)
        # Empty writes bt_tree_<mission>.log into the log directory.
        self.declare_parameter("tree_log_path", "")
        self.declare_parameter("tree_log_every_n", 1)
        # profile: one-line scene profile; full: also every real bin's range,
        # which observed-area scoring needs; none: no range detail.
        self.declare_parameter("tree_log_ranges", "profile")
        self.declare_parameter("max_stamp_skew_sec", 0.5)
        for field in fields(BehaviorConfig):
            self.declare_parameter(field.name, getattr(_BEHAVIOR_DEFAULTS, field.name))

    def _load_parameters(self) -> None:
        get = self.get_parameter
        self.perception_topic = str(get("perception_topic").value)
        self.odom_topic = str(get("odom_topic").value)
        self.collision_topic = str(get("collision_topic").value)
        self.cmd_vel_topic = str(get("cmd_vel_topic").value)
        self.publish_reference_path = bool(get("publish_reference_path").value)
        self.reference_path_pub_topic = str(get("reference_path_pub_topic").value)
        self.reference_path_frame_id = str(get("reference_path_frame_id").value)
        self.reference_path_z = float(get("reference_path_z").value)
        self.mission_xml_path = str(get("mission_xml_path").value)
        self.mesh_path = str(get("mesh_path").value)
        self.start_hab = str(get("start_hab").value)
        self.runner_action_name = str(get("runner_action_name").value)
        self.perception_node_name = str(get("perception_node_name").value)
        self.sync_perception_parameters = bool(get("sync_perception_parameters").value)
        self.num_bins = int(get("num_bins").value)
        self.fov_deg = float(get("fov_deg").value)
        self.fov_padding_bins = int(get("fov_padding_bins").value)
        self.max_range = float(get("max_range").value)
        self.padding_fill_depth_m = float(get("padding_fill_depth_m").value)
        self.sync_model = str(get("sync_model").value)
        if self.sync_model not in ("latest", "interpolation"):
            raise ValueError("sync_model must be 'latest' or 'interpolation'")
        self.enable_cmd_vel = bool(get("enable_cmd_vel").value)
        self.max_stamp_skew_sec = float(get("max_stamp_skew_sec").value)
        self.zero_command_period_sec = float(get("zero_command_period_sec").value)
        self.tree_log_enabled = bool(get("tree_log_enabled").value)
        self.tree_log_path = str(get("tree_log_path").value)
        self.tree_log_every_n = int(get("tree_log_every_n").value)
        self.tree_log_ranges = str(get("tree_log_ranges").value)
        self.budget = ExposureBudget(
            safety_margin_m=float(get("safety_margin").value),
            max_v=float(get("max_v").value),
            deadline_floor_sec=float(get("deadline_floor_sec").value),
        )
        hints = get_type_hints(BehaviorConfig)
        self.behavior_config = BehaviorConfig(**{
            f.name: hints[f.name](get(f.name).value) for f in fields(BehaviorConfig)
        })

    def _perception_param(self, name: str, timeout_sec: float = 1.0):
        """Read one parameter from the perception node, or None if unavailable.

        Called from the mission thread. The future is polled rather than spun
        with `rclpy.spin_until_future_complete`, which would detach this node
        from the main executor and stop its subscriptions.
        """
        client = None
        try:
            client = self.create_client(
                GetParameters, f"{self.perception_node_name}/get_parameters")
            if not client.wait_for_service(timeout_sec=timeout_sec):
                return None
            req = GetParameters.Request()
            req.names = [name]
            future = client.call_async(req)
            deadline = time.monotonic() + timeout_sec
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not future.done():
                future.cancel()
                return None
            result = future.result()
            if result is None or not result.values:
                return None
            return self._parameter_value_to_python(result.values[0])
        except Exception:
            return None
        finally:
            if client is not None:
                try:
                    self.destroy_client(client)
                except Exception:
                    pass

    def _distance_vector_cb(self, msg: DistanceVector) -> None:
        self.buffer.set_distance_vector(msg)
        self.frames_received += 1
        self.frame_event.set()

    def _collision_cb(self, msg: Bool) -> None:
        if msg.data:
            self.contacts_total += 1
            self.contacts_since_tick += 1

    def _odom_cb(self, msg: Odometry) -> None:
        self.buffer.set_odom(msg)

    def _parameter_event_cb(self, event: ParameterEvent) -> None:
        if event.node != self.perception_node_name:
            return
        updated = {
            parameter.name: self._parameter_value_to_python(parameter.value)
            for parameter in [*event.new_parameters, *event.changed_parameters]
            if parameter.name in PERCEPTION_PARAMETER_NAMES
        }
        self._apply_perception_parameter_updates(updated)

    def _try_initial_perception_parameter_sync(self) -> None:
        if self._initial_parameter_sync_started:
            return
        if not self.perception_param_client.service_is_ready():
            self.perception_param_client.wait_for_service(timeout_sec=0.0)
            return
        self._initial_parameter_sync_started = True
        request = GetParameters.Request()
        request.names = list(PERCEPTION_PARAMETER_NAMES)
        future = self.perception_param_client.call_async(request)
        future.add_done_callback(self._initial_perception_parameters_cb)

    def _initial_perception_parameters_cb(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f"initial perception parameter sync failed: {exc}")
            self._initial_parameter_sync_started = False
            return
        updated = {
            name: self._parameter_value_to_python(value)
            for name, value in zip(PERCEPTION_PARAMETER_NAMES, response.values)
        }
        self._apply_perception_parameter_updates(updated)
        self._initial_parameter_sync_timer.cancel()

    def _apply_perception_parameter_updates(self, updated: dict) -> None:
        updated = {name: value for name, value in updated.items() if value is not None}
        if not updated:
            return
        for name, value in updated.items():
            setattr(self, name, value)
        if {"num_bins", "fov_deg", "fov_padding_bins"} & set(updated):
            self.buffer.set_geometry(
                num_bins=self.num_bins,
                fov_deg=self.fov_deg,
                fov_padding_bins=self.fov_padding_bins,
            )
        self.get_logger().info(
            "synced perception parameters: "
            f"num_bins={self.num_bins} fov_deg={self.fov_deg} "
            f"fov_padding_bins={self.fov_padding_bins} max_range={self.max_range} "
            f"padding_fill_depth_m={self.padding_fill_depth_m}"
        )

    @staticmethod
    def _parameter_value_to_python(value):
        if value.type == ParameterType.PARAMETER_INTEGER:
            return int(value.integer_value)
        if value.type == ParameterType.PARAMETER_DOUBLE:
            return float(value.double_value)
        if value.type == ParameterType.PARAMETER_STRING:
            return str(value.string_value)
        if value.type == ParameterType.PARAMETER_BOOL:
            return bool(value.bool_value)
        return None

    def _cancel_cb(self, goal_handle):
        del goal_handle
        return CancelResponse.ACCEPT

    def _execute_tree(self, goal_handle):
        result = self._run_mission(
            goal_handle.request,
            feedback_cb=lambda feedback: goal_handle.publish_feedback(feedback),
            cancel_cb=lambda: self._goal_cancel_requested(goal_handle),
        )
        self._set_terminal_goal_state(goal_handle, result)
        return result

    def _run_mission(self, request, feedback_cb, cancel_cb):
        xml_text = request.mission_xml or self._read_mission_xml(
            request.mission_xml_path or self.mission_xml_path)
        tree = load_tree(xml_text)
        self.blackboard = Blackboard()
        self._seed_mesh_keys(self.blackboard)
        context = BehaviorContext(
            self,
            self.cmd_vel_pub,
            self.behavior_config,
            enable_cmd_vel=bool(request.enable_cmd_vel or self.enable_cmd_vel),
        )
        seeds = _effective_seeds(tree, self.behavior_config.seed)
        self.get_logger().info(
            f"Mission started: sync_model={self.sync_model} "
            f"enable_cmd_vel={context.enable_cmd_vel} seed={seeds} "
            f"budget={self.budget.total_sec:.3f}s (M={self.budget.safety_margin_m:.2f}m / "
            f"v_max={self.budget.max_v:.2f}m/s)"
        )

        self._mission_index += 1
        tree_log = self._open_tree_log(self._mission_index)
        tree_log.write_preamble({
            "scene": self._scene,
            # Full precision: this is the coverage denominator.
            "navigable_area_m2": (repr(float(self._navigable_area_m2))
                                  if self._navigable_area_m2 else None),
            "depth_scale": self._perception_param("depth_scale"),
            "num_bins": self.num_bins,
            "fov_deg": self.fov_deg,
            "max_v": self.behavior_config.max_v,
            "safety_margin": self.behavior_config.safety_margin,
            "mission": os.path.basename(str(request.mission_xml_path or self.mission_xml_path)),
            "seed": seeds,
        })
        mission_start = time.monotonic()
        context.tick_id = 0
        stats = _TickStats(self, tree_log)
        # Discard a frame that arrived before the mission, so the first tick
        # acts on a frame this mission waited for.
        self.frame_event.clear()
        frames_at_start = self.frames_received
        tick_count = 0
        canceled = False
        while rclpy.ok() and self.context.ok():
            if cancel_cb():
                canceled = True
                break
            if not self._await_frame(context, stats):
                continue

            snapshot = self.buffer.snapshot()
            stats.observe(snapshot)
            context.tick_id = tick_count + 1
            status = tree.tick(snapshot, self.blackboard, context)
            tick_count += 1
            contacts = self.contacts_since_tick
            self.contacts_since_tick = 0
            self.blackboard.set("contacts_this_tick", contacts)
            self.blackboard.set("contacts_total", self.contacts_total)
            self.blackboard.set("true_clearance", self._true_clearance)
            self.blackboard.set("true_pose", self._true_pose)
            tree_log.write_tick(
                tree, context.tick_id, time.monotonic() - mission_start,
                snapshot, self.blackboard,
                self.behavior_config.safety_margin,
                self.behavior_config.safety_margin + self.behavior_config.warning_margin,
            )
            self._publish_reference_path(snapshot)
            if feedback_cb is not None:
                feedback = ExecuteMission.Feedback()
                feedback.current_state = status.value
                feedback.tick_count = tick_count
                try:
                    feedback_cb(feedback)
                except Exception as exc:
                    if self.context.ok() and "publisher is invalid" not in str(exc):
                        raise
                    break
            if status == Status.SUCCESS and self.blackboard.get("mission_complete", False):
                break

        if self.context.ok():
            context.publish_twist(0.0, 0.0, self.blackboard)
        stats.report(self.frames_received - frames_at_start, tick_count)
        if self.blackboard.get("mission_complete", False):
            outcome = "complete"
        elif canceled:
            outcome = "canceled"
        elif not (rclpy.ok() and self.context.ok()):
            outcome = "shutdown"
        else:
            outcome = "aborted"
        tree_log.close(outcome,
                       ticks=tick_count,
                       elapsed_s=f"{time.monotonic() - mission_start:.1f}",
                       contacts=self.contacts_total)

        shutdown_requested = not rclpy.ok() or not self.context.ok()
        result = ExecuteMission.Result()
        result.success = not canceled and not shutdown_requested
        if canceled:
            result.message = "mission canceled"
        elif shutdown_requested:
            result.message = "mission interrupted by shutdown"
        else:
            result.message = "mission stopped"
        result.tick_count = tick_count
        return result

    def _seed_mesh_keys(self, blackboard) -> None:
        """Seed the mesh arm inputs; empty on the other arms."""
        blackboard.set("mesh_path", self.mesh_path)
        blackboard.set("start_hab", self.start_hab)

    def _open_tree_log(self, mission_index: int) -> TreeLogger:
        if not self.tree_log_enabled:
            return TreeLogger(None)
        if self.tree_log_path:
            path = Path(self.tree_log_path)
        else:
            path = _default_log_dir() / f"bt_tree_{mission_index}.log"
        try:
            return TreeLogger(path, self.tree_log_every_n, self.get_logger(),
                              self.tree_log_ranges)
        except OSError as exc:
            self.get_logger().warn(f"BT tick log disabled, cannot open {path}: {exc}")
            return TreeLogger(None)

    def _publish_reference_path(self, snapshot) -> None:
        """Publish the robot pose followed by the committed waypoints.

        Stamped with the source frame's capture time and published every tick,
        empty or not.
        """
        if self.reference_path_pub is None or snapshot is None:
            return
        msg = PathMsg()
        msg.header.frame_id = self.reference_path_frame_id
        if snapshot.perception_msg is not None:
            msg.header.stamp = snapshot.perception_msg.header.stamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()

        points = []
        if snapshot.odom is not None:
            points.append((snapshot.odom[0], snapshot.odom[1], snapshot.odom[2]))
        previous = points[0][:2] if points else None
        for x, y in self.blackboard.get("reference_path", []):
            heading = points[-1][2] if points else 0.0
            if previous is not None:
                heading = math.atan2(y - previous[1], x - previous[0])
            points.append((float(x), float(y), heading))
            previous = (x, y)

        for x, y, yaw in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = self.reference_path_z
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            msg.poses.append(pose)
        self.reference_path_pub.publish(msg)

    def _await_frame(self, context, stats) -> bool:
        """Wait for the next frame; on timeout command zero and return False.

        The wait is bounded by the freshness deadline, so an outage brakes the
        robot instead of leaving the last command in force. While starved, the
        zero command is repeated every `zero_command_period_sec`.
        """
        deadline = self.buffer.deadline_sec()
        timeout = min(deadline, self.zero_command_period_sec) if stats.starved else deadline
        if self.frame_event.wait(timeout=timeout):
            self.frame_event.clear()
            stats.frame_resumed()
            return True

        context.publish_twist(0.0, 0.0, self.blackboard)
        if not stats.starved:
            stats.tree_log.write_note(
                f"perception starved: no frame within {deadline * 1e3:.0f}ms, commanding zero"
            )
        stats.deadline_missed(deadline)
        return False

    def _goal_cancel_requested(self, goal_handle) -> bool:
        try:
            return bool(goal_handle.is_cancel_requested)
        except Exception:
            return False

    def _set_terminal_goal_state(self, goal_handle, result: ExecuteMission.Result) -> None:
        try:
            if self._goal_cancel_requested(goal_handle):
                goal_handle.canceled()
            elif result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
        except Exception as exc:
            if self.context.ok():
                self.get_logger().warn(f"failed to set terminal mission goal state: {exc}")

    def _read_mission_xml(self, mission_xml_path: str) -> str:
        return _resolve_share_path(mission_xml_path).read_text(encoding="utf-8")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BtRunner()
    executor = MultiThreadedExecutor()
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
