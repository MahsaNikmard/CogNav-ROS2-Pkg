#!/usr/bin/env python3
"""Public /execute_mission action that forwards missions to the bt_runner action."""

from __future__ import annotations

import threading
import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from cognav_mission.action import ExecuteMission


class MissionNode(Node):
    """Accepts one mission at a time and relays it to `runner_action_name`."""

    def __init__(self) -> None:
        super().__init__("mission")
        self._declare_parameters()
        self._load_parameters()

        self.callback_group = ReentrantCallbackGroup()
        self.runner_client = ActionClient(
            self,
            ExecuteMission,
            self.runner_action_name,
            callback_group=self.callback_group,
        )
        self.action_server = ActionServer(
            self,
            ExecuteMission,
            self.public_action_name,
            execute_callback=self._execute_mission,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self.callback_group,
        )
        self._active_goal_lock = threading.Lock()
        self._active_goal = None
        self._autostart_started = False

        if self.autostart:
            self.create_timer(0.5, self._autostart_once, callback_group=self.callback_group)

        self.get_logger().info(
            f"mission node ready: public_action={self.public_action_name} "
            f"runner_action={self.runner_action_name}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("public_action_name", "/execute_mission")
        self.declare_parameter("runner_action_name", "/bt_runner/execute_tree")
        self.declare_parameter("mission_xml_path", "cognav_mission/mission/random_explore.xml")
        self.declare_parameter("autostart", True)
        self.declare_parameter("enable_cmd_vel", False)

    def _load_parameters(self) -> None:
        self.public_action_name = str(self.get_parameter("public_action_name").value)
        self.runner_action_name = str(self.get_parameter("runner_action_name").value)
        self.mission_xml_path = str(self.get_parameter("mission_xml_path").value)
        self.autostart = bool(self.get_parameter("autostart").value)
        self.enable_cmd_vel = bool(self.get_parameter("enable_cmd_vel").value)

    def _goal_cb(self, goal_request):
        del goal_request
        with self._active_goal_lock:
            if self._active_goal is not None:
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        del goal_handle
        return CancelResponse.ACCEPT

    def _autostart_once(self) -> None:
        if self._autostart_started:
            return
        self._autostart_started = True
        threading.Thread(target=self._run_autostart_mission, daemon=True).start()

    def _run_autostart_mission(self) -> None:
        request = ExecuteMission.Goal()
        request.mission_xml_path = self.mission_xml_path
        request.enable_cmd_vel = self.enable_cmd_vel
        self._send_to_runner(
            request,
            feedback_cb=None,
            cancel_cb=lambda: False,
        )

    def _execute_mission(self, goal_handle):
        with self._active_goal_lock:
            self._active_goal = goal_handle
        try:
            request = goal_handle.request
            if not request.mission_xml and not request.mission_xml_path:
                request.mission_xml_path = self.mission_xml_path
            request.enable_cmd_vel = bool(request.enable_cmd_vel or self.enable_cmd_vel)

            result = self._send_to_runner(
                request,
                feedback_cb=lambda feedback: goal_handle.publish_feedback(feedback),
                cancel_cb=lambda: goal_handle.is_cancel_requested,
            )

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            elif result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result
        finally:
            with self._active_goal_lock:
                self._active_goal = None

    def _send_to_runner(self, request, feedback_cb, cancel_cb):
        if not self.runner_client.wait_for_server(timeout_sec=10.0):
            result = ExecuteMission.Result()
            result.success = False
            result.message = f"runner action unavailable: {self.runner_action_name}"
            result.tick_count = 0
            return result

        send_done = threading.Event()
        send_state = {}

        def _on_send_done(future):
            send_state["goal_handle"] = future.result()
            send_done.set()

        send_future = self.runner_client.send_goal_async(
            request,
            feedback_callback=lambda msg: feedback_cb(msg.feedback) if feedback_cb else None,
        )
        send_future.add_done_callback(_on_send_done)
        while not send_done.is_set() and rclpy.ok():
            time.sleep(0.01)

        runner_goal = send_state.get("goal_handle")
        if runner_goal is None or not runner_goal.accepted:
            result = ExecuteMission.Result()
            result.success = False
            result.message = "runner rejected mission"
            result.tick_count = 0
            return result

        result_done = threading.Event()
        result_state = {}
        result_future = runner_goal.get_result_async()
        result_future.add_done_callback(lambda future: (result_state.update(result=future.result()), result_done.set()))

        cancel_sent = False
        while not result_done.is_set() and rclpy.ok():
            if cancel_cb() and not cancel_sent:
                runner_goal.cancel_goal_async()
                cancel_sent = True
            time.sleep(0.02)

        if "result" not in result_state:
            result = ExecuteMission.Result()
            result.success = False
            result.message = "runner result unavailable"
            result.tick_count = 0
            return result
        return result_state["result"].result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionNode()
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
