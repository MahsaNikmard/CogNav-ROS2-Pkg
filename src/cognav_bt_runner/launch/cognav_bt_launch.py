#!/usr/bin/env python3
"""Bring up the platform adapter, perception, the behaviour tree and the mission node.

    ros2 launch cognav_bt_runner cognav_bt_launch.py \\
        scene:=hm3d_large enable_cmd_vel:=true tree_log_ranges:=full

`depth_scale` is looked up by scene alias in cognav_perception's
config/depth_scale.yaml. A scene given as a path falls back to the platform
value, so pass `depth_scale:=<measured>` with it.
"""

import os
import re

import yaml
from ament_index_python.packages import get_package_share_directory
from cognav_perception.defaults import (
    PERCEPTION_DEFAULTS,
    PLATFORM_TOKEN,
    SCENE_TOKEN,
    depth_scale_template,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue

#: Habitat renders through its own pinhole camera; the robot uses the OAK-D.
_INTRINSICS_BY_PLATFORM = {
    "habitat": "cognav_perception/intrinsic/habitat/intrinsics.npy",
}
_INTRINSICS_FALLBACK = "cognav_perception/intrinsic/turtlebot4/intrinsics.npy"


def _yaml_params(package: str, name: str) -> dict:
    path = os.path.join(get_package_share_directory(package), "config", name)
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    except OSError:
        return {}
    return data.get("/**", {}).get("ros__parameters", {})


def _depth_scale_expression() -> PythonExpression:
    """depth_scale resolved from the measured table by scene, then platform."""
    table_path = os.path.join(
        get_package_share_directory("cognav_perception"), "config", "depth_scale.yaml")
    try:
        with open(table_path, "r", encoding="utf-8") as stream:
            table = yaml.safe_load(stream) or {}
    except OSError:
        table = {}
    substitutions = {
        SCENE_TOKEN: LaunchConfiguration("scene"),
        PLATFORM_TOKEN: LaunchConfiguration("platform"),
    }
    pattern = "(" + "|".join(re.escape(token) for token in substitutions) + ")"
    template = depth_scale_template(table)
    return PythonExpression(
        [substitutions.get(part, part) for part in re.split(pattern, template) if part])


def _per_platform(mapping: dict, fallback) -> PythonExpression:
    return PythonExpression(
        [repr(mapping), ".get('", LaunchConfiguration("platform"), "', ", repr(fallback), ")"])


def generate_launch_description() -> LaunchDescription:
    platform_defaults = _yaml_params("cognav_platform_adapter", "params.yml")
    platform_adapter_launch = os.path.join(
        get_package_share_directory("cognav_platform_adapter"),
        "launch", "platform_adapter_launch.py")
    perception_params_default = os.path.join(
        get_package_share_directory("cognav_perception"), "config", "params.yml")
    runner_params_default = os.path.join(
        get_package_share_directory("cognav_bt_runner"), "config", "params.yml")

    args = [
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument(
            "platform", default_value="habitat",
            description="habitat or turtlebot."),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="true for Habitat, false on the robot. Must match the "
                        "platform, or every frame is judged stale."),
        DeclareLaunchArgument("launch_perception", default_value="true"),
        DeclareLaunchArgument(
            "scene", default_value="",
            description="Habitat scene alias or .glb path. Empty selects hm3d_large."),

        DeclareLaunchArgument(
            "enable_cmd_vel", default_value="false",
            description="Allow the tree to publish /cmd_vel."),
        DeclareLaunchArgument(
            "mission_autostart", default_value="true",
            description="Send the mission goal at startup instead of waiting "
                        "for an /execute_mission goal."),
        DeclareLaunchArgument(
            "mission_xml_path",
            default_value="cognav_mission/mission/random_explore.xml"),

        DeclareLaunchArgument(
            "free_space_source", default_value="polar",
            description="polar or mesh. mesh also needs mesh_path and start_hab."),
        DeclareLaunchArgument(
            "mesh_path", default_value="",
            description="Navmesh raster from scripts/export_navmesh.py."),
        DeclareLaunchArgument("max_v", default_value=str(0.22)),
        DeclareLaunchArgument("max_w", default_value=str(0.8)),
        DeclareLaunchArgument(
            "maneuver_clearance", default_value="0.18",
            description="Standoff added to the body radius in the fit tests."),
        DeclareLaunchArgument("seed", default_value="-1"),
        DeclareLaunchArgument(
            "trial_timeout_sec", default_value="500.0",
            description="Mission time budget."),

        DeclareLaunchArgument(
            "tree_log_path", default_value="",
            description="Tick log file. Empty writes bt_tree_<mission>.log in the log directory."),
        DeclareLaunchArgument(
            "tree_log_ranges", default_value="profile",
            description="profile, full or none. Observed-area scoring needs full."),
        DeclareLaunchArgument(
            "sync_model", default_value="interpolation",
            description="interpolation or latest: how odometry is paired with a frame."),

        DeclareLaunchArgument("perception_params", default_value=perception_params_default),
        DeclareLaunchArgument("runner_params", default_value=runner_params_default),
        DeclareLaunchArgument("perception_env", default_value="perception"),
        DeclareLaunchArgument(
            "depth_scale", default_value=_depth_scale_expression(),
            description="Multiplicative correction on DA2 depth. Measure it with "
                        "scripts/calibrate_depth_scale.sh."),
        DeclareLaunchArgument(
            "intrinsics_path",
            default_value=_per_platform(_INTRINSICS_BY_PLATFORM, _INTRINSICS_FALLBACK)),
        DeclareLaunchArgument("fov_deg", default_value=str(PERCEPTION_DEFAULTS.fov_deg)),
        DeclareLaunchArgument("num_bins", default_value=str(PERCEPTION_DEFAULTS.num_bins)),
        DeclareLaunchArgument("perception_log_level", default_value="info"),
        DeclareLaunchArgument("bt_log_level", default_value="info"),

        DeclareLaunchArgument("robot_ns", default_value=str(platform_defaults.get("robot_ns", ""))),
        DeclareLaunchArgument("rgb_image_topic",
                              default_value=str(platform_defaults.get("rgb_image_topic", ""))),
        DeclareLaunchArgument("domain_id", default_value=str(platform_defaults.get("domain_id", ""))),
        DeclareLaunchArgument("endpoint", default_value="tcp://127.0.0.1:5561"),
        DeclareLaunchArgument("sim_hz", default_value="15.0"),
        DeclareLaunchArgument(
            "start_hab", default_value="",
            description="Episode start pose x,y,z,yaw in Habitat coordinates. "
                        "Empty samples one from the seed."),
        DeclareLaunchArgument(
            "lab_map", default_value="",
            description="map_server YAML of the laboratory, for platform:=turtlebot."),
        DeclareLaunchArgument(
            "pose_source", default_value="amcl",
            description="Where lab_bridge takes the scoring pose from: amcl or odom."),
    ]

    platform_adapter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(platform_adapter_launch),
        launch_arguments={
            "platform": LaunchConfiguration("platform"),
            "namespace": LaunchConfiguration("namespace"),
            "robot_ns": LaunchConfiguration("robot_ns"),
            "rgb_image_topic": LaunchConfiguration("rgb_image_topic"),
            "scene": LaunchConfiguration("scene"),
            "domain_id": LaunchConfiguration("domain_id"),
            "endpoint": LaunchConfiguration("endpoint"),
            "sim_hz": LaunchConfiguration("sim_hz"),
            "start_hab": LaunchConfiguration("start_hab"),
            "seed": LaunchConfiguration("seed"),
            "lab_map": LaunchConfiguration("lab_map"),
            "pose_source": LaunchConfiguration("pose_source"),
        }.items(),
    )

    # Run in the perception conda environment, where torch and DA2 live.
    # The -p overrides come after --params-file so they take precedence.
    perception_node = ExecuteProcess(
        cmd=[
            "conda", "run", "--no-capture-output", "-n",
            LaunchConfiguration("perception_env"),
            "python", "-m", "cognav_perception.cognav_perception",
            "--ros-args",
            "-r", "__node:=cognav_perception",
            "-r", ["__ns:=/", LaunchConfiguration("namespace")],
            "--params-file", LaunchConfiguration("perception_params"),
            "-p", ["depth_scale:=", LaunchConfiguration("depth_scale")],
            "-p", ["intrinsics_path:=", LaunchConfiguration("intrinsics_path")],
            "-p", ["use_sim_time:=", LaunchConfiguration("use_sim_time")],
            "-p", ["num_bins:=", LaunchConfiguration("num_bins")],
            "--log-level", LaunchConfiguration("perception_log_level"),
        ],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("launch_perception")),
    )

    bt_runner = Node(
        package="cognav_bt_runner",
        executable="bt_runner",
        namespace=LaunchConfiguration("namespace"),
        name="bt_runner",
        parameters=[
            LaunchConfiguration("runner_params"),
            {
                "sync_model": LaunchConfiguration("sync_model"),
                "mission_xml_path": LaunchConfiguration("mission_xml_path"),
                "enable_cmd_vel": ParameterValue(
                    LaunchConfiguration("enable_cmd_vel"), value_type=bool),
                "tree_log_path": LaunchConfiguration("tree_log_path"),
                "tree_log_ranges": LaunchConfiguration("tree_log_ranges"),
                "free_space_source": LaunchConfiguration("free_space_source"),
                # start_hab is shared with the platform adapter, which places
                # the robot there, so the two cannot disagree.
                "mesh_path": LaunchConfiguration("mesh_path"),
                "start_hab": LaunchConfiguration("start_hab"),
                "max_v": ParameterValue(LaunchConfiguration("max_v"), value_type=float),
                "max_w": ParameterValue(LaunchConfiguration("max_w"), value_type=float),
                "maneuver_clearance": ParameterValue(
                    LaunchConfiguration("maneuver_clearance"), value_type=float),
                "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
                "trial_timeout_sec": ParameterValue(
                    LaunchConfiguration("trial_timeout_sec"), value_type=float),
                "fov_deg": ParameterValue(LaunchConfiguration("fov_deg"), value_type=float),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("bt_log_level")],
        output="screen",
        emulate_tty=True,
    )

    mission_node = Node(
        package="cognav_mission",
        executable="mission_node.py",
        namespace=LaunchConfiguration("namespace"),
        name="mission",
        parameters=[{
            "mission_xml_path": LaunchConfiguration("mission_xml_path"),
            "autostart": ParameterValue(
                LaunchConfiguration("mission_autostart"), value_type=bool),
            "enable_cmd_vel": ParameterValue(
                LaunchConfiguration("enable_cmd_vel"), value_type=bool),
        }],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        *args,
        SetParameter("use_sim_time",
                     ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)),
        platform_adapter,
        perception_node,
        bt_runner,
        mission_node,
    ])
