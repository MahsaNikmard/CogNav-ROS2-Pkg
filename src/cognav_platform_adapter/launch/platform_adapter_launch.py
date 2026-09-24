#!/usr/bin/env python3
"""Start the platform side: the Habitat server and bridge, or the TurtleBot4 relay."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml


def _platform_defaults() -> dict:
    params_path = os.path.join(
        get_package_share_directory("cognav_platform_adapter"),
        "config",
        "params.yml",
    )
    try:
        with open(params_path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    except OSError:
        return {}
    return data.get("/**", {}).get("ros__parameters", {})


#: Scene aliases, relative to the mounted habitat_data directory. Any other
#: value is passed through as a path.
HABITAT_SCENE_ALIASES = {
    "hm3d_large": "versioned_data/hm3d-0.2/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.glb",
    "hm3d_medium": "versioned_data/hm3d-0.2/hm3d/example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.glb",
    "hm3d_small": "versioned_data/hm3d-0.2/hm3d/example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.glb",
    "van_gogh_room": "versioned_data/habitat_test_scenes/van-gogh-room.glb",
    "apartment_1": "versioned_data/habitat_test_scenes/apartment_1.glb",
    "skokloster_castle": "versioned_data/habitat_test_scenes/skokloster-castle.glb",
}

#: Must match `platform_default_scene` in cognav_perception/config/depth_scale.yaml.
DEFAULT_HABITAT_SCENE = "hm3d_large"


def _habitat_scene_path(share_dir: str, scene: str) -> str:
    """Resolve an alias to a packaged scene, or pass a real path through."""
    if scene in HABITAT_SCENE_ALIASES:
        return os.path.join(
            share_dir, "simulation", "habitat_data", *HABITAT_SCENE_ALIASES[scene].split("/")
        )
    return scene


def _setup_platform(context, *args, **kwargs):
    share_dir = get_package_share_directory("cognav_platform_adapter")
    habitat_server = os.path.join(share_dir, "scripts", "habitat_sim_server.py")
    default_habitat_scene = _habitat_scene_path(share_dir, DEFAULT_HABITAT_SCENE)

    platform = LaunchConfiguration("platform").perform(context).strip().lower()
    namespace = LaunchConfiguration("namespace")
    robot_ns = LaunchConfiguration("robot_ns").perform(context).strip()
    rgb_image_topic = LaunchConfiguration("rgb_image_topic").perform(context).strip()
    scene = LaunchConfiguration("scene").perform(context).strip()
    domain_id = LaunchConfiguration("domain_id").perform(context).strip()

    actions = [LogInfo(msg=[f"[platform_adapter] platform={platform}"])]
    if domain_id:
        actions.append(SetEnvironmentVariable(name="ROS_DOMAIN_ID", value=domain_id))

    if platform == "habitat":
        habitat_scene = _habitat_scene_path(share_dir, scene) if scene else default_habitat_scene
        habitat_server_process = ExecuteProcess(
            cmd=[
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                LaunchConfiguration("habitat_env"),
                "python",
                habitat_server,
                "--scene",
                habitat_scene,
                "--endpoint",
                LaunchConfiguration("endpoint"),
                "--hfov-deg",
                LaunchConfiguration("hfov_deg"),
                "--width",
                LaunchConfiguration("width"),
                "--height",
                LaunchConfiguration("height"),
                "--sensor-height",
                LaunchConfiguration("sensor_height"),
                "--agent-radius",
                LaunchConfiguration("agent_radius"),
                "--agent-max-climb",
                LaunchConfiguration("agent_max_climb"),
                "--agent-height",
                LaunchConfiguration("agent_height"),
                "--seed",
                LaunchConfiguration("seed"),
                # One token, so argparse accepts a pose starting with '-'.
                ["--start-hab=", LaunchConfiguration("start_hab")],
            ],
            output="screen",
            emulate_tty=True,
        )
        actions.extend(
            [
                habitat_server_process,
                # Stop the whole launch if the simulator exits.
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=habitat_server_process,
                        on_exit=[
                            LogInfo(msg="[platform_adapter] habitat_sim_server exited, shutting down"),
                            Shutdown(reason="habitat_sim_server exited"),
                        ],
                    )
                ),
                Node(
                    package="cognav_platform_adapter",
                    executable="habitat_bridge",
                    namespace=namespace,
                    name="platform_bridge",
                    parameters=[
                        {
                            "endpoint": LaunchConfiguration("endpoint"),
                            "sim_hz": LaunchConfiguration("sim_hz"),
                            "lockstep": ParameterValue(
                                LaunchConfiguration("lockstep"),
                                value_type=bool),
                            "use_sim_time": False,
                        }
                    ],
                    output="screen",
                    emulate_tty=True,
                ),
            ]
        )
        return actions

    if platform != "turtlebot":
        raise RuntimeError("platform must be one of: turtlebot, habitat")

    # With a lab map, also bring up map_server, AMCL and lab_bridge for the
    # scoring topics. Without one, the robot is only relayed.
    lab_map = LaunchConfiguration("lab_map").perform(context).strip()
    if platform == "turtlebot" and lab_map:
        if not os.path.isfile(lab_map):
            raise RuntimeError(f"lab_map does not exist: {lab_map!r}")
        amcl_params = {
            "use_sim_time": False,
            # Re-seeded each episode, since the robot is placed by hand.
            "set_initial_pose": True,
            "robot_base_frame": "base_link",
            "odom_frame_id": "odom",
            "global_frame_id": "map",
            "scan_topic": "/scan",
        }
        actions.append(
            Node(package="nav2_map_server", executable="map_server",
                 name="map_server", namespace=namespace, output="screen",
                 parameters=[{"use_sim_time": False, "yaml_filename": lab_map}])
        )
        actions.append(
            Node(package="nav2_amcl", executable="amcl", name="amcl",
                 namespace=namespace, output="screen", parameters=[amcl_params])
        )
        actions.append(
            Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                 name="lifecycle_manager_lab", namespace=namespace, output="screen",
                 parameters=[{"use_sim_time": False, "autostart": True,
                              "node_names": ["map_server", "amcl"]}])
        )
        actions.append(
            Node(package="cognav_platform_adapter", executable="lab_bridge",
                 name="lab_bridge", namespace=namespace, output="screen",
                 parameters=[{
                     "map_yaml": lab_map,
                     "robot_radius": ParameterValue(
                         LaunchConfiguration("robot_radius"), value_type=float),
                     "pose_source": LaunchConfiguration("pose_source"),
                 }])
        )

    actions.append(
        Node(
            package="cognav_platform_adapter",
            executable="namespace_bridge",
            namespace=namespace,
            name="platform_bridge",
            parameters=[
                {
                    "platform": platform,
                    "robot_ns": robot_ns,
                    "rgb_image_topic": rgb_image_topic,
                }
            ],
            output="screen",
            emulate_tty=True,
        )
    )
    return actions


def generate_launch_description() -> LaunchDescription:
    defaults = _platform_defaults()
    args = [
        DeclareLaunchArgument(
            "platform",
            default_value=str(defaults.get("platform", "turtlebot")),
            description="Platform backend: turtlebot or habitat.",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="ROS namespace for the adapter nodes. Topics stay absolute.",
        ),
        DeclareLaunchArgument(
            "robot_ns",
            default_value=str(defaults.get("robot_ns", "")),
            description="Robot namespace on the TurtleBot4. Empty selects the default.",
        ),
        DeclareLaunchArgument(
            "rgb_image_topic",
            default_value=str(defaults.get("rgb_image_topic", "")),
            description="Robot RGB topic relayed to /oakd/rgb/preview/image_raw, "
                        "relative to robot_ns.",
        ),
        DeclareLaunchArgument(
            "domain_id",
            default_value=str(defaults.get("domain_id", "")),
            description="Optional ROS_DOMAIN_ID applied to launched platform processes.",
        ),
        DeclareLaunchArgument(
            "endpoint",
            default_value=str(defaults.get("endpoint", "tcp://127.0.0.1:5561")),
            description="ZMQ endpoint between Habitat server and ROS bridge.",
        ),
        DeclareLaunchArgument(
            "sim_hz",
            default_value=str(defaults.get("sim_hz", "15.0")),
            description="Habitat lockstep rate. Simulation advances 1/sim_hz per bridge tick.",
        ),
        DeclareLaunchArgument(
            "lockstep",
            default_value="true",
            description="Advance the simulator one step per command, which makes "
                        "an episode deterministic. false lets the bridge free-run.",
        ),
        DeclareLaunchArgument(
            "lab_map",
            default_value="",
            description="map_server YAML of the laboratory. With platform:=turtlebot "
                        "it adds map_server, AMCL and lab_bridge.",
        ),
        DeclareLaunchArgument(
            "robot_radius",
            default_value="0.18",
            description="Body radius for the lab clearance field and coverage denominator.",
        ),
        DeclareLaunchArgument(
            "pose_source",
            default_value="amcl",
            description="Where lab_bridge takes the scoring pose from: amcl or odom.",
        ),
        DeclareLaunchArgument(
            "scene",
            default_value="",
            description=(
                "Habitat scene: one of "
                f"{', '.join(sorted(HABITAT_SCENE_ALIASES))}, or a .glb path. "
                f"Empty selects {DEFAULT_HABITAT_SCENE}."
            ),
        ),
        DeclareLaunchArgument("seed", default_value="7", description="Habitat random start seed."),
        DeclareLaunchArgument(
            "start_hab",
            default_value="",
            description="Start pose 'x,y,z[,yaw]' in Habitat coordinates. Empty draws one from seed.",
        ),
        DeclareLaunchArgument("agent_radius", default_value="0.18", description="Habitat navmesh agent radius."),
        DeclareLaunchArgument("agent_max_climb", default_value="0.05",
                              description="Habitat navmesh step allowance; 0.05 m keeps storeys apart."),
        DeclareLaunchArgument("agent_height", default_value="0.40", description="Habitat navmesh agent height."),
        DeclareLaunchArgument("sensor_height", default_value="0.32", description="Habitat camera height above base."),
        DeclareLaunchArgument(
            "hfov_deg",
            default_value="89.0",
            description="Habitat camera horizontal FOV. Keep aligned with cognav_perception fov_deg.",
        ),
        DeclareLaunchArgument("width", default_value="250", description="Habitat RGB render width."),
        DeclareLaunchArgument("height", default_value="250", description="Habitat RGB render height."),
        DeclareLaunchArgument(
            "habitat_env",
            default_value="habitat",
            description="Conda environment containing habitat-sim.",
        ),
    ]

    return LaunchDescription([*args, OpaqueFunction(function=_setup_platform)])
