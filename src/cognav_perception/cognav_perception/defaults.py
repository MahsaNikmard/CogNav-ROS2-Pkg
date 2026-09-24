"""Default value, valid range and description of every cognav_perception parameter.

The node declares its ROS parameters from `PerceptionConfig`, and
config/params.yml holds deployment overrides only. No torch, cv2 or rclpy
imports, so launch files and the ROS environment can read it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class PerceptionConfig:
    """Field names are the ROS parameter names."""

    image_topic: str = "/oakd/rgb/preview/image_raw"
    perception_output_topic: str = "/perception/distance_vector"
    #: `<package>/<relative path>` resolves through the package share
    #: directory. The launch file selects the TurtleBot4 file on the robot.
    intrinsics_path: str = "cognav_perception/intrinsic/habitat/intrinsics.npy"

    encoder: str = "vits"
    #: Must be the metric checkpoint: the model head is Sigmoid() * max_depth,
    #: and relative weights load without error but saturate it.
    weights_path: str = (
        "cognav_perception/model_weights/depth_anything_v2_metric_hypersim_vits.pth"
    )
    max_depth: float = 20.0

    num_bins: int = 72
    fov_deg: float = 89.0
    max_range: float = 5.0
    #: Ceiling exclusion. The camera frame is y-down, so the ceiling is y < 0.
    vertical_margin: float = 0.10
    #: Floor exclusion. The camera sits about 0.32 m above the floor.
    floor_margin: float = 0.20

    #: Multiplicative correction on DA2 depth for a standalone run. Launches
    #: take the measured per-scene value from config/depth_scale.yaml; this is
    #: the Habitat default scene's entry there.
    depth_scale: float = 0.48

    #: Range below which a bin counts toward the danger flag and RViz colour.
    safety_threshold: float = 2.0
    #: Per-bin median window, in frames, over observed readings only.
    temporal_window: int = 5

    #: Virtual bins appended on each side of the real field of view.
    fov_padding_bins: int = 10
    #: Depth given to the padding bins. Must exceed robot_radius + safety_margin
    #: or a straight-ahead command is refused by its own FOV edge.
    padding_fill_depth_m: float = 1.5

    frame_id: str = "base_link"
    marker_lifetime_sec: float = 0.5
    show_padding_bins: bool = True
    publish_depth_image: bool = True

    #: Inference timer period. Images are also processed as they arrive, so
    #: this is a fallback cadence rather than a throttle.
    timer_period_sec: float = 1.0 / 15.0


PERCEPTION_DEFAULTS = PerceptionConfig()


#: Valid ranges of the numeric parameters, used for the ROS descriptors.
LIMITS: Dict[str, Tuple[float, float]] = {
    "max_depth": (0.01, 1000.0),
    "num_bins": (1, 4096),
    "fov_deg": (1.0, 360.0),
    "max_range": (0.01, 1000.0),
    "vertical_margin": (0.0, 10.0),
    "floor_margin": (-10.0, 10.0),
    "depth_scale": (0.0, 100.0),
    "safety_threshold": (0.0, 1000.0),
    "temporal_window": (1, 1024),
    "fov_padding_bins": (0, 1024),
    "padding_fill_depth_m": (0.0, 1000.0),
    "marker_lifetime_sec": (0.0, 60.0),
    "timer_period_sec": (0.001, 60.0),
}

DESCRIPTIONS: Dict[str, str] = {
    "image_topic": "RGB image topic used for depth inference.",
    "perception_output_topic": "Stamped cognav_msgs/DistanceVector topic consumed by the behaviour tree.",
    "intrinsics_path": "Camera intrinsics .npy. Package-relative paths resolve through package share.",
    "weights_path": "Depth Anything V2 metric checkpoint .pth.",
    "encoder": "DA2 encoder variant.",
    "max_depth": "DA2 metric depth ceiling in metres.",
    "num_bins": "Number of angular bins across the real FOV.",
    "fov_deg": "Camera horizontal field of view in degrees.",
    "max_range": "Ignore back-projected points farther than this distance.",
    "vertical_margin": "Ceiling exclusion threshold in metres.",
    "floor_margin": "Floor exclusion threshold in metres.",
    "depth_scale": "Multiplicative correction on raw DA2 depth; see config/depth_scale.yaml.",
    "safety_threshold": "Distance threshold for the danger flag and marker colouring only.",
    "temporal_window": "Observed-frame median filter window size.",
    "fov_padding_bins": "Virtual wall bins added per FOV side.",
    "padding_fill_depth_m": "Depth assigned to each virtual FOV padding bin.",
    "frame_id": "TF frame attached to RViz markers.",
    "marker_lifetime_sec": "Lifetime of RViz depth markers.",
    "show_padding_bins": "Whether RViz markers include the virtual FOV padding bins.",
    "publish_depth_image": "Publish the colorized depth image topic.",
    "timer_period_sec": "Inference and publish cadence for the newest cached image.",
}


#: Placeholders in the depth_scale lookup expression. The launch file replaces
#: them with LaunchConfigurations.
SCENE_TOKEN = "@SCENE@"
PLATFORM_TOKEN = "@PLATFORM@"


def depth_scale_template(table: Dict[str, Any]) -> str:
    """The depth_scale lookup as a Python expression with placeholders.

    Resolution order: the named scene, the platform's default scene when the
    scene is empty, the platform row, then the table's fallback.
    """
    scenes = {
        name: entry["depth_scale"]
        for name, entry in (table.get("scenes") or {}).items()
        if isinstance(entry, dict) and "depth_scale" in entry
    }
    platforms = table.get("platforms") or {}
    default_scene = table.get("platform_default_scene") or {}
    fallback = table.get("fallback", PERCEPTION_DEFAULTS.depth_scale)
    return (
        f"{scenes!r}.get('{SCENE_TOKEN}' or "
        f"{default_scene!r}.get('{PLATFORM_TOKEN}', ''), "
        f"{platforms!r}.get('{PLATFORM_TOKEN}', {fallback!r}))"
    )


def parameter_names() -> Tuple[str, ...]:
    """Every declared ROS parameter name, in declaration order."""
    return tuple(f.name for f in fields(PerceptionConfig))
