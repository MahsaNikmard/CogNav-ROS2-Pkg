#!/usr/bin/env python3
"""ROS 2 node: RGB image in, stamped polar distance vector out.

Publishes

    /perception/distance_vector       cognav_msgs/DistanceVector
    /perception/distance_vector_raw   std_msgs/Float32MultiArray
    /perception/danger                std_msgs/Float32MultiArray
    /perception/depth_markers         visualization_msgs/MarkerArray
    /perception/depth_image           sensor_msgs/Image

The distance vector carries the stamp of the source image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, get_type_hints

import numpy as np
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
import rclpy
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from cognav_msgs.msg import DistanceVector
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray

from cognav_perception.defaults import (
    DESCRIPTIONS,
    LIMITS,
    PERCEPTION_DEFAULTS,
    PerceptionConfig,
    parameter_names,
)
from cognav_perception.perception import Perception
from cognav_visualization.markers import (
    as_float_array,
    bgr_to_imgmsg,
    build_depth_markers,
    depth_to_colormapped_bgr,
    imgmsg_to_bgr,
)


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
    depth=5,
)

_SRC_WS_ROOT = Path(__file__).resolve().parents[2]


def _float_descriptor(description: str, *, from_value: float, to_value: float) -> ParameterDescriptor:
    return ParameterDescriptor(
        description=description,
        floating_point_range=[FloatingPointRange(from_value=from_value, to_value=to_value, step=0.0)],
    )


def _integer_descriptor(description: str, *, from_value: int, to_value: int) -> ParameterDescriptor:
    return ParameterDescriptor(
        description=description,
        integer_range=[IntegerRange(from_value=from_value, to_value=to_value, step=1)],
    )


def _resolve_ws_path(path_value: str) -> Path:
    """Resolve `<package>/<relative path>` through the package share directory."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if path.parts:
        try:
            package_share = Path(get_package_share_directory(path.parts[0]))
        except PackageNotFoundError:
            pass
        else:
            candidate = package_share.joinpath(*path.parts[1:])
            if candidate.exists():
                return candidate
    return _SRC_WS_ROOT / path


class CognavPerception(Node):
    """Runs DA2 and the polar encoder on the newest received image."""

    def __init__(self) -> None:
        super().__init__("cognav_perception")

        self._declare_parameters()
        self._load_parameters()

        self._latest_image_msg: Optional[Image] = None
        self._latest_image_token = 0
        self._last_processed_token = 0
        self._frame_count = 0

        self.perception = Perception(
            intrinsics_path=self.intrinsics_path,
            weights_path=self.weights_path,
            encoder=self.encoder,
            max_depth=self.max_depth,
            num_bins=self.num_bins,
            fov_deg=self.fov_deg,
            max_range=self.max_range,
            vertical_margin=self.vertical_margin,
            floor_margin=self.floor_margin,
            depth_scale=self.depth_scale,
            safety_threshold=self.safety_threshold,
            temporal_window=self.temporal_window,
            fov_padding_bins=self.fov_padding_bins,
            fov_padding_depth=self.padding_fill_depth_m,
        )
        self.num_bins = self.perception.num_bins
        self.max_plot_range = self.perception.max_range

        self.distance_vector_pub = self.create_publisher(
            DistanceVector, self.perception_output_topic, QOS_RELIABLE
        )
        self.dv_raw_pub = self.create_publisher(
            Float32MultiArray, "/perception/distance_vector_raw", QOS_RELIABLE
        )
        self.danger_pub = self.create_publisher(
            Float32MultiArray, "/perception/danger", QOS_RELIABLE
        )
        self.markers_pub = self.create_publisher(
            MarkerArray, "/perception/depth_markers", QOS_RELIABLE
        )
        self.depth_image_pub = None
        if self.publish_depth_image:
            self.depth_image_pub = self.create_publisher(
                Image, "/perception/depth_image", QOS_BEST_EFFORT
            )

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, QOS_BEST_EFFORT
        )
        self.timer = self.create_timer(self.timer_period_sec, self._timer_cb)

        self.get_logger().info("cognav_perception starting:")
        self.get_logger().info(f"  image topic         : {self.image_topic}")
        self.get_logger().info(f"  output topic        : {self.perception_output_topic}")
        self.get_logger().info(f"  intrinsics          : {self.intrinsics_path}")
        self.get_logger().info(f"  weights             : {self.weights_path}")
        self.get_logger().info(f"  depth_scale         : {self.depth_scale}")
        self.get_logger().info(f"  encoder             : {self.encoder}")
        self.get_logger().info(f"  frame_id            : {self.frame_id}")
        self.get_logger().info(f"  timer_period_sec    : {self.timer_period_sec}")
        self.get_logger().info(f"  publish_depth_image : {self.publish_depth_image}")
        self.get_logger().info("cognav_perception ready.")

    def _declare_parameters(self) -> None:
        """Declare every field of PerceptionConfig as a ROS parameter."""
        for name in parameter_names():
            default = getattr(PERCEPTION_DEFAULTS, name)
            description = DESCRIPTIONS.get(name, "")
            if name in LIMITS:
                low, high = LIMITS[name]
                if isinstance(default, bool) or not isinstance(default, int):
                    descriptor = _float_descriptor(
                        description, from_value=float(low), to_value=float(high)
                    )
                else:
                    descriptor = _integer_descriptor(
                        description, from_value=int(low), to_value=int(high)
                    )
            elif name == "encoder":
                descriptor = ParameterDescriptor(
                    description=description,
                    additional_constraints="Must be one of: vits, vitb, vitl.",
                )
            else:
                descriptor = ParameterDescriptor(description=description)
            self.declare_parameter(name, default, descriptor)

    def _load_parameters(self) -> None:
        """Bind every declared parameter to an attribute of the same name."""
        hints = get_type_hints(PerceptionConfig)
        for name in parameter_names():
            value = self.get_parameter(name).value
            setattr(self, name, hints[name](value))
        self.intrinsics_path = _resolve_ws_path(self.intrinsics_path)
        self.weights_path = _resolve_ws_path(self.weights_path)

    def _image_cb(self, msg: Image) -> None:
        # Processed on arrival: under simulator lockstep the next step waits
        # for the command this frame produces. The timer is a fallback.
        self._latest_image_msg = msg
        self._latest_image_token += 1
        self._timer_cb()

    def _timer_cb(self) -> None:
        src_msg = self._latest_image_msg
        src_token = self._latest_image_token
        if src_msg is None or src_token == self._last_processed_token:
            return

        try:
            out = self.perception.update(imgmsg_to_bgr(src_msg))
        except Exception as exc:
            self.get_logger().error(f"Perception failed: {exc}")
            self._last_processed_token = src_token
            return

        self._last_processed_token = src_token
        self._frame_count += 1

        if self._frame_count == 1 or self._frame_count % 50 == 0:
            n_obs = int(np.isfinite(out.raw_distance_vector).sum())
            self.get_logger().info(
                f"  frame {self._frame_count}: {n_obs}/{self.num_bins} real bins observed"
            )

        self.distance_vector_pub.publish(self._to_distance_vector_msg(out, src_msg))
        self.dv_raw_pub.publish(as_float_array(out.raw_distance_vector, "raw_distance_vector"))
        self.danger_pub.publish(as_float_array(out.danger, "danger"))

        self.markers_pub.publish(
            build_depth_markers(
                out,
                frame_id=self.frame_id,
                stamp=src_msg.header.stamp,
                num_real_bins=self.num_bins,
                fov_padding_bins=self.perception.fov_padding_bins,
                max_range=self.max_plot_range,
                lifetime_sec=self.marker_lifetime_sec,
                show_padding_bins=self.show_padding_bins,
            )
        )

        if self.publish_depth_image and self.depth_image_pub is not None:
            self._publish_depth_image(out.depth_map, src_msg)

    def _to_distance_vector_msg(self, out, src_msg: Image) -> DistanceVector:
        msg = DistanceVector()
        msg.header.stamp = src_msg.header.stamp
        msg.header.frame_id = self.frame_id
        msg.distance_vector = np.asarray(out.distance_vector, dtype=np.float32).tolist()
        return msg

    def _publish_depth_image(self, depth_map: np.ndarray, src_msg: Image) -> None:
        colored = depth_to_colormapped_bgr(depth_map, self.max_plot_range)
        try:
            msg = bgr_to_imgmsg(
                colored,
                frame_id=src_msg.header.frame_id,
                stamp=src_msg.header.stamp,
            )
        except Exception as exc:
            self.get_logger().warn(f"depth viz publish failed: {exc}")
            return
        self.depth_image_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CognavPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
