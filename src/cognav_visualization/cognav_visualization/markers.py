"""Image conversion, Float32MultiArray and RViz marker helpers for CogNav."""

from __future__ import annotations

import math

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA, Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray


# cv_bridge is avoided: in the conda environment it conflicts with libtiff, and
# for bgr8/rgb8 the conversion is a reshape.
def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image (bgr8 or rgb8) to a writable HxWx3 uint8 BGR ndarray."""
    enc = msg.encoding.lower()
    if enc not in ("bgr8", "rgb8"):
        raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if enc == "rgb8":
        arr = arr[..., ::-1]
    # frombuffer gives a read-only view, which cv2 inside DA2 rejects.
    return np.array(arr, dtype=np.uint8, order="C")


def bgr_to_imgmsg(bgr: np.ndarray, *, frame_id: str = "", stamp=None) -> Image:
    """Convert an HxWx3 uint8 BGR ndarray to sensor_msgs/Image (bgr8)."""
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 BGR, got shape {bgr.shape} dtype {bgr.dtype}")
    h, w = bgr.shape[:2]
    msg = Image()
    msg.height = int(h)
    msg.width = int(w)
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(w * 3)
    msg.data = np.ascontiguousarray(bgr).tobytes()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    return msg


def as_float_array(arr: np.ndarray, label: str) -> Float32MultiArray:
    """Wrap a 1-D ndarray as a Float32MultiArray. np.inf passes through as float32 inf."""
    msg = Float32MultiArray()
    msg.layout.dim.append(
        MultiArrayDimension(label=label, size=int(arr.size), stride=int(arr.size))
    )
    msg.data = arr.astype(np.float32).tolist()
    return msg


def _distance_color(dist: float, max_range: float) -> ColorRGBA:
    """Green (far) -> yellow -> red (close)."""
    t = max(0.0, min(1.0, dist / max(max_range, 1e-6)))
    if t > 0.5:
        s = (t - 0.5) * 2.0
        r, g = 1.0 - s, 1.0
    else:
        s = t * 2.0
        r, g = 1.0, s
    return ColorRGBA(r=float(r), g=float(g), b=0.0, a=0.9)


def _make_lifetime(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))


def build_depth_markers(
    out,
    *,
    frame_id: str,
    stamp,
    num_real_bins: int,
    fov_padding_bins: int,
    max_range: float,
    lifetime_sec: float = 0.5,
    show_padding_bins: bool = True,
) -> MarkerArray:
    """One arrow per bin of the padded distance vector.

    `out` exposes `distance_vector` and the matching `bin_angles`. Arrow length
    is the range and colour goes from green (far) to red (near). Unobserved
    bins are dim grey and padding bins short grey arrows.
    """
    ma = MarkerArray()
    lifetime = _make_lifetime(lifetime_sec)

    clear = Marker()
    clear.header.frame_id = frame_id
    clear.header.stamp = stamp
    clear.action = Marker.DELETEALL
    ma.markers.append(clear)

    n_padded = num_real_bins + 2 * fov_padding_bins
    for i in range(n_padded):
        is_padding = (i < fov_padding_bins) or (i >= fov_padding_bins + num_real_bins)
        if is_padding and not show_padding_bins:
            continue
        angle = float(out.bin_angles[i])
        dist = float(out.distance_vector[i])
        is_inf = math.isinf(dist) or math.isnan(dist)

        if is_padding:
            plot_dist = 0.25
            color = ColorRGBA(r=0.4, g=0.4, b=0.4, a=0.35)
        elif is_inf:
            plot_dist = max_range * 0.15
            color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.2)
        else:
            plot_dist = min(dist, max_range)
            color = _distance_color(dist, max_range)

        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "padding_bins" if is_padding else "depth_vector"
        m.id = i
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.lifetime = lifetime
        m.points = [
            Point(x=0.0, y=0.0, z=0.1),
            Point(
                x=plot_dist * math.cos(angle),
                y=plot_dist * math.sin(angle),
                z=0.1,
            ),
        ]
        m.scale = Vector3(x=0.02, y=0.04, z=0.04)
        m.color = color
        ma.markers.append(m)

    return ma


def depth_to_colormapped_bgr(depth_map: np.ndarray, max_range: float) -> np.ndarray:
    """Normalise depth_map to [0, max_range] and apply TURBO colormap. BGR uint8 out."""
    import cv2
    d = np.clip(depth_map, 0.0, max_range)
    d = (d / max(max_range, 1e-6) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
