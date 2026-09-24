#!/usr/bin/env python3
"""Import RGB and ground-truth depth pairs into the calibration sample format.

DA2's metric checkpoint is fine-tuned on Hypersim, so estimating the scale on
Hypersim frames is a control for the tooling and should return about 1.0.

    python -m cognav_perception.calibration.import_samples \
        --source hypersim --root <hypersim scene dir> \
        --out /opt/ws/logs/calibration/hypersim

    python -m cognav_perception.calibration.import_samples \
        --source pairs --rgb 'frames/*_rgb.npy' --depth 'frames/*_depth.npy' \
        --out /opt/ws/logs/calibration/mine --fx 127.2

Hypersim `depth_meters` is distance along the ray, which is converted to planar
depth by default for that source (`--planar` disables it).
"""
import argparse
import glob
import json
import os

import numpy as np

#: Hypersim render geometry; override --fx for re-rendered subsets.
HYPERSIM_DEFAULTS = {"width": 1024, "height": 768, "fx": 886.81}


def ray_to_planar(depth, fx, fy, cx, cy):
    """Convert distance along the ray to planar Z.

        Z = d / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)
    """
    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    return depth / np.sqrt(1.0 + x * x + y * y)


def linear_to_srgb8(rgb):
    """Gamma-encode linear float colour to 8-bit, as the model expects."""
    if rgb.dtype == np.uint8:
        return rgb
    return (np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2) * 255.0).astype(np.uint8)


def _load_hypersim(root):
    """Yield (rgb uint8 HxWx3, depth float HxW) from a Hypersim scene tree."""
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "reading Hypersim needs h5py: conda run -n perception pip install h5py"
        ) from exc

    colors = sorted(glob.glob(os.path.join(root, "**", "*.color.hdf5"), recursive=True))
    if not colors:
        raise SystemExit(
            f"no *.color.hdf5 under {root}; expected a Hypersim scene directory "
            f"containing images/scene_cam_XX_final_hdf5/"
        )
    for color_path in colors:
        depth_path = (color_path
                      .replace("_final_hdf5", "_geometry_hdf5")
                      .replace(".color.hdf5", ".depth_meters.hdf5"))
        if not os.path.exists(depth_path):
            continue
        with h5py.File(color_path, "r") as fh:
            rgb = np.asarray(fh["dataset"])
        with h5py.File(depth_path, "r") as fh:
            depth = np.asarray(fh["dataset"], dtype=np.float64)
        yield linear_to_srgb8(rgb)[..., :3], depth


def _load_pairs(rgb_glob, depth_glob):
    rgbs, depths = sorted(glob.glob(rgb_glob)), sorted(glob.glob(depth_glob))
    if not rgbs or len(rgbs) != len(depths):
        raise SystemExit(
            f"{len(rgbs)} rgb vs {len(depths)} depth files; expected equal and non-zero"
        )
    for r, d in zip(rgbs, depths):
        yield np.load(r), np.load(d).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("hypersim", "pairs"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", help="hypersim scene directory")
    ap.add_argument("--rgb", help="glob for RGB .npy (pairs source)")
    ap.add_argument("--depth", help="glob for depth .npy (pairs source)")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--ray-distance", dest="ray", action="store_true", default=None,
                    help="depth is distance along the ray rather than planar Z")
    ap.add_argument("--planar", dest="ray", action="store_false")
    args = ap.parse_args()

    if args.source == "hypersim":
        if not args.root:
            raise SystemExit("--root is required for the hypersim source")
        stream = _load_hypersim(args.root)
        fx = args.fx or HYPERSIM_DEFAULTS["fx"]
        ray = True if args.ray is None else args.ray
        label = f"hypersim:{os.path.basename(os.path.normpath(args.root))}"
    else:
        if not (args.rgb and args.depth):
            raise SystemExit("--rgb and --depth are required for the pairs source")
        stream = _load_pairs(args.rgb, args.depth)
        if args.fx is None:
            raise SystemExit("--fx is required for the pairs source")
        fx, ray = args.fx, bool(args.ray)
        label = "pairs"

    os.makedirs(args.out, exist_ok=True)
    written, shape = 0, None
    for rgb, depth in stream:
        if written >= args.limit:
            break
        shape = depth.shape
        h, w = shape
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        if ray:
            depth = ray_to_planar(depth, fx, fx, cx, cy)
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth) | (depth <= 0.0)] = np.nan
        np.save(os.path.join(args.out, f"rgb_{written:03d}.npy"),
                np.ascontiguousarray(rgb))
        np.save(os.path.join(args.out, f"depth_{written:03d}.npy"), depth)
        written += 1

    if not written:
        raise SystemExit("no usable pairs found")

    h, w = shape
    manifest = {
        "scene": label,
        "samples": written,
        "width": w,
        "height": h,
        "hfov_deg": float(np.degrees(2.0 * np.arctan((w / 2.0) / fx))),
        "sensor_height": None,
        "intrinsics": {"fx": float(fx), "fy": float(fx), "cx": cx, "cy": cy},
        "depth_was_ray_distance": bool(ray),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {written} samples to {args.out}  ({label}, {w}x{h}, "
          f"fx={fx:.1f}, ray->planar={'yes' if ray else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
