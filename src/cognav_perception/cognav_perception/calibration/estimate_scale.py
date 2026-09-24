#!/usr/bin/env python3
"""Estimate depth_scale as the median ratio of ground-truth to DA2 depth.

    conda run -n perception python -m cognav_perception.calibration.estimate_scale \
        --samples <sample dir> --weights <metric checkpoint>

Only pixels the encoder keeps are compared: 0 < Z <= max_range and
-vertical_margin <= Y <= floor_margin. The interquartile range is reported
with the median; a wide one means a single scalar fits the scene poorly.
"""
import argparse
import json
import os
import sys

import numpy as np


def usable_mask(depth, K, max_range, vertical_margin, floor_margin):
    """Pixels the polar encoder keeps."""
    h, w = depth.shape
    fy, cy = K["fy"], K["cy"]
    _, vv = np.meshgrid(np.arange(w, dtype=np.float64),
                        np.arange(h, dtype=np.float64))
    Z = depth.astype(np.float64)
    Y = ((vv - cy) * Z) / fy
    with np.errstate(invalid="ignore"):
        return ((Z > 0) & (Z <= max_range)
                & (Y >= -vertical_margin) & (Y <= floor_margin)
                & np.isfinite(Z))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--vertical-margin", type=float, default=0.10)
    ap.add_argument("--floor-margin", type=float, default=0.20)
    ap.add_argument("--min-pixels", type=int, default=500)
    args = ap.parse_args()

    with open(os.path.join(args.samples, "manifest.json")) as fh:
        manifest = json.load(fh)
    K = manifest["intrinsics"]

    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    configs = {
        "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
        "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
        "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
    }
    model = DepthAnythingV2(**{**configs[args.encoder], "max_depth": args.max_depth})
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    per_sample, pooled = [], []
    for i in range(manifest["samples"]):
        rgb = np.load(os.path.join(args.samples, f"rgb_{i:03d}.npy"))
        gt = np.load(os.path.join(args.samples, f"depth_{i:03d}.npy"))
        pred = model.infer_image(rgb[..., ::-1].copy())      # expects BGR

        mask = usable_mask(gt, K, args.max_range, args.vertical_margin, args.floor_margin)
        mask &= np.isfinite(pred) & (pred > 1e-3)
        if mask.sum() < args.min_pixels:
            per_sample.append(None)
            continue
        ratio = gt[mask] / pred[mask]
        per_sample.append(float(np.median(ratio)))
        pooled.append(ratio)

    usable = [r for r in per_sample if r is not None]
    if not usable:
        print("no sample had enough usable pixels; is the scene path right, "
              "and is the checkpoint the metric one?", file=sys.stderr)
        return 1

    allr = np.concatenate(pooled)
    q1, med, q3 = np.percentile(allr, [25, 50, 75])
    print(f"\nscene   : {manifest['scene']}")
    print(f"camera  : {manifest['width']}x{manifest['height']} hfov={manifest['hfov_deg']}deg "
          f"h={manifest['sensor_height']}m  fx={K['fx']:.1f}")
    print(f"samples : {len(usable)}/{manifest['samples']} usable, {allr.size} pixels pooled")
    print(f"\nper-sample median ratio: min={min(usable):.3f}  max={max(usable):.3f}  "
          f"spread={max(usable) - min(usable):.3f}")
    print(f"pooled  median ratio   : {med:.3f}   IQR [{q1:.3f}, {q3:.3f}]")
    print(f"\n  depth_scale: {med:.2f}")

    rel = (q3 - q1) / med if med else float("inf")
    if rel > 0.25:
        print(f"\n  WARNING: interquartile spread is {100 * rel:.0f}% of the median. One scalar "
              f"\n  does not describe this scene well; the correction is view-dependent, and "
              f"\n  whichever value you pick will be wrong somewhere in it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
