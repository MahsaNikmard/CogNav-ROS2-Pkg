"""Measure `depth_scale` for a scene from ground-truth depth.

Two stages, because habitat-sim and torch live in different conda
environments:

    render_samples.py    habitat env      RGB and ground-truth depth
    import_samples.py    any env          the same layout from Hypersim or .npy pairs
    estimate_scale.py    perception env   median ground truth / DA2 ratio

The stages exchange a directory of samples plus manifest.json.
"""
