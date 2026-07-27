#!/usr/bin/env python3
"""Visualize GT-sampled pseudo-LiDAR points against the source depth GT."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--sampled-lidar-root", type=Path, required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resize-height", type=int, default=768)
    parser.add_argument("--resize-width", type=int, default=1024)
    parser.add_argument("--crop-fraction", type=float, default=0.1)
    return parser.parse_args()


def resize_and_crop(image, target_hw, crop_fraction, resample):
    height, width = target_hw
    resized = image.resize((width, height), resample)
    crop_y = int(round(height * crop_fraction))
    crop_x = int(round(width * crop_fraction))
    return np.asarray(
        resized.crop((crop_x, crop_y, width - crop_x, height - crop_y))
    )


def main():
    cli = parse_args()
    frame = cli.frame.removeprefix("frame").zfill(6)
    target_hw = (cli.resize_height, cli.resize_width)

    rgb_path = (
        cli.sequence_root
        / "images_rectified"
        / "left_out"
        / f"{frame}.png"
    )
    depth_path = cli.sequence_root / "depth_gt_rectified" / f"{frame}.png"
    sample_path = cli.sampled_lidar_root / "samples" / f"{frame}.npz"
    for path in (rgb_path, depth_path, sample_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rgb = resize_and_crop(
        Image.open(rgb_path).convert("RGB"),
        target_hw,
        cli.crop_fraction,
        Image.Resampling.BILINEAR,
    )
    depth_gt = (
        resize_and_crop(
            Image.open(depth_path),
            target_hw,
            cli.crop_fraction,
            Image.Resampling.NEAREST,
        ).astype(np.float64)
        / 256.0
    )
    with np.load(sample_path) as samples:
        u = samples["u"].astype(np.int32)
        v = samples["v"].astype(np.int32)
        sampled_depth = samples["depth_gt_m"].astype(np.float64)

    comparison_gt = depth_gt[v, u]
    signed_error = sampled_depth - comparison_gt
    absolute_error = np.abs(signed_error)
    relative_error = absolute_error / np.maximum(comparison_gt, 1e-6) * 100.0

    valid_gt = np.isfinite(depth_gt) & (depth_gt > 0)
    depth_min, depth_max = np.quantile(depth_gt[valid_gt], [0.01, 0.99])
    error_limit = max(float(np.quantile(absolute_error, 0.99)), 1e-4)

    figure, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    figure.suptitle(
        f"{cli.sequence_root.name} / frame{frame} — "
        "GT-sampled LiDAR vs depth GT",
        fontsize=18,
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Rectified left RGB (model input crop)")

    gt_plot = axes[0, 1].imshow(
        np.where(valid_gt, depth_gt, np.nan),
        cmap="turbo",
        vmin=depth_min,
        vmax=depth_max,
    )
    axes[0, 1].set_title(
        f"Dense depth GT ({depth_min:.2f}–{depth_max:.2f} m color range)"
    )
    figure.colorbar(gt_plot, ax=axes[0, 1], label="Depth (m)")

    axes[1, 0].imshow(
        np.where(valid_gt, depth_gt, np.nan),
        cmap="turbo",
        vmin=depth_min,
        vmax=depth_max,
    )
    lidar_plot = axes[1, 0].scatter(
        u,
        v,
        c=sampled_depth,
        cmap="turbo",
        vmin=depth_min,
        vmax=depth_max,
        s=12,
        edgecolors="white",
        linewidths=0.15,
    )
    axes[1, 0].set_title(
        f"Generated LiDAR over depth GT — {len(u):,} points\n"
        "Same depth color scale; white-edged dots are generated LiDAR"
    )
    figure.colorbar(lidar_plot, ax=axes[1, 0], label="Depth (m)")

    axes[1, 1].imshow(rgb, alpha=0.32)
    error_plot = axes[1, 1].scatter(
        u,
        v,
        c=signed_error,
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
        s=11,
        linewidths=0,
    )
    axes[1, 1].set_title(
        "Generated LiDAR − depth GT at sampled pixels\n"
        f"MAE {absolute_error.mean():.8f} m | "
        f"max {absolute_error.max():.8f} m | "
        f"mean relative error {relative_error.mean():.8f}%"
    )
    figure.colorbar(
        error_plot,
        ax=axes[1, 1],
        label="Signed depth error (m)",
    )

    for axis in axes.flat:
        axis.set_xlim(0, rgb.shape[1] - 1)
        axis.set_ylim(rgb.shape[0] - 1, 0)
        axis.set_axis_off()

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(cli.output, dpi=180, facecolor="white")
    plt.close(figure)
    print(
        f"saved={cli.output}\n"
        f"points={len(u)}\n"
        f"mae_m={absolute_error.mean():.12g}\n"
        f"max_abs_error_m={absolute_error.max():.12g}\n"
        f"mean_relative_error_percent={relative_error.mean():.12g}"
    )


if __name__ == "__main__":
    main()
