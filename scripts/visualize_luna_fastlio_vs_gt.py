#!/usr/bin/env python3
"""Visualize projected FAST-LIO depth together with Luna dense depth GT."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.luna_dataset import LunaOrganized
from scripts.infer_one_luna import model_and_dataset_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-subdir", default="images_rectified")
    parser.add_argument("--left-dirname", default="left")
    parser.add_argument("--right-dirname", default="right")
    return parser.parse_args()


def main():
    cli = parse_args()
    frame = cli.frame.removeprefix("frame").zfill(6)

    args = model_and_dataset_args()
    args.luna_exclude_sequences = []
    args.luna_val_fraction = 0.0
    args.luna_test_fraction = 0.0
    args.luna_image_subdir = cli.image_subdir
    args.luna_left_dirname = cli.left_dirname
    args.luna_right_dirname = cli.right_dirname
    dataset = LunaOrganized(
        aug_params={},
        root=str(cli.root),
        image_set="all",
        args=args,
    )
    sample_index = next(
        (
            index
            for index, sample in enumerate(dataset.extra_info)
            if sample["sequence"] == cli.sequence
            and sample["frame"] == frame
        ),
        None,
    )
    if sample_index is None:
        raise FileNotFoundError(
            f"No paired sample found for {cli.sequence}/frame{frame}"
        )

    paths, image1, _, flow_gt, valid_gt, hint, conversion_rate = dataset[
        sample_index
    ]
    rgb = image1.permute(1, 2, 0).byte().numpy()
    flow = flow_gt.squeeze().numpy().astype(np.float64)
    sparse_disparity = hint.squeeze().numpy().astype(np.float64)
    conversion = float(conversion_rate)

    gt_valid = valid_gt.bool().numpy() & np.isfinite(flow) & (flow > 0)
    fastlio_valid = (
        np.isfinite(sparse_disparity) & (sparse_disparity > 0)
    )
    compared = gt_valid & fastlio_valid

    gt_depth = np.full(flow.shape, np.nan, dtype=np.float64)
    gt_depth[gt_valid] = conversion / flow[gt_valid]
    fastlio_depth = np.full(flow.shape, np.nan, dtype=np.float64)
    fastlio_depth[fastlio_valid] = (
        conversion / sparse_disparity[fastlio_valid]
    )
    ys, xs = np.nonzero(compared)
    lidar_values = fastlio_depth[compared]
    gt_values = gt_depth[compared]
    signed_error = lidar_values - gt_values
    absolute_error = np.abs(signed_error)
    relative_error = absolute_error / np.maximum(gt_values, 1e-6) * 100.0

    depth_values = np.concatenate((gt_depth[gt_valid], lidar_values))
    depth_min, depth_max = np.quantile(depth_values, [0.01, 0.99])
    error_limit = max(float(np.quantile(np.abs(signed_error), 0.95)), 0.1)

    figure, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    figure.suptitle(
        f"{cli.sequence} / frame{frame} — FAST-LIO vs depth GT",
        fontsize=18,
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Rectified left RGB (model input crop)")

    gt_image = axes[0, 1].imshow(
        gt_depth,
        cmap="turbo",
        vmin=depth_min,
        vmax=depth_max,
    )
    axes[0, 1].set_title(
        f"Dense depth GT ({depth_min:.2f}–{depth_max:.2f} m color range)"
    )
    figure.colorbar(gt_image, ax=axes[0, 1], label="Depth (m)")

    axes[1, 0].imshow(
        gt_depth,
        cmap="turbo",
        vmin=depth_min,
        vmax=depth_max,
    )
    lidar_plot = axes[1, 0].scatter(
        xs,
        ys,
        c=lidar_values,
        cmap="turbo",
        vmin=depth_min,
        vmax=depth_max,
        s=12,
        edgecolors="white",
        linewidths=0.15,
    )
    axes[1, 0].set_title(
        f"FAST-LIO points over depth GT — {len(xs):,} points\n"
        "Same depth color scale; white-edged dots are FAST-LIO"
    )
    figure.colorbar(lidar_plot, ax=axes[1, 0], label="Depth (m)")

    axes[1, 1].imshow(rgb, alpha=0.32)
    error_plot = axes[1, 1].scatter(
        xs,
        ys,
        c=signed_error,
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
        s=11,
        linewidths=0,
    )
    axes[1, 1].set_title(
        "FAST-LIO − depth GT at projected pixels\n"
        f"MAE {absolute_error.mean():.3f} m | "
        f"mean relative error {relative_error.mean():.2f}% | "
        f"median {np.median(relative_error):.2f}%"
    )
    figure.colorbar(
        error_plot,
        ax=axes[1, 1],
        label="Signed depth error (m; clipped at frame P95)",
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
        f"input={paths[0]}\n"
        f"points={len(xs)}\n"
        f"mae_m={absolute_error.mean():.6f}\n"
        f"mean_relative_error_percent={relative_error.mean():.6f}\n"
        f"median_relative_error_percent={np.median(relative_error):.6f}"
    )


if __name__ == "__main__":
    main()
