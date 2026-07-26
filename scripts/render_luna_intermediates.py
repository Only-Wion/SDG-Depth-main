#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render saved Luna intermediate NumPy arrays."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def save_map(path, values, valid, cmap_name, vmin, vmax, title, label):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = ax.imshow(
        np.ma.masked_where(~valid, values),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, shrink=0.85, label=label)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    directory = args.input_dir
    sparse_depth = np.load(directory / "sparse_lidar_depth.npy")
    propagated = np.load(directory / "propagated_disparity.npy")
    confidence = np.load(directory / "confidence_map.npy")
    depth = np.load(directory / "prediction_depth.npy")
    left = np.asarray(Image.open(directory / "left_image.png").convert("RGB"))
    right = np.asarray(Image.open(directory / "right_image.png").convert("RGB"))

    sparse_valid = np.isfinite(sparse_depth) & (sparse_depth > 0)
    propagated_valid = np.isfinite(propagated) & (propagated > 0)
    confidence_valid = np.isfinite(confidence)
    depth_valid = np.isfinite(depth) & (depth > 0)
    sparse_max = float(np.quantile(sparse_depth[sparse_valid], 0.99))
    propagated_max = float(np.quantile(propagated[propagated_valid], 0.99))
    depth_min, depth_max = np.quantile(depth[depth_valid], [0.01, 0.99])

    save_map(
        directory / "sparse_lidar.png",
        sparse_depth,
        sparse_valid,
        "turbo",
        0.0,
        sparse_max,
        "Sparse LiDAR",
        "Depth (m)",
    )
    save_map(
        directory / "propagated_disparity.png",
        propagated,
        propagated_valid,
        "turbo",
        0.0,
        propagated_max,
        "Propagated disparity",
        "Disparity (px)",
    )
    save_map(
        directory / "confidence_map.png",
        confidence,
        confidence_valid,
        "gray_r",
        0.0,
        1.0,
        "Confidence map",
        "Confidence",
    )

    sparse_cmap = plt.get_cmap("turbo").copy()
    propagated_cmap = plt.get_cmap("turbo").copy()
    confidence_cmap = plt.get_cmap("gray_r").copy()
    depth_cmap = plt.get_cmap("turbo").copy()
    for cmap in (sparse_cmap, propagated_cmap, confidence_cmap, depth_cmap):
        cmap.set_bad("white")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(left)
    axes[0].set_title("Left image")
    axes[1].imshow(right)
    axes[1].set_title("Right image")
    sparse_image = axes[2].imshow(
        np.ma.masked_where(~sparse_valid, sparse_depth),
        cmap=sparse_cmap,
        vmin=0.0,
        vmax=sparse_max,
    )
    axes[2].set_title("Sparse LiDAR")
    propagated_image = axes[3].imshow(
        np.ma.masked_where(~propagated_valid, propagated),
        cmap=propagated_cmap,
        vmin=0.0,
        vmax=propagated_max,
    )
    axes[3].set_title("Propagated disparity")
    confidence_image = axes[4].imshow(
        np.ma.masked_where(~confidence_valid, confidence),
        cmap=confidence_cmap,
        vmin=0.0,
        vmax=1.0,
    )
    axes[4].set_title("Confidence map")
    depth_image = axes[5].imshow(
        np.ma.masked_where(~depth_valid, depth),
        cmap=depth_cmap,
        vmin=float(depth_min),
        vmax=float(depth_max),
    )
    axes[5].set_title("Depth")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(sparse_image, ax=axes[2], shrink=0.72, label="Depth (m)")
    fig.colorbar(
        propagated_image, ax=axes[3], shrink=0.72, label="Disparity (px)"
    )
    fig.colorbar(confidence_image, ax=axes[4], shrink=0.72, label="Confidence")
    fig.colorbar(depth_image, ax=axes[5], shrink=0.72, label="Depth (m)")
    fig.savefig(directory / "six_outputs.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
