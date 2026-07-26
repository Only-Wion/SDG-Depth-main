#!/usr/bin/env python3
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.luna_dataset import LunaOrganized
from core.sdg_depth.net import SDGDepth
from core.utils.depth_mask import depth_percentile_mask
from core.utils.utils import InputPadder


def parse_args():
    parser = argparse.ArgumentParser(description="Run SDG-Depth on one Luna sample.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def model_and_dataset_args():
    return Namespace(
        max_disp=192,
        guided_flag=1,
        luna_resize=[768, 1024],
        luna_depth_scale=1.0 / 256.0,
        luna_val_fraction=0.0,
        luna_test_fraction=0.0,
        luna_lidar_max_time_diff_ns=100000000,
        luna_require_lidar=1,
        luna_camera_key="Cam_Rect_L",
        luna_apply_rectification=1,
        luna_image_subdir="images_rectified",
        luna_depth_subdir="depth_gt_rectified",
        luna_lidar_source="fastlio",
        luna_border_crop_fraction=0.1,
        luna_exclude_sequences=[],
        more_bottom=0.0,
        hints_density=0.05,
        expand_flag=1,
        refine_spn_resolution=4,
        gaussian_h=2,
        gaussian_w=8,
        cfnet_confidence_value=0.4,
        gsm_validhint="conf_04",
        disp_to_depth_convert_flag=1,
        disp_to_depth_convert_resolution=2,
        disp_to_depth_convert_disp_range=0.2,
        disp_to_depth_convert_depth_range=0.6,
        disp_to_depth_convert_gate=0.1,
        refine_spn_r=4,
        refine_spn_offset_flag=1,
        refine_spn_offset_range=1,
        refine_spn_conf_pixel="sparse_valid_all",
    )


def save_heatmap(path, values, valid, cmap, vmin, vmax, title, label):
    masked = np.ma.masked_where(~valid, values)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("black")
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = ax.imshow(masked, cmap=colormap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, shrink=0.85, label=label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    cli = parse_args()
    args = model_and_dataset_args()
    frame = cli.frame.removeprefix("frame").zfill(6)
    cli.output_dir.mkdir(parents=True, exist_ok=True)

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
            if sample["sequence"] == cli.sequence and sample["frame"] == frame
        ),
        None,
    )
    if sample_index is None:
        raise FileNotFoundError(
            f"No paired Luna sample found for {cli.sequence}/frame{frame}"
        )

    paths, image1, image2, flow_gt, valid_gt, hint, conversion_rate = dataset[
        sample_index
    ]
    device = torch.device("cuda")
    model = SDGDepth(args.max_disp, use_concat_volume=True, args=args).to(device)
    checkpoint = torch.load(cli.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    image1_batch = image1[None].to(device)
    image2_batch = image2[None].to(device)
    hint_batch = hint[None].to(device)
    conversion_batch = conversion_rate[None].to(device)
    padder = InputPadder(image1_batch.shape, divis_by=128)
    image1_padded, image2_padded = padder.pad(image1_batch, image2_batch)
    hint_padded = padder.pad(hint_batch)[0]

    with torch.no_grad():
        depth_predictions, _, _, _, _, _ = model(
            image1_padded,
            image2_padded,
            sparse=hint_padded,
            sparse_mask=(hint_padded > 0).int(),
            conversion_rate=conversion_batch,
        )

    prediction = (
        padder.unpad(depth_predictions[-1].unsqueeze(1))
        .cpu()
        .squeeze()
        .numpy()
        .astype(np.float32)
    )
    flow = flow_gt.squeeze().numpy().astype(np.float32)
    base_valid = valid_gt.bool().numpy() & np.isfinite(flow) & (flow > 0)
    gt_depth = np.zeros_like(flow, dtype=np.float32)
    gt_depth[base_valid] = float(conversion_rate) / flow[base_valid]

    metric_base = (
        base_valid
        & np.isfinite(prediction)
        & (prediction > 0)
        & (gt_depth <= 100.0)
    )
    metric_mask = (
        depth_percentile_mask(
            torch.from_numpy(gt_depth[None]),
            torch.from_numpy(metric_base[None]),
            0.05,
            0.95,
        )
        .squeeze(0)
        .numpy()
    )
    if not np.any(metric_mask):
        raise RuntimeError("No valid pixels remain after depth percentile filtering")

    absolute_error = np.full_like(gt_depth, np.nan, dtype=np.float32)
    absolute_error[metric_mask] = np.abs(
        prediction[metric_mask] - gt_depth[metric_mask]
    )
    errors = absolute_error[metric_mask]
    gt_values = gt_depth[metric_mask]
    depth_min = float(gt_values.min())
    depth_max = float(gt_values.max())
    error_max = max(float(np.quantile(errors, 0.95)), 0.1)
    left_rgb = image1.permute(1, 2, 0).byte().numpy()

    np.save(cli.output_dir / "prediction_depth.npy", prediction)
    np.save(cli.output_dir / "gt_depth.npy", gt_depth)
    np.save(cli.output_dir / "absolute_error.npy", absolute_error)
    np.save(cli.output_dir / "evaluation_mask.npy", metric_mask)
    plt.imsave(cli.output_dir / "left_rectified_crop10.png", left_rgb)

    prediction_valid = np.isfinite(prediction) & (prediction > 0)
    save_heatmap(
        cli.output_dir / "prediction_depth.png",
        prediction,
        prediction_valid,
        "turbo",
        depth_min,
        depth_max,
        "Predicted depth",
        "Depth (m)",
    )
    save_heatmap(
        cli.output_dir / "gt_depth.png",
        gt_depth,
        metric_mask,
        "turbo",
        depth_min,
        depth_max,
        "Ground-truth depth (5%-95% mask)",
        "Depth (m)",
    )
    save_heatmap(
        cli.output_dir / "absolute_error.png",
        absolute_error,
        metric_mask,
        "magma",
        0.0,
        error_max,
        "Absolute depth error (5%-95% mask)",
        "Absolute error (m)",
    )

    depth_cmap = plt.get_cmap("turbo").copy()
    error_cmap = plt.get_cmap("magma").copy()
    depth_cmap.set_bad("black")
    error_cmap.set_bad("black")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    pred_image = axes[0].imshow(
        np.ma.masked_where(~metric_mask, prediction),
        cmap=depth_cmap,
        vmin=depth_min,
        vmax=depth_max,
    )
    axes[0].set_title("Prediction")
    axes[1].imshow(
        np.ma.masked_where(~metric_mask, gt_depth),
        cmap=depth_cmap,
        vmin=depth_min,
        vmax=depth_max,
    )
    axes[1].set_title("Ground truth")
    error_image = axes[2].imshow(
        np.ma.masked_where(~metric_mask, absolute_error),
        cmap=error_cmap,
        vmin=0.0,
        vmax=error_max,
    )
    axes[2].set_title("Absolute error")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(
        pred_image, ax=axes[:2], shrink=0.82, label="Depth (m)", location="bottom"
    )
    fig.colorbar(
        error_image,
        ax=axes[2],
        shrink=0.82,
        label="Absolute error (m)",
        location="bottom",
    )
    fig.savefig(cli.output_dir / "comparison.png", dpi=160)
    plt.close(fig)

    metrics = {
        "sequence": cli.sequence,
        "frame": frame,
        "checkpoint": str(cli.checkpoint),
        "input_left": paths[0],
        "input_right": paths[1],
        "depth_gt": paths[2],
        "image_height": int(prediction.shape[0]),
        "image_width": int(prediction.shape[1]),
        "valid_pixels_5_95": int(metric_mask.sum()),
        "depth_5_percent_m": depth_min,
        "depth_95_percent_m": depth_max,
        "mae_m": float(errors.mean()),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mape_percent": float(
            np.mean(errors / np.maximum(gt_values, 1e-6)) * 100.0
        ),
        "absolute_error_p95_m": float(np.quantile(errors, 0.95)),
    }
    with open(cli.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
