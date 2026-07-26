#!/usr/bin/env python3
import argparse
import csv
import json
import sys
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
from scripts.infer_one_luna import model_and_dataset_args, save_heatmap


EXCLUDED_SEQUENCES = [
    "2025-06-26-22-59-05",
    "2025-06-26-23-01-34",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a Luna split and retain the five samples with lowest MAPE."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=5)
    return parser.parse_args()


def infer_sample(model, sample, device):
    paths, image1, image2, flow_gt, valid_gt, hint, conversion_rate = sample
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
        return None

    absolute_error = np.full_like(gt_depth, np.nan, dtype=np.float32)
    absolute_error[metric_mask] = np.abs(
        prediction[metric_mask] - gt_depth[metric_mask]
    )
    errors = absolute_error[metric_mask]
    gt_values = gt_depth[metric_mask]
    info = {
        "sequence": Path(paths[0]).parents[2].name,
        "frame": Path(paths[0]).stem,
        "input_left": paths[0],
        "input_right": paths[1],
        "depth_gt": paths[2],
        "valid_pixels_5_95": int(metric_mask.sum()),
        "depth_5_percent_m": float(gt_values.min()),
        "depth_95_percent_m": float(gt_values.max()),
        "mae_m": float(errors.mean()),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mape_percent": float(
            np.mean(errors / np.maximum(gt_values, 1e-6)) * 100.0
        ),
        "absolute_error_p95_m": float(np.quantile(errors, 0.95)),
    }
    return {
        "metrics": info,
        "left_rgb": image1.permute(1, 2, 0).byte().numpy(),
        "prediction": prediction,
        "gt_depth": gt_depth,
        "absolute_error": absolute_error,
        "metric_mask": metric_mask,
    }


def save_comparison(path, result):
    metrics = result["metrics"]
    mask = result["metric_mask"]
    depth_min = metrics["depth_5_percent_m"]
    depth_max = metrics["depth_95_percent_m"]
    error_max = max(metrics["absolute_error_p95_m"], 0.1)
    depth_cmap = plt.get_cmap("turbo").copy()
    error_cmap = plt.get_cmap("magma").copy()
    depth_cmap.set_bad("black")
    error_cmap.set_bad("black")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    pred_image = axes[0].imshow(
        np.ma.masked_where(~mask, result["prediction"]),
        cmap=depth_cmap,
        vmin=depth_min,
        vmax=depth_max,
    )
    axes[0].set_title("Prediction")
    axes[1].imshow(
        np.ma.masked_where(~mask, result["gt_depth"]),
        cmap=depth_cmap,
        vmin=depth_min,
        vmax=depth_max,
    )
    axes[1].set_title("Ground truth")
    error_image = axes[2].imshow(
        np.ma.masked_where(~mask, result["absolute_error"]),
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
    fig.suptitle(
        f'{metrics["sequence"]} frame{metrics["frame"]} | '
        f'MAPE {metrics["mape_percent"]:.3f}%',
        fontsize=14,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_result(output_dir, rank, result):
    metrics = result["metrics"]
    sample_dir = output_dir / (
        f'rank{rank:02d}_{metrics["sequence"]}_frame{metrics["frame"]}'
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    mask = result["metric_mask"]
    depth_min = metrics["depth_5_percent_m"]
    depth_max = metrics["depth_95_percent_m"]
    error_max = max(metrics["absolute_error_p95_m"], 0.1)

    np.save(sample_dir / "prediction_depth.npy", result["prediction"])
    np.save(sample_dir / "gt_depth.npy", result["gt_depth"])
    np.save(sample_dir / "absolute_error.npy", result["absolute_error"])
    np.save(sample_dir / "evaluation_mask.npy", mask)
    plt.imsave(sample_dir / "left_rectified_crop10.png", result["left_rgb"])
    save_heatmap(
        sample_dir / "prediction_depth.png",
        result["prediction"],
        np.isfinite(result["prediction"]) & (result["prediction"] > 0),
        "turbo",
        depth_min,
        depth_max,
        "Predicted depth",
        "Depth (m)",
    )
    save_heatmap(
        sample_dir / "gt_depth.png",
        result["gt_depth"],
        mask,
        "turbo",
        depth_min,
        depth_max,
        "Ground-truth depth (5%-95% mask)",
        "Depth (m)",
    )
    save_heatmap(
        sample_dir / "absolute_error.png",
        result["absolute_error"],
        mask,
        "magma",
        0.0,
        error_max,
        "Absolute depth error (5%-95% mask)",
        "Absolute error (m)",
    )
    save_comparison(sample_dir / "comparison.png", result)
    with open(sample_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main():
    cli = parse_args()
    if cli.keep <= 0:
        raise ValueError("--keep must be positive")
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    args = model_and_dataset_args()
    args.luna_val_fraction = 0.2
    args.luna_test_fraction = 0.2
    args.luna_exclude_sequences = EXCLUDED_SEQUENCES
    dataset = LunaOrganized(
        aug_params={},
        root=str(cli.root),
        image_set="test",
        args=args,
    )
    device = torch.device("cuda")
    model = SDGDepth(args.max_disp, use_concat_volume=True, args=args).to(device)
    checkpoint = torch.load(cli.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    top_results = []
    all_metrics = []
    skipped = 0
    for index in range(len(dataset)):
        result = infer_sample(model, dataset[index], device)
        if result is None:
            skipped += 1
            continue
        all_metrics.append(result["metrics"])
        top_results.append(result)
        top_results.sort(key=lambda item: item["metrics"]["mape_percent"])
        if len(top_results) > cli.keep:
            top_results.pop()
        print(
            f'[{index + 1}/{len(dataset)}] '
            f'{result["metrics"]["sequence"]}/frame{result["metrics"]["frame"]} '
            f'MAPE={result["metrics"]["mape_percent"]:.4f}%'
        )

    all_metrics.sort(key=lambda item: item["mape_percent"])
    for rank, result in enumerate(top_results, start=1):
        result["metrics"]["rank"] = rank
        save_result(cli.output_dir, rank, result)

    with open(cli.output_dir / "all_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    if all_metrics:
        with open(
            cli.output_dir / "all_metrics.csv", "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(all_metrics)

    summary = {
        "checkpoint": str(cli.checkpoint),
        "split": "test",
        "excluded_sequences": EXCLUDED_SEQUENCES,
        "total_samples": len(dataset),
        "evaluated_samples": len(all_metrics),
        "skipped_samples": skipped,
        "kept_samples": len(top_results),
        "top5": [result["metrics"] for result in top_results],
    }
    with open(cli.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
