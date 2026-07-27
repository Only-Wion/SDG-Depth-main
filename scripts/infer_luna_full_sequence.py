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
from scripts.infer_luna_top5 import infer_sample, save_comparison
from scripts.infer_one_luna import model_and_dataset_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference and aggregate metrics for one complete Luna sequence."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument(
        "--split", choices=["all", "training", "val", "test"], default="all"
    )
    parser.add_argument("--exclude-sequences", nargs="*", default=[])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-subdir", default="images_rectified")
    parser.add_argument("--left-dirname", default="left")
    parser.add_argument("--right-dirname", default="right")
    parser.add_argument("--depth-subdir", default="depth_gt_rectified")
    parser.add_argument(
        "--lidar-source", choices=["raw", "fastlio", "fake"], default="fastlio"
    )
    parser.add_argument("--border-crop-fraction", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main():
    cli = parse_args()
    selected_sequences = cli.sequences or ([cli.sequence] if cli.sequence else [])
    if not selected_sequences:
        raise ValueError("Pass --sequence or --sequences")
    if cli.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    args = model_and_dataset_args()
    args.luna_exclude_sequences = cli.exclude_sequences
    args.luna_val_fraction = cli.val_fraction
    args.luna_test_fraction = cli.test_fraction
    args.luna_image_subdir = cli.image_subdir
    args.luna_left_dirname = cli.left_dirname
    args.luna_right_dirname = cli.right_dirname
    args.luna_depth_subdir = cli.depth_subdir
    args.luna_lidar_source = cli.lidar_source
    args.luna_border_crop_fraction = cli.border_crop_fraction
    dataset = LunaOrganized(
        aug_params={},
        root=str(cli.root),
        image_set=cli.split,
        args=args,
    )
    sample_indices = [
        index
        for index, sample in enumerate(dataset.extra_info)
        if sample["sequence"] in selected_sequences
    ]
    if not sample_indices:
        raise FileNotFoundError(
            f"No valid {cli.lidar_source} LiDAR-paired samples found for "
            f"sequences {selected_sequences}"
        )

    device = torch.device("cuda")
    model = SDGDepth(args.max_disp, use_concat_volume=True, args=args).to(device)
    checkpoint = torch.load(cli.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    for _ in range(cli.warmup):
        infer_sample(model, dataset[sample_indices[0]], device)

    frame_metrics = []
    skipped_frames = []
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    relative_error_sum = 0.0
    valid_pixel_count = 0
    for progress, sample_index in enumerate(sample_indices, start=1):
        result = infer_sample(model, dataset[sample_index], device)
        frame = dataset.extra_info[sample_index]["frame"]
        if result is None:
            skipped_frames.append(frame)
            continue

        metrics = result["metrics"]
        frame_dir = (
            cli.output_dir
            / metrics["sequence"]
            / f'frame{metrics["frame"]}'
        )
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "prediction_depth.npy", result["prediction"])
        np.save(frame_dir / "gt_depth.npy", result["gt_depth"])
        np.save(frame_dir / "absolute_error.npy", result["absolute_error"])
        np.save(frame_dir / "evaluation_mask.npy", result["metric_mask"])
        plt.imsave(frame_dir / "model_input_left.png", result["left_rgb"])
        plt.imsave(frame_dir / "model_input_right.png", result["right_rgb"])
        save_comparison(frame_dir / "comparison.png", result)
        with open(frame_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        frame_metrics.append(metrics)

        mask = result["metric_mask"]
        errors = result["absolute_error"][mask].astype(np.float64)
        gt_values = result["gt_depth"][mask].astype(np.float64)
        absolute_error_sum += float(errors.sum())
        squared_error_sum += float(np.square(errors).sum())
        relative_error_sum += float(
            (errors / np.maximum(gt_values, 1e-6)).sum()
        )
        valid_pixel_count += int(mask.sum())
        print(
            f"[{progress}/{len(sample_indices)}] frame{metrics['frame']} "
            f"MAPE={metrics['mape_percent']:.4f}% "
            f"MAE={metrics['mae_m']:.4f}m"
        )

    if not frame_metrics:
        raise RuntimeError("No sequence frames contained valid evaluation pixels")

    fields = list(frame_metrics[0].keys())
    with open(
        cli.output_dir / "metrics_by_frame.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_metrics)
    ranked_metrics = sorted(
        frame_metrics, key=lambda item: item["mape_percent"]
    )
    with open(
        cli.output_dir / "metrics_ranked_by_mape.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["mape_rank"] + fields
        )
        writer.writeheader()
        for rank, metrics in enumerate(ranked_metrics, start=1):
            writer.writerow({"mape_rank": rank, **metrics})
    with open(
        cli.output_dir / "metrics_by_frame.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(frame_metrics, handle, indent=2)

    inference_times = [
        metrics["inference_time_ms"] for metrics in frame_metrics
    ]
    summary = {
        "sequences": selected_sequences,
        "checkpoint": str(cli.checkpoint),
        "input_mode": {
            "split": cli.split,
            "images": (
                f"{cli.image_subdir}/"
                f"{cli.left_dirname},{cli.right_dirname}"
            ),
            "depth_gt": cli.depth_subdir,
            "lidar": cli.lidar_source,
            "border_crop_fraction": cli.border_crop_fraction,
            "depth_percentile_interval": [0.05, 0.95],
        },
        "paired_frames": len(sample_indices),
        "evaluated_frames": len(frame_metrics),
        "skipped_frames": skipped_frames,
        "valid_pixels_total": valid_pixel_count,
        "pixel_weighted_metrics": {
            "mae_m": absolute_error_sum / valid_pixel_count,
            "rmse_m": float(
                np.sqrt(squared_error_sum / valid_pixel_count)
            ),
            "mape_percent": relative_error_sum / valid_pixel_count * 100.0,
        },
        "per_frame_metrics": {
            "mae_m": distribution(
                [metrics["mae_m"] for metrics in frame_metrics]
            ),
            "rmse_m": distribution(
                [metrics["rmse_m"] for metrics in frame_metrics]
            ),
            "mape_percent": distribution(
                [metrics["mape_percent"] for metrics in frame_metrics]
            ),
        },
        "inference_time_ms": distribution(inference_times),
        "inference_fps_from_mean": 1000.0 / float(np.mean(inference_times)),
        "best_mape_frame": ranked_metrics[0],
        "worst_mape_frame": ranked_metrics[-1],
    }
    with open(cli.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
