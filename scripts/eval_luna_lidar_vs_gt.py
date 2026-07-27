#!/usr/bin/env python3
"""Evaluate projected Luna raw or FAST-LIO LiDAR depth against dense depth GT."""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.luna_dataset import LunaOrganized
from scripts.infer_one_luna import model_and_dataset_args


POINT_FIELDS = [
    "sequence",
    "frame",
    "u",
    "v",
    "lidar_source",
    "lidar_depth_m",
    "depth_gt_m",
    "absolute_error_m",
    "relative_error",
    "relative_error_percent",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--lidar-source", choices=("raw", "fastlio"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-subdir", default="images_rectified")
    parser.add_argument("--left-dirname", default="left")
    parser.add_argument("--right-dirname", default="right")
    return parser.parse_args()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def main():
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    args = model_and_dataset_args()
    args.luna_exclude_sequences = []
    args.luna_val_fraction = 0.0
    args.luna_test_fraction = 0.0
    args.luna_image_subdir = cli.image_subdir
    args.luna_left_dirname = cli.left_dirname
    args.luna_right_dirname = cli.right_dirname
    args.luna_lidar_source = cli.lidar_source
    dataset = LunaOrganized(
        aug_params={},
        root=str(cli.root),
        image_set="all",
        args=args,
    )
    sample_indices = [
        index
        for index, sample in enumerate(dataset.extra_info)
        if sample["sequence"] == cli.sequence
    ]
    if not sample_indices:
        raise FileNotFoundError(
            f"No valid {cli.lidar_source}-LiDAR-paired samples found for "
            f"sequence {cli.sequence}"
        )

    prefix = f"{cli.lidar_source}_lidar"
    point_path = cli.output_dir / f"{prefix}_point_errors.csv.gz"
    frame_path = cli.output_dir / f"{prefix}_point_metrics_by_frame.csv"
    summary_path = cli.output_dir / f"{prefix}_point_summary.json"

    frame_rows = []
    all_lidar_depth = []
    all_gt_depth = []
    projected_total = 0
    missing_gt_total = 0

    with gzip.open(point_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_FIELDS)
        writer.writeheader()

        for progress, sample_index in enumerate(sample_indices, start=1):
            paths, _, _, flow_gt, valid_gt, hint, conversion_rate = dataset[
                sample_index
            ]
            frame = Path(paths[0]).stem
            flow = flow_gt.squeeze().numpy().astype(np.float64)
            sparse_disparity = hint.squeeze().numpy().astype(np.float64)
            conversion = float(conversion_rate)

            lidar_mask = (
                np.isfinite(sparse_disparity) & (sparse_disparity > 0)
            )
            gt_mask = (
                valid_gt.bool().numpy()
                & np.isfinite(flow)
                & (flow > 0)
            )
            valid = lidar_mask & gt_mask
            if not np.any(valid):
                raise RuntimeError(
                    f"No {cli.lidar_source} pixels with valid GT for {paths[0]}"
                )

            projected_count = int(lidar_mask.sum())
            missing_gt_count = int((lidar_mask & ~gt_mask).sum())
            projected_total += projected_count
            missing_gt_total += missing_gt_count

            lidar_depth = conversion / sparse_disparity[valid]
            gt_depth = conversion / flow[valid]
            absolute_error = np.abs(lidar_depth - gt_depth)
            relative_error = absolute_error / np.maximum(gt_depth, 1e-6)
            ys, xs = np.nonzero(valid)

            for index in range(len(xs)):
                writer.writerow(
                    {
                        "sequence": cli.sequence,
                        "frame": frame,
                        "u": int(xs[index]),
                        "v": int(ys[index]),
                        "lidar_source": cli.lidar_source,
                        "lidar_depth_m": lidar_depth[index],
                        "depth_gt_m": gt_depth[index],
                        "absolute_error_m": absolute_error[index],
                        "relative_error": relative_error[index],
                        "relative_error_percent": relative_error[index] * 100.0,
                    }
                )

            frame_rows.append(
                {
                    "sequence": cli.sequence,
                    "frame": frame,
                    "lidar_source": cli.lidar_source,
                    "projected_lidar_pixels": projected_count,
                    "evaluated_lidar_pixels": int(valid.sum()),
                    "lidar_pixels_missing_depth_gt": missing_gt_count,
                    "mae_m": float(absolute_error.mean()),
                    "rmse_m": float(
                        np.sqrt(np.mean(np.square(absolute_error)))
                    ),
                    "mean_relative_error_percent": float(
                        relative_error.mean() * 100.0
                    ),
                    "median_relative_error_percent": float(
                        np.median(relative_error) * 100.0
                    ),
                    "p95_relative_error_percent": float(
                        np.quantile(relative_error, 0.95) * 100.0
                    ),
                }
            )
            all_lidar_depth.append(lidar_depth)
            all_gt_depth.append(gt_depth)
            print(
                f"[{progress}/{len(sample_indices)}] frame{frame}: "
                f"points={int(valid.sum())}, "
                f"relative error={relative_error.mean() * 100.0:.4f}%"
            )

    lidar_depth = np.concatenate(all_lidar_depth)
    gt_depth = np.concatenate(all_gt_depth)
    absolute_error = np.abs(lidar_depth - gt_depth)
    relative_error = absolute_error / np.maximum(gt_depth, 1e-6)

    with frame_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    summary = {
        "sequence": cli.sequence,
        "root": str(cli.root),
        "lidar_source": cli.lidar_source,
        "input_mode": {
            "images": (
                f"{cli.image_subdir}/"
                f"{cli.left_dirname},{cli.right_dirname}"
            ),
            "depth_gt": args.luna_depth_subdir,
            "border_crop_fraction": args.luna_border_crop_fraction,
        },
        "scope": (
            f"Projected {cli.lidar_source} LiDAR pixels with valid depth_gt; "
            "no depth-percentile filtering."
        ),
        "frames": len(frame_rows),
        "projected_lidar_pixels": projected_total,
        "evaluated_lidar_pixels": int(gt_depth.size),
        "lidar_pixels_missing_depth_gt": missing_gt_total,
        "mae_m": float(absolute_error.mean()),
        "rmse_m": float(np.sqrt(np.mean(np.square(absolute_error)))),
        "absolute_error_m": distribution(absolute_error),
        "relative_error": distribution(relative_error),
        "relative_error_percent": distribution(relative_error * 100.0),
        "outputs": {
            "per_point_csv_gzip": str(point_path),
            "per_frame_csv": str(frame_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
