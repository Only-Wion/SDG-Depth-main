#!/usr/bin/env python3
"""Evaluate a Luna depth prediction only at projected FAST-LIO pixels."""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.luna_dataset import LunaOrganized
from core.sdg_depth.net import SDGDepth
from scripts.infer_luna_top5 import infer_sample
from scripts.infer_one_luna import model_and_dataset_args


POINT_FIELDS = [
    "sequence",
    "frame",
    "u",
    "v",
    "prediction_depth_m",
    "fastlio_depth_m",
    "depth_gt_m",
    "prediction_vs_gt_abs_error_m",
    "prediction_vs_gt_relative_error",
    "prediction_vs_gt_relative_error_percent",
    "prediction_vs_fastlio_abs_error_m",
    "prediction_vs_fastlio_relative_error",
    "prediction_vs_fastlio_relative_error_percent",
    "fastlio_vs_gt_abs_error_m",
    "fastlio_vs_gt_relative_error",
    "fastlio_vs_gt_relative_error_percent",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Luna inference and report prediction/FAST-LIO errors at every "
            "projected FAST-LIO pixel that also has valid depth ground truth."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-subdir", default="images_rectified")
    parser.add_argument("--left-dirname", default="left")
    parser.add_argument("--right-dirname", default="right")
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
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


def error_metrics(estimate, reference):
    absolute = np.abs(estimate - reference).astype(np.float64)
    relative = absolute / np.maximum(reference, 1e-6)
    return absolute, relative


def aggregate_metrics(estimate, reference):
    absolute, relative = error_metrics(estimate, reference)
    return {
        "mae_m": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.mean(np.square(absolute)))),
        "relative_error": distribution(relative),
        "relative_error_percent": distribution(relative * 100.0),
        "absolute_error_m": distribution(absolute),
    }


def main():
    cli = parse_args()
    if cli.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    cli.output_dir.mkdir(parents=True, exist_ok=True)

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
    sample_indices = [
        index
        for index, sample in enumerate(dataset.extra_info)
        if sample["sequence"] == cli.sequence
    ]
    if not sample_indices:
        raise FileNotFoundError(
            f"No valid FAST-LIO-paired samples found for sequence {cli.sequence}"
        )

    device = torch.device("cuda")
    model = SDGDepth(args.max_disp, use_concat_volume=True, args=args).to(device)
    checkpoint = torch.load(cli.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    first_sample = dataset[sample_indices[0]]
    for _ in range(cli.warmup):
        infer_sample(model, first_sample, device)

    point_path = cli.output_dir / "fastlio_point_errors.csv.gz"
    frame_rows = []
    all_prediction = []
    all_fastlio = []
    all_gt = []
    projected_fastlio_total = 0
    missing_gt_total = 0
    invalid_prediction_total = 0

    with gzip.open(point_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_FIELDS)
        writer.writeheader()

        for progress, sample_index in enumerate(sample_indices, start=1):
            sample = dataset[sample_index]
            paths, _, _, flow_gt, valid_gt, hint, conversion_rate = sample
            result = infer_sample(model, sample, device)
            if result is None:
                raise RuntimeError(f"Inference produced no result for {paths[0]}")

            frame = Path(paths[0]).stem
            prediction = result["prediction"].astype(np.float64)
            flow = flow_gt.squeeze().numpy().astype(np.float64)
            sparse_disparity = hint.squeeze().numpy().astype(np.float64)
            conversion = float(conversion_rate)

            fastlio_mask = np.isfinite(sparse_disparity) & (sparse_disparity > 0)
            gt_mask = (
                valid_gt.bool().numpy()
                & np.isfinite(flow)
                & (flow > 0)
            )
            prediction_mask = np.isfinite(prediction) & (prediction > 0)
            valid = fastlio_mask & gt_mask & prediction_mask

            projected_count = int(fastlio_mask.sum())
            missing_gt_count = int((fastlio_mask & ~gt_mask).sum())
            invalid_prediction_count = int(
                (fastlio_mask & gt_mask & ~prediction_mask).sum()
            )
            projected_fastlio_total += projected_count
            missing_gt_total += missing_gt_count
            invalid_prediction_total += invalid_prediction_count

            fastlio_depth = conversion / sparse_disparity[valid]
            gt_depth = conversion / flow[valid]
            prediction_depth = prediction[valid]
            ys, xs = np.nonzero(valid)

            pred_gt_abs, pred_gt_rel = error_metrics(
                prediction_depth, gt_depth
            )
            pred_lidar_abs, pred_lidar_rel = error_metrics(
                prediction_depth, fastlio_depth
            )
            lidar_gt_abs, lidar_gt_rel = error_metrics(
                fastlio_depth, gt_depth
            )

            for index in range(len(xs)):
                writer.writerow(
                    {
                        "sequence": cli.sequence,
                        "frame": frame,
                        "u": int(xs[index]),
                        "v": int(ys[index]),
                        "prediction_depth_m": prediction_depth[index],
                        "fastlio_depth_m": fastlio_depth[index],
                        "depth_gt_m": gt_depth[index],
                        "prediction_vs_gt_abs_error_m": pred_gt_abs[index],
                        "prediction_vs_gt_relative_error": pred_gt_rel[index],
                        "prediction_vs_gt_relative_error_percent": (
                            pred_gt_rel[index] * 100.0
                        ),
                        "prediction_vs_fastlio_abs_error_m": (
                            pred_lidar_abs[index]
                        ),
                        "prediction_vs_fastlio_relative_error": (
                            pred_lidar_rel[index]
                        ),
                        "prediction_vs_fastlio_relative_error_percent": (
                            pred_lidar_rel[index] * 100.0
                        ),
                        "fastlio_vs_gt_abs_error_m": lidar_gt_abs[index],
                        "fastlio_vs_gt_relative_error": lidar_gt_rel[index],
                        "fastlio_vs_gt_relative_error_percent": (
                            lidar_gt_rel[index] * 100.0
                        ),
                    }
                )

            frame_rows.append(
                {
                    "sequence": cli.sequence,
                    "frame": frame,
                    "projected_fastlio_pixels": projected_count,
                    "evaluated_fastlio_pixels": int(valid.sum()),
                    "fastlio_pixels_missing_depth_gt": missing_gt_count,
                    "fastlio_pixels_invalid_prediction": invalid_prediction_count,
                    "prediction_vs_gt_mae_m": float(pred_gt_abs.mean()),
                    "prediction_vs_gt_rmse_m": float(
                        np.sqrt(np.mean(np.square(pred_gt_abs)))
                    ),
                    "prediction_vs_gt_mean_relative_error_percent": float(
                        pred_gt_rel.mean() * 100.0
                    ),
                    "prediction_vs_fastlio_mae_m": float(
                        pred_lidar_abs.mean()
                    ),
                    "prediction_vs_fastlio_rmse_m": float(
                        np.sqrt(np.mean(np.square(pred_lidar_abs)))
                    ),
                    "prediction_vs_fastlio_mean_relative_error_percent": float(
                        pred_lidar_rel.mean() * 100.0
                    ),
                    "fastlio_vs_gt_mae_m": float(lidar_gt_abs.mean()),
                    "fastlio_vs_gt_rmse_m": float(
                        np.sqrt(np.mean(np.square(lidar_gt_abs)))
                    ),
                    "fastlio_vs_gt_mean_relative_error_percent": float(
                        lidar_gt_rel.mean() * 100.0
                    ),
                }
            )
            all_prediction.append(prediction_depth)
            all_fastlio.append(fastlio_depth)
            all_gt.append(gt_depth)
            print(
                f"[{progress}/{len(sample_indices)}] frame{frame}: "
                f"points={int(valid.sum())}, "
                f"prediction-vs-GT relative error={pred_gt_rel.mean() * 100:.4f}%, "
                f"FAST-LIO-vs-GT relative error={lidar_gt_rel.mean() * 100:.4f}%"
            )

    prediction = np.concatenate(all_prediction)
    fastlio = np.concatenate(all_fastlio)
    gt = np.concatenate(all_gt)

    frame_path = cli.output_dir / "fastlio_point_metrics_by_frame.csv"
    with frame_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    summary = {
        "sequence": cli.sequence,
        "checkpoint": str(cli.checkpoint),
        "root": str(cli.root),
        "coordinate_convention": {
            "u": "zero-based column in the resized and center-cropped model input",
            "v": "zero-based row in the resized and center-cropped model input",
        },
        "scope": (
            "Projected FAST-LIO pixels with valid depth_gt and a finite, "
            "positive prediction; no depth-percentile filtering."
        ),
        "frames": len(frame_rows),
        "projected_fastlio_pixels": projected_fastlio_total,
        "evaluated_fastlio_pixels": int(gt.size),
        "fastlio_pixels_missing_depth_gt": missing_gt_total,
        "fastlio_pixels_invalid_prediction": invalid_prediction_total,
        "prediction_vs_depth_gt": aggregate_metrics(prediction, gt),
        "prediction_vs_fastlio": aggregate_metrics(prediction, fastlio),
        "fastlio_vs_depth_gt": aggregate_metrics(fastlio, gt),
        "outputs": {
            "per_point_csv_gzip": str(point_path),
            "per_frame_csv": str(frame_path),
        },
    }
    summary_path = cli.output_dir / "fastlio_point_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
