#!/usr/bin/env python3
"""Summarize LUNA ground-truth disparity at the resolution used by training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_THRESHOLDS = (128, 160, 192, 224, 256, 288, 320, 384, 512)
QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute disparity statistics from uint16 LUNA depth PNGs after "
            "the same resize and border crop used by LunaOrganized."
        )
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--depth-subdir", default="depth_gt_rectified")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=(768, 1024))
    parser.add_argument("--border-crop-fraction", type=float, default=0.1)
    parser.add_argument("--depth-scale", type=float, default=1.0 / 256.0)
    parser.add_argument("--camera-key", default="Cam_Rect_L")
    parser.add_argument("--exclude-sequences", nargs="*", default=[])
    parser.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--histogram-bin-width", type=float, default=0.01)
    parser.add_argument("--histogram-max", type=float, default=2048.0)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def load_conversion_rate(sequence: Path, camera_key: str) -> float:
    calibration = sequence / "calibration"
    with (calibration / "intrinsics.json").open(encoding="utf-8") as handle:
        intrinsics = json.load(handle)
    with (calibration / "extrinsics.json").open(encoding="utf-8") as handle:
        extrinsics = json.load(handle)
    focal_length = float(intrinsics[camera_key]["K"][0][0])
    translation_mm = np.asarray(
        extrinsics["Cam_R_to_Cam_L"]["T"], dtype=np.float64
    )
    baseline_m = float(np.linalg.norm(translation_mm) / 1000.0)
    return focal_length * baseline_m


def histogram_quantile(
    histogram: np.ndarray, bin_width: float, quantile: float
) -> float:
    target = quantile * int(histogram.sum())
    index = int(np.searchsorted(np.cumsum(histogram), target, side="left"))
    return (index + 0.5) * bin_width


def summarize_histogram(
    histogram: np.ndarray,
    bin_width: float,
    valid_pixels: int,
    value_sum: float,
    minimum: float,
    maximum: float,
    thresholds: list[int],
) -> dict:
    result = {
        "valid_pixels": valid_pixels,
        "min_px": minimum,
        "mean_px": value_sum / valid_pixels,
        "max_px": maximum,
        "quantiles_px": {
            f"p{100 * q:g}": histogram_quantile(histogram, bin_width, q)
            for q in QUANTILES
        },
        "candidate_max_disp": {},
    }
    cumulative = np.cumsum(histogram)
    for threshold in thresholds:
        below_index = min(int(np.ceil(threshold / bin_width)), len(histogram))
        below = int(cumulative[below_index - 1]) if below_index else 0
        result["candidate_max_disp"][str(threshold)] = {
            "kept_pixels": below,
            "kept_percent": 100.0 * below / valid_pixels,
            "dropped_pixels": valid_pixels - below,
            "dropped_percent": 100.0 * (valid_pixels - below) / valid_pixels,
        }
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.border_crop_fraction < 0.5:
        raise ValueError("--border-crop-fraction must be in [0, 0.5)")
    if args.histogram_bin_width <= 0 or args.histogram_max <= 0:
        raise ValueError("Histogram width and maximum must be positive")

    excluded = set(args.exclude_sequences)
    sequences = [
        path
        for path in sorted(args.root.iterdir())
        if path.is_dir()
        and path.name not in excluded
        and (path / args.depth_subdir).is_dir()
        and (path / "calibration").is_dir()
    ]
    if not sequences:
        raise RuntimeError("No matching sequence directories found")

    bin_count = int(np.ceil(args.histogram_max / args.histogram_bin_width))
    combined_histogram = np.zeros(bin_count, dtype=np.int64)
    combined_valid = 0
    combined_sum = 0.0
    combined_min = float("inf")
    combined_max = float("-inf")
    frame_maxima: list[float] = []
    sequence_results = {}

    for sequence in sequences:
        paths = sorted((sequence / args.depth_subdir).glob("*.png"))
        if not paths:
            continue
        native_conversion = load_conversion_rate(sequence, args.camera_key)
        sequence_histogram = np.zeros_like(combined_histogram)
        sequence_valid = 0
        sequence_sum = 0.0
        sequence_min = float("inf")
        sequence_max = float("-inf")
        sequence_frame_maxima = []

        for path in paths:
            depth_native = np.asarray(Image.open(path), dtype=np.float32)
            native_height, native_width = depth_native.shape
            target_height, target_width = args.resize
            if (native_height, native_width) != (target_height, target_width):
                depth_raw = np.asarray(
                    Image.fromarray(depth_native).resize(
                        (target_width, target_height), Image.Resampling.NEAREST
                    ),
                    dtype=np.float32,
                )
            else:
                depth_raw = depth_native

            crop_y = int(round(target_height * args.border_crop_fraction))
            crop_x = int(round(target_width * args.border_crop_fraction))
            if crop_y or crop_x:
                depth_raw = depth_raw[
                    crop_y : target_height - crop_y,
                    crop_x : target_width - crop_x,
                ]

            depth_m = depth_raw * args.depth_scale
            valid = depth_m > 0.01
            if not np.any(valid):
                continue
            horizontal_scale = target_width / float(native_width)
            conversion_rate = native_conversion * horizontal_scale
            disparity = conversion_rate / depth_m[valid]
            local_min = float(disparity.min())
            local_max = float(disparity.max())
            if local_max >= args.histogram_max:
                raise RuntimeError(
                    f"{path}: disparity {local_max:.3f}px exceeds "
                    f"--histogram-max {args.histogram_max:g}"
                )
            histogram, _ = np.histogram(
                disparity,
                bins=bin_count,
                range=(0.0, args.histogram_max),
            )
            sequence_histogram += histogram
            sequence_valid += int(disparity.size)
            sequence_sum += float(disparity.sum(dtype=np.float64))
            sequence_min = min(sequence_min, local_min)
            sequence_max = max(sequence_max, local_max)
            sequence_frame_maxima.append(local_max)

        if not sequence_valid:
            continue
        combined_histogram += sequence_histogram
        combined_valid += sequence_valid
        combined_sum += sequence_sum
        combined_min = min(combined_min, sequence_min)
        combined_max = max(combined_max, sequence_max)
        frame_maxima.extend(sequence_frame_maxima)
        sequence_summary = summarize_histogram(
            sequence_histogram,
            args.histogram_bin_width,
            sequence_valid,
            sequence_sum,
            sequence_min,
            sequence_max,
            args.thresholds,
        )
        sequence_summary["frames"] = len(paths)
        sequence_summary["conversion_rate_native_px_m"] = native_conversion
        sequence_results[sequence.name] = sequence_summary

    if not combined_valid:
        raise RuntimeError("No valid depth pixels found")

    result = summarize_histogram(
        combined_histogram,
        args.histogram_bin_width,
        combined_valid,
        combined_sum,
        combined_min,
        combined_max,
        args.thresholds,
    )
    frame_maxima_array = np.asarray(frame_maxima, dtype=np.float64)
    result.update(
        {
            "root": str(args.root),
            "depth_subdir": args.depth_subdir,
            "resize_hw": list(args.resize),
            "border_crop_fraction": args.border_crop_fraction,
            "depth_scale": args.depth_scale,
            "excluded_sequences": sorted(excluded),
            "sequences": len(sequence_results),
            "frames": len(frame_maxima),
            "frame_max_quantiles_px": {
                f"p{100 * q:g}": float(np.quantile(frame_maxima_array, q))
                for q in QUANTILES
            },
            "per_sequence": sequence_results,
        }
    )

    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    print(serialized)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
