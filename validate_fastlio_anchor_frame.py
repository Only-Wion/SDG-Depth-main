#!/usr/bin/env python3
"""Rebuild and validate FAST-LIO Tier-B anchors for one RGB frame.

This script validates FAST-LIO anchors as the subject under test.  It reports:

1. Self consistency:
   rebuilt FAST-LIO Tier-B anchors versus the final dense GT at Tier-B pixels.
   This detects projection-grid, export, remap, and quantization problems, but
   is circular because Tier-B anchors overwrite the final GT.

2. Physical consistency:
   rebuilt FAST-LIO anchors versus strict raw-LiDAR measurements within a 7x7
   neighborhood.  Raw LiDAR is used only as an independent reference; the
   evaluated subject remains the FAST-LIO multi-sweep anchor map.

Both raw-distorted and same-K undistorted projection grids are tested.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import evaluate_depth_lidar_frame as core  # noqa: E402
import generate_first50 as generator  # noqa: E402


DEFAULT_ROOT = Path(
    r"F:\502_luna_data_processed\batch_organized\2025-06-26-23-29-56"
)
DEFAULT_OUTPUT = WORKSPACE / "analysis" / "fastlio_anchor_validation_2025-06-26-23-29-56"


def load_timestamp(root: Path, frame_index: int) -> int:
    rows = core.load_image_rows(root / "images" / "timestamps.tsv")
    selected = [row for row in rows if int(row["frame_index"]) == frame_index]
    if len(selected) != 1:
        raise ValueError(f"frame {frame_index} has {len(selected)} timestamp rows")
    return int(selected[0]["timestamp_ns"])


def load_fastlio_rows(root: Path) -> list[dict]:
    path = root / "lidar_fastlio" / "timestamps.tsv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    rows.sort(key=lambda row: int(row["header_time_ns"]))
    times = np.asarray([int(row["header_time_ns"]) for row in rows], dtype=np.int64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("FAST-LIO timestamps are not strictly increasing")
    return rows


def selected_indices(
    rows: list[dict],
    target_time_ns: int,
    *,
    max_age_ms: float,
    maximum_scans: int,
) -> list[int]:
    times = np.asarray([int(row["header_time_ns"]) for row in rows], dtype=np.int64)
    age_ms = np.abs(times - np.int64(target_time_ns)).astype(np.float64) / 1e6
    candidates = np.flatnonzero(age_ms <= float(max_age_ms))
    order = np.argsort(age_ms[candidates], kind="stable")
    chosen = candidates[order[: int(maximum_scans)]]
    return sorted(int(index) for index in chosen)


def build_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def map_cloud_to_target_camera(
    map_xyz: np.ndarray,
    target_pose: tuple[np.ndarray, np.ndarray],
    T_imu_lidar: np.ndarray,
    T_camera_lidar: np.ndarray,
) -> np.ndarray:
    target_rotation, target_position = target_pose
    world = np.asarray(map_xyz, dtype=np.float64)
    target_body = (world - target_position) @ target_rotation
    R_imu_lidar = T_imu_lidar[:3, :3]
    t_imu_lidar = T_imu_lidar[:3, 3]
    target_lidar = (target_body - t_imu_lidar) @ R_imu_lidar
    return (
        target_lidar @ T_camera_lidar[:3, :3].T
        + T_camera_lidar[:3, 3]
    )


def rebuild_fastlio_anchors(
    *,
    root: Path,
    target_time_ns: int,
    trajectory: generator.FastLioTrajectory,
    rows: list[dict],
    calibration: dict[str, np.ndarray],
    T_imu_lidar: np.ndarray,
    settings: dict,
    projection_mode: str,
    shape: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict]:
    indices = selected_indices(
        rows,
        target_time_ns,
        max_age_ms=float(settings["max_age_ms"]),
        maximum_scans=int(settings["maximum_scans"]),
    )
    if len(indices) < int(settings["minimum_support"]):
        raise RuntimeError("not enough FAST-LIO clouds for multi-sweep support")
    target_pose = trajectory.pose(target_time_ns)
    distortion = (
        calibration["distortion"]
        if projection_mode == "raw_distorted"
        else np.zeros_like(calibration["distortion"])
    )
    depth_maps: list[np.ndarray] = []
    ages_ms: list[float] = []
    projected_total = 0
    zbuffer_total = 0
    selected_clouds: list[dict] = []
    for index in indices:
        row = rows[index]
        path = root / "lidar_fastlio" / row["filename"]
        with np.load(path, allow_pickle=False) as item:
            map_xyz = np.asarray(item["xyz"], dtype=np.float32)
            timestamp_ns = int(item["header_time_ns"])
        if timestamp_ns != int(row["header_time_ns"]):
            raise ValueError(f"timestamp mismatch in {path}")
        camera_xyz = map_cloud_to_target_camera(
            map_xyz,
            target_pose,
            T_imu_lidar,
            calibration["T_camera_lidar"],
        )
        depth_map, projected, zbuffered = generator.zbuffer_project(
            camera_xyz,
            calibration["K"],
            distortion,
            shape,
            min_depth_m=0.5,
            max_depth_m=100.0,
        )
        age_ms = (timestamp_ns - int(target_time_ns)) / 1e6
        depth_maps.append(depth_map)
        ages_ms.append(age_ms)
        projected_total += projected
        zbuffer_total += zbuffered
        selected_clouds.append(
            {
                "index": int(index),
                "filename": row["filename"],
                "timestamp_ns": timestamp_ns,
                "age_ms": age_ms,
                "point_count": int(len(map_xyz)),
                "camera_depth_selected_count": int(projected),
                "zbuffer_pixel_count": int(zbuffered),
            }
        )

    combined = np.full(shape, np.inf, dtype=np.float32)
    winning_age = np.full(shape, np.nan, dtype=np.float32)
    for depth_map, age_ms in zip(depth_maps, ages_ms, strict=True):
        update = np.isfinite(depth_map) & (depth_map < combined)
        combined[update] = depth_map[update]
        winning_age[update] = np.float32(age_ms)
    combined[~np.isfinite(combined)] = np.nan

    radius = int(settings["support_radius_px"])
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    support = np.zeros(shape, dtype=np.uint8)
    valid_combined = np.isfinite(combined)
    for depth_map in depth_maps:
        inverse = np.zeros(shape, dtype=np.float32)
        valid = np.isfinite(depth_map) & (depth_map > 0)
        inverse[valid] = 1.0 / depth_map[valid]
        local_inverse = cv2.dilate(inverse, kernel)
        local_depth = np.full(shape, np.nan, dtype=np.float32)
        local_valid = local_inverse > 0
        local_depth[local_valid] = 1.0 / local_inverse[local_valid]
        tolerance = np.maximum(
            np.float32(settings["absolute_tolerance_m"]),
            np.float32(settings["relative_tolerance"]) * combined,
        )
        consistent = valid_combined & local_valid
        consistent &= np.abs(local_depth - combined) <= tolerance
        support[consistent] += np.uint8(1)
    reliable = valid_combined & (support >= int(settings["minimum_support"]))
    depth = np.where(reliable, combined, np.nan).astype(np.float32)
    age = np.where(reliable, winning_age, np.nan).astype(np.float32)
    support[~reliable] = 0
    return (
        {"depth_m": depth, "support": support, "source_age_ms": age},
        {
            "projection_mode": projection_mode,
            "selected_clouds": selected_clouds,
            "cloud_count": len(indices),
            "projected_point_count": int(projected_total),
            "zbuffer_pixel_count": int(zbuffer_total),
            "reliable_anchor_count": int(np.count_nonzero(reliable)),
        },
    )


def build_strict_direct_map(
    *,
    lidar: core.RawLidar,
    target_time_ns: int,
    calibration: dict[str, np.ndarray],
    shape: tuple[int, int],
    mode: str,
) -> np.ndarray:
    projection = core.build_projection(
        lidar,
        target_time_ns,
        calibration["T_camera_lidar"],
        calibration["K"],
        calibration["distortion"],
        calibration["rectification"],
        calibration["rectified_K"],
        shape,
        inner_abs_dt_ms=0.0,
        outer_abs_dt_ms=6.0,
        mode=mode,
    )
    depth = np.full(shape, np.nan, dtype=np.float32)
    depth[projection["y"], projection["x"]] = projection["z_m"]
    return depth


def remap_depth_to_same_k_undistorted(
    encoded_depth: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = encoded_depth.shape
    map_x, map_y = cv2.initUndistortRectifyMap(
        K,
        distortion,
        None,
        K,
        (width, height),
        cv2.CV_32FC1,
    )
    remapped_encoded = cv2.remap(
        encoded_depth,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    depth = remapped_encoded.astype(np.float32) / np.float32(256.0)
    depth[remapped_encoded == 0] = np.nan
    return depth, remapped_encoded


def self_consistency_metrics(
    fastlio_depth: np.ndarray,
    final_gt: np.ndarray,
    tier_b_mask: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    valid = np.asarray(tier_b_mask, dtype=bool)
    valid &= np.isfinite(fastlio_depth) & (fastlio_depth > 0)
    valid &= np.isfinite(final_gt) & (final_gt > 0)
    y, x = np.nonzero(valid)
    fast = fastlio_depth[y, x].astype(np.float64)
    gt = final_gt[y, x].astype(np.float64)
    error = np.abs(gt - fast)
    abs_rel = error / fast
    tolerance = 0.5 / 256.0 + 1e-6
    metrics = {
        "paired_pixel_count": int(len(x)),
        "median_absolute_error_m": (
            float(np.median(error)) if len(error) else float("nan")
        ),
        "median_abs_rel": (
            float(np.median(abs_rel)) if len(abs_rel) else float("nan")
        ),
        "p90_abs_rel": (
            float(np.percentile(abs_rel, 90)) if len(abs_rel) else float("nan")
        ),
        "within_10_percent_fraction": (
            float(np.mean(abs_rel <= 0.10)) if len(abs_rel) else float("nan")
        ),
        "quantization_level_match_fraction": (
            float(np.mean(error <= tolerance)) if len(error) else float("nan")
        ),
    }
    return metrics, {"x": x, "y": y, "abs_rel": abs_rel}


def error_summary(reference: np.ndarray, candidate: np.ndarray) -> dict:
    valid = np.isfinite(reference) & (reference > 0)
    valid &= np.isfinite(candidate) & (candidate > 0)
    count = int(np.count_nonzero(valid))
    if not count:
        return {
            "pair_count": 0,
            "median_abs_rel": float("nan"),
            "p90_abs_rel": float("nan"),
            "p95_abs_rel": float("nan"),
            "median_absolute_error_m": float("nan"),
            "p90_absolute_error_m": float("nan"),
            "within_5_percent_fraction": float("nan"),
            "within_10_percent_fraction": float("nan"),
        }
    absolute_error = np.abs(candidate[valid] - reference[valid])
    abs_rel = absolute_error / reference[valid]
    return {
        "pair_count": count,
        "median_abs_rel": float(np.median(abs_rel)),
        "p90_abs_rel": float(np.percentile(abs_rel, 90)),
        "p95_abs_rel": float(np.percentile(abs_rel, 95)),
        "median_absolute_error_m": float(np.median(absolute_error)),
        "p90_absolute_error_m": float(np.percentile(absolute_error, 90)),
        "within_5_percent_fraction": float(np.mean(abs_rel <= 0.05)),
        "within_10_percent_fraction": float(np.mean(abs_rel <= 0.10)),
    }


def physical_consistency_metrics(
    direct_depth: np.ndarray,
    fastlio_depth: np.ndarray,
    *,
    radius_px: int,
) -> dict:
    center = error_summary(direct_depth, fastlio_depth)
    query_mask = np.isfinite(direct_depth) & (direct_depth > 0)
    local_fastlio, paired = generator.local_depth_at_queries(
        fastlio_depth,
        query_mask,
        radius_px,
    )
    local_reference = np.where(paired, direct_depth, np.nan)
    local = error_summary(local_reference, local_fastlio)
    return {
        "reference": "strict raw LiDAR |dt| <= 6 ms",
        "subject": "FAST-LIO Tier-B multi-sweep anchors",
        "center_pixel": center,
        f"local_{2 * radius_px + 1}x{2 * radius_px + 1}": local,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--frame-index", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve() / f"frame_{args.frame_index:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = load_timestamp(root, args.frame_index)

    calibration = core.load_calibration(root)
    manifest_path = (
        WORKSPACE / "runs" / root.name / "full" / "batch_manifest.json"
    )
    batch_manifest = core.load_json(manifest_path)
    parameters = batch_manifest["parameters"]
    multi_sweep = parameters["multi_sweep"]
    lidar_to_imu = parameters["fastlio_lidar_to_imu"]
    T_imu_lidar = build_transform(
        np.asarray(lidar_to_imu["R"], dtype=np.float64),
        np.asarray(lidar_to_imu["t_m"], dtype=np.float64),
    )

    trajectory = generator.FastLioTrajectory(
        root / "lidar_fastlio" / "trajectory.npz"
    )
    rows = load_fastlio_rows(root)
    if len(rows) != len(trajectory.time_ns):
        raise ValueError("FAST-LIO point-cloud and trajectory lengths differ")
    lidar = core.RawLidar(root / "lidar_raw")

    encoded_gt = cv2.imread(
        str(root / "depth_gt" / f"{args.frame_index:06d}.png"),
        cv2.IMREAD_UNCHANGED,
    )
    if encoded_gt is None or encoded_gt.dtype != np.uint16:
        raise ValueError("failed to load uint16 depth GT")
    current_gt = encoded_gt.astype(np.float32) / np.float32(256.0)
    current_gt[encoded_gt == 0] = np.nan
    undistorted_gt, remapped_encoded = remap_depth_to_same_k_undistorted(
        encoded_gt,
        calibration["K"],
        calibration["distortion"],
    )
    cv2.imwrite(str(output_dir / "depth_gt_runtime_remapped_undistorted.png"), remapped_encoded)

    organized_rgb = cv2.imread(
        str(root / "images" / "left" / f"{args.frame_index:06d}.png"),
        cv2.IMREAD_COLOR,
    )
    if organized_rgb is None:
        raise FileNotFoundError("organized RGB")
    shape = current_gt.shape

    report_modes: dict[str, dict] = {}
    for mode in ("raw_distorted", "undistorted"):
        fastlio, build_metrics = rebuild_fastlio_anchors(
            root=root,
            target_time_ns=timestamp_ns,
            trajectory=trajectory,
            rows=rows,
            calibration=calibration,
            T_imu_lidar=T_imu_lidar,
            settings=multi_sweep,
            projection_mode=mode,
            shape=shape,
        )
        direct_depth = build_strict_direct_map(
            lidar=lidar,
            target_time_ns=timestamp_ns,
            calibration=calibration,
            shape=shape,
            mode=mode,
        )
        direct_valid = np.isfinite(direct_depth)
        tier_b_mask = np.isfinite(fastlio["depth_m"]) & ~direct_valid
        comparison_gt = current_gt if mode == "raw_distorted" else undistorted_gt
        self_metrics, overlay_arrays = self_consistency_metrics(
            fastlio["depth_m"],
            comparison_gt,
            tier_b_mask,
        )
        physical_metrics = physical_consistency_metrics(
            direct_depth,
            fastlio["depth_m"],
            radius_px=int(multi_sweep["support_radius_px"]),
        )
        report_modes[mode] = {
            "build": build_metrics,
            "tier_b_anchor_count_after_direct_precedence": int(
                np.count_nonzero(tier_b_mask)
            ),
            "self_consistency_vs_final_gt": self_metrics,
            "physical_consistency_vs_strict_raw_lidar_7x7": physical_metrics,
        }
        if mode == "undistorted":
            core.save_overlay(
                output_dir / "fastlio_undistorted_vs_runtime_remapped_gt.png",
                organized_rgb,
                overlay_arrays,
            )

    report = {
        "format": "luna.fastlio_tier_b_anchor_validation",
        "format_version": 1,
        "sequence": root.name,
        "frame_index": args.frame_index,
        "timestamp_ns": timestamp_ns,
        "subject_under_test": "FAST-LIO Tier-B multi-sweep anchors",
        "multi_sweep_parameters": multi_sweep,
        "projection_modes": report_modes,
        "interpretation": {
            "self_consistency": (
                "Detects projection/export/remap errors but is circular because "
                "Tier-B anchors overwrite final GT."
            ),
            "physical_consistency": (
                "Uses strict raw LiDAR only as an independent reference; the "
                "evaluated subject is the FAST-LIO anchor map."
            ),
        },
        "artifacts": {
            "runtime_remapped_depth_gt": str(
                output_dir / "depth_gt_runtime_remapped_undistorted.png"
            ),
            "undistorted_overlay": str(
                output_dir / "fastlio_undistorted_vs_runtime_remapped_gt.png"
            ),
        },
    }
    result_path = output_dir / "fastlio_anchor_validation.json"
    result_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
