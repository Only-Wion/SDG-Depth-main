#!/usr/bin/env python3
"""Sample dense Luna depth GT at projected raw-LiDAR pixel locations."""

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.luna_dataset import LunaOrganized
from scripts.infer_one_luna import model_and_dataset_args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project raw LiDAR to the rectified left image, sample depth GT at "
            "those pixels, and back-project the sampled depths into LiDAR XYZ."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-subdir", default="images_rectified")
    parser.add_argument("--left-dirname", default="left_out")
    parser.add_argument("--right-dirname", default="right_out")
    return parser.parse_args()


def write_binary_xyz_pcd(path, xyz):
    xyz = np.asarray(xyz, dtype="<f4")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 XYZ array, got {xyz.shape}")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(xyz)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(xyz)}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(xyz.tobytes(order="C"))


def back_project_to_lidar(
    u,
    v,
    depth,
    original_hw,
    target_hw,
    crop_fraction,
    calibration,
):
    scale_x = target_hw[1] / float(original_hw[1])
    scale_y = target_hw[0] / float(original_hw[0])
    crop_x = int(round(target_hw[1] * crop_fraction))
    crop_y = int(round(target_hw[0] * crop_fraction))

    intrinsics = calibration["K"].astype(np.float64).copy()
    intrinsics[0, :] *= scale_x
    intrinsics[1, :] *= scale_y
    intrinsics[0, 2] -= crop_x
    intrinsics[1, 2] -= crop_y

    x = (u - intrinsics[0, 2]) / intrinsics[0, 0] * depth
    y = (v - intrinsics[1, 2]) / intrinsics[1, 1] * depth
    xyz_rectified = np.column_stack((x, y, depth))

    rectification = calibration["rect_r"].astype(np.float64)
    xyz_camera = xyz_rectified @ rectification
    lidar_to_camera = calibration["lidar_to_cam"].astype(np.float64)
    rotation = lidar_to_camera[:3, :3]
    translation = lidar_to_camera[:3, 3]
    xyz_lidar = np.linalg.solve(
        rotation, (xyz_camera - translation).T
    ).T
    return xyz_rectified, xyz_lidar, intrinsics


def main():
    cli = parse_args()
    output_sequence = cli.output_root / cli.sequence
    pcd_dir = output_sequence / "frames"
    sample_dir = output_sequence / "samples"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    args = model_and_dataset_args()
    args.luna_exclude_sequences = []
    args.luna_val_fraction = 0.0
    args.luna_test_fraction = 0.0
    args.luna_image_subdir = cli.image_subdir
    args.luna_left_dirname = cli.left_dirname
    args.luna_right_dirname = cli.right_dirname
    args.luna_lidar_source = "raw"
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
            f"No raw-LiDAR-paired samples found for {cli.sequence}"
        )

    manifest_rows = []
    total_points = 0
    timestamp_path = output_sequence / "timestamps.txt"
    with timestamp_path.open("w", encoding="utf-8", newline="\n") as timestamps:
        timestamps.write(
            "index filename header_time_ns record_time_ns seq frame_id "
            "point_count\n"
        )
        for sequence_index, sample_index in enumerate(sample_indices):
            metadata = dataset.extra_info[sample_index]
            paths, _, _, flow_gt, valid_gt, hint, conversion_rate = dataset[
                sample_index
            ]
            frame = metadata["frame"]
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
                    f"No raw-LiDAR positions with valid GT for frame{frame}"
                )

            v, u = np.nonzero(valid)
            depth_gt = conversion / flow[valid]
            original_hw = np.asarray(
                Image.open(metadata["depth"])
            ).shape[:2]
            target_hw = tuple(args.luna_resize)
            xyz_rectified, xyz_lidar, effective_k = back_project_to_lidar(
                u.astype(np.float64),
                v.astype(np.float64),
                depth_gt,
                original_hw,
                target_hw,
                args.luna_border_crop_fraction,
                metadata["calib"],
            )

            pcd_path = pcd_dir / f"{frame}.pcd"
            npz_path = sample_dir / f"{frame}.npz"
            write_binary_xyz_pcd(pcd_path, xyz_lidar)
            np.savez_compressed(
                npz_path,
                u=u.astype(np.int32),
                v=v.astype(np.int32),
                depth_gt_m=depth_gt.astype(np.float32),
                xyz_rectified_camera=xyz_rectified.astype(np.float32),
                xyz_lidar=xyz_lidar.astype(np.float32),
                effective_rectified_K=effective_k.astype(np.float64),
                source_raw_lidar=np.asarray(str(metadata["lidar"])),
                source_depth_gt=np.asarray(str(metadata["depth"])),
                image_time_ns=np.asarray(
                    metadata["image_time_ns"], dtype=np.int64
                ),
            )

            point_count = len(xyz_lidar)
            timestamp = int(metadata["image_time_ns"])
            timestamps.write(
                f"{frame} frames/{frame}.pcd {timestamp} {timestamp} "
                f"{sequence_index} depth_gt_sampled {point_count}\n"
            )
            manifest_rows.append(
                {
                    "sequence": cli.sequence,
                    "frame": frame,
                    "image_time_ns": timestamp,
                    "point_count": point_count,
                    "source_raw_lidar": str(metadata["lidar"]),
                    "source_depth_gt": str(metadata["depth"]),
                    "pcd": str(pcd_path),
                    "samples_npz": str(npz_path),
                }
            )
            total_points += point_count
            print(
                f"[{sequence_index + 1}/{len(sample_indices)}] "
                f"frame{frame}: {point_count} GT-sampled LiDAR points"
            )

    manifest_path = output_sequence / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "sequence": cli.sequence,
        "source_lidar": "raw",
        "source_depth": "depth_gt_rectified",
        "sampling": (
            "Depth GT sampled at z-buffered raw-LiDAR projection pixels in "
            "the resized, center-cropped rectified left image."
        ),
        "coordinate_outputs": {
            "pcd": "LiDAR sensor coordinate system",
            "npz_xyz_lidar": "LiDAR sensor coordinate system",
            "npz_xyz_rectified_camera": (
                "rectified left-camera coordinate system"
            ),
            "npz_u_v": (
                "zero-based pixels in the resized and center-cropped model input"
            ),
        },
        "frames": len(manifest_rows),
        "total_points": total_points,
        "mean_points_per_frame": total_points / len(manifest_rows),
        "output_root": str(output_sequence),
        "timestamps": str(timestamp_path),
        "manifest": str(manifest_path),
    }
    summary_path = output_sequence / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
