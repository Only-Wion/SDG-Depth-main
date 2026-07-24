#!/usr/bin/env python3
"""Project one FAST-LIO map cloud into its timestamp-matched left RGB image."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read_tsv(path):
    with path.open("r", encoding="utf-8") as stream:
        return [
            line.strip().split("\t")
            for line in stream
            if line.strip() and not line.startswith("#")
        ]


def find_key(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            result = find_key(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find_key(child, key)
            if result is not None:
                return result
    return None


def find_calibration(root):
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if find_key(data, "LiDAR_to_Cam_L") is not None:
            combined = {path.stem: data}
            for sibling in path.parent.glob("*.json"):
                if sibling == path:
                    continue
                try:
                    combined[sibling.stem] = json.loads(
                        sibling.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue
            return path.parent, combined
    raise FileNotFoundError(f"No LiDAR-to-camera calibration found under {root}")


def matrix(node, keys):
    for key in keys:
        value = find_key(node, key)
        if value is None:
            continue
        if isinstance(value, dict):
            for subkey in ("RT", "R", "K", "data", "matrix"):
                if subkey in value:
                    return np.asarray(value[subkey], dtype=np.float64)
        return np.asarray(value, dtype=np.float64)
    raise KeyError(f"None of the matrix keys were found: {keys}")


def as_transform(value):
    value = np.asarray(value, dtype=np.float64)
    if value.size == 16:
        return value.reshape(4, 4)[:3]
    return value.reshape(3, 4)


def as_rotation(value):
    value = np.asarray(value, dtype=np.float64)
    if value.size == 16:
        return value.reshape(4, 4)[:3, :3]
    return value.reshape(3, 3)


def interpolate_pose(trajectory, timestamp_ns):
    times = trajectory["header_time_ns"].astype(np.int64)
    positions = trajectory["position"].astype(np.float64)
    quaternions = trajectory["orientation"].astype(np.float64)
    order = np.argsort(times)
    times = times[order]
    positions = positions[order]
    quaternions = quaternions[order]

    relative_times = (times - times[0]) * 1e-9
    query_time = float(
        np.clip((timestamp_ns - times[0]) * 1e-9, relative_times[0], relative_times[-1])
    )
    upper = int(np.searchsorted(relative_times, query_time))
    if upper == 0:
        position = positions[0]
        rotation = Rotation.from_quat(quaternions[0]).as_matrix()
    elif upper == len(relative_times):
        position = positions[-1]
        rotation = Rotation.from_quat(quaternions[-1]).as_matrix()
    else:
        lower = upper - 1
        alpha = (query_time - relative_times[lower]) / (
            relative_times[upper] - relative_times[lower]
        )
        position = (1.0 - alpha) * positions[lower] + alpha * positions[upper]
        slerp = Slerp(
            relative_times[[lower, upper]],
            Rotation.from_quat(quaternions[[lower, upper]]),
        )
        rotation = slerp([query_time]).as_matrix()[0]
    clamped = timestamp_ns < times[0] or timestamp_ns > times[-1]
    return position, rotation, bool(clamped)


def zbuffer(points, intrinsics, height, width):
    depth = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (depth > 0.2) & (depth < 120.0)
    points = points[valid]
    depth = depth[valid]
    u = np.rint(
        intrinsics[0, 0] * points[:, 0] / depth + intrinsics[0, 2]
    ).astype(np.int32)
    v = np.rint(
        intrinsics[1, 1] * points[:, 1] / depth + intrinsics[1, 2]
    ).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, depth = u[valid], v[valid], depth[valid]

    linear = v.astype(np.int64) * width + u
    order = np.lexsort((depth, linear))
    sorted_linear = linear[order]
    first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
    selected = order[first]
    return u[selected], v[selected], depth[selected]


def labelled(image, title):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(
        result,
        title,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def overlay(image, u, v, depth, title):
    result = image.copy()
    normalized = ((np.clip(depth, 1.0, 40.0) - 1.0) / 39.0 * 255).astype(
        np.uint8
    )
    colors = cv2.applyColorMap(normalized[:, None], cv2.COLORMAP_TURBO)[:, 0]
    for x, y, color in zip(u[::-1], v[::-1], colors[::-1]):
        cv2.circle(
            result,
            (int(x), int(y)),
            2,
            tuple(map(int, color)),
            -1,
            cv2.LINE_AA,
        )
    return labelled(cv2.addWeighted(image, 0.4, result, 0.6, 0), title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("frame")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path)
    args = parser.parse_args()

    image_rows = read_tsv(args.sequence / "images" / "timestamps.tsv")
    image_row = next(
        row
        for row in image_rows[1:]
        if row[0] == "left" and row[1] == args.frame
    )
    image_time_ns = int(image_row[2])
    image = cv2.imread(str(args.sequence / image_row[3]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.sequence / image_row[3])
    height, width = image.shape[:2]

    calibration_root = args.calibration_root or args.sequence.parent.parent
    calibration_path, calibration = find_calibration(calibration_root)
    lidar_to_camera = as_transform(
        matrix(find_key(calibration, "LiDAR_to_Cam_L"), ("RT", "matrix", "data"))
    )
    camera_k = matrix(find_key(calibration, "Cam_L"), ("K",)).reshape(3, 3)
    rectified_k = matrix(find_key(calibration, "Cam_Rect_L"), ("K",)).reshape(3, 3)
    rectification = as_rotation(
        matrix(find_key(calibration, "Cam_L_to_Cam_L_Rect"), ("R", "matrix", "data"))
    )

    fastlio_rows = read_tsv(args.sequence / "lidar_fastlio" / "timestamps.tsv")
    _, fastlio_time_ns, fastlio_name = min(
        (
            abs(int(row[2]) - image_time_ns),
            int(row[2]),
            row[1],
        )
        for row in fastlio_rows[1:]
    )
    fastlio_frame = Path(fastlio_name).stem
    cloud = np.load(
        args.sequence / "lidar_fastlio" / "frames" / f"{fastlio_frame}.npz"
    )
    map_points = cloud["xyz"].astype(np.float64)

    trajectory = np.load(args.sequence / "lidar_fastlio" / "trajectory.npz")
    position, map_from_body, clamped = interpolate_pose(
        trajectory, image_time_ns
    )
    body_points = (map_points - position) @ map_from_body
    camera_points = (
        body_points @ lidar_to_camera[:, :3].T + lidar_to_camera[:, 3]
    )
    rectified_points = camera_points @ rectification.T

    u, v, depth = zbuffer(camera_points, camera_k, height, width)
    ur, vr, rectified_depth = zbuffer(
        rectified_points, rectified_k, height, width
    )
    comparison = (image * 0.22).astype(np.uint8)
    comparison[v, u] = (0, 255, 0)
    comparison[vr, ur] = (255, 0, 255)

    panels = [
        labelled(image, "Left RGB"),
        overlay(image, u, v, depth, "FAST-LIO + image-time pose + Cam_L K"),
        overlay(
            image,
            ur,
            vr,
            rectified_depth,
            "FAST-LIO + image-time pose + rectified K/R",
        ),
        labelled(comparison, "Green: Cam_L K   Magenta: rectified K/R"),
    ]
    panels = [
        cv2.resize(panel, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        for panel in panels
    ]
    contact_sheet = np.vstack(
        (np.hstack(panels[:2]), np.hstack(panels[2:]))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), contact_sheet):
        raise RuntimeError(f"Failed to write {args.output}")

    metrics = {
        "sequence": args.sequence.name,
        "frame": args.frame,
        "image_time_ns": image_time_ns,
        "fastlio_frame": fastlio_frame,
        "fastlio_time_ns": fastlio_time_ns,
        "time_delta_ms": (fastlio_time_ns - image_time_ns) / 1e6,
        "map_points": len(map_points),
        "projected_cam_l_pixels": len(depth),
        "projected_rectified_pixels": len(rectified_depth),
        "calibration_path": str(calibration_path),
        "image_shape": [height, width],
        "trajectory_clamped": clamped,
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
