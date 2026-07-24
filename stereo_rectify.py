#!/usr/bin/env python3
"""
Calibrated stereo rectification for the supplied JSON schema.

The default extrinsic key is Cam_R_to_Cam_L. OpenCV stereoRectify expects
the transform from the first camera (left) to the second camera (right), so
the script inverts R_right_to_left and T_right_to_left before rectification.

An optional empirical vertical refinement is enabled by default. It estimates
only a smooth y correction for the right rectified image; x coordinates and
therefore horizontal disparity are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Undistort and epipolar-rectify a calibrated stereo image pair."
        )
    )
    parser.add_argument("--left", type=Path, required=True, help="Left image.")
    parser.add_argument("--right", type=Path, required=True, help="Right image.")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=script_dir / "intrinsics.json",
    )
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=script_dir / "extrinsics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "rectified_output",
    )
    parser.add_argument("--left-key", default="Cam_L")
    parser.add_argument("--right-key", default="Cam_R")
    parser.add_argument("--extrinsic-key", default="Cam_R_to_Cam_L")
    parser.add_argument(
        "--extrinsic-direction",
        choices=("right-to-left", "left-to-right"),
        default="right-to-left",
        help=(
            "Direction encoded by the selected extrinsic key. The supplied "
            "Cam_R_to_Cam_L entry is right-to-left."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help=(
            "OpenCV stereoRectify scaling: 0 removes black borders, "
            "1 keeps all source pixels."
        ),
    )
    parser.add_argument(
        "--no-refine-vertical",
        action="store_true",
        help="Disable feature-based residual y-only refinement.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = (
        [cv2.IMWRITE_PNG_COMPRESSION, 3]
        if path.suffix.lower() == ".png"
        else []
    )
    ok, encoded = cv2.imencode(path.suffix, image, parameters)
    if not ok:
        raise RuntimeError(f"Cannot encode output: {path}")
    temporary = path.with_name(f".{path.stem}.rectify_tmp{path.suffix}")
    if temporary.exists():
        raise FileExistsError(f"Temporary file already exists: {temporary}")
    try:
        encoded.tofile(str(temporary))
        if temporary.stat().st_size <= 0:
            raise RuntimeError(f"Temporary output is empty: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def scaled_camera_matrix(
    entry: dict,
    actual_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(entry["K"], dtype=np.float64).copy()
    distortion = np.asarray(entry.get("distortion", []), dtype=np.float64)
    calibration_height, calibration_width = entry["resolution"]
    actual_width, actual_height = actual_size
    scale_x = actual_width / float(calibration_width)
    scale_y = actual_height / float(calibration_height)
    calibration_aspect = calibration_width / float(calibration_height)
    actual_aspect = actual_width / float(actual_height)
    if abs(calibration_aspect - actual_aspect) > 1e-4:
        raise ValueError(
            "Image aspect ratio differs from calibration resolution: "
            f"calibration={calibration_width}x{calibration_height}, "
            f"image={actual_width}x{actual_height}"
        )
    matrix[0, 0] *= scale_x
    matrix[0, 2] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[1, 2] *= scale_y
    return matrix, distortion


def left_to_right_extrinsics(
    entry: dict,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(entry["R"], dtype=np.float64)
    translation = np.asarray(entry["T"], dtype=np.float64).reshape(3, 1)
    if rotation.shape != (3, 3) or translation.shape != (3, 1):
        raise ValueError("Extrinsic R must be 3x3 and T must contain 3 values")
    if direction == "right-to-left":
        rotation_left_to_right = rotation.T
        translation_left_to_right = -rotation.T @ translation
    else:
        rotation_left_to_right = rotation
        translation_left_to_right = translation
    return rotation_left_to_right, translation_left_to_right


def match_points(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.006)
        norm = cv2.NORM_L2
        ratio_threshold = 0.74
    else:
        detector = cv2.ORB_create(nfeatures=15000, fastThreshold=7)
        norm = cv2.NORM_HAMMING
        ratio_threshold = 0.78
    left_keypoints, left_descriptors = detector.detectAndCompute(left_gray, None)
    right_keypoints, right_descriptors = detector.detectAndCompute(right_gray, None)
    if left_descriptors is None or right_descriptors is None:
        raise RuntimeError("Could not extract enough image descriptors")
    raw_matches = cv2.BFMatcher(norm).knnMatch(
        right_descriptors,
        left_descriptors,
        k=2,
    )
    ratio_matches = []
    for candidates in raw_matches:
        if len(candidates) < 2:
            continue
        first, second = candidates
        if first.distance < ratio_threshold * second.distance:
            ratio_matches.append(first)
    if len(ratio_matches) < 12:
        raise RuntimeError(f"Too few descriptor matches: {len(ratio_matches)}")
    right_points = np.float32(
        [right_keypoints[match.queryIdx].pt for match in ratio_matches]
    )
    left_points = np.float32(
        [left_keypoints[match.trainIdx].pt for match in ratio_matches]
    )
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    _, mask = cv2.findFundamentalMat(
        right_points,
        left_points,
        method=method,
        ransacReprojThreshold=1.5,
        confidence=0.999,
        maxIters=10000,
    )
    if mask is None:
        raise RuntimeError("Fundamental matrix validation failed")
    keep = mask.ravel().astype(bool)
    if int(keep.sum()) < 12:
        raise RuntimeError(f"Too few geometric inliers: {int(keep.sum())}")
    return (
        left_points[keep],
        right_points[keep],
        len(ratio_matches),
        int(keep.sum()),
    )


def transform_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rectification_rotation: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    return cv2.undistortPoints(
        points[:, None, :],
        camera_matrix,
        distortion,
        R=rectification_rotation,
        P=projection,
    )[:, 0, :]


def vertical_statistics(
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> dict[str, float]:
    residual = left_points[:, 1] - right_points[:, 1]
    absolute = np.abs(residual)
    disparity = left_points[:, 0] - right_points[:, 0]
    return {
        "median_signed_y_px": float(np.median(residual)),
        "median_abs_y_px": float(np.median(absolute)),
        "p90_abs_y_px": float(np.percentile(absolute, 90)),
        "mean_abs_y_px": float(np.mean(absolute)),
        "median_disparity_px": float(np.median(disparity)),
        "p10_disparity_px": float(np.percentile(disparity, 10)),
        "p90_disparity_px": float(np.percentile(disparity, 90)),
    }


def robust_vertical_model(
    right_rectified_points: np.ndarray,
    target_delta_y: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    x = right_rectified_points[:, 0] / max(1, width - 1) * 2.0 - 1.0
    y = right_rectified_points[:, 1] / max(1, height - 1) * 2.0 - 1.0
    basis = np.stack([np.ones_like(x), x, y], axis=1).astype(np.float64)
    target = target_delta_y.astype(np.float64)
    weights = np.ones(len(target))
    coefficients = np.zeros(3)
    for _ in range(10):
        weighted = np.sqrt(weights)[:, None]
        coefficients, *_ = np.linalg.lstsq(
            basis * weighted,
            target * weighted[:, 0],
            rcond=None,
        )
        errors = basis @ coefficients - target
        median = np.median(errors)
        sigma = 1.4826 * np.median(np.abs(errors - median)) + 1e-8
        threshold = max(2.5 * sigma, 0.25)
        weights = np.minimum(1.0, threshold / (np.abs(errors) + 1e-8))
    return coefficients


def refine_right_vertical(
    right_rectified: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    height, width = right_rectified.shape[:2]
    coefficient_0, coefficient_x, coefficient_y = coefficients
    x_pixels = np.arange(width, dtype=np.float32)[None, :]
    y_pixels = np.arange(height, dtype=np.float32)[:, None]
    x_normalized = x_pixels / max(1, width - 1) * 2.0 - 1.0
    denominator = 1.0 + 2.0 * coefficient_y / max(1, height - 1)
    source_y = (
        y_pixels
        - coefficient_0
        - coefficient_x * x_normalized
        + coefficient_y
    ) / denominator
    map_x = np.broadcast_to(x_pixels, (height, width)).astype(np.float32)
    map_y = np.broadcast_to(source_y, (height, width)).astype(np.float32)
    return cv2.remap(
        right_rectified,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def refined_point_coordinates(
    right_points: np.ndarray,
    coefficients: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    x = right_points[:, 0] / max(1, width - 1) * 2.0 - 1.0
    y = right_points[:, 1] / max(1, height - 1) * 2.0 - 1.0
    delta = coefficients[0] + coefficients[1] * x + coefficients[2] * y
    refined = right_points.copy()
    refined[:, 1] += delta
    return refined


def canonical_positive_disparity_geometry(
    projection_left: np.ndarray,
    baseline: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    focal = float(projection_left[0, 0])
    center_x = float(projection_left[0, 2])
    center_y = float(projection_left[1, 2])
    projection_left_canonical = projection_left.copy()
    projection_left_canonical[:, 3] = 0.0
    projection_right_canonical = projection_left_canonical.copy()
    projection_right_canonical[0, 3] = -focal * baseline
    q_positive = np.array(
        [
            [1.0, 0.0, 0.0, -center_x],
            [0.0, 1.0, 0.0, -center_y],
            [0.0, 0.0, 0.0, focal],
            [0.0, 0.0, 1.0 / baseline, 0.0],
        ],
        dtype=np.float64,
    )
    return projection_left_canonical, projection_right_canonical, q_positive


def build_preview(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    preview = np.concatenate([left, right], axis=1)
    height, width = preview.shape[:2]
    for y in range(40, height, 80):
        cv2.line(preview, (0, y), (width - 1, y), (0, 220, 0), 1)
    cv2.putText(
        preview,
        "LEFT RECTIFIED",
        (24, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "RIGHT RECTIFIED",
        (left.shape[1] + 24, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def serializable_matrix(matrix: np.ndarray) -> list:
    return np.asarray(matrix).tolist()


def main() -> int:
    args = parse_args()
    left_path = args.left.expanduser().resolve()
    right_path = args.right.expanduser().resolve()
    intrinsics_path = args.intrinsics.expanduser().resolve()
    extrinsics_path = args.extrinsics.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    left_image = read_image(left_path)
    right_image = read_image(right_path)
    if left_image.shape[:2] != right_image.shape[:2]:
        raise ValueError(
            f"Left/right dimensions differ: "
            f"{left_image.shape[:2]} vs {right_image.shape[:2]}"
        )
    height, width = left_image.shape[:2]
    image_size = (width, height)

    intrinsics = load_json(intrinsics_path)
    extrinsics = load_json(extrinsics_path)
    camera_left, distortion_left = scaled_camera_matrix(
        intrinsics[args.left_key],
        image_size,
    )
    camera_right, distortion_right = scaled_camera_matrix(
        intrinsics[args.right_key],
        image_size,
    )
    rotation_left_to_right, translation_left_to_right = left_to_right_extrinsics(
        extrinsics[args.extrinsic_key],
        args.extrinsic_direction,
    )
    baseline = float(np.linalg.norm(translation_left_to_right))

    (
        rectification_left,
        rectification_right,
        projection_left,
        projection_right,
        q_opencv,
        roi_left,
        roi_right,
    ) = cv2.stereoRectify(
        camera_left,
        distortion_left,
        camera_right,
        distortion_right,
        image_size,
        rotation_left_to_right,
        translation_left_to_right,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(args.alpha),
        newImageSize=image_size,
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        camera_left,
        distortion_left,
        rectification_left,
        projection_left,
        image_size,
        cv2.CV_32FC1,
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        camera_right,
        distortion_right,
        rectification_right,
        projection_right,
        image_size,
        cv2.CV_32FC1,
    )
    left_rectified = cv2.remap(
        left_image,
        map_left_x,
        map_left_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rectified = cv2.remap(
        right_image,
        map_right_x,
        map_right_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    left_points, right_points, ratio_count, inlier_count = match_points(
        left_image,
        right_image,
    )
    left_points_rectified = transform_points(
        left_points,
        camera_left,
        distortion_left,
        rectification_left,
        projection_left,
    )
    right_points_rectified = transform_points(
        right_points,
        camera_right,
        distortion_right,
        rectification_right,
        projection_right,
    )
    statistics_before_refinement = vertical_statistics(
        left_points_rectified,
        right_points_rectified,
    )

    vertical_coefficients = np.zeros(3, dtype=np.float64)
    if not args.no_refine_vertical:
        vertical_coefficients = robust_vertical_model(
            right_points_rectified,
            left_points_rectified[:, 1] - right_points_rectified[:, 1],
            image_size,
        )
        right_rectified = refine_right_vertical(
            right_rectified,
            vertical_coefficients,
        )
        right_points_rectified = refined_point_coordinates(
            right_points_rectified,
            vertical_coefficients,
            image_size,
        )
    statistics_after_refinement = vertical_statistics(
        left_points_rectified,
        right_points_rectified,
    )

    (
        projection_left_positive,
        projection_right_positive,
        q_positive_disparity,
    ) = canonical_positive_disparity_geometry(projection_left, baseline)

    output_dir.mkdir(parents=True, exist_ok=True)
    left_output = output_dir / "left_rectified.png"
    right_output = output_dir / "right_rectified.png"
    preview_output = output_dir / "epipolar_preview.png"
    parameters_output = output_dir / "rectification_parameters.json"
    write_image(left_output, left_rectified, args.overwrite)
    write_image(right_output, right_rectified, args.overwrite)
    write_image(
        preview_output,
        build_preview(left_rectified, right_rectified),
        args.overwrite,
    )

    parameters = {
        "inputs": {
            "left": str(left_path),
            "right": str(right_path),
            "intrinsics": str(intrinsics_path),
            "extrinsics": str(extrinsics_path),
        },
        "keys": {
            "left": args.left_key,
            "right": args.right_key,
            "extrinsic": args.extrinsic_key,
            "extrinsic_direction": args.extrinsic_direction,
        },
        "image_size_width_height": [width, height],
        "alpha": float(args.alpha),
        "baseline_in_extrinsic_units": baseline,
        "camera_matrix_left": serializable_matrix(camera_left),
        "camera_matrix_right": serializable_matrix(camera_right),
        "distortion_left": serializable_matrix(distortion_left),
        "distortion_right": serializable_matrix(distortion_right),
        "R_left_to_right": serializable_matrix(rotation_left_to_right),
        "T_left_to_right": serializable_matrix(translation_left_to_right),
        "R1": serializable_matrix(rectification_left),
        "R2": serializable_matrix(rectification_right),
        "P1_opencv": serializable_matrix(projection_left),
        "P2_opencv": serializable_matrix(projection_right),
        "Q_opencv": serializable_matrix(q_opencv),
        "P1_positive_disparity": serializable_matrix(projection_left_positive),
        "P2_positive_disparity": serializable_matrix(projection_right_positive),
        "Q_positive_disparity": serializable_matrix(q_positive_disparity),
        "valid_roi_left": list(map(int, roi_left)),
        "valid_roi_right": list(map(int, roi_right)),
        "feature_validation": {
            "ratio_matches": ratio_count,
            "geometric_inliers": inlier_count,
            "before_vertical_refinement": statistics_before_refinement,
            "after_vertical_refinement": statistics_after_refinement,
        },
        "vertical_refinement": {
            "enabled": not args.no_refine_vertical,
            "model": "delta_y = c0 + cx*x_normalized + cy*y_normalized",
            "coefficients_c0_cx_cy": vertical_coefficients.tolist(),
            "note": (
                "This refinement changes only right-image y coordinates. "
                "Horizontal disparity x_left-x_right is unchanged."
            ),
        },
    }
    if parameters_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists; use --overwrite: {parameters_output}"
        )
    temporary_json = parameters_output.with_suffix(".json.tmp")
    try:
        with temporary_json.open("w", encoding="utf-8") as handle:
            json.dump(parameters, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_json, parameters_output)
    finally:
        if temporary_json.exists():
            temporary_json.unlink()

    print(f"image_size={width}x{height}")
    print(f"baseline={baseline:.6f} (extrinsic units)")
    print(f"ratio_matches={ratio_count}")
    print(f"geometric_inliers={inlier_count}")
    print(
        "vertical_before: "
        f"median_abs={statistics_before_refinement['median_abs_y_px']:.4f}px, "
        f"p90_abs={statistics_before_refinement['p90_abs_y_px']:.4f}px"
    )
    print(
        "vertical_after: "
        f"median_abs={statistics_after_refinement['median_abs_y_px']:.4f}px, "
        f"p90_abs={statistics_after_refinement['p90_abs_y_px']:.4f}px"
    )
    print(
        "median_disparity="
        f"{statistics_after_refinement['median_disparity_px']:.4f}px"
    )
    print(f"left_rectified={left_output}")
    print(f"right_rectified={right_output}")
    print(f"preview={preview_output}")
    print(f"parameters={parameters_output}")
    if statistics_after_refinement["p90_abs_y_px"] > 1.5:
        print(
            "warning: p90 vertical residual exceeds 1.5 px; recalibration "
            "may be needed for high-accuracy depth.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
