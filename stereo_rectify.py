#!/usr/bin/env python3
"""
Calibrated stereo rectification for the supplied JSON schema.

The default extrinsic key is Cam_R_to_Cam_L. OpenCV stereoRectify expects
the transform from the first camera (left) to the second camera (right), so
the script inverts R_right_to_left and T_right_to_left before rectification.

Batch mode discovers organized sequences below --input-root, reuses one
rectification map per sequence, writes images_rectified/ beside images/, and
remaps aligned uint16 depth_gt/ into depth_gt_rectified/ with nearest neighbors.
If a sequence manifest says its RGB images are already undistorted, auto mode
uses zero distortion coefficients to avoid applying lens correction twice. Auto
rectification prefers the explicit Cam_*_Rect matrices published with a sequence.

An optional empirical vertical refinement is enabled by default. Batch mode
estimates one fixed smooth y correction per sequence; x coordinates and
therefore horizontal disparity are left unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
            "Rectify one calibrated stereo pair or a root of organized sequences."
        )
    )
    parser.add_argument("--left", type=Path, help="Left image in single-pair mode.")
    parser.add_argument("--right", type=Path, help="Right image in single-pair mode.")
    parser.add_argument("--input-root", type=Path, help="Root of organized sequences.")
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
    parser.add_argument(
        "--output-subdir",
        default="images_rectified",
        help="Per-sequence output directory name in batch mode.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        help="Process only this sequence name; may be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many pairs per sequence (for validation).",
    )
    parser.add_argument(
        "--refine-samples",
        type=int,
        default=5,
        help="Frames sampled per sequence to estimate one fixed y refinement.",
    )
    parser.add_argument(
        "--distortion-mode",
        choices=("auto", "calibrated", "already-undistorted"),
        default="auto",
        help="Auto reads each sequence manifest; organized Luna images are undistorted.",
    )
    parser.add_argument(
        "--rectification-mode",
        choices=("auto", "opencv", "calibrated"),
        default="auto",
        help="Auto prefers explicit Cam_*_Rect calibration when available.",
    )
    parser.add_argument("--left-rectified-key", default="Cam_Rect_L")
    parser.add_argument("--right-rectified-key", default="Cam_Rect_R")
    parser.add_argument(
        "--left-rectification-key", default="Cam_L_to_Cam_L_Rect"
    )
    parser.add_argument(
        "--right-rectification-key", default="Cam_R_to_Cam_R_Rect"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pairs whose two rectified outputs already exist.",
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
        "--no-rectify-depth",
        action="store_true",
        help="Do not remap depth_gt with the left rectification map.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()
    single_requested = args.left is not None or args.right is not None
    if args.input_root is not None and single_requested:
        parser.error("--input-root cannot be combined with --left or --right")
    if args.input_root is None and (args.left is None or args.right is None):
        parser.error("provide --input-root, or both --left and --right")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.refine_samples <= 0:
        parser.error("--refine-samples must be positive")
    output_subdir = Path(args.output_subdir)
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        parser.error("--output-subdir must be relative and cannot contain '..'")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def read_image_unchanged(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
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
    zero_distortion: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(entry["K"], dtype=np.float64).copy()
    distortion = np.asarray(entry.get("distortion", []), dtype=np.float64)
    if zero_distortion:
        distortion = np.zeros_like(distortion)
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


@dataclass
class RectificationContext:
    image_size: tuple[int, int]
    camera_left: np.ndarray
    camera_right: np.ndarray
    distortion_left: np.ndarray
    distortion_right: np.ndarray
    rotation_left_to_right: np.ndarray
    translation_left_to_right: np.ndarray
    rectification_left: np.ndarray
    rectification_right: np.ndarray
    projection_left: np.ndarray
    projection_right: np.ndarray
    q_opencv: np.ndarray
    roi_left: tuple[int, int, int, int]
    roi_right: tuple[int, int, int, int]
    map_left_x: np.ndarray
    map_left_y: np.ndarray
    map_right_x: np.ndarray
    map_right_y: np.ndarray
    baseline: float
    zero_distortion: bool
    rectification_mode: str
    vertical_coefficients: np.ndarray
    vertical_refinement_applied: bool
    feature_validation: dict


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def discover_sequences(root: Path, selected: list[str] | None) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Batch input root does not exist: {root}")
    selected_set = set(selected or [])
    sequences = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        if selected_set and candidate.name not in selected_set:
            continue
        required = (
            candidate / "images" / "left",
            candidate / "images" / "right",
            candidate / "calibration" / "intrinsics.json",
            candidate / "calibration" / "extrinsics.json",
        )
        if all(path.exists() for path in required):
            sequences.append(candidate)
    if selected_set:
        found = {path.name for path in sequences}
        missing = sorted(selected_set - found)
        if missing:
            raise FileNotFoundError(
                "Selected sequences are missing or incomplete: " + ", ".join(missing)
            )
    if not sequences:
        raise RuntimeError(f"No organized stereo sequences found under: {root}")
    return sequences


def paired_image_paths(sequence_dir: Path) -> list[tuple[Path, Path]]:
    left_dir = sequence_dir / "images" / "left"
    right_dir = sequence_dir / "images" / "right"
    left = {path.name: path for path in left_dir.glob("*.png")}
    right = {path.name: path for path in right_dir.glob("*.png")}
    missing_right = sorted(left.keys() - right.keys())
    missing_left = sorted(right.keys() - left.keys())
    if missing_right or missing_left:
        details = []
        if missing_right:
            details.append(f"missing right: {missing_right[:5]}")
        if missing_left:
            details.append(f"missing left: {missing_left[:5]}")
        raise RuntimeError(
            f"Unpaired images in {sequence_dir.name}: " + "; ".join(details)
        )
    if not left:
        raise RuntimeError(f"No PNG stereo pairs in: {sequence_dir / 'images'}")
    return [(left[name], right[name]) for name in sorted(left)]


def manifest_reports_undistorted(sequence_dir: Path) -> bool:
    manifest_path = sequence_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = load_json(manifest_path)
    convention = str(
        manifest.get("coordinate_conventions", {}).get("images", "")
    ).lower()
    return "undistort" in convention


def create_rectification_context(
    sequence_dir: Path,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> RectificationContext:
    intrinsics = load_json(sequence_dir / "calibration" / "intrinsics.json")
    extrinsics = load_json(sequence_dir / "calibration" / "extrinsics.json")
    if args.distortion_mode == "already-undistorted":
        zero_distortion = True
    elif args.distortion_mode == "calibrated":
        zero_distortion = False
    else:
        zero_distortion = manifest_reports_undistorted(sequence_dir)
    camera_left, distortion_left = scaled_camera_matrix(
        intrinsics[args.left_key], image_size, zero_distortion
    )
    camera_right, distortion_right = scaled_camera_matrix(
        intrinsics[args.right_key], image_size, zero_distortion
    )
    rotation_left_to_right, translation_left_to_right = left_to_right_extrinsics(
        extrinsics[args.extrinsic_key], args.extrinsic_direction
    )
    baseline = float(np.linalg.norm(translation_left_to_right))
    explicit_keys = (
        args.left_rectified_key,
        args.right_rectified_key,
        args.left_rectification_key,
        args.right_rectification_key,
    )
    explicit_available = (
        all(key in intrinsics for key in explicit_keys[:2])
        and all(key in extrinsics for key in explicit_keys[2:])
    )
    use_explicit = args.rectification_mode == "calibrated" or (
        args.rectification_mode == "auto" and explicit_available
    )
    if use_explicit and not explicit_available:
        missing = [
            key
            for key in explicit_keys[:2]
            if key not in intrinsics
        ] + [
            key
            for key in explicit_keys[2:]
            if key not in extrinsics
        ]
        raise KeyError("Missing explicit rectification keys: " + ", ".join(missing))
    if use_explicit:
        rectified_left, _ = scaled_camera_matrix(
            intrinsics[args.left_rectified_key], image_size, True
        )
        rectified_right, _ = scaled_camera_matrix(
            intrinsics[args.right_rectified_key], image_size, True
        )
        if not np.allclose(rectified_left, rectified_right, atol=1e-5):
            raise ValueError(
                "Explicit left/right rectified camera matrices must match"
            )
        rectification_left = np.asarray(
            extrinsics[args.left_rectification_key]["R"], dtype=np.float64
        )
        rectification_right = np.asarray(
            extrinsics[args.right_rectification_key]["R"], dtype=np.float64
        )
        projection_left = np.column_stack(
            [rectified_left, np.zeros(3, dtype=np.float64)]
        )
        projection_right = projection_left.copy()
        projection_right[0, 3] = -projection_left[0, 0] * baseline
        _, _, q_opencv = canonical_positive_disparity_geometry(
            projection_left, baseline
        )
        width, height = image_size
        roi_left = (0, 0, width, height)
        roi_right = (0, 0, width, height)
        rectification_mode = "calibrated"
    else:
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
        rectification_mode = "opencv"
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
    return RectificationContext(
        image_size=image_size,
        camera_left=camera_left,
        camera_right=camera_right,
        distortion_left=distortion_left,
        distortion_right=distortion_right,
        rotation_left_to_right=rotation_left_to_right,
        translation_left_to_right=translation_left_to_right,
        rectification_left=rectification_left,
        rectification_right=rectification_right,
        projection_left=projection_left,
        projection_right=projection_right,
        q_opencv=q_opencv,
        roi_left=tuple(map(int, roi_left)),
        roi_right=tuple(map(int, roi_right)),
        map_left_x=map_left_x,
        map_left_y=map_left_y,
        map_right_x=map_right_x,
        map_right_y=map_right_y,
        baseline=baseline,
        zero_distortion=zero_distortion,
        rectification_mode=rectification_mode,
        vertical_coefficients=np.zeros(3, dtype=np.float64),
        vertical_refinement_applied=False,
        feature_validation={},
    )

def estimate_sequence_refinement(
    pairs: list[tuple[Path, Path]],
    context: RectificationContext,
    args: argparse.Namespace,
) -> None:
    sample_count = min(args.refine_samples, len(pairs))
    sample_indices = sorted(
        set(np.linspace(0, len(pairs) - 1, sample_count, dtype=int).tolist())
    )
    left_rectified_batches = []
    right_rectified_batches = []
    sample_reports = []
    for index in sample_indices:
        left_path, right_path = pairs[index]
        try:
            left_image = read_image(left_path)
            right_image = read_image(right_path)
            left_points, right_points, ratio_count, inlier_count = match_points(
                left_image, right_image
            )
            left_rectified = transform_points(
                left_points,
                context.camera_left,
                context.distortion_left,
                context.rectification_left,
                context.projection_left,
            )
            right_rectified = transform_points(
                right_points,
                context.camera_right,
                context.distortion_right,
                context.rectification_right,
                context.projection_right,
            )
            left_rectified_batches.append(left_rectified)
            right_rectified_batches.append(right_rectified)
            sample_reports.append(
                {
                    "frame": left_path.name,
                    "ratio_matches": ratio_count,
                    "geometric_inliers": inlier_count,
                }
            )
        except Exception as error:
            sample_reports.append({"frame": left_path.name, "error": str(error)})
    if not left_rectified_batches:
        raise RuntimeError(
            "Feature validation failed for every sampled pair; cannot validate "
            "the sequence calibration"
        )
    left_all = np.concatenate(left_rectified_batches, axis=0)
    right_all = np.concatenate(right_rectified_batches, axis=0)
    before = vertical_statistics(left_all, right_all)
    after = before
    if not args.no_refine_vertical:
        candidate_coefficients = robust_vertical_model(
            right_all,
            left_all[:, 1] - right_all[:, 1],
            context.image_size,
        )
        candidate_right = refined_point_coordinates(
            right_all, candidate_coefficients, context.image_size
        )
        candidate_statistics = vertical_statistics(left_all, candidate_right)
        improvement = (
            before["p90_abs_y_px"] - candidate_statistics["p90_abs_y_px"]
        )
        if improvement >= 0.01:
            context.vertical_coefficients = candidate_coefficients
            context.vertical_refinement_applied = True
            after = candidate_statistics
    context.feature_validation = {
        "sampled_frames": sample_reports,
        "successful_samples": len(left_rectified_batches),
        "total_geometric_inliers": int(len(left_all)),
        "before_vertical_refinement": before,
        "after_vertical_refinement": after,
    }


def rectify_with_context(
    left_image: np.ndarray,
    right_image: np.ndarray,
    context: RectificationContext,
    refine_vertical: bool,
) -> tuple[np.ndarray, np.ndarray]:
    expected_width, expected_height = context.image_size
    expected_shape = (expected_height, expected_width)
    if left_image.shape[:2] != expected_shape or right_image.shape[:2] != expected_shape:
        raise ValueError(
            "Pair dimensions differ from sequence calibration context: "
            f"expected={expected_shape}, left={left_image.shape[:2]}, "
            f"right={right_image.shape[:2]}"
        )
    left_rectified = cv2.remap(
        left_image,
        context.map_left_x,
        context.map_left_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rectified = cv2.remap(
        right_image,
        context.map_right_x,
        context.map_right_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if refine_vertical:
        right_rectified = refine_right_vertical(
            right_rectified, context.vertical_coefficients
        )
    return left_rectified, right_rectified


def batch_parameters(
    sequence_dir: Path,
    context: RectificationContext,
    args: argparse.Namespace,
) -> dict:
    projection_left_positive, projection_right_positive, q_positive = (
        canonical_positive_disparity_geometry(
            context.projection_left, context.baseline
        )
    )
    return {
        "sequence": sequence_dir.name,
        "source_sequence": str(sequence_dir),
        "image_size_width_height": list(context.image_size),
        "alpha": float(args.alpha),
        "distortion_mode_requested": args.distortion_mode,
        "input_treated_as_already_undistorted": context.zero_distortion,
        "rectification_mode_requested": args.rectification_mode,
        "rectification_mode_used": context.rectification_mode,
        "keys": {
            "left": args.left_key,
            "right": args.right_key,
            "extrinsic": args.extrinsic_key,
            "extrinsic_direction": args.extrinsic_direction,
            "left_rectified": args.left_rectified_key,
            "right_rectified": args.right_rectified_key,
            "left_rectification": args.left_rectification_key,
            "right_rectification": args.right_rectification_key,
        },
        "baseline_in_extrinsic_units": context.baseline,
        "camera_matrix_left": serializable_matrix(context.camera_left),
        "camera_matrix_right": serializable_matrix(context.camera_right),
        "distortion_left_used": serializable_matrix(context.distortion_left),
        "distortion_right_used": serializable_matrix(context.distortion_right),
        "R_left_to_right": serializable_matrix(context.rotation_left_to_right),
        "T_left_to_right": serializable_matrix(context.translation_left_to_right),
        "R1": serializable_matrix(context.rectification_left),
        "R2": serializable_matrix(context.rectification_right),
        "P1_opencv": serializable_matrix(context.projection_left),
        "P2_opencv": serializable_matrix(context.projection_right),
        "Q_opencv": serializable_matrix(context.q_opencv),
        "P1_positive_disparity": serializable_matrix(projection_left_positive),
        "P2_positive_disparity": serializable_matrix(projection_right_positive),
        "Q_positive_disparity": serializable_matrix(q_positive),
        "valid_roi_left": list(context.roi_left),
        "valid_roi_right": list(context.roi_right),
        "feature_validation": context.feature_validation,
        "depth_rectification": {
            "requested": not args.no_rectify_depth,
            "interpolation": "nearest",
            "map": "left rectification map",
            "invalid_value": 0,
        },
        "vertical_refinement": {
            "requested": not args.no_refine_vertical,
            "applied": context.vertical_refinement_applied,
            "minimum_p90_improvement_px": 0.01,
            "scope": "one fixed model per sequence",
            "model": "delta_y = c0 + cx*x_normalized + cy*y_normalized",
            "coefficients_c0_cx_cy": context.vertical_coefficients.tolist(),
        },
    }


def rectification_signature(parameters: dict) -> dict:
    return {
        "image_size_width_height": parameters["image_size_width_height"],
        "alpha": parameters["alpha"],
        "input_treated_as_already_undistorted": parameters[
            "input_treated_as_already_undistorted"
        ],
        "rectification_mode_used": parameters["rectification_mode_used"],
        "keys": parameters["keys"],
        "R1": parameters["R1"],
        "R2": parameters["R2"],
        "P1_opencv": parameters["P1_opencv"],
        "P2_opencv": parameters["P2_opencv"],
        "depth_rectification": parameters["depth_rectification"],
        "vertical_refinement": parameters["vertical_refinement"],
    }


def process_sequence_batch(
    sequence_dir: Path,
    args: argparse.Namespace,
) -> dict:
    all_pairs = paired_image_paths(sequence_dir)
    pairs = all_pairs[: args.limit] if args.limit is not None else all_pairs
    depth_source_dir = sequence_dir / "depth_gt"
    rectify_depth = depth_source_dir.is_dir() and not args.no_rectify_depth
    if rectify_depth:
        missing_depth = [
            left_path.name
            for left_path, _ in pairs
            if not (depth_source_dir / left_path.name).exists()
        ]
        if missing_depth:
            raise FileNotFoundError(
                f"Missing depth_gt files in {sequence_dir.name}: {missing_depth[:5]}"
            )
    first_left = read_image(pairs[0][0])
    first_right = read_image(pairs[0][1])
    if first_left.shape[:2] != first_right.shape[:2]:
        raise ValueError(
            f"First pair dimensions differ in {sequence_dir.name}: "
            f"{first_left.shape[:2]} vs {first_right.shape[:2]}"
        )
    height, width = first_left.shape[:2]
    context = create_rectification_context(sequence_dir, (width, height), args)
    estimate_sequence_refinement(pairs, context, args)
    output_dir = sequence_dir / args.output_subdir
    left_output_dir = output_dir / "left"
    right_output_dir = output_dir / "right"
    if args.output_subdir.startswith("images_"):
        depth_output_name = "depth_gt_" + args.output_subdir[len("images_") :]
    else:
        depth_output_name = args.output_subdir + "_depth_gt"
    depth_output_dir = sequence_dir / depth_output_name
    parameters_path = output_dir / "rectification_parameters.json"
    parameters = batch_parameters(sequence_dir, context, args)
    parameters["depth_rectification"].update(
        {
            "source_available": depth_source_dir.is_dir(),
            "applied": rectify_depth,
            "output_dir": str(depth_output_dir) if rectify_depth else None,
        }
    )
    has_existing_outputs = (
        any(left_output_dir.glob("*.png"))
        or any(right_output_dir.glob("*.png"))
        or (rectify_depth and any(depth_output_dir.glob("*.png")))
    )
    if args.resume and has_existing_outputs:
        if not parameters_path.exists():
            raise FileNotFoundError(
                "Cannot safely resume without existing rectification parameters: "
                f"{parameters_path}"
            )
        previous_parameters = load_json(parameters_path)
        if rectification_signature(previous_parameters) != rectification_signature(
            parameters
        ):
            raise RuntimeError(
                "Existing outputs use different rectification parameters; "
                f"use --overwrite for sequence {sequence_dir.name}"
            )
    parameters["pair_counts"] = {
        "available": len(all_pairs),
        "selected": len(pairs),
        "processed": 0,
        "skipped": 0,
        "depth_processed": 0,
    }
    write_json(parameters_path, parameters)
    processed = 0
    skipped = 0
    depth_processed = 0
    preview_pair = None
    for pair_index, (left_path, right_path) in enumerate(pairs, start=1):
        left_output = left_output_dir / left_path.name
        right_output = right_output_dir / right_path.name
        depth_output = depth_output_dir / left_path.name
        left_exists = left_output.exists()
        right_exists = right_output.exists()
        depth_exists = depth_output.exists() if rectify_depth else True
        complete = left_exists and right_exists and depth_exists
        if complete and args.resume:
            skipped += 1
            continue
        if (left_exists or right_exists or (rectify_depth and depth_exists)) and not (
            args.resume or args.overwrite
        ):
            raise FileExistsError(
                "Rectified output exists; use --resume for matching parameters "
                f"or --overwrite: {left_output}, {right_output}, {depth_output}"
            )
        need_left = args.overwrite or not left_exists
        need_right = args.overwrite or not right_exists
        need_depth = rectify_depth and (args.overwrite or not depth_exists)
        if need_left or need_right:
            left_image = read_image(left_path)
            right_image = read_image(right_path)
            left_rectified, right_rectified = rectify_with_context(
                left_image,
                right_image,
                context,
                context.vertical_refinement_applied,
            )
            if need_left:
                write_image(left_output, left_rectified, args.overwrite)
            if need_right:
                write_image(right_output, right_rectified, args.overwrite)
            if preview_pair is None:
                preview_pair = (left_rectified, right_rectified)
        if need_depth:
            depth = read_image_unchanged(depth_source_dir / left_path.name)
            if depth.ndim != 2 or depth.shape != (height, width):
                raise ValueError(
                    f"Invalid depth shape for {left_path.name}: {depth.shape}; "
                    f"expected {(height, width)}"
                )
            depth_rectified = cv2.remap(
                depth,
                context.map_left_x,
                context.map_left_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            write_image(depth_output, depth_rectified, args.overwrite)
            depth_processed += 1
        processed += 1
        if pair_index % 25 == 0 or pair_index == len(pairs):
            print(
                f"[{sequence_dir.name}] {pair_index}/{len(pairs)} pairs "
                f"(processed={processed}, skipped={skipped}, "
                f"depth={depth_processed})",
                flush=True,
            )
    if preview_pair is not None:
        write_image(
            output_dir / "epipolar_preview.png",
            build_preview(*preview_pair),
            overwrite=True,
        )
    parameters["pair_counts"] = {
        "available": len(all_pairs),
        "selected": len(pairs),
        "processed": processed,
        "skipped": skipped,
        "depth_processed": depth_processed,
    }
    write_json(parameters_path, parameters)
    return {
        "sequence": sequence_dir.name,
        "status": "completed",
        "output_dir": str(output_dir),
        "depth_output_dir": str(depth_output_dir) if rectify_depth else None,
        "pairs_available": len(all_pairs),
        "pairs_selected": len(pairs),
        "pairs_processed": processed,
        "pairs_skipped": skipped,
        "depth_frames_processed": depth_processed,
        "input_treated_as_already_undistorted": context.zero_distortion,
        "vertical_before": context.feature_validation[
            "before_vertical_refinement"
        ],
        "vertical_after": context.feature_validation[
            "after_vertical_refinement"
        ],
    }

def batch_main(args: argparse.Namespace) -> int:
    root = args.input_root.expanduser().resolve()
    sequences = discover_sequences(root, args.sequence)
    print(
        f"batch_root={root} sequences={len(sequences)} "
        f"output_subdir={args.output_subdir}",
        flush=True,
    )
    reports = []
    for index, sequence_dir in enumerate(sequences, start=1):
        print(
            f"sequence {index}/{len(sequences)}: {sequence_dir.name}", flush=True
        )
        try:
            reports.append(process_sequence_batch(sequence_dir, args))
        except Exception as error:
            reports.append(
                {
                    "sequence": sequence_dir.name,
                    "status": "failed",
                    "error": str(error),
                }
            )
            print(f"[{sequence_dir.name}] failed: {error}", file=sys.stderr, flush=True)
    summary = {
        "input_root": str(root),
        "output_subdir": args.output_subdir,
        "sequence_count": len(sequences),
        "completed_sequences": sum(
            report["status"] == "completed" for report in reports
        ),
        "failed_sequences": sum(report["status"] == "failed" for report in reports),
        "pairs_processed": sum(report.get("pairs_processed", 0) for report in reports),
        "pairs_skipped": sum(report.get("pairs_skipped", 0) for report in reports),
        "depth_frames_processed": sum(
            report.get("depth_frames_processed", 0) for report in reports
        ),
        "sequences": reports,
    }
    report_path = root / f"{args.output_subdir}_batch_report.json"
    write_json(report_path, summary)
    print(f"batch_report={report_path}", flush=True)
    print(
        f"completed_sequences={summary['completed_sequences']} "
        f"failed_sequences={summary['failed_sequences']} "
        f"pairs_processed={summary['pairs_processed']} "
        f"pairs_skipped={summary['pairs_skipped']} "
        f"depth_frames_processed={summary['depth_frames_processed']}",
        flush=True,
    )
    return 1 if summary["failed_sequences"] else 0

def single_main(args: argparse.Namespace) -> int:
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
    zero_distortion = args.distortion_mode == "already-undistorted"
    camera_left, distortion_left = scaled_camera_matrix(
        intrinsics[args.left_key],
        image_size,
        zero_distortion,
    )
    camera_right, distortion_right = scaled_camera_matrix(
        intrinsics[args.right_key],
        image_size,
        zero_distortion,
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
            "left_rectified": args.left_rectified_key,
            "right_rectified": args.right_rectified_key,
            "left_rectification": args.left_rectification_key,
            "right_rectification": args.right_rectification_key,
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


def main() -> int:
    args = parse_args()
    if args.input_root is not None:
        return batch_main(args)
    return single_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
