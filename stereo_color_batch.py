#!/usr/bin/env python3
"""
Batch stereo colour correction with verified backup-before-overwrite.

Pairs images by identical relative paths under --left-dir and --right-dir,
uses the left image as the reference, first creates and verifies an exact
backup of every paired right image, and then atomically replaces the original
right images with corrected versions without changing image geometry.

Dependencies:
    Python 3.9+
    numpy
    opencv-python >= 4.5

Typical use:
    python stereo_color_batch.py ^
        --left-dir D:\\dataset\\left ^
        --right-dir D:\\dataset\\right
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class Pair:
    relative_path: Path
    left_path: Path
    right_path: Path
    output_path: Path


@dataclass
class LoadedImage:
    rgb: np.ndarray
    alpha: np.ndarray | None
    dtype: np.dtype
    scale: float


@dataclass
class PairSamples:
    left_linear: np.ndarray
    right_linear: np.ndarray
    positions: np.ndarray
    ratio_matches: int
    geometric_inliers: int
    usable_pairs: int
    median_dx: float
    median_dy: float


@dataclass
class CorrectionModel:
    ccm: np.ndarray
    spatial_coefficients: np.ndarray
    spatial_degree: int
    luminance_intercept: float
    luminance_slope: float


class PairAnalysisError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correct right-eye images to match identically named left-eye "
            "reference images. All paired right originals are backed up and "
            "verified before any right image is overwritten."
        )
    )
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help=(
            "Directory for exact right-image backups. Defaults to a timestamped "
            "sibling such as right_original_backup_20260723_151500."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "shared", "per-pair"),
        default="hybrid",
        help=(
            "hybrid: per-pair luminance plus one shared colour/spatial model "
            "(recommended); shared: one model for everything; per-pair: fit "
            "a complete model independently for every pair."
        ),
    )
    parser.add_argument(
        "--max-calibration-pairs",
        type=int,
        default=30,
        help="Maximum evenly sampled pairs used to estimate a shared model.",
    )
    parser.add_argument(
        "--max-samples-per-pair",
        type=int,
        default=1200,
        help="Maximum reliable colour correspondences retained from each pair.",
    )
    parser.add_argument(
        "--min-colour-pairs",
        type=int,
        default=80,
        help="Minimum reliable correspondences required to fit one pair.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        choices=(2, 3),
        default=2,
        help="Degree of the smooth spatial colour field; 2 is safer.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated image extensions.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only match files directly inside the two input directories.",
    )
    parser.add_argument(
        "--failure",
        choices=("skip", "error"),
        default="skip",
        help=(
            "Behaviour when a pair cannot be corrected after backup. The "
            "original right image remains unchanged when skipped."
        ),
    )
    parser.add_argument(
        "--linear-input",
        action="store_true",
        help="Treat RGB values as linear instead of sRGB-encoded.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        help="Optional .npz path for the fitted shared model.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional CSV report path.",
    )
    return parser.parse_args()


def normalize_extensions(text: str) -> set[str]:
    result = set()
    for item in text.split(","):
        extension = item.strip().lower()
        if not extension:
            continue
        result.add(extension if extension.startswith(".") else f".{extension}")
    if not result:
        raise ValueError("No valid image extensions were supplied")
    return result


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_directories(left_dir: Path, right_dir: Path, backup_dir: Path) -> None:
    if not left_dir.is_dir():
        raise FileNotFoundError(f"Left directory does not exist: {left_dir}")
    if not right_dir.is_dir():
        raise FileNotFoundError(f"Right directory does not exist: {right_dir}")
    if backup_dir == left_dir or backup_dir == right_dir:
        raise ValueError("Backup directory must differ from both input directories")
    if left_dir in backup_dir.parents or right_dir in backup_dir.parents:
        raise ValueError("Backup directory cannot be inside an input directory")
    if backup_dir in left_dir.parents or backup_dir in right_dir.parents:
        raise ValueError("Backup directory cannot contain an input directory")
    if backup_dir.exists():
        raise FileExistsError(
            f"Backup directory already exists; choose a new path: {backup_dir}"
        )


def discover_pairs(
    left_dir: Path,
    right_dir: Path,
    extensions: set[str],
    recursive: bool,
) -> tuple[list[Pair], list[Path]]:
    iterator: Iterable[Path]
    iterator = left_dir.rglob("*") if recursive else left_dir.glob("*")
    pairs = []
    missing_right = []
    for left_path in sorted(iterator):
        if not left_path.is_file() or left_path.suffix.lower() not in extensions:
            continue
        relative = left_path.relative_to(left_dir)
        right_path = right_dir / relative
        if not right_path.is_file():
            missing_right.append(relative)
            continue
        pairs.append(
            Pair(
                relative_path=relative,
                left_path=left_path,
                right_path=right_path,
                output_path=right_path,
            )
        )
    return pairs, missing_right


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * rgb ** (1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def to_linear(rgb: np.ndarray, linear_input: bool) -> np.ndarray:
    return np.clip(rgb, 0.0, 1.0).astype(np.float32) if linear_input else srgb_to_linear(rgb)


def from_linear(rgb: np.ndarray, linear_input: bool) -> np.ndarray:
    return np.clip(rgb, 0.0, 1.0).astype(np.float32) if linear_input else linear_to_srgb(rgb)


def read_image(path: Path) -> LoadedImage:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise RuntimeError(f"Only RGB/RGBA images are supported: {path}")
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise RuntimeError(f"Unsupported image dtype {image.dtype}: {path}")
    alpha = image[:, :, 3].copy() if image.shape[2] == 4 else None
    bgr = image[:, :, :3]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / scale
    return LoadedImage(rgb=rgb, alpha=alpha, dtype=image.dtype, scale=scale)


def write_image(path: Path, rgb: np.ndarray, source: LoadedImage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = np.clip(np.rint(rgb * source.scale), 0, source.scale).astype(source.dtype)
    bgr = cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR)
    output = np.dstack([bgr, source.alpha]) if source.alpha is not None else bgr
    extension = path.suffix.lower()
    parameters = []
    if extension in (".jpg", ".jpeg"):
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif extension == ".png":
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, encoded = cv2.imencode(extension, output, parameters)
    if not ok:
        raise RuntimeError(f"Cannot encode output image: {path}")
    temporary_path = path.with_name(
        f".{path.stem}.stereo_colour_tmp{path.suffix}"
    )
    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary output already exists; remove it before retrying: "
            f"{temporary_path}"
        )
    try:
        encoded.tofile(str(temporary_path))
        if temporary_path.stat().st_size <= 0:
            raise RuntimeError(f"Temporary output is empty: {temporary_path}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def patch_median(rgb: np.ndarray, x: float, y: float, radius: int = 4) -> np.ndarray:
    height, width = rgb.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - radius), min(width, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(height, yi + radius + 1)
    return np.median(rgb[y0:y1, x0:x1].reshape(-1, 3), axis=0)


def grayscale_u8(rgb: np.ndarray) -> np.ndarray:
    rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)


def estimate_correspondences(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    left_gray = grayscale_u8(left_rgb)
    right_gray = grayscale_u8(right_rgb)
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
        raise PairAnalysisError("No usable image descriptors")
    raw_matches = cv2.BFMatcher(norm).knnMatch(
        right_descriptors,
        left_descriptors,
        k=2,
    )
    ratio_matches = [
        first
        for first, second in raw_matches
        if first.distance < ratio_threshold * second.distance
    ]
    if len(ratio_matches) < 12:
        raise PairAnalysisError(f"Too few descriptor matches: {len(ratio_matches)}")
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
        raise PairAnalysisError("Fundamental matrix estimation failed")
    keep = mask.ravel().astype(bool)
    if int(keep.sum()) < 12:
        raise PairAnalysisError(f"Too few geometric inliers: {int(keep.sum())}")
    return left_points[keep], right_points[keep], len(ratio_matches)


def analyse_pair(
    pair: Pair,
    min_colour_pairs: int,
    max_samples: int,
    linear_input: bool,
) -> PairSamples:
    left = read_image(pair.left_path)
    right = read_image(pair.right_path)
    left_points, right_points, ratio_matches = estimate_correspondences(
        left.rgb,
        right.rgb,
    )
    right_height, right_width = right.rgb.shape[:2]
    left_samples = []
    right_samples = []
    positions = []
    displacements = []
    luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    for left_point, right_point in zip(left_points, right_points):
        left_colour = patch_median(left.rgb, *left_point)
        right_colour = patch_median(right.rgb, *right_point)
        left_luminance = float(left_colour @ luminance_weights)
        right_luminance = float(right_colour @ luminance_weights)
        if (
            min(left_luminance, right_luminance) < 0.045
            or max(left_luminance, right_luminance) > 0.92
            or left_colour.max() > 0.97
            or right_colour.max() > 0.97
        ):
            continue
        left_samples.append(left_colour)
        right_samples.append(right_colour)
        positions.append(
            (
                right_point[0] / max(1, right_width - 1) * 2.0 - 1.0,
                right_point[1] / max(1, right_height - 1) * 2.0 - 1.0,
            )
        )
        displacements.append(left_point - right_point)
    if len(left_samples) < min_colour_pairs:
        raise PairAnalysisError(
            f"Only {len(left_samples)} reliable colour pairs; "
            f"need at least {min_colour_pairs}"
        )
    left_samples_array = np.asarray(left_samples, dtype=np.float32)
    right_samples_array = np.asarray(right_samples, dtype=np.float32)
    positions_array = np.asarray(positions, dtype=np.float32)
    displacements_array = np.asarray(displacements, dtype=np.float32)
    if len(left_samples_array) > max_samples:
        seed = sum(pair.relative_path.as_posix().encode("utf-8")) % (2**32)
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            len(left_samples_array),
            size=max_samples,
            replace=False,
        )
        left_samples_array = left_samples_array[selected]
        right_samples_array = right_samples_array[selected]
        positions_array = positions_array[selected]
        displacements_array = displacements_array[selected]
    median_displacement = np.median(displacements_array, axis=0)
    return PairSamples(
        left_linear=to_linear(left_samples_array, linear_input),
        right_linear=to_linear(right_samples_array, linear_input),
        positions=positions_array,
        ratio_matches=ratio_matches,
        geometric_inliers=len(left_points),
        usable_pairs=len(left_samples_array),
        median_dx=float(median_displacement[0]),
        median_dy=float(median_displacement[1]),
    )


def robust_luminance_power(
    right_linear: np.ndarray,
    left_linear: np.ndarray,
) -> tuple[float, float]:
    luminance_weights = np.array([0.2126, 0.7152, 0.0722])
    x = np.log(np.clip(right_linear @ luminance_weights, 1e-4, None))
    y = np.log(np.clip(left_linear @ luminance_weights, 1e-4, None))
    design = np.stack([np.ones_like(x), x], axis=1)
    weights = np.ones(len(x))
    parameters = np.array([0.0, 1.0])
    prior = np.diag([0.02, 0.02])
    prior_target = np.array([0.0, 1.0])
    for _ in range(10):
        weighted = np.sqrt(weights)[:, None]
        a = design * weighted
        b = y * weighted[:, 0]
        parameters = np.linalg.solve(
            a.T @ a + prior,
            a.T @ b + prior @ prior_target,
        )
        errors = design @ parameters - y
        median = np.median(errors)
        sigma = 1.4826 * np.median(np.abs(errors - median)) + 1e-8
        threshold = max(2.5 * sigma, 0.02)
        weights = np.minimum(1.0, threshold / (np.abs(errors) + 1e-8))
    intercept = float(np.clip(parameters[0], -1.2, 1.2))
    slope = float(np.clip(parameters[1], 0.70, 1.30))
    return intercept, slope


def apply_luminance_power(
    rgb_linear: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance = np.clip(rgb_linear @ luminance_weights, 1e-5, None)
    target_luminance = np.exp(intercept) * luminance**slope
    scale = np.clip(target_luminance / luminance, 0.35, 1.8)
    return rgb_linear * scale[..., None]


def robust_ccm(
    right_linear: np.ndarray,
    left_linear: np.ndarray,
    prior_strength: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(len(right_linear), dtype=np.float64)
    identity = np.eye(3)
    matrix = identity.copy()
    source = right_linear.astype(np.float64)
    target = left_linear.astype(np.float64)
    for _ in range(8):
        weighted = np.sqrt(weights)[:, None]
        a = source * weighted
        b = target * weighted
        gram = a.T @ a
        ridge = prior_strength * np.trace(gram) / 3.0
        matrix = np.linalg.solve(
            gram + ridge * identity,
            a.T @ b + ridge * identity,
        )
        residual = np.linalg.norm(source @ matrix - target, axis=1)
        median = np.median(residual)
        sigma = 1.4826 * np.median(np.abs(residual - median)) + 1e-8
        threshold = max(2.5 * sigma, 0.008)
        weights = np.minimum(1.0, threshold / (residual + 1e-8))
    return matrix.astype(np.float32), weights


def polynomial_basis(xy: np.ndarray, degree: int) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    columns = [np.ones_like(x), x, y, x * x, x * y, y * y]
    if degree >= 3:
        columns.extend([x**3, x * x * y, x * y * y, y**3])
    return np.stack(columns, axis=1)


def robust_spatial_fit(
    positions: np.ndarray,
    log_residual: np.ndarray,
    degree: int,
    base_weights: np.ndarray,
) -> np.ndarray:
    basis = polynomial_basis(positions.astype(np.float64), degree)
    penalties = np.ones(basis.shape[1])
    penalties[0] = 0.2
    if degree >= 3:
        penalties[6:] = 4.0
    ridge_strength = 0.12 if degree == 2 else 0.22
    coefficients = np.zeros((basis.shape[1], 3), dtype=np.float64)
    for channel in range(3):
        weights = base_weights.astype(np.float64).copy()
        for _ in range(8):
            weighted = np.sqrt(weights)[:, None]
            a = basis * weighted
            b = log_residual[:, channel] * weighted[:, 0]
            gram = a.T @ a
            ridge = ridge_strength * np.trace(gram) / max(1, gram.shape[0])
            coefficients[:, channel] = np.linalg.solve(
                gram + ridge * np.diag(penalties),
                a.T @ b,
            )
            errors = basis @ coefficients[:, channel] - log_residual[:, channel]
            median = np.median(errors)
            sigma = 1.4826 * np.median(np.abs(errors - median)) + 1e-8
            threshold = max(2.5 * sigma, 0.015)
            robust = np.minimum(1.0, threshold / (np.abs(errors) + 1e-8))
            weights = base_weights * robust
    return coefficients.astype(np.float32)


def fit_colour_and_spatial_model(
    right_luminance_matched: np.ndarray,
    left_linear: np.ndarray,
    positions: np.ndarray,
    degree: int,
    luminance_intercept: float,
    luminance_slope: float,
) -> CorrectionModel:
    ccm, ccm_weights = robust_ccm(right_luminance_matched, left_linear)
    global_prediction = np.clip(right_luminance_matched @ ccm, 1e-5, 1.5)
    safe = (
        (global_prediction.min(axis=1) > 0.006)
        & (left_linear.min(axis=1) > 0.006)
        & (global_prediction.max(axis=1) < 1.2)
    )
    if int(safe.sum()) < 40:
        raise PairAnalysisError("Too few safe samples for the spatial model")
    log_residual = np.log(
        (left_linear[safe] + 0.004) / (global_prediction[safe] + 0.004)
    )
    coefficients = robust_spatial_fit(
        positions[safe],
        log_residual,
        degree,
        np.clip(ccm_weights[safe], 0.05, 1.0),
    )
    return CorrectionModel(
        ccm=ccm,
        spatial_coefficients=coefficients,
        spatial_degree=degree,
        luminance_intercept=luminance_intercept,
        luminance_slope=luminance_slope,
    )


def fit_complete_model(samples: PairSamples, degree: int) -> CorrectionModel:
    intercept, slope = robust_luminance_power(
        samples.right_linear,
        samples.left_linear,
    )
    right_matched = apply_luminance_power(samples.right_linear, intercept, slope)
    return fit_colour_and_spatial_model(
        right_matched,
        samples.left_linear,
        samples.positions,
        degree,
        intercept,
        slope,
    )


def evaluate_spatial_field(
    height: int,
    width: int,
    coefficients: np.ndarray,
    degree: int,
) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :, None]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
    field = (
        coefficients[0][None, None, :]
        + x * coefficients[1][None, None, :]
        + y * coefficients[2][None, None, :]
        + x * x * coefficients[3][None, None, :]
        + x * y * coefficients[4][None, None, :]
        + y * y * coefficients[5][None, None, :]
    )
    if degree >= 3:
        field = (
            field
            + x**3 * coefficients[6][None, None, :]
            + x * x * y * coefficients[7][None, None, :]
            + x * y * y * coefficients[8][None, None, :]
            + y**3 * coefficients[9][None, None, :]
        )
    return np.clip(field, -0.38, 0.38)


def apply_model(
    right_rgb: np.ndarray,
    model: CorrectionModel,
    linear_input: bool,
    luminance_override: tuple[float, float] | None = None,
) -> np.ndarray:
    intercept, slope = (
        luminance_override
        if luminance_override is not None
        else (model.luminance_intercept, model.luminance_slope)
    )
    right_linear = to_linear(right_rgb, linear_input)
    right_matched = apply_luminance_power(right_linear, intercept, slope)
    global_corrected = np.clip(right_matched @ model.ccm, 0.0, 1.0)
    height, width = right_rgb.shape[:2]
    field = evaluate_spatial_field(
        height,
        width,
        model.spatial_coefficients,
        model.spatial_degree,
    )
    corrected_linear = np.clip(global_corrected * np.exp(field), 0.0, 1.0)
    return from_linear(corrected_linear, linear_input)


def apply_model_to_samples(
    samples: PairSamples,
    model: CorrectionModel,
    luminance_override: tuple[float, float] | None = None,
) -> np.ndarray:
    intercept, slope = (
        luminance_override
        if luminance_override is not None
        else (model.luminance_intercept, model.luminance_slope)
    )
    right_matched = apply_luminance_power(samples.right_linear, intercept, slope)
    global_corrected = np.clip(right_matched @ model.ccm, 0.0, 1.0)
    field = np.clip(
        polynomial_basis(samples.positions, model.spatial_degree)
        @ model.spatial_coefficients,
        -0.38,
        0.38,
    )
    return np.clip(global_corrected * np.exp(field), 0.0, 1.0)


def select_evenly(items: list[Pair], maximum: int) -> list[Pair]:
    if maximum <= 0 or len(items) <= maximum:
        return list(items)
    indices = np.linspace(0, len(items) - 1, maximum)
    unique_indices = sorted(set(np.rint(indices).astype(int).tolist()))
    return [items[index] for index in unique_indices]


def fit_shared_model(
    pairs: list[Pair],
    args: argparse.Namespace,
) -> tuple[
    CorrectionModel,
    dict[str, tuple[float, float]],
    dict[str, PairSamples],
]:
    calibration_pairs = select_evenly(pairs, args.max_calibration_pairs)
    left_pool = []
    right_pool = []
    position_pool = []
    luminance_cache: dict[str, tuple[float, float]] = {}
    samples_cache: dict[str, PairSamples] = {}
    successful_samples: list[PairSamples] = []
    successful_pairs: list[Pair] = []
    for index, pair in enumerate(calibration_pairs, start=1):
        print(
            f"[calibration {index}/{len(calibration_pairs)}] "
            f"{pair.relative_path.as_posix()}"
        )
        try:
            samples = analyse_pair(
                pair,
                args.min_colour_pairs,
                args.max_samples_per_pair,
                args.linear_input,
            )
            successful_samples.append(samples)
            successful_pairs.append(pair)
            samples_cache[pair.relative_path.as_posix()] = samples
        except Exception as error:
            print(f"  warning: {error}", file=sys.stderr)
    if not successful_samples:
        raise RuntimeError("No calibration pair produced enough reliable matches")

    if args.mode == "hybrid":
        luminance_parameters = []
        for pair, samples in zip(successful_pairs, successful_samples):
            parameters = robust_luminance_power(
                samples.right_linear,
                samples.left_linear,
            )
            luminance_cache[pair.relative_path.as_posix()] = parameters
            luminance_parameters.append(parameters)
            right_pool.append(
                apply_luminance_power(samples.right_linear, *parameters)
            )
            left_pool.append(samples.left_linear)
            position_pool.append(samples.positions)
        default_intercept = float(np.median([item[0] for item in luminance_parameters]))
        default_slope = float(np.median([item[1] for item in luminance_parameters]))
    else:
        raw_right = np.concatenate(
            [samples.right_linear for samples in successful_samples],
            axis=0,
        )
        raw_left = np.concatenate(
            [samples.left_linear for samples in successful_samples],
            axis=0,
        )
        default_intercept, default_slope = robust_luminance_power(raw_right, raw_left)
        for samples in successful_samples:
            right_pool.append(
                apply_luminance_power(
                    samples.right_linear,
                    default_intercept,
                    default_slope,
                )
            )
            left_pool.append(samples.left_linear)
            position_pool.append(samples.positions)

    model = fit_colour_and_spatial_model(
        np.concatenate(right_pool, axis=0),
        np.concatenate(left_pool, axis=0),
        np.concatenate(position_pool, axis=0),
        args.degree,
        default_intercept,
        default_slope,
    )
    return model, luminance_cache, samples_cache


def model_metrics(
    samples: PairSamples,
    corrected_linear: np.ndarray,
    linear_input: bool,
) -> tuple[float, float]:
    original = from_linear(samples.right_linear, linear_input)
    target = from_linear(samples.left_linear, linear_input)
    corrected = from_linear(corrected_linear, linear_input)
    before = float(np.sqrt(np.mean((original - target) ** 2)))
    after = float(np.sqrt(np.mean((corrected - target) ** 2)))
    return before, after


def report_row(
    pair: Pair,
    status: str,
    backup_dir: Path | None = None,
    samples: PairSamples | None = None,
    luminance: tuple[float, float] | None = None,
    rmse_before: float | None = None,
    rmse_after: float | None = None,
    message: str = "",
) -> dict[str, object]:
    return {
        "relative_path": pair.relative_path.as_posix(),
        "backup_path": (
            str(backup_dir / pair.relative_path) if backup_dir is not None else ""
        ),
        "status": status,
        "ratio_matches": samples.ratio_matches if samples else "",
        "geometric_inliers": samples.geometric_inliers if samples else "",
        "usable_colour_pairs": samples.usable_pairs if samples else "",
        "median_dx": f"{samples.median_dx:.3f}" if samples else "",
        "median_dy": f"{samples.median_dy:.3f}" if samples else "",
        "luminance_intercept": f"{luminance[0]:.6f}" if luminance else "",
        "luminance_slope": f"{luminance[1]:.6f}" if luminance else "",
        "matched_srgb_rmse_before": (
            f"{rmse_before:.6f}" if rmse_before is not None else ""
        ),
        "matched_srgb_rmse_after": (
            f"{rmse_after:.6f}" if rmse_after is not None else ""
        ),
        "message": message,
    }


def save_model(path: Path, model: CorrectionModel, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ccm=model.ccm,
        spatial_coefficients=model.spatial_coefficients,
        spatial_degree=np.array(model.spatial_degree),
        luminance_intercept=np.array(model.luminance_intercept),
        luminance_slope=np.array(model.luminance_slope),
        mode=np.array(mode),
    )


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "relative_path",
        "backup_path",
        "status",
        "ratio_matches",
        "geometric_inliers",
        "usable_colour_pairs",
        "median_dx",
        "median_dy",
        "luminance_intercept",
        "luminance_slope",
        "matched_srgb_rmse_before",
        "matched_srgb_rmse_after",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def backup_right_images(pairs: list[Pair], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    for index, pair in enumerate(pairs, start=1):
        destination = backup_dir / pair.relative_path
        print(f"[backup {index}/{len(pairs)}] {pair.relative_path.as_posix()}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pair.right_path, destination)
        source_size = pair.right_path.stat().st_size
        backup_size = destination.stat().st_size
        if source_size != backup_size:
            raise RuntimeError(
                f"Backup size mismatch for {pair.relative_path}: "
                f"{source_size} != {backup_size}"
            )
        if not filecmp.cmp(pair.right_path, destination, shallow=False):
            raise RuntimeError(
                f"Backup content verification failed: {pair.relative_path}"
            )
        sha256 = sha256_file(destination)
        manifest_rows.append(
            {
                "relative_path": pair.relative_path.as_posix(),
                "source_path": str(pair.right_path),
                "backup_path": str(destination),
                "size_bytes": source_size,
                "sha256": sha256,
                "verified": "true",
            }
        )
    manifest_path = backup_dir / "backup_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "relative_path",
                "source_path",
                "backup_path",
                "size_bytes",
                "sha256",
                "verified",
            ),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def process_with_shared_model(
    pairs: list[Pair],
    model: CorrectionModel,
    luminance_cache: dict[str, tuple[float, float]],
    samples_cache: dict[str, PairSamples],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows = []
    for index, pair in enumerate(pairs, start=1):
        print(f"[process {index}/{len(pairs)}] {pair.relative_path.as_posix()}")
        key = pair.relative_path.as_posix()
        samples = samples_cache.get(key)
        luminance_override = None
        message = ""
        if args.mode == "hybrid":
            if key in luminance_cache:
                luminance_override = luminance_cache[key]
            else:
                try:
                    samples = analyse_pair(
                        pair,
                        args.min_colour_pairs,
                        args.max_samples_per_pair,
                        args.linear_input,
                    )
                    luminance_override = robust_luminance_power(
                        samples.right_linear,
                        samples.left_linear,
                    )
                except Exception as error:
                    luminance_override = (
                        model.luminance_intercept,
                        model.luminance_slope,
                    )
                    message = f"used shared luminance fallback: {error}"
                    print(f"  warning: {message}", file=sys.stderr)
        try:
            right = read_image(pair.right_path)
            corrected = apply_model(
                right.rgb,
                model,
                args.linear_input,
                luminance_override,
            )
            write_image(pair.output_path, corrected, right)
            used_luminance = (
                luminance_override
                if luminance_override is not None
                else (model.luminance_intercept, model.luminance_slope)
            )
            rmse_before = None
            rmse_after = None
            if samples is not None:
                corrected_samples = apply_model_to_samples(
                    samples,
                    model,
                    used_luminance,
                )
                rmse_before, rmse_after = model_metrics(
                    samples,
                    corrected_samples,
                    args.linear_input,
                )
            rows.append(
                report_row(
                    pair,
                    "corrected",
                    backup_dir=args.backup_dir,
                    samples=samples,
                    luminance=used_luminance,
                    rmse_before=rmse_before,
                    rmse_after=rmse_after,
                    message=message,
                )
            )
        except Exception as error:
            rows.append(handle_failure(pair, error, args))
    return rows


def process_per_pair(
    pairs: list[Pair],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows = []
    for index, pair in enumerate(pairs, start=1):
        print(f"[process {index}/{len(pairs)}] {pair.relative_path.as_posix()}")
        try:
            samples = analyse_pair(
                pair,
                args.min_colour_pairs,
                args.max_samples_per_pair,
                args.linear_input,
            )
            model = fit_complete_model(samples, args.degree)
            right = read_image(pair.right_path)
            corrected = apply_model(right.rgb, model, args.linear_input)
            write_image(pair.output_path, corrected, right)
            corrected_samples = apply_model_to_samples(samples, model)
            rmse_before, rmse_after = model_metrics(
                samples,
                corrected_samples,
                args.linear_input,
            )
            luminance = (
                model.luminance_intercept,
                model.luminance_slope,
            )
            rows.append(
                report_row(
                    pair,
                    "corrected",
                    backup_dir=args.backup_dir,
                    samples=samples,
                    luminance=luminance,
                    rmse_before=rmse_before,
                    rmse_after=rmse_after,
                )
            )
        except Exception as error:
            rows.append(handle_failure(pair, error, args))
    return rows


def handle_failure(
    pair: Pair,
    error: Exception,
    args: argparse.Namespace,
) -> dict[str, object]:
    message = str(error)
    print(f"  error: {message}", file=sys.stderr)
    if args.failure == "error":
        raise error
    return report_row(
        pair,
        "failed-skipped-original-unchanged",
        backup_dir=args.backup_dir,
        message=message,
    )


def main() -> int:
    args = parse_args()
    args.left_dir = resolved(args.left_dir)
    args.right_dir = resolved(args.right_dir)
    if args.backup_dir:
        args.backup_dir = resolved(args.backup_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.backup_dir = args.right_dir.with_name(
            f"{args.right_dir.name}_original_backup_{timestamp}"
        )
    validate_directories(args.left_dir, args.right_dir, args.backup_dir)
    extensions = normalize_extensions(args.extensions)
    pairs, missing_right = discover_pairs(
        args.left_dir,
        args.right_dir,
        extensions,
        recursive=not args.no_recursive,
    )
    if not pairs:
        raise RuntimeError("No identically named left/right image pairs were found")
    print(f"paired images: {len(pairs)}")
    if missing_right:
        print(
            f"left images without matching right image: {len(missing_right)}",
            file=sys.stderr,
        )
    print(f"backup directory: {args.backup_dir}")
    backup_right_images(pairs, args.backup_dir)
    print("backup verification: all paired right originals are byte-identical")
    report_path = (
        resolved(args.report)
        if args.report
        else args.backup_dir / "stereo_color_report.csv"
    )
    all_rows: list[dict[str, object]] = []

    if args.mode in ("hybrid", "shared"):
        model, luminance_cache, samples_cache = fit_shared_model(pairs, args)
        model_path = (
            resolved(args.model_out)
            if args.model_out
            else args.backup_dir / "stereo_color_model.npz"
        )
        save_model(model_path, model, args.mode)
        all_rows.extend(
            process_with_shared_model(
                pairs,
                model,
                luminance_cache,
                samples_cache,
                args,
            )
        )
        print(f"model: {model_path}")
    else:
        all_rows.extend(process_per_pair(pairs, args))

    for relative in missing_right:
        dummy = Pair(
            relative_path=relative,
            left_path=args.left_dir / relative,
            right_path=args.right_dir / relative,
            output_path=args.right_dir / relative,
        )
        all_rows.append(
            report_row(
                dummy,
                "missing-right",
                backup_dir=args.backup_dir,
            )
        )

    write_report(report_path, all_rows)
    corrected_count = sum(row["status"] == "corrected" for row in all_rows)
    failed_count = sum(str(row["status"]).startswith("failed") for row in all_rows)
    print(f"corrected: {corrected_count}")
    print(f"failed: {failed_count}")
    print(f"verified backup: {args.backup_dir}")
    print(f"report: {report_path}")
    return 0 if corrected_count > 0 and (failed_count == 0 or args.failure != "error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
