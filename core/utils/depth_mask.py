from __future__ import annotations

import torch


def depth_percentile_mask(
    depth: torch.Tensor,
    valid: torch.Tensor,
    lower: float = 0.05,
    upper: float = 0.95,
) -> torch.Tensor:
    """Keep valid depths within per-sample quantile bounds."""
    if depth.shape != valid.shape:
        raise ValueError(
            f"depth and valid shapes differ: {depth.shape} vs {valid.shape}"
        )
    if depth.ndim < 2:
        raise ValueError(f"depth must include batch and spatial dimensions: {depth.shape}")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(
            f"invalid depth percentile interval: lower={lower}, upper={upper}"
        )

    valid = valid.bool() & torch.isfinite(depth) & (depth > 0)
    result = torch.zeros_like(valid)
    for batch_index in range(depth.shape[0]):
        sample_values = depth[batch_index][valid[batch_index]]
        if sample_values.numel() == 0:
            continue
        bounds = torch.quantile(
            sample_values.float(),
            torch.tensor(
                [lower, upper],
                device=sample_values.device,
                dtype=torch.float32,
            ),
        )
        result[batch_index] = (
            valid[batch_index]
            & (depth[batch_index] >= bounds[0])
            & (depth[batch_index] <= bounds[1])
        )
    return result
