import argparse
import distutils.version
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.datasets import KITTI_completion
from core.sdg_depth.net import SDGDepth
from core.utils.utils import InputPadder
from infer_one_kitti import build_model_args, compute_metrics, save_visualization


def crop_sample(image1, image2, flow_gt, valid_gt, hint, crop_height, crop_width):
    _, height, width = image1.shape
    if crop_height > height or crop_width > width:
        raise ValueError(f"crop {crop_height}x{crop_width} exceeds image {height}x{width}")
    y0 = height - crop_height
    x0 = (width - crop_width) // 2
    return (
        image1[:, y0 : y0 + crop_height, x0 : x0 + crop_width],
        image2[:, y0 : y0 + crop_height, x0 : x0 + crop_width],
        flow_gt[:, y0 : y0 + crop_height, x0 : x0 + crop_width],
        valid_gt[y0 : y0 + crop_height, x0 : x0 + crop_width],
        hint[:, y0 : y0 + crop_height, x0 : x0 + crop_width],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="premodel/model_kitti.pth")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--crop-width", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs/kitti_batch_demo")
    parser.add_argument("--skip-visualizations", action="store_true")
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    model_args = build_model_args()
    dataset = KITTI_completion({}, image_set=args.split, args=model_args)
    end_index = min(args.start_index + args.num_samples, len(dataset))
    if args.start_index < 0 or args.start_index >= end_index:
        raise ValueError(f"invalid range [{args.start_index}, {end_index}) for dataset of {len(dataset)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDGDepth(model_args.max_disp, use_concat_volume=True, args=model_args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        "valid_pixels": 0,
        "abs": 0.0,
        "squared": 0.0,
        "abs_rel": 0.0,
        "sq_rel": 0.0,
        "inverse_abs": 0.0,
        "inverse_squared": 0.0,
    }
    samples = []

    for index in range(args.start_index, end_index):
        paths, image1, image2, flow_gt, valid_gt, hint, conversion_rate = dataset[index]
        image1, image2, flow_gt, valid_gt, hint = crop_sample(
            image1, image2, flow_gt, valid_gt, hint, args.crop_height, args.crop_width
        )
        image_for_visualization = image1.clone()
        conversion_rate_value = conversion_rate.item()
        gt_depth = torch.zeros_like(flow_gt[0])
        gt_depth[valid_gt > 0] = conversion_rate_value / flow_gt[0][valid_gt > 0]

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        hint = hint[None].to(device)
        conversion_rate_gpu = conversion_rate[None].to(device)
        padder = InputPadder(image1.shape, divis_by=64)
        image1, image2 = padder.pad(image1, image2)
        hint = padder.pad(hint)[0]

        with torch.no_grad():
            depth_list, _, _, _, dense_hint, _ = model(
                image1,
                image2,
                sparse=hint,
                sparse_mask=(hint > 0).int(),
                conversion_rate=conversion_rate_gpu,
            )

        depth = padder.unpad(depth_list[-1].unsqueeze(1)).cpu().squeeze()
        dense_hint = padder.unpad(dense_hint).cpu().squeeze()
        propagated_hint_depth = torch.zeros_like(dense_hint)
        propagated_hint_depth[dense_hint > 0] = conversion_rate_value / dense_hint[dense_hint > 0]
        valid = (valid_gt > 0) & (gt_depth > 0) & torch.isfinite(depth) & (depth > 0)
        metrics = compute_metrics(depth, gt_depth, valid)

        error = depth[valid] - gt_depth[valid]
        inverse_error = 1.0 / depth[valid] - 1.0 / gt_depth[valid]
        totals["valid_pixels"] += int(valid.sum().item())
        totals["abs"] += float(error.abs().sum().item())
        totals["squared"] += float(error.square().sum().item())
        totals["abs_rel"] += float((error.abs() / gt_depth[valid]).sum().item())
        totals["sq_rel"] += float((error.square() / gt_depth[valid]).sum().item())
        totals["inverse_abs"] += float(inverse_error.abs().sum().item())
        totals["inverse_squared"] += float(inverse_error.square().sum().item())

        figure_path = output_dir / f"sample_{index:04d}" / "kitti_inference_visualization.png"
        if not args.skip_visualizations:
            figure_path = save_visualization(
                figure_path.parent, image_for_visualization, propagated_hint_depth, depth, gt_depth, valid, metrics
            )
        samples.append({"index": index, "paths": paths, "metrics": metrics, "visualization": str(figure_path)})
        print(f"[{index + 1}/{end_index}] RMSE={metrics['rmse_m']:.3f} m, MAE={metrics['mae_m']:.3f} m")

    count = totals["valid_pixels"]
    aggregate = {
        "valid_pixels": count,
        "rmse_m": (totals["squared"] / count) ** 0.5,
        "mae_m": totals["abs"] / count,
        "abs_rel": totals["abs_rel"] / count,
        "sq_rel_m": totals["sq_rel"] / count,
        "irmse_1km": 1000.0 * (totals["inverse_squared"] / count) ** 0.5,
        "imae_1km": 1000.0 * totals["inverse_abs"] / count,
    }
    report = {
        "dataset_split": args.split,
        "sample_range": [args.start_index, end_index - 1],
        "crop": [args.crop_height, args.crop_width],
        "aggregate_metrics": aggregate,
        "samples": samples,
    }
    report_path = output_dir / "batch_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("aggregate metrics:")
    print(json.dumps(aggregate, indent=2))
    print("report:", report_path)


if __name__ == "__main__":
    main()
