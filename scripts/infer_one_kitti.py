import argparse
import distutils.version
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.datasets import KITTI_completion
from core.sdg_depth.net import SDGDepth
from core.utils.utils import InputPadder


def build_model_args():
    return argparse.Namespace(
        max_disp=192,
        guided_flag=1,
        hints_density=0.05,
        more_bottom=0.0,
        expand_flag=1,
        refine_spn_resolution=4,
        refine_spn_r=4,
        refine_spn_offset_flag=1,
        refine_spn_offset_range=1.0,
        refine_spn_conf_pixel="sparse_valid_all",
        gaussian_h=2,
        gaussian_w=8,
        cfnet_confidence_value=0.4,
        gsm_validhint="conf_04",
        disp_to_depth_convert_resolution=2,
        disp_to_depth_convert_gate=0.1,
        disp_to_depth_convert_disp_range=0.2,
        disp_to_depth_convert_depth_range=0.6,
    )


def summarize(name, tensor):
    valid = tensor[tensor > 0]
    if valid.numel() == 0:
        print(f"{name}: no positive values")
        return
    print(
        f"{name}: shape={tuple(tensor.shape)} "
        f"valid={valid.numel()} min={valid.min().item():.4f} "
        f"mean={valid.mean().item():.4f} max={valid.max().item():.4f}"
    )


def compute_metrics(pred, gt, valid):
    pred = pred[valid]
    gt = gt[valid]
    error = pred - gt
    inv_error = 1.0 / pred - 1.0 / gt
    return {
        "valid_pixels": int(valid.sum().item()),
        "rmse_m": float(torch.sqrt(torch.mean(error.square())).item()),
        "mae_m": float(torch.mean(error.abs()).item()),
        "abs_rel": float(torch.mean(error.abs() / gt).item()),
        "sq_rel_m": float(torch.mean(error.square() / gt).item()),
        "irmse_1km": float((1000.0 * torch.sqrt(torch.mean(inv_error.square()))).item()),
        "imae_1km": float((1000.0 * torch.mean(inv_error.abs())).item()),
    }


def save_visualization(output_dir, image, propagated_hint, prediction, ground_truth, valid, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
    propagated_hint_np = propagated_hint.numpy()
    prediction_np = prediction.numpy()
    ground_truth_np = ground_truth.numpy()
    valid_np = valid.numpy()
    error_np = np.abs(prediction_np - ground_truth_np)
    error_np[~valid_np] = np.nan
    depth_limit = float(np.nanpercentile(ground_truth_np[valid_np], 99))
    error_limit = max(1.0, float(np.nanpercentile(error_np[valid_np], 99)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    panels = [
        (image_np, "Left image", None, None),
        (np.ma.masked_where(propagated_hint_np <= 0, propagated_hint_np), "Propagated depth hint (m)", "turbo", depth_limit),
        (prediction_np, "Predicted depth (m)", "turbo", depth_limit),
        (ground_truth_np, "Ground truth depth (m)", "turbo", depth_limit),
        (np.ma.masked_invalid(error_np), "Absolute error (m)", "magma", error_limit),
    ]
    for axis, (data, title, cmap, vmax) in zip(axes.flat, panels):
        if cmap is None:
            axis.imshow(data)
        else:
            rendered = axis.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
            fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
        axis.axis("off")

    axes.flat[-1].axis("off")
    metrics_text = "\n".join([
        "KITTI validation sample (cropped 256 x 512)",
        f"Valid GT pixels: {metrics['valid_pixels']:,}",
        f"RMSE: {metrics['rmse_m']:.3f} m",
        f"MAE:  {metrics['mae_m']:.3f} m",
        f"iRMSE: {metrics['irmse_1km']:.3f} 1/km",
        f"iMAE:  {metrics['imae_1km']:.3f} 1/km",
    ])
    axes.flat[-1].text(0.03, 0.96, metrics_text, va="top", fontsize=15, family="monospace")
    figure_path = output_dir / "kitti_inference_visualization.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="premodel/model_kitti.pth")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--crop-height", type=int)
    parser.add_argument("--crop-width", type=int)
    parser.add_argument("--output-dir", default="outputs/kitti_demo")
    args = parser.parse_args()

    model_args = build_model_args()
    dataset = KITTI_completion({}, image_set=args.split, args=model_args)
    if len(dataset) == 0:
        raise RuntimeError("KITTI dataset is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDGDepth(model_args.max_disp, use_concat_volume=True, args=model_args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    paths, image1, image2, flow_gt, valid_gt, hint, conversion_rate = dataset[args.index]
    if args.crop_height or args.crop_width:
        _, height, width = image1.shape
        crop_height = args.crop_height or height
        crop_width = args.crop_width or width
        if crop_height > height or crop_width > width:
            raise ValueError(f"crop {crop_height}x{crop_width} exceeds image {height}x{width}")
        y0 = height - crop_height
        x0 = (width - crop_width) // 2
        image1 = image1[:, y0 : y0 + crop_height, x0 : x0 + crop_width]
        image2 = image2[:, y0 : y0 + crop_height, x0 : x0 + crop_width]
        flow_gt = flow_gt[:, y0 : y0 + crop_height, x0 : x0 + crop_width]
        valid_gt = valid_gt[y0 : y0 + crop_height, x0 : x0 + crop_width]
        hint = hint[:, y0 : y0 + crop_height, x0 : x0 + crop_width]

    image_for_visualization = image1.clone()
    conversion_rate_cpu = conversion_rate.item()
    gt_disp = flow_gt[0]
    gt_depth = torch.zeros_like(gt_disp)
    gt_depth[valid_gt > 0] = conversion_rate_cpu / gt_disp[valid_gt > 0]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    hint = hint[None].to(device)
    conversion_rate = conversion_rate[None].to(device)

    pad_size = 64 if image1.shape[-1] not in [960] else 128
    padder = InputPadder(image1.shape, divis_by=pad_size)
    image1, image2 = padder.pad(image1, image2)
    hint = padder.pad(hint)[0]
    sparse_mask = (hint > 0).int()

    with torch.no_grad():
        depth_list, disp_list, _, _, dense_hint, confidence = model(
            image1,
            image2,
            sparse=hint,
            sparse_mask=sparse_mask,
            conversion_rate=conversion_rate,
        )

    depth = padder.unpad(depth_list[-1].unsqueeze(1)).cpu().squeeze(0).squeeze(0)
    disp = padder.unpad(disp_list[-1].unsqueeze(1)).cpu().squeeze(0).squeeze(0)
    dense_hint = padder.unpad(dense_hint).cpu().squeeze(0).squeeze(0)
    confidence = padder.unpad(confidence).cpu().squeeze(0).squeeze(0)
    propagated_hint_depth = torch.zeros_like(dense_hint)
    propagated_hint_depth[dense_hint > 0] = conversion_rate_cpu / dense_hint[dense_hint > 0]

    print("inference ok")
    print("device:", device)
    print("sample:", paths)
    print("conversion_rate:", conversion_rate.item())
    summarize("depth_m", depth)
    summarize("disp_px", disp)
    summarize("dense_hint_px", dense_hint)
    summarize("confidence", confidence)

    valid = (valid_gt > 0) & (gt_depth > 0) & torch.isfinite(depth) & (depth > 0)
    metrics = compute_metrics(depth, gt_depth, valid)
    output_dir = Path(args.output_dir)
    figure_path = save_visualization(
        output_dir, image_for_visualization, propagated_hint_depth, depth, gt_depth, valid, metrics
    )
    metrics_path = output_dir / "kitti_inference_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "sample_paths": paths,
                "crop_height": int(depth.shape[0]),
                "crop_width": int(depth.shape[1]),
                "conversion_rate": conversion_rate_cpu,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("metrics:", json.dumps(metrics, indent=2))
    print("visualization:", figure_path)
    print("metrics file:", metrics_path)


if __name__ == "__main__":
    main()
