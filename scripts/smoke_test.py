import argparse
import distutils.version
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.sdg_depth.net import SDGDepth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--max_disp", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    if args_cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    model_args = argparse.Namespace(
        max_disp=args_cli.max_disp,
        guided_flag=1,
        expand_flag=1,
        refine_spn_resolution=4,
        refine_spn_r=4,
        refine_spn_offset_flag=1,
        refine_spn_offset_range=1.0,
        refine_spn_conf_pixel="sparse_valid_all",
        refine_spn_iter_num=5,
        gaussian_h=2,
        gaussian_w=8,
        cfnet_confidence_value=0.4,
        gsm_validhint="conf_04",
        disp_to_depth_convert_resolution=2,
        disp_to_depth_convert_gate=0.1,
        disp_to_depth_convert_disp_range=0.2,
        disp_to_depth_convert_depth_range=0.6,
    )

    device = torch.device(args_cli.device)
    model = SDGDepth(args_cli.max_disp, use_concat_volume=True, args=model_args).to(device)
    model.eval()

    left = torch.randn(1, 3, args_cli.height, args_cli.width, device=device)
    right = torch.randn_like(left)
    sparse = torch.zeros(1, 1, args_cli.height, args_cli.width, device=device)
    sparse[:, :, args_cli.height // 2, args_cli.width // 2] = 8.0
    sparse_mask = (sparse > 0).float()
    conversion_rate = torch.tensor([386.0], device=device)

    with torch.no_grad():
        outputs = model(left, right, sparse=sparse, sparse_mask=sparse_mask, conversion_rate=conversion_rate)

    shapes = []
    for output in outputs:
        if isinstance(output, (list, tuple)):
            shapes.append([tuple(item.shape) for item in output if torch.is_tensor(item)])
        elif torch.is_tensor(output):
            shapes.append(tuple(output.shape))

    print("smoke test ok")
    print("device:", device)
    print("output shapes:", shapes)


if __name__ == "__main__":
    main()
