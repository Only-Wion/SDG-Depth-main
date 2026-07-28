#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python -u stereo_lidar.py \
  --checkpoint_dir outputs/luna_2025-06-26-23-29-56_rectified_out_crop10_fastlio_body_to_lidar_from_pretrained_lr1e4_ep120 \
  --train_datasets luna \
  --test_datasets luna \
  --luna_root /mnt/data/jianglai/artifacts/SDG-Depth-main/batch_organized \
  --luna_exclude_sequences \
    2025-06-26-20-21-08 \
    2025-06-26-22-40-47 \
    2025-06-26-22-59-05 \
    2025-06-26-23-01-34 \
    2025-06-26-23-32-07 \
    2025-06-26-23-38-44 \
    2025-06-26-23-42-20 \
    2025-06-26-23-47-27 \
  --luna_image_subdir images_rectified \
  --luna_left_dirname left_out \
  --luna_right_dirname right_out \
  --luna_depth_subdir depth_gt_rectified \
  --luna_lidar_source fastlio \
  --luna_fastlio_body_to_lidar \
    -0.00584021 0.99995500 -0.00748795 -0.056199899903680 \
    -0.99987300 -0.00572824 0.01488950 0.002601436907349 \
    0.01484590 0.00757395 0.99986100 0.177498078345765 \
    0 0 0 1 \
  --luna_border_crop_fraction 0.1 \
  --resume_ckpt premodel/model_kitti.pth \
  --strict_resume 0 \
  --max_disp 192 \
  --image_size 256 512 \
  --occlusion_aug_prob 0.5 \
  --guided_flag 1 \
  --cfnet_confidence_value 0.4 \
  --gsm_validhint conf_04 \
  --gaussian_h 2 \
  --gaussian_w 8 \
  --refine_spn_r 4 \
  --refine_spn_resolution 4 \
  --refine_spn_conf_pixel sparse_valid_all \
  --refine_spn_offset_flag 1 \
  --disp_to_depth_convert_flag 1 \
  --disp_to_depth_convert_resolution 2 \
  --disp_to_depth_convert_gate 0.1 \
  --disp_to_depth_convert_disp_range 0.2 \
  --disp_to_depth_convert_depth_range 0.6 \
  --pred_hint_weight 0.5 \
  --disp_to_depth_convert_loss_weight1 0.7 \
  --disp_to_depth_convert_loss_weight2 0.7 \
  --batch_size 4 \
  --num_workers 4 \
  --num_steps 1320 \
  --num_epoch 120 \
  --lr 1e-4 \
  --lr_scheduler_type MultiStepLR \
  --milestones 90 \
  --lr_gamma 0.1 \
  --val_epoch 1 \
  --mixed_precision
