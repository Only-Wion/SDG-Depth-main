#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" python -u stereo_lidar.py \
  --checkpoint_dir outputs/luna_finetune_bs8_lr1e6_ep50_exclude2 \
  --train_datasets luna \
  --test_datasets luna \
  --luna_root /mnt/data/jianglai/artifacts/SDG-Depth-main/batch_organized \
  --luna_exclude_sequences 2025-06-26-22-59-05 2025-06-26-23-01-34 \
  --resume_ckpt premodel/model_kitti.pth \
  --strict_resume 0 \
  --guided_flag 1 \
  --batch_size 8 \
  --num_workers 8 \
  --num_steps 3400 \
  --num_epoch 50 \
  --lr 1e-6 \
  --lr_scheduler_type MultiStepLR \
  --milestones 38 \
  --lr_gamma 0.1 \
  --val_epoch 1 \
  --mixed_precision