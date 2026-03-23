#!/bin/bash
export WANDB_MODE=disabled 
# export HF_HOME=/path/to/your/huggingface/cache
# Select GPUs
export CUDA_VISIBLE_DEVICES=2,3,4,5

DATASET_NAME="libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops,libero_10_no_noops"
VLA_PATH="ckpts/openvla-7b"
VGGT_PATH="ckpts/VGGT-1B/model.pt"

# Define layers to profile
# VLA (OpenVLA-7B) has 32 transformer layers (0-31) + 1 projector layer (32)
# VGGT has 24 layers (0-23)
# Configured as variables for easy modification
# To profile all layers, use the following:
VLA_LAYERS_TO_PROFILE="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"
VGGT_LAYERS_TO_PROFILE="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"

# To profile only the last few layers, uncomment the following:
# VLA_LAYERS_TO_PROFILE="-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12"
# VGGT_LAYERS_TO_PROFILE="-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12"

# Run profiling
# Notes:
# 1. Increased batch_size (1 -> 4) to speed up feature collection
# 2. Increased num_profiling_samples (100 -> 2048) for statistically significant CKA results
# 3. Increase batch_size further if VRAM allows
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/profiling_cka.py \
    --data_root_dir data_libero_rlds \
    --vla_path "$VLA_PATH" \
    --vggt_path "$VGGT_PATH" \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "runs_profiling_2048" \
    --num_profiling_samples 2048 \
    --batch_size 16 \
    --vla_layers_align "$VLA_LAYERS_TO_PROFILE" \
    --vggt_layers_align "$VGGT_LAYERS_TO_PROFILE"
