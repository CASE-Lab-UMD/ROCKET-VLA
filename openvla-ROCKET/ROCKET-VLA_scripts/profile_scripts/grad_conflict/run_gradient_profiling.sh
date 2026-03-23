#!/bin/bash
# Gradient conflict profiling: compute cosine similarity between alignment and task loss gradients
# Corresponds to paper Fig. 1, Fig. 13, Fig. 14
#
# This script profiles gradient directions for different loss components at specified training steps,
# outputting a JSON file with cosine similarity matrices.

export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS=4
VLA_PATH="ckpts/openvla-7b"
VGGT_PATH="ckpts/VGGT-1B/model.pt"
DATA_DIR="data/libero"
DATASET_NAME="libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops,libero_10_no_noops"

# ---- Alignment Configuration ----
VLA_LAYERS_ALIGN="2,4,6,8,10,12,14,16,18,-1"
VGGT_LAYERS_ALIGN="2,4,6,8,10,12,14,16,18,-1"

# ---- Profiling Configuration ----
PROFILING_START_STEP=1000         # Step at which to start gradient profiling
PROFILING_NUM_SAMPLES=32          # Number of samples to collect for profiling
PROFILING_OUTPUT="profiling_results/gradient_profiling.json"

torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS \
  vla-scripts/profiling_gradient_filtering.py \
  --vla_path $VLA_PATH \
  --vggt_path $VGGT_PATH \
  --data_root_dir $DATA_DIR \
  --dataset_name $DATASET_NAME \
  --run_root_dir ckpts/profiling_runs/ \
  --pooling_func bilinear \
  --vla_layers_align "$VLA_LAYERS_ALIGN" \
  --vggt_layers_align "$VGGT_LAYERS_ALIGN" \
  --align_loss_type cosine \
  --align_loss_coeff 0.5 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --use_vlm_norm True \
  --use_vggt_pe True \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 2000 \
  --image_aug True \
  --lora_rank 32 \
  --share_projector True \
  --ensemble_size 1 \
  --profiling_start_step $PROFILING_START_STEP \
  --profiling_num_samples $PROFILING_NUM_SAMPLES \
  --profiling_output_file $PROFILING_OUTPUT \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_override "gradient_profiling"
