#!/bin/bash
# ============================================================================
# Gradient Cosine Similarity Analysis
# ============================================================================
#
# Analyzes gradient coherence between alignment losses at different layers.
# This reproduces the gradient cosine similarity visualizations (Fig. 1, 13, 14).
#
# Usage:
#   bash scripts/gradient_analysis.sh
#
# ============================================================================

# ---- GPU Configuration ----
NUM_GPUS=4

# ---- Model Paths ----
VLA_PATH="ckpts/openvla-7b"
VGGT_PATH="ckpts/VGGT-1B/model.pt"

# ---- Dataset ----
DATA_ROOT_DIR="data/libero"
DATASET_NAME="libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops,libero_10_no_noops"

# ---- Output ----
RUN_ROOT_DIR="ckpts/gradient_analysis"

# ---- Layer Alignment ----
VLA_LAYERS_ALIGN="2,4,6,8,10,12,14,16,18,-1"
VGGT_LAYERS_ALIGN="2,4,6,8,10,12,14,16,18,-1"

# ---- Gradient Filtering ----
USE_GRADIENT_FILTERING=True
GRADIENT_FILTERING_START_STEP=2000
GRADIENT_FILTERING_THRESHOLD=0.0

# ============================================================================

torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} \
  vla-scripts/profiling_gradient_filtering.py \
  --vla_path ${VLA_PATH} \
  --vggt_path ${VGGT_PATH} \
  --data_root_dir ${DATA_ROOT_DIR} \
  --dataset_name ${DATASET_NAME} \
  --run_root_dir ${RUN_ROOT_DIR} \
  --vla_layers_align "${VLA_LAYERS_ALIGN}" \
  --vggt_layers_align "${VGGT_LAYERS_ALIGN}" \
  --align_loss_coeff 0.5 \
  --share_projector True \
  --batch_size 8 \
  --max_steps 50005 \
  --use_gradient_filtering ${USE_GRADIENT_FILTERING} \
  --gradient_filtering_start_step ${GRADIENT_FILTERING_START_STEP} \
  --gradient_filtering_threshold ${GRADIENT_FILTERING_THRESHOLD} \
  --use_l1_regression True \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_vlm_norm True \
  --use_vggt_pe True
