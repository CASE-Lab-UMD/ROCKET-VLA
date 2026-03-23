#!/bin/bash
# ============================================================================
# Layer Profiling Script (CKA Similarity between OpenVLA and VGGT)
# ============================================================================
#
# Computes pairwise CKA / cosine similarity between all VLA layers (0-32)
# and VGGT layers (0-23). Outputs a JSON file with the similarity matrix,
# which can be analyzed with `analyze_profiling_results.py`.
#
# Usage:
#   bash scripts/profile_layers.sh
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
RUN_ROOT_DIR="profiling_results"

# ---- Profiling Configuration ----
#   All VLA layers: 0,1,2,...,32 (32 transformer layers + 1 projector)
#   All VGGT layers: 0,1,2,...,23
VLA_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"
VGGT_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
NUM_PROFILING_SAMPLES=2048
BATCH_SIZE=16

# ============================================================================

torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} \
  vla-scripts/profiling_cka.py \
  --vla_path ${VLA_PATH} \
  --vggt_path ${VGGT_PATH} \
  --data_root_dir ${DATA_ROOT_DIR} \
  --dataset_name ${DATASET_NAME} \
  --run_root_dir ${RUN_ROOT_DIR} \
  --vla_layers_align "${VLA_LAYERS}" \
  --vggt_layers_align "${VGGT_LAYERS}" \
  --num_profiling_samples ${NUM_PROFILING_SAMPLES} \
  --batch_size ${BATCH_SIZE} \
  --use_l1_regression True \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_vlm_norm True \
  --use_vggt_pe True

echo ""
echo "Profiling complete. Analyze results with:"
echo "  python vla-scripts/analyze_profiling_results.py profiling_results/<run_dir>/profiling_results.json"
