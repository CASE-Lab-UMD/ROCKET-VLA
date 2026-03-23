#!/bin/bash
# ============================================================================
# ROCKET Training Script (OpenVLA + VGGT Multi-layer Alignment)
# ============================================================================
#
# This script fine-tunes OpenVLA-7B with ROCKET multi-layer alignment.
# By configuring the parameters below, you can run:
#   - Full ROCKET (shared projector + Matryoshka sparse activation)
#   - Baseline (no alignment, set ALIGN_LOSS_COEFF=0)
#   - Single-layer alignment (set one layer index in VLA/VGGT_LAYERS_ALIGN)
#   - Naive multi-layer (set SHARE_PROJECTOR=False)
#   - Shared-only ablation (set SHARE_PROJECTOR=True, WIDTH_RATIOS="None")
#
# Usage:
#   bash scripts/train_rocket.sh
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
RUN_ROOT_DIR="ckpts/training_results"
RUN_ID="rocket-libero-mix4"

# ---- ROCKET Alignment Configuration ----
#   N=10 aligned layer pairs by default (see paper Sec. 6.1)
#   Layer indices: comma-separated, -1 = last layer (VLA layer 33, VGGT layer 23)
#   Default E2M-Last1 strategy for OpenVLA (32 transformer layers + 1 projector = 33 total):
#     VLA:  2,4,6,8,10,12,14,16,18,-1  (early-to-middle uniform + last)
#     VGGT: 2,4,6,8,10,12,14,16,18,-1  (corresponding VGGT layers)
VLA_LAYERS_ALIGN="2,4,6,8,10,12,14,16,18,-1"
VGGT_LAYERS_ALIGN="2,4,6,8,10,12,14,16,18,-1"
ALIGN_LOSS_TYPE="cosine"
ALIGN_LOSS_COEFF=0.5            # lambda in Eq. 16

# ---- Shared Projector + Matryoshka ----
SHARE_PROJECTOR=True
ENSEMBLE_SIZE=1
#   Width ratios: "None" = auto Matryoshka schedule (shallow→deep: 0.2→1.0)
#   Or specify manually, e.g., "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,1.0"
WIDTH_RATIOS="None"
SHALLOW_TO_DEEP_INCREASE=True

# ---- Training Hyperparameters ----
BATCH_SIZE=8              # per GPU, ~74GB VRAM; use 1 for ~30GB
                          # total batch size = BATCH_SIZE * NUM_GPUS (paper uses 32)
LEARNING_RATE=5e-4        # decays to 5e-5 after NUM_STEPS_BEFORE_DECAY (MultiStepLR)
MAX_STEPS=50005
NUM_STEPS_BEFORE_DECAY=10000
SAVE_FREQ=10000

# ---- LoRA ----
USE_LORA=True
LORA_RANK=32

# ---- Input Configuration ----
NUM_IMAGES=2              # number of camera views
USE_PROPRIO=True          # include proprioceptive state
USE_VLM_NORM=True
USE_VGGT_PE=True
POOLING_FUNC="bilinear"

# ---- Logging ----
WANDB_ENTITY="YOUR_WANDB_ENTITY"
WANDB_PROJECT="YOUR_WANDB_PROJECT"

# ============================================================================

torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} \
  vla-scripts/finetune_rocket.py \
  --vla_path ${VLA_PATH} \
  --vggt_path ${VGGT_PATH} \
  --data_root_dir ${DATA_ROOT_DIR} \
  --dataset_name ${DATASET_NAME} \
  --run_root_dir ${RUN_ROOT_DIR} \
  --run_id_override ${RUN_ID} \
  --vla_layers_align "${VLA_LAYERS_ALIGN}" \
  --vggt_layers_align "${VGGT_LAYERS_ALIGN}" \
  --align_loss_type ${ALIGN_LOSS_TYPE} \
  --align_loss_coeff ${ALIGN_LOSS_COEFF} \
  --share_projector ${SHARE_PROJECTOR} \
  --ensemble_size ${ENSEMBLE_SIZE} \
  --projector_width_ratios "${WIDTH_RATIOS}" \
  --projector_shallow_to_deep_increase ${SHALLOW_TO_DEEP_INCREASE} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --max_steps ${MAX_STEPS} \
  --num_steps_before_decay ${NUM_STEPS_BEFORE_DECAY} \
  --save_freq ${SAVE_FREQ} \
  --use_lora ${USE_LORA} \
  --lora_rank ${LORA_RANK} \
  --num_images_in_input ${NUM_IMAGES} \
  --use_proprio ${USE_PROPRIO} \
  --use_vlm_norm ${USE_VLM_NORM} \
  --use_vggt_pe ${USE_VGGT_PE} \
  --pooling_func ${POOLING_FUNC} \
  --use_l1_regression True \
  --merge_lora_during_training True \
  --image_aug True \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}"
