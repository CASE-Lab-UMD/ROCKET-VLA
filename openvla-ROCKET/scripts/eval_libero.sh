#!/bin/bash
# ============================================================================
# LIBERO Evaluation Script
# ============================================================================
#
# Evaluates a fine-tuned ROCKET checkpoint on LIBERO task suites.
#
# Usage:
#   bash scripts/eval_libero.sh
#
# ============================================================================

# ---- Checkpoint Path ----
CKPT_DIR="ckpts/training_results/rocket-libero-mix4"

# ---- Task Suite ----
#   Options: libero_spatial, libero_object, libero_goal, libero_10
TASK_SUITE="libero_spatial"

# ---- Evaluation Settings ----
NUM_TRIALS=50             # trials per task
SEED=42

# ============================================================================

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint ${CKPT_DIR} \
  --task_suite_name ${TASK_SUITE} \
  --num_trials_per_task ${NUM_TRIALS} \
  --seed ${SEED} \
  --center_crop True
