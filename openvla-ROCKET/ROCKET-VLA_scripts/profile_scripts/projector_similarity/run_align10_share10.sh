# sleep 4.8h
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=3

# Base path for checkpoints
BASE_CKPT_DIR="ckpts/motivation/shared"
RUN_NAME="YOUR_SHARED_RUN_NAME"

# Output directory for results
OUTPUT_DIR="profiling_results/projector_similarity"
mkdir -p "$OUTPUT_DIR"

# Manually specify the checkpoints you want to profile here
# Example: STEPS=(10000 30000 50000)
STEPS=(10000 30000 50000)

for STEP in "${STEPS[@]}"; do
    CHECKPOINT_DIR="${BASE_CKPT_DIR}/${RUN_NAME}--${STEP}_chkpt"
    
    # Check if checkpoint exists
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "Warning: Checkpoint not found at $CHECKPOINT_DIR, skipping..."
        continue
    fi
    
    echo "Processing Checkpoint Step: $STEP"
    echo "Checkpoint Path: $CHECKPOINT_DIR"
    
    OUTPUT_JSON="${OUTPUT_DIR}/profiling_results_share10_${STEP}.json"

    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/profiling_projector_similarity.py \
      --vla_path ckpts/openvla-7b \
      --vggt_path ckpts/VGGT-1B/model.pt \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --output_json_path "$OUTPUT_JSON" \
      --num_profiling_samples 128 \
      --data_root_dir data_libero_rlds \
      --dataset_name libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops,libero_10_no_noops \
      --run_root_dir ckpts/training_results_0103/ \
      --pooling_func bilinear \
      --vla_layers_align "2,4,6,8,10,12,14,16,18,32" \
      --vggt_layers_align "2,4,6,8,10,12,14,16,18,23" \
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
      --image_aug True \
      --lora_rank 32 \
      --share_projector True \
      --use_naive_projector True \
      --ensemble_size 1 \
      --projector_width_ratios "None" \
      --projector_shallow_to_deep_increase True \
      --use_vggt_cache False \
      --generate_vggt_cache False \
      --vggt_cache_dir None \
      --wandb_entity "YOUR_WANDB_ENTITY" \
      --wandb_project "YOUR_WANDB_PROJECT" \
      --run_id_override "profiling_naive_${STEP}"
      
    echo "Finished profiling step $STEP. Results saved to $OUTPUT_JSON"
    echo "----------------------------------------"
done
