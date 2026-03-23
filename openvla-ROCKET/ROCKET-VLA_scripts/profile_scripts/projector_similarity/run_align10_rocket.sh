# sleep 4.8h
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=1

# Base path for checkpoints
BASE_CKPT_DIR="ckpts/training_results_0103"
RUN_NAME="YOUR_ROCKET_RUN_NAME"

# Output directory for results
OUTPUT_DIR="profiling_results/projector_similarity"
mkdir -p "$OUTPUT_DIR"

# Find all checkpoint directories matching the pattern
CKPT_DIRS=$(ls -d ${BASE_CKPT_DIR}/${RUN_NAME}--*_chkpt | sort -V)

for CHECKPOINT_DIR in $CKPT_DIRS; do
    # Extract step number from checkpoint path (e.g. 10000 from ...--10000_chkpt)
    STEP_NUM=$(echo $CHECKPOINT_DIR | grep -oP '(?<=--)\d+(?=_chkpt)')
    
    echo "Processing Checkpoint Step: $STEP_NUM"
    echo "Checkpoint Path: $CHECKPOINT_DIR"
    
    OUTPUT_JSON="${OUTPUT_DIR}/profiling_results_v12_${STEP_NUM}.json"

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
      --ensemble_size 1 \
      --projector_width_ratios "None" \
      --projector_shallow_to_deep_increase True \
      --use_vggt_cache False \
      --generate_vggt_cache False \
      --vggt_cache_dir None \
      --wandb_entity "YOUR_WANDB_ENTITY" \
      --wandb_project "YOUR_WANDB_PROJECT" \
      --run_id_override "profiling_rocket_${STEP_NUM}"
      
    echo "Finished profiling step $STEP_NUM. Results saved to $OUTPUT_JSON"
    echo "----------------------------------------"
done
