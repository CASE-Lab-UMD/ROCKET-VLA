
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=4,5

# Run layer importance measurement
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/profiling_layer_importance.py \
  --vla_path ckpts/openvla-7b \
  --data_root_dir data_libero_rlds \
  --dataset_name libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops,libero_10_no_noops \
  --run_root_dir ckpts/training_results_libero_plus_1231/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 50005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_override "YOUR_RUN_ID" \
  --measure_importance True \
  --importance_save_file "profiling_results/libero_importance.pt" \
  --num_calibration_batches 128
