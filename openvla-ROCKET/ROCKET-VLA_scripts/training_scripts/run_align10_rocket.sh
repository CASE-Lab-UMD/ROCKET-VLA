# sleep 4.8h
export WANDB_MODE="online"
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_rocket.py \
  --vla_path ckpts/openvla-7b \
  --vggt_path ckpts/VGGT-1B/model.pt \
  --data_root_dir data_libero_rlds \
  --dataset_name libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops,libero_10_no_noops \
  --run_root_dir ckpts/training_results_0103/ \
  --pooling_func bilinear \
  --vla_layers_align "2,4,6,8,10,12,14,16,18,-1" \
  --vggt_layers_align "2,4,6,8,10,12,14,16,18,-1" \
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
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 50005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --image_aug True \
  --lora_rank 32 \
  --share_projector True \
  --ensemble_size 1 \
  --use_matryoshka True \
  --projector_width_ratios "None" \
  --projector_shallow_to_deep_increase True \
  --use_vggt_cache False \
  --generate_vggt_cache False \
  --vggt_cache_dir None \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_override "YOUR_RUN_ID"


    # --batch_size 16 \
