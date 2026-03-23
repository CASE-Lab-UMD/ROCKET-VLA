python vla-scripts/merge_lora_weights_and_save.py \
    --base_checkpoint ckpts/openvla-7b \
    --lora_finetuned_checkpoint_dir ckpts/training_results_0103/openvla-sf-align-my10-tw-v12-bsz32--50000_lora


# python vla-scripts/compare_model_weights.py \
#     --model_dir1 ckpts/training_results_0103/openvla-sf-align-my10-tw-v12-bsz32--50000_lora \
#     --model_dir2 ckpts/training_results_0103/openvla-sf-align-my10-tw-v12-bsz32--50000_chkpt \
#     --verbose