"""
profile_similarity.py

Profiles OpenVLA via LoRA.
"""

import os
import time
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors_rocket import (
    NoisyActionProjector,
    ProprioProjector,
    AlignProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.util.pooling_utils import custom_pooling
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES
import random

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_normed_images

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    vggt_path: str = 'official_ckpts/vggt_model.pt'  # Path to VGGT model (on HuggingFace Hub or stored locally)
    
    # Checkpoint Paths (New)
    checkpoint_dir: str = "checkpoints/latest"       # Path to the checkpoint directory containing lora_adapter and align_projectors
    
    # Profiling Config (New)
    num_profiling_samples: int = 100                 # Number of samples to profile
    output_json_path: str = "profiling_results.json" # Path to save profiling results
    layer_offset: int = -1                            # Offset to add to VLA and VGGT layer indices for OOD testing

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)
    seed: int = 42                                   # Random seed for reproducibility

    # Algorithm and architecture
    align_loss_type: str = "cosine"                  # Loss function for alignment loss, "cosine"
    align_loss_coeff: float = 0.5                    # Coefficient for alignment loss (multiplied by align_loss)
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    pooling_func: str = "bilinear"                   # resize VGGT state pixels to vla pixels
    vla_layers_align: str = "-1"                     # Comma-separated string of VLA hidden state layers to align
    vggt_layers_align: str = "-1"                    # Comma-separated string of VGGT hidden state layers to align
    use_vlm_norm: bool = False                       # whether to use VLM normalization for the VLM output vision embeddings
    use_vggt_pe: bool = False                        # position embedding for vggt state before pooling
    gain_feat_1move: bool = True                     # whether to gain the VLA feature moved one position backward

    share_projector: bool = False
    ensemble_size: int = 1
    
    # Projector Dynamic Sub-network Config
    projector_width_ratios: str = "None"             # Comma-separated ratios, e.g., "0.25,0.5,0.75,1.0". If "None", uses default linear.
    projector_shallow_to_deep_increase: bool = True  # True: shallow layers use fewer params.
    use_naive_projector: bool = False                # If True, use the naive AlignProjector (non-dynamic)

    # VGGT Feature Cache Config
    use_vggt_cache: bool = False                     # If True, load VGGT features from cache if available
    generate_vggt_cache: bool = False                # If True, save computed VGGT features to cache
    vggt_cache_dir: str = "vggt_features_cache"      # Directory to store cached features

    # Training configuration (Ignored for profiling but kept for compatibility)
    batch_size: int = 8                              
    learning_rate: float = 5e-4                      
    lr_warmup_steps: int = 0                         
    num_steps_before_decay: int = 100_000            
    grad_accumulation_steps: int = 1                 
    max_steps: int = 200_000                         
    use_val_set: bool = False                        
    val_freq: int = 10_000                           
    val_time_limit: int = 180                        
    save_freq: int = 10_000                          
    save_latest_checkpoint_only: bool = False        
    scheduler: str = 'MultiStepLR'                   
    resume: bool = False                             
    resume_step: Optional[int] = None                
    image_aug: bool = False                          # Default to False for profiling
    diffusion_sample_freq: int = 50                  

    # LoRA
    use_lora: bool = True                            
    lora_rank: int = 32                              
    lora_dropout: float = 0.0                        
    merge_lora_during_training: bool = True          
    
    # Logging
    wandb_entity: str = "your-wandb-entity"          
    wandb_project: str = "your-wandb-project"        
    run_id_note: Optional[str] = None                
    run_id_override: Optional[str] = None            
    wandb_log_freq: int = 10                         
    # fmt: on


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def get_image_hash_keys(pixel_values):
    import hashlib
    imgs = pixel_values.detach().cpu().to(torch.float32).numpy()
    keys = []
    for img in imgs:
        keys.append(hashlib.md5(img.tobytes()).hexdigest())
    return keys


@draccus.wrap()
def profile(cfg: FinetuneConfig) -> None:
    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Parse comma-separated layer alignment strings
    cfg.vla_layers_align = [int(x) for x in cfg.vla_layers_align.split(',')]
    cfg.vggt_layers_align = [int(x) for x in cfg.vggt_layers_align.split(',')]

    print(f"Profiling OpenVLA Model based on `{cfg.vla_path}`")

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Load processor and VLA Base
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Resolve -1 in vla_layers_align
    if hasattr(vla.config, "text_config") and hasattr(vla.config.text_config, "num_hidden_layers"):
        total_layers = vla.config.text_config.num_hidden_layers
    elif hasattr(vla.config, "num_hidden_layers"):
        total_layers = vla.config.num_hidden_layers
    else:
        total_layers = 32
    cfg.vla_layers_align = [(l if l != -1 else total_layers - 1) for l in cfg.vla_layers_align]
    
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # Load LoRA Adapter
    adapter_path = Path(cfg.checkpoint_dir) / "lora_adapter"
    if adapter_path.exists():
        print(f"Loading LoRA adapter from {adapter_path}")
        vla = PeftModel.from_pretrained(vla, adapter_path)
    else:
        print(f"Warning: LoRA adapter not found at {adapter_path}. Using base model.")

    # FiLM setup (if used in training)
    if cfg.use_film:
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        # Load film weights if they are separate or part of base
        # Usually they are part of base if merged, or separate.
        # Assuming they are handled by loading mechanism if relevant.

    # Load VGGT
    vggt_model = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
        feature_only=True,
    )
    vggt_model.load_state_dict(torch.load(cfg.vggt_path), strict=False)
    vggt_model = vggt_model.to(device_id)
    vggt_model.eval()

    # Initialize Projectors
    if cfg.projector_width_ratios == "None":
        width_ratios = None
    else:
        width_ratios = [float(x) for x in cfg.projector_width_ratios.split(',')]
    
    num_align_layers = len(cfg.vla_layers_align)
    
    # Force ensemble_size to be at least 1, but we might have loaded more from checkpoint
    # Actually, we need to know the structure to load correctly.
    # Assuming the user provided correct ensemble_size in args (e.g. 1) but checkpoint has 1.
    # Wait, if ensemble_size is 1, then input/output similarity IS 1.0 because there's only 1 item.
    
    if cfg.share_projector:
        if cfg.use_naive_projector:
            from prismatic.models.projectors_rocket import AlignProjector as NaiveAlignProjector
            # For naive projector, we don't need num_layers or width_ratios logic.
            # But we are sharing it, so we replicate the same instance.
            single_projector = NaiveAlignProjector(
                llm_dim=vla.module.llm_dim if hasattr(vla, "module") else vla.llm_dim,
                vggt_dim=vggt_model.embed_dim,
                align_loss_type=cfg.align_loss_type,
                use_vlm_norm=cfg.use_vlm_norm,
            )
            # Replicate the same instance into a ModuleList to mimic the structure, 
            # or just wrap it. But wait, we need an ensemble.
            shared_projectors_pool = nn.ModuleList([single_projector for _ in range(cfg.ensemble_size)])
            
            # Since naive projector doesn't support layer_index arg in align_dimension, 
            # we need to wrapper or handle it in the loop.
            # But AlignProjector in projectors.py doesn't take layer_index.
            # So we can just use it.
            # However, the loop below calls projector.align_dimension(..., layer_index=i).
            # We'll handle this by monkey-patching or wrapping.
            
            # Better approach: Wrap it to ignore layer_index
            class NaiveWrapper(nn.Module):
                def __init__(self, proj):
                    super().__init__()
                    self.proj = proj
                def align_dimension(self, x, layer_index=None):
                    return self.proj.align_dimension(x)
            
            # Re-create pool with wrappers
            shared_projectors_pool = nn.ModuleList([
                NaiveWrapper(
                    NaiveAlignProjector(
                        llm_dim=vla.module.llm_dim if hasattr(vla, "module") else vla.llm_dim,
                        vggt_dim=vggt_model.embed_dim,
                        align_loss_type=cfg.align_loss_type,
                        use_vlm_norm=cfg.use_vlm_norm,
                    )
                ) for _ in range(cfg.ensemble_size)
            ])
        else:
            shared_projectors_pool = nn.ModuleList([
                AlignProjector(
                    llm_dim=vla.module.llm_dim if hasattr(vla, "module") else vla.llm_dim,
                    vggt_dim=vggt_model.embed_dim,
                    align_loss_type=cfg.align_loss_type,
                    use_vlm_norm=cfg.use_vlm_norm,
                    num_layers=num_align_layers,
                    width_ratios=width_ratios,
                    shallow_to_deep_increase=cfg.projector_shallow_to_deep_increase,
                ) for _ in range(cfg.ensemble_size)
            ])
        align_projectors_list = [shared_projectors_pool for _ in cfg.vla_layers_align]
    else:
        if cfg.use_naive_projector:
             from prismatic.models.projectors_rocket import AlignProjector as NaiveAlignProjector
             class NaiveWrapper(nn.Module):
                def __init__(self, proj):
                    super().__init__()
                    self.proj = proj
                def align_dimension(self, x, layer_index=None):
                    return self.proj.align_dimension(x)
                    
             align_projectors_list = [
                nn.ModuleList([
                    NaiveWrapper(
                        NaiveAlignProjector(
                            llm_dim=vla.module.llm_dim if hasattr(vla, "module") else vla.llm_dim,
                            vggt_dim=vggt_model.embed_dim,
                            align_loss_type=cfg.align_loss_type,
                            use_vlm_norm=cfg.use_vlm_norm,
                        )
                    ) for _ in range(cfg.ensemble_size)
                ]) for _ in cfg.vla_layers_align
            ]
        else:
            align_projectors_list = [
                nn.ModuleList([
                    AlignProjector(
                        llm_dim=vla.module.llm_dim if hasattr(vla, "module") else vla.llm_dim,
                        vggt_dim=vggt_model.embed_dim,
                        align_loss_type=cfg.align_loss_type,
                        use_vlm_norm=cfg.use_vlm_norm,
                        num_layers=1,
                    ) for _ in range(cfg.ensemble_size)
                ]) for _ in cfg.vla_layers_align
            ]
    align_projectors = nn.ModuleList(align_projectors_list).to(device_id)
    align_projectors.eval()

    # Load Projector Weights
    # Find projector checkpoint
    checkpoint_dir = Path(cfg.checkpoint_dir)
    # Try finding exact match or use provided name
    # We look for a file starting with align_projectors
    proj_files = list(checkpoint_dir.glob("align_projectors*.pt"))
    if proj_files:
        proj_path = proj_files[0]
        print(f"Loading Align Projectors from {proj_path}")
        state_dict = torch.load(proj_path, map_location=f"cuda:{device_id}")
        
        # Handle DDP prefix removal if necessary
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            clean_state_dict[k] = v
            
        new_state_dict = {}
        if cfg.use_naive_projector:
             # Strategy: Robust mapping for shared projectors
             # We assume share_projector=True (based on context), so we replicate weights.
             # We scan the checkpoint keys for components ('fc1', 'fc2', 'vlm_norm') and replicate them to all model layers.
             
             for k, v in clean_state_dict.items():
                 parts = k.split('.')
                 
                 # Identify component and parameter type
                 comp = None
                 if 'fc1' in parts: comp = 'fc1'
                 elif 'fc2' in parts: comp = 'fc2'
                 elif 'vlm_norm' in parts: comp = 'vlm_norm'
                 
                 param = None
                 if 'weight' in parts: param = 'weight'
                 elif 'bias' in parts: param = 'bias'
                 
                 if comp and param:
                     # Replicate to all layers (l) and ensembles (e)
                     # Target format: "{l}.{e}.proj.{comp}.{param}"
                     for l in range(len(cfg.vla_layers_align)):
                         for e in range(cfg.ensemble_size):
                             target_key = f"{l}.{e}.proj.{comp}.{param}"
                             new_state_dict[target_key] = v
                             
                 # Keep original key to avoid empty dict issues (though unexpected keys will be reported)
                 new_state_dict[k] = v

             # Fallback: Allow partial loading
             missing_keys, unexpected_keys = align_projectors.load_state_dict(new_state_dict, strict=False)
             print(f"Loaded AlignProjectors with use_naive_projector=True (Robust Replication).")
             if missing_keys:
                 print(f"Missing keys (sample): {missing_keys[:5]}")
             # if unexpected_keys:
             #    print(f"Unexpected keys (sample): {unexpected_keys[:5]}")
                 
        else:
            align_projectors.load_state_dict(clean_state_dict)
    else:
        raise ValueError(f"Error: No align_projectors checkpoint found in {checkpoint_dir}. Profiling requires trained weights!")

    # Dataset Setup
    use_wrist_image = cfg.num_images_in_input > 1
    dataset_names = [name.strip() for name in cfg.dataset_name.split(',')]
    if len(dataset_names) > 1:
        mixture_name = "+".join(sorted(dataset_names))
        mixture_spec = [(name, 1.0) for name in dataset_names]
        OXE_NAMED_MIXTURES[mixture_name] = mixture_spec
        cfg.dataset_name = mixture_name

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=1000, # Small buffer for profiling
        image_aug=False,
    )
    
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    # Profiling Loop
    results = []
    vla.eval()
    
    # Proprio Projector (if needed, just load it to avoid errors if used in forward, though we might not use it)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = ProprioProjector(vla.llm_dim, PROPRIO_DIM).to(device_id)
        # Try load if exists
        pp_files = list(checkpoint_dir.glob("proprio_projector*.pt"))
        if pp_files:
             state_dict = torch.load(pp_files[0], map_location=f"cuda:{device_id}")
             # Handle DDP prefix removal if necessary
             new_state_dict = {}
             for k, v in state_dict.items():
                 if k.startswith("module."):
                     new_state_dict[k[7:]] = v
                 else:
                     new_state_dict[k] = v
             proprio_projector.load_state_dict(new_state_dict)
        else:
             raise ValueError(f"Error: No proprio_projector checkpoint found in {checkpoint_dir}, but use_proprio is True!")
        proprio_projector.eval()

    # Noisy Action Projector (if needed)
    noisy_action_projector = None

    print("Starting profiling...")
    processed_samples = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, total=cfg.num_profiling_samples // cfg.batch_size):
            if processed_samples >= cfg.num_profiling_samples:
                break
                
            # Move batch to device
            batch_input_ids = batch["input_ids"].to(device_id)
            batch_attention_mask = batch["attention_mask"].to(device_id)
            batch_pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)
            batch_labels = batch["labels"].to(device_id)
            batch_proprio = batch["proprio"].to(device_id) if "proprio" in batch and batch["proprio"] is not None else None

            # VLA Forward
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values=batch_pixel_values,
                    labels=batch_labels,
                    output_hidden_states=True,
                    proprio=batch_proprio if cfg.use_proprio else None,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                )

            # VGGT Forward
            unnorm_imgs = preprocess_normed_images(
                batch['pixel_values'], processor.image_processor, cfg.num_images_in_input
            ).to(device_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vggt_output = vggt_model(unnorm_imgs)
            
            patch_start_idx = vggt_output["patch_start_idx"]
            original_img = vggt_output["images"]
            H, W = original_img.shape[-2:]
            patch_h, patch_w = H // vggt_model.patch_size, W // vggt_model.patch_size
            
            # Common params
            projected_embeddings = output.projector_features
            vision_length = projected_embeddings.shape[-2] - 1 if cfg.use_proprio else projected_embeddings.shape[-2]
            boi_ids = 2 if cfg.gain_feat_1move else 1
            
            current_batch_size = batch_input_ids.shape[0]

            # Collect features for inter-layer similarity calculation
            # We want to compute similarity between layers (hidden states) BEFORE projector and AFTER projector.
            # But the user asked for "for each sample...".
            # Re-reading task:
            # 1. Before projector: similarity between inputs. 
            #    Wait, "inputs" could mean "ensemble inputs" (which are identical) OR "different layers' inputs".
            #    "pairwise similarity of 10 pre-projector inputs" -> similarity between the 10 LAYERS for a single sample.
            # 3. After projector: similarity between outputs.
            #    "pairwise similarity of 10 post-projector hidden states" -> similarity between the 10 LAYERS' outputs.
            
            for b in range(current_batch_size):
                if processed_samples >= cfg.num_profiling_samples:
                    break
                
                sample_res = {
                    "sample_id": processed_samples,
                }
                
                # Collect features across all layers for this sample
                layer_inputs = []   # List of (N, D) tensors
                layer_outputs = []  # List of (N, D) tensors
                layer_targets = []  # List of (N, D) tensors (optional, for validation)
                
                # First pass: collect all features
                for i, layer_projectors in enumerate(align_projectors):
                    # 1. Inputs (VLA features)
                    vla_hidden_i = output.hidden_states[cfg.vla_layers_align[i] + cfg.layer_offset]
                    vision_hidden_sample = vla_hidden_i[b:b+1, boi_ids : vision_length + boi_ids, :].clone() # (1, N, D_vla)
                    
                    # Normalize input for similarity calc
                    layer_inputs.append(F.normalize(vision_hidden_sample, dim=-1))
                    
                    # 2. Target (VGGT features) - just for checking teacher sim
                    agg_vggt_hidden_i = vggt_output["features"][cfg.vggt_layers_align[i] + cfg.layer_offset]
                    vggt_hidden_sample = agg_vggt_hidden_i[b:b+1, :, patch_start_idx:, :]
                    pooled_vggt_sample = custom_pooling(
                        vggt_hidden_sample, (patch_h, patch_w), (H, W), vision_hidden_sample, cfg.pooling_func, cfg.use_vggt_pe
                    )
                    layer_targets.append(F.normalize(pooled_vggt_sample, dim=-1))

                    # 3. Projector Outputs
                    # Since ensemble_size is likely 1, we take the first one. 
                    # If ensemble > 1, user might want average or specific one. Assuming mean if ensemble > 1?
                    # Or just take the first one since usually ensemble is for uncertainty not diversity in feature space?
                    # Let's take the mean of ensemble outputs for the layer representation.
                    
                    ens_outputs = []
                    for j, projector in enumerate(layer_projectors):
                         with torch.autocast("cuda", dtype=torch.bfloat16):
                            proj_feat = projector.align_dimension(vision_hidden_sample, layer_index=i)
                            ens_outputs.append(proj_feat)
                    
                    # Stack and mean
                    mean_output = torch.stack(ens_outputs).mean(dim=0) # (1, N, D_vggt)
                    layer_outputs.append(F.normalize(mean_output, dim=-1))

                # Task 1: Inter-layer Input Similarity Matrix (10x10)
                num_layers = len(layer_inputs)
                input_sim_matrix = np.zeros((num_layers, num_layers))
                for m in range(num_layers):
                    for n in range(num_layers):
                         # Cosine similarity between layer m and layer n features
                         # Shape: (1, N, D). We calculate mean cosine sim over tokens.
                         # Note: D might differ if VLA layers have different dims? No, VLA layers are same dim.
                         sim = (layer_inputs[m] * layer_inputs[n]).sum(dim=-1).mean().item()
                         input_sim_matrix[m, n] = sim
                sample_res["input_layer_similarity"] = input_sim_matrix.tolist()

                # Task 2: Projector Output vs Teacher Similarity Vector (Length 10)
                teacher_sims = []
                for m in range(num_layers):
                    # We are already normalized
                    # (1, N, D) * (1, N, D) -> sum dim -1 -> (1, N) -> mean -> scalar
                    sim = (layer_outputs[m] * layer_targets[m]).sum(dim=-1).mean().item()
                    teacher_sims.append(sim)
                sample_res["projector_teacher_similarity"] = teacher_sims

                # Task 3: Inter-layer Output Similarity Matrix (10x10)
                output_sim_matrix = np.zeros((num_layers, num_layers))
                for m in range(num_layers):
                    for n in range(num_layers):
                        # Shape: (1, N, D). VGGT dim is same across layers? Yes usually.
                        sim = (layer_outputs[m] * layer_outputs[n]).sum(dim=-1).mean().item()
                        output_sim_matrix[m, n] = sim
                sample_res["output_layer_similarity"] = output_sim_matrix.tolist()

                # Task 4: Inter-layer VGGT Similarity Matrix (10x10)
                vggt_sim_matrix = np.zeros((num_layers, num_layers))
                for m in range(num_layers):
                    for n in range(num_layers):
                        # layer_targets contains normalized VGGT features
                        sim = (layer_targets[m] * layer_targets[n]).sum(dim=-1).mean().item()
                        vggt_sim_matrix[m, n] = sim
                sample_res["vggt_layer_similarity"] = vggt_sim_matrix.tolist()
                
                results.append(sample_res)
                processed_samples += 1
                
    # Compute Average across all samples
    if results:
        avg_res = {
            "input_layer_similarity": np.mean([r["input_layer_similarity"] for r in results], axis=0).tolist(),
            "projector_teacher_similarity": np.mean([r["projector_teacher_similarity"] for r in results], axis=0).tolist(),
            "output_layer_similarity": np.mean([r["output_layer_similarity"] for r in results], axis=0).tolist(),
            "vggt_layer_similarity": np.mean([r["vggt_layer_similarity"] for r in results], axis=0).tolist(),
        }
        
        # Save detailed results
        with open(cfg.output_json_path, 'w') as f:
            json.dump({"average": avg_res, "detailed": results}, f, indent=2)
    
    print(f"Saved profiling results to {cfg.output_json_path}")

if __name__ == "__main__":
    profile()
