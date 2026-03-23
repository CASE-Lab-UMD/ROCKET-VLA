# ROCKET on Real-World Robots (openpi-ROCKET)

> PI0 / PI0.5 backbone with ROCKET multi-layer alignment for real-world deployment and RoboTwin 2.0.
> See the [main README](../README.md) for full documentation.

## 1. Environment Setup

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies:

```bash
cd openpi-ROCKET

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
source .venv/bin/activate
```

> `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

## 2. Data Preparation

Collect task-specific raw data with your robot, then convert to LeRobot format:

```bash
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py \
  --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
# Converted data is stored in ~/.cache/huggingface/lerobot/<org>/<dataset-name>/
```

## 3. Model Preparation

```bash
# Convert PI0 JAX checkpoint to PyTorch (needed for PI0.5 PyTorch training)
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir gs://openpi-assets/checkpoints/pi0_base \
  --config_name <config_name> \
  --output_path ./checkpoints/pi0.5_base

# Download VGGT-1B (not needed for baseline configs)
mkdir -p checkpoints/vggt/VGGT-1B
# https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt -> checkpoints/vggt/VGGT-1B/model.pt
```

Expected directory structure:

```
openpi-ROCKET/
├── checkpoints/
│   ├── pi0.5_base/             # PyTorch-converted PI0.5 weights
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── ...
│   └── vggt/
│       └── VGGT-1B/
│           └── model.pt
```

## 4. Training

Compute normalization statistics first:

```bash
uv run scripts/compute_norm_stats.py --config-name <config_name>
```

### Available Configs

All configs are defined in [src/openpi/training/config.py](src/openpi/training/config.py).

**LIBERO (PI0.5, full fine-tuning):**

| Config Name | Method | Training Script | Key Differences |
|-------------|--------|-----------------|-----------------|
| `pi05_libero_align10_rocket_64bsz` | **ROCKET** | `train_pytorch_ROCKET.py` | 10 layer pairs, shared projector |
| `pi05_libero_align1_spatial_forcing_64bsz` | Spatial Forcing | `train_pytorch_ROCKET.py` | Single layer (VLA=12, VGGT=-1) |
| `pi05_libero_align0_baseline_64bsz` | Baseline | `train_pytorch.py` | No VGGT, no alignment |

```bash
# ROCKET on LIBERO (PI0.5, PyTorch, multi-GPU)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch_ROCKET.py pi05_libero_align10_rocket_64bsz \
  --exp_name rocket_libero

# Spatial Forcing on LIBERO
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch_ROCKET.py pi05_libero_align1_spatial_forcing_64bsz \
  --exp_name sf_libero

# Baseline on LIBERO (no VGGT needed — uses standard train_pytorch.py)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch.py pi05_libero_align0_baseline_64bsz \
  --exp_name baseline_libero

# Resume from latest checkpoint
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch_ROCKET.py pi05_libero_align10_rocket_64bsz \
  --exp_name rocket_libero --resume
```

**RoboTwin 2.0 (PI0, LoRA, single-task):**

| Config Name | Method | Training Script |
|-------------|--------|-----------------|
| `pi0_base_aloha_robotwin_lora_rocket_MPA` | **ROCKET** | `train_jax_ROCKET.py` |
| `pi0_base_aloha_robotwin_lora_spatial_forcing_MPA` | Spatial Forcing | `train_jax_ROCKET.py` |
| `pi0_base_aloha_robotwin_lora_baseline_MPA` | Baseline | `train.py` |

```bash
# ROCKET on RoboTwin (PI0, JAX)
uv run scripts/train_jax_ROCKET.py pi0_base_aloha_robotwin_lora_rocket_MPA \
  --exp_name rocket_robotwin

# Spatial Forcing on RoboTwin
uv run scripts/train_jax_ROCKET.py pi0_base_aloha_robotwin_lora_spatial_forcing_MPA \
  --exp_name sf_robotwin

# Baseline on RoboTwin (no VGGT needed — uses standard train.py)
uv run scripts/train.py pi0_base_aloha_robotwin_lora_baseline_MPA \
  --exp_name baseline_robotwin
```

> These RoboTwin configs use a single-task dataset (`move_playingcard_away`). To train on your own task, duplicate the config and change `repo_id` to your LeRobot dataset.

**Real-Robot (PI0.5, full fine-tuning, JAX):**

| Config Name | Method | Training Script |
|-------------|--------|-----------------|
| `pi05_0312_250_3_15_74_resize_chunksize10_rocket` | **ROCKET** | `train_jax_ROCKET.py` |
| `pi05_0312_250_3_15_74_resize_chunksize10_spatial_forcing` | Spatial Forcing | `train_jax_ROCKET.py` |
| `pi05_0312_250_3_15_74_resize_chunksize10_baseline` | Baseline | `train.py` |

```bash
# ROCKET on real-robot task (PI0.5, JAX)
uv run scripts/train_jax_ROCKET.py pi05_0312_250_3_15_74_resize_chunksize10_rocket \
  --exp_name rocket_real
```

> These configs reference a local dataset (`local/pick_packages_...`). Replace `repo_id` with your own dataset in config.py.

### ROCKET-Specific Parameters

| Parameter | ROCKET | Spatial Forcing | Baseline |
|-----------|--------|-----------------|----------|
| `vla_layers_align` | `"2,3,4,5,6,7,8,9,10,-1"` | `"12"` | *(not set)* |
| `vggt_layers_align` | `"2,4,6,8,10,12,14,16,18,-1"` | `"-1"` | *(not set)* |
| `share_projector` | `True` | `False` | *(not set)* |
| `align_loss_coeff` | `0.5` (LIBERO) / `0.125` (RoboTwin) | `0.5` | *(not set)* |

## 5. Inference

Launch a model server, then run your robot client:

```bash
# Start model server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<config_name> \
  --policy.dir=checkpoints/<config_name>/<exp_name>/<step>

# Run client (example for ALOHA)
uv run examples/simple_client/main.py --env ALOHA
```

## Notes

- **Baseline configs** do not set any VGGT/alignment parameters, so they use the standard openpi training scripts (`train_pytorch.py` / `train.py`) without VGGT overhead.
- **RoboTwin (single-task):** Uses `align_loss_coeff=0.125` instead of `0.5`, as the action loss converges faster in single-task settings (see paper Appendix A).
- **PI0.5 on LIBERO:** Uses batch size 64, full fine-tuning, 30,000 steps with cosine LR schedule (peak 2.5e-5).
- **VGGT weight path:** All ROCKET/SF configs reference `checkpoints/vggt/VGGT-1B`. Update in [config.py](src/openpi/training/config.py) if your path differs.
