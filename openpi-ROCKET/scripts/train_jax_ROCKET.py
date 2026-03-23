"""JAX training script with Spatial Forcing (multi-layer alignment) support.

Only VGGT runs in PyTorch. Everything else (VLA model, projectors, alignment loss) stays in JAX.

Usage:
  python scripts/train_jax_ROCKET.py <config_name> --exp_name <run_name>
"""

import dataclasses
import functools
import logging
import os
import platform
from contextlib import nullcontext
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config
from openpi.models.pi0_align import Pi0Align, TOKENS_PER_IMAGE
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_images_from_openpi
from vggt.heads.utils import custom_pooling


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def compute_vggt_features(
    vggt_model, observation_images, vggt_layers_align, pooling_func, use_vggt_pe, vggt_device, img_len,
    vggt_stream=None,
):
    """Run VGGT inference in PyTorch and return pooled features as numpy arrays.

    Args:
        observation_images: dict of image arrays {name: (B, C, H, W)} as numpy arrays
        vggt_layers_align: list of VGGT layer indices
        pooling_func: interpolation mode (e.g., 'bilinear')
        use_vggt_pe: whether to use positional embeddings
        vggt_device: torch device for VGGT
        img_len: total number of image tokens in VLA (num_images * tokens_per_image)
        vggt_stream: optional CUDA stream for async execution

    Returns:
        numpy array of shape (num_align_layers, B, img_len, 2*vggt_dim) in bfloat16
    """
    stream_ctx = torch.cuda.stream(vggt_stream) if vggt_stream is not None else nullcontext()

    with stream_ctx:
        # Convert images from (B, H, W, C) to (B, C, H, W) bf16 tensors on GPU
        image_list = [
            torch.from_numpy(np.ascontiguousarray(img)).permute(0, 3, 1, 2).to(vggt_device, dtype=torch.bfloat16, non_blocking=True)
            for img in observation_images.values()
        ]

        # Preprocess for VGGT (resize to 518x518)
        images_vggt = preprocess_images_from_openpi(image_list).contiguous()  # (B, N, C, 518, 518)

        # Run VGGT in bf16 (autocast ensures LayerNorm/softmax/pos_embed stay in bf16)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            vggt_output = vggt_model(images_vggt)

        patch_start_idx = vggt_output["patch_start_idx"]
        original_img = vggt_output["images"]
        H, W = original_img.shape[-2:]
        patch_h, patch_w = H // vggt_model.patch_size, W // vggt_model.patch_size

        # Dummy reference for computing resample ratio (only shape[1] is used)
        reference_dummy = torch.zeros(1, img_len, 1, device=vggt_device, dtype=torch.bfloat16)

        # Pool features for each alignment layer — stay in bf16
        pooled_features = []
        for vggt_layer_idx in vggt_layers_align:
            feats = vggt_output["features"][vggt_layer_idx]
            feats = feats[:, :, patch_start_idx:, :]
            pooled = custom_pooling(feats, (patch_h, patch_w), (H, W), reference_dummy, pooling_func, use_vggt_pe)
            pooled_features.append(pooled)

        # Stack on GPU, single transfer to CPU
        stacked = torch.stack(pooled_features, dim=0)  # (num_layers, B, img_len, dim)

    # Sync stream before CPU transfer
    if vggt_stream is not None:
        vggt_stream.synchronize()

    return stacked.cpu().numpy()


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)

        # Create Pi0Align model (with alignment projectors)
        model = Pi0Align(config.model, config, rngs=nnx.Rngs(model_rng))

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    pooled_vggt_features,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions, vggt_features):
        action_loss, align_loss, details = model.compute_loss_with_alignment(
            rng, observation, actions, vggt_features, train=True
        )
        total_loss = action_loss + config.align_loss_coeff * align_loss
        return total_loss, {"action_loss": action_loss, "align_loss": align_loss}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    # Convert stacked vggt features to list for model
    vggt_list = [pooled_vggt_features[i] for i in range(pooled_vggt_features.shape[0])]

    (total_loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions, vggt_list
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": total_loss,
        "action_loss": aux["action_loss"],
        "align_loss": aux["align_loss"],
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # ==================== Initialize VGGT (PyTorch, bf16) ====================
    vggt_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Enable TF32 for any remaining float32 matmuls (faster on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    vggt_model = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
        feature_only=True,
    ).to(vggt_device)

    if config.vggt_weight_path is not None:
        vggt_path = os.path.join(config.vggt_weight_path, "model.pt")
        if not os.path.exists(vggt_path):
            raise FileNotFoundError(f"VGGT weight file not found at {vggt_path}")
        vggt_model.load_state_dict(torch.load(vggt_path, weights_only=False), strict=False)
        logging.info(f"Loaded VGGT weights from {config.vggt_weight_path}")

    # Cast entire model to bf16 — avoids autocast overhead, halves memory
    vggt_model = vggt_model.to(dtype=torch.bfloat16)
    vggt_model.eval()

    # torch.compile for graph optimization (default mode; reduce-overhead uses CUDA Graphs
    # which conflicts with VGGT's internal position caching in RoPE layers)
    try:
        vggt_model = torch.compile(vggt_model)
        logging.info("VGGT compiled with torch.compile (default mode)")
    except Exception as e:
        logging.warning(f"torch.compile failed, falling back to eager mode: {e}")

    # Create dedicated CUDA stream for VGGT inference (overlaps with JAX compute)
    vggt_stream = torch.cuda.Stream(device=vggt_device) if torch.cuda.is_available() else None

    # Parse alignment config
    vla_layers_align = [int(x) for x in config.vla_layers_align.split(",")] if config.vla_layers_align else [-1]
    vggt_layers_align = [int(x) for x in config.vggt_layers_align.split(",")] if config.vggt_layers_align else [-1]
    pooling_func = config.pooling_func or "bilinear"
    use_vggt_pe = config.use_vggt_pe or False
    num_images = len(batch[0].images)
    img_len = num_images * TOKENS_PER_IMAGE

    logging.info(f"Alignment config: vla_layers={vla_layers_align}, vggt_layers={vggt_layers_align}")
    logging.info(f"  pooling={pooling_func}, use_vggt_pe={use_vggt_pe}, img_len={img_len}")
    logging.info(f"  align_loss_coeff={config.align_loss_coeff}")

    # ==================== Model Layer Information ====================
    logging.info("=" * 20 + " Model Layer Information " + "=" * 20)
    try:
        if hasattr(vggt_model, 'aggregator') and hasattr(vggt_model.aggregator, 'layers'):
            vggt_num_layers = len(vggt_model.aggregator.layers)
            logging.info(f"VGGT (Teacher) has {vggt_num_layers} aggregator layers (indices 0-{vggt_num_layers - 1})")
        elif hasattr(vggt_model, 'blocks'):
            vggt_num_layers = len(vggt_model.blocks)
            logging.info(f"VGGT (Teacher) has {vggt_num_layers} blocks (indices 0-{vggt_num_layers - 1})")
        else:
            logging.info("VGGT (Teacher) layer count could not be automatically determined.")
    except AttributeError:
        logging.info("VGGT (Teacher) layer count could not be automatically determined.")

    paligemma_config = openpi.models.pi0_config.Pi0Config()
    import openpi.models.gemma as _gemma
    gemma_cfg = _gemma.get_config(config.model.paligemma_variant if hasattr(config.model, 'paligemma_variant') else 'gemma_2b')
    logging.info(f"VLA (Student) has {gemma_cfg.depth} layers (indices 0-{gemma_cfg.depth - 1}), width={gemma_cfg.width}")
    logging.info("=" * 65)

    # ==================== Initialize JAX train state (with Pi0Align) ====================
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # ---- Prefetch: compute VGGT features for the first batch ----
    observation, actions = batch
    images_host = {name: np.asarray(img) for name, img in jax.device_get(observation.images).items()}
    pooled_vggt_np = compute_vggt_features(
        vggt_model, images_host, vggt_layers_align, pooling_func, use_vggt_pe, vggt_device, img_len,
        vggt_stream=vggt_stream,
    )

    infos = []
    for step in pbar:
        # VGGT features for this step are already computed (prefetched)
        pooled_vggt_jax = jnp.array(pooled_vggt_np)  # (num_layers, B, img_len, dim)

        # ---- JAX train step (runs while next VGGT prefetch happens below) ----
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch, pooled_vggt_jax)

        # ---- Prefetch next batch + VGGT features (overlaps with JAX compute) ----
        batch = next(data_iter)
        observation, actions = batch
        images_host = {name: np.asarray(img) for name, img in jax.device_get(observation.images).items()}
        pooled_vggt_np = compute_vggt_features(
            vggt_model, images_host, vggt_layers_align, pooling_func, use_vggt_pe, vggt_device, img_len,
            vggt_stream=vggt_stream,
        )

        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
