"""JAX Pi0 model with Spatial Forcing (multi-layer alignment) support.

VGGT runs in PyTorch externally. Everything else (VLA, projectors, alignment loss) is in JAX.
"""

import logging

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask, posemb_sincos
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# SigLIP with 224x224 images and patch_size=14 produces 16x16=256 tokens per image
TOKENS_PER_IMAGE = (224 // 14) ** 2


class AlignProjector(nnx.Module):
    """JAX AlignProjector with dynamic sub-network width for multi-layer alignment.

    Supports shared projector mode where a single projector is used across multiple layers,
    with each layer using a different sub-network width (slicing fc1/fc2 weights).
    """

    def __init__(
        self,
        llm_dim: int,
        vggt_dim: int,
        hidden_dim: int | None = None,
        use_vlm_norm: bool = False,
        num_layers: int = 1,
        width_ratios: list | None = None,
        shallow_to_deep_increase: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 2 * vggt_dim
        self.num_layers = num_layers
        self.shallow_to_deep_increase = shallow_to_deep_increase

        if width_ratios is None:
            self.width_ratios = [0.2 + 0.8 * i / max(num_layers - 1, 1) for i in range(num_layers)]
        else:
            assert len(width_ratios) == num_layers
            self.width_ratios = list(width_ratios)

        if not shallow_to_deep_increase:
            self.width_ratios = self.width_ratios[::-1]

        self.layer_hidden_dims = [max(1, int(self.hidden_dim * r)) for r in self.width_ratios]

        self.fc1 = nnx.Linear(llm_dim, self.hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(self.hidden_dim, 2 * vggt_dim, rngs=rngs)
        self.vlm_norm = nnx.LayerNorm(llm_dim, rngs=rngs) if use_vlm_norm else None

    def _get_layer_hidden_dim(self, layer_index: int) -> int:
        if layer_index == -1:
            layer_index = self.num_layers - 1
        layer_index = max(0, min(layer_index, self.num_layers - 1))
        return self.layer_hidden_dims[layer_index]

    def align_dimension(self, x, layer_index: int = -1):
        if self.vlm_norm is not None:
            x = self.vlm_norm(x)

        hdim = self._get_layer_hidden_dim(layer_index)

        # fc1: (llm_dim, hidden_dim) → slice first hdim output channels
        x = x @ self.fc1.kernel.value[:, :hdim] + self.fc1.bias.value[:hdim]
        x = nnx.gelu(x)

        # fc2: (hidden_dim, 2*vggt_dim) → slice first hdim input channels
        x = x @ self.fc2.kernel.value[:hdim, :] + self.fc2.bias.value

        return x

    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden, align_mask):
        """Compute cosine similarity alignment loss.

        Args:
            vision_hidden: projected VLA features (B, N, 2*vggt_dim)
            vggt_hidden: target VGGT features (B, N, 2*vggt_dim)
            align_mask: boolean mask (B, N)
        """
        vision_norm = vision_hidden / (jnp.linalg.norm(vision_hidden, axis=-1, keepdims=True) + 1e-8)
        vggt_norm = vggt_hidden / (jnp.linalg.norm(vggt_hidden, axis=-1, keepdims=True) + 1e-8)
        cosine_sim = jnp.sum(vision_norm * vggt_norm, axis=-1)  # (B, N)
        loss_per_token = 1 - cosine_sim
        masked_loss = jnp.where(align_mask, loss_per_token, 0.0)
        num_valid = jnp.maximum(jnp.sum(align_mask, axis=-1), 1.0)  # (B,)
        per_batch_loss = jnp.sum(masked_loss, axis=-1) / num_valid
        return jnp.mean(per_batch_loss)

    def __call__(self, llm_emb, target_emb, align_mask, layer_index: int = -1):
        projected = self.align_dimension(llm_emb, layer_index=layer_index)
        return self.compute_align_loss_cosine(projected, target_emb, align_mask)

    def get_layer_info(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "full_hidden_dim": self.hidden_dim,
            "shallow_to_deep_increase": self.shallow_to_deep_increase,
            "width_ratios": self.width_ratios,
            "layer_hidden_dims": self.layer_hidden_dims,
        }


class Pi0Align(Pi0):
    """Pi0 with Spatial Forcing alignment loss.

    Extends the base Pi0 model with:
    - Multi-layer alignment projectors (shared or independent)
    - Alignment loss computation using pre-computed VGGT features
    - Hidden state extraction from specified transformer layers
    """

    def __init__(self, config: pi0_config.Pi0Config, extra_config, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        # Parse alignment config
        self.vla_layers_align = (
            [int(x) for x in extra_config.vla_layers_align.split(",")]
            if extra_config.vla_layers_align
            else [-1]
        )
        self.vggt_layers_align = (
            [int(x) for x in extra_config.vggt_layers_align.split(",")]
            if extra_config.vggt_layers_align
            else [-1]
        )
        assert len(self.vla_layers_align) == len(self.vggt_layers_align), (
            "vla_layers_align and vggt_layers_align must have the same number of entries"
        )

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self.llm_width = paligemma_config.width
        num_align_layers = len(self.vla_layers_align)
        self.share_projector = extra_config.share_projector
        self.ensemble_size = extra_config.ensemble_size

        # Parse width ratios
        width_ratios = None
        if extra_config.projector_width_ratios is not None:
            width_ratios = [float(x) for x in extra_config.projector_width_ratios.split(",")]

        # Initialize alignment projectors
        if extra_config.share_projector:
            # Shared: one pool reused across all layers (with sub-network slicing)
            shared_pool = [
                AlignProjector(
                    llm_dim=self.llm_width,
                    vggt_dim=extra_config.vggt_dim,
                    use_vlm_norm=extra_config.use_vlm_norm,
                    num_layers=num_align_layers,
                    width_ratios=width_ratios,
                    shallow_to_deep_increase=extra_config.projector_shallow_to_deep_increase,
                    rngs=rngs,
                )
                for _ in range(extra_config.ensemble_size)
            ]
            # All layers reference the same pool
            self.align_projectors = [shared_pool for _ in self.vla_layers_align]
        else:
            # Independent: each layer gets its own projector(s)
            self.align_projectors = [
                [
                    AlignProjector(
                        llm_dim=self.llm_width,
                        vggt_dim=extra_config.vggt_dim,
                        use_vlm_norm=extra_config.use_vlm_norm,
                        num_layers=1,
                        rngs=rngs,
                    )
                    for _ in range(extra_config.ensemble_size)
                ]
                for _ in self.vla_layers_align
            ]

    def compute_loss_with_alignment(
        self,
        rng,
        observation: _model.Observation,
        actions: _model.Actions,
        pooled_vggt_features,
        *,
        train: bool = False,
    ):
        """Compute action loss + alignment loss.

        Args:
            rng: JAX random key
            observation: model observation
            actions: target actions
            pooled_vggt_features: list of JAX arrays, one per alignment layer,
                each shape (B, img_len, 2*vggt_dim). Pre-computed and pooled VGGT features.
            train: training mode flag

        Returns:
            action_loss: scalar
            align_loss: scalar
            align_loss_details: dict of per-layer scalar losses
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix (images + language) and suffix (actions + time)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward pass with hidden states collection
        (prefix_out, suffix_out), _, all_hidden_states = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            output_hidden_states=True,
        )

        # ---- Action loss ----
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.mean(jnp.square(v_t - u_t), axis=-1))

        # ---- Alignment loss ----
        num_images = len(observation.images)
        img_len = num_images * TOKENS_PER_IMAGE

        # Build alignment mask from per-image masks
        img_masks_list = [observation.image_masks[name] for name in observation.images]
        img_masks_stack = jnp.stack(img_masks_list, axis=1)  # (B, num_images)
        align_mask = jnp.repeat(img_masks_stack, TOKENS_PER_IMAGE, axis=1)  # (B, img_len)

        # Multi-layer alignment
        align_loss = 0.0
        align_loss_details = {}

        for i, vla_layer_idx in enumerate(self.vla_layers_align):
            # all_hidden_states: tuple of (stacked_expert_0, stacked_expert_1)
            # stacked_expert_0: shape (depth, batch, prefix_len, dim)
            prefix_hidden_i = all_hidden_states[0][vla_layer_idx]  # (batch, prefix_len, dim)
            vision_hidden_i = prefix_hidden_i[:, :img_len, :]  # only image tokens

            # Pre-computed VGGT features for this layer
            vggt_pooled_i = pooled_vggt_features[i]  # (batch, img_len, 2*vggt_dim)

            # Ensemble of projectors
            layer_projectors = self.align_projectors[i]
            current_layer_loss = 0.0
            for j, projector in enumerate(layer_projectors):
                ens_loss = projector(vision_hidden_i, vggt_pooled_i, align_mask, layer_index=i)
                align_loss_details[f"align_loss_layer_{i}_ens_{j}"] = ens_loss
                current_layer_loss = current_layer_loss + ens_loss

            if len(layer_projectors) > 0:
                current_layer_loss = current_layer_loss / len(layer_projectors)

            align_loss_details[f"align_loss_layer_{i}_avg"] = current_layer_loss
            align_loss = align_loss + current_layer_loss

        align_loss = align_loss / len(self.vla_layers_align)
        align_loss_details["align_loss"] = align_loss

        return action_loss, align_loss, align_loss_details
