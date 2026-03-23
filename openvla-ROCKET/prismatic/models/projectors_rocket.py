"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class AlignProjector(nn.Module):
    """
    Calculate the alignment between LLM and VGGT embeddings.
    
    Supports multiple layers sharing the same projector, dynamically selecting sub-networks based on layer_index:
    - num_layers: Total number of layers sharing this projector
    - width_ratios: Ratio of hidden_dim width for each layer, e.g., [0.25, 0.5, 0.75, 1.0]
    - shallow_to_deep_increase: 
        - True: Shallow layers (small index) use fewer parameters, deep layers (large index) use more parameters
        - False: Shallow layers use more parameters, deep layers use fewer parameters
    - layer_index = -1 represents the last layer (the deepest layer)
    """
    def __init__(
            self,
            llm_dim: int,
            vggt_dim: int,
            hidden_dim: int = None,
            align_loss_type: str = "cosine",
            use_vlm_norm: bool = False,
            # Dynamic sub-network parameters
            num_layers: int = 1,
            width_ratios: list = None,
            use_matryoshka: bool = True,
            shallow_to_deep_increase: bool = True,
        ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 2 * vggt_dim
        self.align_loss_type = align_loss_type

        # Dynamic sub-network configuration
        self.num_layers = num_layers
        self.use_matryoshka = use_matryoshka
        self.shallow_to_deep_increase = shallow_to_deep_increase

        if not use_matryoshka:
            # All layers use full hidden_dim (shared baseline)
            self.width_ratios = [1.0] * num_layers
        elif width_ratios is not None:
            assert len(width_ratios) == num_layers, \
                f"width_ratios length ({len(width_ratios)}) must match num_layers ({num_layers})"
            self.width_ratios = width_ratios
        else:
            # Default linear distribution from 0.2 to 1.0
            self.width_ratios = [0.2 + 0.8 * i / max(num_layers - 1, 1) for i in range(num_layers)]

        # Reverse ratios if shallow layers use more parameters (only meaningful when matryoshka is on)
        if use_matryoshka and not shallow_to_deep_increase:
            self.width_ratios = self.width_ratios[::-1]

        # Pre-calculate hidden_dim sub-network width for each layer
        self.layer_hidden_dims = [max(1, int(self.hidden_dim * r)) for r in self.width_ratios]

        self.fc1 = nn.Linear(self.llm_dim, self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim, 2 * self.vggt_dim, bias=True)
        self.act_fn1 = nn.GELU()
        
        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
    
    def _get_layer_hidden_dim(self, layer_index: int) -> int:
        """Get corresponding hidden_dim width based on layer_index"""
        # Handle -1 representing the last layer
        if layer_index == -1:
            layer_index = self.num_layers - 1
        
        # Ensure index is within valid range
        layer_index = max(0, min(layer_index, self.num_layers - 1))
        return self.layer_hidden_dims[layer_index]

    def align_dimension(self, LLM_embedding: torch.Tensor = None, layer_index: int = -1) -> torch.Tensor:
        """
        Select sub-network for forward pass based on layer_index
        
        Args:
            LLM_embedding: Input LLM embedding
            layer_index: Layer index, -1 represents the last layer (using the deepest layer configuration)
        """
        if self.vlm_norm is not None:
            LLM_embedding = self.vlm_norm(LLM_embedding)
        
        # Get hidden_dim to be used for the current layer
        current_hidden_dim = self._get_layer_hidden_dim(layer_index)
        
        # Use sub-network: slice part of the weights from fc1 and fc2
        # fc1: (llm_dim -> hidden_dim) => use the first current_hidden_dim output channels
        projected_features = nn.functional.linear(
            LLM_embedding,
            self.fc1.weight[:current_hidden_dim, :],
            self.fc1.bias[:current_hidden_dim] if self.fc1.bias is not None else None
        )
        
        projected_features = self.act_fn1(projected_features)
        
        # fc2: (hidden_dim -> 2*vggt_dim) => use the first current_hidden_dim input channels
        projected_features = nn.functional.linear(
            projected_features,
            self.fc2.weight[:, :current_hidden_dim],
            self.fc2.bias
        )
        
        return projected_features
    
    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden):
        # vision_hidden has a shape of (bs, N, D)
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        align_loss = 0
        bsz = vision_hidden.shape[0]
        for _vision, _vggt in zip(vision_hidden, vggt_hidden):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _vggt = torch.nn.functional.normalize(_vggt, dim=-1)
            # align_loss += 1 - torch.mean(vision_hidden * vggt_hidden).sum(dim=-1).mean()
            align_loss += 1 - mean_flat((_vision * _vggt).sum(dim=-1))  # Cosine similarity loss
        align_loss /= bsz  # Average over batch size
        return align_loss
    
    def forward(self, LLM_emb, target_emb, layer_index: int = -1):
        """
        Args:
            LLM_emb: LLM embedding
            target_emb: Target embedding (VGGT)
            layer_index: Layer index, used to select sub-network
                - 0, 1, 2, ... : Specific layer index
                - -1: Last layer (the deepest layer)
        """
        if self.align_loss_type == "cosine":
            # project vla dimension and calculate align loss
            with torch.autocast("cuda", dtype=torch.bfloat16):
                LLM_emb = self.align_dimension(LLM_emb, layer_index=layer_index)
            align_loss = self.compute_align_loss_cosine(LLM_emb, target_emb).mean()  # mean for sequence length
            return align_loss
        else:
            raise NotImplementedError(f"Align loss type {self.align_loss_type} is not implemented.")
    
    def get_layer_info(self) -> dict:
        """Return configuration info for each layer, for debug purpose"""
        return {
            "num_layers": self.num_layers,
            "full_hidden_dim": self.hidden_dim,
            "use_matryoshka": self.use_matryoshka,
            "shallow_to_deep_increase": self.shallow_to_deep_increase,
            "width_ratios": self.width_ratios,
            "layer_hidden_dims": self.layer_hidden_dims,
        }

