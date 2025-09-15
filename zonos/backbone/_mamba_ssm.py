import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import create_block
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn

from zonos.config import BackboneConfig, InferenceParams

class MambaSSMZonosBackbone(nn.Module):
    """
    Zonos backbone using Mamba State Space Models with optional attention layers.
    
    This backbone implementation uses the mamba-ssm library to create a hybrid
    architecture that can combine Mamba layers (for efficiency) with transformer
    attention layers (for quality) based on the configuration.
    
    Mamba SSMs provide linear scaling with sequence length while maintaining
    strong modeling capabilities, making them ideal for long audio sequences.
    
    Attributes:
        supported_architectures (list): Architecture types this backbone supports
        config (BackboneConfig): Configuration containing layer specifications
        layers (nn.ModuleList): Stack of Mamba/attention blocks  
        norm_f (nn.LayerNorm): Final layer normalization
    """
    supported_architectures = ["transformer", "hybrid"]

    def __init__(self, config: BackboneConfig):
        """
        Initialize the Mamba SSM backbone.
        
        Args:
            config (BackboneConfig): Configuration specifying architecture details
            
        Notes:
            - Creates layers based on config.n_layer
            - Layers at indices in config.attn_layer_idx use attention instead of SSM
            - Uses the mamba-ssm library's create_block for optimal implementations
            - Enables fused operations and custom normalization options
        """
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=config.d_model,
                    d_intermediate=config.d_intermediate
                    if (i not in config.attn_layer_idx)
                    else config.attn_mlp_d_intermediate,
                    ssm_cfg=config.ssm_cfg,
                    layer_idx=i,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    norm_epsilon=config.norm_epsilon,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=True,
                    rms_norm=config.rms_norm,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        """
        Allocate inference cache for efficient generation.
        
        Creates per-layer key-value caches and other state needed for efficient
        autoregressive generation without recomputing past activations.
        
        Args:
            batch_size (int): Maximum batch size for generation
            max_seqlen (int): Maximum sequence length to support
            dtype (torch.dtype): Data type for cache tensors
            
        Returns:
            dict: Dictionary mapping layer indices to their allocated caches
            
        Notes:
            - Each layer allocates its own cache based on its type (SSM vs attention)
            - Cache allocation is handled by the mamba-ssm library
            - Essential for efficient long-sequence generation
        """
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams | None = None):
        """
        Forward pass through the Mamba SSM backbone.
        
        Args:
            hidden_states (torch.Tensor): Input embeddings of shape [batch, seq_len, d_model]
            inference_params (InferenceParams, optional): Caching parameters for efficient generation
            
        Returns:
            torch.Tensor: Processed hidden states of shape [batch, seq_len, d_model]
            
        Notes:
            - Processes through all layers sequentially
            - Each layer maintains residual connections and applies fused normalization
            - Final layer normalization uses optimized triton kernel when available
            - Supports both training (inference_params=None) and generation modes
        """
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params)

        return layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            residual,
            eps=self.norm_f.eps,
            residual_in_fp32=self.config.residual_in_fp32,
            is_rms_norm=self.config.rms_norm,
        )
