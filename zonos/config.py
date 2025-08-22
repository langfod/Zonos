from dataclasses import dataclass, field
from typing import Literal

import torch


# https://github.com/state-spaces/mamba/blob//mamba_ssm/utils/generation.py#L18
@dataclass
class InferenceParams:
    """
    Inference parameters for efficient context management during generation.
    
    This class stores all the state needed for efficient autoregressive generation,
    including sequence position tracking, batch management, and key-value caches
    for attention mechanisms.
    
    Based on: https://github.com/state-spaces/mamba/blob//mamba_ssm/utils/generation.py#L18
    
    Attributes:
        max_seqlen (int): Maximum sequence length for this inference session
        max_batch_size (int): Maximum batch size for this inference session  
        seqlen_offset (int): Current position in the sequence (for incremental generation)
        batch_size_offset (int): Current batch position (for batched generation)
        key_value_memory_dict (dict): Storage for attention key-value caches by layer
        lengths_per_sample (torch.Tensor, optional): Actual sequence length for each sample
    """

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: torch.Tensor | None = None

    def reset(self, max_seqlen, max_batch_size):
        """
        Reset inference parameters for a new generation session.
        
        Args:
            max_seqlen (int): New maximum sequence length
            max_batch_size (int): New maximum batch size
            
        Notes:
            - Resets sequence offset to 0 for fresh generation
            - Clears length tracking if it exists
            - Key-value caches are preserved but will be rewritten
        """
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


@dataclass
class BackboneConfig:
    """
    Configuration for the neural network backbone architecture.
    
    This dataclass defines all parameters needed to configure the transformer
    or hybrid transformer-mamba backbone of the Zonos model.
    
    Attributes:
        d_model (int): Model dimension/hidden size
        d_intermediate (int): Intermediate dimension for feedforward layers (0 = auto-compute)
        attn_mlp_d_intermediate (int): Intermediate dimension for attention MLP layers  
        n_layer (int): Number of transformer/mamba layers
        ssm_cfg (dict): Configuration for State Space Model (Mamba) components
        attn_layer_idx (list): Indices of layers that should use attention instead of SSM
        attn_cfg (dict): Configuration for attention layers
        rms_norm (bool): Whether to use RMSNorm instead of LayerNorm
        residual_in_fp32 (bool): Whether to compute residuals in float32 for stability
        norm_epsilon (float): Epsilon value for layer normalization
    """
    d_model: int = 1024
    d_intermediate: int = 0
    attn_mlp_d_intermediate: int = 0
    n_layer: int = 16
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = False
    residual_in_fp32: bool = False
    norm_epsilon: float = 1e-5


@dataclass
class PrefixConditionerConfig:
    """
    Configuration for the prefix conditioning system.
    
    Defines how different conditioning inputs (text, speaker, acoustic parameters)
    are processed and combined to guide audio generation.
    
    Attributes:
        conditioners (list[dict]): List of conditioner configurations, each specifying
            type and parameters for a specific conditioning modality
        projection (str): Type of projection applied to combined conditioning
            ('none', 'linear', 'mlp')
    """
    conditioners: list[dict]
    projection: Literal["none", "linear", "mlp"]


@dataclass
class ZonosConfig:
    """
    Main configuration class for the Zonos model.
    
    This class combines all configuration components needed to instantiate
    and run a complete Zonos text-to-speech model.
    
    Attributes:
        backbone (BackboneConfig): Neural network backbone configuration
        prefix_conditioner (PrefixConditionerConfig): Conditioning system configuration
        eos_token_id (int): End-of-sequence token ID for generation termination
        masked_token_id (int): Token ID used for masked positions during generation  
        pad_vocab_to_multiple_of (int): Padding factor for vocabulary alignment
    """
    backbone: BackboneConfig
    prefix_conditioner: PrefixConditionerConfig
    eos_token_id: int = 1024
    masked_token_id: int = 1025
    pad_vocab_to_multiple_of: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "ZonosConfig":
        """
        Create ZonosConfig instance from dictionary.
        
        Args:
            d (dict): Configuration dictionary containing 'backbone' and 
                'prefix_conditioner' keys, plus any additional parameters
                
        Returns:
            ZonosConfig: Configured ZonosConfig instance
            
        Notes:
            - Automatically handles nested config object creation
            - Extracts backbone and prefix_conditioner configs into separate objects
            - Passes remaining parameters as top-level config attributes
        """
        d = d.copy()
        backbone_config = BackboneConfig(**d.pop("backbone"))
        prefix_conditioner_config = PrefixConditionerConfig(**d.pop("prefix_conditioner"))
        config = cls(backbone_config, prefix_conditioner_config, **d)
        return config
