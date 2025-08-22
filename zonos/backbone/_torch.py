# Based on gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/095b2229ee3a40e379c11f05b94bd6923db63b4b/model.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from zonos.config import BackboneConfig, InferenceParams


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000) -> torch.Tensor:
    """
    Precompute rotary position encoding frequencies.
    
    Creates the complex exponential frequencies used in RoPE (Rotary Position Embedding)
    for efficient position encoding in transformer attention.
    
    Args:
        seq_len (int): Maximum sequence length to precompute for
        n_elem (int): Number of embedding dimensions (typically head_dim)
        base (float): Base for the exponential (10000 is standard)
        
    Returns:
        torch.Tensor: Precomputed frequencies as [seq_len, n_elem//2, 2] (cos/sin pairs)
        
    Notes:
        - Based on the RoPE paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        - Frequencies decrease exponentially to encode position information
        - Cached as cos/sin pairs for efficient application during attention
    """
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.
    
    Applies RoPE (Rotary Position Embedding) by rotating query/key vectors
    based on their position in the sequence. This enables relative position
    encoding while maintaining translation equivariance.
    
    Args:
        x (torch.Tensor): Input tensor to apply RoPE to [batch, seq_len, n_heads, head_dim]
        freqs_cis (torch.Tensor): Precomputed frequencies [seq_len, head_dim//2, 2]
        
    Returns:
        torch.Tensor: Input tensor with RoPE applied, same shape as input
        
    Notes:
        - Operates on pairs of dimensions to apply 2D rotation
        - More efficient than applying rotation matrices directly
        - Critical for maintaining position information in transformer attention
    """
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def _update_kv_cache(
    k: torch.Tensor, v: torch.Tensor, inference_params: InferenceParams, layer_idx: int
) -> torch.Tensor:
    """
    Update key-value cache for efficient autoregressive inference.
    
    Updates the cached key and value tensors for a specific transformer layer
    during inference. This enables incremental decoding without recomputing
    attention for previous tokens.
    
    Args:
        k (torch.Tensor): Key tensor [batch_size, seqlen, nheads, head_dim] or [batch_size, 1, nheads, head_dim]
        v (torch.Tensor): Value tensor [batch_size, seqlen, nheads, head_dim] or [batch_size, 1, nheads, head_dim] 
        inference_params (InferenceParams): Inference state containing cache and offsets
        layer_idx (int): Index of the transformer layer being updated
        
    Returns:
        torch.Tensor: Updated key-value cache for this layer [batch, seq_len, 2, nheads, head_dim]
        
    Notes:
        - Cache format: [batch, seq, kv_pair, heads, dim] where kv_pair=0 is keys, kv_pair=1 is values
        - Handles batch and sequence offsets for efficient memory management
        - Critical for fast autoregressive generation in transformer models
    """
    assert layer_idx in inference_params.key_value_memory_dict
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + k.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + k.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, 0, ...] = k
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, 1, ...] = v
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class TorchZonosBackbone(nn.Module):
    """
    Transformer-based backbone architecture for Zonos TTS model.
    
    Implements a transformer backbone using multi-head attention with RoPE
    (Rotary Position Embedding), layer normalization, and feedforward networks.
    Supports both training and efficient inference with KV caching.
    
    Attributes:
        supported_architectures (list): List of supported model architectures
        freqs_cis (torch.Tensor): Cached rotary position embeddings
        config (BackboneConfig): Model configuration
        layers (nn.ModuleList): Stack of transformer blocks
        norm_f (nn.LayerNorm): Final layer normalization
        
    Notes:
        - Only supports transformer architecture, not SSM/Mamba
        - Uses RoPE for positional encoding instead of absolute position embeddings
        - Optimized for both training and autoregressive inference
    """
    supported_architectures = ["transformer"]
    freqs_cis: torch.Tensor

    def __init__(self, config: BackboneConfig):
        """
        Initialize the TorchZonosBackbone.
        
        Sets up the transformer architecture with the specified configuration,
        creating the layer stack and final normalization.
        
        Args:
            config (BackboneConfig): Configuration object containing model parameters
                
        Raises:
            AssertionError: If config specifies SSM architecture (not supported)
            
        Notes:
            - Only transformer architecture is supported by this implementation
            - Each layer is initialized with its layer index for position-aware operations
        """
        assert not config.ssm_cfg, "This backbone implementation only supports the Transformer model."
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config, i) for i in range(config.n_layer))
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        """
        Allocate memory for key-value caches during inference.
        
        Pre-allocates memory for storing key and value tensors across all transformer
        layers to enable efficient autoregressive generation without reallocation.
        
        Args:
            batch_size (int): Number of sequences to process in parallel
            max_seqlen (int): Maximum sequence length to support
            dtype (torch.dtype): Data type for cache tensors (default: bfloat16)
            
        Returns:
            dict: Mapping from layer index to allocated cache tensors
            
        Notes:
            - Cache format optimized for transformer attention computation
            - RoPE frequencies precomputed and cached for position encoding
            - Memory allocation is performed once and reused across generation steps
            - Pure function version available via allocate_inference_cache_pure()
        """
        kv_cache_dict, freqs_cis = self.allocate_inference_cache_pure(batch_size, max_seqlen, dtype)
        self.freqs_cis = freqs_cis  # Cache for forward pass
        return kv_cache_dict
    
    def allocate_inference_cache_pure(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        """
        Pure version of allocate_inference_cache with no side effects.
        
        Pre-allocates memory for storing key and value tensors across all transformer
        layers to enable efficient autoregressive generation without reallocation.
        
        Args:
            batch_size (int): Number of sequences to process in parallel
            max_seqlen (int): Maximum sequence length to support
            dtype (torch.dtype): Data type for cache tensors (default: bfloat16)
            
        Returns:
            tuple: (kv_cache_dict, freqs_cis) where:
                - kv_cache_dict: Mapping from layer index to allocated cache tensors
                - freqs_cis: Precomputed rotary embeddings for position encoding
            
        Notes:
            - Pure function with no side effects
            - Caller is responsible for managing freqs_cis state
            - Cache format optimized for transformer attention computation
            - RoPE frequencies precomputed up to length 16384 for efficiency
        """
        head_dim = self.config.d_model // self.config.attn_cfg["num_heads"]
        freqs_cis = precompute_freqs_cis(16384, head_dim)
        kv_cache_dict = {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }
        return kv_cache_dict, freqs_cis

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams) -> torch.Tensor:
        """
        Forward pass through the transformer backbone.
        
        Processes input embeddings through all transformer layers with rotary position
        embeddings, applying attention, feedforward, and residual connections.
        
        Args:
            hidden_states (torch.Tensor): Input embeddings [batch, seq_len, d_model]
            inference_params (InferenceParams): Inference state containing caches and offsets
            
        Returns:
            torch.Tensor: Processed hidden states [batch, seq_len, d_model]
            
        Notes:
            - Position encodings are computed based on sequence offsets for incremental generation
            - Each layer receives the same RoPE frequencies for consistency
            - Final layer normalization applied before returning
        """
        input_pos = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        input_pos = input_pos + inference_params.lengths_per_sample.unsqueeze(-1)

        freqs_cis = self.freqs_cis[input_pos].expand(hidden_states.shape[0], -1, -1, -1)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, inference_params, freqs_cis)
        return self.norm_f(hidden_states)


class TransformerBlock(nn.Module):
    """
    Individual transformer block with multi-head attention and feedforward network.
    
    Implements a standard transformer layer with pre-layer normalization, multi-head
    self-attention with RoPE, and a feedforward network with residual connections.
    
    Attributes:
        layer_idx (int): Index of this layer in the transformer stack
        ln_1 (nn.LayerNorm): Layer normalization before attention
        attn (Attention): Multi-head self-attention mechanism
        ln_2 (nn.LayerNorm): Layer normalization before feedforward
        mlp (FeedForward): Feedforward network
        
    Notes:
        - Uses pre-layer normalization for training stability
        - Residual connections applied around both attention and feedforward
        - Layer index used for KV cache management during inference
    """
    def __init__(self, config: BackboneConfig, layer_idx: int) -> None:
        """
        Initialize a transformer block.
        
        Sets up the layer normalization, attention, and feedforward components
        with configuration parameters.
        
        Args:
            config (BackboneConfig): Model configuration containing layer parameters
            layer_idx (int): Index of this layer in the transformer stack
            
        Notes:
            - Layer index is passed to attention for KV cache management
            - Head dimensions computed from model size and attention configuration
        """
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mixer = Attention(config, layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mlp = FeedForward(config)

        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // config.attn_cfg["num_heads"]

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        """
        Allocate KV cache for this transformer layer.
        
        Creates a tensor to store key and value states for efficient autoregressive
        inference without recomputing attention for past tokens.
        
        Args:
            batch_size (int): Number of sequences to process in parallel
            max_seqlen (int): Maximum sequence length to support
            dtype (torch.dtype): Data type for cache tensor (default: bfloat16)
            
        Returns:
            tuple: (kv_cache_tensor, None) where cache is [batch, seq, 2, heads_kv, head_dim]
            
        Notes:
            - Cache dimension 2 corresponds to key (index 0) and value (index 1)
            - Uses num_heads_kv for efficient grouped-query attention
        """
        return torch.empty(batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim, dtype=dtype), None

    def forward(self, x: torch.Tensor, inference_params: InferenceParams, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Applies pre-layer normalization, self-attention, and feedforward network
        with residual connections.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, d_model]
            inference_params (InferenceParams): Inference state and caches
            freqs_cis (torch.Tensor): Rotary position embeddings
            
        Returns:
            torch.Tensor: Output tensor [batch, seq_len, d_model]
            
        Notes:
            - Pre-normalization applied before attention and feedforward
            - Residual connections ensure gradient flow during training
        """
        x = x + self.mixer(self.norm(x), inference_params, freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention with rotary position embeddings.
    
    Implements scaled dot-product attention with grouped-query attention (GQA)
    for efficiency and RoPE for relative position encoding.
    
    Attributes:
        num_heads (int): Number of query/attention heads
        num_heads_kv (int): Number of key/value heads (for GQA)
        head_dim (int): Dimension per attention head
        layer_idx (int): Layer index for KV cache management
        in_proj (nn.Linear): Linear projection for Q, K, V
        out_proj (nn.Linear): Output projection after attention
        
    Notes:
        - Supports grouped-query attention where fewer key/value heads than query heads
        - Uses rotary position embeddings instead of absolute position encodings
        - Optimized for both training and cached inference
    """
    def __init__(self, config: BackboneConfig, layer_idx: int):
        """
        Initialize multi-head attention.
        
        Sets up the query, key, value projections and output projection
        with support for grouped-query attention.
        
        Args:
            config (BackboneConfig): Model configuration
            layer_idx (int): Layer index for KV cache management
            
        Notes:
            - Total projection size accounts for Q heads + K heads + V heads
            - Grouped-query attention uses fewer KV heads than Q heads for efficiency
        """
        super().__init__()
        self.num_heads = config.attn_cfg["num_heads"]
        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // self.num_heads
        self.layer_idx = layer_idx

        total_head_dim = (self.num_heads + 2 * self.num_heads_kv) * self.head_dim
        self.in_proj = nn.Linear(config.d_model, total_head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, inference_params: InferenceParams, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Computes scaled dot-product attention with RoPE position encoding
        and KV caching for efficient autoregressive inference.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, d_model]
            inference_params (InferenceParams): Inference state and caches
            freqs_cis (torch.Tensor): Rotary position embeddings [batch, seq_len, heads, head_dim//2, 2]
            
        Returns:
            torch.Tensor: Attention output [batch, seq_len, d_model]
            
        Notes:
            - Applies RoPE to queries and keys before attention computation
            - Updates and retrieves from KV cache for efficient inference
            - Uses grouped-query attention with enable_gqa=True
            - Causal masking applied for sequences longer than 1 token
        """
        batch_size, seqlen, _ = x.shape

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim
        q, k, v = self.in_proj(x).split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(batch_size, seqlen, self.num_heads, self.head_dim)
        k = k.view(batch_size, seqlen, self.num_heads_kv, self.head_dim)
        v = v.view(batch_size, seqlen, self.num_heads_kv, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        kv = _update_kv_cache(k, v, inference_params, self.layer_idx)
        k, v = kv.unbind(dim=-3)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        y = F.scaled_dot_product_attention(q, k, v, is_causal=seqlen > 1, enable_gqa=True)

        y = y.transpose(1, 2).contiguous().view(batch_size, seqlen, q_size)

        y = self.out_proj(y)
        return self.out_proj(y)


class FeedForward(nn.Module):
    """
    Gated feedforward network with SiLU activation.
    
    Implements a gated linear unit (GLU) variant using SiLU (Swish) activation.
    The gating mechanism allows the network to control information flow.
    
    Attributes:
        fc1 (nn.Linear): First linear layer projecting to 2x intermediate dimension
        fc2 (nn.Linear): Second linear layer projecting back to model dimension
        
    Notes:
        - Uses gated linear unit (GLU) with SiLU activation for better performance
        - Input is projected to 2x intermediate size then split into value and gate
        - Gate controls which parts of the value are passed through
    """
    def __init__(self, config: BackboneConfig) -> None:
        """
        Initialize the feedforward network.
        
        Sets up the linear layers for the gated feedforward mechanism.
        
        Args:
            config (BackboneConfig): Model configuration containing layer dimensions
            
        Notes:
            - First layer projects to 2x intermediate size (for value and gate)
            - Second layer projects back to original model dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.attn_mlp_d_intermediate, bias=False)
        self.fc2 = nn.Linear(config.attn_mlp_d_intermediate, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the gated feedforward network.
        
        Applies gated linear unit with SiLU activation function.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output tensor [batch, seq_len, d_model]
            
        Notes:
            - Input projected to 2x intermediate size then split into value and gate
            - Gate uses SiLU activation to control information flow
            - Final projection returns to original model dimension
        """
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))
