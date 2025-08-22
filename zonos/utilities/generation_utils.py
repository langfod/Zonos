"""
Generation utilities for neural audio synthesis.

This module provides core generation functions for autoregressive audio token
generation, including single-token decoding, prefill operations, and CUDA graph
management for high-performance inference.
"""

import torch
from typing import Callable, Optional
from zonos.config import InferenceParams


class CUDAGraphManager:
    """
    Manages CUDA graph capture and replay for optimized inference.
    
    Provides stateful CUDA graph management with automatic capture when
    batch size changes and efficient replay for repeated inference.
    """
    
    def __init__(self):
        """Initialize CUDA graph manager with empty state."""
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.batch_size: Optional[int] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.logits: Optional[torch.Tensor] = None
        self.inference_params: Optional[InferenceParams] = None
        self.cfg_scale: Optional[float] = None
    
    def needs_capture(self, batch_size: int, inference_params: InferenceParams = None) -> bool:
        """
        Check if CUDA graph needs to be captured.
        
        Args:
            batch_size (int): Current batch size
            inference_params (InferenceParams, optional): Current inference parameters
            
        Returns:
            bool: True if capture is needed (first time, batch size changed, or inference_params changed)
        """
        # Always need to capture if no graph exists
        if self.graph is None:
            return True
            
        # Check if batch size changed
        if self.batch_size != batch_size:
            return True
            
        # Check if inference_params object has changed (different instance)
        if inference_params is not None and self.inference_params is not inference_params:
            return True
            
        return False
    
    def prepare_capture(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float
    ) -> None:
        """
        Prepare for CUDA graph capture.
        
        Sets up state variables and allocates buffers needed for graph capture.
        
        Args:
            input_ids (torch.Tensor): Input token IDs for sizing
            inference_params (InferenceParams): Inference configuration
            cfg_scale (float): Classifier-free guidance scale
        """
        self.batch_size = input_ids.size(0)
        self.inference_params = inference_params
        self.cfg_scale = cfg_scale
        self.input_ids = input_ids.clone()
    
    def capture_graph(
        self,
        compute_logits_fn: Callable[[torch.Tensor, InferenceParams, float], torch.Tensor],
        embed_codes_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """
        Capture CUDA graph for single-token decoding.
        
        Creates a CUDA graph that can be replayed for efficient inference
        without kernel launch overhead.
        
        Args:
            compute_logits_fn (Callable): Function to compute logits from hidden states
            embed_codes_fn (Callable): Function to embed input codes
            
        Notes:
            - Performs warmup runs before capture to ensure stable memory layout
            - Uses captured tensors for input/output to ensure proper graph replay
        """
        # Warmup runs to ensure stable memory allocation
        for _ in range(3):
            hidden_states = embed_codes_fn(self.input_ids)
            hidden_states = hidden_states.repeat(2, 1, 1)
            logits = compute_logits_fn(hidden_states, self.inference_params, self.cfg_scale)
        
        # Allocate output buffer
        self.logits = torch.empty_like(logits)
        
        # Capture the computation graph
        graph = torch.cuda.CUDAGraph()
        
        def capture_region():
            hidden_states_local = embed_codes_fn(self.input_ids)
            hidden_states_local = hidden_states_local.repeat(2, 1, 1)
            self.logits.copy_(compute_logits_fn(hidden_states_local, self.inference_params, self.cfg_scale))
        
        with torch.cuda.graph(graph):
            capture_region()
        
        self.graph = graph
    
    def replay(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Replay captured CUDA graph with new input.
        
        Args:
            input_ids (torch.Tensor): New input token IDs
            
        Returns:
            torch.Tensor: Computed logits from graph replay
        """
        self.input_ids.copy_(input_ids)
        self.graph.replay()
        return self.logits
    
    def clear(self) -> None:
        """Clear all captured state."""
        self.graph = None
        self.batch_size = None
        self.input_ids = None
        self.logits = None
        self.inference_params = None
        self.cfg_scale = None


def decode_one_token_static(
    input_ids: torch.Tensor,
    inference_params: InferenceParams,
    cfg_scale: float,
    embed_codes_fn: Callable[[torch.Tensor], torch.Tensor],
    compute_logits_fn: Callable[[torch.Tensor, InferenceParams, float], torch.Tensor],
    cuda_graph_manager: Optional[CUDAGraphManager] = None,
    allow_cudagraphs: bool = True,
) -> torch.Tensor:
    """
    Decode single token with optional CUDA graph acceleration.
    
    Performs single-step autoregressive decoding with support for classifier-free
    guidance and CUDA graph optimization for improved performance.
    
    Args:
        input_ids (torch.Tensor): Input token IDs [batch, num_codebooks, seq_len]
        inference_params (InferenceParams): Inference configuration and cache state
        cfg_scale (float): Classifier-free guidance scale (1.0 = no guidance)
        embed_codes_fn (Callable): Function to embed input codes to hidden states
        compute_logits_fn (Callable): Function to compute logits from hidden states
        cuda_graph_manager (CUDAGraphManager, optional): CUDA graph manager for acceleration
        allow_cudagraphs (bool): Whether to allow CUDA graph usage
        
    Returns:
        torch.Tensor: Output logits [batch, num_codebooks, vocab_size]
        
    Notes:
        - Uses CUDA graphs when available for significant speedup
        - Automatically handles batch size changes by recapturing graphs
        - Falls back to standard computation for non-CUDA devices
        - CFG requires doubling batch size for conditional/unconditional paths
    """
    # Simple case: no classifier-free guidance
    if cfg_scale == 1.0:
        hidden_states = embed_codes_fn(input_ids)
        return compute_logits_fn(hidden_states, inference_params, cfg_scale)

    batch_size = input_ids.size(0)
    
    # Check if CUDA graphs can be used
    can_use_graphs = (
        allow_cudagraphs and 
        input_ids.device.type == "cuda" and 
        cuda_graph_manager is not None
    )
    
    if not can_use_graphs:
        # Standard computation without CUDA graphs
        hidden_states = embed_codes_fn(input_ids)
        hidden_states = hidden_states.repeat(2, 1, 1)  # Double for CFG
        return compute_logits_fn(hidden_states, inference_params, cfg_scale)
    
    # CUDA graph path
    if cuda_graph_manager.needs_capture(batch_size, inference_params):
        # Clear old graph and prepare new capture
        cuda_graph_manager.clear()
        cuda_graph_manager.prepare_capture(input_ids, inference_params, cfg_scale)
        cuda_graph_manager.capture_graph(compute_logits_fn, embed_codes_fn)
    
    # Replay captured graph
    return cuda_graph_manager.replay(input_ids)


def prefill_static(
    prefix_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    inference_params: InferenceParams,
    cfg_scale: float,
    embed_codes_fn: Callable[[torch.Tensor], torch.Tensor],
    compute_logits_fn: Callable[[torch.Tensor, InferenceParams, float], torch.Tensor],
) -> torch.Tensor:
    """
    Prefill operation for efficient context processing.
    
    Combines pre-computed prefix hidden states with new input embeddings
    to process contexts efficiently without recomputing the entire sequence.
    
    Args:
        prefix_hidden_states (torch.Tensor): Pre-computed prefix states [batch, prefix_len, d_model]
        input_ids (torch.Tensor): New input token IDs [batch, num_codebooks, new_len]
        inference_params (InferenceParams): Inference configuration and cache state
        cfg_scale (float): Classifier-free guidance scale
        embed_codes_fn (Callable): Function to embed input codes to hidden states
        compute_logits_fn (Callable): Function to compute logits from hidden states
        
    Returns:
        torch.Tensor: Output logits [batch, num_codebooks, vocab_size]
        
    Notes:
        - Concatenates prefix states with newly embedded codes
        - Handles CFG by expanding input_ids to match prefix batch size
        - Efficient for processing long contexts incrementally
    """
    # Handle classifier-free guidance by expanding input if needed
    if cfg_scale != 1.0:
        input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1)
    
    # Embed new input codes and concatenate with prefix
    new_hidden_states = embed_codes_fn(input_ids)
    hidden_states = torch.cat([prefix_hidden_states, new_hidden_states], dim=1)
    
    return compute_logits_fn(hidden_states, inference_params, cfg_scale)


def setup_inference_cache_static(
    backbone_allocate_fn: Callable[[int, int, torch.dtype, torch.device], InferenceParams],
    batch_size: int,
    max_seqlen: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = None
) -> InferenceParams:
    """
    Set up inference cache for efficient generation.
    
    Creates key-value caches and other state needed for efficient autoregressive
    generation without recomputing past states.
    
    Args:
        backbone_allocate_fn (Callable): Backbone-specific cache allocation function
        batch_size (int): Batch size for generation
        max_seqlen (int): Maximum sequence length to allocate for
        dtype (torch.dtype): Data type for cache tensors
        device (torch.device, optional): Device for cache allocation
        
    Returns:
        InferenceParams: Configured inference parameters with allocated caches
        
    Notes:
        - Delegates to backbone-specific allocation for compatibility
        - Pre-allocates caches to avoid memory allocations during generation
        - Cache size should account for maximum expected sequence length
    """
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    
    return backbone_allocate_fn(batch_size, max_seqlen, dtype, device)


def can_use_cudagraphs_static(device: torch.device, backbone_class_name: str) -> bool:
    """
    Check if CUDA graphs can be used for optimization.
    
    Determines whether the current configuration supports CUDA graph acceleration
    based on device type and backbone architecture.
    
    Args:
        device (torch.device): Target device for inference
        backbone_class_name (str): Name of the backbone class
        
    Returns:
        bool: True if CUDA graphs are supported and recommended
        
    Notes:
        - Currently only mamba-ssm backbone supports CUDA graphs
        - Requires CUDA device for graph capture and replay
        - CUDA graphs provide significant speedup for inference loops
    """
    return device.type == "cuda" and "_mamba_ssm" in backbone_class_name


def warmup_cuda_graphs(
    input_ids: torch.Tensor,
    inference_params: InferenceParams,
    cfg_scale: float,
    embed_codes_fn: Callable[[torch.Tensor], torch.Tensor],
    compute_logits_fn: Callable[[torch.Tensor, InferenceParams, float], torch.Tensor],
    warmup_steps: int = 3
) -> None:
    """
    Warm up CUDA operations before graph capture.
    
    Performs warmup iterations to ensure stable memory allocation and
    kernel initialization before capturing CUDA graphs.
    
    Args:
        input_ids (torch.Tensor): Representative input for warmup
        inference_params (InferenceParams): Inference configuration
        cfg_scale (float): Classifier-free guidance scale
        embed_codes_fn (Callable): Function to embed codes
        compute_logits_fn (Callable): Function to compute logits
        warmup_steps (int): Number of warmup iterations
        
    Notes:
        - Essential for stable CUDA graph capture
        - Ensures consistent memory layout across runs
        - Should match the exact computation being captured
    """
    for _ in range(warmup_steps):
        hidden_states = embed_codes_fn(input_ids)
        if cfg_scale != 1.0:
            hidden_states = hidden_states.repeat(2, 1, 1)
        _ = compute_logits_fn(hidden_states, inference_params, cfg_scale)
        
        # Force CUDA synchronization to complete operations
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize()


class GenerationState:
    """
    Manages state for multi-step generation processes.
    
    Provides a stateful container for managing generation parameters,
    caches, and optimization state across multiple generation steps.
    """
    
    def __init__(
        self,
        inference_params: InferenceParams,
        cuda_graph_manager: Optional[CUDAGraphManager] = None
    ):
        """
        Initialize generation state.
        
        Args:
            inference_params (InferenceParams): Inference configuration and caches
            cuda_graph_manager (CUDAGraphManager, optional): CUDA graph manager
        """
        self.inference_params = inference_params
        self.cuda_graph_manager = cuda_graph_manager or CUDAGraphManager()
        self.step_count = 0
        self.total_tokens = 0
    
    def reset(self) -> None:
        """Reset generation state for new sequence."""
        if hasattr(self.inference_params, 'reset'):
            self.inference_params.reset()
        self.step_count = 0
        self.total_tokens = 0
    
    def step(self) -> None:
        """Increment step counters."""
        self.step_count += 1
        self.total_tokens += 1
    
    def clear_cuda_graphs(self) -> None:
        """Clear CUDA graph cache."""
        self.cuda_graph_manager.clear()
