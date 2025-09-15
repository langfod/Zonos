"""
Optimized tensor operations for efficient neural network inference.

This module provides fused tensor operations that reduce kernel launches and improve
memory access patterns, particularly useful for autoregressive generation loops.
"""

import torch
from zonos.config import InferenceParams


def fused_frame_update(
    delayed_codes: torch.Tensor,
    offset: int, 
    batch_size: int,
    next_token: torch.Tensor,
    unknown_token: int,
    codebook_dimension: int,
    temp_mask_buffer: torch.Tensor = None
) -> None:
    """
    Efficiently update frame slice with new tokens using fused operations.
    
    Combines frame extraction, masking, and assignment with contiguous memory operations
    to reduce kernel launches and improve memory access patterns.
    
    Args:
        delayed_codes (torch.Tensor): Delayed pattern codes [batch, codebooks, seq_len]
        offset (int): Current position offset in the sequence
        batch_size (int): Number of sequences being processed
        next_token (torch.Tensor): New tokens to assign [batch, codebooks]
        unknown_token (int): Sentinel value indicating unknown/masked positions
        codebook_dimension (int): Number of codebooks in the autoencoder
        temp_mask_buffer (torch.Tensor, optional): Pre-allocated mask buffer [batch, codebooks]
            for memory-efficient operations. If None, allocates temporary mask.
            
    Notes:
        - Operates in-place on delayed_codes tensor
        - Uses provided buffer to avoid memory allocations when available
        - Handles edge cases where frame slice is empty
    """
    frame_slice = delayed_codes[..., offset: offset + 1]
    if frame_slice.numel() > 0:
        frame_view = frame_slice.view(batch_size, codebook_dimension)
        
        if temp_mask_buffer is not None:
            # Use pre-allocated buffer for memory efficiency
            torch.eq(frame_view, unknown_token, out=temp_mask_buffer)
            frame_view.copy_(torch.where(temp_mask_buffer, next_token, frame_view))
        else:
            # Fallback to temporary allocation
            unknown_mask = (frame_view == unknown_token)  # [B, codebook_dimension]
            frame_view.copy_(torch.where(unknown_mask, next_token, frame_view))


def fused_parameter_updates(
    inference_params: InferenceParams,
    remaining_steps: torch.Tensor,
    step_idx: int,
    cpu_step_counter: int = None
) -> bool:
    """
    Efficiently update inference parameters with early termination check.
    
    Combines multiple parameter updates and early termination check to reduce 
    kernel launches and improve efficiency. Minimizes CPU-GPU synchronization 
    by using CPU-side counters for most checks.
    
    Args:
        inference_params (InferenceParams): Inference state containing sequence offsets
        remaining_steps (torch.Tensor): Per-batch remaining step counters [batch]
        step_idx (int): Current step index in the generation loop
        cpu_step_counter (int, optional): CPU-side step counter for estimation
        
    Returns:
        bool: True if generation should terminate early, False to continue
        
    Notes:
        - Updates seqlen_offset, lengths_per_sample, and remaining_steps in-place
        - Uses adaptive checking frequency to reduce GPU synchronization
        - Checks every 16 steps by default, every 8 steps with CPU estimation
        - Early termination occurs when all sequences have completed
    """
    # Update parameters in-place
    inference_params.seqlen_offset += 1
    inference_params.lengths_per_sample.add_(1)
    remaining_steps.sub_(1)
    
    # Adaptive checking to reduce CPU-GPU synchronization
    should_check_16 = (step_idx % 16 == 15)
    should_check_8 = (step_idx % 8 == 7)
    
    if should_check_16:
        # Full check every 16 steps
        if (remaining_steps <= 0).all():
            return True
    elif should_check_8 and cpu_step_counter is not None:
        # Estimated check every 8 steps when CPU counter available
        estimated_remaining = max(0, len(remaining_steps) * 10 - cpu_step_counter)
        if estimated_remaining < 5:
            # Only perform expensive GPU sync when estimate suggests we're close
            if (remaining_steps <= 0).all():
                return True
    
    return False


def create_boolean_workspace(batch_size: int, device: torch.device, codebook_dimension: int) -> torch.Tensor:
    """
    Create pre-allocated boolean workspace for mask operations.
    
    Creates a contiguous memory workspace for various boolean mask operations
    to avoid repeated allocations during generation loops.
    
    Args:
        batch_size (int): Number of sequences being processed
        device (torch.device): Target device for tensor allocation
        codebook_dimension (int): Number of codebooks in the autoencoder
        
    Returns:
        torch.Tensor: Boolean workspace of shape [batch, codebook_dimension, 3] for mask operations
        
    Notes:
        - Dimension 2 (size 3) provides space for different mask types:
          - Index 0: Conditional mask 1
          - Index 1: Conditional mask 2  
          - Index 2: Temporary operations buffer
        - Contiguous allocation improves memory access patterns
    """
    return torch.zeros((batch_size, codebook_dimension, 3), dtype=torch.bool, device=device).contiguous()


def create_comparison_workspace(batch_size: int, device: torch.device, codebook_dimension: int) -> torch.Tensor:
    """
    Create pre-allocated comparison workspace for tensor operations.
    
    Creates a workspace for comparison operations that need temporary storage
    for multiple comparison results.
    
    Args:
        batch_size (int): Number of sequences being processed
        device (torch.device): Target device for tensor allocation
        codebook_dimension (int): Number of codebooks in the autoencoder
        
    Returns:
        torch.Tensor: Comparison workspace of shape [3, batch, codebook_dimension]
        
    Notes:
        - Dimension 0 (size 3) provides space for different comparison types
        - Optimized memory layout for efficient tensor operations
    """
    return torch.zeros((3, batch_size, codebook_dimension), dtype=torch.bool, device=device).contiguous()


def apply_eos_masking(
    next_token: torch.Tensor,
    stopping: torch.Tensor,
    eos_idx_buffer: torch.Tensor,
    cb_mask_expanded: torch.Tensor,
    comparison_workspace: torch.Tensor,
    mask_cond_1_buffer: torch.Tensor,
    mask_cond_2_buffer: torch.Tensor,
    masked_token_id: int,
    eos_token_id: int,
    codebook_dimension: int
) -> torch.Tensor:
    """
    Apply end-of-sequence masking with pre-allocated buffers.
    
    Efficiently applies EOS masking logic using pre-allocated workspace tensors
    to avoid memory allocations in the generation loop.
    
    Args:
        next_token (torch.Tensor): Next tokens to potentially mask [batch, codebooks]
        stopping (torch.Tensor): Per-batch stopping flags [batch]
        eos_idx_buffer (torch.Tensor): EOS position indices [batch]
        cb_mask_expanded (torch.Tensor): Codebook mask for comparisons [batch, codebooks]
        comparison_workspace (torch.Tensor): Pre-allocated comparison space [3, batch, codebooks]
        mask_cond_1_buffer (torch.Tensor): Buffer for condition 1 mask [batch, codebooks]
        mask_cond_2_buffer (torch.Tensor): Buffer for condition 2 mask [batch, codebooks]
        masked_token_id (int): Token ID for masked positions
        eos_token_id (int): Token ID for end-of-sequence
        codebook_dimension (int): Number of codebooks in the autoencoder
        
    Returns:
        torch.Tensor: Masked next tokens [batch, codebooks]
        
    Notes:
        - Uses pre-allocated buffers to avoid memory allocations
        - Implements complex EOS masking logic with fused operations
        - Handles both before-EOS and at-EOS position masking
    """
    eos_expanded = eos_idx_buffer.unsqueeze(1)  # [B, 1]
    stopping_expanded = stopping.unsqueeze(1)   # [B, 1]
    
    # Use comparison workspace views
    torch.eq(cb_mask_expanded, eos_expanded, out=comparison_workspace[0])
    torch.lt(cb_mask_expanded, eos_expanded, out=comparison_workspace[1])
    comparison_workspace[2] = stopping_expanded.expand(-1, codebook_dimension)
    
    eos_pos_view = comparison_workspace[0]
    before_eos_view = comparison_workspace[1] 
    stop_mask_view = comparison_workspace[2]
    
    # Apply masking conditions using pre-allocated buffers
    torch.logical_and(stop_mask_view, before_eos_view, out=mask_cond_1_buffer)
    torch.logical_and(stop_mask_view, eos_pos_view, out=mask_cond_2_buffer)
    
    # Apply the masking
    return torch.where(mask_cond_1_buffer, masked_token_id, 
                      torch.where(mask_cond_2_buffer, eos_token_id, next_token))
