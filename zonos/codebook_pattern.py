import torch
import torch.nn.functional as F


def apply_delay_pattern(codes: torch.Tensor, mask_token: int):
    """
    Apply a delay pattern to codebook sequences for parallel generation.
    
    This function implements a delay pattern that staggers the generation of different
    codebooks, allowing for causal generation while maintaining parallelism. Each
    codebook is delayed by an increasing amount, with masking tokens filling the gaps.
    
    Args:
        codes (torch.Tensor): Input codes tensor of shape [batch, num_codebooks, sequence_length]
        mask_token (int): Token ID used to mask future positions in the delayed pattern
        
    Returns:
        torch.Tensor: Delayed codes with shape [batch, num_codebooks, extended_sequence_length]
        
    Notes:
        - The output sequence length is extended by num_codebooks positions
        - Each codebook k is delayed by (k+1) positions and rolled accordingly  
        - This enables parallel generation while respecting causal dependencies
        - Used in conjunction with revert_delay_pattern for inference
    
    Example:
        Original:  [[1,2,3], [4,5,6], [7,8,9]]
        Delayed:   [[M,1,2,3,M,M], [M,M,4,5,6,M], [M,M,M,7,8,9]]
        where M is the mask_token
    """
    codes = F.pad(codes, (0, codes.shape[1]), value=mask_token)
    return torch.stack([codes[:, k].roll(k + 1) for k in range(codes.shape[1])], dim=1)


def revert_delay_pattern(codes: torch.Tensor):
    """
    Revert the delay pattern applied by apply_delay_pattern.
    
    This function reverses the delay pattern transformation, extracting the original
    sequential codes from the delayed representation. It removes the delays and
    padding introduced during the delay pattern application.
    
    Args:
        codes (torch.Tensor): Delayed codes tensor of shape [batch, num_codebooks, extended_sequence_length]
        
    Returns:
        torch.Tensor: Original codes with shape [batch, num_codebooks, original_sequence_length]
        
    Notes:
        - Extracts the valid portion of each delayed codebook sequence
        - Removes the artificial delays and padding added by apply_delay_pattern
        - The output sequence length is reduced by num_codebooks positions
        - Essential for converting generated delayed sequences back to standard format
        
    Example:
        Delayed:   [[M,1,2,3,M,M], [M,M,4,5,6,M], [M,M,M,7,8,9]]  
        Reverted:  [[1,2,3], [4,5,6], [7,8,9]]
        where M represents masked positions that are discarded
    """
    _, n_q, seq_len = codes.shape
    return torch.stack([codes[:, k, k + 1 : seq_len - n_q + k + 1] for k in range(n_q)], dim=1)
