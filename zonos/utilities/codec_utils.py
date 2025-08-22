"""
Codec utilities for audio token processing and neural network operations.

This module provides utilities for processing audio codec tokens, including
embedding operations, output head computations, and codebook pattern manipulation
used in autoregressive audio generation.
"""

import torch
import torch.nn as nn
from zonos.codebook_pattern import apply_delay_pattern, revert_delay_pattern
from zonos.utilities.utils import pad_weight_


def embed_codes_static(
    codes: torch.Tensor,
    embeddings: nn.ModuleList
) -> torch.Tensor:
    """
    Convert discrete codes to embeddings using provided embedding layers.
    
    Sums embeddings from all codebooks to create a unified representation
    suitable for processing by the backbone model.
    
    Args:
        codes (torch.Tensor): Discrete codes of shape [batch, num_codebooks, seq_len]
        embeddings (nn.ModuleList): Embedding layers for each codebook
        
    Returns:
        torch.Tensor: Embedded codes of shape [batch, seq_len, d_model]
        
    Notes:
        - Each codebook gets its own embedding layer to capture unique patterns
        - Final representation is the sum across all codebook embeddings
        - Optimized for batch processing with efficient tensor operations
    """
    return sum(emb(codes[:, i]) for i, emb in enumerate(embeddings))


def apply_heads_static(
    hidden_states: torch.Tensor,
    fused_heads: nn.Linear,
    num_codebooks: int
) -> torch.Tensor:
    """
    Apply output heads to compute logits for each codebook position.
    
    Transforms hidden states into per-codebook logits using a fused linear layer,
    then reshapes and transposes for proper codebook alignment.
    
    Args:
        hidden_states (torch.Tensor): Hidden states of shape [batch, seq_len, d_model]
        fused_heads (nn.Linear): Fused linear layer for all codebooks
        num_codebooks (int): Number of codebooks in the autoencoder
        
    Returns:
        torch.Tensor: Logits of shape [batch, num_codebooks, seq_len, vocab_size]
        
    Raises:
        RuntimeError: If fused head output size is not divisible by num_codebooks
        
    Notes:
        - Uses fused linear layer to reduce kernel launches and improve performance
        - Reshapes output to separate logits for each codebook
        - Transpose operation aligns dimensions for downstream processing
    """
    # hidden_states: [B, seq_len, D]; fused linear -> [B, seq_len, num_codebooks*vocab]
    out = fused_heads(hidden_states)
    B, seq_len, prod = out.shape
    
    if prod % num_codebooks != 0:
        # Should not happen since we avoided padding fused head
        raise RuntimeError(
            f"Fused head output size {prod} not divisible by num_codebooks={num_codebooks}."
        )
    
    vocab = prod // num_codebooks
    out = out.view(B, seq_len, num_codebooks, vocab).transpose(1, 2)  # [B, num_codebooks, seq_len, vocab]
    return out


def prepare_codec_input(
    codes: torch.Tensor,
    masked_token_id: int
) -> torch.Tensor:
    """
    Prepare codec input by applying delay pattern for autoregressive generation.
    
    Applies delay pattern to input codes to create the proper causal structure
    for autoregressive audio token generation.
    
    Args:
        codes (torch.Tensor): Input codes of shape [batch, num_codebooks, seq_len]
        masked_token_id (int): Token ID used for masked/unknown positions
        
    Returns:
        torch.Tensor: Delayed codes ready for autoregressive processing
        
    Notes:
        - Delay pattern ensures proper causal dependencies between codebooks
        - Masked tokens are used to indicate positions to be predicted
        - Essential for maintaining autoregressive generation properties
    """
    return apply_delay_pattern(codes, masked_token_id)


def finalize_codec_output(delayed_codes: torch.Tensor) -> torch.Tensor:
    """
    Finalize codec output by reverting delay pattern.
    
    Reverts the delay pattern applied during generation to restore the
    original codebook alignment and produce final output codes.
    
    Args:
        delayed_codes (torch.Tensor): Delayed codes from generation process
        
    Returns:
        torch.Tensor: Final output codes with delay pattern reverted
        
    Notes:
        - Removes delay pattern applied during autoregressive generation
        - Restores proper temporal alignment across all codebooks
        - Final step in the audio token generation pipeline
    """
    return revert_delay_pattern(delayed_codes)


def pad_embeddings_and_heads(
    embeddings: nn.ModuleList,
    heads: nn.ModuleList = None,
    fused_heads: nn.Linear = None,
    target_multiple: int = None
) -> None:
    """
    Pad embedding and head weights for vocabulary alignment.
    
    Pads embedding and output head weights to align vocabulary size with
    specified multiples for optimal memory access and hardware utilization.
    
    Args:
        embeddings (nn.ModuleList): Embedding layers to pad
        heads (nn.ModuleList, optional): Individual head layers to pad
        fused_heads (nn.Linear, optional): Fused head layer to pad
        target_multiple (int, optional): Target multiple for padding
        
    Notes:
        - Only pads if target_multiple is specified and non-zero
        - Improves memory alignment and hardware utilization
        - Applied automatically during model loading via state dict hooks
        - Uses utility function pad_weight_ for actual padding operation
    """
    if not target_multiple:
        return
        
    # Pad embedding layers
    for emb in embeddings:
        pad_weight_(emb, target_multiple)
    
    # Pad individual head layers if present
    if heads is not None:
        for head in heads:
            pad_weight_(head, target_multiple)
    
    # Note: Fused heads padding would require more complex logic
    # and is typically handled during model initialization


class CodebookEmbedder(nn.Module):
    """
    Efficient codebook embedding layer with support for multiple codebooks.
    
    Provides embedding functionality for multi-codebook audio tokens with
    optional weight padding for hardware optimization.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_codebooks: int,
        pad_vocab_to_multiple_of: int = None
    ):
        """
        Initialize codebook embedder.
        
        Args:
            vocab_size (int): Vocabulary size for each codebook
            d_model (int): Model dimension
            num_codebooks (int): Number of codebooks
            pad_vocab_to_multiple_of (int, optional): Pad vocabulary to this multiple
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.pad_vocab_to_multiple_of = pad_vocab_to_multiple_of
        
        # Create embedding layers for each codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) 
            for _ in range(num_codebooks)
        ])
        
        # Apply padding if specified
        if pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_hook)
    
    def _pad_embeddings_hook(self, *args, **kwargs):
        """Hook to pad embeddings after loading state dict."""
        pad_embeddings_and_heads(
            embeddings=self.embeddings,
            target_multiple=self.pad_vocab_to_multiple_of
        )
    
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Embed input codes.
        
        Args:
            codes (torch.Tensor): Input codes [batch, num_codebooks, seq_len]
            
        Returns:
            torch.Tensor: Embedded codes [batch, seq_len, d_model]
        """
        return embed_codes_static(codes, self.embeddings)


class CodebookHead(nn.Module):
    """
    Efficient codebook output head with fused linear operations.
    
    Provides output head functionality for multi-codebook audio token prediction
    with fused linear operations for improved performance.
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        num_codebooks: int,
        bias: bool = False
    ):
        """
        Initialize codebook head.
        
        Args:
            d_model (int): Model dimension
            vocab_size (int): Vocabulary size for each codebook
            num_codebooks (int): Number of codebooks
            bias (bool): Whether to use bias in linear layer
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        
        # Fused linear layer for all codebooks
        self.fused_heads = nn.Linear(
            d_model, 
            num_codebooks * vocab_size, 
            bias=bias
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply output heads to compute logits.
        
        Args:
            hidden_states (torch.Tensor): Hidden states [batch, seq_len, d_model]
            
        Returns:
            torch.Tensor: Logits [batch, num_codebooks, seq_len, vocab_size]
        """
        return apply_heads_static(
            hidden_states, 
            self.fused_heads, 
            self.num_codebooks
        )
