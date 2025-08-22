import json
import logging
from typing import Callable

import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from zonos.autoencoder import DACAutoencoder, preload_dac_autoencoder
from zonos.backbone import BACKBONES
from zonos.conditioning import PrefixConditioner
from zonos.config import InferenceParams, ZonosConfig
from zonos.sampling import sample_from_logits
from zonos.utilities.utils import DEFAULT_DEVICE, find_multiple, pad_weight_
from zonos.utilities.conditioning_cache import ConditioningCache, prepare_conditioning_with_cache
from zonos.utilities.tensor_ops import (
    fused_frame_update, 
    fused_parameter_updates, 
    create_boolean_workspace,
    create_comparison_workspace,
    apply_eos_masking
)
from zonos.utilities.codec_utils import (
    embed_codes_static,
    apply_heads_static,
    prepare_codec_input,
    finalize_codec_output,
    pad_embeddings_and_heads
)
from zonos.utilities.generation_utils import (
    CUDAGraphManager,
    decode_one_token_static,
    prefill_static,
    setup_inference_cache_static,
    can_use_cudagraphs_static,
    GenerationState
)

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))


class Zonos(nn.Module):
    """
    The main Zonos text-to-speech model.
    
    Zonos combines a neural audio codec (DAC) with a powerful language model backbone
    to generate high-quality speech from text and conditioning inputs. It supports both
    transformer and hybrid transformer-mamba architectures with advanced optimization
    techniques like CUDA graphs and torch.compile.
    
    The model uses a prefix conditioning approach where various inputs (text, speaker,
    acoustic parameters) are processed into conditioning embeddings that guide the
    autoregressive generation of audio codebook tokens.
    
    Args:
        config (ZonosConfig): Model configuration containing backbone and conditioning parameters
        backbone_cls: Backbone architecture class (defaults to available implementation)
        
    Attributes:
        config (ZonosConfig): Model configuration  
        autoencoder (DACAutoencoder): Audio codec for encoding/decoding
        backbone: Neural network backbone (transformer or mamba)
        prefix_conditioner (PrefixConditioner): Conditioning input processor
        embeddings (nn.ModuleList): Embedding layers for each codebook
        fused_heads (nn.Linear): Output projection layer for all codebooks
    """
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id
        self.autoencoder: DACAutoencoder = preload_dac_autoencoder(device=DEFAULT_DEVICE, warmup=True)
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self._conditioning_cache = ConditioningCache(max_size=32)
        self._persistent_spk_model = None
        # 1024 (DAC vocab) + 1 (EOS) + 1 (MASK) = 1026.
        vocab_size = find_multiple(1026, 8)  # 1026 -> 1032
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.fused_heads = nn.Linear(dim, self.autoencoder.num_codebooks * 1025, bias=False)
        self._cuda_graph_manager = CUDAGraphManager()

        if getattr(config, "pad_vocab_to_multiple_of", None):
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    # ---------------------------- Utility & Loading ----------------------------
    def _pad_embeddings_and_heads(self, *args, **kwargs):  # pragma: no cover (hook)
        """Pad embedding and head weights for vocabulary alignment."""
        pad_embeddings_and_heads(
            embeddings=self.embeddings,
            fused_heads=self.fused_heads if hasattr(self, 'fused_heads') else None,
            heads=self.heads if hasattr(self, 'heads') else None,
            target_multiple=self.config.pad_vocab_to_multiple_of
        )

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        """
        Load a pretrained Zonos model from Hugging Face Hub.
        
        Args:
            repo_id (str): Hugging Face repository ID (e.g., 'username/model-name')
            revision (str, optional): Git revision (branch, tag, or commit hash)
            device (str): Target device for model placement
            **kwargs: Additional arguments passed to from_local
            
        Returns:
            Zonos: Loaded and initialized model instance
            
        Notes:
            - Downloads config.json and model.safetensors from the repository
            - Automatically selects appropriate backbone architecture
            - Model is placed on the specified device with appropriate dtype
        """
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision, local_dir_use_symlinks=False)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision, local_dir_use_symlinks=False)
        return cls.from_local(config_path, model_path, device, **kwargs)

    @classmethod
    def from_local(
        cls, config_path: str, model_path: str, device: str = DEFAULT_DEVICE, backbone: str | None = None
    ) -> "Zonos":
        """
        Load a Zonos model from local configuration and weight files.
        
        Args:
            config_path (str): Path to config.json file
            model_path (str): Path to model.safetensors file  
            device (str): Target device for model placement
            backbone (str, optional): Force specific backbone type ('torch', 'mamba_ssm')
            
        Returns:
            Zonos: Loaded and initialized model instance
            
        Notes:
            - Automatically detects transformer vs hybrid architecture from config
            - Handles embedding weight padding for vocabulary alignment
            - Loads model in bfloat16 precision for efficiency
            - Ensures DAC autoencoder is placed on the same device
        """
        config = ZonosConfig.from_dict(json.load(open(config_path)))
        if backbone:
            backbone_cls = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            backbone_cls = DEFAULT_BACKBONE_CLS
            if is_transformer and "torch" in BACKBONES:
                backbone_cls = BACKBONES["torch"]
        model = cls(config, backbone_cls).to(device, torch.bfloat16)
        model.autoencoder.dac.to(device)
        sd = model.state_dict()
        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                tensor = f.get_tensor(k)
                if k.startswith("embeddings.") and k.endswith(".weight"):
                    current_shape = sd[k].shape
                    loaded_shape = tensor.shape
                    if current_shape[0] != loaded_shape[0] and current_shape[1] == loaded_shape[1]:
                        padded_tensor = torch.zeros(current_shape, dtype=tensor.dtype, device=tensor.device)
                        padded_tensor[:loaded_shape[0]] = tensor
                        sd[k] = padded_tensor
                    else:
                        sd[k] = tensor
                else:
                    sd[k] = tensor
        model.load_state_dict(sd)
        return model

    # ---------------------------- Embedding / Heads ----------------------------
    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete codes to embeddings.
        
        Sums embeddings from all codebooks to create a unified representation
        suitable for processing by the backbone model.
        
        Args:
            codes (torch.Tensor): Discrete codes of shape [batch, num_codebooks, seq_len]
            
        Returns:
            torch.Tensor: Embedded codes of shape [batch, seq_len, d_model]
        """
        return embed_codes_static(codes, self.embeddings)

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply output heads to compute logits for each codebook position.
        
        Transforms hidden states into per-codebook logits using a fused linear layer.
        
        Args:
            hidden_states (torch.Tensor): Hidden states of shape [batch, seq_len, d_model]
            
        Returns:
            torch.Tensor: Logits of shape [batch, num_codebooks, seq_len, vocab_size]
        """
        return apply_heads_static(hidden_states, self.fused_heads, self.autoencoder.num_codebooks)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        first_key = f"{prefix}heads.0.weight"
        if first_key in state_dict:
            weights = []
            i = 0
            while f"{prefix}heads.{i}.weight" in state_dict:
                weights.append(state_dict.pop(f"{prefix}heads.{i}.weight"))
                i += 1
            if weights:
                fused_weight = torch.cat(weights, dim=0)
                state_dict[f"{prefix}fused_heads.weight"] = fused_weight
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        logits = self.apply_heads(last_hidden_states).squeeze(2).float()
        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        logits[..., 1025:].fill_(-torch.inf)
        return logits

    # ---------------------------- Conditioning (with optional cache) -----------
    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None, use_cache: bool = False, cfg_scale: float = 1.0) -> torch.Tensor:  # type: ignore[override]
        """
        Prepare conditioning embeddings with optional caching.
        
        Processes conditioning dictionaries through the prefix conditioner with
        optional caching to avoid recomputation of identical inputs.
        
        Args:
            cond_dict (dict): Primary conditioning inputs
            uncond_dict (dict, optional): Unconditional inputs for classifier-free guidance
            use_cache (bool): Whether to use caching (default: False)
            cfg_scale (float): Classifier-free guidance scale (default: 1.0)
            
        Returns:
            torch.Tensor: Processed conditioning embeddings
            
        Notes:
            - Uses centralized conditioning cache utilities for consistency
            - Cache is automatically managed with LRU eviction
            - For CFG, concatenates conditional and unconditional embeddings
        """
        return prepare_conditioning_with_cache(
            self.prefix_conditioner,
            cond_dict=cond_dict,
            uncond_dict=uncond_dict,
            use_cache=use_cache,
            cfg_scale=cfg_scale,
            cache=self._conditioning_cache if use_cache else None
        )

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True,
    ) -> torch.Tensor:
        """Single-step decode with optional CUDA Graph capture."""
        return decode_one_token_static(
            input_ids=input_ids,
            inference_params=inference_params,
            cfg_scale=cfg_scale,
            embed_codes_fn=self.embed_codes,
            compute_logits_fn=self._compute_logits,
            cuda_graph_manager=self._cuda_graph_manager,
            allow_cudagraphs=allow_cudagraphs
        )

    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        return prefill_static(
            prefix_hidden_states=prefix_hidden_states,
            input_ids=input_ids,
            inference_params=inference_params,
            cfg_scale=cfg_scale,
            embed_codes_fn=self.embed_codes,
            compute_logits_fn=self._compute_logits
        )

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16) -> InferenceParams:
        """
        Set up inference cache for efficient generation.
        
        Creates key-value caches and other state needed for efficient autoregressive
        generation without recomputing past states.
        
        Args:
            batch_size (int): Batch size for generation
            max_seqlen (int): Maximum sequence length to allocate for
            dtype (torch.dtype): Data type for cache tensors
            
        Returns:
            InferenceParams: Configured inference parameters with allocated caches
            
        Notes:
            - Sequence length is automatically padded to multiple of 8
            - Cache allocation is backend-specific (attention vs mamba)
            - Length tracking is initialized to zero for all samples
        """
        return setup_inference_cache_static(
            backbone_allocate_fn=lambda bs, ml, dt, dev: self._allocate_cache_helper(bs, ml, dt),
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            dtype=dtype,
            device=self.device
        )
    
    def _allocate_cache_helper(self, batch_size: int, max_seqlen: int, dtype: torch.dtype) -> InferenceParams:
        """Helper method to maintain cache allocation logic."""
        max_seqlen = find_multiple(max_seqlen, 8)
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def can_use_cudagraphs(self) -> bool:
        """
        Check if CUDA graphs can be used for optimization.
        
        Returns:
            bool: True if CUDA graphs are supported and recommended
            
        Notes:
            - Currently only mamba-ssm backbone supports CUDA graphs
            - Requires CUDA device 
            - CUDA graphs provide significant speedup for inference
        """
        return can_use_cudagraphs_static(self.device, str(self.backbone.__class__))

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor = None,  # [bsz, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: dict = dict(min_p=0.1),
        disable_torch_compile: bool = False,
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        """
        Generate audio codes autoregressively from conditioning inputs.
        
        This is the main generation method that produces discrete audio codes which can
        be decoded to audio using the DAC autoencoder. It supports advanced optimizations
        like CUDA graphs, classifier-free guidance, and efficient delay pattern handling.
        
        Args:
            prefix_conditioning (torch.Tensor): Conditioning embeddings of shape 
                [batch_size, cond_seq_len, d_model] containing text, speaker, and 
                acoustic parameter representations
            audio_prefix_codes (torch.Tensor, optional): Pre-existing audio codes to 
                continue from, shape [batch_size, 9, prefix_audio_seq_len]  
            max_new_tokens (int): Maximum number of new tokens to generate. 
                Default 86*30 ≈ 30 seconds at 44.1kHz
            cfg_scale (float): Classifier-free guidance scale. Higher values follow 
                conditioning more closely. Must be > 1.0
            batch_size (int): Batch size for generation
            sampling_params (dict): Sampling configuration (temperature, top_p, min_p, etc.)
            disable_torch_compile (bool): Whether to disable torch.compile optimizations
            callback (Callable, optional): Progress callback function that receives 
                (frame, step, max_steps) and returns bool to continue
                
        Returns:
            torch.Tensor: Generated audio codes of shape [batch_size, 9, generated_length]
            
        Notes:
            - Uses delay pattern for parallel codebook generation
            - Supports CUDA graphs on compatible hardware for maximum speed
            - Automatically handles EOS detection and sequence truncation
            - Applies various optimizations like fused operations and memory reuse
            - Generated codes are clamped to valid range [0, 1023]
        """
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
        device = self.device

        # Use CUDA Graphs if supported, and torch.compile otherwise.
        cg = self.can_use_cudagraphs()
        decode_one_token = self._decode_one_token
        decode_one_token = torch.compile(decode_one_token, dynamic=True, disable=cg or disable_torch_compile)

        unknown_token = -1
        audio_seq_len = prefix_audio_len + max_new_tokens
        seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

        with torch.device(device):
            inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
            codes = torch.full((batch_size, 9, audio_seq_len), unknown_token)

        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes

        delayed_codes = prepare_codec_input(codes, self.masked_token_id)
        delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
        next_token = sample_from_logits(logits, **sampling_params).squeeze(-1)  # [B,9]
        offset = delayed_prefix_audio_codes.shape[2]
        frame = delayed_codes[..., offset: offset + 1]

        unknown_mask = (frame == unknown_token)
        frame.copy_(torch.where(unknown_mask, next_token.unsqueeze(-1), frame))
        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample[:] += prefix_length

        logit_bias = torch.zeros_like(logits)
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf
        # Lower EOS bias reduces risk of abrupt cutoffs that can cause audio artifacts
        eos_log_factor = torch.log(torch.tensor(2.0, device=logits.device))
        logit_bias[:, 0, self.eos_token_id] -= eos_log_factor

        stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_steps = delayed_codes.shape[2] - offset
        remaining_steps = torch.full((batch_size,), max_steps, device=device)
        cfg_scale_tensor = torch.tensor(cfg_scale)

        step = 0
        nine_const = torch.tensor(9, device=device)
        arange_codebooks = torch.arange(9, device=device)
        
        cb_mask_expanded = arange_codebooks.unsqueeze(0).expand(batch_size, -1)
        
        # Use utility functions for workspace creation
        boolean_workspace = create_boolean_workspace(batch_size, device)
        comparison_workspace = create_comparison_workspace(batch_size, device)
        
        # Create views into the workspace for different mask types (eliminates separate allocations)
        mask_cond_1_buffer = boolean_workspace[..., 0]    # [B, 9]
        mask_cond_2_buffer = boolean_workspace[..., 1]    # [B, 9]
        temp_mask_buffer = boolean_workspace[..., 2]      # [B, 9] (for intermediate operations)
        
        temp_remaining_buffer = torch.zeros(batch_size, device=device)
        eos_idx_buffer = torch.zeros(batch_size, device=device)
        
        max_context_len = min(max_new_tokens, 100)
        
        cpu_step_counter = 0
        
        for step_idx in range(max_steps):
            offset += 1
            cpu_step_counter += 1
            
            if offset >= delayed_codes.shape[2]:
                break
                
            input_ids = delayed_codes[..., offset - 1: offset]
            logits = decode_one_token(input_ids, inference_params, cfg_scale_tensor, allow_cudagraphs=cg)
            logits += logit_bias
            
            context_start = max(0, offset - max_context_len)
            context_slice = delayed_codes[..., context_start:offset]
            
            next_token = sample_from_logits(logits, generated_tokens=context_slice, **sampling_params).squeeze(-1)

            eos_in_cb0 = (next_token[:, 0] == self.eos_token_id)

            torch.minimum(remaining_steps, nine_const, out=temp_remaining_buffer)
            remaining_steps.copy_(torch.where(eos_in_cb0, temp_remaining_buffer, remaining_steps))
            stopping.logical_or_(eos_in_cb0)
            
            torch.sub(nine_const, remaining_steps, out=eos_idx_buffer)
            eos_idx_buffer.clamp_(max=8)

            # Apply EOS masking using utility function
            next_token = apply_eos_masking(
                next_token, stopping, eos_idx_buffer, cb_mask_expanded,
                comparison_workspace, mask_cond_1_buffer, mask_cond_2_buffer,
                self.masked_token_id, self.eos_token_id
            )

            if offset < delayed_codes.shape[2]:
                fused_frame_update(delayed_codes, offset, batch_size, next_token, unknown_token, temp_mask_buffer)

            should_break = fused_parameter_updates(inference_params, remaining_steps, step_idx, cpu_step_counter)
            if should_break:
                break
            
            step = step_idx + 1
            
            if callback is not None and not callback(frame if 'frame' in locals() else torch.empty(0), step, max_steps):
                break

        out_codes = finalize_codec_output(delayed_codes)
        
        batch_size, num_codebooks, seq_len = out_codes.shape
        valid_length = offset - 9
        
        search_window = min(50, valid_length // 4)
        search_start = max(0, valid_length - search_window)
        
        if search_start < valid_length:
            eos_boundary = None
            for pos in range(search_start, valid_length):
                eos_count = (out_codes[:, :, pos] == self.eos_token_id).sum().item()
                if eos_count >= num_codebooks // 2:
                    eos_boundary = pos
                    break
            
            if eos_boundary is not None:
                valid_length = eos_boundary
                logging.info(f"Found EOS boundary at position {eos_boundary}, truncating sequence")

        invalid_mask = out_codes > 1024
        eos_mask = out_codes == 1024
        
        out_codes = torch.where(invalid_mask, 512, out_codes)
        out_codes = torch.where(eos_mask, 0, out_codes)
        
        final_codes = out_codes[..., :valid_length]
        
        final_codes = torch.clamp(final_codes, 0, 1023)

        actual_new_tokens = offset - (prefix_audio_len + 1)
        logging.info(f"Max Token: {max_new_tokens}. Actual decoded tokens (excluding prompt): {actual_new_tokens}")
        logging.info(f"Final offset: {offset}, prefix_audio_len: {prefix_audio_len}")
        logging.info(f"out_codes shape before slicing: {out_codes.shape}")
        logging.info(f"final_codes shape: {final_codes.shape}")

        self._cg_graph = None
        return final_codes
