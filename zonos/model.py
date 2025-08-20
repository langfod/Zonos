import json
import logging
import hashlib
from typing import Callable

import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from zonos.autoencoder import DACAutoencoder, preload_dac_autoencoder
from zonos.backbone import BACKBONES
from zonos.codebook_pattern import apply_delay_pattern, revert_delay_pattern
from zonos.conditioning import PrefixConditioner
from zonos.config import InferenceParams, ZonosConfig
from zonos.sampling import sample_from_logits
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.utils import DEFAULT_DEVICE, find_multiple, pad_weight_

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id
        self.autoencoder: DACAutoencoder = preload_dac_autoencoder(device=DEFAULT_DEVICE, warmup=True)
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None
        self._persistent_spk_model = None
        # Pad embedding vocab size to multiple of 8 for better memory alignment
        # 1024 (DAC vocab) + 1 (EOS) + 1 (MASK) = 1026.
        vocab_size = find_multiple(1026, 8)  # 1026 -> 1032
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, dim) for _ in range(self.autoencoder.num_codebooks)])

        # Fused projection replacing per-codebook heads: a single Linear whose weight rows are the old heads concatenated.
        # This reduces kernel launches while keeping identical math/output ordering.
        self.fused_heads = nn.Linear(dim, self.autoencoder.num_codebooks * 1025, bias=False)

        # CUDA graph related state
        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if getattr(config, "pad_vocab_to_multiple_of", None):
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    # ---------------------------- Utility & Loading ----------------------------
    def _pad_embeddings_and_heads(self, *args, **kwargs):  # pragma: no cover (hook)
        target_multiple = self.config.pad_vocab_to_multiple_of
        if not target_multiple:
            return
        # Support legacy checkpoints that may still have self.heads
        head_modules = []
        if hasattr(self, "fused_heads"):
            head_modules.append(self.fused_heads)
        elif hasattr(self, "heads"):
            head_modules.extend(self.heads)
        for w in self.embeddings:
            pad_weight_(w, target_multiple)
        # Skip padding fused_heads to keep output divisible by num_codebooks*1025
        # (legacy per-codebook heads could be padded individually without affecting divisibility)
        if hasattr(self, "heads"):
            for w in self.heads:
                pad_weight_(w, target_multiple)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision, local_dir_use_symlinks=False)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision, local_dir_use_symlinks=False)
        return cls.from_local(config_path, model_path, device, **kwargs)

    @classmethod
    def from_local(
        cls, config_path: str, model_path: str, device: str = DEFAULT_DEVICE, backbone: str | None = None
    ) -> "Zonos":
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
                # Handle embedding size mismatch due to vocab padding
                if k.startswith("embeddings.") and k.endswith(".weight"):
                    current_shape = sd[k].shape
                    loaded_shape = tensor.shape
                    if current_shape[0] != loaded_shape[0] and current_shape[1] == loaded_shape[1]:
                        # Pad the embedding weight from old vocab size to new vocab size
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
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, 1, D]; fused linear -> [B,1,num_codebooks*V]
        out = self.fused_heads(hidden_states)
        B, one, prod = out.shape
        num_codebooks = self.autoencoder.num_codebooks
        if prod % num_codebooks != 0:
            # Should not happen since we avoided padding fused head
            raise RuntimeError(
                f"Fused head output size {prod} not divisible by num_codebooks={num_codebooks}."
            )
        vocab = prod // num_codebooks
        out = out.view(B, one, num_codebooks, vocab).transpose(1, 2)  # [B, num_codebooks, 1, vocab]
        return out

    # Legacy checkpoint conversion: map heads.* weights to fused_heads.weight
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Detect old per-head weights
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

    # ---------------------------- Speaker Embedding ----------------------------
    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA()
        _, spk_embedding = self.spk_clone_model(wav.to(self.spk_clone_model.device), sr)
        return spk_embedding.unsqueeze(0).bfloat16()

    # ---------------------------- Conditioning (with optional cache) -----------
    def _create_conditioning_cache_key(self, cond_dict: dict, uncond_dict: dict | None) -> str:
        def value_to_key(value):
            if value is None:
                return "None"
            if isinstance(value, torch.Tensor):
                return f"tensor_{tuple(value.shape)}_{value.dtype}_{hash(value.cpu().float().numpy().tobytes())}"
            if isinstance(value, (int, float, str, bool)):
                return str(value)
            if isinstance(value, (list, tuple)):
                return f"list_{[value_to_key(v) for v in value]}"
            return f"other_{type(value).__name__}_{value}"
        cond_items = sorted((k, value_to_key(v)) for k, v in cond_dict.items())
        uncond_items = None if uncond_dict is None else sorted((k, value_to_key(v)) for k, v in uncond_dict.items())
        cache_string = f"cond:{cond_items}_uncond:{uncond_items}"
        return hashlib.sha512(cache_string.encode()).hexdigest()

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None, use_cache: bool = False, cfg_scale: float = 1.0) -> torch.Tensor:  # type: ignore[override]
        if not use_cache:
            if cfg_scale == 1.0:
                return self.prefix_conditioner(cond_dict)
            if uncond_dict is None:
                uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
            return torch.cat([self.prefix_conditioner(cond_dict), self.prefix_conditioner(uncond_dict)])
        cache_key = self._create_conditioning_cache_key(cond_dict, uncond_dict)
        if hasattr(self, "_conditioning_cache") and cache_key in self._conditioning_cache:
            return self._conditioning_cache[cache_key].clone()
        if cfg_scale == 1.0:
            conditioning = self.prefix_conditioner(cond_dict)
        else:
            if uncond_dict is None:
                uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
            conditioning = torch.cat([self.prefix_conditioner(cond_dict), self.prefix_conditioner(uncond_dict)])
        if not hasattr(self, "_conditioning_cache"):
            self._conditioning_cache = {}
        if len(self._conditioning_cache) >= 32:
            oldest = next(iter(self._conditioning_cache))
            del self._conditioning_cache[oldest]
        self._conditioning_cache[cache_key] = conditioning.clone()
        return conditioning

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True,
    ) -> torch.Tensor:
        """Single-step decode with optional CUDA Graph capture."""
        if cfg_scale == 1.0:
            hidden_states = self.embed_codes(input_ids)
            return self._compute_logits(hidden_states, inference_params, cfg_scale)

        bsz = input_ids.size(0)
        if not allow_cudagraphs or input_ids.device.type != "cuda":
            hidden_states_local = self.embed_codes(input_ids)
            hidden_states_local = hidden_states_local.repeat(2, 1, 1)
            return self._compute_logits(hidden_states_local, inference_params, cfg_scale)

        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz)
        if need_capture:
            self._cg_graph = None
            self._cg_batch_size = bsz
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale

            for _ in range(3):
                hidden_states = self.embed_codes(input_ids)
                hidden_states = hidden_states.repeat(2, 1, 1)
                logits = self._compute_logits(hidden_states, inference_params, cfg_scale)

            self._cg_input_ids = input_ids.clone()
            self._cg_logits = torch.empty_like(logits)

            g = torch.cuda.CUDAGraph()
            def capture_region():
                hidden_states_local = self.embed_codes(self._cg_input_ids)
                hidden_states_local = hidden_states_local.repeat(2, 1, 1)
                self._cg_logits = self._compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale)
            with torch.cuda.graph(g):
                capture_region()
            self._cg_graph = g
        else:
            self._cg_input_ids.copy_(input_ids)

        self._cg_graph.replay()
        return self._cg_logits

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
        # Replicate input_ids if CFG is enabled
        if cfg_scale != 1.0:
            input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([prefix_hidden_states, self.embed_codes(input_ids)], dim=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16) -> InferenceParams:
        max_seqlen = find_multiple(max_seqlen, 8)
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    # (Original simple prepare_conditioning replaced by cached variant above)

    def can_use_cudagraphs(self) -> bool:
        # Only the mamba-ssm backbone supports CUDA Graphs at the moment
        return self.device.type == "cuda" and "_mamba_ssm" in str(self.backbone.__class__)

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

        delayed_codes = apply_delay_pattern(codes, self.masked_token_id)
        delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

        # Mask-based implementation (stable baseline)
        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
        next_token = sample_from_logits(logits, **sampling_params).squeeze(-1)  # [B,9]
        offset = delayed_prefix_audio_codes.shape[2]
        frame = delayed_codes[..., offset: offset + 1]
        frame.masked_scatter_(frame == unknown_token, next_token.unsqueeze(-1))

        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample[:] += prefix_length

        logit_bias = torch.zeros_like(logits)
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf
        #logit_bias[:, 0, self.eos_token_id] -= torch.log(torch.tensor(2.0, device=logits.device)) # Make EOS less likely because audio often is cut off

        stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_steps = delayed_codes.shape[2] - offset
        remaining_steps = torch.full((batch_size,), max_steps, device=device)
        cfg_scale_tensor = torch.tensor(cfg_scale)

        step = 0
        nine_const = torch.tensor(9, device=device)
        arange_codebooks = torch.arange(9, device=device)
        while torch.max(remaining_steps) > 0:
            offset += 1
            if offset >= delayed_codes.shape[2]:
                break
            input_ids = delayed_codes[..., offset - 1: offset]
            logits = decode_one_token(input_ids, inference_params, cfg_scale_tensor, allow_cudagraphs=cg)
            logits += logit_bias
            next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params).squeeze(-1)

            eos_in_cb0 = (next_token[:, 0] == self.eos_token_id)
            if eos_in_cb0.any():
                remaining_steps[eos_in_cb0] = torch.minimum(remaining_steps[eos_in_cb0], nine_const)
            stopping |= eos_in_cb0
            eos_codebook_idx = 9 - remaining_steps
            eos_codebook_idx.clamp_(max=8)

            stop_rows = torch.nonzero(stopping, as_tuple=False).flatten()
            if stop_rows.numel():
                idxs = eos_codebook_idx[stop_rows]
                cb_mask = arange_codebooks.unsqueeze(0)
                eos_pos = (cb_mask == idxs.unsqueeze(1))
                before_eos = (cb_mask < idxs.unsqueeze(1))
                row_tokens = next_token[stop_rows]
                row_tokens = torch.where(before_eos, self.masked_token_id, row_tokens)
                row_tokens = torch.where(eos_pos, self.eos_token_id, row_tokens)
                next_token[stop_rows] = row_tokens

            if offset < delayed_codes.shape[2]:
                frame = delayed_codes[..., offset: offset + 1]
                if frame.numel() > 0:
                    mask = (frame == unknown_token).view(batch_size, 9)
                    if mask.any():
                        frame.view(batch_size, 9)[mask] = next_token[mask]

            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample.add_(1)
            remaining_steps.sub_(1)
            step += 1
            if callback is not None and not callback(frame if frame.numel() > 0 else torch.empty(0), step, max_steps):
                break

        out_codes = revert_delay_pattern(delayed_codes)
        out_codes.masked_fill_(out_codes >= 1024, 0)
        out_codes = out_codes[..., : offset - 9]

        actual_new_tokens = offset - (prefix_audio_len + 1)
        logging.info(f"Max Token: {max_new_tokens}. Actual decoded tokens (excluding prompt): {actual_new_tokens}")

        self._cg_graph = None  # reset cuda graph to avoid cache changes
        return out_codes
