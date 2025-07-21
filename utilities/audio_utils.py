"""
Audio processing utilities for Zonos application.
"""
import torch
import torchaudio
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np


# Global cache for audio prefixes
PREFIX_AUDIO_CACHE: Dict[str, torch.Tensor] = {}


def process_speaker_audio(speaker_audio_path: str, model, device: torch.device,
                         speaker_cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Process speaker audio and return speaker embedding.
    Uses caching to avoid recomputing embeddings for the same audio.
    """
    if speaker_audio_path not in speaker_cache:
        logging.info("Computing speaker embedding")
        wav, sr = torchaudio.load(speaker_audio_path)
        speaker_embedding = model.make_speaker_embedding(wav, sr)
        speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)
        speaker_cache[speaker_audio_path] = speaker_embedding
        return speaker_embedding
    else:
        logging.info("Reused cached speaker embedding")
        return speaker_cache[speaker_audio_path]


def process_prefix_audio(prefix_audio_path: str, model, device: torch.device) -> torch.Tensor:
    """
    Process prefix audio and return encoded audio codes.
    Uses global caching to avoid recomputing codes for the same audio.
    """
    global PREFIX_AUDIO_CACHE

    if prefix_audio_path in PREFIX_AUDIO_CACHE:
        logging.info("Using cached audio prefix.")
        return PREFIX_AUDIO_CACHE[prefix_audio_path]
    else:
        logging.info("Encoding and caching new audio prefix.")
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))
        PREFIX_AUDIO_CACHE[prefix_audio_path] = audio_prefix_codes
        return audio_prefix_codes


def convert_audio_to_numpy(wav_gpu_f32: torch.Tensor, sampling_rate: int) -> Tuple[int, np.ndarray]:
    """
    Convert GPU tensor audio to numpy array for output.
    """
    wav_gpu_i16 = (wav_gpu_f32.clamp(-1, 1) * 32767).to(torch.int16)
    wav_np = wav_gpu_i16.cpu().squeeze().numpy()
    return sampling_rate, wav_np
