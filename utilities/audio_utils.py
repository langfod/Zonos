"""
Audio processing utilities for Zonos application.
"""
import torch
import torchaudio
import logging
from typing import Dict, Any, Tuple, List
import numpy as np
from pathlib import Path


# Global cache for audio prefixes
PREFIX_AUDIO_CACHE: Dict[str, torch.Tensor] = {}


def process_speaker_audio(speaker_audio_path: str, model, device: torch.device,
                         speaker_cache: Dict[str, torch.Tensor], speaker_audio_uuid = None, model_choice: str = None) -> torch.Tensor:
    """
    Process speaker audio and return speaker embedding with enhanced caching.
    Uses both the old dict cache and the new disk/memory cache system.

    Args:
        speaker_audio_path: Path to the speaker audio file
        model: The Zonos model instance
        device: PyTorch device
        speaker_cache: Legacy cache dictionary for backward compatibility
        speaker_audio_uuid: C++ unsigned int UUID for unique identification
        model_choice: Full model choice string (e.g., "Zyphra/Zonos-v0.1-hybrid")
    """
    # Check old-style cache first for backward compatibility
    if speaker_audio_path in speaker_cache:
        logging.info("Reused cached speaker embedding (legacy cache)")
        return speaker_cache[speaker_audio_path]

    # Use the model's built-in caching if available
    logging.info("Computing speaker embedding with caching")
    wav, sr = torchaudio.load(speaker_audio_path)

    # Check if model supports caching (new SpeakerEmbeddingLDA)
    if hasattr(model, 'spk_clone_model') and hasattr(model.spk_clone_model, 'forward'):
        # Call the cached version with audio_path, UUID, and model_choice
        _, speaker_embedding = model.spk_clone_model(wav.to(model.spk_clone_model.device), sr,
                                                    audio_path=speaker_audio_path,
                                                    audio_uuid=speaker_audio_uuid,
                                                    model_choice=model_choice)
    else:
        # Fallback to regular method
        speaker_embedding = model.make_speaker_embedding(wav, sr)

    speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)

    # Store in legacy cache for backward compatibility
    speaker_cache[speaker_audio_path] = speaker_embedding

    return speaker_embedding


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


def configure_speaker_cache(cache_dir: str = None,
                          memory_cache_size: int = 100,
                          enable_disk_cache: bool = True,
                          cache_expiry_hours: float = 24 * 7) -> None:
    """
    Configure the global speaker embedding cache system.

    Args:
        cache_dir: Directory path string for disk cache storage
        memory_cache_size: Maximum number of embeddings in memory cache
        enable_disk_cache: Whether to enable persistent disk caching
        cache_expiry_hours: Hours after which cache entries expire
    """
    try:
        from utilities.speaker_cache import configure_global_cache
        configure_global_cache(
            cache_dir=cache_dir,
            memory_cache_size=memory_cache_size,
            enable_disk_cache=enable_disk_cache,
            cache_expiry_hours=cache_expiry_hours
        )
        logging.info(f"Speaker cache configured: {cache_dir}, memory_size={memory_cache_size}")
    except ImportError as e:
        logging.warning(f"Speaker cache system not available {e}")
    except Exception as e:
        logging.error(f"Error configuring speaker cache: {e}")


def get_speaker_cache_stats() -> Dict[str, Any]:
    """Get speaker embedding cache statistics."""
    try:
        from utilities.speaker_cache import get_global_speaker_cache
        cache = get_global_speaker_cache()
        return cache.get_cache_stats()
    except ImportError:
        return {"cache_available": False}


def print_speaker_cache_stats() -> None:
    """Print speaker embedding cache statistics."""
    try:
        from utilities.speaker_cache import get_global_speaker_cache
        cache = get_global_speaker_cache()
        cache.print_cache_stats()
    except ImportError:
        print("Speaker cache system not available")


def clear_speaker_cache() -> None:
    """Clear all speaker embedding caches."""
    try:
        from utilities.speaker_cache import get_global_speaker_cache
        cache = get_global_speaker_cache()
        cache.clear_all_cache()
        logging.info("Speaker cache cleared")
    except ImportError:
        logging.warning("Speaker cache system not available")
