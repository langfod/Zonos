"""
Audio processing utilities for Zonos application.
"""
import torch
import torchaudio
import logging
from typing import Dict, Any, Tuple
import numpy as np


# Global cache for audio prefixes
PREFIX_AUDIO_CACHE: Dict[str, torch.Tensor] = {}


def process_speaker_audio(speaker_audio_path: str, model, device: torch.device,
                         speaker_cache: Dict[str, torch.Tensor], speaker_audio_uuid=None, model_choice: str = None) -> torch.Tensor:
    """
    Process speaker audio and return speaker embedding with caching.
    """
    # Check legacy cache first for backward compatibility
    if speaker_audio_path in speaker_cache:
        logging.debug("Reused cached speaker embedding (legacy cache)")
        return speaker_cache[speaker_audio_path]

    # Load and process audio
    logging.debug("Computing speaker embedding with caching")
    wav, sr = torchaudio.load(speaker_audio_path)

    # Use enhanced caching if available and initialized
    if hasattr(model, 'spk_clone_model') and model.spk_clone_model is not None:
        _, speaker_embedding = model.spk_clone_model(
            wav.to(model.spk_clone_model.device), sr,
            audio_path=speaker_audio_path,
            audio_uuid=speaker_audio_uuid,
            model_choice=model_choice
        )
    else:
        # Fallback to regular method (which will initialize spk_clone_model if needed)
        speaker_embedding = model.make_speaker_embedding(wav, sr)

    speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)

    # Store in legacy cache for backward compatibility
    speaker_cache[speaker_audio_path] = speaker_embedding
    return speaker_embedding


def process_prefix_audio(prefix_audio_path: str, model, device: torch.device) -> torch.Tensor:
    """
    Process prefix audio and return encoded audio codes with simple caching.
    """
    global PREFIX_AUDIO_CACHE

    if prefix_audio_path in PREFIX_AUDIO_CACHE:
        logging.debug("Using cached audio prefix")
        return PREFIX_AUDIO_CACHE[prefix_audio_path]

    logging.debug("Encoding and caching new audio prefix")
    wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
    wav_prefix = wav_prefix.mean(0, keepdim=True)
    wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
    wav_prefix = wav_prefix.to(device, dtype=torch.float32)
    audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))
    PREFIX_AUDIO_CACHE[prefix_audio_path] = audio_prefix_codes
    return audio_prefix_codes


def convert_audio_to_numpy(wav_gpu_f32: torch.Tensor, sampling_rate: int) -> Tuple[int, np.ndarray]:
    """Convert GPU tensor audio to numpy array for output."""
    wav_gpu_i16 = (wav_gpu_f32.clamp(-1, 1) * 32767).to(torch.int16)
    wav_np = wav_gpu_i16.cpu().squeeze().numpy()
    return sampling_rate, wav_np


def configure_speaker_cache(cache_dir: str = None, memory_cache_size: int = 100,
                          enable_disk_cache: bool = True, cache_expiry_hours: float = 24 * 7) -> None:
    """Configure the global speaker embedding cache system."""
    try:
        from utilities.speaker_cache import configure_global_cache
        configure_global_cache(
            cache_dir=cache_dir,
            memory_cache_size=memory_cache_size,
            enable_disk_cache=enable_disk_cache,
            cache_expiry_hours=cache_expiry_hours
        )
        logging.info(f"Speaker cache configured: {cache_dir}")
    except Exception as e:
        logging.warning(f"Failed to configure speaker cache: {e}")


def get_speaker_cache_stats() -> Dict[str, Any]:
    """Get speaker embedding cache statistics."""
    try:
        from utilities.speaker_cache import get_global_speaker_cache
        return get_global_speaker_cache().get_cache_stats()
    except Exception:
        return {"cache_available": False}


def print_speaker_cache_stats() -> None:
    """Print speaker embedding cache statistics."""
    try:
        from utilities.speaker_cache import get_global_speaker_cache
        get_global_speaker_cache().print_cache_stats()
    except Exception:
        print("Speaker cache system not available")


def clear_speaker_cache() -> None:
    """Clear all speaker embedding caches."""
    try:
        from utilities.speaker_cache import get_global_speaker_cache
        get_global_speaker_cache().clear_all_cache()
        logging.info("Speaker cache cleared")
    except Exception:
        logging.warning("Speaker cache system not available")
