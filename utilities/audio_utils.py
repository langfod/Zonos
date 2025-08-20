"""
Audio processing utilities for Zonos application.
"""
import threading

import torch
import torchaudio
import logging
from typing import Dict

from utilities.cache_utils import get_cache_key, load_from_disk, save_to_disk

# Global cache for audio prefixes
PREFIX_AUDIO_CACHE: Dict[str, torch.Tensor] = {}
SPEAKER_CACHE: Dict[str, torch.Tensor] = {}


async def process_speaker_audio(speaker_audio_path: str, uuid: int, model, device: torch.device, enable_disk_cache=True) -> torch.Tensor:
    """
    Process speaker audio and return speaker embedding.
    Uses caching to avoid recomputing embeddings for the same audio.
    """
    cache_key = get_cache_key(speaker_audio_path, uuid)
    if cache_key in SPEAKER_CACHE:
        logging.info("Reused cached speaker embedding (memory).")
        return SPEAKER_CACHE[cache_key]

    if enable_disk_cache:
        speaker_embedding = load_from_disk(cache_key, "embeds", device=device)
        if speaker_embedding is not None:
            logging.info(f"Loaded speaker embedding for {cache_key} from disk cache.")
            # Keep on target device to avoid unnecessary transfers
            SPEAKER_CACHE[cache_key] = speaker_embedding
            return speaker_embedding

    try:
        wav, sr = torchaudio.load(speaker_audio_path)
        if wav.size(0) > 1:  # mix to mono
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device, non_blocking=True)  # Use non_blocking for async transfer
    except Exception as e:
        logging.error(f"Failed to load speaker audio '{speaker_audio_path}': {e}")
        raise

    logging.info("Computing speaker embedding.")
    autocast_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if autocast_enabled else None

    with torch.no_grad():
        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                speaker_embedding = model.make_speaker_embedding(wav, sr)
        else:
            speaker_embedding = model.make_speaker_embedding(wav, sr)

    # Detach & cache
    speaker_embedding = speaker_embedding.detach()
    SPEAKER_CACHE[cache_key] = speaker_embedding

    if enable_disk_cache:
        threading.Thread(
            target=save_to_disk,
            args=(cache_key, "embeds", speaker_embedding.cpu()),
            daemon=True
        ).start()

    return speaker_embedding


async def process_prefix_audio(prefix_audio_path: str, model, device: torch.device, enable_disk_cache=True) -> torch.Tensor:
    """
    Process prefix audio and return encoded audio codes.
    Uses global caching to avoid recomputing codes for the same audio.
    """
    prefix_audio_cache_key = get_cache_key(prefix_audio_path)
    if prefix_audio_cache_key in PREFIX_AUDIO_CACHE:
        logging.info("Using cached audio prefix.")
        return PREFIX_AUDIO_CACHE[prefix_audio_cache_key]

    if enable_disk_cache:
        audio_prefix_codes = load_from_disk(prefix_audio_cache_key, "prefix", device=device)
        if audio_prefix_codes is not None:
            logging.info(f"Loaded prefix audio for {prefix_audio_cache_key} from disk cache")
            # Keep directly (no clone) unless you plan to mutate downstream
            PREFIX_AUDIO_CACHE[prefix_audio_cache_key] = audio_prefix_codes
            return audio_prefix_codes
        else:
            logging.info(f"Prefix embedding for {prefix_audio_cache_key} not found in disk cache, computing new embedding")

    logging.info("Encoding and caching new audio prefix.")
    wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
    if wav_prefix.size(0) > 1:
        wav_prefix = wav_prefix.mean(dim=0, keepdim=True)
    wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix).unsqueeze(0)
    wav_prefix = wav_prefix.to(device, non_blocking=True)

    with torch.no_grad():
        audio_prefix_codes = model.autoencoder.encode(wav_prefix)

    PREFIX_AUDIO_CACHE[prefix_audio_cache_key] = audio_prefix_codes

    if enable_disk_cache:
        threading.Thread(
            target=save_to_disk,
            args=(prefix_audio_cache_key, "prefix", audio_prefix_codes.detach().to("cpu")),
            daemon=True
        ).start()

    return audio_prefix_codes
