"""
Audio processing utilities for Zonos application.
"""
import threading

import torch
import torchaudio
import logging
from typing import Dict

from utilities.cache_utils import get_cache_key, load_from_disk, save_to_disk
from zonos.speaker_cloning import SpeakerEmbeddingLDA

# Global cache for audio prefixes
PREFIX_AUDIO_CACHE: Dict[str, torch.Tensor] = {}
SPEAKER_CACHE: Dict[str, torch.Tensor] = {}
spk_clone_model = None
spk_clone_model_device = "cuda" #"cpu"

def make_speaker_embedding(wav: torch.Tensor, sr: int) -> torch.Tensor:
    global spk_clone_model
    if spk_clone_model is None:
        spk_clone_model = SpeakerEmbeddingLDA(device=torch.device(spk_clone_model_device))
    if wav.device != spk_clone_model.device:
        wav = wav.to(spk_clone_model.device)
    _, spk_embedding = spk_clone_model(wav, sr)
    return spk_embedding.unsqueeze(0).bfloat16()

async def process_speaker_audio(speaker_audio_path: str, uuid: int, model, device: torch.device, enable_disk_cache=True) -> torch.Tensor:
    """
    Process speaker audio and return speaker embedding.
    Uses caching to avoid recomputing embeddings for the same audio.
    """
    cache_key = get_cache_key(speaker_audio_path, uuid)
    if cache_key in SPEAKER_CACHE:
        logging.info("Reused cached speaker embedding (memory).")
        cached_embedding = SPEAKER_CACHE[cache_key]

        if cached_embedding.device != device:
            cached_embedding = cached_embedding.to(device)
            SPEAKER_CACHE[cache_key] = cached_embedding
        return cached_embedding

    if enable_disk_cache:
        speaker_embedding = load_from_disk(cache_key, "embeds", device=device)
        if speaker_embedding is not None:
            logging.info(f"Loaded speaker embedding for {cache_key} from disk cache.")
            SPEAKER_CACHE[cache_key] = speaker_embedding
            return speaker_embedding

    try:
        wav, sr = torchaudio.load(speaker_audio_path, normalize=True)
        if wav.size(0) > 1:  # mix to mono
            wav = wav.mean(dim=0, keepdim=True)
    except Exception as e:
        logging.error(f"Failed to load speaker audio '{speaker_audio_path}': {e}")
        raise

    logging.info("Computing speaker embedding.")

    with torch.no_grad():
        speaker_embedding = make_speaker_embedding(wav, sr).detach()

    if enable_disk_cache:
        threading.Thread(
            target=save_to_disk,
            args=(cache_key, "embeds", speaker_embedding.cpu()),
            daemon=True
        ).start()

    if speaker_embedding.device != device:
        speaker_embedding = speaker_embedding.to(device)
    
    SPEAKER_CACHE[cache_key] = speaker_embedding

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
            PREFIX_AUDIO_CACHE[prefix_audio_cache_key] = audio_prefix_codes
            return audio_prefix_codes
        else:
            logging.info(f"Prefix embedding for {prefix_audio_cache_key} not found in disk cache, computing new embedding")

    logging.info("Encoding and caching new audio prefix.")
    wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
    if wav_prefix.size(0) > 1:
        wav_prefix = wav_prefix.mean(dim=0, keepdim=True)
    wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix).unsqueeze(0)
    if wav_prefix.device != device:
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
