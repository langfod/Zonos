"""
Audio processing utilities for Zonos application.
"""
import asyncio
import torch
import torchaudio
from loguru import logger
from utilities.cache_utils import (
    get_cache_key,
    get_embed_cache_manager, 
    get_prefix_cache_manager,
    get_embed_cache_dir,
    get_speakers_dir
)
from zonos.speaker_cloning import SpeakerEmbeddingLDA
spk_clone_model = None
spk_clone_model_device = "cuda" if torch.cuda.is_available() else "cpu"

def make_speaker_embedding(wav: torch.Tensor, sr: int) -> torch.Tensor:
    global spk_clone_model
    if spk_clone_model is None:
        spk_clone_model = SpeakerEmbeddingLDA(device=torch.device(spk_clone_model_device))
    if wav.device != spk_clone_model.device:
        wav = wav.to(spk_clone_model.device)
    _, spk_embedding = spk_clone_model(wav, sr)
    return spk_embedding.unsqueeze(0).bfloat16()

async def process_speaker_audio(speaker_audio_path: str, device: torch.device = torch.device(spk_clone_model_device), enable_disk_cache=True) -> torch.Tensor:
    """
    Process speaker audio and return speaker embedding.
    Uses caching to avoid recomputing embeddings for the same audio.
    """
    if device is None:
        device = torch.device(spk_clone_model_device)

    cache_key = get_cache_key(speaker_audio_path)
    speaker_cache = get_embed_cache_manager(device)
    
    # Check cache (memory first, then disk if enabled)
    cached_embedding = speaker_cache.get(cache_key, auto_load=enable_disk_cache)
    if cached_embedding is not None:
        logger.info("Reused cached speaker embedding.")
        if cached_embedding.device != device:
            cached_embedding = cached_embedding.to(device)
            # Update cache with device-corrected tensor
            speaker_cache.set(cache_key, cached_embedding, save_to_disk_flag=False)
        return cached_embedding

    try:
        wav, sr = torchaudio.load(speaker_audio_path, normalize=True)
        if wav.size(0) > 1:  # mix to mono
            wav = wav.mean(dim=0, keepdim=True)
    except Exception as e:
        logger.error(f"Failed to load speaker audio '{speaker_audio_path}': {e}")
        raise

    logger.info("Computing speaker embedding.")

    with torch.no_grad():
        speaker_embedding = make_speaker_embedding(wav, sr).detach()

    if speaker_embedding.device != device:
        speaker_embedding = speaker_embedding.to(device)
    
    # Cache the embedding (both memory and disk)
    speaker_cache.set(cache_key, speaker_embedding, save_to_disk_flag=enable_disk_cache)

    return speaker_embedding


async def process_prefix_audio(prefix_audio_path: str, model, device: torch.device, enable_disk_cache=True) -> torch.Tensor:
    """
    Process prefix audio and return encoded audio codes.
    Uses global caching to avoid recomputing codes for the same audio.
    """
    prefix_audio_cache_key = get_cache_key(prefix_audio_path)
    prefix_cache = get_prefix_cache_manager(device)
    
    # Check cache (memory first, then disk if enabled)
    cached_codes = prefix_cache.get(prefix_audio_cache_key, auto_load=enable_disk_cache)
    if cached_codes is not None:
        logger.info("Using cached audio prefix.")
        return cached_codes

    logger.info("Encoding and caching new audio prefix.")
    wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
    if wav_prefix.size(0) > 1:
        wav_prefix = wav_prefix.mean(dim=0, keepdim=True)
    wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix).unsqueeze(0)
    wav_prefix = wav_prefix.to(device, non_blocking=True)

    with torch.no_grad():
        audio_prefix_codes = model.autoencoder.encode(wav_prefix)

    # Cache the codes (both memory and disk)
    prefix_cache.set(prefix_audio_cache_key, audio_prefix_codes, save_to_disk_flag=enable_disk_cache)

    return audio_prefix_codes


def init_latent_cache() -> None:
    """Initialize latent cache from disk for all supported languages."""
    get_embed_cache_dir.cache_clear()

    cache_manager = get_embed_cache_manager()
    cache_manager.clear_memory()

    latent_dir = get_embed_cache_dir()
    
    # Load existing .pt files from latents directory
    for filename in latent_dir.glob("*.pt"):
        try:
            cache_manager.get(filename.stem)
        except Exception as e:
            logger.error(f"Failed to load latents from {filename}: {e}")
    
    speaker_dir = get_speakers_dir()

    # Get all speaker files and organize by base name
    speaker_files = {}
    for file_path in speaker_dir.iterdir():
        if file_path.is_file() and file_path.suffix in ['.wav']:
            base_name = file_path.stem
            if base_name not in speaker_files:
                speaker_files[base_name] = {}
            speaker_files[base_name][file_path.suffix] = file_path
    
    # Process each speaker, preferring .wav over .json
    for base_name, files in speaker_files.items():
       
        try:
            if '.wav' in files:
                # .wav files - compute latents from audio
                speaker_wav_path = files['.wav']
                logger.info(f"Processing .wav file: {speaker_wav_path}")
                speaker_embedding = asyncio.run(process_speaker_audio(str(speaker_wav_path), enable_disk_cache=True))                                       
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to load latents for speaker {base_name}: {e}") 
    cache_size = cache_manager.get_memory_cache_size()
    logger.info(f"Initialized latent cache with {cache_size} entries.")
