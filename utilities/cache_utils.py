import datetime
import functools
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Dict

import psutil
import torch
import torchaudio
import soundfile as sf

_cache_lock = threading.Lock()
# Global cache for audio prefixes
PREFIX_AUDIO_CACHE: Dict[str, torch.Tensor] = {}
SPEAKER_CACHE: Dict[str, torch.Tensor] = {}

@functools.lru_cache(1)
def get_process_creation_time():
    """Get the process creation time as a datetime object"""
    p = psutil.Process(os.getpid())
    creation_timestamp = p.create_time()
    return datetime.datetime.fromtimestamp(creation_timestamp)

@functools.cache
def get_embed_cache_dir():
    """Get or create the conditionals cache directory"""
    cache_dir = Path("cache/embeds")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@functools.cache
def get_prefix_cache_dir():
    """Get or create the conditionals cache directory"""
    cache_dir = Path("cache/prefixes")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def save_to_disk(cache_key:str, cache_type:str, speaker_embedding: torch.Tensor):
    try:
        if cache_type == "prefix":
            cache_dir = get_prefix_cache_dir()
        elif cache_type == "embeds":
            cache_dir = get_embed_cache_dir()
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
        cache_file = cache_dir.joinpath(cache_key + ".pt")

        torch.save(speaker_embedding, cache_file)

        logging.info(f"Saved {cache_type} cache to disk: {cache_key} as {cache_file}")

    except Exception as e:
        logging.error(f"Failed to save {cache_type} from disk cache: {e}")
        print(f"Failed to save {cache_type} from disk cache: {e}")
        print(traceback.format_exc())

def load_from_disk(cache_key, cache_type, device: torch.device):
    try:
        if cache_type == "prefix":
            cache_dir = get_prefix_cache_dir()
        elif cache_type == "embeds":
            cache_dir = get_embed_cache_dir()
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
        cache_file = cache_dir.joinpath(cache_key + ".pt")

        if not cache_file.exists():
            logging.warning(f"Cache file {cache_file} does not exist.")
            return None

        return torch.load(cache_file, map_location=device, weights_only=True)

    except Exception as e:
        import traceback
        logging.error(f"Failed to load {cache_type} disk cache: {e}")
        print(f"Failed to load {cache_type} disk cache: {e}")
        print(traceback.format_exc())
        return None

@functools.cache
def get_cache_key(audio_path, uuid: int = None):
    """Generate a cache key based on audio file, UUID, and exaggeration"""
    if audio_path is None:
        return None

    # Extract just the filename without extension as prefix
    try:
        filename = Path(audio_path).stem  # Gets filename without extension
        # Remove any temp directory prefixes, just keep the actual filename
        cache_prefix = filename
    except Exception:
        cache_prefix = "unknown"

    if uuid is not None:
        # Ensure the cache prefix is a valid string
        # Convert UUID to hex string for readability
        try:
            uuid_hex = hex(uuid)[2:]  # Remove '0x' prefix
        except (TypeError, ValueError):
            uuid_hex = str(uuid)

        cache_key = f"{cache_prefix}_{uuid_hex}"
    else:
        cache_key = cache_prefix

    return cache_key

@functools.lru_cache(1)
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = Path("output_temp").joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir

async def save_torchaudio_wav(wav_tensor, sr, audio_path, uuid):
    """Save a tensor as a WAV file using torchaudio"""

    formatted_now_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path, uuid)}"
    path = get_wavout_dir().joinpath(f"{filename}.wav")
    torchaudio.save(path, wav_tensor, sr, encoding="PCM_S", bits_per_sample=16)
    return str(path.resolve())
