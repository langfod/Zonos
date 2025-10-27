"""
Cache utilities for managing audio embeddings and prefixes.

This module provides caching functionality for speaker embeddings and audio prefixes,
with both memory and disk storage capabilities. 

Example usage:

    # Using the cache manager directly
    from cache_utils import TensorCacheManager
    
    manager = TensorCacheManager("embeds")
    manager.set("speaker_1", my_embedding_tensor)
    cached_embedding = manager.get("speaker_1")
    
    # Using convenience functions
    from cache_utils import cache_speaker_embedding, get_cached_speaker_embedding
    
    cache_speaker_embedding("speaker_1", my_embedding_tensor)
    cached_embedding = get_cached_speaker_embedding("speaker_1")
    
    # Using global cache managers
    from cache_utils import get_embed_cache_manager, get_cache_stats
    
    manager = get_embed_cache_manager()
    stats = get_cache_stats()

"""
import datetime
import functools
from loguru import logger
import os
import threading
import traceback
from pathlib import Path
from typing import Dict, Optional, Union

import psutil
import torch
import torchaudio
import warnings


_cache_lock = threading.Lock()
# Global cache for audio prefixes
PREFIX_AUDIO_CACHE: Dict[str, torch.Tensor] = {}
SPEAKER_CACHE: Dict[str, torch.Tensor] = {}
WAV_OUTPUT_DIR = Path("output_temp")

class TensorCacheManager:
    """
    Manages tensor cache for both memory and disk storage.
    
    Provides stateful cache management with automatic memory and disk 
    synchronization, thread-safe operations, and efficient lookup for
    PyTorch tensors (speaker embeddings, audio prefix codes, etc.).
    """
    
    def __init__(self, cache_type: str = "embeds", device: Optional[torch.device] = None):
        """
        Initialize tensor cache manager.
        
        Args:
            cache_type (str): Type of cache ("embeds" or "prefix")
            device (torch.device, optional): Default device for loading tensors
        """
        self.cache_type = cache_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        
        # Memory cache - use existing global caches based on type
        if cache_type == "embeds":
            self._memory_cache = SPEAKER_CACHE
        elif cache_type == "prefix":
            self._memory_cache = PREFIX_AUDIO_CACHE
        else:
            raise ValueError(f"Unknown cache type: {cache_type}. Must be 'embeds' or 'prefix'")
    
    def get(self, cache_key: str, auto_load: bool = True) -> Optional[torch.Tensor]:
        """
        Get tensor from cache, checking memory first then disk.
        
        Args:
            cache_key (str): Cache key to retrieve
            auto_load (bool): Whether to automatically load from disk if not in memory
            
        Returns:
            torch.Tensor or None: Cached tensor or None if not found
        """
        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                logger.debug(f"Cache hit (memory): {cache_key}")
                return self._memory_cache[cache_key]
            
            # Try loading from disk if enabled
            if auto_load:
                tensor = load_from_disk(cache_key, self.cache_type, self.device)
                if tensor is not None:
                    # Store in memory cache for faster future access
                    self._memory_cache[cache_key] = tensor
                    logger.debug(f"Cache hit (disk->memory): {cache_key}")
                    return tensor
            
            logger.debug(f"Cache miss: {cache_key}")
            return None
    
    def set(self, cache_key: str, tensor: torch.Tensor, save_to_disk_flag: bool = True) -> None:
        """
        Store tensor in cache (both memory and optionally disk).
        
        Args:
            cache_key (str): Cache key for storage
            tensor (torch.Tensor): Tensor to cache
            save_to_disk_flag (bool): Whether to also save to disk
        """
        with self._lock:
            # Store in memory cache
            self._memory_cache[cache_key] = tensor
            logger.debug(f"Cached to memory: {cache_key}")
            
            # Optionally save to disk
            if save_to_disk_flag:
                save_to_disk(cache_key, self.cache_type, tensor)
    
    def exists(self, cache_key: str, check_disk: bool = True) -> bool:
        """
        Check if cache entry exists in memory and/or disk.
        
        Args:
            cache_key (str): Cache key to check
            check_disk (bool): Whether to also check disk
            
        Returns:
            bool: True if cache entry exists
        """
        with self._lock:
            # Check memory first
            if cache_key in self._memory_cache:
                return True
            
            # Check disk if requested
            if check_disk:
                cache_dir = get_embed_cache_dir() if self.cache_type == "embeds" else get_prefix_cache_dir()
                cache_file = cache_dir.joinpath(cache_key + ".pt")
                return cache_file.exists()
            
            return False
    
    def clear_memory(self) -> None:
        """Clear all entries from memory cache."""
        with self._lock:
            self._memory_cache.clear()
            logger.info(f"Cleared {self.cache_type} memory cache")
    
    def clear_disk(self) -> None:
        """Clear all entries from disk cache."""
        cache_dir = get_embed_cache_dir() if self.cache_type == "embeds" else get_prefix_cache_dir()
        
        try:
            for cache_file in cache_dir.glob("*.pt"):
                cache_file.unlink()
            logger.info(f"Cleared {self.cache_type} disk cache")
        except Exception as e:
            logger.error(f"Failed to clear {self.cache_type} disk cache: {e}")
    
    def clear_all(self) -> None:
        """Clear both memory and disk caches."""
        self.clear_memory()
        self.clear_disk()
    
    def get_memory_cache_size(self) -> int:
        """Get number of entries in memory cache."""
        with self._lock:
            return len(self._memory_cache)
    
    def get_memory_cache_keys(self) -> list:
        """Get list of keys in memory cache."""
        with self._lock:
            return list(self._memory_cache.keys())
    
    def remove(self, cache_key: str, remove_from_disk: bool = True) -> bool:
        """
        Remove specific cache entry from memory and/or disk.
        
        Args:
            cache_key (str): Cache key to remove
            remove_from_disk (bool): Whether to also remove from disk
            
        Returns:
            bool: True if anything was removed
        """
        removed = False
        
        with self._lock:
            # Remove from memory
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
                removed = True
                logger.debug(f"Removed from memory cache: {cache_key}")
        
        # Remove from disk
        if remove_from_disk:
            cache_dir = get_embed_cache_dir() if self.cache_type == "embeds" else get_prefix_cache_dir()
            cache_file = cache_dir.joinpath(cache_key + ".pt")
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    removed = True
                    logger.debug(f"Removed from disk cache: {cache_key}")
                except Exception as e:
                    logger.error(f"Failed to remove {cache_key} from disk: {e}")
        
        return removed


# Global cache manager instances for convenience
_embed_cache_manager: Optional[TensorCacheManager] = None
_prefix_cache_manager: Optional[TensorCacheManager] = None


def get_embed_cache_manager(device: Optional[torch.device] = None) -> TensorCacheManager:
    """
    Get or create global embed cache manager instance.
    
    Args:
        device (torch.device, optional): Device for loading tensors
        
    Returns:
        TensorCacheManager: Global embed cache manager
    """
    global _embed_cache_manager
    if _embed_cache_manager is None:
        _embed_cache_manager = TensorCacheManager("embeds", device)
    return _embed_cache_manager


def get_prefix_cache_manager(device: Optional[torch.device] = None) -> TensorCacheManager:
    """
    Get or create global prefix cache manager instance.
    
    Args:
        device (torch.device, optional): Device for loading tensors
        
    Returns:
        TensorCacheManager: Global prefix cache manager
    """
    global _prefix_cache_manager
    if _prefix_cache_manager is None:
        _prefix_cache_manager = TensorCacheManager("prefix", device)
    return _prefix_cache_manager


def clear_all_caches():
    """Clear all cache managers (both memory and disk)."""
    global _embed_cache_manager, _prefix_cache_manager
    
    if _embed_cache_manager:
        _embed_cache_manager.clear_all()
    if _prefix_cache_manager:
        _prefix_cache_manager.clear_all()


def get_cache_stats() -> Dict[str, Dict[str, Union[int, list]]]:
    """
    Get statistics about all cache managers.
    
    Returns:
        dict: Cache statistics including memory cache sizes and keys
    """
    stats = {}
    
    if _embed_cache_manager:
        stats["embeds"] = {
            "memory_size": _embed_cache_manager.get_memory_cache_size(),
            "memory_keys": _embed_cache_manager.get_memory_cache_keys()
        }
    
    if _prefix_cache_manager:
        stats["prefix"] = {
            "memory_size": _prefix_cache_manager.get_memory_cache_size(),
            "memory_keys": _prefix_cache_manager.get_memory_cache_keys()
        }
    
    return stats


@functools.lru_cache(1)
def get_process_creation_time():
    """Get the process creation time as a datetime object"""
    p = psutil.Process(os.getpid())
    creation_timestamp = p.create_time()
    return datetime.datetime.fromtimestamp(creation_timestamp)

@functools.cache
def get_embed_cache_dir():
    """Get or create the conditionals cache directory"""
    # Lazy import to avoid circular dependency
    from utilities.model_utils import CURRENT_MODEL_TYPE
    
    model_ext = CURRENT_MODEL_TYPE.split('/')[-1] if CURRENT_MODEL_TYPE else None
    cache_dir = Path("cache").joinpath("embeds")
    if model_ext:
        cache_dir = cache_dir.joinpath(model_ext)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@functools.cache
def get_prefix_cache_dir():
    """Get or create the conditionals cache directory"""
    # Lazy import to avoid circular dependency
    #from utilities.model_utils import CURRENT_MODEL_TYPE    
    #model_ext = CURRENT_MODEL_TYPE.split('/')[-1] if CURRENT_MODEL_TYPE else None
    cache_dir = Path("cache").joinpath("prefixes")
    #if model_ext:
    #    cache_dir = cache_dir.joinpath(model_ext)
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

        logger.info(f"Saved {cache_type} cache to disk: {cache_key} as {cache_file}")

    except Exception as e:
        logger.error(f"Failed to save {cache_type} from disk cache: {e}")
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
            logger.warning(f"Cache file {cache_file} does not exist.")
            return None

        return torch.load(cache_file, map_location=device, weights_only=True)

    except Exception as e:
        import traceback
        logger.error(f"Failed to load {cache_type} disk cache: {e}")
        print(f"Failed to load {cache_type} disk cache: {e}")
        print(traceback.format_exc())
        return None

@functools.cache
def get_cache_key(audio_path):
    """Generate a cache key based on audio file"""
    if audio_path is None:
        return None

    return Path(audio_path).stem


@functools.cache
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = WAV_OUTPUT_DIR.joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir

def save_torchaudio_wav(wav_tensor, sr, audio_path):
    """Save a tensor as a WAV file using torchaudio"""

    formatted_now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path)}"
    path = get_wavout_dir() / f"{filename}.wav"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torchaudio.save(path, wav_tensor.cpu(), sr, encoding="PCM_S")
    return path.resolve()


# Convenience functions using the new cache managers
def cache_speaker_embedding(cache_key: str, embedding: torch.Tensor, device: Optional[torch.device] = None) -> None:
    """
    Cache a speaker embedding using the TensorCacheManager.
    
    Args:
        cache_key (str): Cache key for the embedding
        embedding (torch.Tensor): Speaker embedding tensor
        device (torch.device, optional): Device for the cache manager
    """
    manager = get_embed_cache_manager(device)
    manager.set(cache_key, embedding)


def get_cached_speaker_embedding(cache_key: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """
    Retrieve a cached speaker embedding using the TensorCacheManager.
    
    Args:
        cache_key (str): Cache key for the embedding
        device (torch.device, optional): Device for the cache manager
        
    Returns:
        torch.Tensor or None: Cached embedding or None if not found
    """
    manager = get_embed_cache_manager(device)
    return manager.get(cache_key)


def cache_audio_prefix(cache_key: str, prefix: torch.Tensor, device: Optional[torch.device] = None) -> None:
    """
    Cache an audio prefix using the TensorCacheManager.
    
    Args:
        cache_key (str): Cache key for the prefix
        prefix (torch.Tensor): Audio prefix tensor
        device (torch.device, optional): Device for the cache manager
    """
    manager = get_prefix_cache_manager(device)
    manager.set(cache_key, prefix)


def get_cached_audio_prefix(cache_key: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """
    Retrieve a cached audio prefix using the TensorCacheManager.
    
    Args:
        cache_key (str): Cache key for the prefix
        device (torch.device, optional): Device for the cache manager
        
    Returns:
        torch.Tensor or None: Cached prefix or None if not found
    """
    manager = get_prefix_cache_manager(device)
    return manager.get(cache_key)


def is_cached(cache_key: str, cache_type: str = "embeds", device: Optional[torch.device] = None) -> bool:
    """
    Check if an item is cached (memory or disk).
    
    Args:
        cache_key (str): Cache key to check
        cache_type (str): Type of cache ("embeds" or "prefix")
        device (torch.device, optional): Device for the cache manager
        
    Returns:
        bool: True if cached
    """
    if cache_type == "embeds":
        manager = get_embed_cache_manager(device)
    elif cache_type == "prefix":
        manager = get_prefix_cache_manager(device)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
    
    return manager.exists(cache_key)

@functools.cache
def get_speakers_dir(language: str = "en") -> Path:
    """Get or create the speakers directory"""
    speakers_dir = Path("speakers").joinpath(language)
    speakers_dir.mkdir(parents=True, exist_ok=True)
    return speakers_dir

