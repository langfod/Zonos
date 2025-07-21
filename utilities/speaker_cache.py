"""
Speaker embedding cache system for Zonos application.
Provides both memory and disk caching with torch.compile compatibility and thread safety.
"""

import threading
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from collections import OrderedDict
import torch

from threading import RLock
from utilities.file_utils import (get_cache_dir)

class SpeakerEmbeddingCache:
    """
    Thread-safe cache system for speaker embeddings with memory and disk storage.

    Features:
    - LRU memory cache with configurable size
    - Persistent disk cache using torch.save/torch.load
    - Thread-safe operations with RLock
    - Cache key generation from filename and audio properties
    - Torch.compile compatible
    - Automatic cache cleanup and maintenance
    """

    def __init__(
        self,
        cache_dir: str = None,
        memory_cache_size: int = 100,
        enable_disk_cache: bool = True,
        cache_expiry_hours: float = 24 * 7,  # 1 week default
        auto_cleanup: bool = True
    ):
        self.cache_dir = get_cache_dir(cache_dir)
        self.memory_cache_size = memory_cache_size
        self.enable_disk_cache = enable_disk_cache
        self.cache_expiry_seconds = cache_expiry_hours * 3600
        self.auto_cleanup = auto_cleanup

        # Thread safety
        self._lock = RLock()

        # Memory cache: LRU implementation using OrderedDict
        self._memory_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # Cache statistics
        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'saves': 0,
            'evictions': 0
        }

        # Initialize cache directory
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.auto_cleanup:
                self._cleanup_expired_cache()

    def _generate_cache_key(self, audio_path: str, model_type: str = "default", audio_uuid: str = None) -> str:
        """
        Generate a unique cache key from audio file path, model type, and optional UUID.
        Uses filename stem plus UUID for a readable and unique cache key.
        """
        try:
            path = Path(audio_path)
            file_stem = path.stem  # filename without extension

            # Extract model name from full model path (e.g., "Zonos-v0.1-hybrid" from "Zyphra/Zonos-v0.1-hybrid")
            if "/" in model_type:
                model_name = model_type.split("/")[-1]
            else:
                model_name = model_type

            # If UUID is provided, convert to hex format if it's a numeric value
            if audio_uuid is not None:
                if isinstance(audio_uuid, int):
                    # Convert C++ unsigned int to hex format
                    uuid_hex = hex(audio_uuid)[2:]  # Remove '0x' prefix
                elif isinstance(audio_uuid, str) and audio_uuid.isdigit():
                    # Handle string representation of numeric UUID
                    uuid_hex = hex(int(audio_uuid))[2:]
                else:
                    # Use as-is for non-numeric UUIDs
                    uuid_hex = str(audio_uuid)

                cache_key = f"spk_emb_{file_stem}_{uuid_hex}_{model_name}"
            else:
                # Fallback: use file size and mtime if file exists
                if path.exists():
                    stat = path.stat()
                    cache_key = f"spk_emb_{file_stem}_{stat.st_size}_{int(stat.st_mtime)}_{model_name}"
                else:
                    # For non-existent files, use just the filename and model type
                    cache_key = f"spk_emb_{file_stem}_{model_name}"

            # Sanitize the cache key for filesystem compatibility
            # Replace any potentially problematic characters
            cache_key = cache_key.replace(" ", "_").replace("/", "_").replace("\\", "_")
            cache_key = "".join(c for c in cache_key if c.isalnum() or c in "._-")

            return cache_key

        except Exception as e:
            logging.warning(f"Error generating cache key for {audio_path}: {e}")
            # Fallback to simple filename-based key
            file_stem = Path(audio_path).stem
            model_name = model_type.split("/")[-1] if "/" in model_type else model_type
            fallback_key = f"spk_emb_{file_stem}_{model_name}"
            return "".join(c for c in fallback_key if c.isalnum() or c in "._-")

    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """Get the disk cache file path for a given cache key."""
        return self.cache_dir / f"{cache_key}.pt"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if a disk cache file is still valid (not expired)."""
        if not cache_path.exists():
            return False

        try:
            age = time.time() - cache_path.stat().st_mtime
            return age < self.cache_expiry_seconds
        except Exception:
            return False

    def _cleanup_expired_cache(self):
        """Remove expired cache files from disk."""
        if not self.enable_disk_cache or not self.cache_dir.exists():
            return

        try:
            current_time = time.time()
            removed_count = 0

            for cache_file in self.cache_dir.glob("spk_emb_*.pt"):
                try:
                    if current_time - cache_file.stat().st_mtime > self.cache_expiry_seconds:
                        cache_file.unlink()
                        removed_count += 1
                except Exception as e:
                    logging.warning(f"Error removing expired cache file {cache_file}: {e}")

            if removed_count > 0:
                logging.info(f"Cleaned up {removed_count} expired cache files")

        except Exception as e:
            logging.error(f"Error during cache cleanup: {e}")

    def _evict_lru_memory(self):
        """Remove least recently used item from memory cache."""
        if self._memory_cache:
            evicted_key, _ = self._memory_cache.popitem(last=False)
            self._stats['evictions'] += 1
            logging.debug(f"Evicted LRU cache entry: {evicted_key}")

    def get(self, audio_path: str, model_type: str = "default", audio_uuid = None) -> Optional[torch.Tensor]:
        """
        Retrieve speaker embedding from cache.

        Args:
            audio_path: Path to the audio file
            model_type: Type/version of the embedding model
            audio_uuid: Optional UUID for unique identification (C++ unsigned int or equivalent)

        Returns:
            Cached embedding tensor or None if not found
        """
        cache_key = self._generate_cache_key(audio_path, model_type, audio_uuid)

        with self._lock:
            # Try memory cache first
            if cache_key in self._memory_cache:
                # Move to end (most recently used)
                embedding = self._memory_cache.pop(cache_key)
                self._memory_cache[cache_key] = embedding
                self._stats['memory_hits'] += 1
                logging.debug(f"Memory cache hit for {cache_key}")
                return embedding.clone()  # Return clone for safety

            # Try disk cache
            if self.enable_disk_cache:
                cache_path = self._get_disk_cache_path(cache_key)
                if self._is_cache_valid(cache_path):
                    try:
                        embedding = torch.load(cache_path, map_location='cpu', weights_only=True)

                        # Add to memory cache
                        if len(self._memory_cache) >= self.memory_cache_size:
                            self._evict_lru_memory()
                        self._memory_cache[cache_key] = embedding.clone()

                        self._stats['disk_hits'] += 1
                        logging.debug(f"Disk cache hit for {cache_key}")
                        return embedding.clone()

                    except Exception as e:
                        logging.warning(f"Error loading cached embedding from {cache_path}: {e}")
                        # Remove corrupted cache file
                        try:
                            cache_path.unlink()
                        except Exception:
                            pass

            # Cache miss
            self._stats['misses'] += 1
            return None

    def put(self, audio_path: str, embedding: torch.Tensor, model_type: str = "default", audio_uuid = None) -> bool:
        """
        Store speaker embedding in cache.

        Args:
            audio_path: Path to the audio file
            embedding: The embedding tensor to cache (can be a single tensor or tuple of tensors)
            model_type: Type/version of the embedding model
            audio_uuid: Optional UUID for unique identification (C++ unsigned int or equivalent)

        Returns:
            True if successfully cached, False otherwise
        """
        if embedding is None:
            return False

        cache_key = self._generate_cache_key(audio_path, model_type, audio_uuid)

        with self._lock:
            try:
                # Store in memory cache
                if len(self._memory_cache) >= self.memory_cache_size:
                    self._evict_lru_memory()

                # Handle both single tensors and tuples of tensors
                if isinstance(embedding, tuple):
                    # For tuples (like from SpeakerEmbeddingLDA), the tensors should already be detached
                    cpu_embedding = embedding
                else:
                    # For single tensors, detach and move to CPU
                    cpu_embedding = embedding.detach().cpu()

                self._memory_cache[cache_key] = cpu_embedding

                # Store to disk cache
                if self.enable_disk_cache:
                    cache_path = self._get_disk_cache_path(cache_key)
                    try:
                        # Use atomic write to prevent corruption
                        temp_path = cache_path.with_suffix('.tmp')
                        torch.save(cpu_embedding, temp_path)
                        temp_path.replace(cache_path)
                        logging.debug(f"Saved embedding to disk cache: {cache_path}")
                    except Exception as e:
                        logging.warning(f"Error saving embedding to disk cache: {e}")
                        # Clean up temp file if it exists
                        if temp_path.exists():
                            try:
                                temp_path.unlink()
                            except Exception:
                                pass

                self._stats['saves'] += 1
                return True

            except Exception as e:
                logging.error(f"Error caching embedding for {audio_path}: {e}")
                return False

    def clear_memory_cache(self):
        """Clear all entries from memory cache."""
        with self._lock:
            self._memory_cache.clear()
            logging.info("Memory cache cleared")

    def clear_disk_cache(self):
        """Clear all entries from disk cache."""
        if not self.enable_disk_cache:
            return

        with self._lock:
            try:
                removed_count = 0
                for cache_file in self.cache_dir.glob("spk_emb_*.pt"):
                    try:
                        cache_file.unlink()
                        removed_count += 1
                    except Exception as e:
                        logging.warning(f"Error removing cache file {cache_file}: {e}")

                logging.info(f"Disk cache cleared, removed {removed_count} files")

            except Exception as e:
                logging.error(f"Error clearing disk cache: {e}")

    def clear_all_cache(self):
        """Clear both memory and disk caches."""
        self.clear_memory_cache()
        self.clear_disk_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._stats['memory_hits'] + self._stats['disk_hits'] + self._stats['misses']
            hit_rate = ((self._stats['memory_hits'] + self._stats['disk_hits']) / total_requests * 100) if total_requests > 0 else 0

            return {
                'memory_cache_size': len(self._memory_cache),
                'memory_cache_max_size': self.memory_cache_size,
                'disk_cache_enabled': self.enable_disk_cache,
                'cache_directory': str(self.cache_dir),
                'total_requests': total_requests,
                'memory_hits': self._stats['memory_hits'],
                'disk_hits': self._stats['disk_hits'],
                'misses': self._stats['misses'],
                'hit_rate_percent': round(hit_rate, 2),
                'saves': self._stats['saves'],
                'evictions': self._stats['evictions']
            }

    def print_cache_stats(self):
        """Print formatted cache statistics."""
        stats = self.get_cache_stats()
        print("\n=== Speaker Embedding Cache Statistics ===")
        print(f"Memory Cache: {stats['memory_cache_size']}/{stats['memory_cache_max_size']} entries")
        print(f"Disk Cache: {'Enabled' if stats['disk_cache_enabled'] else 'Disabled'}")
        print(f"Cache Directory: {stats['cache_directory']}")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Memory Hits: {stats['memory_hits']}")
        print(f"Disk Hits: {stats['disk_hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Hit Rate: {stats['hit_rate_percent']}%")
        print(f"Saves: {stats['saves']}")
        print(f"Evictions: {stats['evictions']}")
        print("=" * 42)


# Global cache instance
_global_speaker_cache = None
_cache_lock = threading.Lock()


def get_global_speaker_cache() -> SpeakerEmbeddingCache:
    """Get or create the global speaker embedding cache instance."""
    global _global_speaker_cache

    if _global_speaker_cache is None:
        with _cache_lock:
            if _global_speaker_cache is None:
                _global_speaker_cache = SpeakerEmbeddingCache()

    return _global_speaker_cache


def configure_global_cache(**kwargs) -> SpeakerEmbeddingCache:
    """Configure the global speaker embedding cache with custom settings."""
    global _global_speaker_cache

    with _cache_lock:
        _global_speaker_cache = SpeakerEmbeddingCache(**kwargs)

    return _global_speaker_cache
