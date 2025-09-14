"""
Speaker embedding cache system for Zonos application.
Thread-safe disk and memory caching with torch.compile compatibility.
GPU-optimized for PyTorch 2.7.1 cu128.
"""

import threading
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union
from collections import OrderedDict
import torch
from threading import RLock
from utilities.file_utils import get_cache_dir


class SpeakerEmbeddingCache:
    """Thread-safe cache system for speaker embeddings with GPU-optimized memory and disk storage."""

    def __init__(self, cache_dir: str = None, memory_cache_size: int = 100,
                 enable_disk_cache: bool = True, cache_expiry_hours: float = 24 * 7,
                 gpu_memory_cache: bool = True):
        self.cache_dir = get_cache_dir(cache_dir)
        self.memory_cache_size = memory_cache_size
        self.enable_disk_cache = enable_disk_cache
        self.cache_expiry_seconds = cache_expiry_hours * 3600
        self.gpu_memory_cache = gpu_memory_cache and torch.cuda.is_available()

        # Thread safety
        self._lock = RLock()

        # Memory cache: LRU implementation (can store GPU or CPU tensors)
        self._memory_cache: OrderedDict[str, Union[torch.Tensor, tuple]] = OrderedDict()

        # Cache statistics
        self._stats = {'memory_hits': 0, 'disk_hits': 0, 'misses': 0, 'saves': 0, 'gpu_hits': 0}

        # Initialize cache directory
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Speaker cache: GPU memory cache {'enabled' if self.gpu_memory_cache else 'disabled'}")

    def _generate_cache_key(self, audio_path: str, model_type: str = "default", audio_uuid=None) -> str:
        """Generate cache key from audio path, model type, and UUID."""
        path = Path(audio_path)
        file_stem = path.stem

        # Extract model name (e.g., "Zonos-v0.1-hybrid" from "Zyphra/Zonos-v0.1-hybrid")
        model_name = model_type.split("/")[-1] if "/" in model_type else model_type

        # Use UUID if provided, otherwise use file properties
        if audio_uuid is not None:
            uuid_hex = hex(audio_uuid)[2:] if isinstance(audio_uuid, int) else str(audio_uuid)
            cache_key = f"spk_emb_{file_stem}_{uuid_hex}_{model_name}"
        else:
            cache_key = f"spk_emb_{file_stem}_{model_name}"

        # Sanitize for filesystem
        return "".join(c for c in cache_key if c.isalnum() or c in "._-")

    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """Get disk cache file path."""
        return self.cache_dir / f"{cache_key}.pt"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid (not expired)."""
        if not cache_path.exists():
            return False
        try:
            age = time.time() - cache_path.stat().st_mtime
            return age < self.cache_expiry_seconds
        except Exception:
            return False

    @staticmethod
    def _move_to_target_device(embedding, target_device: torch.device):
        """Move embedding(s) to target device, handling both single tensors and tuples."""
        if isinstance(embedding, tuple):
            return tuple(tensor.to(target_device) for tensor in embedding)
        else:
            return embedding.to(target_device)

    def get(self, audio_path: str, model_type: str = "default", audio_uuid=None,
            target_device: torch.device = None) -> Optional[torch.Tensor]:
        """Retrieve embedding from cache."""

        cache_key = self._generate_cache_key(audio_path, model_type, audio_uuid)

        with self._lock:
            # Try memory cache first
            if cache_key in self._memory_cache:
                embedding = self._memory_cache.pop(cache_key)
                self._memory_cache[cache_key] = embedding  # Move to end (LRU)
                self._stats['memory_hits'] += 1

                # Track GPU cache hits
                if isinstance(embedding, tuple):
                    if embedding[0].is_cuda:
                        self._stats['gpu_hits'] += 1
                elif hasattr(embedding, 'is_cuda') and embedding.is_cuda:
                    self._stats['gpu_hits'] += 1

                # Move to target device if specified
                if target_device is not None:
                    embedding = self._move_to_target_device(embedding, target_device)

                return embedding

            # Try disk cache
            if self.enable_disk_cache:
                cache_path = self._get_disk_cache_path(cache_key)
                if self._is_cache_valid(cache_path):
                    try:
                        # Load to CPU first, then move to appropriate device
                        embedding = torch.load(cache_path, map_location='cpu', weights_only=True)

                        # Determine target device for memory cache storage
                        if self.gpu_memory_cache and target_device and target_device.type == 'cuda':
                            # Store in GPU memory cache and return on target device
                            gpu_embedding = self._move_to_target_device(embedding, target_device)
                            cache_embedding = gpu_embedding
                            result_embedding = gpu_embedding
                        elif target_device:
                            # Move to target device for return, but store CPU version in cache
                            result_embedding = self._move_to_target_device(embedding, target_device)
                            cache_embedding = embedding  # Keep CPU version for cache
                        else:
                            # No target device specified, use original
                            cache_embedding = result_embedding = embedding

                        # Add to memory cache
                        if len(self._memory_cache) >= self.memory_cache_size:
                            self._memory_cache.popitem(last=False)  # Remove LRU
                        self._memory_cache[cache_key] = cache_embedding

                        self._stats['disk_hits'] += 1
                        return result_embedding

                    except Exception as e:
                        logging.warning(f"Error loading cached embedding: {e}")
                        cache_path.unlink(missing_ok=True)  # Remove corrupted file

            # Cache miss
            self._stats['misses'] += 1
            return None

    def put(self, audio_path: str, embedding, model_type: str = "default", audio_uuid=None) -> bool:
        """Store embedding in cache with GPU optimization."""
        if embedding is None:
            return False

        cache_key = self._generate_cache_key(audio_path, model_type, audio_uuid)

        with self._lock:
            try:
                # Handle both single tensors and tuples
                if isinstance(embedding, tuple):
                    # For tuples, tensors should already be detached in SpeakerEmbeddingLDA
                    memory_embedding = embedding
                    # Always save CPU version to disk
                    disk_embedding = tuple(tensor.detach().cpu() for tensor in embedding)
                else:
                    # For single tensors
                    if self.gpu_memory_cache and embedding.is_cuda:
                        # Keep GPU version in memory, save CPU version to disk
                        memory_embedding = embedding.detach()
                        disk_embedding = embedding.detach().cpu()
                    else:
                        # Store CPU version
                        memory_embedding = disk_embedding = embedding.detach().cpu()

                # Store in memory cache
                if len(self._memory_cache) >= self.memory_cache_size:
                    self._memory_cache.popitem(last=False)  # Remove LRU
                self._memory_cache[cache_key] = memory_embedding

                # Store to disk cache (always CPU for compatibility)
                if self.enable_disk_cache:
                    cache_path = self._get_disk_cache_path(cache_key)
                    torch.save(disk_embedding, cache_path)

                self._stats['saves'] += 1
                return True
            except Exception as e:
                logging.error(f"Error caching embedding: {e}")
                return False

    def clear_all_cache(self):
        """Clear both memory and disk caches."""
        with self._lock:
            self._memory_cache.clear()
            if self.enable_disk_cache:
                for cache_file in self.cache_dir.glob("spk_emb_*.pt"):
                    cache_file.unlink(missing_ok=True)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total = self._stats['memory_hits'] + self._stats['disk_hits'] + self._stats['misses']
            hit_rate = ((self._stats['memory_hits'] + self._stats['disk_hits']) / total * 100) if total > 0 else 0
            gpu_hit_rate = (self._stats['gpu_hits'] / self._stats['memory_hits'] * 100) if self._stats['memory_hits'] > 0 else 0

            return {
                'memory_cache_size': len(self._memory_cache),
                'memory_cache_max_size': self.memory_cache_size,
                'disk_cache_enabled': self.enable_disk_cache,
                'gpu_memory_cache_enabled': self.gpu_memory_cache,
                'cache_directory': str(self.cache_dir),
                'total_requests': total,
                'memory_hits': self._stats['memory_hits'],
                'disk_hits': self._stats['disk_hits'],
                'misses': self._stats['misses'],
                'hit_rate_percent': round(hit_rate, 2),
                'gpu_hits': self._stats['gpu_hits'],
                'gpu_hit_rate_percent': round(gpu_hit_rate, 2),
                'saves': self._stats['saves']
            }

    def print_cache_stats(self):
        """Print formatted cache statistics."""
        stats = self.get_cache_stats()
        print(f"\n=== Speaker Embedding Cache Statistics ===")
        print(f"Memory Cache: {stats['memory_cache_size']}/{stats['memory_cache_max_size']} entries")
        print(f"GPU Memory Cache: {'Enabled' if stats['gpu_memory_cache_enabled'] else 'Disabled'}")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Hit Rate: {stats['hit_rate_percent']}%")
        print(f"GPU Hit Rate: {stats['gpu_hit_rate_percent']}% of memory hits")
        print(f"Cache Directory: {stats['cache_directory']}")
        print("=" * 47)


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
