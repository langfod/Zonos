"""
Audio generation pipeline utilities
"""
import math
from loguru import logger
from time import perf_counter_ns
from typing import Optional, Callable, Dict, Any, Tuple

import torch
import gradio as gr

from utilities.app_constants import AudioGenerationConfig, PerformanceConfig
from utilities.audio_utils import process_speaker_audio, process_prefix_audio
from utilities.cache_utils import save_torchaudio_wav
from zonos.conditioning import make_cond_dict
from zonos.utilities.utils import DEFAULT_DEVICE


class PerformanceTimer:
    """Utility class for performance timing"""
    
    def __init__(self, name: str, threshold_ms: float = PerformanceConfig.MIN_TIMING_THRESHOLD_MS):
        self.name = name
        self.threshold_ms = threshold_ms
        self.start_time = None
    
    def __enter__(self):
        self.start_time = perf_counter_ns()
        return self
    
    def __exit__(self, *args):
        if self.start_time:
            duration_ms = (perf_counter_ns() - self.start_time) / 1000000
            if duration_ms > self.threshold_ms:
                logger.info(f"{self.name} took: {duration_ms:.4f} ms")


def prepare_generation_params(text: str, seed: int, randomize_seed: bool, 
                            speaker_noised: bool, **kwargs) -> Dict[str, Any]:
    """Convert and validate generation parameters"""
    
    # Handle seed randomization
    final_seed = seed
    if randomize_seed:
        final_seed = torch.randint(0, PerformanceConfig.SEED_MAX, (1,)).item()
    torch.manual_seed(final_seed)
    
    # Calculate max tokens based on text length
    max_new_tokens = min(
        max(
            AudioGenerationConfig.MIN_TOKENS, 
            AudioGenerationConfig.TOKEN_BUFFER + math.ceil(len(text) * AudioGenerationConfig.TEXT_TO_TOKENS_MULTIPLIER)
        ), 
        AudioGenerationConfig.MAX_NEW_TOKENS_CEILING
    )
    
    # Convert parameters to appropriate types
    params = {
        'seed': final_seed,
        'max_new_tokens': max_new_tokens,
        'speaker_noised': bool(speaker_noised),
        'fmax': float(kwargs.get('fmax', 24000)),
        'pitch_std': float(kwargs.get('pitch_std', 45.0)),
        'speaking_rate': float(kwargs.get('speaking_rate', 15.0)),
        'dnsmos_ovrl': float(kwargs.get('dnsmos_ovrl', 4.0)),
        'cfg_scale': float(kwargs.get('cfg_scale', 2.0)),
        'top_p': float(kwargs.get('top_p', 0)),
        'top_k': int(kwargs.get('top_k', 0)),
        'min_p': float(kwargs.get('min_p', 0)),
        'linear': float(kwargs.get('linear', 0.5)),
        'confidence': float(kwargs.get('confidence', 0.40)),
        'quadratic': float(kwargs.get('quadratic', 0.00))
    }
    
    return params


async def setup_speaker_conditioning(speaker_audio: Optional[str], unconditional_keys: list, model, enable_disk_cache: bool = True) -> Optional[Any]:
    """Process speaker audio for conditioning"""
    if speaker_audio is None or "speaker" in unconditional_keys:
        return None
    
    with PerformanceTimer("speaker_embedding"):
        return await process_speaker_audio(
            speaker_audio_path=speaker_audio, 
            device=DEFAULT_DEVICE,
            enable_disk_cache=enable_disk_cache
        )


def create_conditioning_dict(text: str, language: str, speaker_embedding: Optional[Any],
                           emotions: list, params: Dict[str, Any], unconditional_keys: list) -> Dict[str, Any]:
    """Create the conditioning dictionary for generation"""
    vq_val = [params['vq_single']] * 8 if 'vq_single' in params else None
    
    return make_cond_dict(
        text=text, 
        language=language, 
        speaker=speaker_embedding,
        emotion=emotions,
        vqscore_8=vq_val, 
        fmax=params['fmax'], 
        pitch_std=params['pitch_std'], 
        speaking_rate=params['speaking_rate'],
        dnsmos_ovrl=params['dnsmos_ovrl'], 
        speaker_noised=params['speaker_noised'], 
        device=DEFAULT_DEVICE,
        unconditional_keys=unconditional_keys
    )


async def setup_prefix_audio(prefix_audio: Optional[str], model) -> Optional[Any]:
    """Process prefix audio if provided"""
    if prefix_audio is None:
        return None
    return await process_prefix_audio(
        prefix_audio_path=prefix_audio, 
        model=model, 
        device=DEFAULT_DEVICE
    )


def create_progress_callback(do_progress: bool, text: str, progress: gr.Progress) -> Optional[Callable]:
    """Create progress callback if needed"""
    if not do_progress:
        return None
    
    estimated_generation_duration = 30 * len(text) / 400
    estimated_total_steps = int(estimated_generation_duration * AudioGenerationConfig.TOKENS_PER_SECOND)

    def update_progress(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
        progress((step, estimated_total_steps))
        return True
    
    return update_progress


def generate_and_save_audio(model, conditioning: torch.Tensor, params: Dict[str, Any], 
                          audio_prefix_codes: Optional[Any], callback: Optional[Callable],
                          speaker_audio: Optional[str]) -> Tuple[str, float]:
    """Execute the main generation and save the result"""
    
    with PerformanceTimer("generate"):
        codes = model.generate(
            prefix_conditioning=conditioning, 
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=params['max_new_tokens'], 
            cfg_scale=params['cfg_scale'], 
            batch_size=1,
            disable_torch_compile=params.get('disable_torch_compile', False),
            sampling_params={
                'top_p': params['top_p'], 
                'top_k': params['top_k'], 
                'min_p': params['min_p'], 
                'linear': params['linear'], 
                'conf': params['confidence'], 
                'quad': params['quadratic']
            },
            callback=callback
        )

    wav_np = model.autoencoder.decode(codes)
    output_wav_path = save_torchaudio_wav(
        wav_np.squeeze(0), 
        model.autoencoder.sampling_rate, 
        audio_path=speaker_audio,
    )
    
    wav_length = wav_np.shape[-1] / model.autoencoder.sampling_rate
    return output_wav_path, wav_length

