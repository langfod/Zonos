import math

import numpy as np
import torch
import torchaudio
from transformers.models.dac import DacModel

_GLOBAL_DAC_AUTOENCODER = None

def preload_dac_autoencoder(device: torch.device | None = None, warmup: bool = False):
    """
    Ensure the DAC model weights are loaded early (and optionally CUDA kernels warmed).
    
    This function implements a singleton pattern to ensure only one DAC autoencoder instance
    is created globally, avoiding redundant model loading and memory usage.
    
    Args:
        device (torch.device, optional): Target device to move the DAC model to. 
            If None, model stays on default device.
        warmup (bool, optional): Whether to warm up CUDA kernels by running a dummy 
            forward pass. Defaults to False.
    
    Returns:
        DACAutoencoder: The global DAC autoencoder instance.
        
    Notes:
        This function is idempotent - calling it multiple times returns the same instance.
        The warmup process involves decoding dummy codes to trigger JIT compilation
        and kernel loading for improved performance on subsequent calls.
    """
    global _GLOBAL_DAC_AUTOENCODER
    if _GLOBAL_DAC_AUTOENCODER is None:
        _GLOBAL_DAC_AUTOENCODER = DACAutoencoder()
        if device is not None:
            _GLOBAL_DAC_AUTOENCODER.dac.to(device)
        if warmup:
            with torch.no_grad():
                # minimal dummy codes to force first decode graph / kernel load
                dummy = torch.zeros(
                    1,
                    _GLOBAL_DAC_AUTOENCODER.num_codebooks,
                    1,
                    dtype=torch.long,
                    device=_GLOBAL_DAC_AUTOENCODER.dac.device,
                )
                _ = _GLOBAL_DAC_AUTOENCODER.decode(dummy)
    return _GLOBAL_DAC_AUTOENCODER

class DACAutoencoder:
    """
    A wrapper around the Descript Audio Codec (DAC) model for audio encoding and decoding.
    
    This class provides a convenient interface for DAC operations including audio preprocessing,
    encoding to discrete codes, and decoding back to audio. It handles device management,
    automatic casting, and provides optimized paths for both float and int16 outputs.
    
    The DAC model uses Vector Quantization to compress audio into discrete tokens while
    maintaining high fidelity reconstruction. It operates at 44.1kHz sample rate.
    
    Attributes:
        dac: The underlying DAC model from transformers
        codebook_size: Size of the quantization codebook
        num_codebooks: Number of parallel codebooks used
        sampling_rate: Target sampling rate (44100 Hz)
    """
    def __init__(self):
        """
        Initialize the DAC autoencoder.
        
        Loads the pretrained DAC model from HuggingFace, sets it to evaluation mode,
        and extracts important configuration parameters.
        """
        super().__init__()
        self.dac = DacModel.from_pretrained("descript/dac_44khz")
        self.dac.eval().requires_grad_(False)
        self.codebook_size = self.dac.config.codebook_size
        self.num_codebooks = self.dac.quantizer.n_codebooks
        self.sampling_rate = self.dac.config.sampling_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Preprocess audio for DAC encoding.
        
        Resamples audio to 44.1kHz and pads to ensure the length is a multiple of 512,
        which is required by the DAC model's architecture.
        
        Args:
            wav (torch.Tensor): Input audio tensor of shape [batch, time] or [time]
            sr (int): Original sample rate of the input audio
            
        Returns:
            torch.Tensor: Preprocessed audio tensor at 44.1kHz, padded to multiple of 512 samples
            
        Notes:
            Left padding with zeros is used to maintain temporal alignment while meeting
            the DAC model's length requirements.
        """
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        left_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
        return torch.nn.functional.pad(wav, (left_pad, 0), value=0)

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to discrete codes using DAC quantization.
        
        Args:
            wav (torch.Tensor): Preprocessed audio tensor at 44.1kHz
            
        Returns:
            torch.Tensor: Discrete audio codes of shape [batch, num_codebooks, sequence_length]
            
        Notes:
            The output codes are integers in the range [0, codebook_size-1] representing
            quantized audio features. Multiple codebooks are used in parallel for higher
            fidelity reconstruction.
        """
        return self.dac.encode(wav).audio_codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes back to audio.
        
        Uses automatic mixed precision for better performance on modern GPUs while maintaining
        numerical stability. Falls back to float16 on non-CUDA devices.
        
        Args:
            codes (torch.Tensor): Discrete audio codes of shape [batch, num_codebooks, sequence_length]
            
        Returns:
            torch.Tensor: Reconstructed audio tensor of shape [batch, 1, time] in float32 format
            
        Notes:
            - Uses bfloat16 autocast on CUDA devices for optimal performance
            - Output is automatically converted to float32 for compatibility
            - Adds channel dimension (unsqueeze(1)) for consistency with audio processing pipelines
        """
        # Use bfloat16 for better performance on modern GPUs
        autocast_dtype = torch.bfloat16 if self.dac.device.type == "cuda" else torch.float16
        with torch.autocast(self.dac.device.type, autocast_dtype, enabled=self.dac.device.type != "cpu"):
            return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1).float()

    def decode_to_int16(self, codes: torch.Tensor) -> np.ndarray:
        """
        Decode discrete codes to 16-bit integer audio format.
        
        This method is optimized for applications requiring integer audio output,
        such as file I/O or real-time playback systems. It includes device-aware
        optimizations and efficient memory transfers.
        
        Args:
            codes (torch.Tensor): Discrete audio codes of shape [batch, num_codebooks, sequence_length]
            
        Returns:
            np.ndarray: Reconstructed audio as int16 array, shape [time, 1] for mono audio
            
        Notes:
            - Uses non-blocking GPU transfers when possible for better performance  
            - Applies proper clamping to prevent overflow in int16 conversion
            - Uses mixed precision for optimal GPU utilization
            - Squeezes batch dimension and adds channel dimension for standard audio format
        """
        device = self.dac.device
        codes = codes.to(device, non_blocking=True)  # Use non_blocking transfer

        autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=device.type != "cpu"):
            audio_values = self.dac.decode(audio_codes=codes).audio_values
            # More efficient clamping and conversion
            audio_int16 = torch.clamp(audio_values * 32767.0, -32767.0, 32767.0).to(torch.int16)
            return audio_int16.squeeze(0).unsqueeze(1)
