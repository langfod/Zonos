import math

import numpy as np
import torch
import torchaudio
from transformers.models.dac import DacModel

_GLOBAL_DAC_AUTOENCODER = None

def preload_dac_autoencoder(device: torch.device | None = None, warmup: bool = False):
    """
    Ensure the DAC model weights are loaded early (and optionally CUDA kernels warmed).
    Idempotent: returns existing singleton if already created.
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
    def __init__(self):
        super().__init__()
        self.dac = DacModel.from_pretrained("descript/dac_44khz")
        self.dac.eval().requires_grad_(False)
        self.codebook_size = self.dac.config.codebook_size
        self.num_codebooks = self.dac.quantizer.n_codebooks
        self.sampling_rate = self.dac.config.sampling_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        left_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
        return torch.nn.functional.pad(wav, (left_pad, 0), value=0)

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.dac.encode(wav).audio_codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.dac.device.type, torch.float16, enabled=self.dac.device.type != "cpu"):
            return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1).float()

    def decode_to_int16(self, codes: torch.Tensor) -> np.ndarray:
        device = self.dac.device
        codes = codes.to(device)

        with torch.autocast(self.dac.device.type, dtype=torch.float16, enabled=self.dac.device.type != "cpu"):
            audio_values = self.dac.decode(audio_codes=codes).audio_values
            audio_int16 = (audio_values.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
            return audio_int16.squeeze(0).unsqueeze(1)
