"""
Backbone architecture registry for Zonos models.

This module provides a registry of available backbone implementations that can be used
with Zonos models. It handles conditional imports to gracefully fall back when
optional dependencies (like mamba-ssm) are not available.

The BACKBONES dictionary maps string identifiers to backbone classes:
- "mamba_ssm": MambaSSMZonosBackbone (hybrid transformer-mamba, requires mamba-ssm)
- "torch": TorchZonosBackbone (pure transformer) or MambaSSMZonosBackbone (if available)

The module attempts to import mamba-ssm first, falling back to pure PyTorch implementation
if the mamba-ssm package is not installed.
"""

import traceback
import os
if os.name == 'nt' and os.environ.get('CUDA_PATH'):
    dll_path = os.path.join(os.environ['CUDA_PATH'], 'bin')
    print(dll_path)
    if os.path.isdir(dll_path):
        os.add_dll_directory(dll_path)

BACKBONES = {}

try:
    from ._mamba_ssm import MambaSSMZonosBackbone

    BACKBONES["mamba_ssm"] = MambaSSMZonosBackbone
    BACKBONES["torch"] = MambaSSMZonosBackbone

except ImportError:
    print(traceback.format_exc())
    from ._torch import TorchZonosBackbone
    
    BACKBONES["torch"] = TorchZonosBackbone
