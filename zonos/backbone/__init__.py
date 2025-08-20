BACKBONES = {}

try:
    from ._mamba_ssm import MambaSSMZonosBackbone

    BACKBONES["mamba_ssm"] = MambaSSMZonosBackbone
    BACKBONES["torch"] = MambaSSMZonosBackbone

except ImportError:

    from ._torch import TorchZonosBackbone
    
    BACKBONES["torch"] = TorchZonosBackbone
