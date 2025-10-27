"""
Model loading and management utilities for Zonos application.
"""

import os
import torch
from loguru import logger
from typing import Optional
from zonos.model import Zonos
from utilities.config_utils import is_online_model
from torch._inductor.utils import is_big_gpu

CURRENT_MODEL_TYPE: Optional[str] = None
CURRENT_MODEL: Optional[Zonos] = None

def load_model_if_needed(model_choice: str,
                        device: torch.device,
                        needed_models: set, disable_torch_compile:bool = False, reset_compiler: bool = False) -> Zonos:

    global CURRENT_MODEL_TYPE, CURRENT_MODEL

    if CURRENT_MODEL_TYPE != model_choice:
        logger.info(f"Model type changed from {CURRENT_MODEL_TYPE} to {model_choice}. Reloading model...")
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()

        logger.info(f"Loading {model_choice} model...")

        if is_online_model(model_choice, needed_models, debug_mode=False):
            model = Zonos.from_pretrained(model_choice, device=device.type)
        else:
            config_path = f"{model_choice}{os.sep}config.json"
            model_path = f"{model_choice}{os.sep}model.safetensors"
            model = Zonos.from_local(config_path, model_path, device=device.type)

        model.requires_grad_(False).eval()

        if not disable_torch_compile:
            try:
                if is_big_gpu():
                    # High-end GPUs: Use custom options for max performance but disable CUDA graphs
                    # to avoid inference tensor inplace operation conflicts
                    model.autoencoder.decode = torch.compile(
                        model.autoencoder.decode, 
                        fullgraph=True,
                        options={
                            "triton.cudagraphs": False,  # Disable CUDA graphs for this method
                            "max_autotune": True,        # Enable max-autotune optimizations
                            "epilogue_fusion": True,     # Enable operation fusion
                            "max_autotune_pointwise": True  # Aggressive pointwise optimizations
                        }
                    )
                else:
                    # Mid/lower-end GPUs: balanced optimization with default settings
                    model.autoencoder.decode = torch.compile(
                        model.autoencoder.decode, 
                        fullgraph=True, 
                        mode="default",
                        options={
                            "triton.cudagraphs": False,  # Disable CUDA graphs for this method
                            "max_autotune": False,        # Disable max-autotune optimizations
                            "epilogue_fusion": True,     # Enable operation fusion
                            "max_autotune_pointwise": True  # Aggressive pointwise optimizations
                        }
                    )
                if reset_compiler:
                    torch.compiler.reset()
            except Exception as e:
                logger.info(f"Warning: Could not compile the autoencoder decoder. It will run unoptimized. Error: {e}")

            logger.info(f"{model_choice} model loaded successfully!")

        CURRENT_MODEL = model
        CURRENT_MODEL_TYPE = model_choice
    
        from utilities.audio_utils import init_latent_cache
        init_latent_cache() 
    return CURRENT_MODEL


def get_supported_models(backbone_cls, ai_model_dir_hy: str, ai_model_dir_tf: str) -> list[str]:
    """
    Get list of supported models based on available architectures.
    """
    supported_models = []

    if "hybrid" in backbone_cls.supported_architectures:
        supported_models.append(ai_model_dir_hy)
    else:
        print(
            "| The current ZonosBackbone does not support the hybrid architecture, meaning only the transformer model will be available in the model selector.\n"
            "| This probably means the mamba-ssm library has not been installed.")

    if "transformer" in backbone_cls.supported_architectures:
        supported_models.append(ai_model_dir_tf)

    return supported_models

