"""
Model loading and management utilities for Zonos application.
"""
import os
import torch
import logging
from typing import Optional, Union, Dict, Any
from zonos.model import Zonos
from utilities.config_utils import is_online_model
from torch._inductor.utils import is_big_gpu

# Optional DeepSpeed integration
try:
    from utilities.deepspeed_utils import (
        ZonosDeepSpeedEngine, 
        initialize_deepspeed_model, 
        DEEPSPEED_AVAILABLE
    )
    if DEEPSPEED_AVAILABLE:
        logging.info("DeepSpeed integration available")
    else:
        logging.info("DeepSpeed not available (not installed or missing dependencies)")
        ZonosDeepSpeedEngine = None  # For runtime checks
except ImportError as e:
    DEEPSPEED_AVAILABLE = False
    ZonosDeepSpeedEngine = None
    initialize_deepspeed_model = None
    logging.info(f"DeepSpeed integration not available: {e}")

CURRENT_MODEL_TYPE: Optional[str] = None
CURRENT_MODEL: Optional[Any] = None  # Can be Zonos or ZonosDeepSpeedEngine
DEEPSPEED_ENABLED: bool = False

def is_deepspeed_available() -> bool:
    """Check if DeepSpeed is available for use"""
    return DEEPSPEED_AVAILABLE

def load_model_if_needed(model_choice: str,
                        device: torch.device,
                        needed_models: set, 
                        disable_torch_compile: bool = False,
                        enable_deepspeed: bool = False,
                        deepspeed_config: Optional[dict] = None) -> Any:  # Returns Zonos or ZonosDeepSpeedEngine

    global CURRENT_MODEL_TYPE, CURRENT_MODEL, DEEPSPEED_ENABLED

    # Check if we need to reload due to DeepSpeed setting change
    need_reload = (CURRENT_MODEL_TYPE != model_choice or 
                   DEEPSPEED_ENABLED != enable_deepspeed)

    if need_reload:
        logging.info(f"Model type changed from {CURRENT_MODEL_TYPE} to {model_choice}. Reloading model...")
        if enable_deepspeed and not DEEPSPEED_AVAILABLE:
            logging.warning("DeepSpeed requested but not available, falling back to standard model")
            enable_deepspeed = False
            
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()

        logging.info(f"Loading {model_choice} model{'with DeepSpeed' if enable_deepspeed else ''}...")

        # Load the base model first
        if is_online_model(model_choice, needed_models):
            model = Zonos.from_pretrained(model_choice, device=device.type)
        else:
            config_path = f"{model_choice}{os.sep}config.json"
            model_path = f"{model_choice}{os.sep}model.safetensors"
            model = Zonos.from_local(config_path, model_path, device=device.type)

        model.requires_grad_(False).eval()

        # Apply torch.compile if not disabled (only for non-DeepSpeed models)
        if not disable_torch_compile and not enable_deepspeed:
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
                        mode="default"
                    )
            except Exception as e:
                logging.info(f"Warning: Could not compile the autoencoder decoder. It will run unoptimized. Error: {e}")

        # Wrap with DeepSpeed if requested
        if enable_deepspeed and DEEPSPEED_AVAILABLE:
            try:
                model = initialize_deepspeed_model(
                    model=model,
                    phase=1,  # Start with Phase 1
                    dtype="bf16",  # Use BF16 to match model's natural dtype
                    cpu_offload=False,  # Disable CPU offloading for speed priority
                    enable_profiling=True,
                    **(deepspeed_config or {})
                )
                logging.info(f"{model_choice} model loaded successfully with DeepSpeed!")
            except Exception as e:
                logging.error(f"Failed to initialize DeepSpeed, falling back to standard model: {e}")
                # Keep the original model if DeepSpeed fails
        else:
            logging.info(f"{model_choice} model loaded successfully!")

        CURRENT_MODEL = model
        CURRENT_MODEL_TYPE = model_choice
        DEEPSPEED_ENABLED = enable_deepspeed

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
