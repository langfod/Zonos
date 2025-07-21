"""
Model loading and management utilities for Zonos application.
"""
import os
import torch
import logging
from typing import Optional
from zonos.model import Zonos
from utilities.config_utils import is_online_model


def load_model_if_needed(model_choice: str, current_model_type: Optional[str],
                        current_model: Optional[Zonos], device: torch.device,
                        needed_models: set) -> tuple[Zonos, str]:
    """
    Load model if needed, with caching to avoid reloading the same model.

    Returns:
        tuple: (loaded_model, model_type)
    """
    if current_model_type != model_choice:
        if current_model is not None:
            del current_model
            torch.cuda.empty_cache()

        print(f"Loading {model_choice} model...")
        logging.info(f"Loading {model_choice} model...")

        if is_online_model(model_choice, needed_models):
            model = Zonos.from_pretrained(model_choice, device=device)
        else:
            config_path = f"{model_choice}{os.sep}config.json"
            model_path = f"{model_choice}{os.sep}model.safetensors"
            model = Zonos.from_local(config_path, model_path, device=device)

        model.requires_grad_(False).eval()

        # Optimization: compile autoencoder decoder
        logging.info("Compiling the autoencoder decoder for faster waveform generation...")
        try:
            model.autoencoder.decode = torch.compile(
                model.autoencoder.decode, mode="reduce-overhead", fullgraph=True
            )
            logging.info("Decoder compiled successfully!")
        except Exception as e:
            logging.info(f"Warning: Could not compile the autoencoder decoder. It will run unoptimized. Error: {e}")

        logging.info(f"{model_choice} model loaded successfully!")
        print(f"{model_choice} model loaded successfully!")

        return model, model_choice

    return current_model, current_model_type


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
