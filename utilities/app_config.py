"""
Application configuration management
"""
import os
import logging
from sys import stdout, platform
from typing import Dict, Set, Any, Tuple

from utilities.config_utils import update_model_paths_file, parse_model_paths_file
from utilities.app_constants import ModelConfig


class AppConfiguration:
    """Manages application configuration and setup"""
    
    def __init__(self):
        self.debug_mode = False
        self.disable_torch_compile_default = False
        self.models = {}
        self.paths = {}
        self.params = {}
        
    def setup_logging(self):
        """Configure application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(stdout)]
        )
    
    def load_configuration(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load and parse configuration from file"""
        
        # Define needed models and paths
        needed_models = ModelConfig.DEFAULT_MODELS
        needed_paths = {"HF_HOME": ModelConfig.HF_HOME}
        needed_params = {
            "DISABLE_TORCH_COMPILE_DEFAULT": self.disable_torch_compile_default,
            "DEBUG_MODE": False
        }
        
        # Configuration file prefixes
        PREFIX_MODEL = "PATH_MODEL_"
        PREFIX_PATH = "PATH_NEEDED_"
        LOG_PREFIX = "CROSSOS_LOG"
        
        # Update configuration file
        update_model_paths_file(
            needed_models, needed_paths, needed_params,
            ModelConfig.CONFIG_FILE, PREFIX_MODEL, PREFIX_PATH, LOG_PREFIX
        )
        
        # Read back the values
        loaded_models, loaded_paths, loaded_params, loaded_models_values = parse_model_paths_file(
            ModelConfig.CONFIG_FILE, needed_models, needed_paths, 
            PREFIX_MODEL, PREFIX_PATH
        )
        
        # Store configuration
        self.models = loaded_models
        self.paths = loaded_paths  
        self.params = loaded_params
        self.debug_mode = loaded_params.get("DEBUG_MODE", False)
        
        # Set environment variables
        if "HF_HOME" in needed_paths:
            os.environ['HF_HOME'] = loaded_paths["HF_HOME"]
        
        # Debug output if enabled
        if self.debug_mode:
            print("Loaded models:", loaded_models)
            print("Loaded models values:", loaded_models_values)
            print("Loaded paths:", loaded_paths)
            print("Loaded params:", loaded_params)
            
        return loaded_models, loaded_models_values
    
    def get_model_paths(self) -> Tuple[str, str]:
        """Get the transformer and hybrid model paths"""
        return (
            self.models["Zyphra/Zonos-v0.1-transformer"],
            self.models["Zyphra/Zonos-v0.1-hybrid"]
        )
    
    def get_disable_torch_compile_default(self) -> bool:
        """Get the default torch compile setting"""
        return self.params.get("DISABLE_TORCH_COMPILE_DEFAULT", False)
