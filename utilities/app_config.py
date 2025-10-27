"""
Application configuration management
"""
import os
from loguru import logger
from sys import stdout, platform
from typing import Dict, Set, Any, Tuple

from utilities.config_utils import update_model_paths_file, parse_model_paths_file
from utilities.app_constants import ModelConfig

# Global flag to prevent multiple logging initializations
_LOGGING_CONFIGURED = False

class AppConfiguration:
    """Manages application configuration and setup"""
    
    def __init__(self):
        self.debug_mode = False
        self.disable_torch_compile_default = False
        self.models = {}
        self.paths = {}
        self.params = {}    
   
    # LOGGING CONFIGURATION UTILITIES
    def setup_logging(
        self,
        log_to_file: bool = None,
        log_file_path: str = None,
        console_level: str = "INFO",
        file_level: str = "INFO"
    ) -> None:
        """
        Setup standardized logging configuration for SkyrimNet applications.

        Args:
            log_to_file: Whether to enable file logging (checks env var if None)
            log_file_path: Path to log file (uses env var or default if None)
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        global _LOGGING_CONFIGURED

        # Only configure logging once to prevent conflicts
        if _LOGGING_CONFIGURED:
            return

        # Remove ALL existing loggers to avoid conflicts
        logger.remove()

        # Setup console logging with consistent format
        logger.add(
            stdout, 
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", 
            level=console_level, 
            enqueue=True,
            catch=True
        )

        # Determine file logging settings
        if log_to_file is None:
            log_to_file = os.getenv('LOG_TO_FILE', 'false').lower() == 'true'

        if log_file_path is None:
            log_file_path = os.getenv('LOG_FILE_PATH', 'logs/skyrimnet.log')

        # Setup file logging if enabled
        if log_to_file:
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            logger.add(
                log_file_path,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                level=file_level,
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                enqueue=True
            )
            logger.info(f"File logging enabled. Logs will be written to: {log_file_path}")

        # Set the flag to prevent reinitialization
        _LOGGING_CONFIGURED = True


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
            self.models["Zyphra/Zonos-v0.1-hybrid"],
            self.models["Zyphra/Zonos-v0.1-transformer"],
        )
    
    def get_disable_torch_compile_default(self) -> bool:
        """Get the default torch compile setting"""
        return self.params.get("DISABLE_TORCH_COMPILE_DEFAULT", False)
