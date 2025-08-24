"""
DeepSpeed integration utilities for Zonos
Provides memory optimization through DeepSpeed-Inference (optional)
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch

# Optional DeepSpeed imports with availability check
try:
    from transformers.integrations import is_deepspeed_available
    
    if is_deepspeed_available():
        import deepspeed
        from deepspeed.inference.config import DeepSpeedInferenceConfig
        from deepspeed.utils import groups as deepspeed_groups
        DEEPSPEED_AVAILABLE = True
    else:
        deepspeed = None
        DeepSpeedInferenceConfig = None
        deepspeed_groups = None
        DEEPSPEED_AVAILABLE = False
        
except ImportError:
    # Fallback for environments without transformers or DeepSpeed
    deepspeed = None
    DeepSpeedInferenceConfig = None
    deepspeed_groups = None
    DEEPSPEED_AVAILABLE = False
    
    def is_deepspeed_available():
        return False


class ZonosDeepSpeedConfig:
    """Configuration management for DeepSpeed integration with Zonos"""
    
    @staticmethod
    def is_available() -> bool:
        """Check if DeepSpeed is available for use"""
        return DEEPSPEED_AVAILABLE
    
    @staticmethod
    def get_phase1_config(
        dtype: str = "bf16",  # Changed to bf16 to match baseline model
        enable_cuda_graph: bool = True,
        cpu_offload: bool = False,
        nvme_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Phase 1: Speed-optimized configuration with memory balance
        
        Args:
            dtype: Data type for inference ("bf16", "fp16", "fp32")
            enable_cuda_graph: Enable CUDA graphs for better performance
            cpu_offload: Enable modest CPU offloading (only if needed)
            nvme_path: Path to NVMe storage for offloading (optional)
            
        Returns:
            DeepSpeed configuration dictionary balanced for speed + memory
            
        Raises:
            RuntimeError: If DeepSpeed is not available
        """
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError(
                "DeepSpeed is not available. Please install DeepSpeed: "
                "pip install deepspeed"
            )
            
        config = {
            "inference": {
                "dtype": dtype,
                "replace_with_kernel_inject": True,
                "enable_cuda_graph": enable_cuda_graph,
                "triangular_masking": False,  # Not needed for Zonos
                "return_tuple": False
            },
            "tensor_parallel": {
                "tp_size": 1  # Single GPU for Phase 1
            }
        }
        
        # Only add memory offloading if explicitly requested (prioritize speed)
        if cpu_offload:
            # Use lighter memory optimization that doesn't hurt performance as much
            config["zero_optimization"] = {
                "stage": 2,  # ZeRO Stage 2 (less aggressive than Stage 3)
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,  # Overlap communication with computation
                "contiguous_gradients": True,
                "allgather_bucket_size": 5e8,  # Larger bucket for better performance
                "reduce_bucket_size": 5e8
            }
            
            if nvme_path:
                config["zero_optimization"]["offload_optimizer"]["nvme_path"] = nvme_path
                
        return config
    
    @staticmethod
    def get_phase2_config(
        dtype: str = "fp16",
        tensor_parallel_size: int = 1,
        enable_advanced_kernels: bool = True
    ) -> Dict[str, Any]:
        """
        Phase 2: Advanced inference optimization configuration
        
        Args:
            dtype: Data type for inference
            tensor_parallel_size: Number of GPUs for tensor parallelism
            enable_advanced_kernels: Enable advanced kernel optimizations
            
        Returns:
            Advanced DeepSpeed inference configuration dictionary
            
        Raises:
            RuntimeError: If DeepSpeed is not available
        """
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError(
                "DeepSpeed is not available. Please install DeepSpeed: "
                "pip install deepspeed"
            )
        config = {
            "inference": {
                "dtype": dtype,
                "replace_with_kernel_inject": True,
                "enable_cuda_graph": True,
                "max_tokens": 2048,  # Zonos typical generation length
                "triangular_masking": False,
                "return_tuple": False
            },
            "tensor_parallel": {
                "tp_size": tensor_parallel_size
            }
        }
        
        if enable_advanced_kernels:
            config["inference"]["kernel_inject"] = True
            config["inference"]["enable_autotuning"] = True
            
        return config


class ZonosDeepSpeedEngine:
    """DeepSpeed inference engine wrapper for Zonos models"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[Dict[str, Any]] = None,
        device: Union[str, torch.device] = "cuda",
        enable_profiling: bool = False
    ):
        """
        Initialize DeepSpeed inference engine
        
        Args:
            model: Zonos model to optimize
            config: DeepSpeed configuration (uses Phase 1 default if None)
            device: Target device
            enable_profiling: Enable performance profiling
            
        Raises:
            RuntimeError: If DeepSpeed is not available
        """
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError(
                "DeepSpeed is not available. Please install DeepSpeed: "
                "pip install deepspeed"
            )
            
        self.original_model = model
        self.device = device
        self.enable_profiling = enable_profiling
        self.config = config or ZonosDeepSpeedConfig.get_phase1_config()
        
        # Store original methods for comparison
        self.original_generate = model.generate
        self.original_autoencoder = model.autoencoder
        
        # Initialize DeepSpeed engine
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the DeepSpeed inference engine"""
        try:
            # Detect the model's actual dtype from its parameters
            model_dtype = next(self.original_model.parameters()).dtype
            logging.info(f"Detected model dtype: {model_dtype}")
            
            # Ensure model is on the correct device (but keep original dtype)
            self.original_model = self.original_model.to(device=self.device)
            
            logging.info("Initializing DeepSpeed inference engine...")
            logging.info(f"Using model dtype: {model_dtype}")
            logging.info(f"DeepSpeed config: {json.dumps(self.config, indent=2)}")
            
            # Use DeepSpeed inference API for speed optimization
            # This provides better speed/memory balance than training API
            logging.info("Using DeepSpeed init_inference for speed optimization")
            self.engine = deepspeed.init_inference(
                model=self.original_model,
                mp_size=self.config["tensor_parallel"]["tp_size"],  
                dtype=model_dtype,  
                replace_with_kernel_inject=self.config["inference"]["replace_with_kernel_inject"],
            )
            
            logging.info("DeepSpeed inference engine initialized successfully!")
            
            # Store performance baseline if profiling enabled
            if self.enable_profiling:
                self._log_memory_usage("after_initialization")
                
        except Exception as e:
            logging.error(f"Failed to initialize DeepSpeed engine: {e}")
            logging.warning("Falling back to original model")
            self.engine = self.original_model
    
    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype"""
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16, 
            "fp32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    def _log_memory_usage(self, stage: str):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logging.info(f"GPU Memory {stage}: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
    
    def generate(self, *args, **kwargs):
        """Generate method with DeepSpeed optimization"""
        if self.enable_profiling:
            self._log_memory_usage("before_generation")
            
        # Use DeepSpeed optimized model - access underlying model through .module
        if hasattr(self.engine, 'module'):
            result = self.engine.module.generate(*args, **kwargs)
        else:
            # Fallback to original model if not wrapped
            result = self.original_model.generate(*args, **kwargs)
        
        if self.enable_profiling:
            self._log_memory_usage("after_generation")
            
        return result
    
    @property
    def autoencoder(self):
        """Access to autoencoder (unchanged by DeepSpeed)"""
        return self.original_autoencoder
    
    def prepare_conditioning(self, *args, **kwargs):
        """Conditioning preparation (delegated to underlying model)"""
        # DeepSpeed engine wraps the model in .module
        if hasattr(self.engine, 'module'):
            return self.engine.module.prepare_conditioning(*args, **kwargs)
        else:
            # Fallback to original model if not wrapped
            return self.original_model.prepare_conditioning(*args, **kwargs)
    
    def make_speaker_embedding(self, *args, **kwargs):
        """Speaker embedding creation (delegated to underlying model)"""
        # DeepSpeed engine wraps the model in .module
        if hasattr(self.engine, 'module'):
            return self.engine.module.make_speaker_embedding(*args, **kwargs)
        else:
            # Fallback to original model if not wrapped
            return self.original_model.make_speaker_embedding(*args, **kwargs)


def initialize_deepspeed_model(
    model: torch.nn.Module,
    phase: int = 1,
    dtype: str = "fp16",
    cpu_offload: bool = False,
    enable_profiling: bool = False,
    **kwargs
) -> ZonosDeepSpeedEngine:
    """
    Initialize a Zonos model with DeepSpeed optimization
    
    Args:
        model: Zonos model to optimize
        phase: DeepSpeed integration phase (1 or 2)
        dtype: Data type for inference
        cpu_offload: Enable CPU offloading
        enable_profiling: Enable performance profiling
        **kwargs: Additional configuration options
        
    Returns:
        Optimized model wrapped in ZonosDeepSpeedEngine
        
    Raises:
        RuntimeError: If DeepSpeed is not available
    """
    if not DEEPSPEED_AVAILABLE:
        raise RuntimeError(
            "DeepSpeed is not available. Please install DeepSpeed: "
            "pip install deepspeed"
        )
        
    if phase == 1:
        config = ZonosDeepSpeedConfig.get_phase1_config(
            dtype=dtype,
            cpu_offload=cpu_offload,
            **kwargs
        )
    elif phase == 2:
        config = ZonosDeepSpeedConfig.get_phase2_config(
            dtype=dtype,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported phase: {phase}. Use 1 or 2.")
    
    return ZonosDeepSpeedEngine(
        model=model,
        config=config,
        enable_profiling=enable_profiling
    )


def compare_performance(
    original_model: torch.nn.Module,
    deepspeed_model: ZonosDeepSpeedEngine,
    test_inputs: Dict[str, Any],
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Compare performance between original and DeepSpeed-optimized models
    
    Args:
        original_model: Original Zonos model
        deepspeed_model: DeepSpeed-optimized model
        test_inputs: Test inputs for generation
        num_runs: Number of test runs for averaging
        
    Returns:
        Performance comparison results
    """
    import time
    
    results = {
        "original": {"times": [], "memory_usage": []},
        "deepspeed": {"times": [], "memory_usage": []}
    }
    
    # Test original model
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.perf_counter()
        _ = original_model.generate(**test_inputs)
        end_time = time.perf_counter()
        
        results["original"]["times"].append(end_time - start_time)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            results["original"]["memory_usage"].append(peak_memory)
    
    # Test DeepSpeed model
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.perf_counter()
        _ = deepspeed_model.generate(**test_inputs)
        end_time = time.perf_counter()
        
        results["deepspeed"]["times"].append(end_time - start_time)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            results["deepspeed"]["memory_usage"].append(peak_memory)
    
    # Calculate averages and improvements
    avg_original_time = sum(results["original"]["times"]) / num_runs
    avg_deepspeed_time = sum(results["deepspeed"]["times"]) / num_runs
    
    avg_original_memory = sum(results["original"]["memory_usage"]) / num_runs if results["original"]["memory_usage"] else 0
    avg_deepspeed_memory = sum(results["deepspeed"]["memory_usage"]) / num_runs if results["deepspeed"]["memory_usage"] else 0
    
    speedup = avg_original_time / avg_deepspeed_time if avg_deepspeed_time > 0 else 1.0
    memory_reduction = (avg_original_memory - avg_deepspeed_memory) / avg_original_memory if avg_original_memory > 0 else 0.0
    
    return {
        "original_avg_time": avg_original_time,
        "deepspeed_avg_time": avg_deepspeed_time,
        "speedup": speedup,
        "original_avg_memory": avg_original_memory,
        "deepspeed_avg_memory": avg_deepspeed_memory,
        "memory_reduction_pct": memory_reduction * 100,
        "raw_results": results
    }
