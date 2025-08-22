"""
Application constants for Gradio-Zonos
"""

class AudioGenerationConfig:
    """Constants for audio generation"""
    TOKENS_PER_SECOND = 86
    MAX_DURATION_SECONDS = 30
    MAX_NEW_TOKENS_CEILING = 2580  # 86 tokens per second, 30 seconds ceiling
    MIN_TOKENS = 86
    TEXT_TO_TOKENS_MULTIPLIER = 6.5
    TOKEN_BUFFER = 2

class UIConfig:
    """Constants for UI components"""
    TEXT_INPUT_LINES = 4
    TEXT_MAX_LENGTH = 500
    GENERATION_CONCURRENCY_LIMIT = 2
    
    # Slider ranges and defaults
    DNSMOS_RANGE = (1.0, 5.0, 4.0, 0.1)
    FMAX_RANGE = (0, 24000, 24000, 1)
    VQ_SCORE_RANGE = (0.5, 0.8, 0.78, 0.01)
    PITCH_STD_RANGE = (0.0, 300.0, 45.0, 1)
    SPEAKING_RATE_RANGE = (5.0, 30.0, 15.0, 0.5)
    CFG_SCALE_RANGE = (1.0, 5.0, 2.0, 0.1)

class PerformanceConfig:
    """Constants for performance monitoring"""
    MIN_TIMING_THRESHOLD_MS = 0.005
    SEED_MAX = 2 ** 32 - 1
    DEFAULT_SEED = 420
    
class ModelConfig:
    """Model-related constants"""
    DEFAULT_MODELS = {"Zyphra/Zonos-v0.1-transformer","Zyphra/Zonos-v0.1-hybrid"}
    CONFIG_FILE = "configmodel.txt"
    HF_HOME = "./models/hf_download"
