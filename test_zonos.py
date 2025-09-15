#!/usr/bin/env python3
"""
Zonos Text-to-Speech Application with Gradio Interface
"""
import asyncio
import logging
import math

from argparse import ArgumentParser
from os import environ as os_environ
from sys import (stdout, exit)
from time import  perf_counter_ns

# Third-party imports


import traceback
from pathlib import Path
from scipy.io.wavfile import write
import torch
from loguru import logger
import warnings
#from test_utils.audio_graph import plot_audio
from utilities.cache_utils import save_torchaudio_wav
# Local imports - utilities  
from utilities.app_config import AppConfiguration
from utilities.app_constants import AudioGenerationConfig, PerformanceConfig
from utilities.config_utils import (update_model_paths_file, parse_model_paths_file)
from utilities.file_utils import (lcx_checkmodels)
from utilities.report import generate_troubleshooting_report
from utilities.audio_utils import (process_speaker_audio, process_prefix_audio)
from utilities.model_utils import (load_model_if_needed)
from utilities.audio_generation_pipeline import (
    prepare_generation_params, setup_speaker_conditioning, 
    create_conditioning_dict, setup_prefix_audio, generate_and_save_audio
)

#from test_utils.model_whisper_utils import (initialize_whisper_model, transcribe_audio_with_whisper)

# Zonos-specific imports
from zonos.conditioning import make_cond_dict
from zonos.utilities.utils import DEFAULT_DEVICE
import importlib
skyrimnet_zonos = importlib.import_module("SkyrimNet-Zonos")



# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Platform-specific defaults
de_disable_torch_compile_default = False

# Model and path configuration
in_dotenv_needed_models = {"Zyphra/Zonos-v0.1-hybrid", "Zyphra/Zonos-v0.1-transformer"}
in_dotenv_needed_paths = {"HF_HOME": "./models/hf_download"}
in_dotenv_needed_params = {
    "DISABLE_TORCH_COMPILE_DEFAULT": de_disable_torch_compile_default,
    "DEBUG_MODE": True
}
in_files_to_check_in_paths = []

# Configuration file
in_model_config_file = "configmodel.txt"

# Dotenv prefixes
PREFIX_MODEL = "PATH_MODEL_"
PREFIX_PATH = "PATH_NEEDED_"
LOG_PREFIX = "CROSSOS_LOG"


# =============================================================================
# LOGGING AND DEVICE SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(stdout)]
)

# =============================================================================
# CONFIGURATION SETUP
# =============================================================================

# Initialize configuration using the new AppConfiguration class
config = AppConfiguration()
config.setup_logging()

# Load configuration and models (like SkyrimNet-Zonos.py does)
models_dict, models_values = config.load_configuration()
AI_MODEL_DIR_TF, AI_MODEL_DIR_HY = config.get_model_paths()
disable_torch_compile_default = config.get_disable_torch_compile_default()


# However, for test_zonos.py we'll keep the explicit setup for testing purposes
# while also having access to the centralized config for consistency

# Update the config file
update_model_paths_file(in_dotenv_needed_models, in_dotenv_needed_paths, in_dotenv_needed_params,
                        in_model_config_file, PREFIX_MODEL, PREFIX_PATH, LOG_PREFIX)

# Read back the values
out_dotenv_loaded_models, out_dotenv_loaded_paths, out_dotenv_loaded_params, out_dotenv_loaded_models_values = parse_model_paths_file(
    in_model_config_file, in_dotenv_needed_models, in_dotenv_needed_paths, PREFIX_MODEL, PREFIX_PATH)

if out_dotenv_loaded_params["DEBUG_MODE"]:
    print("Loaded models:", out_dotenv_loaded_models)
    print("Loaded models values:", out_dotenv_loaded_models_values)
    print("Loaded paths:", out_dotenv_loaded_paths)
    print("Loaded params:", out_dotenv_loaded_params)

# Set environment variables
if "HF_HOME" in in_dotenv_needed_paths:
    os_environ['HF_HOME'] = out_dotenv_loaded_paths["HF_HOME"]


# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

parser = ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--output_dir", type=str, default='./outputs')
parser.add_argument("--checkmodels", action='store_true')
parser.add_argument("--integritycheck", action='store_true')
parser.add_argument("--sysreport", action='store_true')
args = parser.parse_args()

# Handle command line options
if args.checkmodels:
    lcx_checkmodels(in_dotenv_needed_models, out_dotenv_loaded_paths, out_dotenv_loaded_models,
                    out_dotenv_loaded_models_values, in_files_to_check_in_paths)

if args.sysreport:
    full_report = generate_troubleshooting_report(in_model_config_file=in_model_config_file)
    print(full_report)
    exit()

if out_dotenv_loaded_params["DEBUG_MODE"]:
    print("---current model paths---------")
    for id in out_dotenv_loaded_models:
        print(f"{id}: {out_dotenv_loaded_models[id]}")

# Configuration values are now loaded from config object above
# disable_torch_compile_default = config.get_disable_torch_compile_default() - already set
# AI_MODEL_DIR_TF, AI_MODEL_DIR_HY = config.get_model_paths() - already set

# =============================================================================
# MAIN APPLICATION FUNCTIONS
# =============================================================================

ALLOW_TF32 = True         # enable TF32 matmul on Ampere+ for faster GEMMs with minimal quality loss

if ALLOW_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # optimize convolution algorithms
    torch.set_float32_matmul_precision("medium")  # hint for matmul precision (PyTorch 2.x)

def summarize_profiler(prof: torch.profiler.profile, out_dir: str = "profile_logs", topk: int = 30):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    def dump(table, name):
        p = Path(out_dir) / name
        with open(p, "w", encoding="utf-8") as f:
            f.write(table)
        logger.info(f"Wrote profiler summary: {p}")

    # Key averages object
    ka = prof.key_averages(group_by_input_shape=False)

    # 1. CUDA (self) time
    cuda_table = ka.table(sort_by="self_cuda_time_total", row_limit=topk)
    dump(cuda_table, "top_cuda_time.txt")

    # 2. CPU (self) time
    cpu_table = ka.table(sort_by="self_cpu_time_total", row_limit=topk)
    dump(cpu_table, "top_cpu_time.txt")

    # 3. CPU + CUDA total time
    total_table = ka.table(sort_by="cuda_time_total", row_limit=topk)
    dump(total_table, "top_total_time.txt")

    # 4. Memory (if enabled)
    try:
        mem_table = ka.table(sort_by="self_cpu_memory_usage", row_limit=topk)
        dump(mem_table, "top_cpu_mem.txt")
        
        # CUDA Memory table
        try:
            cuda_mem_table = ka.table(sort_by="self_cuda_memory_usage", row_limit=topk)
            dump(cuda_mem_table, "top_cuda_mem.txt")
        except Exception:
            pass
    except Exception:
        pass

    # 5. CSV (full)
    # Manual CSV export (EventList has no csv() API)
    csv_path = Path(out_dir) / "key_averages.csv"
    cols = [
        "name",
        "count",
        "cpu_time_total_ms",
        "cpu_time_self_ms",
        "cuda_time_total_ms",
        "cuda_time_self_ms",
        "cpu_mem_self_bytes",
        "cuda_mem_self_bytes",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for evt in ka:
            # Times are reported in microseconds; convert to ms
            def us_to_ms(v):
                return f"{(v or 0.0)/1000.0:.4f}"
            row = [
                evt.key,
                str(evt.count),
                us_to_ms(getattr(evt, "cpu_time_total", 0.0)),
                us_to_ms(getattr(evt, "self_cpu_time_total", 0.0)),
                us_to_ms(getattr(evt, "cuda_time_total", 0.0)),
                us_to_ms(getattr(evt, "self_cuda_time_total", 0.0)),
                str(getattr(evt, "self_cpu_memory_usage", 0)),
                str(getattr(evt, "self_cuda_memory_usage", 0)),
            ]
            f.write(",".join(row) + "\n")
    logger.info(f"Wrote profiler CSV: {csv_path}")
    
    # 6. Memory Analysis Summary
    print("\n=== MEMORY ANALYSIS ===")
    
    # Check current GPU memory state
    if torch.cuda.is_available():
        current_allocated = torch.cuda.memory_allocated() / (1024**2)
        current_reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"Current GPU memory: {current_allocated:.1f} MB allocated, {current_reserved:.1f} MB reserved")
    
    memory_ops = []
    total_cuda_mem = 0
    total_cpu_mem = 0
    
    for evt in ka:
        cuda_mem = getattr(evt, "self_cuda_memory_usage", 0) or 0
        cpu_mem = getattr(evt, "self_cpu_memory_usage", 0) or 0
        if cuda_mem > 0 or cpu_mem > 0:
            memory_ops.append({
                'name': evt.key,
                'count': evt.count,
                'cuda_mem_bytes': cuda_mem,
                'cpu_mem_bytes': cpu_mem,
                'cuda_mem_mb': cuda_mem / (1024**2),
                'cpu_mem_mb': cpu_mem / (1024**2),
                'cuda_time_ms': (getattr(evt, "cuda_time_total", 0) or 0) / 1000.0
            })
            total_cuda_mem += cuda_mem
            total_cpu_mem += cpu_mem
    
    print(f"Profiler captured CUDA memory: {total_cuda_mem / (1024**2):.2f} MB")
    print(f"Profiler captured CPU memory: {total_cpu_mem / (1024**2):.2f} MB")
    
    if memory_ops:
        # Sort by CUDA memory usage
        memory_ops.sort(key=lambda x: x['cuda_mem_bytes'], reverse=True)
        
        print(f"\nTop {min(15, len(memory_ops))} memory-consuming operations:")
        print("Operation | Count | CUDA MB | CPU MB | Time(ms)")
        print("-" * 70)
        
        for i, op in enumerate(memory_ops[:15]):
            name_short = op['name'][:35] + "..." if len(op['name']) > 35 else op['name']
            print(f"{name_short:38} | {op['count']:5} | {op['cuda_mem_mb']:7.2f} | {op['cpu_mem_mb']:6.2f} | {op['cuda_time_ms']:8.2f}")
        
        # Look for optimization-related operations
        opt_keywords = ['where', 'masked', 'scatter', 'workspace', 'copy', 'index', 'gather']
        opt_ops = []
        for op in memory_ops:
            name_lower = op['name'].lower()
            if any(kw in name_lower for kw in opt_keywords):
                opt_ops.append(op)
        
        if opt_ops:
            print(f"\n=== OPTIMIZATION-RELATED OPERATIONS ({len(opt_ops)} found) ===")
            print("Operation | Count | CUDA MB | Time(ms)")
            print("-" * 55)
            for op in opt_ops:
                name_short = op['name'][:40] + "..." if len(op['name']) > 40 else op['name']
                print(f"{name_short:43} | {op['count']:5} | {op['cuda_mem_mb']:7.2f} | {op['cuda_time_ms']:8.2f}")
        else:
            print("\n=== No optimization-specific operations found in memory profile ===")
    else:
        print("No memory usage data found in profiler results")
        
    # Show general profiler statistics
    print(f"\nProfiler Statistics:")
    print(f"  Total events: {len(ka)}")
    print(f"  Events with CUDA memory > 0: {len([op for op in memory_ops if op['cuda_mem_bytes'] > 0])}")
    print(f"  Events with CPU memory > 0: {len([op for op in memory_ops if op['cpu_mem_bytes'] > 0])}")

# compare_with_baseline function removed as it was not useful


# Use load_model_wrapper from skyrimnet_zonos module instead of duplicating
# def load_model_wrapper(model_choice: str, disable_torch_compile: bool = disable_torch_compile_default):
#     # Use the models set from config for consistency
#     model_keys = config.models.keys() if hasattr(config, 'models') and config.models else in_dotenv_needed_models
#     return load_model_if_needed(model_choice, DEFAULT_DEVICE, model_keys, disable_torch_compile)


async def generate_audio_test_wrapper(model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
                  disable_torch_compile=disable_torch_compile_default, do_progress=False, profiling=False):
    """
    Test wrapper for generate_audio that adds profiling and memory tracking capabilities.
    Uses the main skyrimnet_zonos.generate_audio function for most functionality.
    """
    func_start_time = perf_counter_ns()
    
    # For non-profiling runs, use the main generate_audio function
    if not profiling:
        # Import gradio Progress to match the signature
        import gradio as gr
        progress = gr.Progress()
        return await skyrimnet_zonos.generate_audio(
            model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
            vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
            top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
            disable_torch_compile, progress, do_progress
        )
    
    # For profiling, use the enhanced test version below
    return await generate_audio_with_testing_features(
        model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
        vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
        top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
        disable_torch_compile, do_progress, profiling, func_start_time
    )


async def generate_audio_with_testing_features(model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
                  disable_torch_compile=disable_torch_compile_default, do_progress=False, profiling=False, func_start_time=None):
    """
    Enhanced generate_audio function with profiling and memory tracking.
    This function contains the test-specific code that can't be easily reused from the main function.
    """
    if func_start_time is None:
        func_start_time = perf_counter_ns()

    # Prepare generation parameters using the utility function
    emotions = [e1, e2, e3, e4, e5, e6, e7, e8]
    params = prepare_generation_params(
        text=text, seed=seed, randomize_seed=randomize_seed,
        speaker_noised=speaker_noised, vq_single=vq_single,
        fmax=fmax, pitch_std=pitch_std, speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl, cfg_scale=cfg_scale, top_p=top_p,
        top_k=top_k, min_p=min_p, linear=linear, confidence=confidence,
        quadratic=quadratic, disable_torch_compile=disable_torch_compile
    )
    
    uuid = params['seed']
    vq_val = [float(vq_single)] * 8 if model_choice != "Zyphra/Zonos-v0.1-hybrid" else None

    if not profiling:
        selected_model = skyrimnet_zonos.load_model_wrapper(model_choice)
        
        # Setup conditioning using utility functions
        speaker_embedding = await setup_speaker_conditioning(
            speaker_audio, unconditional_keys, uuid, selected_model
        )
        
        cond_dict = create_conditioning_dict(
            text, language, speaker_embedding, emotions, params, unconditional_keys
        )
        
        conditioning = selected_model.prepare_conditioning(
            cond_dict, cfg_scale=params['cfg_scale'], use_cache=True
        )
        
        # Setup prefix audio
        audio_prefix_codes = await setup_prefix_audio(prefix_audio, selected_model)
    
    # Track memory before generation
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024**2)
        reserved_before = torch.cuda.memory_reserved() / (1024**2)
        torch.cuda.reset_peak_memory_stats()  # Reset peak tracking
        
    if profiling:
        # Single-run profiling (Strategy A)
        with torch.profiler.profile(
            activities=[    
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_logs"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as prof:
            def mark_step(*args) -> bool:
                prof.step
                return True
            callback = mark_step

            with torch.inference_mode():
                torch.cuda.nvtx.range_push("infer_single")

                selected_model = skyrimnet_zonos.load_model_wrapper(model_choice)
                prof.step()

                # Setup conditioning using utility functions with profiling steps
                speaker_embedding = await setup_speaker_conditioning(
                    speaker_audio, unconditional_keys, uuid, selected_model
                )
                prof.step()
                
                cond_dict = create_conditioning_dict(
                    text, language, speaker_embedding, emotions, params, unconditional_keys
                )
                prof.step()

                conditioning = selected_model.prepare_conditioning(cond_dict, cfg_scale=params['cfg_scale'], use_cache=False)
                prof.step()
               
                # Process prefix audio if provided
                audio_prefix_codes = await setup_prefix_audio(prefix_audio, selected_model)
                prof.step()
                
                # Generate audio codes
                generate_start_time = perf_counter_ns()

                codes = selected_model.generate(
                    prefix_conditioning=conditioning, audio_prefix_codes=audio_prefix_codes,
                    max_new_tokens=params['max_new_tokens'], cfg_scale=params['cfg_scale'], batch_size=1,
                    disable_torch_compile=disable_torch_compile,
                    sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear,
                       conf=confidence, quad=quadratic), callback=callback
                )
                end1 = perf_counter_ns()
                prof.step()
                codes = selected_model.generate(
                    prefix_conditioning=conditioning, audio_prefix_codes=audio_prefix_codes,
                    max_new_tokens=params['max_new_tokens'], cfg_scale=params['cfg_scale'], batch_size=1,
                    disable_torch_compile=disable_torch_compile,
                    sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear,
                       conf=confidence, quad=quadratic), callback=callback
                )
                end2 = perf_counter_ns()
                torch.cuda.nvtx.range_pop()
        logging.info(f"'generate1' took {(end1 - generate_start_time) /1000000:.4f} ms")
        logging.info(f"'generate2' took {(end2 - end1) /1000000:.4f} ms")
        summarize_profiler(prof, out_dir="profile_logs", topk=50)
        
        # Track memory after generation  
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / (1024**2)
            reserved_after = torch.cuda.memory_reserved() / (1024**2)
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            
            print(f"\n=== MEMORY TRACKING AROUND GENERATION ===")
            print(f"Memory before: {mem_before:.1f} MB allocated, {reserved_before:.1f} MB reserved")
            print(f"Memory after:  {mem_after:.1f} MB allocated, {reserved_after:.1f} MB reserved")
            print(f"Peak memory:   {peak_mem:.1f} MB (max allocated during generation)")
            print(f"Memory delta:  {mem_after - mem_before:+.1f} MB allocated, {reserved_after - reserved_before:+.1f} MB reserved")
            if peak_mem > mem_before + 100:  # Spike of >100MB
                print(f"⚠️  MEMORY SPIKE DETECTED: Peak was {peak_mem - mem_before:.1f} MB above starting memory")
    else:
        generate_start_time = perf_counter_ns()  # Define timing start
        codes = selected_model.generate(
                        prefix_conditioning=conditioning, audio_prefix_codes=audio_prefix_codes,
                        max_new_tokens=params['max_new_tokens'], cfg_scale=params['cfg_scale'], batch_size=1,
                        disable_torch_compile=disable_torch_compile,
                        sampling_params=dict(top_p=params['top_p'], top_k=params['top_k'], min_p=params['min_p'], linear=params['linear'],
                           conf=params['confidence'], quad=params['quadratic'])
                    )
        end = perf_counter_ns()
        logging.info(f"'generate' took {(end - generate_start_time) /1000000:.4f} ms")

    # Decode audio and convert to numpy
    wav_np = selected_model.autoencoder.decode(codes)
    
    output_wav_path = save_torchaudio_wav(wav_np.squeeze(0), selected_model.autoencoder.sampling_rate, audio_path=speaker_audio,uuid=uuid)
    
    # Log performance
    func_end_time = perf_counter_ns()
    total_duration_s = (func_end_time - func_start_time) / 1_000_000_000  # Convert nanoseconds to seconds
    wav_length = wav_np.shape[-1] / selected_model.autoencoder.sampling_rate
    logging.info(f"Generated audio length: {wav_length:.2f} seconds {selected_model.autoencoder.sampling_rate}. Speed: {wav_length / total_duration_s:.2f}x")
    stdout.flush()

    return [output_wav_path, uuid]

def test_generate_audio(
    model_choice,
    text,  # Main text input
    language = "en-us",
    speaker_audio = None,  # Reference audio file (Gradio file object)
    prefix_audio = "assets/silence_100ms.wav",
    e1 = 1,
    e2 = .05, 
    e3 = .05, 
    e4 = .05, 
    e5 = .05, 
    e6 = .05, 
    e7 = 0.1, 
    e8  = 0.2,
    vq_single  = 0.78,
    fmax  = 22050,  # Keep original frequency value
    pitch_std  = 60,
    speaking_rate  = 15,
    dnsmos_ovrl  = 4,
    speaker_noised  = True,
    cfg_scale  = 2,
    top_p =0,
    top_k = 0,
    min_p  = 0 ,
    linear  = 0.5,
    confidence  = 0.4,
    quadratic  = 0,
    seed = PerformanceConfig.DEFAULT_SEED,  # Use constant from PerformanceConfig
    randomize_seed = False,
    unconditional_keys = ["emotion"],
    profiling = False
):
    return asyncio.run(generate_audio_test_wrapper(model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
                  disable_torch_compile=disable_torch_compile_default, profiling=profiling))




# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    #initialize_whisper_model()
    #modelchoice= "Zyphra/Zonos-v0.1-hybrid"
    model_choice = "Zyphra/Zonos-v0.1-transformer"
    model = skyrimnet_zonos.load_model_if_needed(model_choice, DEFAULT_DEVICE, config.models.keys(), disable_torch_compile=False,reset_compiler=True)
    test_asset=Path.cwd().joinpath("assets", "dlc1seranavoice.wav")
    #test_asset=Path.cwd().joinpath("assets", "fishaudio_horror.wav")
    test_text_short="Testing Text. This is great!"
    test_text_long= "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible."
    try:
        # Run twice to warm model and caches
        #[output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=PerformanceConfig.DEFAULT_SEED*10)
        #[output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=PerformanceConfig.DEFAULT_SEED*10)
        #[output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=PerformanceConfig.DEFAULT_SEED*10)

        # Reset for next run
        sampling_rate, seed_int = None, 0

        #[output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=PerformanceConfig.DEFAULT_SEED*10,profiling=True)
        modelchoice = "Zyphra/Zonos-v0.1-transformer"
        [output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text_short,speaker_audio=test_asset,seed=PerformanceConfig.DEFAULT_SEED*10)
        [output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text_long,speaker_audio=test_asset,seed=PerformanceConfig.DEFAULT_SEED*10)
        [output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text_long,speaker_audio=test_asset,seed=PerformanceConfig.DEFAULT_SEED*10)


    except Exception as e:
       print(traceback.format_exc())
       logger.error(f"Error occurred: {e}")
