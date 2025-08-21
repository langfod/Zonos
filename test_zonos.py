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

from test_utils.audio_graph import plot_audio
from utilities.cache_utils import save_torchaudio_wav
# Local imports - utilities
from utilities.config_utils import (update_model_paths_file, parse_model_paths_file)
from utilities.file_utils import (lcx_checkmodels)
from utilities.report import generate_troubleshooting_report
from utilities.audio_utils import (process_speaker_audio, process_prefix_audio)
from utilities.model_utils import (load_model_if_needed)

from test_utils.model_whisper_utils import (initialize_whisper_model, transcribe_audio_with_whisper)

# Zonos-specific imports
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE

import warnings

# Filter out the specific UserWarning
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.save_with_torchcodec",
    category=UserWarning
)
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

# Application configuration
#debug_mode = True
LCX_APP_NAME = "CROSSOS_FILE_CHECK"
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
# CONFIGURATION PARSING AND SETUP
# =============================================================================

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

# Extract configuration values
disable_torch_compile_default = out_dotenv_loaded_params["DISABLE_TORCH_COMPILE_DEFAULT"]
AI_MODEL_DIR_TF = out_dotenv_loaded_models["Zyphra/Zonos-v0.1-transformer"]
AI_MODEL_DIR_HY = out_dotenv_loaded_models["Zyphra/Zonos-v0.1-hybrid"]

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

def compare_with_baseline(baseline_path: str, cur_codes: torch.tensor):
    if not Path(baseline_path).exists():
        print(f"[compare] Baseline file not found: {baseline_path}")
        return 2
    ref = torch.load(baseline_path, map_location="cpu")
    ref_codes = ref["codes"]
    ref_shape = tuple(ref["shape"])
    cur_shape = tuple(cur_codes.shape)
    # Shape check
    shape_ok = (cur_shape[0] == ref_shape[0] and cur_shape[1] == ref_shape[1] and (abs(cur_shape[2] - ref_shape[2])/ ref_shape[2]) < 0.0099)
    print(f"[compare] baseline shape: {ref_shape} Current shape: {cur_shape} diff: {abs(cur_shape[2] - ref_shape[2])/ ref_shape[2]:.4f} PASS: {shape_ok}")
    # Hamming distance ratio
    min_len = min(ref_codes.shape[-1], cur_codes.shape[-1])
    ref_trim = ref_codes[..., :min_len]
    cur_trim = cur_codes[..., :min_len]
    diff = (ref_trim != cur_trim).sum().item()
    total = ref_trim.numel()
    hamming_ratio = diff / total
    # Token overlap (unique)
    ref_set = set(ref_trim.reshape(-1).tolist())
    cur_set = set(cur_trim.reshape(-1).tolist())
    jaccard = len(ref_set & cur_set) / max(1, len(ref_set | cur_set))
    # Simple summary slice
    #sample_slice = min(32, min_len)
    #ref_head = ref_trim[0, 0, :sample_slice].tolist()
    #cur_head = cur_trim[0, 0, :sample_slice].tolist()
    print(f"[compare] Hamming ratio: {hamming_ratio:.4f} (fraction of differing positions)")
    print(f"[compare] Jaccard token overlap: {jaccard:.4f}")
    #print(f"[compare] First {sample_slice} tokens (codebook 0):")
    #print(f"          baseline: {ref_head}")
    #print(f"          current : {cur_head}")
    # Heuristic pass criteria:
    # 1. Shape matches
    # 2. Hamming ratio not extreme (allow up to 0.999 since sampling may vary heavily)
    # 3. Some minimal overlap (Jaccard > 0.05)
    passed = shape_ok and hamming_ratio < 0.999 and jaccard > 0.05
    if passed:
        print("[compare][PASS] Regression check passed.")
        return 0
    else:
        print("[compare][FAIL] Regression check failed.")
        return 1


def load_model_wrapper(model_choice: str, disable_torch_compile: bool = disable_torch_compile_default):
    return load_model_if_needed(model_choice, DEFAULT_DEVICE, in_dotenv_needed_models, disable_torch_compile)


async def generate_audio(model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
                  disable_torch_compile=disable_torch_compile_default, do_progress=False, profiling=False, baseline_save=False, baseline_compare=False):
    """
    Generates audio based on the provided UI parameters.
    """
    func_start_time = perf_counter_ns()
    

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    top_p = float(top_p)
    top_k = int(top_k)
    min_p = float(min_p)
    linear = float(linear)
    confidence = float(confidence)
    quadratic = float(quadratic)
    seed = int(seed)
    max_new_tokens_ceiling =  2580  # 86 tokens per second, 30 seconds ceiling
    max_new_tokens = min(max(86, 2+(math.ceil(len(text) *6.5))), max_new_tokens_ceiling)

    uuid = seed
    if randomize_seed:
        seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
    torch.manual_seed(seed)
    vq_val = [float(vq_single)] * 8 if model_choice != "Zyphra/Zonos-v0.1-hybrid" else None

    if not profiling:
        selected_model = load_model_wrapper(model_choice)
        speaker_embedding_start_time = perf_counter_ns()
        # Process speaker audio if provided
        speaker_embedding = None
        if speaker_audio is not None and "speaker" not in unconditional_keys:
            speaker_embedding = process_speaker_audio(speaker_audio_path=speaker_audio, uuid=uuid, model=selected_model, device=DEFAULT_DEVICE,enable_disk_cache=True)
    
        # Create conditioning dictionary
        cond_dict = make_cond_dict(
            text=text, language=language, speaker=await speaker_embedding if speaker_embedding is not None else None, emotion=[e1, e2, e3, e4, e5, e6, e7, e8],
            vqscore_8=vq_val, fmax=fmax, pitch_std=pitch_std, speaking_rate=speaking_rate,
            dnsmos_ovrl=dnsmos_ovrl, speaker_noised=speaker_noised_bool, device=DEFAULT_DEVICE,
            unconditional_keys=unconditional_keys
        )
        conditioning = selected_model.prepare_conditioning(cond_dict, cfg_scale=cfg_scale, use_cache=True)
    
        speaker_embedding_duration_ms = (perf_counter_ns() - speaker_embedding_start_time) / 1000000
        logging.info(f"speaker_embedding took: {speaker_embedding_duration_ms:.4f} ms")
       
        # Process prefix audio if provided
        audio_prefix_codes = None
        if prefix_audio is not None:
            audio_prefix_codes = process_prefix_audio(prefix_audio_path=prefix_audio, model=selected_model, device=DEFAULT_DEVICE)
    
        # Generate audio codes
        generate_start_time = perf_counter_ns()
    
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

                selected_model = load_model_wrapper(model_choice)
                prof.step()

                speaker_embedding_start_time = perf_counter_ns()
                # Process speaker audio if provided
                speaker_embedding = None
                if speaker_audio is not None and "speaker" not in unconditional_keys:
                    speaker_embedding = process_speaker_audio(speaker_audio_path=speaker_audio, uuid=uuid, model=selected_model, device=DEFAULT_DEVICE,enable_disk_cache=True)
                    prof.step()
                # Create conditioning dictionary
                cond_dict = make_cond_dict(
                    text=text, language=language, speaker=await speaker_embedding if speaker_embedding is not None else None, emotion=[e1, e2, e3, e4, e5, e6, e7, e8],
                    vqscore_8=vq_val, fmax=fmax, pitch_std=pitch_std, speaking_rate=speaking_rate,
                    dnsmos_ovrl=dnsmos_ovrl, speaker_noised=speaker_noised_bool, device=DEFAULT_DEVICE,
                    unconditional_keys=unconditional_keys
                )
                prof.step()

                conditioning = selected_model.prepare_conditioning(cond_dict, cfg_scale=cfg_scale, use_cache=True)
                prof.step()

                speaker_embedding_duration_ms = (perf_counter_ns() - speaker_embedding_start_time) / 1000000
                #logging.info(f"speaker_embedding took: {speaker_embedding_duration_ms:.4f} ms")
               
                # Process prefix audio if provided
                audio_prefix_codes = None
                if prefix_audio is not None:
                    audio_prefix_codes = await process_prefix_audio(prefix_audio_path=prefix_audio, model=selected_model, device=DEFAULT_DEVICE)
                    prof.step()
                # Generate audio codes
                generate_start_time = perf_counter_ns()

                codes = selected_model.generate(
                    prefix_conditioning=conditioning, audio_prefix_codes=audio_prefix_codes,
                    max_new_tokens=max_new_tokens, cfg_scale=cfg_scale, batch_size=1,
                    disable_torch_compile=disable_torch_compile,
                    sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear,
                       conf=confidence, quad=quadratic), callback=callback
                )
                end1 = perf_counter_ns()
                prof.step()
                codes = selected_model.generate(
                    prefix_conditioning=conditioning, audio_prefix_codes=audio_prefix_codes,
                    max_new_tokens=max_new_tokens, cfg_scale=cfg_scale, batch_size=1,
                    disable_torch_compile=disable_torch_compile,
                    sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear,
                       conf=confidence, quad=quadratic), callback=callback
                )
                end2 = perf_counter_ns()
                torch.cuda.nvtx.range_pop()
        #torch.cuda.synchronize()  # flush kernels before exiting profiler
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
        codes = selected_model.generate(
                        prefix_conditioning=conditioning, audio_prefix_codes=await audio_prefix_codes if audio_prefix_codes is not None else None,
                        max_new_tokens=max_new_tokens, cfg_scale=cfg_scale, batch_size=1,
                        disable_torch_compile=disable_torch_compile,
                        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear,
                           conf=confidence, quad=quadratic)
                    )
        end = perf_counter_ns()
        logging.info(f"'generate' took {(end - generate_start_time) /1000000:.4f} ms")


    # Decode audio and convert to numpy
    wav_np16 = selected_model.autoencoder.decode_to_int16(codes)
    wav_np = selected_model.autoencoder.decode(codes)
    words = transcribe_audio_with_whisper(wav_np.squeeze(0), selected_model.autoencoder.sampling_rate)
    if baseline_save and not baseline_compare:
        torch.save({"codes": codes.cpu(), "shape": tuple(codes.shape), "seed": seed}, "baseline_codes.pt")
        torch.save({"codes": wav_np.cpu(), "shape": tuple(wav_np.cpu().shape), "seed": seed}, "baseline_wav.pt")
    if baseline_compare:
        compare_with_baseline("baseline_codes.pt", codes.cpu())
        compare_with_baseline("baseline_wav.pt", wav_np.cpu())
    output_wav_path = save_torchaudio_wav(wav_np.squeeze(0), selected_model.autoencoder.sampling_rate, audio_path=speaker_audio,uuid=uuid)
    # Log execution time
    func_end_time = perf_counter_ns()
    total_duration_s = (func_end_time - func_start_time)  / 1_000_000_000  # Convert nanoseconds to seconds
    wav_length = wav_np.shape[-1]   / selected_model.autoencoder.sampling_rate
    #wav_length = len(wav_np) / selected_model.autoencoder.sampling_rate
    #logging.info(f"Total 'generate_audio' for {speaker_audio} execution time: {total_duration_s:.2f} seconds")
    logging.info(f"Generated audio length: {wav_length:.2f} seconds {selected_model.autoencoder.sampling_rate}. Speed: {wav_length / total_duration_s:.2f}x")
    stdout.flush()
    plot_audio(wav_np16, selected_model.autoencoder.sampling_rate, words=words, audio_path=speaker_audio, uuid=uuid)

    #return (selected_model.autoencoder.sampling_rate, wav_np), uuid
    return [await output_wav_path, uuid]

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
    fmax  = 22050,
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
    seed = 420,
    randomize_seed = False,
    unconditional_keys = ["emotion"],
    profiling = False,
    baseline_save = False,
    baseline_compare = False

):
    return asyncio.run(generate_audio(model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
                  disable_torch_compile=disable_torch_compile_default, profiling=profiling, baseline_save=baseline_save, baseline_compare=baseline_compare))




# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    initialize_whisper_model()
    #modelchoice= "Zyphra/Zonos-v0.1-hybrid"
    modelchoice = "Zyphra/Zonos-v0.1-transformer"
    #load_model_wrapper(modelchoice)
    test_asset=Path.cwd().joinpath("assets", "dlc1seranavoice.wav")
    #test_asset=Path.cwd().joinpath("assets", "fishaudio_horror.wav")
    #test_text="Testing Text. This is great!"
    test_text= "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible."
    try:
        # Run twice to warm model and caches
        #[output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=4200,baseline_save=True)
        #[output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=4200,baseline_compare=False)
        #[output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=4200,baseline_compare=True)

        # Reset for next run
        sampling_rate, seed_int = None, 0
        [output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=4200,profiling=True,baseline_compare=True)


    except Exception as e:
       print(traceback.format_exc())
       logger.error(f"Error occurred: {e}")
