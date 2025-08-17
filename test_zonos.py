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

from utilities.cache_utils import save_torchaudio_wav
# Local imports - utilities
from utilities.config_utils import (update_model_paths_file, parse_model_paths_file)
from utilities.file_utils import (lcx_checkmodels)
from utilities.report import generate_troubleshooting_report
from utilities.audio_utils import (process_speaker_audio, process_prefix_audio)
from utilities.model_utils import (load_model_if_needed)


# Zonos-specific imports
from zonos.model import DEFAULT_BACKBONE_CLS as ZONOS_BACKBONE
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE

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
    "DEBUG_MODE": False
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
    
def estimate_tokens(text: str) -> int:
    """
    Rough rule: assume ≈ 86 tokens per ~2-second *second*, rounded up.
    We map as:
        every 1 character  ≈ 1 token (char-count itself)
        every paragraph ≈ 1.5
    Then clamp to an upper bound quickly.
    """
    chars  = len(text)
    wpm    = 160                                 # conservative WPM
    secs   = max(1, chars // wpm * 60)           # seconds spoken
    # empirical: ~86 tokens per second of compressed audio
    tokens = int(86 * secs)
    return min(tokens, 3000)

def load_model_wrapper(model_choice: str, disable_torch_compile: bool = disable_torch_compile_default):
    return load_model_if_needed(model_choice, DEFAULT_DEVICE, in_dotenv_needed_models, disable_torch_compile)


async def generate_audio(model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
                  disable_torch_compile=disable_torch_compile_default, do_progress=False, profiling=False):
    """
    Generates audio based on the provided UI parameters.
    """
    #logging.info(f"Requested: \"{text}\"")
    # Start timing the entire function
    #func_start_time = perf_counter_ns()

    # Time the model loading specifically
    #load_start_time = perf_counter_ns()
    selected_model = load_model_wrapper(model_choice)
    #load_end_time = perf_counter_ns()
    #load_duration_ms = (load_end_time - load_start_time) / 1000000  # Convert to milliseconds
    #if load_duration_ms > 0.005:
    #    logging.info(f"Model loading took: {load_duration_ms:.4f} ms")

    # Convert parameters to appropriate types
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

    speaker_embedding_start_time = perf_counter_ns()
    # Process speaker audio if provided
    speaker_embedding = None
    if speaker_audio is not None and "speaker" not in unconditional_keys:
        speaker_embedding = process_speaker_audio(speaker_audio_path=speaker_audio, uuid=uuid, model=selected_model, device=DEFAULT_DEVICE,enable_disk_cache=True)

    # Create conditioning dictionary
    vq_val = [float(vq_single)] * 8 if model_choice != "Zyphra/Zonos-v0.1-hybrid" else None
    cond_dict = make_cond_dict(
        text=text, language=language, speaker=await speaker_embedding if speaker_embedding is not None else None, emotion=[e1, e2, e3, e4, e5, e6, e7, e8],
        vqscore_8=vq_val, fmax=fmax, pitch_std=pitch_std, speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl, speaker_noised=speaker_noised_bool, device=DEFAULT_DEVICE,
        unconditional_keys=unconditional_keys
    )
    conditioning = selected_model.prepare_conditioning(cond_dict)
    speaker_embedding_duration_ms = (perf_counter_ns() - speaker_embedding_start_time) / 1000000
    logging.info(f"speaker_embedding took: {speaker_embedding_duration_ms:.4f} ms")
   
    # Process prefix audio if provided
    audio_prefix_codes = None
    if prefix_audio is not None:
        audio_prefix_codes = process_prefix_audio(prefix_audio_path=prefix_audio, model=selected_model, device=DEFAULT_DEVICE)

    # Generate audio codes
    generate_start_time = perf_counter_ns()
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
                with torch.inference_mode():
                    torch.cuda.nvtx.range_push("infer_single")
                    codes = selected_model.generate(
                        prefix_conditioning=conditioning, audio_prefix_codes=await audio_prefix_codes,
                        max_new_tokens=max_new_tokens, cfg_scale=cfg_scale, batch_size=1,
                        disable_torch_compile=disable_torch_compile,
                        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear,
                           conf=confidence, quad=quadratic)
                    )
                    torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()  # flush kernels before exiting profiler
            end = perf_counter_ns()
            summarize_profiler(prof, out_dir="profile_logs", topk=50)
    else:
        codes = selected_model.generate(
                        prefix_conditioning=conditioning, audio_prefix_codes=await audio_prefix_codes,
                        max_new_tokens=max_new_tokens, cfg_scale=cfg_scale, batch_size=1,
                        disable_torch_compile=disable_torch_compile,
                        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear,
                           conf=confidence, quad=quadratic)
                    )
        end = perf_counter_ns()

    logging.info(f"'generate' took {(end - generate_start_time) /1000000:.4f} ms")
    # Decode audio and convert to numpy
    #wav_np = selected_model.autoencoder.decode_to_int16(codes)
    wav_np = selected_model.autoencoder.decode(codes)

    output_wav_path = save_torchaudio_wav(wav_np.squeeze(0), selected_model.autoencoder.sampling_rate, audio_path=speaker_audio,uuid=uuid)
    # Log execution time
    #func_end_time = perf_counter_ns()
    #total_duration_s = (func_end_time - func_start_time)  / 1_000_000_000  # Convert nanoseconds to seconds
    #wav_length = wav_np.shape[-1]   / selected_model.autoencoder.sampling_rate
    #wav_length = len(wav_np) / selected_model.autoencoder.sampling_rate
    #logging.info(f"Total 'generate_audio' for {speaker_audio} execution time: {total_duration_s:.2f} seconds")
    #logging.info(f"Generated audio length: {wav_length:.2f} seconds {selected_model.autoencoder.sampling_rate}. Speed: {wav_length / total_duration_s:.2f}x")
    stdout.flush()

    #return (selected_model.autoencoder.sampling_rate, wav_np), uuid
    return [await output_wav_path, uuid]

def test_generate_audio(
    model_choice,  # Ignored parameter for Zonos compatibility
    text,  # Main text input
    language = "en-us",  # Ignored parameter
    speaker_audio = None,  # Reference audio file (Gradio file object)
    prefix_audio = "assets/silence_100ms.wav",  # Ignored parameter
    e1 = 1,  # Ignored emotion parameter
    e2 = .05,  # Ignored emotion parameter
    e3 = .05,  # Ignored emotion parameter
    e4 = .05,  # Ignored emotion parameter
    e5 = .05,  # Ignored emotion parameter
    e6 = .05,  # Ignored emotion parameter
    e7 = 0.1,  # Ignored emotion parameter
    e8  = 0.2,  # Ignored emotion parameter
    vq_single  = 0.78,  # Ignored parameter
    fmax  = 24000,  # Ignored parameter
    pitch_std  = 45,  # Ignored parameter
    speaking_rate  = 15,  # Ignored parameter
    dnsmos_ovrl  = 4,  # Ignored parameter
    speaker_noised  = True,  # Ignored parameter
    cfg_scale  = 2,  # Ignored parameter
    top_p =0,  # Can be used for generation
    top_k = 0,  # Ignored parameter
    min_p  = 0 ,  # Ignored parameter
    linear  = 0.5,  # Ignored parameter
    confidence  = 0.4,  # Ignored parameter
    quadratic  = 0,  # Ignored parameter
    seed = 0,  # Can be used for generation
    randomize_seed = False,  # Ignored parameter
    unconditional_keys = ["emotion"],  # Ignored parameter
    profiling = False
):
    return asyncio.run(generate_audio(model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
                  disable_torch_compile=disable_torch_compile_default, profiling=profiling))




# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    modelchoice= "Zyphra/Zonos-v0.1-hybrid"
    # modelchoice = "Zyphra/Zonos-v0.1-transformer"
    load_model_wrapper(modelchoice)
    test_asset=Path.cwd().joinpath("assets", "dlc1seranavoice.wav")
    #test_asset=Path.cwd().joinpath("assets", "fishaudio_horror.wav")
    #test_text="Testing Text. This is great!"
    test_text= "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Ladle it over some fresh Khajiit meat. Now smell that. Oh boy this is going to be incredible."
    try:
        # Run twice to warm model and caches
        [output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=42)

        [output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=42)
        [output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=42)
        [output_wav_path, seed_int] = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=42)

        # Reset for next run
        #sampling_rate, wav_numpy, seed_int = None,  None , 0
        #[sampling_rate, wav_numpy], seed_int = test_generate_audio(model_choice=modelchoice,text=test_text,speaker_audio=test_asset,seed=42,profiling=True)


    except Exception as e:
        print(traceback.format_exc())

    #print("Sleeping for 1 seconds to settle model warmup...")
    #time.sleep(1)
    ##
    #try:
    #    sampling_rate, wav_numpy, seed_int = None,  None , 0# Reset for next run
    #    [sampling_rate, wav_numpy], seed_int = generate_audio(text=test_text,speaker_audio=test_asset,seed=42, profiling=True)
    #    #np.save("test_audio_output",wav_numpy)
    #    #write("test_audio_output.wav", sampling_rate, wav_numpy)
    #    print(f"Generated audio with sampling rate: {sampling_rate}, seed: {seed_int}")
    #except Exception as e:
    #    print(traceback.format_exc())
    exit(0)

