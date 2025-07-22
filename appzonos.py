#!/usr/bin/env python3
"""
Zonos Text-to-Speech Application with Gradio Interface
Enhanced with disk and memory caching for speaker embeddings
"""

# Standard library imports
import os
import sys
import time
import argparse
import logging

# Third-party imports
import torch
import gradio as gr

# Local imports - utilities
from utilities.config_utils import (update_model_paths_file, parse_model_paths_file, is_online_model)
from utilities.file_utils import (lcx_checkmodels, get_cache_dir)
from utilities.report import generate_troubleshooting_report
from utilities.system_info import (get_gpu_device)
from utilities.audio_utils import (process_speaker_audio, process_prefix_audio, convert_audio_to_numpy,
                                 configure_speaker_cache, get_speaker_cache_stats, print_speaker_cache_stats,
                                 clear_speaker_cache)
from utilities.model_utils import (load_model_if_needed, get_supported_models)
from utilities.gradio_utils import (update_ui_visibility)

# Zonos-specific imports
from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================
# Fix torch.compile C++ compilation issues on Windows
if sys.platform == "win32":
    os.environ["TORCH_COMPILE_CPP_FORCE_X64"] = "1"
    # Alternative approach - force specific compiler architecture
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"

# Platform-specific defaults
de_disable_torch_compile_default = False
if sys.platform == "linux":
    de_disable_torch_compile_default = False
if sys.platform == "darwin":
    de_disable_torch_compile_default = False

# Model and path configuration
in_dotenv_needed_models = {"Zyphra/Zonos-v0.1-hybrid", "Zyphra/Zonos-v0.1-transformer"}
in_dotenv_needed_paths = {"HF_HOME": "./models/hf_download"}
in_dotenv_needed_params = {
    "DISABLE_TORCH_COMPILE_DEFAULT": de_disable_torch_compile_default,
    "DEBUG_MODE": False,
    "CACHE_DIR": "./cache/speaker_embeddings",
    "CACHE_MEMORY_SIZE": 100,
    "CACHE_EXPIRY_HOURS": 24 * 7  # 1 week
}
in_files_to_check_in_paths = []

# Application configuration
debug_mode = True
LCX_APP_NAME = "CROSSOS_FILE_CHECK"
in_model_config_file = "configmodel.txt"

# Dotenv prefixes
PREFIX_MODEL = "PATH_MODEL_"
PREFIX_PATH = "PATH_NEEDED_"
LOG_PREFIX = "CROSSOS_LOG"

# Global model state
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None
SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None
SPEAKER_AUDIO_PATH_DICT = {}

# =============================================================================
# LOGGING AND DEVICE SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Initialize GPU device
cpu = torch.device('cpu')
try:
    gpu = get_gpu_device()
except RuntimeError as e:
    logging.error(f"GPU initialization failed: {e}")
    gpu = None

# =============================================================================
# CONFIGURATION PARSING AND SETUP
# =============================================================================

# Update the config file
update_model_paths_file(in_dotenv_needed_models, in_dotenv_needed_paths, in_dotenv_needed_params,
                        in_model_config_file, PREFIX_MODEL, PREFIX_PATH, LOG_PREFIX)

# Read back the values
out_dotenv_loaded_models, out_dotenv_loaded_paths, out_dotenv_loaded_params, out_dotenv_loaded_models_values = parse_model_paths_file(
    in_model_config_file, in_dotenv_needed_models, in_dotenv_needed_paths, PREFIX_MODEL, PREFIX_PATH)

if debug_mode:
    print("Loaded models:", out_dotenv_loaded_models)
    print("Loaded models values:", out_dotenv_loaded_models_values)
    print("Loaded paths:", out_dotenv_loaded_paths)
    print("Loaded params:", out_dotenv_loaded_params)

# Set environment variables
if "HF_HOME" in in_dotenv_needed_paths:
    os.environ['HF_HOME'] = out_dotenv_loaded_paths["HF_HOME"]
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

# Configure speaker embedding cache to use local script directory
cache_dir = out_dotenv_loaded_params.get("CACHE_DIR")
cache_memory_size = int(out_dotenv_loaded_params.get("CACHE_MEMORY_SIZE", 100))
cache_expiry_hours = float(out_dotenv_loaded_params.get("CACHE_EXPIRY_HOURS", 24 * 7))

# Ensure cache directory exists and is writable

logging.info(f"Cache directory set to: {cache_dir}")

try:
    configure_speaker_cache(
        cache_dir=cache_dir,
        memory_cache_size=cache_memory_size,
        enable_disk_cache=True,
        cache_expiry_hours=cache_expiry_hours
    )
    logging.info(f"Speaker embedding cache configured: {cache_dir}")
except Exception as e:
    logging.warning(f"Failed to configure speaker cache: {e}")

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--output_dir", type=str, default='./outputs')
parser.add_argument("--checkmodels", action='store_true')
parser.add_argument("--integritycheck", action='store_true')
parser.add_argument("--sysreport", action='store_true')
parser.add_argument("--cache-stats", action='store_true', help='Show speaker cache statistics')
parser.add_argument("--clear-cache", action='store_true', help='Clear speaker embedding cache')
parser.add_argument("--disable-cache", action='store_true', help='Disable speaker embedding caching')
args = parser.parse_args()

# Handle command line options
if args.checkmodels:
    lcx_checkmodels(in_dotenv_needed_models, out_dotenv_loaded_paths, out_dotenv_loaded_models,
                    out_dotenv_loaded_models_values, in_files_to_check_in_paths)

if args.sysreport:
    full_report = generate_troubleshooting_report(in_model_config_file=in_model_config_file)
    print(full_report)
    sys.exit()

if args.cache_stats:
    print_speaker_cache_stats()
    sys.exit()

if args.clear_cache:
    clear_speaker_cache()
    print("Speaker embedding cache cleared.")
    sys.exit()

if debug_mode:
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

def load_model_wrapper(model_choice: str):
    """Wrapper function for model loading that maintains global state."""
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    CURRENT_MODEL, CURRENT_MODEL_TYPE = load_model_if_needed(
        model_choice, CURRENT_MODEL_TYPE, CURRENT_MODEL, device, in_dotenv_needed_models)
    return CURRENT_MODEL


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    """
    model = load_model_wrapper(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    return update_ui_visibility(model, cond_names)


def generate_audio(model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
                  disable_torch_compile=disable_torch_compile_default, progress=gr.Progress()):
    """
    Generates audio based on the provided UI parameters with enhanced caching.
    """
    logging.info(f"inputs: model_choice={model_choice}, text={text}, language={language}, speaker_audio={speaker_audio}, seed={seed}")
    # Start timing the entire function
    func_start_time = time.perf_counter()

    # Time the model loading specifically
    #logging.info("Checking/loading model...")
    load_start_time = time.perf_counter()
    selected_model = load_model_wrapper(model_choice)
    load_end_time = time.perf_counter()
    load_duration_ms = (load_end_time - load_start_time) * 1000
    if load_duration_ms > .005:
        logging.info(f"Model loading took: {load_duration_ms:.10f} ms")

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
    max_new_tokens = 86 * 30

    # Handle speaker audio caching
    global SPEAKER_AUDIO_PATH, SPEAKER_AUDIO_PATH_DICT, SPEAKER_EMBEDDING

    speaker_audio_uuid = seed
    if randomize_seed:
        seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
    torch.manual_seed(seed)

    # Process speaker audio if provided (with enhanced caching)
    if speaker_audio is not None and "speaker" not in unconditional_keys:
        if speaker_audio not in SPEAKER_AUDIO_PATH_DICT:
            SPEAKER_EMBEDDING = process_speaker_audio(speaker_audio, selected_model, device, SPEAKER_AUDIO_PATH_DICT,speaker_audio_uuid, model_choice)
            SPEAKER_AUDIO_PATH = speaker_audio
        else:
            if speaker_audio != SPEAKER_AUDIO_PATH:
                SPEAKER_AUDIO_PATH = speaker_audio
                SPEAKER_EMBEDDING = SPEAKER_AUDIO_PATH_DICT[speaker_audio]

    # Process prefix audio if provided
    audio_prefix_codes = None
    if prefix_audio is not None:
        audio_prefix_codes = process_prefix_audio(prefix_audio, selected_model, device)

    # Prepare emotion and VQ tensors
    emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)
    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    # Create conditioning dictionary
    cond_dict = make_cond_dict(
        text=text, language=language, speaker=SPEAKER_EMBEDDING, emotion=emotion_tensor,
        vqscore_8=vq_tensor, fmax=fmax, pitch_std=pitch_std, speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl, speaker_noised=speaker_noised_bool, device=device,
        unconditional_keys=unconditional_keys
    )
    conditioning = selected_model.prepare_conditioning(cond_dict)

    # Progress tracking
    estimated_generation_duration = 30 * len(text) / 400
    estimated_total_steps = int(estimated_generation_duration * 86)

    def update_progress(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
        progress((step, estimated_total_steps))
        return True

    # Generate audio codes
    codes = selected_model.generate(
        prefix_conditioning=conditioning, audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens, cfg_scale=cfg_scale, batch_size=1,
        disable_torch_compile=disable_torch_compile,
        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear,
                           conf=confidence, quad=quadratic),
        callback=update_progress,
    )

    # Decode audio and convert to numpy
    #wav_gpu_f32 = selected_model.autoencoder.decode(codes)
    #sr_out, wav_np = convert_audio_to_numpy(wav_gpu_f32, selected_model.autoencoder.sampling_rate)
    wav_np = selected_model.autoencoder.decode_to_int16(codes)

    # Log execution time
    func_end_time = time.perf_counter()
    total_duration_s = func_end_time - func_start_time
    if speaker_audio:
        logging.info(f"Total 'generate_audio' for {speaker_audio.split('\\')[-1]} execution time: {total_duration_s:.2f} seconds")
    else:
        logging.info(f"Total 'generate_audio' execution time: {total_duration_s:.2f} seconds")
    sys.stdout.flush()

    return (selected_model.autoencoder.sampling_rate, wav_np), seed


def show_cache_stats():
    """Return cache statistics as a formatted string."""
    try:
        stats = get_speaker_cache_stats()
        if stats.get("cache_available", True):
            return f"""
**Speaker Embedding Cache Statistics:**
- Memory Cache: {stats['memory_cache_size']}/{stats['memory_cache_max_size']} entries
- Total Requests: {stats['total_requests']}
- Hit Rate: {stats['hit_rate_percent']}%
- Memory Hits: {stats['memory_hits']}
- Disk Hits: {stats['disk_hits']}
- Misses: {stats['misses']}
- Cache Directory: {stats['cache_directory']}
            """
        else:
            return "Speaker embedding cache is not available."
    except Exception as e:
        return f"Error retrieving cache stats: {e}"


def clear_cache_interface():
    """Clear the speaker embedding cache and return confirmation."""
    try:
        clear_speaker_cache()
        return "✅ Speaker embedding cache cleared successfully!"
    except Exception as e:
        return f"❌ Error clearing cache: {e}"


def build_interface():
    """Build and return the Gradio interface with cache management."""
    supported_models = get_supported_models(ZonosBackbone, AI_MODEL_DIR_HY, AI_MODEL_DIR_TF)

    with gr.Blocks(analytics_enabled=False, title="Zonos TTS with Enhanced Caching") as demo:
        gr.Markdown("# Zonos Text-to-Speech with Speaker Embedding Cache")

        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(choices=supported_models, value=supported_models[0],
                    label="Zonos Model Selection")
                text = gr.Textbox(label="Text to Synthesize",
                    value="Zonos uses eSpeak for text to phoneme conversion!",
                    lines=4, max_length=500)
                language = gr.Dropdown(choices=supported_language_codes, value="en-us", label="Language Code")
            prefix_audio = gr.Audio(value="assets/silence_100ms.wav",
                label="Optional Prefix Audio (continue from this audio)", type="filepath")
            with gr.Column():
                speaker_audio = gr.Audio(label="Optional Speaker Audio (for cloning)", type="filepath")
                speaker_noised_checkbox = gr.Checkbox(label="Denoise Speaker? (only Hybrid model)", value=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                dnsmos_slider = gr.Slider(1.0, 5.0, value=4.0, step=0.1, label="DNSMOS Overall")
                fmax_slider = gr.Slider(0, 24000, value=24000, step=1,
                                        label="Fmax (Hz) (T+H) Use 22050 for voice cloning")
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ Score")
                pitch_std_slider = gr.Slider(0.0, 300.0, value=45.0, step=1,
                                             label="Pitch Std deviation. Controls Tone: normal(20-45) or expressive (60-150)")
                speaking_rate_slider = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Speaking Rate")

            with gr.Column():
                gr.Markdown("## Generation Parameters")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")
                seed_number = gr.Number(label="Seed", value=420, precision=0)
                with gr.Row():
                    randomize_seed_toggle = gr.Checkbox(label="Randomize Seed (before generation)", value=True)
                    disable_torch_compile = gr.Checkbox(label="Disable Torch Compile",
                                                        info="Only Transformer Windows:To enable Compile you must start the app in a dev console",
                                                        value=disable_torch_compile_default)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### NovelAi's unified sampler")
                    linear_slider = gr.Slider(-2.0, 2.0, 0.5, 0.01,
                                              label="Linear (set to 0 to disable unified sampling)",
                                              info="High values make the output less random.")
                    confidence_slider = gr.Slider(-2.0, 2.0, 0.40, 0.01, label="Confidence",
                                                  info="Low values make random outputs more random.")
                    quadratic_slider = gr.Slider(-2.0, 2.0, 0.00, 0.01, label="Quadratic",
                                                 info="High values make low probablities much lower.")
                with gr.Column():
                    gr.Markdown("### Legacy sampling")
                    top_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Top P")
                    min_k_slider = gr.Slider(0.0, 1024, 0, 1, label="Min K")
                    min_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Min P")

        with gr.Accordion("Advanced Parameters", open=False):
            gr.Markdown("### Unconditional Toggles\n"
                        "Checking a box will make the model ignore the corresponding conditioning value and make it unconditional.\n"
                        'Practically this means the given conditioning feature will be unconstrained and "filled in automatically".')
            with gr.Row():
                unconditional_keys = gr.CheckboxGroup(
                    ["speaker", "emotion", "vqscore_8", "fmax", "pitch_std", "speaking_rate", "dnsmos_ovrl",
                        "speaker_noised", ], value=["emotion"], label="Unconditional Keys", )

            gr.Markdown("### Emotion Sliders\n"
                        "Warning: The way these sliders work is not intuitive and may require some trial and error to get the desired effect.\n"
                        "Certain configurations can cause the model to become unstable. Setting emotion to unconditional may help.")
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 1.0, 0.05, label="Happiness")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                emotion7 = gr.Slider(0.0, 1.0, 0.1, 0.05, label="Other")
                emotion8 = gr.Slider(0.0, 1.0, 0.2, 0.05, label="Neutral")

        # Cache Management Section
        with gr.Accordion("Cache Management", open=False):
            gr.Markdown("### Speaker Embedding Cache")
            gr.Markdown("The application uses an advanced caching system to store speaker embeddings, "
                       "improving performance for repeated use of the same speaker audio.")

            with gr.Row():
                cache_stats_button = gr.Button("📊 Show Cache Stats")
                clear_cache_button = gr.Button("🗑️ Clear Cache")

            cache_info = gr.Markdown("Click 'Show Cache Stats' to view current cache status.")

        with gr.Column():
            generate_button = gr.Button("Generate Audio", variant="primary")
            output_audio = gr.Audio(label="Generated Audio", type="numpy", autoplay=True)

        # Event handlers
        model_choice.change(fn=update_ui, inputs=[model_choice],
            outputs=[text, language, speaker_audio, prefix_audio, emotion1, emotion2, emotion3, emotion4, emotion5,
                emotion6, emotion7, emotion8, vq_single_slider, fmax_slider, pitch_std_slider, speaking_rate_slider,
                dnsmos_slider, speaker_noised_checkbox, unconditional_keys, disable_torch_compile, ], )

        demo.load(fn=update_ui, inputs=[model_choice],
            outputs=[text, language, speaker_audio, prefix_audio, emotion1, emotion2, emotion3, emotion4, emotion5,
                emotion6, emotion7, emotion8, vq_single_slider, fmax_slider, pitch_std_slider, speaking_rate_slider,
                dnsmos_slider, speaker_noised_checkbox, unconditional_keys, ], )

        generate_button.click(fn=generate_audio,
            inputs=[model_choice, text, language, speaker_audio, prefix_audio, emotion1, emotion2, emotion3, emotion4,
                emotion5, emotion6, emotion7, emotion8, vq_single_slider, fmax_slider, pitch_std_slider,
                speaking_rate_slider, dnsmos_slider, speaker_noised_checkbox, cfg_scale_slider, top_p_slider,
                min_k_slider, min_p_slider, linear_slider, confidence_slider, quadratic_slider, seed_number,
                randomize_seed_toggle, unconditional_keys, ], outputs=[output_audio, seed_number], )

        # Cache management event handlers
        cache_stats_button.click(fn=show_cache_stats, outputs=cache_info)
        clear_cache_button.click(fn=clear_cache_interface, outputs=cache_info)

    return demo


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    gr.set_static_paths(paths=["assets/","cache"])

    # Initialize cache if not disabled
    if not args.disable_cache:
        try:
            # Print initial cache stats
            print_speaker_cache_stats()
        except Exception as e:
            logging.warning(f"Error displaying initial cache stats: {e}")

    load_model_wrapper("Zyphra/Zonos-v0.1-hybrid")
    demo = build_interface()
    demo.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser)
