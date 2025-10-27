#!/usr/bin/env python3
"""
Zonos Text-to-Speech Application with Gradio Interface
"""

# Standard library imports
from argparse import ArgumentParser
import os
from pathlib import Path
from sys import exit, stdout
from time import perf_counter_ns

# Third-party imports
import torch
import gradio as gr
from loguru import logger
# Local imports
from utilities.app_config import AppConfiguration
from utilities.app_constants import UIConfig
from utilities.audio_generation_pipeline import (
    prepare_generation_params, setup_speaker_conditioning, 
    create_conditioning_dict, setup_prefix_audio,
    create_progress_callback, generate_and_save_audio
)
from utilities.cache_utils import get_embed_cache_dir, get_wavout_dir, get_speakers_dir
from utilities.file_utils import lcx_checkmodels
from utilities.gradio_utils import update_ui_visibility  
from utilities.model_utils import load_model_if_needed, get_supported_models
from utilities.report import generate_troubleshooting_report
from utilities.ui_components import (
    create_model_and_text_controls, create_audio_controls,
    create_conditioning_controls, create_generation_controls,
    create_sampling_controls, create_advanced_controls, create_output_controls
)
# Zonos-specific imports
from zonos.model import DEFAULT_BACKBONE_CLS as ZONOS_BACKBONE
from zonos.utilities.utils import DEFAULT_DEVICE

# =============================================================================
# APPLICATION SETUP
# =============================================================================

# Initialize configuration
current_dir = Path(__file__).parent
config = AppConfiguration()
config.setup_logging()

# Load configuration and models
models_dict, models_values = config.load_configuration()
AI_MODEL_DIR_TF, AI_MODEL_DIR_HY = config.get_model_paths()
disable_torch_compile_default = config.get_disable_torch_compile_default()
IGNORE_PING = None

# Enable TF32 for better performance on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True  
torch.set_float32_matmul_precision("medium")
os.environ["GRADIO_TEMP_DIR"] = "temp_dir"
# =============================================================================
# COMMAND LINE ARGUMENT PARSING  
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument("--server", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, required=False)
    parser.add_argument("--inbrowser", action='store_true')
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument("--checkmodels", action='store_true')
    parser.add_argument("--integritycheck", action='store_true')
    parser.add_argument("--sysreport", action='store_true')
    return parser.parse_args()


def handle_cli_options(args, config):
    """Handle command line options that exit early"""
    if args.checkmodels:
        lcx_checkmodels(
            config.models.keys(), config.paths, config.models,
            models_values, []
        )

    if args.sysreport:
        full_report = generate_troubleshooting_report(in_model_config_file=config.CONFIG_FILE)
        print(full_report)
        exit()

# =============================================================================
# CORE APPLICATION FUNCTIONS
# =============================================================================

def load_model_wrapper(model_choice: str, disable_torch_compile: bool = disable_torch_compile_default):
    """Wrapper for model loading"""
    return load_model_if_needed(model_choice, DEFAULT_DEVICE, config.models.keys(), disable_torch_compile=disable_torch_compile)


def update_ui(model_choice, disable_torch_compile):
    """Dynamically show/hide UI elements based on the model's conditioners"""
    model = load_model_wrapper(model_choice, disable_torch_compile)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    return update_ui_visibility(cond_names)


async def generate_audio(model_choice, text, language, speaker_audio, prefix_audio, e1, e2, e3, e4, e5, e6, e7, e8,
                  vq_single, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p,
                  top_k, min_p, linear, confidence, quadratic, seed, randomize_seed, unconditional_keys,
                  disable_torch_compile=disable_torch_compile_default, progress=gr.Progress(), do_progress=False):
    """Generate audio based on the provided UI parameters"""
    global IGNORE_PING
    job_id = seed

    if text == "ping":
       if IGNORE_PING is None:
        IGNORE_PING = "pending"
       else:
          logger.info("Ping request received, sending silence audio.")
          return ["assets/silence_100ms.wav", job_id]

    emotions = [e1, e2, e3, e4, e5, e6, e7, e8]

    if speaker_audio is None:
        speaker_audio = 'malecommoner'
        logger.info(f'Requested: default {speaker_audio} with{f"out emotion." if "emotion" in unconditional_keys else f" emotion values: {emotions}"} Text: "{text}"')
    else:
        logger.info(f'Requested: {Path(speaker_audio).stem} with{f"out emotion." if "emotion" in unconditional_keys else f" emotion values: {emotions}"} Text: "{text}"')


    func_start_time = perf_counter_ns()
    
    # Load model
    selected_model = load_model_wrapper(model_choice, disable_torch_compile)
    
    # Prepare generation parameters
    
    params = prepare_generation_params(
        text=text, seed=seed, randomize_seed=randomize_seed,
        speaker_noised=speaker_noised, vq_single=vq_single,
        fmax=fmax, pitch_std=pitch_std, speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl, cfg_scale=cfg_scale, top_p=top_p,
        top_k=top_k, min_p=min_p, linear=linear, confidence=confidence,
        quadratic=quadratic, disable_torch_compile=disable_torch_compile
    )
    
    
    # Setup conditioning
    speaker_embedding = await setup_speaker_conditioning(
        speaker_audio, unconditional_keys, selected_model
    )
    
    cond_dict = create_conditioning_dict(
        text, language, speaker_embedding, emotions, params, unconditional_keys
    )
    
    conditioning = selected_model.prepare_conditioning(
        cond_dict, cfg_scale=params['cfg_scale'], use_cache=False
    )
    
    # Setup prefix audio
    audio_prefix_codes = await setup_prefix_audio(prefix_audio, selected_model)
    
    # Setup progress callback
    callback = create_progress_callback(do_progress, text, progress)
    
    # Generate and save audio
    output_wav_path, wav_length = generate_and_save_audio(
        selected_model, conditioning, params, audio_prefix_codes, 
        callback, speaker_audio
    )
    
    # Log performance
    total_duration_s = (perf_counter_ns() - func_start_time) / 1_000_000_000
    logger.info(f"Generated audio length: {wav_length:.2f}s. Execution time: {total_duration_s:.2f}s.Speed: {wav_length / total_duration_s:.2f}x")
    stdout.flush()

    if IGNORE_PING == "pending":
        IGNORE_PING = True
        os.remove(output_wav_path)
        return ["assets/silence_100ms.wav", job_id]
    
    truct_path_str = str(Path(output_wav_path).relative_to(current_dir))
    return [truct_path_str, job_id]

def build_interface():
    output_temp = get_wavout_dir().parent.absolute()
    latents_dir = get_embed_cache_dir().parent.absolute()
    speakers_dir = get_speakers_dir().parent.absolute()

    gr.set_static_paths([output_temp, latents_dir, speakers_dir])
    """Build and return the Gradio interface"""
    supported_models = get_supported_models(ZONOS_BACKBONE, AI_MODEL_DIR_HY, AI_MODEL_DIR_TF)

    with gr.Blocks(analytics_enabled=False) as demo:
        # Create UI components using modular functions
        with gr.Row():
            model_choice, text, language = create_model_and_text_controls(supported_models)
            prefix_audio, speaker_audio, speaker_noised_checkbox = create_audio_controls()

        with gr.Row():
            dnsmos_slider, fmax_slider, vq_single_slider, pitch_std_slider, speaking_rate_slider = create_conditioning_controls()
            cfg_scale_slider, seed_number, randomize_seed_toggle, disable_torch_compile = create_generation_controls(disable_torch_compile_default)

        # Sampling controls
        linear_slider, confidence_slider, quadratic_slider, top_p_slider, min_k_slider, min_p_slider = create_sampling_controls()
        
        # Advanced controls
        unconditional_keys, emotions = create_advanced_controls()
        
        # Output controls
        generate_button, output_audio = create_output_controls()

        # Hidden progress control
        do_progress = gr.Checkbox(label="Progress", value=False, visible=False)

        # Event handlers
        model_choice.change(
            fn=update_ui, 
            inputs=[model_choice, disable_torch_compile],
            outputs=[text, language, speaker_audio, prefix_audio] + emotions + 
                    [vq_single_slider, fmax_slider, pitch_std_slider, speaking_rate_slider,
                     dnsmos_slider, speaker_noised_checkbox, unconditional_keys]
        )

        demo.load(
            fn=update_ui, 
            inputs=[model_choice, disable_torch_compile],
            outputs=[text, language, speaker_audio, prefix_audio] + emotions + 
                    [vq_single_slider, fmax_slider, pitch_std_slider, speaking_rate_slider,
                     dnsmos_slider, speaker_noised_checkbox, unconditional_keys]
        )

        generate_button.click(
            fn=generate_audio, 
            concurrency_limit=UIConfig.GENERATION_CONCURRENCY_LIMIT,
            inputs=[model_choice, text, language, speaker_audio, prefix_audio] + emotions +
                   [vq_single_slider, fmax_slider, pitch_std_slider, speaking_rate_slider,
                    dnsmos_slider, speaker_noised_checkbox, cfg_scale_slider, top_p_slider,
                    min_k_slider, min_p_slider, linear_slider, confidence_slider, 
                    quadratic_slider, seed_number, randomize_seed_toggle, unconditional_keys],
            outputs=[output_audio, seed_number]
        )

    return demo


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    handle_cli_options(args, config)
    
    # Set up Gradio static paths and preload model
    gr.set_static_paths(paths=["assets/"])
    default_model = "Zyphra/Zonos-v0.1-transformer"
    model = load_model_wrapper(default_model)

    
    #warmup_file, _ = asyncio.run(generate_audio(model_choice='Zyphra/Zonos-v0.1-transformer', text='Warmup Time.', language='en-us', speaker_audio=None, prefix_audio="empty_100ms.wav", e1=0.0, e2=0.0, e3=0.0, e4=0.0, e5=0.0, e6=0.0, e7=0.0, e8=1.0, vq_single=0.699999988079071, fmax=24000, pitch_std=45.0, speaking_rate=14.600000381469727, dnsmos_ovrl=4, speaker_noised=False, cfg_scale=4.5, top_p=0.0, top_k=0.0, min_p=0.0, linear=0.5, confidence=0.4000000059604645, quadratic=0.0, seed=6298667263556447910, randomize_seed=False, unconditional_keys=['emotion']))
    #os.remove(warmup_file)

    # Build and launch interface
    demo = build_interface().queue()
    demo.launch(
        server_name=args.server, 
        server_port=args.port, 
        share=args.share, 
        inbrowser=args.inbrowser
    )
    