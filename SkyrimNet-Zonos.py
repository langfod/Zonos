#!/usr/bin/env python3
"""
Zonos Text-to-Speech Application with Gradio Interface
"""

# Standard library imports
import logging
from argparse import ArgumentParser
from pathlib import Path
from sys import exit, stdout
from time import perf_counter_ns

# Third-party imports
import torch
import gradio as gr
from gradio import processing_utils as gr_processing_utils
from gradio.data_classes import GradioModel, GradioRootModel
from gradio_client import utils as gr_client_utils

# Local imports
from utilities.app_config import AppConfiguration
from utilities.app_constants import UIConfig, PerformanceConfig, AudioGenerationConfig
from utilities.audio_generation_pipeline import (
    prepare_generation_params, setup_speaker_conditioning, 
    create_conditioning_dict, setup_prefix_audio,
    create_progress_callback, generate_and_save_audio
)
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
# GRADIO PATH HANDLING PATCH
# =============================================================================

def _path_is_relative_to(path: Path, base: Path) -> bool:
    """Compatibility helper for checking whether path is inside base."""
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _install_gradio_path_patch():
    """Allow trusted local files to be accepted by Gradio preprocessing."""

    trusted_roots = [
        Path.cwd(),
        Path.cwd() / "assets",
        Path.cwd() / "cache",
        Path.cwd() / "output_temp",
    ]

    original_check_allowed = getattr(gr_processing_utils, "_check_allowed", None)

    if original_check_allowed is None:
        logging.warning("Could not locate gradio.processing_utils._check_allowed")
        return

    def patched_check_allowed(path, check_in_upload_folder):
        logging.info(
            "Gradio path check: %s (check_in_upload_folder=%s)",
            path,
            check_in_upload_folder,
        )
        if check_in_upload_folder:
            try:
                abs_path = Path(path).resolve()
            except Exception:  # noqa: BLE001 - fall back to original handler
                return original_check_allowed(path, check_in_upload_folder)

            if abs_path.is_file() and any(
                _path_is_relative_to(abs_path, root) for root in trusted_roots
            ):
                return

        return original_check_allowed(path, check_in_upload_folder)

    gr_processing_utils._check_allowed = patched_check_allowed

    original_async_move_files_to_cache = gr_processing_utils.async_move_files_to_cache

    async def patched_async_move_files_to_cache(
        data,
        block,
        postprocess: bool = False,
        check_in_upload_folder: bool = False,
        keep_in_cache: bool = False,
    ):
        if isinstance(data, (GradioRootModel, GradioModel)):
            data = data.model_dump()

        def _scrub_payload(d):
            if gr_client_utils.is_file_obj_with_meta(d):
                path = d.get("path")
                if not path:
                    logging.info("Dropping empty file payload for block %s", getattr(block, "label", block.__class__.__name__))
                    return None
                try:
                    resolved_path = Path(path).resolve()
                except Exception:  # noqa: BLE001 - leave payload unchanged if path invalid
                    return d

                if resolved_path.is_dir():
                    logging.info("Dropping directory payload for block %s: %s", getattr(block, "label", block.__class__.__name__), resolved_path)
                    return None

            return d

        cleaned_data = gr_client_utils.traverse(
            data,
            _scrub_payload,
            gr_client_utils.is_file_obj_with_meta,
        )

        return await original_async_move_files_to_cache(
            cleaned_data,
            block,
            postprocess=postprocess,
            check_in_upload_folder=check_in_upload_folder,
            keep_in_cache=keep_in_cache,
        )

    gr_processing_utils.async_move_files_to_cache = patched_async_move_files_to_cache


_install_gradio_path_patch()

# =============================================================================
# APPLICATION SETUP
# =============================================================================

# Initialize configuration
config = AppConfiguration()
config.setup_logging()

# Load configuration and models
models_dict, models_values = config.load_configuration()
AI_MODEL_DIR_TF, AI_MODEL_DIR_HY = config.get_model_paths()
disable_torch_compile_default = config.get_disable_torch_compile_default()

# Enable TF32 for better performance on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True  
torch.set_float32_matmul_precision("medium")

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


def run_startup_warmup(model_choice: str, disable_torch_compile: bool) -> None:
    """Run a short generation pass at startup to trigger compilation and autotune."""
    warmup_label = "warmup sample"
    emotions = [0.0] * 8
    start_ns = perf_counter_ns()

    try:
        selected_model = load_model_wrapper(model_choice, disable_torch_compile)

        params = prepare_generation_params(
            text=warmup_label,
            seed=PerformanceConfig.DEFAULT_SEED,
            randomize_seed=False,
            speaker_noised=False,
            vq_single=UIConfig.VQ_SCORE_RANGE[2],
            fmax=UIConfig.FMAX_RANGE[2],
            pitch_std=UIConfig.PITCH_STD_RANGE[2],
            speaking_rate=UIConfig.SPEAKING_RATE_RANGE[2],
            dnsmos_ovrl=UIConfig.DNSMOS_RANGE[2],
            cfg_scale=UIConfig.CFG_SCALE_RANGE[2],
            top_p=0.0,
            top_k=0,
            min_p=0.0,
            linear=0.5,
            confidence=0.40,
            quadratic=0.0,
            disable_torch_compile=disable_torch_compile
        )

        params['vq_single'] = UIConfig.VQ_SCORE_RANGE[2]
        params['disable_torch_compile'] = disable_torch_compile

        cond_dict = create_conditioning_dict(
            text=warmup_label,
            language="en-us",
            speaker_embedding=None,
            emotions=emotions,
            params=params,
            unconditional_keys=[]
        )

        conditioning = selected_model.prepare_conditioning(
            cond_dict,
            cfg_scale=params['cfg_scale'],
            use_cache=False
        )

        warmup_tokens = min(AudioGenerationConfig.TOKENS_PER_SECOND, params['max_new_tokens'])

        codes = selected_model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=None,
            max_new_tokens=warmup_tokens,
            cfg_scale=params['cfg_scale'],
            batch_size=1,
            sampling_params={
                'top_p': params['top_p'],
                'top_k': params['top_k'],
                'min_p': params['min_p'],
                'linear': params['linear'],
                'conf': params['confidence'],
                'quad': params['quadratic']
            },
            disable_torch_compile=disable_torch_compile,
            callback=None
        )

        selected_model.autoencoder.decode(codes)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        duration_s = (perf_counter_ns() - start_ns) / 1_000_000_000
        logging.info(f"Startup warmup completed in {duration_s:.2f} seconds")

    except Exception as exc:  # noqa: BLE001 - guard warmup failures without blocking launch
        logging.warning("Warmup failed: %s", exc, exc_info=True)


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
    logging.info(f'Requested: "{text}"')
    
    func_start_time = perf_counter_ns()
    
    # Load model
    selected_model = load_model_wrapper(model_choice, disable_torch_compile)
    
    # Prepare generation parameters
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
    
    # Setup conditioning
    speaker_embedding = await setup_speaker_conditioning(
        speaker_audio, unconditional_keys, uuid, selected_model
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
        callback, speaker_audio, uuid
    )
    
    # Log performance
    total_duration_s = (perf_counter_ns() - func_start_time) / 1_000_000_000
    logging.info(f"Total 'generate_audio' execution time: {total_duration_s:.2f} seconds")
    logging.info(f"Generated audio length: {wav_length:.2f} seconds. Speed: {wav_length / total_duration_s:.2f}x")
    stdout.flush()

    return [output_wav_path, uuid]

def build_interface():
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
    default_model_choice = "Zyphra/Zonos-v0.1-transformer"
    load_model_wrapper(default_model_choice)
    run_startup_warmup(default_model_choice, disable_torch_compile_default)
    
    # Build and launch interface
    demo = build_interface().queue()
    demo.launch(
        server_name=args.server, 
        server_port=args.port, 
        share=args.share, 
        inbrowser=args.inbrowser
    )
