"""
Gradio UI component builders
"""
import gradio as gr
from utilities.app_constants import UIConfig
from zonos.conditioning import supported_language_codes


def create_model_and_text_controls(supported_models: list) -> tuple:
    """Create model selection and text input controls"""
    with gr.Column():
        model_choice = gr.Dropdown(
            choices=supported_models, 
            value=supported_models[0],
            label="Zonos Model Selection"
        )
        text = gr.Textbox(
            label="Text to Synthesize",
            value="Zonos uses eSpeak for text to phoneme conversion!",
            lines=UIConfig.TEXT_INPUT_LINES, 
            max_length=UIConfig.TEXT_MAX_LENGTH
        )
        language = gr.Dropdown(
            choices=supported_language_codes, 
            value="en-us", 
            label="Language Code"
        )
    return model_choice, text, language


def create_audio_controls() -> tuple:
    """Create audio input controls"""
    prefix_audio = gr.Audio(
        value="assets/silence_100ms.wav",
        label="Optional Prefix Audio (continue from this audio)", 
        type="filepath"
    )
    
    with gr.Column():
        speaker_audio = gr.Audio(
            label="Optional Speaker Audio (for cloning)", 
            type="filepath"
        )
        speaker_noised_checkbox = gr.Checkbox(
            label="Denoise Speaker? (only Hybrid model)", 
            value=False
        )
    
    return prefix_audio, speaker_audio, speaker_noised_checkbox


def create_conditioning_controls() -> tuple:
    """Create conditioning parameter controls"""
    with gr.Column():
        gr.Markdown("## Conditioning Parameters")
        
        min_val, max_val, default_val, step = UIConfig.DNSMOS_RANGE
        dnsmos_slider = gr.Slider(min_val, max_val, value=default_val, step=step, label="DNSMOS Overall")
        
        min_val, max_val, default_val, step = UIConfig.FMAX_RANGE
        fmax_slider = gr.Slider(min_val, max_val, value=default_val, step=step,
                               label="Fmax (Hz) (T+H) Use 22050 for voice cloning")
        
        min_val, max_val, default_val, step = UIConfig.VQ_SCORE_RANGE
        vq_single_slider = gr.Slider(min_val, max_val, default_val, step, label="VQ Score")
        
        min_val, max_val, default_val, step = UIConfig.PITCH_STD_RANGE
        pitch_std_slider = gr.Slider(min_val, max_val, value=default_val, step=step,
                                    label="Pitch Std deviation. Controls Tone: normal(20-45) or expressive (60-150)")
        
        min_val, max_val, default_val, step = UIConfig.SPEAKING_RATE_RANGE
        speaking_rate_slider = gr.Slider(min_val, max_val, value=default_val, step=step, label="Speaking Rate")
    
    return dnsmos_slider, fmax_slider, vq_single_slider, pitch_std_slider, speaking_rate_slider


def create_generation_controls(disable_torch_compile_default: bool) -> tuple:
    """Create generation parameter controls"""
    with gr.Column():
        gr.Markdown("## Generation Parameters")
        
        min_val, max_val, default_val, step = UIConfig.CFG_SCALE_RANGE
        cfg_scale_slider = gr.Slider(min_val, max_val, default_val, step, label="CFG Scale")
        
        seed_number = gr.Number(label="Seed", value=420, precision=0)
        
        with gr.Row():
            randomize_seed_toggle = gr.Checkbox(label="Randomize Seed (before generation)", value=True)
            disable_torch_compile = gr.Checkbox(
                label="Disable Torch Compile",
                info="Only Transformer Windows:To enable Compile you must start the app in a dev console",
                value=disable_torch_compile_default
            )
    
    return cfg_scale_slider, seed_number, randomize_seed_toggle, disable_torch_compile


def create_sampling_controls() -> tuple:
    """Create sampling parameter controls"""
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
    
    return linear_slider, confidence_slider, quadratic_slider, top_p_slider, min_k_slider, min_p_slider


def create_advanced_controls() -> tuple:
    """Create advanced parameter controls"""
    with gr.Accordion("Advanced Parameters", open=False):
        gr.Markdown("### Unconditional Toggles\n"
                   "Checking a box will make the model ignore the corresponding conditioning value and make it unconditional.\n"
                   'Practically this means the given conditioning feature will be unconstrained and "filled in automatically".')
        
        with gr.Row():
            unconditional_keys = gr.CheckboxGroup(
                ["speaker", "emotion", "vqscore_8", "fmax", "pitch_std", "speaking_rate", "dnsmos_ovrl", "speaker_noised"],
                value=["emotion"], 
                label="Unconditional Keys"
            )

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
    
    emotions = [emotion1, emotion2, emotion3, emotion4, emotion5, emotion6, emotion7, emotion8]
    return unconditional_keys, emotions


def create_output_controls() -> tuple:
    """Create output controls"""
    with gr.Column():
        generate_button = gr.Button("Generate Audio")
        output_audio = gr.Audio(label="Generated Audio", type="filepath", autoplay=True)
    
    return generate_button, output_audio
